from collections import deque
from dataclasses import asdict, dataclass
import math
from time import perf_counter

import numpy as np


CONTROL_HZ = 20
EPISODE_STEPS = 300
LIFT_STEPS = 16
HOLD_STEPS = 11
LIFT_HEIGHT_M = 0.080


@dataclass(frozen=True)
class LiftGateConfig:
    close_threshold_m: float = 0.071
    close_duration_s: float = 0.25
    arm_window_s: float = 0.20
    arm_displacement_m: float = 0.010
    arm_max_joint_speed_rad_s: float = 0.10
    jaw_stable_duration_s: float = 0.15
    jaw_delta_m: float = 0.0005
    min_pad_force_n: float = 5.0
    mechanical_close_m: float = 0.002
    min_closing_reserve_m: float = 0.006

    def __post_init__(self):
        positive = {k: v for k, v in asdict(self).items() if k != "min_closing_reserve_m"}
        if (any(not math.isfinite(value) or value <= 0 for value in positive.values())
                or not math.isfinite(self.min_closing_reserve_m) or self.min_closing_reserve_m < 0):
            raise ValueError("Trial lift thresholds must be finite and positive")


def ticks(seconds):
    return math.ceil(seconds * CONTROL_HZ - 1e-9)


def guarded_gripper_width(requested, observed, previous, pad_forces, config):
    """Apply the normal slew guard, with bounded preloading against measured contact."""
    lower, upper = observed - 0.004, observed + 0.004
    if config.min_closing_reserve_m and requested <= previous:
        if min(pad_forces.get(pad, 0.0) for pad in ("pad_1", "pad_2")) >= config.min_pad_force_n:
            lower = max(previous - 0.004, observed - config.min_closing_reserve_m - 0.003)
            upper = max(lower, previous + 0.004)
        else:
            lower = min(lower, previous)
    return float(np.clip(np.clip(requested, lower, upper), 0.00107, 0.09007))


class LiftGate:
    def __init__(self, config, initial):
        self.config = config
        self.window = ticks(config.arm_window_s)
        self.history = deque([initial], maxlen=self.window + 1)
        self.closed = self.stable = 0

    def update(self, width_command, feedback):
        c = self.config
        before = self.history[-1]
        self.history.append(feedback)
        self.closed = self.closed + 1 if width_command <= c.close_threshold_m else 0
        delta = abs(feedback["width_m"] - before["width_m"])
        forces = feedback["pad_normal_force_n"]
        load = min(forces.get(pad, 0.0) for pad in ("pad_1", "pad_2")) >= c.min_pad_force_n
        at_limit = feedback["width_m"] <= c.mechanical_close_m
        reserve = feedback["width_m"] - width_command
        loaded = load and (c.min_closing_reserve_m == 0 or reserve >= c.min_closing_reserve_m)
        empty_at_limit = at_limit and not any(force > 0 for force in forces.values())
        stable = delta < c.jaw_delta_m and (loaded or empty_at_limit)
        self.stable = self.stable + 1 if stable and self.closed else 0
        positions = np.array([item["eef_xyz_m"] for item in self.history])
        displacement = float(np.linalg.norm(positions - positions[0], axis=1).max())
        speed = max(item["max_joint_speed_rad_s"] for item in list(self.history)[1:])
        arm_stable = len(self.history) == self.window + 1 and (
            displacement < c.arm_displacement_m or speed < c.arm_max_joint_speed_rad_s)
        flags = {"sustained_close": self.closed >= ticks(c.close_duration_s),
                 "arm_stable": arm_stable, "jaw_stable_loaded": self.stable >= ticks(c.jaw_stable_duration_s)}
        return {**flags, "ready": all(flags.values()), "width_command_m": float(width_command),
                "width_actual_m": feedback["width_m"], "jaw_delta_m": delta, "close_steps": self.closed,
                "jaw_stable_steps": self.stable, "arm_displacement_m": displacement,
                "closing_reserve_m": reserve, "target_load_ready": loaded,
                "max_joint_speed_rad_s": speed, "pad_normal_force_n": forces, "at_mechanical_limit": at_limit,
                "empty_at_mechanical_limit": empty_at_limit}


def execute(*, initial_frame, capture, source, feedback, apply, lift, config):
    """Consume a recorded/native source or a policy source with the same 20 Hz deadline and trial."""
    began = perf_counter()
    initial = feedback()
    previous_control = current_control = initial
    gate = LiftGate(config, initial)
    steps, decisions, gate_log, trials = 0, [], [], []
    status = "timeout"
    while steps < EPISODE_STEPS and not trials:
        capture_started = perf_counter()
        rgb, frame = initial_frame if steps == 0 else capture()
        captured = perf_counter()
        capture_started = frame.get("acquisition_started_wall_s", capture_started)
        captured = frame.get("acquisition_finished_wall_s", captured)
        sensor_time = frame["sensor_sim_time_s"]
        live = feedback()
        if not math.isclose(sensor_time, live["sim_time_s"], abs_tol=2e-7, rel_tol=0):
            raise ValueError(f"Sensor and controller timestamps differ at step {steps}")
        if not math.isclose(sensor_time - initial["sim_time_s"], steps / CONTROL_HZ, abs_tol=2e-7, rel_tol=0):
            raise ValueError(f"Control-step clock gap at step {steps}")
        commands, latency = source(steps, rgb, frame, {**live, "previous_control": previous_control})
        if not 1 <= len(commands) <= 2:
            raise ValueError("A decision must produce one or two 20 Hz commands")
        decision = {"step": steps, "image_time_s": sensor_time,
                    "capture_ms": (captured - capture_started) * 1000,
                    "steps_applied": 0, **latency}
        decisions.append(decision)
        for command in commands[:EPISODE_STEPS - steps]:
            width, sent = apply(command)
            if not decision["steps_applied"]:
                decision["to_command_ms"] = (sent - capture_started) * 1000
            steps += 1
            decision["steps_applied"] += 1
            observed = feedback()
            previous_control, current_control = current_control, observed
            state = gate.update(width, observed)
            state.update(step=steps, time_s=steps / CONTROL_HZ, sim_time_s=observed["sim_time_s"])
            gate_log.append(state)
            if state["ready"] and EPISODE_STEPS - steps >= LIFT_STEPS + HOLD_STEPS:
                result = lift(state)
                steps += result["steps_used"]
                if not 0 <= result["steps_used"] <= LIFT_STEPS + HOLD_STEPS or steps > EPISODE_STEPS:
                    raise ValueError("Trial lift exceeded its time reservation")
                trials.append(result)
                status = "success" if result["success"] else "lift_failed"
                break
    return {"status": status, "success": status == "success", "steps": steps, "sim_duration_s": steps / CONTROL_HZ,
            "wall_s": perf_counter() - began, "gate_config": asdict(config), "gate": gate_log,
            "decisions": decisions, "trials": trials, "part_hit": trials[0].get("part_hit") if trials else None}
