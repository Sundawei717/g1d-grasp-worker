from dataclasses import dataclass
from math import isclose, isfinite

from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PhysicsSample:
    time_s: float
    target_z_m: float
    wrist_z_m: float
    relative_xyz_m: tuple[float, float, float]
    relative_xyzw: tuple[float, float, float, float]
    target_supported: bool
    target_part_pads: frozenset[str]
    forbidden_contact: bool
    pose_override_or_constraint: bool
    target_inner_pads: frozenset[str] = frozenset()


def evaluate(samples, *, lift_start, hold_start, dt, baseline_contract="instantaneous_v1", unified=False,
             expected_lift_m=0.030):
    """Score an independently instrumented trace, sampled on every physics step."""
    if not 0 <= lift_start < hold_start < len(samples) or not isfinite(dt) or dt <= 0:
        return {"success": False, "reasons": ["incomplete_trace"]}
    if any(
        not all(isfinite(value) for value in (
            sample.time_s, sample.target_z_m, sample.wrist_z_m,
            *sample.relative_xyz_m, *sample.relative_xyzw))
        or len(sample.relative_xyz_m) != 3 or len(sample.relative_xyzw) != 4
        or not isclose(sum(value * value for value in sample.relative_xyzw), 1, abs_tol=1e-3)
        for sample in samples
    ) or any(
        not isclose(after.time_s - before.time_s, dt, rel_tol=1e-3, abs_tol=1e-7)
        for before, after in zip(samples, samples[1:])
    ):
        return {"success": False, "reasons": ["invalid_or_gapped_physics_trace"]}

    before = samples[lift_start]
    hold = samples[hold_start:]
    hold_duration = hold[-1].time_s - hold[0].time_s
    wrist_lift = hold[0].wrist_z_m - before.wrist_z_m
    min_target_lift = min(sample.target_z_m - before.target_z_m for sample in hold)
    relative_start = Rotation.from_quat(hold[0].relative_xyzw)
    max_angle = max((relative_start.inv() * Rotation.from_quat(sample.relative_xyzw)).magnitude() for sample in hold)
    max_slip = max(sum((left - right) ** 2 for left, right in zip(
        sample.relative_xyz_m, hold[0].relative_xyz_m)) ** 0.5 for sample in hold)
    reasons = []
    supported_at_baseline = samples[0].target_supported if unified else before.target_supported
    if not unified and baseline_contract == "stable_support_v2" and not supported_at_baseline:
        for prior in reversed(samples[:lift_start]):
            if before.time_s - prior.time_s > 0.010000001:
                break
            if (prior.target_supported and abs(prior.target_z_m - before.target_z_m) <= 0.00025
                    and abs(prior.wrist_z_m - before.wrist_z_m) <= 0.00025
                    and sum((a - b) ** 2 for a, b in zip(prior.relative_xyz_m, before.relative_xyz_m)) <= 0.00025**2
                    and (Rotation.from_quat(prior.relative_xyzw).inv() * Rotation.from_quat(before.relative_xyzw)).magnitude()
                    <= 0.005):
                supported_at_baseline = True
                break
    elif baseline_contract not in {"instantaneous_v1", "stable_support_v2"}:
        raise ValueError(f"Unknown baseline contract {baseline_contract}")
    if not supported_at_baseline:
        reasons.append("invalid_pre_lift_baseline")
    if unified:
        edge = max(1, round(0.1 / dt))
        paired = {"pad_1", "pad_2"}
        if not all(any(sample.target_inner_pads == paired for sample in segment)
                   for segment in (hold[:edge], hold[-edge:])):
            reasons.append("no_bilateral_target_hold")
    elif before.target_part_pads != {"pad_1", "pad_2"} or hold[0].target_part_pads != {"pad_1", "pad_2"}:
        reasons.append("wrong_target_or_part_contact")
    if not unified and any(sample.forbidden_contact for sample in samples):
        reasons.append("forbidden_contact")
    if any(sample.pose_override_or_constraint for sample in samples):
        reasons.append("nonphysical_grasp")
    if not isclose(wrist_lift, expected_lift_m, abs_tol=0.005):
        reasons.append("wrong_trial_lift_height")
    if min_target_lift < 0.02 or any(sample.target_supported for sample in hold):
        reasons.append("target_not_lifted_clear")
    if hold_duration < 0.5 - 1e-6:
        reasons.append("short_hold")
    if max_slip > 0.01 + 1e-9 or max_angle > 0.2617993877991494 + 1e-9:
        reasons.append("target_slipped")
    return {"success": not reasons, "reasons": reasons, "hold_duration_s": hold_duration,
            "wrist_lift_m": wrist_lift, "minimum_target_lift_m": min_target_lift,
            "max_relative_slip_m": max_slip, "max_relative_turn_rad": max_angle}
