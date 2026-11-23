from time import perf_counter

import numpy as np
import torch

from chunk_model import SharedResNet18ChunkTransformer
from prompt_views import PROMPT_AGE_CAP_S, encode_views


PROMPT_MODES = ("gt_visible", "gt_full_projection", "hold_last_reliable", "frozen_init")


class PolicyStop(RuntimeError):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status


class LoadedPolicy:
    def __init__(self, path, device):
        self.device = torch.device(device)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.norm = checkpoint["normalization"]
        self.action_reference = checkpoint["config"].get("action_reference", "absolute")
        self.model = SharedResNet18ChunkTransformer(fixture_random_init=True)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, rgb, box, point, state, age, crop):
        first, second, crop = encode_views(rgb, box, point, min_crop_px=64, previous=crop)
        height, width = rgb.shape[:2]
        meta = (crop.center_xy[0] / (width - 1), crop.center_xy[1] / (height - 1),
                crop.side_px / (width - 1), crop.side_px / (height - 1), 1.0, age)
        state_norm = self.norm["state"]
        normalized = (state - np.asarray(state_norm["mean"])) / np.asarray(state_norm["scale"])
        inputs = [torch.as_tensor(value, device=self.device, dtype=torch.float32).unsqueeze(0)
                  for value in (first.channels, second.channels, normalized, meta)]
        output = self.model(*inputs)[0].cpu().numpy()
        commands = output * np.asarray(self.norm["action"]["scale"]) + np.asarray(self.norm["action"]["mean"])
        if self.action_reference == "measured":
            commands += state[-1]
        if commands.shape != (8, 8) or not np.isfinite(commands).all():
            raise PolicyStop("unverified", "policy produced invalid commands")
        return commands, crop


def run_approach(policy, first_frame, first_prompt, capture, state, apply, receipt, prompt_mode="gt_visible"):
    """Select the 2D prompt while keeping RGB and arm observations live."""
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unknown prompt mode {prompt_mode}")
    if not first_prompt.get("prompt_valid"):
        raise PolicyStop("invalid_prompt", "initial target anchor is not visible")
    observation = np.asarray(state())
    previous = observation.copy()
    steps = 0
    closure = []
    crop = None
    last_prompt = first_prompt
    last_visible = 0
    receipt.update(phase="APPROACH_AND_GRASP", policy_calls=0, executed_control_steps=0, decisions=[])
    while steps < 120:
        start = perf_counter()
        if steps == 0:
            rgb, frame, prompt = (*first_frame, first_prompt)
        else:
            rgb, frame, prompt = capture()
        captured = perf_counter()
        if prompt_mode == "frozen_init":
            current_valid = True
        else:
            current_valid = bool(prompt and prompt.get("prompt_valid"))
        if current_valid and prompt_mode != "frozen_init":
            last_prompt, last_visible = prompt, steps
        if prompt_mode == "gt_full_projection" and not current_valid:
            raise PolicyStop("invalid_prompt", "the full geometric projection was outside the image")
        age_steps = 0 if prompt_mode == "frozen_init" else steps - last_visible
        if prompt_mode == "gt_visible" and age_steps > 4:
            apply(observation[:7], observation[7], "approach", 0.0)
            previous, observation = observation, np.asarray(state())
            steps += 1
            receipt["executed_control_steps"] = steps
            if age_steps >= 6:
                raise PolicyStop("tracking_lost", "fixed anchor was unavailable after the freeze window")
            continue
        model_age = min(age_steps / 20, PROMPT_AGE_CAP_S)
        chunk, crop = policy.predict(rgb, last_prompt["bbox_0"], last_prompt["affordance_0"],
                                     np.stack([previous, observation]), model_age, crop)
        inferred = perf_counter()
        count = min(1 if age_steps else 2, 120 - steps)
        receipt["policy_calls"] += 1
        requested = chunk[:count]
        completed = 0
        for request in requested:
            arm_step = 0.0175 if age_steps else 0.035
            narrowed = observation[7] - request[7] >= 0.002
            previous = observation.copy()
            applied_width = apply(request[:7], request[7], "close" if narrowed else "approach", arm_step)
            observation = np.asarray(state())
            steps += 1
            completed += 1
            receipt["executed_control_steps"] = steps
            if previous[7] - applied_width >= 0.002:
                closure.append(float(previous[7] - observation[7]))
            else:
                closure.clear()
            if len(closure) >= 2 and sum(closure[-2:]) >= 0.003:
                break
        receipt["decisions"].append({"step": steps - completed, "steps_applied": completed,
                                     "capture_and_prompt_ms": (captured - start) * 1000,
                                     "crop_and_inference_ms": (inferred - captured) * 1000,
                                     "total_ms": (perf_counter() - start) * 1000,
                                     "prompt_age_s": (steps - completed) / 20 if prompt_mode == "frozen_init" else age_steps / 20,
                                     "model_prompt_age_s": model_age, "source": (
                                         "initial" if prompt_mode == "frozen_init" else
                                         last_prompt.get("prompt_source", "gt_visible") if not age_steps else "last_reliable"),
                                     "image_time_s": frame.get("sensor_sim_time_s") if isinstance(frame, dict) else None,
                                     "bbox_0": last_prompt["bbox_0"], "affordance_0": last_prompt["affordance_0"]})
        if len(closure) >= 2 and sum(closure[-2:]) >= 0.003:
            break
    else:
        raise PolicyStop("timeout", "no executed closure with width feedback within 120 steps")

    receipt.update(phase="CLOSE_WAIT", close_started_step=steps)
    locked = observation[:7].copy()
    settled = 0
    for _ in range(20):
        previous = observation
        apply(locked, previous[7] - 0.004, "close", 0.035)
        observation = np.asarray(state())
        steps += 1
        receipt["executed_control_steps"] = steps
        settled = settled + 1 if abs(observation[7] - previous[7]) < 0.001 else 0
        if settled >= 2:
            receipt.update(phase="LIFT_VERIFY", close_completed_step=steps, measured_hold_width_m=float(observation[7]))
            return
    raise PolicyStop("timeout", "the gripper did not stabilize during the one-second close wait")
