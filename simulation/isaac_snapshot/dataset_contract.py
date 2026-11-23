from dataclasses import dataclass

import cv2
import numpy as np

from prompt_views import encode_views


TRAIN_PHASES = {"pre_grasp", "approach", "align", "close", "close_hold"}
NO_TRAIN_PHASES = {"lift", "hold", "trial_lift", "trial_hold"}


@dataclass(frozen=True)
class EpisodeInfo:
    actions: int
    valid_training_actions: int
    sample_hz: float


def action_windows(commands, train_valid, horizon=8):
    """Generate labels at all eligible starts, including action zero."""
    commands = np.asarray(commands)
    train_valid = np.asarray(train_valid, dtype=bool)
    if commands.ndim != 2 or train_valid.shape != (len(commands),) or horizon < 1:
        raise ValueError("Invalid action window inputs")
    for start in np.flatnonzero(train_valid):
        labels = np.zeros((horizon, commands.shape[1]), dtype=commands.dtype)
        mask = np.zeros(horizon, dtype=bool)
        for offset in range(horizon):
            index = start + offset
            if index >= len(commands) or not train_valid[index]:
                break
            labels[offset] = commands[index]
            mask[offset] = True
        yield int(start), labels, mask


def _numeric(h5, key, shape):
    array = np.asarray(h5[key])
    if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"Invalid numeric dataset {key}: {array.shape}/{array.dtype}")
    return array


def read_rgb(h5, index, key="obs/rgb_left"):
    dataset = h5[key]
    representation = dataset.attrs.get("encoding", "raw")
    if representation == "raw":
        rgb = np.asarray(dataset[index])
    elif representation == "jpeg":
        encoded = np.asarray(dataset[index], dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to decode left-eye JPEG at {index}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unknown RGB encoding: {representation}")
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected uint8 RGB, not depth, mask or RGBA")
    return rgb


def validate(h5, *, joint_names, camera_path, resolution, q_min, q_max, width_limits, seconds_per_step=0.05):
    joints = tuple(h5["meta/joint_names"].asstr()[...])
    metadata = h5["meta"].attrs
    if (joints != tuple(joint_names) or metadata.get("camera_path") != camera_path
            or metadata.get("arm_side") != "right" or metadata.get("joint_unit") != "rad"
            or metadata.get("width_unit") != "m"):
        raise ValueError("Wrong left-eye, arm, order or units")
    action_shape = h5["action/qpos_target"].shape
    if len(action_shape) != 2 or action_shape[1] != len(joints) or action_shape[0] < 1:
        raise ValueError("Unexpected joint command shape")
    steps = action_shape[0]
    ts = _numeric(h5, "obs/timestamp", (steps + 1,))
    if not np.allclose(np.diff(ts), seconds_per_step, rtol=0, atol=1e-5):
        raise ValueError("RGB/state/action clock is not consecutive")
    frame_ids = h5["obs/frame_id"].asstr()[...]
    if frame_ids.shape != (steps + 1,) or not all(frame_ids) or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Frame identity is stale or missing")
    qpos = _numeric(h5, "obs/qpos", (steps + 1, len(joints)))
    action = _numeric(h5, "action/qpos_target", (steps, len(joints)))
    actual_gap = _numeric(h5, "obs/gripper_width", (steps + 1,))
    gap_command = _numeric(h5, "action/gripper_width_target", (steps,))
    _numeric(h5, "action/native_command", (steps, len(joints) + 2))
    if np.any(action < np.asarray(q_min)) or np.any(action > np.asarray(q_max)):
        raise ValueError("Action command violates the arm limits")
    if np.any(qpos < np.asarray(q_min) - 0.02) or np.any(qpos > np.asarray(q_max) + 0.02):
        raise ValueError("Measured arm exceeds the physical position envelope")
    if (np.any(gap_command < width_limits[0]) or np.any(gap_command > width_limits[1])
            or np.any(actual_gap < width_limits[0] - 0.002) or np.any(actual_gap > width_limits[1] + 0.002)):
        raise ValueError("Gripper width inconsistent with measured metre limits")
    elements = h5["obs/rgb_left"]
    width, height = resolution
    if len(elements) != steps + 1 or any(read_rgb(h5, index).shape != (height, width, 3) for index in (0, steps)):
        raise ValueError("Raw RGB resolution or T+1 length mismatch")
    phases = h5["phase"].asstr()[...]
    flags = np.asarray(h5["train_action_valid"])
    if phases.shape != (steps,) or flags.shape != (steps,) or flags.dtype != np.bool_:
        raise ValueError("Action-phase and eligibility shape mismatch")
    expected = np.array([phase in TRAIN_PHASES for phase in phases])
    if not set(phases) <= TRAIN_PHASES | NO_TRAIN_PHASES or not np.array_equal(flags, expected) or not flags[0]:
        raise ValueError("Invalid training labels, missing first action or trial-lift leakage")
    first_invalid = np.flatnonzero(~flags)
    if not len(first_invalid) or np.any(flags[first_invalid[0]:]):
        raise ValueError("Expected a contiguous pre-lift trajectory and a masked final trial")
    bbox = _numeric(h5, "policy_prompt/bbox", (steps + 1, 4))
    point = _numeric(h5, "policy_prompt/point", (steps + 1, 2))
    prompt_valid = np.asarray(h5["policy_prompt/valid"])
    if prompt_valid.shape != (steps + 1,) or prompt_valid.dtype != np.bool_ or not prompt_valid[0]:
        raise ValueError("Initial prompt is not valid")
    if (np.any(bbox[prompt_valid] < 0) or np.any(bbox[prompt_valid] > 1)
            or np.any(bbox[prompt_valid, :2] >= bbox[prompt_valid, 2:]) or np.any(point[prompt_valid] < 0)
            or np.any(point[prompt_valid] > 1)):
        raise ValueError("Normalized RGB prompt is malformed")
    return EpisodeInfo(steps, int(flags.sum()), 1 / seconds_per_step)


def policy_inputs(h5, index, *, min_crop_px, previous_crop=None, prompt_step=None):
    prompt_step = index if prompt_step is None else prompt_step
    if not 0 <= prompt_step <= index or not bool(h5["policy_prompt/valid"][prompt_step]):
        raise ValueError("Observation has no valid box/point; apply the tracker failure policy")
    rgb = read_rgb(h5, index)
    first, second, crop = encode_views(rgb, h5["policy_prompt/bbox"][prompt_step], h5["policy_prompt/point"][prompt_step],
                                       min_crop_px=min_crop_px, previous=previous_crop)
    indices = [max(0, index - 1), index]
    state = np.stack([np.r_[h5["obs/qpos"][row], h5["obs/gripper_width"][row]] for row in indices])
    return {"global_view": first.channels, "local_view": second.channels, "recent_state": state,
            "crop": crop, "raw_to_global": first.raw_to_view, "raw_to_local": second.raw_to_view}
