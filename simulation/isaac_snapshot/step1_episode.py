import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from episode_store import jaw_width_m
from three_view_detector import sha256
from three_view_inputs import rgb224


VIEWS = {"head_left": "obs/rgb_left", "left_wrist": "obs/left_wrist/rgb", "right_wrist": "obs/right_wrist/rgb"}


def orange_mug_reference(rgb):
    red, green, blue = rgb.astype(np.float32).transpose(2, 0, 1)
    mask = ((red > 1.38 * green + 4) & (green > 1.21 * blue + 2) & (red - blue > 27)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return {"bbox": [0.0] * 4, "pixels": 0}
    x, y, w, h, pixels = max(stats[1:], key=lambda row: row[cv2.CC_STAT_AREA])
    height, width = rgb.shape[:2]
    return {"bbox": [x / width, y / height, (x + w) / width, (y + h) / height], "pixels": int(pixels)}


class AlignedEpisode:
    def __init__(self, case, joint_names, cameras, prompt):
        self.case = case
        self.joint_names = joint_names
        self.cameras = cameras
        self.prompt = prompt
        self.observations = []
        self.actions = []

    def observe(self, frame, arm, width, wrist):
        if len(self.actions) != len(self.observations):
            raise ValueError("Expected a single observation per 20 Hz transition")
        rgb, info = frame
        images = {"head_left": (rgb, info), **info["wrist_views"]}
        resized = {name: rgb224(value) for name, (value, _) in images.items()}
        self.observations.append({
            "sim_time_s": info["sensor_sim_time_s"], "arm": arm, "width": width, "wrist": wrist,
            "ids": {name: f"{meta['reference_time']['numerator']}/{meta['reference_time']['denominator']}"
                    for name, (_, meta) in images.items()},
            "times": {name: meta["sensor_sim_time_s"] for name, (_, meta) in images.items()},
            "images": resized,
        })

    def action(self, phase, native):
        commands = np.asarray(native, dtype=np.float32)
        if (len(self.observations) != len(self.actions) + 1 or commands.shape != (30, 9)
                or not np.isfinite(commands).all()):
            raise ValueError("Control requires an observation and all thirty applied native targets")
        self.actions.append((phase, commands))

    def save(self, path, score):
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        try:
            rows = self.observations
            step_count = len(self.actions)
            if len(rows) != step_count + 1 or not score["success"]:
                raise ValueError("An accepted demonstration needs complete observations and verified lift")
            stamps = [row["sim_time_s"] for row in rows]
            if not np.allclose(np.diff(stamps), 0.05, atol=2e-7, rtol=0):
                raise ValueError("Observation timestamps are not a continuous 20 Hz clock")
            native_phase = [phase for phase, _ in self.actions]
            phases = [phase if phase in {"approach", "close"} else "lift" for phase in native_phase]
            rank = {"approach": 0, "close": 1, "lift": 2}
            if set(phases) != rank.keys() or phases != sorted(phases, key=rank.get):
                raise ValueError("Incorrect approach/close/lift phase order")
            native = np.stack([value for _, value in self.actions])
            targets = native[:, -1]
            with h5py.File(temporary, "x") as f:
                f.attrs.update(schema="g1d_v3_step_aligned", control_hz=20, servo_hz=600,
                               rgb_resampling="cv2.INTER_AREA to 224x224 before lossless HDF5 write")
                f["obs/control_step_id"] = np.arange(step_count + 1)
                f["action/control_step_id"] = np.arange(step_count)
                f["obs/timestamp"], f["obs/qpos"] = stamps, [x["arm"] for x in rows]
                f["obs/gripper_width"], f["obs/eef_pose"] = [x["width"] for x in rows], [x["wrist"] for x in rows]
                for name, dataset in VIEWS.items():
                    value = np.stack([x["images"][name] for x in rows])
                    if value.shape != (step_count + 1, 224, 224, 3) or value.dtype != np.uint8:
                        raise ValueError(f"Invalid RGB frames: {name}")
                    f.create_dataset(dataset, data=value, chunks=(1, 224, 224, 3),
                                     compression="gzip", compression_opts=2).attrs["encoding"] = "raw"
                    prefix = "obs" if name == "head_left" else f"obs/{name}"
                    view_times = [x["times"][name] for x in rows]
                    ids = [x["ids"][name] for x in rows]
                    if view_times != stamps or len(set(ids)) != len(rows):
                        raise ValueError(f"Out-of-step camera frames: {name}")
                    f[f"{prefix}/frame_id"] = ids
                    if name != "head_left":
                        f[f"{prefix}/timestamp"] = view_times
                f["action/qpos_target"] = targets[:, :7]
                f["action/gripper_width_target"] = [jaw_width_m(x) for x in targets[:, 7:]]
                f["action/native_command"] = targets
                f["action/native_substep_command"] = native
                f["action/native_phase"], f["phase"] = native_phase, phases
                f["train_action_valid"] = [phase != "lift" for phase in phases]
                f["meta/joint_names"] = list(self.joint_names)
                f["meta"].attrs.update(camera_path=self.cameras["head_left"]["prim_path"], arm_side="right",
                                       joint_unit="rad", width_unit="m", control_hz=20, servo_hz=600)
                f["meta/case_json"] = json.dumps(self.case)
                f["meta/strict_score_json"] = json.dumps(score)
                f["meta/three_view_cameras_json"] = json.dumps(self.cameras)
                f["meta/initial_prompt_json"] = json.dumps({**self.prompt, "head_bbox_0": self.prompt["bbox_0"],
                                                          "head_affordance_0": self.prompt["affordance_0"]})
                f.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return {"case": self.case["case_id"], "path": str(path.resolve()), "sha256": sha256(path),
                "steps": step_count, "bytes": path.stat().st_size}
