import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import h5py
import numpy as np

from dataset_contract import TRAIN_PHASES, read_rgb, validate


WRIST_VIEWS = ("left_wrist", "right_wrist")


def encode_rgb(rgb, resolution):
    width, height = resolution
    if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
        raise ValueError("Native RGB shape changed")
    ok, data = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Native JPEG encoding failed")
    return data


def write_rgb(h5, key, frames):
    rgb = h5.create_dataset(key, (len(frames),), dtype=h5py.vlen_dtype(np.uint8))
    rgb.attrs.update(encoding="jpeg", quality=95)
    for index, image in enumerate(frames):
        rgb[index] = image


def verify_three_view(h5):
    stamps = h5["obs/timestamp"][...]
    cameras = json.loads(h5["meta/three_view_cameras_json"].asstr()[()])
    prompt = json.loads(h5["meta/initial_prompt_json"].asstr()[()])
    aligned_rgb = h5.attrs.get("schema") == "g1d_v3_step_aligned"
    if prompt["frame_id"] != h5["obs/frame_id"].asstr()[0] or not prompt["target_query"]:
        raise ValueError("Initial prompt does not match the first head frame")
    for name in WRIST_VIEWS:
        base = f"obs/{name}"
        ts = h5[f"{base}/timestamp"][...]
        ids = h5[f"{base}/frame_id"].asstr()[...]
        # V3 keeps the native camera calibration and stores resampled RGB.
        width, height = (224, 224) if aligned_rgb else cameras[name]["resolution"]
        if (ts.shape != stamps.shape or not np.allclose(ts, stamps, atol=1e-5, rtol=0)
                or len(ids) != len(stamps) or len(set(ids)) != len(stamps)
                or len(h5[f"{base}/rgb"]) != len(stamps)
                or any(read_rgb(h5, i, f"{base}/rgb").shape != (height, width, 3) for i in (0, len(stamps)-1))):
            raise ValueError(f"Wrist RGB/clock is invalid for {name}")
    return len(stamps)


def jaw_width_m(finger_positions):
    # Pinned Dex1 tip-link separation is 90.07 mm at the two -20 mm joint limits.
    return 0.09007 - sum(position + 0.020 for position in finger_positions)


class EpisodeStore:
    def __init__(self, camera, joint_names, lower, upper, wrist_cameras=None, target_query=None):
        self.camera = camera
        self.wrist_cameras = wrist_cameras or {}
        self.target_query = target_query
        self.joint_names = joint_names
        self.lower = lower
        self.upper = upper
        self.observations = []
        self.commands = []

    def observation(self, *, rgb, timestamp, frame_id, arm, fingers, wrist, target, prompt=None, contacts=None,
                    extra_views=None):
        if len(self.observations) != len(self.commands):
            raise ValueError("A transition already has its initial observation")
        data = encode_rgb(rgb, self.camera["resolution"])
        views = {}
        for name, camera in self.wrist_cameras.items():
            image, frame = extra_views[name]
            stamp = frame["reference_time"]
            views[name] = (encode_rgb(image, camera["resolution"]), frame["sensor_sim_time_s"],
                           f'{stamp["numerator"]}/{stamp["denominator"]}')
        self.observations.append({"rgb": data, "time_s": timestamp, "frame_id": frame_id, "arm": arm,
                                  "width_m": jaw_width_m(fingers), "wrist": wrist, "target": target,
                                  "prompt": prompt, "contacts": contacts, "views": views})

    def command(self, phase, arm, fingers, native_substeps):
        if len(self.observations) != len(self.commands) + 1:
            raise ValueError("A command must follow its source observation")
        native = [*arm, *fingers]
        substeps = np.asarray(native_substeps)
        if (substeps.shape != (30, len(self.joint_names) + 2) or not np.isfinite(substeps).all()
                or not np.allclose(substeps[-1], native, atol=1e-6, rtol=0)):
            raise ValueError("Expected thirty 600 Hz commands ending at the actual control target")
        self.commands.append({"phase": phase, "arm": arm, "width_m": jaw_width_m(fingers),
                              "native": native, "native_substeps": substeps})

    def write_success(self, path, case, score):
        if score.get("success") is not True or len(self.observations) != len(self.commands) + 1:
            raise ValueError("An episode needs strict success and a complete T+1 observation sequence")
        fields = {"case_id", "seed", "asset_family", "scene_family", "split", "config_sha256", "part"}
        if not fields <= case.keys():
            raise ValueError("Collection needs frozen split, family, seed and configuration bindings")
        observations, commands = self.observations, self.commands
        path = Path(path)
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            with h5py.File(temporary, "w") as h5:
                h5["meta/joint_names"] = list(self.joint_names)
                metadata = h5["meta"]
                metadata.attrs.update(camera_path=self.camera["prim_path"], arm_side="right", joint_unit="rad",
                                      width_unit="m", control_hz=20, servo_hz=600)
                metadata["camera_json"] = json.dumps(self.camera)
                metadata["case_json"] = json.dumps(case)
                metadata["strict_score_json"] = json.dumps(score)
                write_rgb(h5, "obs/rgb_left", [row["rgb"] for row in observations])
                if self.wrist_cameras:
                    initial = observations[0]["prompt"]
                    metadata["initial_prompt_json"] = json.dumps({
                        "frame_id": observations[0]["frame_id"], "bbox_0": initial["bbox_0"],
                        "affordance_0": initial["affordance_0"], "target_query": self.target_query, "arm_side": "right"})
                    metadata["three_view_cameras_json"] = json.dumps({"head_left": self.camera, **self.wrist_cameras})
                    for name in WRIST_VIEWS:
                        frames = [row["views"][name] for row in observations]
                        base = f"obs/{name}"
                        write_rgb(h5, f"{base}/rgb", [row[0] for row in frames])
                        h5[f"{base}/timestamp"] = [row[1] for row in frames]
                        h5[f"{base}/frame_id"] = [row[2] for row in frames]
                h5["obs/timestamp"] = [row["time_s"] for row in observations]
                h5["obs/frame_id"] = [row["frame_id"] for row in observations]
                h5["obs/qpos"] = [row["arm"] for row in observations]
                h5["obs/gripper_width"] = [row["width_m"] for row in observations]
                h5["obs/eef_pose"] = [row["wrist"] for row in observations]
                h5["action/qpos_target"] = [row["arm"] for row in commands]
                h5["action/gripper_width_target"] = [row["width_m"] for row in commands]
                h5["action/native_command"] = [row["native"] for row in commands]
                h5["action/native_substep_command"] = [row["native_substeps"] for row in commands]
                phases = [row["phase"] for row in commands]
                h5["phase"] = phases
                h5["train_action_valid"] = [phase in TRAIN_PHASES for phase in phases]
                prompts = [row["prompt"] for row in observations]
                visible = [item is not None and item.get("prompt_valid", True) for item in prompts]
                h5["policy_prompt/bbox"] = [item["bbox_0"] if ok else [0.0] * 4 for item, ok in zip(prompts, visible)]
                h5["policy_prompt/point"] = [item["affordance_0"] if ok else [0.0] * 2 for item, ok in zip(prompts, visible)]
                h5["policy_prompt/valid"] = visible
                h5["policy_prompt/updated_at"] = [row["time_s"] if ok else -1 for row, ok in zip(observations, visible)]
                h5["policy_prompt/source"] = ["visible_truth" if ok else "unavailable" for ok in visible]
                h5["gt/target_pose_xyz_xyzw"] = [row["target"] for row in observations]
                h5["gt/part_visibility"] = visible
                h5["gt/part_anchor_local"] = [item["part_hit_local_xyz_m"] if item else [0.0] * 3 for item in prompts]
                h5["gt/part_anchor_projected"] = [(item.get("projected_anchor") or [0.0] * 2) if item else [0.0] * 2
                                                  for item in prompts]
                h5["gt/part_anchor_projection_valid"] = [item is not None and item.get("projected_anchor") is not None
                                                         for item in prompts]
                width, height = self.camera["resolution"]
                packed = np.zeros((width * height + 7) // 8, dtype=np.uint8)
                for name in ("target_mask_packed", "part_mask_packed"):
                    values = [item.get(name, packed) if item else packed for item in prompts]
                    dataset = h5.create_dataset(f"gt/{name}", data=values, compression="gzip", compression_opts=3)
                    dataset.attrs.update(raw_resolution=(width, height), bits="big", flatten_order="row_major")
                h5["gt/contact_flags"] = [row["contacts"] if row["contacts"] is not None
                                          else [False] * 4 for row in observations]
                h5["gt/contact_flags"].attrs["columns"] = ["supported", "pad_1", "pad_2", "forbidden"]
                h5["gt/part"] = case["part"]
                info = validate(h5, joint_names=self.joint_names, camera_path=self.camera["prim_path"],
                                resolution=self.camera["resolution"], q_min=self.lower, q_max=self.upper,
                                width_limits=(0.00106, 0.09008))
                if self.wrist_cameras:
                    verify_three_view(h5)
            with temporary.open("rb") as completed:
                os.fsync(completed.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return {"bytes": path.stat().st_size, "actions": info.actions,
                "valid_training_actions": info.valid_training_actions}
