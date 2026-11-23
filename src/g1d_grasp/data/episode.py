"""Read and validate g1d_v3_step_aligned HDF5 archives."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

SCHEMA = "g1d_v3_step_aligned"
CONTROL_HZ = 20
TRAIN_PHASES = ("approach", "close")
RGB_PATHS = {
    "head": "obs/rgb_left",
    "left_wrist": "obs/left_wrist/rgb",
    "right_wrist": "obs/right_wrist/rgb",
}


@dataclass(frozen=True)
class Episode:
    case: dict
    prompt: dict
    phases: np.ndarray
    valid: np.ndarray
    qpos: np.ndarray
    gripper_width_m: np.ndarray
    eef_pose_xyz_xyzw: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray
    joint_names: tuple[str, ...]

    @property
    def steps(self):
        return len(self.phases)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(h5, key):
    return json.loads(h5[key].asstr()[()])


def _array(h5, key, shape):
    values = h5[key][:]
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"{key} must be finite with shape {shape}, found {values.shape}")
    return values


def read_episode(path, *, sha256=None, verify_native=False):
    """Load metadata and CPU arrays. Images stay in HDF5 and load per sample."""
    path = Path(path)
    try:
        if sha256 is not None and file_sha256(path) != sha256:
            raise ValueError("SHA-256 mismatch")
        with h5py.File(path, "r") as h5:
            return _episode(h5, verify_native=verify_native)
    except (KeyError, ValueError, TypeError, OSError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _episode(h5, *, verify_native):
    if h5.attrs.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA}")
    phases = h5["phase"].asstr()[:]
    n = len(phases)
    if not 0 < n <= 300:
        raise ValueError("expected 1-300 transitions at 20 Hz")
    seq = {k: np.flatnonzero(phases == k) for k in (*TRAIN_PHASES, "lift")}
    if (any(not len(ids) for ids in seq.values())
            or not np.array_equal(np.concatenate(tuple(seq.values())), np.arange(n))):
        raise ValueError("expected consecutive approach, close and lift phases")
    valid = h5["train_action_valid"][:]
    if valid.dtype != np.dtype(bool) or not np.array_equal(valid, np.isin(phases, TRAIN_PHASES)):
        raise ValueError("train_action_valid must preserve all approach and close actions")
    if (not np.array_equal(h5["obs/control_step_id"][:], np.arange(n+1))
            or not np.array_equal(h5["action/control_step_id"][:], np.arange(n))):
        raise ValueError("nonconsecutive control_step_id")
    ts = _array(h5, "obs/timestamp", (n+1,))
    if not np.allclose(np.diff(ts), 1/CONTROL_HZ, rtol=0, atol=2e-7):
        raise ValueError("expected a 20 Hz observation clock")
    head_ids = h5["obs/frame_id"].asstr()[:]
    if head_ids.shape != (n+1,) or len(set(head_ids)) != n+1:
        raise ValueError("invalid head frame IDs")
    for camera, rgb_key in RGB_PATHS.items():
        rgb = h5[rgb_key]
        if rgb.shape != (n+1,224,224,3) or rgb.dtype != np.uint8 or rgb.compression != "gzip":
            raise ValueError(f"expected lossless (T+1,224,224,3) RGB for {camera}")
        if camera != "head":
            wrist_ts = _array(h5, f"obs/{camera}/timestamp", (n+1,))
            wrist_ids = h5[f"obs/{camera}/frame_id"].asstr()[:]
            if (not np.allclose(wrist_ts, ts, rtol=0, atol=1e-5)
                    or wrist_ids.shape != (n+1,) or len(set(wrist_ids)) != n+1):
                raise ValueError(f"{camera} frame sequence is out of alignment")
    case = _read_json(h5, "meta/case_json")
    prompt = _read_json(h5, "meta/initial_prompt_json")
    bbox = np.asarray(prompt["bbox_0"], dtype=np.float64)
    point = np.asarray(prompt["affordance_0"], dtype=np.float64)
    if (prompt["frame_id"] != head_ids[0] or not prompt["target_query"] or not prompt.get("prompt_valid", True)
            or bbox.shape != (4,) or point.shape != (2,) or not np.isfinite(np.r_[bbox, point]).all()
            or np.any(np.r_[bbox, point] < 0) or np.any(np.r_[bbox, point] > 1) or np.any(bbox[:2] >= bbox[2:])):
        raise ValueError("invalid first-frame bbox, grasp point or frame ID")
    if _read_json(h5, "meta/strict_score_json").get("success") is not True:
        raise ValueError("archive must have a successful lift score")
    qpos = _array(h5, "obs/qpos", (n+1,7))
    width = _array(h5, "obs/gripper_width", (n+1,))
    pose = _array(h5, "obs/eef_pose", (n+1,7))
    if np.any(np.linalg.norm(pose[:,3:], axis=1) <= 1e-7):
        raise ValueError("invalid wrist quaternion")
    actions = np.column_stack((_array(h5, "action/qpos_target", (n,7)),
                               _array(h5, "action/gripper_width_target", (n,))))
    names = tuple(h5["meta/joint_names"].asstr()[:])
    if len(names) != 7 or len(set(names)) != 7:
        raise ValueError("expected seven unique commanded joints")
    if verify_native:
        native = _array(h5, "action/native_command", (n,9))
        substeps = _array(h5, "action/native_substep_command", (n,30,9))
        after = seq["lift"][0]
        if (not np.array_equal(substeps[:,-1], native)
                or not np.allclose(substeps[after:,:,7:], substeps[after-1,-1,7:], atol=1e-7, rtol=0)):
            raise ValueError("invalid 600 Hz command binding or open-width change after lift")
    return Episode(case, prompt, phases, valid, qpos, width, pose, actions, ts, names)
