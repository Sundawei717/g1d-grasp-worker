"""Finite-horizon samples shared by custom models and the V3-A adapter."""
from bisect import bisect_right
from itertools import accumulate
import operator

import h5py
import numpy as np

from .episode import RGB_PATHS, read_episode
from .manifest import episode_path


def match_receipt(episode, row):
    for field in ("case_id","asset_family","part"):
        if episode.case.get(field) != row[field]:
            raise ValueError(f"{row['case_id']}: HDF5 {field} differs from manifest")
    if episode.steps != row["steps"]:
        raise ValueError(f"{row['case_id']}: HDF5 transition count differs from manifest")


class WindowDataset:
    """Indexable NumPy dataset directly compatible with PyTorch DataLoader.

    Each valid transition is one index. History pads with the first observation.
    Future actions stop at lift or the end of the configured horizon.
    """

    def __init__(self, manifest, data_root, *, split="train", horizon=8, history=2, verify_sha256=False):
        if type(horizon) is not int or type(history) is not int or horizon < 1 or history < 1:
            raise ValueError("horizon and history must be positive integers")
        if split not in ("train","validation","test"):
            raise ValueError("select train, validation or test explicitly")
        self.split = split
        self.horizon = horizon
        self.history = history
        self._records = []
        self.joint_names = None
        for row in manifest.select(split):
            path = episode_path(data_root,row)
            episode = read_episode(path,sha256=row["sha256"] if verify_sha256 else None)
            match_receipt(episode,row)
            if self.joint_names is not None and self.joint_names != episode.joint_names:
                raise ValueError("joint order differs within split")
            self.joint_names = episode.joint_names
            self._records.append((path,episode,row["episode_id"]))
        if not self._records:
            raise ValueError(f"no episodes selected for split {split}")
        self._ends = list(accumulate(int(e.valid.sum()) for _,e,_ in self._records))

    def __len__(self):
        return self._ends[-1]

    def iter_episodes(self):
        """Iterate verified metadata and numerical trajectories, without loading RGB."""
        for _, episode, _ in self._records:
            yield episode

    def __getitem__(self, idx):
        idx = operator.index(idx)
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        i = bisect_right(self._ends,idx)
        step = idx - (self._ends[i-1] if i else 0)
        path, episode, episode_id = self._records[i]
        valid_end = self._ends[i] - (self._ends[i-1] if i else 0)
        end = min(valid_end,step+self.horizon)
        count = end-step
        previous = np.maximum(0,np.arange(step-self.history+1,step+1))
        actions = np.zeros((self.horizon,8),dtype=np.float32)
        actions[:count] = episode.actions[step:end]
        valid = np.arange(self.horizon) < count
        close = np.zeros(self.horizon,dtype=bool)
        close[:count] = episode.phases[step:end] == "close"
        with h5py.File(path,"r") as h5:
            rgb = {name:h5[key][step] for name,key in RGB_PATHS.items()}
        return {"episode_id":episode_id,
                "observation_step":step, "timestamp_s":float(episode.timestamps[step]),
                "rgb":rgb,
                "prompt":{"bbox":np.asarray(episode.prompt["bbox_0"],dtype=np.float64),
                          "point":np.asarray(episode.prompt["affordance_0"],dtype=np.float64),
                          "target_query":episode.prompt["target_query"]},
                "history":{"steps":previous,"qpos_rad":episode.qpos[previous],
                           "gripper_width_m":episode.gripper_width_m[previous],
                           "eef_pose_xyz_xyzw":episode.eef_pose_xyz_xyzw[previous]},
                "actions":actions, "action_mask":valid, "close_mask":close}
