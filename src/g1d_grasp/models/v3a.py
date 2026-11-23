"""Adapter between model-neutral recorded windows and a V3-A checkpoint."""
import numpy as np
import torch

from .inputs import head_input, proprioception, wrist_input
from .network import SharedResNet18ChunkTransformer

ACTION_FLOOR = np.asarray([.001]*7+[.0001])
STATE_FLOOR = np.asarray([.001]*7+[.0001]*4+[.001]*6)


def fit_normalization(train_dataset):
    """Fit physical-value statistics on the training split only."""
    states, actions = [], []
    if train_dataset.split != "train":
        raise ValueError("normalization must be fitted on the training split")
    for ep in train_dataset.iter_episodes():
        state = proprioception(ep.qpos,ep.gripper_width_m,ep.eef_pose_xyz_xyzw)
        states.append(state[:ep.steps][ep.valid])
        actions.append(ep.actions[ep.valid])

    def stats(items, floor):
        values = np.concatenate(items)
        return {"mean":values.mean(0).tolist(),"scale":np.maximum(values.std(0),floor).tolist()}

    return {"state":stats(states,STATE_FLOOR),"action":stats(actions,ACTION_FLOOR)}


def _normalize(array, stats):
    return ((array-np.asarray(stats["mean"]))/np.asarray(stats["scale"])).astype(np.float32)


def encode_sample(sample, normalizer):
    """Return the five tensors consumed by V3-A, plus normalized labels and masks."""
    images, prompt, history = sample["rgb"],sample["prompt"],sample["history"]
    state = proprioception(history["qpos_rad"],history["gripper_width_m"],history["eef_pose_xyz_xyzw"])
    if state.shape != (2,17) or sample["actions"].shape != (8,8):
        raise ValueError("V3-A requires history=2 and horizon=8")
    head, meta = head_input(images["head"],prompt["bbox"],prompt["point"],point_geometry=True)
    inputs = (head, wrist_input(images["left_wrist"],detection_channel=False),
              _normalize(state,normalizer["state"]), meta,
              wrist_input(images["right_wrist"],detection_channel=False))
    mask = sample["action_mask"]
    actions = np.zeros((8,8),dtype=np.float32)
    actions[mask] = _normalize(sample["actions"][mask],normalizer["action"])
    return inputs, actions, mask, sample["close_mask"]


def load_checkpoint(path, device="cpu"):
    """Load a V3-A query-bbox checkpoint with its original normalization."""
    saved = torch.load(path,map_location="cpu",weights_only=True)
    cfg = saved["config"]
    if (cfg.get("architecture") != "v3_final" or cfg.get("wrist_channels") != 3
            or not all(cfg.get(k) for k in ("prompt_fusion","point_queries","query_bbox"))):
        raise ValueError("expected the final 3-channel V3-A box-conditioned checkpoint")
    model = SharedResNet18ChunkTransformer(fixture_random_init=True,wrist_channels=3,v3_final=True,
                                            freeze_bn=cfg.get("freeze_bn",False),prompt_fusion=True,
                                            point_queries=True,query_bbox=True)
    model.load_state_dict(saved["model"],strict=True)
    return model.to(device).eval(), saved["normalization"]


@torch.inference_mode()
def predict(model, sample, normalizer):
    """Run one forward pass. Return eight absolute joint/jaw targets in rad and m."""
    inputs,_,_,_ = encode_sample(sample,normalizer)
    device = next(model.parameters()).device
    batch = [torch.as_tensor(x,dtype=torch.float32,device=device)[None] for x in inputs]
    raw = model(*batch)[0].cpu().numpy()
    scale = np.asarray(normalizer["action"]["scale"])
    center = np.asarray(normalizer["action"]["mean"])
    commands = raw*scale + center
    if commands.shape != (8,8) or not np.isfinite(commands).all():
        raise ValueError("invalid V3-A action chunk")
    return commands
