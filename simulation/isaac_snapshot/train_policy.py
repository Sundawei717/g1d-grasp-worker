import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from chunk_model import SharedResNet18ChunkTransformer, masked_chunk_loss
from dataset_contract import policy_inputs
from prompt_views import PROMPT_AGE_CAP_S
from school_planner import sha256


GROUP_WEIGHTS = {"A_target": 0.40, "B_obstacle": 0.35, "C_start": 0.25}
MOTION_SCALE_FLOOR = np.asarray([0.001] * 7 + [0.0001])
PROMPT_TRAINING_MODES = ("visible_only", "hold_last_reliable")


def observations(h5, step, prompt_step=None):
    source = step if prompt_step is None else prompt_step
    inputs = policy_inputs(h5, step, min_crop_px=64, prompt_step=source)
    camera = json.loads(h5["meta/camera_json"].asstr()[()])
    width, height = camera["resolution"]
    crop = inputs["crop"]
    age = float(h5["obs/timestamp"][step] - h5["policy_prompt/updated_at"][source])
    if prompt_step is not None:
        age = min(age, PROMPT_AGE_CAP_S)
    prompt_meta = np.array([crop.center_xy[0] / (width - 1), crop.center_xy[1] / (height - 1),
                            crop.side_px / (width - 1), crop.side_px / (height - 1), 1.0, age], dtype=np.float32)
    return inputs["global_view"], inputs["local_view"], inputs["recent_state"], prompt_meta


def statistics(values):
    stacked = np.concatenate(values)
    return {"mean": stacked.mean(axis=0).tolist(),
            "scale": np.maximum(stacked.std(axis=0), MOTION_SCALE_FLOOR).tolist()}


def normalize(value, stats):
    return np.asarray((value - np.asarray(stats["mean"])) / np.asarray(stats["scale"]), dtype=np.float32)


class Windows(Dataset):
    def __init__(self, episodes, action_reference="absolute", prompt_training="visible_only"):
        if prompt_training not in PROMPT_TRAINING_MODES:
            raise ValueError(f"Unknown training prompt mode: {prompt_training}")
        self.files, self.windows, self.weights, self.diagnostics = [], [], [], []
        self.prompt_sources = []
        self.action_reference = action_reference
        action_values, state_values = [], []
        group_sizes = Counter(row["group"] for row in episodes)
        for row in episodes:
            path = Path(row["path"])
            if sha256(path) != row["sha256"]:
                raise ValueError(f"Training HDF5 changed: {path}")
            with h5py.File(path, "r") as h5:
                metadata = json.loads(h5["meta/case_json"].asstr()[()])
                if metadata["split"] != "train" or metadata["case_id"] != row["case"]:
                    raise ValueError(f"Episode has incorrect training scope: {path}")
                action = np.c_[h5["action/qpos_target"][...], h5["action/gripper_width_target"][...]]
                observed = np.c_[h5["obs/qpos"][...], h5["obs/gripper_width"][...]]
                flags = h5["train_action_valid"][...]
                stop = int(np.flatnonzero(~flags)[0])
                if np.any(flags[stop:]):
                    raise ValueError(f"Training actions are not contiguous: {path}")
                prompt_valid = h5["policy_prompt/valid"][:-1]
                valid = np.flatnonzero(flags & prompt_valid)
                if not len(valid) or valid[0] != 0:
                    raise ValueError(f"Episode is missing its initial training window: {path}")
                phases = h5["phase"].asstr()[...]
            episode_id = len(self.files)
            step_ids = np.arange(stop)
            last_visible = np.maximum.accumulate(np.where(prompt_valid[:stop], step_ids, -1))
            if prompt_training == "hold_last_reliable":
                valid = step_ids
                window_stops = np.full(len(valid), stop, dtype=int)
                supported_actions = flags[:stop]
                self.prompt_sources.append(last_visible)
            else:
                expired = np.flatnonzero(step_ids - last_visible > 4)
                window_stops = np.minimum(stop, np.r_[expired, stop][np.searchsorted(expired, valid)])
                supported_actions = step_ids - last_visible <= 4
                self.prompt_sources.append(None)
            if action_reference == "measured":
                action_values.extend(action[step:min(window_end, step + 8)] - observed[step]
                                     for step, window_end in zip(valid, window_stops))
            else:
                action_values.append(action[:stop][supported_actions])
            state_values.append(observed[valid])
            self.files.append((path, action, observed, phases))
            offset = len(self.windows)
            self.windows.extend((episode_id, int(step), int(window_end)) for step, window_end in zip(valid, window_stops))
            self.weights.extend([GROUP_WEIGHTS[row["group"]] / group_sizes[row["group"]] / len(valid)] * len(valid))
            for phase in ("pre_grasp", "approach", "close"):
                matching = np.flatnonzero(phases[valid] == phase)
                if len(matching):
                    self.diagnostics.append(offset + int(matching[len(matching) // 2]))
        self.norm = {"action": statistics(action_values), "state": statistics(state_values)}

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, step, stop = self.windows[index]
        path, action, observed, phases = self.files[episode]
        sources = self.prompt_sources[episode]
        with h5py.File(path, "r") as h5:
            first, second, state, metadata = observations(h5, step, None if sources is None else int(sources[step]))
        count = min(stop - step, 8)
        padded = np.zeros((8, 8), dtype=np.float32)
        targets = action[step:step + count]
        if self.action_reference == "measured":
            targets = targets - observed[step]
        padded[:count] = normalize(targets, self.norm["action"])
        return first, second, normalize(state, self.norm["state"]), metadata, padded, np.arange(8) < count, phases[step]


def atomic_checkpoint(path, payload):
    stage = path.with_suffix(".tmp")
    torch.save(payload, stage)
    stage.replace(path)


@torch.inference_mode()
def diagnostics(model, windows, device):
    model.eval()
    sums = {}
    for data in DataLoader(windows, sampler=windows.diagnostics, batch_size=4):
        first, second, state, metadata, labels, _, phases = data
        prediction = model(first.to(device), second.to(device), state.to(device), metadata.to(device)).cpu().numpy()
        delta = np.abs(prediction[:, 0] - labels[:, 0].numpy()) * np.asarray(windows.norm["action"]["scale"])
        for label, error in zip(phases, delta):
            sums.setdefault(label, []).append([float(error[:-1].mean()), float(error[-1] * 1000)])
    result = {phase: {"mean_joint_error_rad": float(np.mean(values, axis=0)[0]),
                      "mean_jaw_error_mm": float(np.mean(values, axis=0)[1]), "windows": len(values)}
              for phase, values in sums.items()}
    model.train()
    return result


def corrected_episodes(episodes, corrections, visibility_replay):
    episodes = list(episodes)
    use = {}
    original = {row["case"]: row for row in episodes}
    if visibility_replay:
        verified = json.loads(visibility_replay.read_text())
        marker = json.loads(visibility_replay.parent.with_suffix(".complete").read_text())
        if (marker["exit_code"] != 0 or verified["strict_success"] != 1 or verified["passed"] is not True
                or verified["case"] not in original):
            raise ValueError("The visibility-aware expert replay did not pass")
        key = verified["case"]
        shard = visibility_replay.parent.with_name(visibility_replay.parent.name + "_carton")
        path = shard / key / "episode.h5"
        episodes = [{**row, "path": str(path.resolve()), "sha256": sha256(path)} if row["case"] == key else row
                    for row in episodes]
        use["visibility_replay_sha256"] = sha256(visibility_replay)
    if corrections:
        result = json.loads(corrections.read_text())
        marker = json.loads(corrections.parent.with_suffix(".complete").read_text())
        if result.get("phase") != "policy_correction_capture":
            raise ValueError("Incorrect corrective data provenance")
        if marker["exit_code"] == 0 and result["passed"]:
            accepted = result["episodes"]
            if (any(row["parent_case"] not in original or row["scene_family"] != original[row["parent_case"]]["scene_family"]
                    for row in accepted) or len({row["case"] for row in (*episodes, *accepted)}) != len(episodes) + len(accepted)):
                raise ValueError("Corrective HDF5 cases leave the original training scene families")
            episodes.extend(accepted)
        use["corrections_sha256"] = sha256(corrections)
    return episodes, use


def train(args):
    receipt = json.loads(args.dataset.read_text())
    marker = json.loads(args.dataset.parent.with_suffix(".complete").read_text())
    if receipt["phase"] != "training_bootstrap" or not receipt["passed"] or marker["exit_code"] != 0:
        raise ValueError("Small-sample fitting requires the completed independent training bootstrap")
    episodes = [row for row in receipt["episodes"] if row["use_for_fit20"]]
    if len(episodes) != 20 or len({row["case"] for row in episodes}) != 20:
        raise ValueError("Expected 20 frozen, distinct training episodes")
    episodes, supplements = corrected_episodes(episodes, args.corrections, args.visibility_replay)
    if not 0 < args.steps <= 30000 or args.microbatch < 1 or 32 % args.microbatch:
        raise ValueError("The microbatch must divide the effective batch of 32")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    windows = Windows(episodes, args.action_reference, args.prompt_training)
    model = SharedResNet18ChunkTransformer(weights_path=args.weights).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": [param for name, param in model.named_parameters() if not name.startswith("backbone.")], "lr": 1e-4},
    ], weight_decay=1e-4)
    config = {"dataset_sha256": sha256(args.dataset), "weights_sha256": model.weight_sha256, "seed": args.seed,
              "effective_batch": 32, "microbatch": args.microbatch, "sampled_windows": len(windows),
              "action_reference": args.action_reference, **supplements}
    if args.prompt_training != "visible_only":
        config["prompt_training"] = args.prompt_training
    checkpoint_path = args.output / "checkpoint.pt"
    start = 0
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        resume_config = {"action_reference": "absolute", **checkpoint["config"]}
        if resume_config != config or checkpoint["normalization"] != windows.norm:
            raise ValueError("Cannot resume from a different data, weights or batch configuration")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.random.set_rng_state(checkpoint["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["device_rng"], device)
        start = checkpoint["step"]
        if start > args.steps:
            raise ValueError("The checkpoint is beyond the requested update count")
    indices = torch.multinomial(torch.as_tensor(windows.weights, dtype=torch.double), args.steps * 32,
                                replacement=True, generator=torch.Generator().manual_seed(args.seed + 1))
    loader = DataLoader(windows, sampler=indices[start * 32:].tolist(), batch_size=args.microbatch,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(args.seed + 2))
    per_step = 32 // args.microbatch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for batch, (first, second, state, metadata, labels, mask, _) in enumerate(loader, 1):
        values = [item.to(device, non_blocking=True) for item in (first, second, state, metadata, labels, mask)]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = masked_chunk_loss(model(*values[:4]), values[4], values[5])
        if not torch.isfinite(loss):
            raise ValueError("Non-finite normalized action loss")
        (loss / per_step).backward()
        if batch % per_step:
            continue
        step = start + batch // per_step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 100 == 0:
            print(f"{step}/{args.steps} loss={loss.item():.6f}", flush=True)
        if step % 500 == 0 or step == args.steps:
            checkpoint = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "normalization": windows.norm, "config": config, "rng": torch.random.get_rng_state(),
                          "device_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
            atomic_checkpoint(checkpoint_path, checkpoint)
    result = {"phase": "fit20_in_sample", "complete_optimization": True, "updates": args.steps,
              "episodes": len(episodes), "base_episodes": 20, "config": config, "normalization": windows.norm,
              "in_sample_action_errors": diagnostics(model, windows, device),
              "closed_loop_evaluated": False, "checkpoint": str(checkpoint_path.resolve())}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for flag in ("dataset", "weights", "output"):
        parser.add_argument(flag, type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--steps", default=2000, type=int)
    parser.add_argument("--seed", default=33889, type=int)
    parser.add_argument("--microbatch", default=4, type=int)
    parser.add_argument("--workers", default=3, type=int)
    parser.add_argument("--action-reference", choices=("absolute", "measured"), default="absolute")
    parser.add_argument("--prompt-training", choices=PROMPT_TRAINING_MODES, default="visible_only")
    parser.add_argument("--corrections", type=Path)
    parser.add_argument("--visibility-replay", type=Path)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
