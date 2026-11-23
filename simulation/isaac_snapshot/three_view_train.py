import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from chunk_model import SharedResNet18ChunkTransformer, masked_chunk_loss
from dataset_contract import read_rgb
from episode_store import WRIST_VIEWS, verify_three_view
from three_view_detector import sha256, write_json
from three_view_inputs import head_input, proprioception, wrist_input
from train_policy import GROUP_WEIGHTS, atomic_checkpoint, normalize, statistics


def make_split(collection, output):
    source = json.loads(collection.read_text())
    episodes = source["episodes"]
    v3 = source.get("phase") == "tier1_v3"
    family = lambda row: row.get("pair_group_id") or row["scene_family"]
    validation = set()
    for group in GROUP_WEIGHTS:
        families = {family(row) for row in episodes if row["group"] == group}
        if len(families) < 2:
            if v3 and not families:
                continue
            raise ValueError(f"Need independent train and validation families in {group}")
        ordered = sorted(families, key=lambda name: hashlib.sha256(name.encode()).digest())
        validation.update(ordered[:max(1, len(ordered) // 5)])
    if source.get("asset_balance"):
        strata = {(row["asset_family"], row.get("part", "upper_side_panels")) for row in episodes}
        for stratum in sorted(strata):
            families = {family(row) for row in episodes
                        if (row["asset_family"], row.get("part", "upper_side_panels")) == stratum}
            ordered = sorted(families, key=lambda name: hashlib.sha256(name.encode()).digest())
            if len(ordered) >= 2:
                validation.update(ordered[:max(1, len(ordered) // 5)])
            if families <= validation:
                validation.remove(ordered[-1])
    rows = [{**row, "split": "validation" if family(row) in validation else "train"} for row in episodes]
    write_json(output, {"phase": "three_view_v3_split" if v3 else "three_view_split",
                        "source_sha256": sha256(collection), "episodes": rows,
                        "asset_balance": bool(source.get("asset_balance")),
                        "assets_by_split": {split: Counter(row["asset_family"] for row in rows if row["split"] == split)
                                            for split in ("train", "validation")},
                        "parts_by_split": {split: Counter(row.get("part", "upper_side_panels") for row in rows if row["split"] == split)
                                            for split in ("train", "validation")}})


class ThreeViewWindows(Dataset):
    def __init__(self, manifest, split, boxes, norm=None, *, phase_metrics=False, point_geometry=False):
        self.files, self.windows, self.weights = [], [], []
        self.with_boxes = boxes is not None
        self.v3_final = manifest.get("phase") in {"tier1_v3", "three_view_v3_split"}
        self.close_starts = []
        self.phase_metrics = phase_metrics
        self.point_geometry = point_geometry
        rows = [row for row in manifest["episodes"] if row["split"] == split]
        counts = Counter(row["group"] for row in rows)
        assets = Counter((row["group"], row["asset_family"], row.get("part", "upper_side_panels")) for row in rows)
        asset_mass = Counter()
        if manifest.get("asset_balance"):
            for (group, *_), count in assets.items():
                asset_mass[group] += count ** 0.5
        action_values, state_values = [], []
        for row in rows:
            path = Path(row["path"])
            if sha256(path) != row["sha256"]:
                raise ValueError(f"RGB episode changed: {path}")
            detections = boxes[row["case"]] if self.with_boxes else None
            with h5py.File(path, "r") as h5:
                verify_three_view(h5)
                prompt = json.loads(h5["meta/initial_prompt_json"].asstr()[()])
                action = np.c_[h5["action/qpos_target"][...], h5["action/gripper_width_target"][...]]
                phases = h5["phase"].asstr()[...]
                flags = h5["train_action_valid"][...]
                if self.v3_final:
                    if (h5.attrs.get("schema") != "g1d_v3_step_aligned"
                            or not np.array_equal(h5["obs/control_step_id"][:], np.arange(len(action)+1))
                            or not np.array_equal(h5["action/control_step_id"][:], np.arange(len(action)))
                            or not np.array_equal(flags, np.isin(phases, ["approach", "close"]))):
                        raise ValueError(f"V3-Final requires aligned lossless RGB and all close labels: {path}")
                    state = proprioception(h5["obs/qpos"][:], h5["obs/gripper_width"][:], h5["obs/eef_pose"][:])
                else:
                    state = np.c_[h5["obs/qpos"][...], h5["obs/gripper_width"][...]]
                valid = np.flatnonzero(flags)
                if not len(valid) or not np.array_equal(valid, np.arange(len(valid))):
                    raise ValueError(f"Missing continuous action labels: {path}")
                if detections:
                    for name in (("right_wrist",) if self.v3_final else WRIST_VIEWS):
                        ids = h5[f"obs/{name}/frame_id"].asstr()[...].tolist()
                        if [item["frame_id"] for item in detections[name]] != ids:
                            raise ValueError(f"Wrist detection cache is misaligned: {path} {name}")
            index = len(self.files)
            self.files.append((path, prompt, state, action, detections, phases))
            for step in valid:
                if phases[step] == "close":
                    self.close_starts.append(len(self.windows))
                self.windows.append((index, int(step), min(len(valid), step + 8)))
                if not self.v3_final:
                    action_values.append(action[step:min(len(valid), step + 8)] - state[step])
            if self.v3_final:
                action_values.append(action[valid])
            state_values.append(state[valid])
            group = row["group"]
            part = row.get("part", "upper_side_panels")
            denominator = counts[group]
            if asset_mass:
                denominator = asset_mass[group] * assets[group, row["asset_family"], part] ** 0.5
            self.weights.extend([GROUP_WEIGHTS[group] / denominator / len(valid)] * len(valid))
        if not self.windows:
            raise ValueError(f"Empty {split} split")
        if norm is not None:
            self.norm = norm
        elif self.v3_final:
            stacked = np.concatenate(state_values)
            scale_floor = [.001]*7+[.0001]*4+[.001]*6
            self.norm = {"action": statistics(action_values), "state": {"mean": stacked.mean(0).tolist(),
                         "scale": np.maximum(stacked.std(0), scale_floor).tolist()}}
        else:
            self.norm = {"action": statistics(action_values), "state": statistics(state_values)}

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, step, end = self.windows[index]
        path, prompt, state, action, boxes, phases = self.files[episode]
        with h5py.File(path, "r") as h5:
            head, meta = head_input(read_rgb(h5, step), prompt["bbox_0"], prompt["affordance_0"],
                                    point_geometry=self.point_geometry)
            wrists = []
            for name in WRIST_VIEWS:
                mask = self.with_boxes and (not self.v3_final or name == "right_wrist")
                bbox = boxes[name][step]["bbox"] if mask else None
                wrists.append(wrist_input(read_rgb(h5, step, f"obs/{name}/rgb"), detection_channel=mask, bbox=bbox))
        history_indices = (max(0, step-1), step) if self.v3_final else (step, step)
        history = normalize(state[list(history_indices)], self.norm["state"])
        padded = np.zeros((8, 8), dtype=np.float32)
        commands = action[step:end] if self.v3_final else action[step:end] - state[step]
        padded[:end-step] = normalize(commands, self.norm["action"])
        sample = (head, wrists[0], history, meta, wrists[1], padded, np.arange(8) < end-step)
        if self.phase_metrics:
            close = np.zeros(8, dtype=bool)
            close[:end-step] = phases[step:end] == "close"
            return (*sample, close)
        return sample


def read_boxes(manifest, result):
    receipt = json.loads(result.read_text())
    entries = {row["case"]: row for row in receipt["episodes"]}
    if manifest.get("phase") in {"tier1_v3", "three_view_v3_split"}:
        from v3_detector import cache_key, detector_config, verify_episode_cache

        cfg = detector_config()
        if (not receipt.get("passed") or receipt["config"] != cfg or receipt["config_sha256"] != cache_key(cfg)
                or manifest.get("source_sha256", receipt["collection_sha256"]) != receipt["collection_sha256"]):
            raise ValueError("Training requires the pinned Tier-1 Swin-T BF16 cache")
        cases = {}
        for row in manifest["episodes"]:
            entry = entries[row["case"]]
            with h5py.File(row["path"], "r") as data:
                ids = data["obs/right_wrist/frame_id"].asstr()[:]
                query = json.loads(data["meta/initial_prompt_json"].asstr()[()])["target_query"]
                if entry["target_query"] != query:
                    raise ValueError(f"Wrong target text prompt in cache for {row['case']}")
                cases[row["case"]] = {"right_wrist": verify_episode_cache(row, entry, ids, cfg)}
        return cases, receipt["config_sha256"]
    cases = {}
    for row in manifest["episodes"]:
        entry = entries[row["case"]]
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("Wrist box cache changed")
        payload = json.loads(path.read_text())
        if payload["rgb_sha256"] != row["sha256"] or payload["config_sha256"] != receipt["config_sha256"]:
            raise ValueError("Wrist box cache points to another image or detector")
        cases[row["case"]] = payload["views"]
    return cases, receipt["config_sha256"]


def supervision(dataset, sampled):
    counts = np.bincount(sampled.numpy(), minlength=len(dataset))
    rows = [np.zeros(len(episode[-1]), dtype=np.int64) for episode in dataset.files]
    for n, (episode, start, end) in zip(counts, dataset.windows):
        rows[episode][start:end] += n
    phases = {}
    for episode, terms in zip(dataset.files, rows):
        labels = episode[-1]
        for phase in np.unique(labels):
            group = terms[labels == phase]
            totals = phases.setdefault(phase, {"recorded": 0, "sampled_unique": 0, "loss_terms": 0})
            totals["recorded"] += len(group)
            totals["sampled_unique"] += int(np.count_nonzero(group))
            totals["loss_terms"] += int(group.sum())
    return {"window_draws_sha256": hashlib.sha256(sampled.numpy().tobytes()).hexdigest(), "phases": phases}


def supervised_indices(dataset, count, seed):
    rng = torch.Generator().manual_seed(seed)
    required = torch.as_tensor(dataset.close_starts if dataset.v3_final else [], dtype=torch.long)
    if len(required) > count:
        raise ValueError(f"The requested schedule has {count} windows for {len(required)} required close steps")
    if count > len(required):
        remainder = torch.multinomial(torch.as_tensor(dataset.weights, dtype=torch.double), count-len(required),
                                     replacement=True, generator=rng)
        required = torch.cat((required, remainder))
    return required[torch.randperm(count, generator=rng)] if dataset.v3_final else required


def phase_loss(prediction, labels, mask, close):
    errors = F.smooth_l1_loss(prediction.float(), labels, reduction="none")
    joint, gripper = errors[..., :-1].mean(-1), errors[..., -1]
    joint_sum, gripper_sum = (joint * mask).sum(), (gripper * mask).sum()
    count = mask.sum()
    loss = (joint_sum + gripper_sum) / count
    metrics = torch.stack((joint_sum.detach(), gripper_sum.detach(), count,
                           ((joint.detach() + gripper.detach()) * close).sum(), close.sum()))
    return loss, metrics


def loss_record(totals):
    joint, gripper, count, close, close_count = totals.tolist()
    return {"total": (joint + gripper) / count, "joint": joint / count,
            "gripper": gripper / count, "close": close / close_count if close_count else None,
            "action_terms": int(count), "close_terms": int(close_count)}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    values = []
    totals = torch.zeros(5, device=device)
    phase_metrics = loader.dataset.phase_metrics
    for batch in loader:
        tensors = [item.to(device) for item in batch]
        if phase_metrics:
            *images_and_state, label, mask, close = tensors
            _, metrics = phase_loss(model(*images_and_state), label, mask, close)
            totals += metrics
        else:
            *images_and_state, label, mask = tensors
            prediction = model(*images_and_state)
            values.append((float(masked_chunk_loss(prediction, label, mask)), int(mask.sum())))
    model.train()
    if phase_metrics:
        return loss_record(totals)
    return sum(error * count for error, count in values) / sum(count for _, count in values)


def train(args):
    manifest = json.loads(args.manifest.read_text())
    boxes, detector_config = read_boxes(manifest, args.boxes) if args.variant == "B" else (None, None)
    v3 = manifest.get("phase") in {"tier1_v3", "three_view_v3_split"}
    training = ThreeViewWindows(manifest, "train", boxes, phase_metrics=v3)
    validation = ThreeViewWindows(manifest, "validation", boxes, training.norm, phase_metrics=v3)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    channels = 3 if args.variant == "A" else 4
    architecture = "v3_final" if training.v3_final else "legacy_three_view"
    train_options = {"v3_final": training.v3_final, "freeze_bn": args.freeze_bn}
    model = SharedResNet18ChunkTransformer(weights_path=args.weights, wrist_channels=channels, **train_options).to(device)
    parent = None
    if args.initialize_from:
        checkpoint = torch.load(args.initialize_from, map_location="cpu", weights_only=True)
        previous = checkpoint["config"]
        if (previous["variant"] != args.variant or previous["manifest_sha256"] != sha256(args.manifest)
                or previous.get("architecture", "legacy_three_view") != architecture
                or previous["detector_config_sha256"] != detector_config
                or checkpoint["normalization"] != training.norm):
            raise ValueError("Continuation differs from the original variant, data or detector")
        model.load_state_dict(checkpoint["model"], strict=True)
        parent = {"archived_step": checkpoint["step"], "sha256": sha256(args.initialize_from)}
    backbone = {k for k, _ in model.named_parameters()
                if k.startswith(("backbone.", "wrist_stem.", "head_stem.", "left_wrist_stem.", "right_wrist_stem."))}
    opt = torch.optim.AdamW([{"params": [v for k, v in model.named_parameters() if k in backbone], "lr": 1e-5},
                             {"params": [v for k, v in model.named_parameters() if k not in backbone], "lr": 1e-4}],
                            weight_decay=1e-4)
    indices = supervised_indices(training, args.steps * 32, args.seed + 1)
    sampled = supervision(training, indices)
    close = sampled["phases"].get("close", {})
    if training.v3_final and close.get("recorded") != close.get("sampled_unique"):
        raise ValueError("The schedule skipped eligible close steps")
    write_json(args.output / "supervision.json", sampled)
    train_loader = DataLoader(training, sampler=indices.tolist(), batch_size=4, num_workers=3,
                              generator=torch.Generator().manual_seed(args.seed + 2))
    val_loader = DataLoader(validation, batch_size=8)
    config = {"variant": args.variant, "wrist_channels": channels, "seed": args.seed, "effective_batch": 32,
              "architecture": architecture, "state_dim": model.state_dim, "freeze_bn": args.freeze_bn,
              "steps": args.steps, "manifest_sha256": sha256(args.manifest), "weights_sha256": model.weight_sha256,
              "detector_config_sha256": detector_config}
    if parent is not None:
        config["parent"] = parent
    if v3:
        config.update(optimizer="AdamW", weight_decay=1e-4, backbone_lr=1e-5, head_lr=1e-4,
                      lr_schedule="constant", microbatch=4, accumulation=8, grad_clip=1.0,
                      autocast="bf16" if device.type == "cuda" else None, augmentation="none",
                      validation_batch=8, validation_interval=500)
    best = float("inf")
    best_step = None
    train_totals = torch.zeros(5, device=device)
    history = {"train": [], "validation": []}
    checkpoint = args.output / "checkpoint.pt"
    for batch_index, batch in enumerate(train_loader, 1):
        tensors = [item.to(device) for item in batch]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if v3:
                *images_and_state, labels, mask, close_mask = tensors
                loss, metrics = phase_loss(model(*images_and_state), labels, mask, close_mask)
                train_totals += metrics
            else:
                *images_and_state, labels, mask = tensors
                loss = masked_chunk_loss(model(*images_and_state), labels, mask)
        (loss / 8).backward()
        if batch_index % 8:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step = batch_index // 8
        if step == 1 or step % 100 == 0 or step == args.steps:
            if v3:
                record = loss_record(train_totals)
                history["train"].append({"step": step, **record})
                write_json(args.output / "loss_curves.json", history)
                train_totals.zero_()
                print(f"{step}/{args.steps} train={json.dumps(record)}", flush=True)
            else:
                print(f"{step}/{args.steps} loss={loss.item():.6f}", flush=True)
        if step % 500 == 0 or step == args.steps:
            validation_metrics = validate(model, val_loader, device)
            score = validation_metrics["total"] if v3 else validation_metrics
            print(f"{step}/{args.steps} validation_masked_loss={score:.6f}", flush=True)
            payload = {"model": model.state_dict(), "step": step, "config": config,
                       "normalization": training.norm}
            if v3:
                history["validation"].append({"step": step, **validation_metrics})
                write_json(args.output / "loss_curves.json", history)
                atomic_checkpoint(args.output / f"step_{step:06}.pt", payload)
                if step == args.steps:
                    atomic_checkpoint(args.output / "last.pt", payload)
                if score < min((row["total"] for row in history["validation"][:-1]), default=float("inf")):
                    atomic_checkpoint(args.output / "best_val.pt", payload)
            selected = step == args.steps if args.select_final else score < best
            if selected:
                best, best_step = score, step
                atomic_checkpoint(checkpoint, payload)
    write_json(args.output / "result.json", {"phase": "three_view_fit", "complete_optimization": True, "config": config,
               "updates": args.steps, "selected_step": best_step, "validation_masked_loss": best,
               "training_windows": len(training), "validation_windows": len(validation),
               "sampler_sha256": sampled["window_draws_sha256"],
               "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint)})


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    modes = cli.add_subparsers(dest="command", required=True)
    split = modes.add_parser("split")
    split.add_argument("collection", type=Path)
    split.add_argument("output", type=Path)
    fit = modes.add_parser("train")
    for name in ("manifest", "weights", "output"):
        fit.add_argument(name, type=Path)
    fit.add_argument("--variant", choices=("A", "B"), required=True)
    fit.add_argument("--boxes", type=Path)
    fit.add_argument("--device", default="cuda:2")
    fit.add_argument("--steps", type=int, default=2000)
    fit.add_argument("--seed", type=int, default=33889)
    fit.add_argument("--initialize-from", type=Path)
    fit.add_argument("--select-final", action="store_true")
    fit.add_argument("--freeze-bn", action="store_true", help="Use the stored ResNet-18 BN running statistics")
    args = cli.parse_args()
    if args.command == "split":
        make_split(args.collection, args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        train(args)
