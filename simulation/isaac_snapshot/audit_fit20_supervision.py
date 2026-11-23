import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch

from prompt_views import PROMPT_AGE_CAP_S
from school_planner import sha256
from train_policy import Windows, corrected_episodes


PHASE_GROUPS = {
    "pre_grasp": ("pre_grasp",),
    "align_approach": ("approach", "align"),
    "close": ("close",),
    "close_hold": ("close_hold",),
}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_fit(root, job="fit20_03"):
    fit = root / "runs" / job
    receipt = json.loads((fit / "result.json").read_text())
    bootstrap = root / "runs/train_bootstrap_01/result.json"
    base = json.loads(bootstrap.read_text())
    replay = root / "runs/visibility_replay_01/result.json"
    corrections = root / "runs/policy_correction_capture_01/result.json"
    episodes, supplements = corrected_episodes(
        [row for row in base["episodes"] if row["use_for_fit20"]], corrections, replay)
    config = receipt["config"]
    checkpoint = fit / "checkpoint.pt"
    snapshot = torch.load(checkpoint, map_location="cpu", weights_only=True)
    log = fit.with_suffix(".log").read_text()
    logged = [(int(step), int(total)) for step, total in re.findall(r"(?m)^(\d+)/(\d+) loss=", log)]
    updates = receipt["updates"]
    expected_logged = [(step, updates) for step in (1, *range(100, updates + 1, 100))]
    marker = json.loads(fit.with_suffix(".complete").read_text())
    if (marker["outcome"] != "PASS" or not receipt["complete_optimization"]
            or config["dataset_sha256"] != sha256(bootstrap)
            or any(config[key] != digest for key, digest in supplements.items())
            or updates != snapshot["step"] or config != snapshot["config"]
            or logged != expected_logged or len(episodes) != receipt["episodes"]):
        raise ValueError("Training receipt, checkpoint and sampler sources disagree")
    dataset = Windows(episodes, config["action_reference"], config.get("prompt_training", "visible_only"))
    if (len(dataset) != config["sampled_windows"] or dataset.norm != snapshot["normalization"]
            or dataset.norm != receipt["normalization"]):
        raise ValueError("Reconstructed training windows or normalization differ from the checkpoint")
    return episodes, dataset, config, updates, sha256(checkpoint)


def coverage(dataset, config, updates):
    draws = torch.multinomial(
        torch.as_tensor(dataset.weights, dtype=torch.double), updates * config["effective_batch"],
        replacement=True, generator=torch.Generator().manual_seed(config["seed"] + 1))
    frequencies = np.bincount(draws.numpy(), minlength=len(dataset))
    possible = [np.zeros(len(action), dtype=bool) for _, action, _, _ in dataset.files]
    terms = [np.zeros(len(action), dtype=np.int64) for _, action, _, _ in dataset.files]
    starts = [np.zeros(len(action), dtype=np.int64) for _, action, _, _ in dataset.files]
    for index, (episode, start, stop) in enumerate(dataset.windows):
        active = slice(start, min(start + 8, stop))
        possible[episode][active] = True
        terms[episode][active] += frequencies[index]
        starts[episode][start] = frequencies[index]
    return possible, terms, starts, frequencies


def phase_counts(selected, flags, visible, possible, terms, starts):
    return {
        "recorded_actions": int(selected.sum()),
        "eligible_actions": int((selected & flags).sum()),
        "valid_prompt_starts": int((selected & flags & visible).sum()),
        "invalid_prompt_actions_covered": int((selected & ~visible & (terms > 0)).sum()),
        "eligible_actions_unreachable": int((selected & flags & ~possible).sum()),
        "reachable_unique_actions": int((selected & possible).sum()),
        "sampled_unique_actions": int((selected & (terms > 0)).sum()),
        "sampled_window_starts": int(starts[selected].sum()),
        "sampled_unique_held_prompt_starts": int((selected & ~visible & (starts > 0)).sum()),
        "sampled_held_prompt_starts": int(starts[selected & ~visible].sum()),
        "sampled_loss_terms": int(terms[selected].sum()),
    }


def audit(root, output, job):
    episodes, dataset, config, updates, checkpoint_digest = load_fit(root, job)
    possible, terms, starts, frequencies = coverage(dataset, config, updates)
    rows, boundary, details = [], [], []
    input_prompt_ages = []
    for i, (episode, (path, actions, _, phases)) in enumerate(zip(episodes, dataset.files)):
        with h5py.File(path, "r") as h5:
            flags = h5["train_action_valid"][...]
            visible = h5["policy_prompt/valid"][:-1]
            sampled = np.flatnonzero(starts[i])
            sources = dataset.prompt_sources[i]
            frames = sampled if sources is None else sources[sampled]
            ages = h5["obs/timestamp"][...][sampled] - h5["policy_prompt/updated_at"][...][frames]
        input_prompt_ages.extend(ages if sources is None else np.minimum(ages, PROMPT_AGE_CAP_S))
        if np.any(possible[i] & ~flags) or np.any((terms[i] > 0) & ~possible[i]):
            raise ValueError(f"Action loss coverage is inconsistent for {episode['case']}")
        identity = {"case": episode["case"], "group": episode["group"], "source_h5": str(path)}
        for label, phase_names in PHASE_GROUPS.items():
            selected = np.isin(phases, phase_names)
            rows.append({**identity, "phase": label,
                         **phase_counts(selected, flags, visible, possible[i], terms[i], starts[i])})
        close = np.flatnonzero(phases == "close")
        if not len(close):
            raise ValueError(f"Missing close phase for {episode['case']}")
        first = int(close[0])
        details.append({
            **identity,
            "first_close_action": first,
            "first_close_prompt_valid": bool(visible[first]),
            "first_close_loss_terms": int(terms[i][first]),
            "previous_5_actions_total": min(first, 5),
            "previous_5_actions_sampled": int(np.count_nonzero(terms[i][max(0, first - 5):first])),
            "first_3_close_actions_total": min(len(close), 3),
            "first_3_close_actions_sampled": int(np.count_nonzero(terms[i][close[:3]])),
            "jaw_target_before_close_mm": round(float(actions[max(0, first - 1), -1] * 1000), 3),
            "lowest_close_jaw_target_mm": round(float(actions[close, -1].min() * 1000), 3),
        })
        for step in np.r_[np.arange(max(0, first - 5), first), close[:3]]:
            boundary.append({**identity, "action_index": int(step), "phase": phases[step],
                             "offset_from_close": int(step - first), "prompt_valid": bool(visible[step]),
                             "reachable_by_training_window": bool(possible[i][step]),
                             "sampled_window_starts": int(starts[i][step]), "sampled_loss_terms": int(terms[i][step]),
                             "jaw_target_mm": round(float(actions[step, -1] * 1000), 3)})
    phase_totals = {}
    count_fields = tuple(key for key in rows[0] if key not in ("case", "group", "source_h5", "phase"))
    for phase in PHASE_GROUPS:
        phase_rows = [row for row in rows if row["phase"] == phase]
        phase_totals[phase] = {key: sum(row[key] for row in phase_rows) for key in count_fields}
    result = {
        "checkpoint_sha256": checkpoint_digest,
        "fit": job,
        "train_policy_sha256": sha256(root / "train_policy.py"),
        "chunk_model_sha256": sha256(root / "chunk_model.py"),
        "train_config": config,
        "updates": updates,
        "trajectory_count": len(episodes),
        "window_draws": int(frequencies.sum()),
        "unique_window_starts_sampled": int(np.count_nonzero(frequencies)),
        "input_prompt_age_s_at_sampled_unique_starts": {
            "min": float(np.min(input_prompt_ages)), "max": float(np.max(input_prompt_ages)),
            "nonzero": int(np.count_nonzero(np.abs(input_prompt_ages) > 1e-6)),
        },
        "phase_mapping": PHASE_GROUPS,
        "phases_present_in_hdf5": sorted({p for _, _, _, phases in dataset.files for p in phases}),
        "ignored_action_rows": sum(len(action) for _, action, _, _ in dataset.files)
        - sum(row["recorded_actions"] for row in rows),
        "total_sampled_loss_terms": sum(int(values.sum()) for values in terms),
        "phase_totals": phase_totals,
        "episodes_with_no_close_loss": [
            row["case"] for row in rows if row["phase"] == "close" and not row["sampled_loss_terms"]],
        "episodes_with_no_first_close_loss": [row["case"] for row in details if not row["first_close_loss_terms"]],
        "close_boundary_per_trajectory": details,
    }
    if sum(row["sampled_loss_terms"] for row in rows) != result["total_sampled_loss_terms"]:
        raise ValueError("An action phase contributing to the loss was omitted")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "trajectory_phase.csv", rows)
    write_csv(output / "close_boundary.csv", boundary)
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "close_boundary_per_trajectory"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--fit", default="fit20_03")
    args = parser.parse_args()
    if not re.fullmatch(r"fit20_[0-9]+", args.fit):
        parser.error("--fit must be a fit20 job name")
    audit(Path(__file__).resolve().parent, args.output, args.fit)
