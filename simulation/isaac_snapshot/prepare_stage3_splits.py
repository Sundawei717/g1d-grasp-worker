import argparse
import concurrent.futures
import json
from collections import Counter
from pathlib import Path

import numpy as np

from prepare_collection_preview import plan_preview, verified_expert
from prepare_grouped import generate_layouts, write_json
from school_planner import sha256


TRAIN_SUCCESS_QUOTAS = {"A_target": 100, "B_obstacle": 88, "C_start": 62}
TRAIN_RESERVE = {"A_target": 116, "B_obstacle": 108, "C_start": 72}
VALIDATION_QUOTAS = {"A_target": 12, "B_obstacle": 10, "C_start": 8}


def draw_cases(root, config, evidence, config_hash, split, seed, quotas, start):
    rng = np.random.default_rng(seed)
    drawn = []
    for index in range(4):
        drawn.extend(generate_layouts(root, config, evidence, seed + 10000 * index))
    ordered = []
    for group in quotas:
        raw = [case for case in drawn if case["group"] == group]
        offset = 0
        while offset < quotas[group]:
            paired = group != "C_start"
            family_index = offset // 2 if paired else offset
            if group == "B_obstacle":
                selected = raw[family_index]
                displacement = rng.uniform([-0.016, -0.025], [0.016, 0.015])
                for pose in (selected["target_pose_xyz_xyzw"], selected["fixtures"][0]["pose"]):
                    pose[0] += float(displacement[0])
                    pose[1] += float(displacement[1])
                pair = (selected, {**selected, "fixtures": []})
            else:
                count = 2 if paired else 1
                pair = raw[offset:offset + count]
            if len(pair) != (2 if paired else 1):
                raise ValueError(f"Too few source layouts for {group}")
            for sample in pair:
                numbered = start + len(ordered)
                case = {**sample, "case_id": f"{group[0]}_{numbered}", "seed": seed * 10 + numbered,
                        "scene_family": f"{split}_{group}_{family_index:03d}", "split": split,
                        "asset_family": "/".join(Path(sample["target_usd"]).parts[-3:-1]),
                        "config_sha256": config_hash}
                if group == "C_start":
                    home = case["home_q"].copy()
                    home[1] = -abs(home[1])
                    case["home_q"] = home
                if split == "train":
                    case["command_delay_substeps"] = 6 if family_index % 5 == 3 else 0
                ordered.append(case)
            offset += len(pair)
    if Counter(row["group"] for row in ordered) != quotas:
        raise ValueError("Stage 3 scene quotas were not reproduced")
    return ordered


def validate_splits(*sources):
    for field in ("case_id", "seed", "scene_family"):
        existing = set()
        for source in sources:
            values = {row[field] for row in source}
            if existing & values:
                raise ValueError(f"Stage 3 split reused {field}: {existing & values}")
            existing.update(values)
    scene_by_family = {}
    for source in sources:
        for item in source:
            actors = [item["target_pose_xyz_xyzw"], *(fixture["pose"] for fixture in item["fixtures"])]
            fingerprint = tuple(sorted(tuple("%.8f" % value for value in pose) for pose in actors))
            signature = (fingerprint, tuple("%.8f" % value for value in item["home_q"]))
            previous = scene_by_family.setdefault(signature, item["scene_family"])
            if previous != item["scene_family"]:
                raise ValueError(f"One physical layout is shared by {previous} and {item['scene_family']}")


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("urdf", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    root = args.root.resolve()
    runs = root / "runs"
    config_path = root / "env_resolved.json"
    config = json.loads(config_path.read_text())
    evidence = json.loads((runs / "lab_school_03/result.json").read_text())
    receipt = runs / "grouped_eval_03/result.json"
    expert = verified_expert({"path": str(receipt), "sha256": sha256(receipt)})
    preview = json.loads((runs / "collection_preview_02/plans/frozen_cases.json").read_text())["cases"]
    bootstrap = json.loads((runs / "train_bootstrap_01/plans/frozen_cases.json").read_text())["cases"]
    training = draw_cases(root, config, evidence, sha256(config_path), "train", 906917, TRAIN_RESERVE, 500)
    validation = draw_cases(root, config, evidence, sha256(config_path), "seen_layout_validation", 937453,
                            VALIDATION_QUOTAS, 850)
    validate_splits(expert, preview, bootstrap, training, validation)
    args.output.mkdir()
    write_json(args.output / "training_candidates.json", {"seed": 906917, "cases": training,
               "success_quotas": TRAIN_SUCCESS_QUOTAS, "maximum_attempts_in_this_manifest": len(training),
               "delayed_success_target": 50, "bootstrap_successes_already_recorded": 24,
               "scope": "two admitted assets; fixed blocked/unblocked scene pairs; one admitted region per asset"})
    write_json(args.output / "frozen_cases.json", {"seed": 937453, "cases": validation, "quotas": VALIDATION_QUOTAS,
                                                 "split": "seen_layout_validation"})
    guide = json.loads((runs / "grouped_eval_03/plans/B_040.json").read_text())["feasible"][0]
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(plan_preview, (config, args.urdf, evidence, case, guide)): case for case in validation}
        for future in concurrent.futures.as_completed(futures):
            case = futures[future]
            key = case["case_id"]
            outcome = future.result()
            write_json(args.output / f"{key}.json", outcome)
            results[key] = {**case, "plan": f"{key}.json", "offline_feasible": bool(outcome["feasible"]),
                            "offline_failure": outcome.get("offline_failure")}
            print(f"{len(results)}/{len(validation)} {key}: {outcome.get('offline_failure') or 'planned'}", flush=True)
    ordered = [results[case["case_id"]] for case in validation]
    feasible = Counter(row["group"] for row in ordered if row["offline_feasible"])
    write_json(args.output / "result.json", {"phase": "stage3_frozen_splits", "validation_cases": len(validation),
                                             "validation_offline_feasible": sum(feasible.values()),
                                             "validation_feasible_by_group": dict(feasible), "cases": ordered,
                                             "training_candidates": len(training), "training_success_quotas": TRAIN_SUCCESS_QUOTAS})
    print("Frozen 30 seen-asset validation scenes; offline feasible groups:", dict(feasible), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
