import argparse
import concurrent.futures
import json
import shutil
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from prepare_collection_preview import plan_preview, verified_expert
from prepare_grouped import TARGETS, generate_layouts, verify_gates, write_json
from school_planner import sha256


QUOTAS = {"A_target": 8, "B_obstacle": 7, "C_start": 5}
SEED = 620174


def successful_receipt(binding, rows, phase):
    path = Path(binding["path"])
    if sha256(path) != binding["sha256"]:
        raise ValueError(f"A prerequisite receipt changed: {path}")
    status = json.loads(path.parent.with_suffix(".complete").read_text())
    result = json.loads(path.read_text())
    if (status["exit_code"] != 0 or result.get("passed") is not True or result.get("phase") != phase
            or result.get("strict_success") < rows):
        raise ValueError(f"Prerequisite did not pass: {path}")
    return result


def verify_train_inputs(gates, cases):
    checked_expert = verified_expert(gates["expert"])
    successful_receipt(gates["capacity"], 20, "collection_capacity_preview")
    successful_receipt(gates["alignment"], 10, "recorded_command_alignment")
    preview_path = Path(gates["capacity"]["path"]).parent / "plans/frozen_cases.json"
    preview = json.loads(preview_path.read_text())["cases"]
    evaluation = checked_expert + preview
    for field in ("case_id", "seed", "scene_family"):
        if {row[field] for row in cases} & {row[field] for row in evaluation}:
            raise ValueError(f"The training set shares evaluation {field}")
    for case in cases:
        if (case["split"] != "train" or not case["asset_family"] or not case["config_sha256"]
                or case["command_delay_substeps"] not in {0, 6}):
            raise ValueError(f"Invalid training provenance: {case['case_id']}")


def frozen_cases(root, config, evidence, config_hash):
    samples = generate_layouts(root, config, evidence, SEED)
    cases = samples[:10]
    rng = np.random.default_rng(SEED + 73)
    for index, sample in enumerate(samples[40:44]):
        dx, dy = rng.uniform(-0.009, 0.009, size=2) + [0.005, -0.012]
        for pose in (sample["target_pose_xyz_xyzw"], sample["fixtures"][0]["pose"]):
            pose[0] += float(dx)
            pose[1] += float(dy)
        sample["scene_family"] = f"obstacle_pair_{index:02d}"
        cases.extend((sample, {**sample, "fixtures": []}))
    cases.extend(samples[75:81])
    for index, case in enumerate(cases):
        case["case_id"] = f"{case['group'][0]}_{400 + index}"
        case["seed"] = SEED * 2 + index
        case["scene_family"] = f"train_bootstrap_{case['scene_family']}"
        case["split"] = "train"
        case["asset_family"] = "/".join(Path(case["target_usd"]).parts[-3:-1])
        case["config_sha256"] = config_hash
        case["command_delay_substeps"] = 6 if index in {2, 3, 12, 13, 20} else 0
        if case["group"] == "C_start":
            case["home_q"][1] = -abs(case["home_q"][1])
    return cases


def prepare(root, urdf, expert, capacity, alignment, output):
    receipts = {key: {"path": str(path.resolve()), "sha256": sha256(path)}
                for key, path in {"expert": expert, "capacity": capacity, "alignment": alignment}.items()}
    config_path = root / "env_resolved.json"
    config = json.loads(config_path.read_text())
    evidence = json.loads((root / "runs/lab_school_03/result.json").read_text())
    cases = frozen_cases(root, config, evidence, sha256(config_path))
    verify_train_inputs(receipts, cases)
    capacity_result = json.loads(capacity.read_text())
    if shutil.disk_usage(root).free <= capacity_result["projected_2100_h5_bytes"]:
        raise ValueError("Available filesystem space fell below the measured full-dataset estimate")
    prerequisites = verify_gates(root)
    output.mkdir()
    write_json(output / "frozen_cases.json", {"seed": SEED, "split": "train", "cases": cases, "quotas": QUOTAS,
                                              "training_gates": receipts})
    reference = root / "runs/grouped_eval_03/plans/B_040.json"
    guide = json.loads(reference.read_text())["feasible"][0]
    planned = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(plan_preview, (config, urdf, evidence, case, guide)): case for case in cases}
        for future in concurrent.futures.as_completed(futures):
            case = futures[future]
            key = case["case_id"]
            result = future.result()
            write_json(output / f"{key}.json", result)
            planned[key] = {**case, "plan": f"{key}.json", "offline_feasible": bool(result["feasible"]),
                            "offline_failure": result.get("offline_failure")}
            print(f"{len(planned)}/{len(cases)} {key}: {result.get('offline_failure') or 'planned'}", flush=True)
    ordered = [planned[case["case_id"]] for case in cases]
    for role in TARGETS:
        shard = [row for row in ordered if row["asset_role"] == role]
        fixtures = {fixture["name"]: fixture["usd"] for row in shard for fixture in row["fixtures"]}
        write_json(output / f"shard_{role}.json", {"phase": "collection_train", "asset": shard[0]["target_usd"],
                   "cases": shard, "fixture_assets": fixtures, "gates": prerequisites, "training_gates": receipts})


def finish(output):
    frozen = json.loads((output / "plans/frozen_cases.json").read_text())
    observed = {}
    for role in TARGETS:
        run = output.parent / f"{output.name}_{role}"
        for name in ("result.json", "batch_progress.json"):
            path = run / name
            if path.is_file():
                rows = json.loads(path.read_text())["cases"]
                observed.update({row["case"]: {**row, "shard": run.name} for row in rows})
                break
    accepted, attempts = [], []
    counts = Counter()
    delay_count = 0
    for case in frozen["cases"]:
        key, group = case["case_id"], case["group"]
        row = observed.get(key, {"case": key, "status": "infrastructure_fail"})
        attempts.append({**row, "group": group})
        if row["status"] != "strict_success" or not row.get("dataset_receipt"):
            continue
        path = output.parent / row["shard"] / key / "episode.h5"
        with h5py.File(path, "r") as h5:
            if json.loads(h5["meta/case_json"].asstr()[()]) != case:
                raise ValueError(f"Training receipt changed its frozen case: {key}")
            anchors = h5["gt/part_anchor_local"][...]
            if not np.allclose(anchors, anchors[0], atol=1e-8, rtol=0):
                raise ValueError(f"Training anchor moved within the object frame: {key}")
            if case["command_delay_substeps"]:
                native = h5["action/native_substep_command"]
                if not np.allclose(native[5, :6], np.repeat(native[4, -1][None], 6, axis=0), atol=1e-6, rtol=0):
                    raise ValueError(f"Scheduled servo perturbation was absent: {key}")
                delay_count += 1
        accept_fit = counts[group] < QUOTAS[group]
        counts[group] += 1
        accepted.append({"case": key, "group": group, "scene_family": case["scene_family"], "asset": case["asset_family"],
                         "command_delay_substeps": case["command_delay_substeps"], "use_for_fit20": accept_fit,
                         "path": str(path.resolve()), "sha256": sha256(path)})
    passed = all(counts[group] >= quota for group, quota in QUOTAS.items())
    write_json(output / "result.json", {"phase": "training_bootstrap", "configured": len(attempts),
                                       "strict_success": len(accepted), "groups": dict(counts), "passed": passed,
                                       "delayed_successes": delay_count, "episodes": accepted, "cases": attempts})
    print(f"Training bootstrap: {len(accepted)}/{len(attempts)} strict, {dict(counts)}, delayed={delay_count}", flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser()
    stages = parser.add_subparsers(dest="command", required=True)
    prep = stages.add_parser("prepare")
    for field in ("root", "urdf", "expert", "capacity", "alignment", "output"):
        prep.add_argument(field, type=Path)
    stages.add_parser("finish").add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.urdf, args.expert, args.capacity, args.alignment, args.output)
        return 0
    return finish(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
