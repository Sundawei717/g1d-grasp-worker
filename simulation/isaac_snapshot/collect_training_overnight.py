import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import h5py
import numpy as np

from audit_grasp_receipt import audit
from prepare_collection_preview import plan_preview
from prepare_grouped import verify_gates, write_json
from prepare_stage3_splits import TRAIN_RESERVE, draw_cases, validate_splits
from prepare_train_bootstrap import verify_train_inputs
from school_planner import sha256


QUOTAS = {"A_target": (160, 40), "B_obstacle": (140, 36), "C_start": (100, 24)}
MILESTONE = {"A_target": (80, 20), "B_obstacle": (70, 18), "C_start": (50, 12)}
CONFIGS = {"carton": ("env_resolved.json", 1), "can": ("runs/candidate_can_config.json", 2)}


def atomic(path, payload):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def stratum(row):
    return row["group"], int(bool(row["command_delay_substeps"]))


def select(available, quotas):
    count = Counter()
    output = []
    for row in available:
        group, delay = stratum(row)
        if count[group, delay] < quotas[group][delay]:
            output.append(row)
            count[group, delay] += 1
    return output, count


def freeze(root, output, cfg, evidence):
    runs = root / "runs"
    bootstrap = json.loads((runs / "train_bootstrap_01/result.json").read_text())
    bootstrap_plans = json.loads((runs / "train_bootstrap_01/plans/frozen_cases.json").read_text())
    source = runs / "stage3_splits_01/training_candidates.json"
    initial = json.loads(source.read_text())["cases"]
    hash_config = sha256(root / "env_resolved.json")
    reserve = draw_cases(root, cfg, evidence, hash_config, "overnight_reserve", 982697, TRAIN_RESERVE, 1000)
    for case in reserve:
        case["split"] = "train"
        family = int(case["scene_family"].rsplit("_", 1)[-1])
        case["command_delay_substeps"] = 6 if family % 5 == 3 else 0
    reference = ["grouped_plan_03/frozen_cases.json", "collection_preview_02/plans/frozen_cases.json",
                 "stage3_splits_01/frozen_cases.json"]
    prior = [json.loads((runs / filename).read_text())["cases"] for filename in reference]
    validate_splits(*prior, bootstrap_plans["cases"], initial, reserve)
    cases = initial + reserve
    verify_train_inputs(bootstrap_plans["training_gates"], cases)
    base = bootstrap["episodes"]
    for row in base:
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"Bootstrap episode hash changed: {row['path']}")
    frozen = {"phase": "overnight_expert_data", "train_candidates": cases, "bootstrap": base, "quotas": QUOTAS,
              "milestone": MILESTONE, "training_gates": bootstrap_plans["training_gates"],
              "candidate_source_sha256": sha256(source), "held_out_layouts": str(runs / reference[-1])}
    write_json(output / "frozen_cases.json", frozen)
    return frozen


def choose_wave(remaining, accepted):
    picked = []
    by_asset = Counter()
    pending = Counter()
    size = 24
    while len(picked) < size:
        bucket = sorted(remaining, key=lambda key: (-(QUOTAS[key[0]][key[1]] - accepted[key] - pending[key]) /
                                                     QUOTAS[key[0]][key[1]], pending[key], key))
        for key in bucket:
            items = remaining[key]
            if not items or accepted[key] + pending[key] >= QUOTAS[key[0]][key[1]]:
                continue
            selected = next((index for index, item in enumerate(items) if by_asset[item["asset_role"]] < 16), None)
            if selected is None:
                continue
            row = items[selected]
            del items[selected]
            picked.append(row)
            by_asset[row["asset_role"]] += 1
            pending[key] += 1
            break
        else:
            break
    return picked


def ready_gpu(gpu):
    memory = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
                                      "--format=csv,noheader,nounits"], text=True)
    return int(memory.strip()) >= 20480


def shard_trial(root, output, plans, attempts, previous, role, gate, round_number):
    cfg, gpu = CONFIGS[role]
    tag = f"{output.name}_w{round_number:03d}_{role}" + ("_retry" if previous else "")
    directory = root / "runs" / tag
    files = {item["name"]: item["usd"] for row in attempts for item in row["fixtures"]}
    manifest = plans / f"{tag}.json"
    write_json(manifest, {"phase": "collection_train", "asset": attempts[0]["target_usd"],
                          "cases": attempts, "fixture_assets": files,
                          "gates": gate["expert"], "training_gates": gate["training"]})
    with directory.with_suffix(".log").open("x") as stream:
        child = subprocess.Popen(["bash", str(root / "run_lab_school.sh"), str(directory), "--grouped-manifest",
                                  str(manifest), "--grasp-after", str(root / "runs/lab_school_03/result.json"),
                                  "--gpu", str(gpu)], cwd=root,
                                 env={**os.environ, "G1D_WORKER_CONFIG": str(root / cfg)},
                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    directory.with_suffix(".pid").write_text(f"{child.pid}\n")
    print(f"{tag} PID {child.pid} on GPU {gpu} ({len(attempts)} layouts)", flush=True)
    return child, directory


def harvest(directory):
    for filename in ("result.json", "batch_progress.json"):
        path = directory / filename
        if path.is_file():
            return {row["case"]: row for row in json.loads(path.read_text())["cases"]}
    return {}


def run_wave(root, output, plans, ready, gate, round_number, events):
    active = []
    accepted = {}
    for role, (_, gpu) in CONFIGS.items():
        selected = [row for row in ready if row["asset_role"] == role]
        if selected:
            if not ready_gpu(gpu):
                print(f"GPU {gpu} has insufficient free space for {role}; retrying the GPU check", flush=True)
                for _ in range(30):
                    time.sleep(15)
                    if ready_gpu(gpu):
                        break
                else:
                    raise RuntimeError(f"No available memory on GPU {gpu}")
            child, directory = shard_trial(root, output, plans, selected, False, role, gate, round_number)
            active.append((child, directory, role, selected))
    for child, directory, role, selected in active:
        status = child.wait()
        print(f"{directory.name} completed with exit {status}", flush=True)
        rows = harvest(directory)
        accepted.update({key: (row, directory) for key, row in rows.items()})
        missing = [row for row in selected if row["case_id"] not in rows]
        if missing:
            events.append({"shard": directory.name, "exit": status, "missing_cases": [row["case_id"] for row in missing]})
            child, retry = shard_trial(root, output, plans, missing, True, role, gate, round_number)
            code = child.wait()
            accepted.update({key: (row, retry) for key, row in harvest(retry).items()})
            print(f"{retry.name} completed with exit {code}", flush=True)
    return accepted


def quality(item, directory, receipt):
    key = item["case_id"]
    episode = directory / key
    path = episode / "episode.h5"
    if path.stat().st_size != receipt["dataset_receipt"]["bytes"] or not audit(episode)["strict_success"]:
        raise ValueError(f"Expert episode failed independent integrity: {key}")
    with h5py.File(path, "r") as h5:
        metadata = json.loads(h5["meta/case_json"].asstr()[()])
        if any(item[field] != metadata[field] for field in metadata) or metadata["split"] != "train":
            raise ValueError(f"Case identity changed during physical capture: {key}")
        anchors = h5["gt/part_anchor_local"][...]
        if not np.allclose(anchors, anchors[0], atol=1e-8, rtol=0):
            raise ValueError(f"Part anchor changes within the object frame: {key}")
        if item["command_delay_substeps"]:
            native = h5["action/native_substep_command"]
            if not np.allclose(native[5, :6], native[4, -1], atol=1e-6, rtol=0):
                raise ValueError(f"Missing scheduled execution delay: {key}")
    return {"case": key, "group": item["group"], "scene_family": item["scene_family"],
            "asset": item["asset_family"], "command_delay_substeps": item["command_delay_substeps"],
            "path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def wait_for_recorders(root, deadline):
    jobs = ("tracking_motion_01", "tracking_benchmark_02")
    print(f"Waiting for GPUs to finish {', '.join(jobs)}", flush=True)
    while any(not (root / "runs" / (name + ".complete")).is_file() for name in jobs):
        if time.monotonic() >= deadline:
            raise TimeoutError("Overnight collection window ended before GPUs became available")
        time.sleep(5)


def main():
    cli = argparse.ArgumentParser()
    for name in ("root", "urdf", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--hours", type=float, default=7)
    cli.add_argument("--reserve-gib", type=int, default=50)
    cli.add_argument("--resume-from", type=Path)
    args = cli.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    deadline = time.monotonic() + args.hours * 3600
    cfg = json.loads((root / "env_resolved.json").read_text())
    evidence = json.loads((root / "runs/lab_school_03/result.json").read_text())
    plans = output / "plans"
    plans.mkdir()
    frozen = freeze(root, output, cfg, evidence)
    recorded, events = [], []
    sources = list(frozen["bootstrap"])
    if args.resume_from:
        prior = args.resume_from
        if not prior.with_suffix(".complete").is_file():
            raise ValueError("Resume requires a completed prior collection")
        if sha256(prior / "frozen_cases.json") != sha256(output / "frozen_cases.json"):
            raise ValueError("Resume candidates differ from the original freeze")
        checkpoint = json.loads((prior / "journal.json").read_text())
        recorded, events = checkpoint["attempts"], checkpoint["infrastructure_retries"]
        sources = checkpoint["successful_hdf5"]
        interrupted = [row for row in recorded if row["status"] == "infrastructure_fail"]
        if interrupted:
            events.append({"resumed_infrastructure_cases": [row["case"] for row in interrupted]})
            recorded = [row for row in recorded if row["status"] != "infrastructure_fail"]
    used = {row["case"] for row in recorded}
    remaining = defaultdict(deque)
    for case in frozen["train_candidates"]:
        if case["case_id"] not in used:
            remaining[stratum(case)].append(case)
    guide = json.loads((root / "runs/grouped_eval_03/plans/B_040.json").read_text())["feasible"][0]
    gate = {"expert": verify_gates(root), "training": frozen["training_gates"]}
    capacity = json.loads((root / "runs/collection_preview_02/result.json").read_text())
    # Each episode also retains its native trace, receipts and sparse graphics.
    bytes_per_episode = max(max(capacity["measured_h5_file_sizes_bytes"]) * 2, 20 * 1024**2)
    reserve = args.reserve_gib * 1024**3
    wave = 0
    recorders_finished = False
    stop = "candidate pool exhausted"
    while True:
        chosen, counts = select(sources, QUOTAS)
        checkpoint = {"attempts": recorded, "successful_hdf5": sources, "infrastructure_retries": events,
                      "selected": len(chosen), "target": 500}
        atomic(output / "journal.json", checkpoint)
        milestone, _ = select(sources, MILESTONE)
        if len(milestone) == 250 and not (output / "milestone_250.json").is_file():
            write_json(output / "milestone_250.json", {"accepted": 250, "episodes": milestone})
            print("250-episode manifest written", flush=True)
        atomic(output / "batch_progress.json", {"phase": "overnight_expert_data", "accepted": len(chosen),
                                                "configured": 500, "cases": recorded})
        if len(chosen) == 500:
            stop = "target reached"
            break
        if time.monotonic() >= deadline:
            stop = "overnight time budget reached"
            break
        physical = [row for row in recorded if row["status"] != "infrastructure_fail"]
        if len(physical) >= 100 and any(sum(case["status"] == "strict_success" for case in physical[i:i+100]) < 80
                                         for i in range(len(physical)-99)):
            stop = "a 100-attempt window fell below 80 percent strict success"
            break
        if shutil.disk_usage(root).free < reserve + (500 - len(chosen)) * bytes_per_episode:
            stop = "measured capacity and shared disk reserve reached"
            break
        batch = choose_wave(remaining, counts)
        if not batch:
            break
        wave += 1
        print(f"Wave {wave}: planning {len(batch)} frozen layouts, manifest has {len(chosen)}/500", flush=True)
        planned = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(plan_preview, (cfg, args.urdf, evidence, case, guide)): case for case in batch}
            for future in concurrent.futures.as_completed(futures):
                case = futures[future]
                result = future.result()
                filename = f"{case['case_id']}.json"
                write_json(plans / filename, result)
                planned[case["case_id"]] = {**case, "plan": filename, "offline_feasible": bool(result["feasible"]),
                                           "offline_failure": result.get("offline_failure")}
                print(f"  {case['case_id']} {result.get('offline_failure') or 'planned'}", flush=True)
        ordered = [planned[row["case_id"]] for row in batch]
        feasible = [row for row in ordered if row["offline_feasible"]]
        if feasible and not recorders_finished:
            wait_for_recorders(root, deadline)
            recorders_finished = True
        if time.monotonic() >= deadline:
            stop = "overnight time budget reached before dispatch"
            break
        if shutil.disk_usage(root).free < reserve + (500 - len(chosen)) * bytes_per_episode:
            stop = "shared disk reserve reached before dispatch"
            break
        receipts = run_wave(root, output, plans, feasible, gate, wave, events) if feasible else {}
        results = []
        for case in ordered:
            key = case["case_id"]
            if not case["offline_feasible"]:
                item = {"case": key, "status": "offline_reject", "failure": case["offline_failure"],
                        "completed_at": (plans / case["plan"]).stat().st_mtime}
            elif key in receipts:
                row, directory = receipts[key]
                trace = directory / key / "expert_diagnostic.json"
                completion = trace.stat().st_mtime if trace.is_file() else directory.stat().st_mtime
                item = {**row, "shard": directory.name, "completed_at": completion}
                if item["status"] == "strict_success":
                    sources.append(quality(case, directory, item))
            else:
                item = {"case": key, "status": "infrastructure_fail", "completed_at": time.time()}
            results.append({**item, "group": case["group"], "scene_family": case["scene_family"]})
        recorded.extend(sorted(results, key=lambda row: row["completed_at"]))
        if any(row["status"] == "infrastructure_fail" for row in results):
            stop = "infrastructure interruption persisted after a retry"
            atomic(output / "journal.json", {"attempts": recorded, "successful_hdf5": sources,
                                            "infrastructure_retries": events})
            break
    chosen, counts = select(sources, QUOTAS)
    write_json(output / "result.json", {"phase": "overnight_expert_data", "passed": len(chosen) == 500,
                                        "accepted": len(chosen), "configured": 500, "stop_reason": stop,
                                        "groups": {group: {"without_delay": counts[group, 0], "with_delay": counts[group, 1]}
                                                   for group in QUOTAS},
                                        "episodes": chosen, "cases": recorded, "reserve_gib": args.reserve_gib})
    print(f"Stopped at {len(chosen)}/500: {stop}", flush=True)
    return 0 if len(chosen) == 500 else 1


if __name__ == "__main__":
    raise SystemExit(main())
