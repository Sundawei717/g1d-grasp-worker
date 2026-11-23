import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import h5py
import numpy as np

from background_status import record_complete
from episode_store import verify_three_view
from step1_episode import VIEWS
from three_view_detector import sha256, write_json
from unified_grasp import EPISODE_STEPS, LiftGateConfig


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
SIM = Path("/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python")
QUOTAS = {"white_box": 40, "pepsi": 40, "mug": 140, "basket": 120, "carton": 120}
OBSTACLE_MIN = {"mug": 14, "basket": 12, "carton": 12}
REUSED_H5 = {"white_box": "sand3d_packaging/00152", "pepsi": "sand3d_packaging/00156"}
FREE_MIB = 22 * 1024
MAX_ATTEMPTS = 2


def read(path):
    return json.loads(Path(path).read_text())


def scene_signature(case):
    payload = [case["asset_family"], np.round(case["target_pose_xyz_xyzw"], 5).tolist(),
               np.round(case["home_q"], 5).tolist(), case["part"],
               [(f["sha256"], np.round(f["pose"], 5).tolist()) for f in case["fixtures"]]]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def write_candidate(work, bucket, plan, before=None):
    case = dict(plan["case"])
    key = case["case_id"]
    place = work / "plans" / key
    place.mkdir(parents=True, exist_ok=True)
    if (place / "ready.json").exists():
        return
    receipt = dict(case, bucket=bucket, episode_id=case.get("episode_id", key),
                   scene_id=case.get("scene_id", key), pair_group_id=case.get("pair_group_id"),
                   scene_signature=scene_signature(case))
    if before:
        receipt["replay_h5"] = before
    write_json(place / "plan.json", plan)
    write_json(place / "ready.json", receipt)


def prepare_legacy(work):
    split = read(RUNS / "three_view_pipeline_03/split.json")
    config = read(ROOT / "env_resolved.json")
    grouped = defaultdict(list)
    for row in split["episodes"]:
        for bucket, family in REUSED_H5.items():
            if row["asset_family"] == family and row["group"] != "B_obstacle":
                grouped[bucket, row["group"]].append(row)
    for bucket, start in (("white_box", 7000), ("pepsi", 7130)):
        groups = [sorted(grouped[bucket, group], key=lambda row: hashlib.sha256(row["case"].encode()).hexdigest())
                  for group in ("A_target", "C_start")]
        selected = []
        for i in range(max(map(len, groups))):
            selected.extend(rows[i] for rows in groups if i < len(rows))
        seen = set()
        for i, row in enumerate(selected):
            path = Path(row["path"])
            if not path.exists():
                continue
            with h5py.File(path, "r") as data:
                stored = json.loads(data["meta/case_json"].asstr()[()])
                score = json.loads(data["meta/strict_score_json"].asstr()[()])
                if not score["success"] or "action/native_substep_command" not in data:
                    continue
                joints = data["meta/joint_names"].asstr()[:].tolist()
            if scene_signature(stored) in seen:
                continue
            seen.add(scene_signature(stored))
            number = start+i
            key = f"{row['group'][0]}_{number}"
            case = dict(stored, case_id=key, episode_id=f"tier1_{key}", scene_id=f"tier1_scene_{key}",
                        scene_family=f"tier1_{bucket}_{number}", split="train", source_case=stored["case_id"],
                        pair_group_id=None, order=i)
            plan = {"case": case, "feasible": [{"candidate": "archived_native_prefix"}],
                    "home_q": case["home_q"], "obstacles": case["fixtures"], "right_joint_names": joints,
                    "target_pose_xyz_xyzw": case["target_pose_xyz_xyzw"], "school_sha256": sha256(config["school_usd"]),
                    "target_sha256": sha256(case["target_usd"])}
            binding = {"path": str(path), "sha256": row["sha256"], "rerecord": True, "source_case": stored["case_id"]}
            write_candidate(work, bucket, plan, binding)
    write_json(work / "legacy_ready.json", {"candidates": len(list((work / 'plans').glob('*/ready.json')))})


def gpu_manifest():
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"], text=True)
    return [{"gpu": int(i), "uuid": uid, "free_mib": int(free)}
            for row in query.splitlines() for i, uid, free in [list(map(str.strip, row.split(",")))]]


def verify_receipt(receipt):
    path = Path(receipt["path"])
    if sha256(path) != receipt["sha256"]:
        raise ValueError(f"Trajectory hash mismatch: {path}")
    with h5py.File(path, "r") as data:
        t = int(receipt["steps"])
        if data.attrs.get("schema") != "g1d_v3_step_aligned" or not 0 < t <= EPISODE_STEPS or verify_three_view(data) != t+1:
            raise ValueError(f"Wrong RGB contract: {path}")
        if (not np.array_equal(data["obs/control_step_id"][:], np.arange(t+1))
                or not np.array_equal(data["action/control_step_id"][:], np.arange(t))
                or not np.allclose(np.diff(data["obs/timestamp"][:]), .05, rtol=0, atol=2e-7)):
            raise ValueError(f"Nonconsecutive control steps: {path}")
        phase = data["phase"].asstr()[:]
        if set(phase) != {"approach", "close", "lift"} or not np.array_equal(data["train_action_valid"][:], phase != "lift"):
            raise ValueError(f"Invalid approach and close supervision: {path}")
        transition = int(np.flatnonzero(phase == "lift")[0])
        n = data["action/native_substep_command"][:]
        if (not np.array_equal(n[:, -1], data["action/native_command"][:])
                or not np.allclose(n[transition:, :, 7:], n[transition-1, -1, 7:], atol=1e-7, rtol=0)
                or any(data[key].shape != shape or not np.isfinite(data[key][:]).all() for key, shape in
                       (("obs/eef_pose", (t+1, 7)), ("obs/qpos", (t+1, 7)), ("obs/gripper_width", (t+1,))))):
            raise ValueError(f"Invalid native commands or proprioception: {path}")
        for key in VIEWS.values():
            rgb = data[key]
            if rgb.shape != (t+1, 224, 224, 3) or rgb.dtype != np.uint8 or rgb.compression != "gzip":
                raise ValueError(f"Invalid lossless image dataset: {path} {key}")
        case = json.loads(data["meta/case_json"].asstr()[()])
        prompt = json.loads(data["meta/initial_prompt_json"].asstr()[()])
        bbox = np.asarray(prompt["bbox_0"])
        point = np.asarray(prompt["affordance_0"])
        coordinates = np.r_[bbox, point]
        if (not json.loads(data["meta/strict_score_json"].asstr()[()])["success"] or not prompt.get("prompt_valid", True)
                or bbox.shape != (4,) or point.shape != (2,) or not np.isfinite(coordinates).all()
                or np.any((coordinates < 0) | (coordinates > 1)) or np.any(bbox[:2] >= bbox[2:])):
            raise ValueError(f"Missing physical success or first-frame prompt: {path}")
        return case, prompt, data["obs/qpos"][0].tolist(), float(data["obs/gripper_width"][0])


def admit_episode(row, receipt, diagnostic, *, enforce_contact_filter=True):
    roll = diagnostic.get("policy_rollout", {})
    if not roll.get("success") or not roll.get("part_hit") or roll["steps"] > EPISODE_STEPS:
        raise ValueError("Episode failed the fixed executor")
    if enforce_contact_filter and row["bucket"] not in REUSED_H5 and diagnostic.get("forbidden_contact_pairs"):
        raise ValueError("A new demonstration reported forbidden contact")
    case, prompt, arm, width = verify_receipt(receipt)
    if scene_signature(case) != row["scene_signature"]:
        raise ValueError("Trajectory belongs to a different layout")
    return {**receipt, **{k: row[k] for k in ("bucket", "group", "asset_family", "part", "episode_id", "scene_id",
                                               "pair_group_id", "scene_family", "scene_signature")},
            "bbox": prompt["bbox_0"], "point": prompt["affordance_0"], "initial_q": arm, "initial_width": width,
            "initial_target": diagnostic["target_initial_xyz_xyzw"], "source_case": row.get("source_case"),
            "yaw_deg": row.get("yaw_deg"), "fixture_assets": [f["usd"] for f in case["fixtures"]],
            "capture_wall_s": roll["wall_s"], "case": row["case_id"]}


def paired(first, second):
    return (first["part"] != second["part"] and first["bbox"] == second["bbox"] and first["point"] != second["point"]
            and np.allclose(first["initial_q"], second["initial_q"], atol=1e-5, rtol=0)
            and abs(first["initial_width"] - second["initial_width"]) <= 1e-5
            and np.linalg.norm(np.asarray(first["initial_target"][:3])-second["initial_target"][:3]) < .001
            and 1-abs(np.dot(first["initial_target"][3:], second["initial_target"][3:])) < 1e-6)


def seed_accepted(work):
    if (work / "pilot_import.json").exists():
        return
    sources = (("mug", "step1_mug_plan_lift8"), ("mug", "step1b_plan_mug_pairs"),
               ("mug", "step1b_plan_mug_extra"), ("mug", "step1b_plan_mug_plus75"),
               ("basket", "step1b_basket_verified_plan"), ("carton", "step1b_plan_carton_pairs_v2"))
    proposed = defaultdict(list)
    imported = []
    for bucket, name in sources:
        path = RUNS / name / "episodes.json"
        for receipt in read(path)["episodes"] if path.exists() else ():
            diag = read(Path(receipt["path"]).with_name("expert_diagnostic.json"))
            try:
                case, *_ = verify_receipt(receipt)
            except (ValueError, KeyError, OSError):
                continue
            if case["fixtures"]:
                continue  # Only complete part pairs are reusable against the paired quotas.
            pair_id = f"pilot_{bucket}_{case['scene_family']}_{case.get('yaw_deg', 0)}"
            key = receipt["case"]
            row = dict(case, bucket=bucket, case_id=key, episode_id=f"tier1_pilot_{key}", scene_id=f"tier1_pilot_scene_{key}",
                       scene_signature=scene_signature(case), pair_group_id=pair_id, scene_family=pair_id)
            # Physical collision diagnostics were recorded during the pilot. Apply the same intake checks.
            try:
                proposed[pair_id].append(admit_episode(row, receipt, diag))
            except ValueError:
                continue
    for group in proposed.values():
        if len(group) == 2 and paired(*group):
            for row in group:
                write_json(work / "accepted" / f"{row['case']}.json", row)
                imported.append(row["case"])
    write_json(work / "pilot_import.json", {"accepted": imported, "count": len(imported)})


def completed_groups(verified):
    groups = defaultdict(list)
    singles = []
    for row in verified.values():
        if row["pair_group_id"]:
            groups[row["pair_group_id"]].append(row)
        else:
            singles.append([row])
    return singles + [rows for rows in groups.values() if len(rows) == 2 and paired(*rows)]


def selected_episodes(verified):
    selected, counts, clean, signatures = [], Counter(), Counter(), set()
    for group in sorted(completed_groups(verified), key=lambda pair: not pair[0]["fixture_assets"]):
        bucket = group[0]["bucket"]
        if counts[bucket]+len(group) > QUOTAS[bucket] or any(row["scene_signature"] in signatures for row in group):
            continue
        if not group[0]["fixture_assets"]:
            if clean[bucket]+len(group) > QUOTAS[bucket]-OBSTACLE_MIN.get(bucket,0):
                continue
            clean[bucket]+=len(group)
        selected.extend(group)
        counts[bucket] += len(group)
        signatures.update(row["scene_signature"] for row in group)
    return selected, counts


def status_receipts(work):
    verified = {p.stem: read(p) for p in sorted((work / "accepted").glob("*.json"))}
    attempts = Counter()
    failures = defaultdict(Counter)
    live = {}
    for batch_file in sorted((work / "batches").glob("*/launch.json")):
        batch = read(batch_file)
        job = batch["job"]
        output = RUNS / job
        marker = output.with_suffix(".complete")
        pidpath = batch_file.parent / "pid"
        pid = int(pidpath.read_text()) if pidpath.exists() else 0
        cmdline = Path(f"/proc/{pid}/cmdline")
        try:
            running = job.encode() in cmdline.read_bytes() if not marker.exists() else False
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            running = False
        if running:
            batch["pid"] = pid
            live[batch["gpu"]] = batch
            continue
        if (batch_file.parent / "collected.json").exists():
            batch_failures = read(batch_file.parent / "collected.json")
        else:
            batch_failures = {}
            for row in batch["rows"]:
                key = row["case_id"]
                diag_path = output / key / "expert_diagnostic.json"
                if not diag_path.exists():
                    batch_failures[key] = "interrupted_or_sim_startup"
                    continue
                diag = read(diag_path)
                receipt = diag.get("dataset_receipt")
                if not receipt:
                    roll = diag.get("policy_rollout", {})
                    trials = roll.get("trials", [])
                    batch_failures[key] = ((trials[0].get("failure_category") if trials else None)
                                           or roll.get("status") or "preflight_rejected")
                    continue
                try:
                    admitted = admit_episode(row, receipt, diag)
                    if key not in verified:
                        write_json(work / "accepted" / f"{key}.json", admitted)
                        verified[key] = admitted
                except (ValueError, KeyError, OSError) as error:
                    batch_failures[key] = f"intake: {error}"
            write_json(batch_file.parent / "collected.json", batch_failures)
        for row in batch["rows"]:
            key = row["case_id"]
            if key in batch_failures:
                reason = batch_failures[key]
                # Physical legacy failures draw a fresh candidate. Runtime interruptions can retry the same identity.
                attempts[key] += MAX_ATTEMPTS if row["bucket"] in REUSED_H5 and reason != "interrupted_or_sim_startup" else 1
                failures[row["bucket"]][reason] += 1
    return verified, attempts, failures, live


def launch(work, gpu, rows):
    batches = work / "batches"
    sequence = len(list(batches.iterdir()))
    batch = batches / f"{sequence:05}"
    batch.mkdir()
    filename = lambda row: f"{row['case_id']}.json"
    for row in rows:
        (batch / filename(row)).hardlink_to(work / "plans" / row["case_id"] / "plan.json")
    usd = rows[0]["target_usd"]
    config = {**read(ROOT / "env_resolved.json"), "target_usd": usd}
    query = {"white_box": "box", "pepsi": "can", "mug": "mug", "basket": "basket", "carton": "box"}[rows[0]["bucket"]]
    from step1_mug_pairs import template

    manifest = {**template(), "phase": "three_view_expert_v3", "asset": usd, "target_query": query,
                "unified_lift_gate": asdict(LiftGateConfig()), "total_steps": EPISODE_STEPS,
                "fixture_assets": {f["name"]: f["usd"] for row in rows for f in row["fixtures"]},
                "cases": [{**row, "plan": filename(row), "offline_feasible": True} for row in rows]}
    write_json(batch / "config.json", config)
    write_json(batch / "manifest.json", manifest)
    job = f"{work.name}_g{gpu['gpu']}_{sequence:05}"
    runtime = Path("/dev/shm") / f"g1d-{os.getuid()}-{job}"
    runtime.mkdir(exist_ok=True)
    record = {**gpu, "job": job, "rows": rows, "log": str(RUNS / (job+".log"))}
    write_json(batch / "launch.json", record)
    env = {**os.environ, "G1D_GPU_UUID": gpu["uuid"], "G1D_RENDER_GPU": str(gpu["gpu"]),
           "G1D_RUNTIME_DIR": str(runtime), "G1D_WORKER_CONFIG": str(batch / "config.json"),
           "G1D_GROUPED_WALL_SECONDS": "3600", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}
    with open(record["log"], "w") as log:
        process = subprocess.Popen(["bash", str(ROOT / "run_lab_school.sh"), str(RUNS / job),
                                    "--grouped-manifest", str(batch / "manifest.json"), "--grasp-after",
                                    str(RUNS / "lab_school_03/result.json"), "--gpu", "0"],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (batch / "pid").write_text(str(process.pid))
    record["pid"] = process.pid
    print(f"{job} PID {process.pid} GPU {gpu['gpu']} {query} {len(rows)} episodes", flush=True)
    return process, record


def fill_batch(candidates, verified, attempts, live, counts, gpu, selected=()):
    running = {row["case_id"] for batch in live.values() for row in batch["rows"]}
    active_buckets = Counter(row["bucket"] for batch in live.values() for row in batch["rows"])
    clean = Counter(row["bucket"] for row in selected if not row["fixture_assets"])
    clean.update(row["bucket"] for batch in live.values() for row in batch["rows"] if not row["fixtures"])
    groups = defaultdict(list)
    for row in candidates:
        groups[row["pair_group_id"] or row["case_id"]].append(row)
    slots = defaultdict(list)
    for group in groups.values():
        bucket = group[0]["bucket"]
        if counts[bucket] + active_buckets[bucket] >= QUOTAS[bucket]:
            continue
        if not group[0]["fixtures"] and clean[bucket] >= QUOTAS[bucket]-OBSTACLE_MIN.get(bucket,0):
            continue
        if group[0]["pair_group_id"] and len(group) != 2:
            continue
        if group[0].get("seed_lane", gpu) != gpu:
            continue
        if len(group) == 2 and any(group[0][k] != group[1][k] for k in
                                  ("target_pose_xyz_xyzw", "home_q", "fixtures", "seed")):
            continue
        if any(row["case_id"] in running or attempts[row["case_id"]] >= MAX_ATTEMPTS for row in group):
            continue
        pending = [row for row in group if row["case_id"] not in verified]
        if not pending:
            continue
        fixtures = tuple(sorted((f["name"], f["usd"]) for f in group[0]["fixtures"]))
        slot = (bucket, group[0]["target_usd"], fixtures)
        slots[slot].append(pending)
    if not slots:
        return []
    order = list(QUOTAS)
    slot = min(slots, key=lambda s: (counts[s[0]]/QUOTAS[s[0]], active_buckets[s[0]] > 0,
                                    not bool(s[2]) if s[0] in OBSTACLE_MIN else False,
                                    -sum(map(len, slots[s])), order.index(s[0])))
    batches = slots[slot]
    batches.sort(key=lambda g: (g[0].get("seed_lane", gpu) != gpu, g[0].get("order", 0), g[0]["case_id"]))
    selected = []
    maximum = min(12, QUOTAS[slot[0]]-counts[slot[0]]-active_buckets[slot[0]])
    if not slot[2]:
        maximum = min(maximum, QUOTAS[slot[0]]-OBSTACLE_MIN.get(slot[0],0)-clean[slot[0]])
    for rows in batches:
        if len(selected)+len(rows) <= maximum:
            selected.extend(rows)
    return selected


def collect(work):
    for name in ("plans", "accepted", "batches"):
        (work / name).mkdir(parents=True, exist_ok=True)
    with (work / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (work / "legacy_ready.json").exists():
            prepare_legacy(work)
        seed_accepted(work)
        handles = []
        while True:
            handles = [p for p in handles if p.poll() is None]
            verified, attempts, failures, live = status_receipts(work)
            selected, counts = selected_episodes(verified)
            gpus = gpu_manifest()
            candidates = [read(p) for p in sorted((work / "plans").glob("*/ready.json"))]
            assert len({r["scene_id"] for r in candidates}) == len(candidates)
            assert len({r["episode_id"] for r in candidates}) == len(candidates)
            assert len({r["scene_signature"] for r in candidates}) == len(candidates)
            if counts == Counter(QUOTAS) and not live:
                if len({row["scene_id"] for row in selected}) != sum(QUOTAS.values()):
                    raise ValueError("Duplicate admitted scene identity")
                original = {r["case"]: r for r in read(RUNS/"three_view_pipeline_03/split.json")["episodes"]}
                grouped = defaultdict(list)
                for row in selected:
                    if row["bucket"] in REUSED_H5:
                        row["scene_family"] = f"tier1_legacy_{original[row['source_case']]['scene_family']}"
                        grouped[row["scene_family"]].append(row)
                for family, rows in grouped.items():
                    if len(rows)==2 and len({r["bucket"] for r in rows})==2:
                        for row in rows:
                            row["pair_group_id"] = family
                write_json(work / "collection.json", {"phase": "tier1_v3", "asset_balance": True,
                           "counts": counts, "episodes": selected, "failures": failures,
                           "passed": True})
                if (work / "progress.json").exists():
                    last = read(work / "progress.json")
                    last.update(accepted=counts, total=len(selected), remaining=dict.fromkeys(QUOTAS,0), active=[],
                                updated_unix_s=time.time())
                    write_json(work / "progress.json",last)
                record_complete(RUNS / (work.name+"_collection"), 0)
                return
            for gpu in gpus:
                index = gpu["gpu"]
                if index in live or gpu["free_mib"] < FREE_MIB:
                    continue
                if rows := fill_batch(candidates, verified, attempts, live, counts, index, selected):
                    handle, batch = launch(work, gpu, rows)
                    handles.append(handle)
                    live[index] = batch
            progress = {"phase": "tier1_collection_progress", "accepted": counts, "target": QUOTAS,
                        "total": sum(counts.values()), "planned": Counter(row["bucket"] for row in candidates),
                        "accepted_obstacle_episodes": Counter(r["bucket"] for r in selected if r["group"]=="B_obstacle"),
                        "planned_obstacle_episodes": Counter(r["bucket"] for r in candidates if r["group"]=="B_obstacle"),
                        "remaining": {k: v-counts[k] for k,v in QUOTAS.items()}, "failures": failures,
                        "active": [{k: v for k, v in b.items() if k != "rows"} for b in live.values()],
                        "gpu": gpus, "headroom_mib": FREE_MIB, "updated_unix_s": time.time()}
            write_json(work / "progress.json", progress)
            if not live and (work / "planning_exhausted.json").exists() and not any(
                    fill_batch(candidates, verified, attempts, {}, counts, g["gpu"], selected) for g in gpus):
                raise RuntimeError("Frozen candidate supply exhausted before all dataset quotas were met")
            time.sleep(12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    collect(args.output.resolve())
