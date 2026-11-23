"""Verify the independent Step 2 cases by recording and replaying expert commands."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import os
import subprocess
import time

import numpy as np

from background_status import record_complete
from step1_mug_pairs import template
from three_view_detector import sha256, write_json
from three_view_eval import read
from tier1_collect import FREE_MIB, ROOT, RUNS, admit_episode, gpu_manifest, paired, scene_signature
from unified_grasp import EPISODE_STEPS, LiftGateConfig


WORK = RUNS / "step2_v3_01"
TIER = RUNS / "tier1_v3_01"
CONFIG = ROOT / "env_resolved.json"
EVIDENCE = RUNS / "lab_school_03/result.json"
QUERIES = {"mug": "mug", "basket": "basket", "carton": "box", "box": "box", "can": "can",
           "white_box": "box", "pepsi": "can"}
BUCKETS = {"mug": 10, "basket": 10, "carton": 8}
OBSTACLES = {"mug": 4, "basket": 4, "carton": 2}


def groups(rows):
    items = defaultdict(list)
    for row in rows:
        items[row["pair_group_id"]].append(row)
    if not all(len(pair) == 2 for pair in items.values()):
        raise ValueError("Incomplete part pair")
    return list(items.values())


def wait_gpu(preferred, exclude=(), timeout=3600):
    headroom_mib = int(os.environ.get("G1D_STEP2_LAUNCH_FREE_MIB", FREE_MIB))
    if headroom_mib < 18 * 1024:
        raise ValueError("Step 2 simulator requires at least 18 GiB of free GPU memory")
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        gpus = sorted(gpu_manifest(), key=lambda gpu: (gpu["gpu"] != preferred, -gpu["free_mib"]))
        for gpu in gpus:
            if gpu["gpu"] not in exclude and gpu["free_mib"] >= headroom_mib:
                return gpu
        print(f"Waiting for a GPU with {headroom_mib} MiB free for Isaac Sim", flush=True)
        time.sleep(30)
    raise RuntimeError("No GPU reached the Isaac Sim memory threshold")


def adopt(job):
    """Keep an existing shard running when replacing its coordinator."""
    done = RUNS / f"{job}.complete"
    if done.exists():
        return
    path = RUNS / f"{job}.pid"
    if not path.exists():
        return
    proc = Path("/proc") / path.read_text().strip()
    command = str(RUNS / job).encode()
    print(f"Checking existing shard {job}", flush=True)
    while not done.exists():
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            return
        if command not in args:
            return
        time.sleep(10)


def simulate(stage, rows, phase, *, policy=None, detector=None, prompts=None, archives=None, preferred=2, exclude=(), lanes=None, resume_completed=False, allocator=None, max_shard_cases=None, retries=3):
    plans = WORK / "sim"
    plans.mkdir(exist_ok=True)
    buckets = defaultdict(list)
    for row in rows:
        key = row["target_usd"], tuple(sorted((f["name"], f["usd"]) for f in row["fixtures"]))
        buckets[key].append(row)
    if max_shard_cases:
        chunks = [(key,rows[i:i+max_shard_cases]) for key,rows in sorted(buckets.items())
                  for i in range(0,len(rows),max_shard_cases)]
    else:
        chunks = sorted(buckets.items())
    indexed = list(enumerate(chunks))
    if not indexed:
        return {}

    def shard(i, item, preferred, exclude):
        (usd, fixtures), subset = item
        output = {}
        tag = f"{WORK.name}_{stage}_{i:02}"
        folder = plans / tag
        folder.mkdir(exist_ok=True)
        role = subset[0]["bucket"]
        definitions = []
        for row in subset:
            case_id = row["case_id"]
            original = read(row["plan"])
            name = f"{case_id}.json"
            write_json(folder / name, original)
            spec = dict(row, plan=name, offline_feasible=True)
            if archives:
                spec["replay_h5"] = {key: archives[case_id][key] for key in ("path", "sha256")}
            definitions.append(spec)
        cfg = folder / "config.json"
        write_json(cfg, dict(read(CONFIG), target_usd=usd))
        manifest = {**template(), "phase": phase, "asset": usd, "asset_role": role, "three_view": True,
                    "target_query": subset[0].get("target_query") or QUERIES[role], "total_steps": EPISODE_STEPS,
                    "fixture_assets": dict(fixtures), "cases": definitions,
                    "unified_lift_gate": asdict(LiftGateConfig()),
                    "initial_head_prompts": {r["case_id"]: prompts[r["case_id"]] for r in subset} if prompts else {}}
        if policy:
            manifest.update(policy=policy, detector=detector)
        else:
            manifest.pop("policy", None)
            manifest.pop("detector", None)
        path = folder / "manifest.json"
        write_json(path, manifest)
        if resume_completed and phase not in {"three_view_closed_loop","three_view_expert_v3","three_view_unified_replay"}:
            raise ValueError("This simulator phase has no supported per-case resume contract")
        expected = {r["case_id"] for r in subset}

        def harvest(root, filename):
            receipt = root / filename
            if not receipt.is_file():
                return
            summary = read(receipt)
            if summary["phase"] != phase or summary["asset_usd"] != usd:
                raise ValueError(f"Unexpected shard identity in {receipt}")
            new = []
            for row in summary["cases"]:
                case = row["case"]
                if case not in expected:
                    raise ValueError(f"Unplanned case in {receipt}: {case}")
                if case in output:
                    continue
                diagnostic = root / case / "expert_diagnostic.json"
                if phase != "three_view_closed_loop":
                    if row["status"] in {"offline_reject","preflight_fail","physical_fail","strict_success"}:
                        if row["status"]=="strict_success" and not diagnostic.is_file():
                            continue
                        output[case] = row, read(diagnostic) if diagnostic.is_file() else {}, str(diagnostic)
                        new.append(case)
                    continue
                if not diagnostic.is_file() or not (diagnostic.parent / "physics_trace.jsonl").is_file():
                    continue
                raw = read(diagnostic)
                result = row.get("policy_result", {})
                roll = raw.get("policy_rollout", {})
                status, steps = roll.get("status"), roll.get("steps", 0)
                if (status in {"success", "lift_failed", "timeout"}
                        and status == result.get("status") and steps == result.get("steps")
                        and 0 < steps <= EPISODE_STEPS
                        and (status != "timeout" or steps == EPISODE_STEPS)):
                    output[case] = row, raw, str(diagnostic)
                    new.append(case)
            if new and filename == "batch_progress.json":
                print(f"Preserved {len(new)} finished cases from {root.name}", flush=True)

        for attempt in range(retries):
            job = tag if not attempt else f"{tag}_retry{attempt}"
            root = RUNS / job
            if root.exists() or (RUNS / f"{job}.pid").exists():
                adopt(job)
            marker = RUNS / f"{job}.complete"
            if resume_completed and root.exists():
                receipt = "result.json" if marker.exists() and read(marker)["outcome"] == "PASS" else "batch_progress.json"
                harvest(root, receipt)
                if set(output) == expected:
                    return output
            if marker.exists():
                if not resume_completed and read(marker)["outcome"] == "PASS":
                    break
                continue
            if root.exists():
                print(f"Preserving incomplete simulator run {root}", flush=True)
                continue
            pending = [r for r in definitions if r["case_id"] not in output]
            attempt_path = path
            if resume_completed and len(pending) < len(definitions):
                attempt_path = folder / f"manifest_retry{attempt}.json"
                include = {r["case_id"] for r in pending}
                write_json(attempt_path, {**manifest, "cases":pending,
                    "initial_head_prompts":{k:v for k,v in manifest["initial_head_prompts"].items() if k in include}})
            wall_seconds = (max(3000, 1200 * len(pending) + 600)
                            if resume_completed and phase=="three_view_closed_loop" else 3000)
            lease = allocator.claim(job) if allocator else nullcontext(wait_gpu(preferred, exclude))
            with lease as gpu:
                runtime = Path("/dev/shm") / f"g1d-{os.getuid()}-{job}"
                runtime.mkdir(exist_ok=True)
                env = {**os.environ, "G1D_WORKER_CONFIG": str(cfg), "G1D_GPU_UUID": gpu["uuid"],
                       "G1D_RENDER_GPU": str(gpu["gpu"]), "G1D_RUNTIME_DIR": str(runtime),
                       "G1D_GROUPED_WALL_SECONDS": str(wall_seconds), "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}
                command = ["bash", str(ROOT / "run_lab_school.sh"), str(root), "--grouped-manifest", str(attempt_path),
                           "--grasp-after", str(EVIDENCE), "--gpu", "0"]
                with (RUNS / f"{job}.log").open("w") as log:
                    fds = (gpu["lease_fd"],) if "lease_fd" in gpu else ()
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            pass_fds=fds)
                    (RUNS / f"{job}.pid").write_text(str(proc.pid))
                    print(f"{job} on GPU {gpu['gpu']}, PID {proc.pid}, {len(pending)} cases", flush=True)
                    returncode = proc.wait()
            if resume_completed:
                harvest(root, "result.json" if returncode == 0 else "batch_progress.json")
                if set(output) == expected:
                    return output
            elif returncode == 0:
                break
        else:
            raise RuntimeError(f"Simulator failed all attempts for {tag}")
        summary = read(root / "result.json")
        if {r["case"] for r in summary["cases"]} != {r["case_id"] for r in subset}:
            raise ValueError(f"Incomplete physical shard {root}")
        for row in summary["cases"]:
            diag = root / row["case"] / "expert_diagnostic.json"
            output[row["case"]] = row, read(diag) if diag.is_file() else {}, str(diag)
        return output

    if allocator is not None:
        with ThreadPoolExecutor(max_workers=3) as pool:
            pending = [pool.submit(shard,i,item,preferred,exclude) for i,item in indexed]
            return {case:receipt for job in pending for case,receipt in job.result().items()}

    if lanes is None:
        return {case: receipt for i,item in indexed for case,receipt in shard(i,item,preferred,exclude).items()}

    def lane(number, gpu):
        excluded = tuple(g for g in (0,1,2) if g != gpu)
        return {case: receipt for i,item in indexed[number::len(lanes)]
                for case,receipt in shard(i,item,gpu,excluded).items()}

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        pending = [pool.submit(lane, i, gpu) for i,gpu in enumerate(lanes)]
        return {case: receipt for job in pending for case,receipt in job.result().items()}


@lru_cache(maxsize=1)
def rejection_sets():
    entries = read(TIER / "collection.json")["episodes"]
    final = read(TIER / "final_test_reservation.json")
    return (set(final["existing_frozen_signatures"]) | {r["scene_signature"] for r in entries},
            {r["seed"] for r in final["slots"]}, {r["scene_id"] for r in final["slots"]},
            set(final["heldout_asset_families"]), {r["scene_family"] for r in entries})


def verify_isolation(rows):
    signatures, seeds, ids, assets, families = rejection_sets()
    seen = set()
    for row in rows:
        signature = scene_signature(row)
        if (signature != row["scene_signature"] or signature in signatures or signature in seen
                or row["seed"] in seeds or row["scene_id"] in ids or row["asset_family"] in assets
                or row["scene_family"] in families):
            raise ValueError(f"Overlapping or changed development definition: {row['case_id']}")
        seen.add(signature)


def replacement_pairs(bucket, obstacle, used):
    from tier1_plan import START, candidate, fixture_bank, references, solve_pair

    config = read(CONFIG)
    fixtures = fixture_bank(config)
    config_evidence = read(EVIDENCE)
    catalog = references(bucket)
    cfg = dict(config, target_usd=next(iter(catalog.values()))[0][0]["target_usd"])
    folder = WORK / "replacement_plans"
    folder.mkdir(exist_ok=True)
    indices = list(range(90, 210))
    if bucket == "carton" and not obstacle:
        # Retain the existing first replacement. Prioritize independently perturbed
        # positive-yaw placements after negative yaw hid the requested end.
        indices.sort(key=lambda index: (index != 90, index % 6 != 5, index))
    for index in indices:
        if (index % 3 == 1) != obstacle:
            continue
        seq = 1000 + 200 * list(START).index(bucket) + index
        docs = solve_pair(cfg, config_evidence, list(candidate(bucket, index, catalog, fixtures, cfg,
                                                               dev=True, dev_index=seq)))
        if len(docs) != 2:
            continue
        rows = []
        for part_index, doc in enumerate(docs):
            row = doc["case"]
            case_id = f"{row['group'][0]}_{6000+240*list(START).index(bucket)+2*(index-90)+part_index}"
            row.update(case_id=case_id, episode_id=f"step2_dev_{case_id}")
            group = row["pair_group_id"] + "_step2"
            row.update(scene_id=row["scene_id"]+"_step2", scene_family=group, pair_group_id=group)
            path = folder / f"{row['case_id']}.json"
            write_json(path, doc)
            rows.append(dict(row, bucket=bucket, scene_signature=scene_signature(row), plan=str(path)))
        if any(row["scene_signature"] in used for row in rows):
            continue
        try:
            verify_isolation(rows)
        except ValueError:
            continue
        yield rows


def expert_round(rows, number):
    verify_isolation(rows)
    recorded = simulate(f"demo{number}", rows, "three_view_expert_v3", lanes=(2,1))
    demos, audit = {}, {}
    for row in rows:
        key = row["case_id"]
        result, diag, path = recorded[key]
        audit[key] = {"case": key, "diagnostic": path, "record": result["status"], "failure": result.get("failure")}
        receipt = result.get("dataset_receipt")
        if not receipt:
            continue
        try:
            demos[key] = admit_episode(row, receipt, diag)
        except ValueError as exc:
            audit[key]["failure"] = str(exc)
    candidates = [row for pair in groups(rows) if all(r["case_id"] in demos for r in pair)
                  and paired(*(demos[r["case_id"]] for r in pair)) for row in pair]
    replay = simulate(f"replay{number}", candidates, "three_view_unified_replay", archives=demos, lanes=(2,1))
    valid = {}
    for r in candidates:
        key = r["case_id"]
        summary, diag, path = replay[key]
        roll = diag.get("policy_rollout", {})
        alignment = diag.get("replay_alignment", {})
        ok = (roll.get("success") and roll.get("part_hit") and roll["steps"] <= EPISODE_STEPS
              and alignment.get("max_timestamp_error_s", 1) <= 2e-7
              and alignment.get("max_native_setpoint_error", 1) <= 1e-6
              and abs(alignment.get("trigger_delta_s", 1)) <= .10 + 1e-8)
        audit[key].update(replay=summary["status"], replay_diagnostic=path, replay_status=roll.get("status"),
                          part_hit=roll.get("part_hit"), alignment=alignment, passed=bool(ok))
        if ok:
            demo = demos[key]
            valid[key] = {**r, "expert_path": demo["path"], "expert_sha256": demo["sha256"],
                          "expert_diagnostic": path, "expert_alignment": alignment, "replay_verified": True,
                          "initial_q": demo["initial_q"], "initial_target": demo["initial_target"],
                          "initial_width": demo["initial_width"],
                          "head_prompt": {"bbox_0": demo["bbox"], "affordance_0": demo["point"]}}
    keep = [valid[r["case_id"]] for pair in groups(rows) if all(r["case_id"] in valid for r in pair) for r in pair]
    write_json(WORK / f"expert_round_{number}.json", {"attempts": list(audit.values()), "accepted": len(keep)})
    print(f"Expert round {number}: {len(keep)}/{len(rows)} passed recording and replay in complete pairs", flush=True)
    return keep


def verify_dev():
    original = read(TIER / "development.json")["cases"]
    reserve = [read(p) for p in sorted((TIER / "dev_plans").glob("*/ready.json"))]
    used = {row["scene_signature"] for row in original}
    accepted = expert_round(original, 0)
    tried = {r["pair_group_id"] for r in original}
    generators = {}
    for attempt in range(1, 13):
        missing = Counter((r["bucket"], bool(r["fixtures"])) for r in original)
        missing.subtract(Counter((r["bucket"], bool(r["fixtures"])) for r in accepted))
        if not +missing:
            break
        replacements = []
        for (bucket, obstacle), count in (+missing).items():
            spare = [p for p in groups(reserve) if p[0]["bucket"] == bucket
                     and bool(p[0]["fixtures"]) == obstacle and p[0]["pair_group_id"] not in tried]
            key = bucket, obstacle
            for _ in range(count // 2):
                pair = next((p for p in spare if p[0]["pair_group_id"] not in tried and
                             all(row["scene_signature"] not in used for row in p)), None)
                if pair is None:
                    if key not in generators:
                        generators[key] = replacement_pairs(bucket, obstacle, used)
                    pair = next(generators[key])
                tried.add(pair[0]["pair_group_id"])
                used.update(r["scene_signature"] for r in pair)
                replacements.extend(pair)
        accepted.extend(expert_round(replacements, attempt))
    verify_isolation(accepted)
    if (Counter(row["bucket"] for row in accepted) != BUCKETS
            or Counter(row["bucket"] for row in accepted if row["fixtures"]) != OBSTACLES):
        raise RuntimeError("Did not obtain the 28 independent replay-verified scenes")
    initial = {r["case_id"] for r in original}
    report = {"phase": "step2_dev_verified", "replay_verified": True, "count": len(accepted),
              "source_sha256": sha256(TIER / "development.json"), "cases": accepted,
              "replacements": sorted(initial - {r["case_id"] for r in accepted})}
    write_json(WORK / "development.json", report)
    print("Verified and froze 28/28 development scenarios", flush=True)


if __name__ == "__main__":
    try:
        verify_dev()
    except BaseException:
        record_complete(RUNS / f"{WORK.name}_dev", 1)
        raise
    else:
        record_complete(RUNS / f"{WORK.name}_dev", 0)
