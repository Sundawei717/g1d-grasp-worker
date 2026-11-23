"""Collect only vetted assets, with durable per-attempt and per-pair admission."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
import hashlib
import json
import random

import h5py

import numpy as np
from scipy.spatial.transform import Rotation

import step2_dev
from step3a_planner import adapt, build_planner, fixture, placement, pose_matrix
from step3a_verify import admits_pair, fixture_or_pregrasp_contact, layout, original_cases, replay_ok
from step3b_common import ASSETS, DEV, FROZEN, FROZEN_SHA, NAME, TIER1, WORK, Resources, check_pause, now, read, save
from three_view_detector import sha256
from tier1_collect import admit_episode, scene_signature, verify_receipt
from tier1_plan import base_mesh_signature

PILOTS = ASSETS.parent
COLLECT = WORK / "collect"
MUG, BASKET, BOX, WHITE, PEPSI = ("mug/00007", "basket/00001", "sand3d_packaging/00208",
                                    "sand3d_packaging/00152", "sand3d_packaging/00156")
SUGAR, YCB_MUG, RAISED, MARKER, COMPACT, ROUND, SODA = (
    "ycb/004_sugar_box", "ycb/025_mug", "basket/raised_rim_train", "marker_box/00003",
    "sand3d_packaging/00000", "sand3d_packaging/00225", "soda_can/00000")
ASSET_ORDER = (MUG, BASKET, BOX, WHITE, PEPSI, SUGAR, YCB_MUG, RAISED, MARKER, COMPACT, ROUND, SODA)
TARGET = 1140
MODE_TARGETS = {"paired": 456, "obstacle": 400, "pose": 284}
PHYSICAL_ATTEMPTS = 40
PLAN_TRIALS = 10


def schedule():
    groups = [("paired", (k, k)) for k, quota in
              {MUG: 50, BASKET: 34, BOX: 29, SUGAR: 23, YCB_MUG: 24, RAISED: 23}.items() for _ in range(quota)]
    groups += [("paired", pair) for pair, quota in
               { (WHITE, PEPSI): 8, (WHITE, COMPACT): 7, (PEPSI, ROUND): 7,
                 (MARKER, COMPACT): 8, (MARKER, SODA): 7, (ROUND, SODA): 8}.items() for _ in range(quota)]
    for mode, quotas in (("obstacle", (28,28,36,8,8,8,8,10,17,17,16,16)),
                         ("pose", (17,13,15,5,5,9,8,7,16,16,16,15))):
        groups += [(mode, (k, k)) for k, quota in zip(ASSET_ORDER, quotas, strict=True) for _ in range(quota)]
    warmup = [groups.pop(groups.index(("paired", (k, k)))) for k in (MUG, MUG, BASKET)]
    random.Random(33889).shuffle(groups)
    result = [{"id": i, "mode": mode, "assets": list(assets)} for i, (mode, assets) in enumerate(warmup + groups)]
    assert Counter(t["mode"] for t in result) == {m: v//2 for m, v in MODE_TARGETS.items()}
    assert sum(2 for _ in result) == TARGET
    return result


@lru_cache(maxsize=None)
def digest(path):
    return sha256(path)


@lru_cache(maxsize=None)
def mesh_hash(path):
    return base_mesh_signature(path)


@lru_cache(maxsize=1)
def inputs():
    if digest(FROZEN) != FROZEN_SHA:
        raise ValueError("Frozen benchmark digest changed")
    frozen = read(FROZEN)["cases"]
    old = read(TIER1)
    assets = read(ASSETS)
    if len(old["episodes"]) != 460 or assets["count"] != 12 or len(frozen) != 80 or len(read(DEV)["cases"]) != 28:
        raise ValueError("Missing the signed Tier-1, 12-asset, or benchmark inputs")
    old_rows = original_cases()
    held = {row["base_mesh_hash"] for row in frozen if row["tier"] == 4}
    approved = set(assets["tier1"].values()) | {row["base_mesh_hash"] for row in assets["additional"].values()}
    if len(approved) != 12 or len(held) != 5 or held & approved:
        raise ValueError("Training base mesh isolation failed")
    sealed = old_rows + frozen
    return old, assets, approved, {
        "layouts": {layout(r) for r in sealed},
        "scene_id": {r.get("scene_id") for r in sealed},
        "episode_id": {r.get("episode_id") for r in sealed},
        "seed": {r.get("seed") for r in sealed},
        "scene_signature": {scene_signature(r) for r in sealed}}


def archived_reference(row, case):
    """Recover wrist goals from the final straight approach of a verified rerecording."""
    with h5py.File(row["path"], "r") as episode:
        recorded = json.loads(episode["meta/case_json"].asstr()[()])
        close = np.flatnonzero(episode["phase"].asstr()[:] == "close")
        if (recorded["scene_id"] != case["scene_id"] or not len(close)
                or int(close[0]) < 12):
            raise ValueError(f"Misaligned archived grasp reference for {row['case']}")
        end = int(close[0])
        poses = episode["obs/eef_pose"][:end+1]
        if not np.isfinite(poses).all():
            raise ValueError(f"Nonfinite archived wrist poses for {row['case']}")
        wrist = poses[end]
        hover = wrist.copy()
        hover[:3] -= .072 * Rotation.from_quat(wrist[3:]).apply([1, 0, 0])
        distance = np.linalg.norm(poses[:end, :3]-hover[:3], axis=1)
        near = int(distance.argmin())
        hover_q = episode["obs/qpos"][near]
    if not np.isfinite(hover_q).all() or distance[near] > .02:
        raise ValueError(f"Invalid archived final approach for {row['case']}")
    return {"candidate": "archived_observed_wrist",
            "hover_xyz_xyzw": hover.tolist(), "wrist_xyz_xyzw": wrist.tolist(),
            "pre_grasp_q": [case["home_q"]], "approach_q": [hover_q.tolist()]}


@lru_cache(maxsize=1)
def bank():
    old, assets, approved, _ = inputs()
    refs = [(r, TIER1.parent / "plans" / r["case"] / "plan.json") for r in old["episodes"]]
    keep = {sid for asset in assets["additional"].values() for sid in asset["accepted_cases"]}
    receipts = [*PILOTS.glob("pilots_guide*_result.json"), *PILOTS.glob("pilots_sugar*_result.json"),
                PILOTS / "pilots_progress.json", PILOTS / "substitutes_progress.json"]
    added = {r["scene_id"]: r for file in receipts for r in read(file)["cases"] if r["scene_id"] in keep}
    if set(added) != keep:
        raise ValueError("Missing a verified Tier-2 source plan")
    refs.extend((r, Path(r["plan"])) for r in added.values())
    groups, clean, blocked, pool = defaultdict(list), defaultdict(list), defaultdict(list), {}
    for row, p in refs:
        if not p.exists():
            continue
        doc = read(p)
        case = doc["case"]
        asset = case["asset_family"]
        if not doc["feasible"] or asset not in ASSET_ORDER:
            continue
        if mesh_hash(case["target_usd"]) not in approved:
            raise ValueError(f"Source uses an unapproved training base mesh: {asset}")
        case["bucket"] = row["bucket"]
        case["base_mesh_hash"] = mesh_hash(case["target_usd"])
        goal = doc["feasible"][0]
        if goal["candidate"] == "archived_native_prefix":
            doc["feasible"] = [archived_reference(row, case)]
        elif not goal.get("pre_grasp_q") or not goal.get("approach_q") or not {
            "hover_xyz_xyzw", "wrist_xyz_xyzw",
        } <= goal.keys():
            raise ValueError(f"Incomplete source plan for {row['case']}")
        clean[asset].append(doc)
        if case["fixtures"]:
            blocked[asset].append(doc)
            for f in case["fixtures"]:
                pool[f["usd"]] = f
        elif row.get("pair_group_id"):
            groups[asset, row["pair_group_id"]].append(doc)
    pairs = defaultdict(list)
    for (asset, _), group in groups.items():
        if len(group) == 2 and len({r["case"]["part"] for r in group}) == 2:
            pairs[asset].append(group)
    if set(clean) != set(ASSET_ORDER) or any(not pairs[a] for a in (MUG, BASKET, BOX, SUGAR, YCB_MUG, RAISED)):
        raise ValueError("A scheduled group type lacks vetted references")
    for tag, p in (("panel", PILOTS / "new_red_panel.usda"), ("panel", PILOTS / "tall_blue_panel.usda")):
        pool[str(p)] = {"usd": str(p), "sha256": digest(p), "asset_family": f"step3a/{p.stem}"}
    return clean, pairs, blocked, tuple(pool.values())


def retarget(source, pose, home, seed):
    old = source["case"]
    fit = source["feasible"][0]
    case = {**old, "target_pose_xyz_xyzw": pose, "home_q": home, "fixtures": [], "seed": seed,
            "planner_hint": {"guide": [*fit["pre_grasp_q"], fit["approach_q"][0]]},
            "lift_height_m": .08, "command_delay_substeps": 0}
    case.pop("gate_override", None)
    case["yaw_deg"] = float(Rotation.from_quat(pose[3:]).as_euler("zyx", degrees=True)[0])
    planner = build_planner(case)
    transform = pose_matrix(pose) * pose_matrix(old["target_pose_xyz_xyzw"]).inverse()
    hover = transform * pose_matrix(fit["hover_xyz_xyzw"])
    wrist = transform * pose_matrix(fit["wrist_xyz_xyzw"])
    planner.candidate_goals = lambda: iter([("reference_retarg", hover, wrist)])
    result = planner.plan(seed=seed, stop_after=1, max_candidates=1, lift_height_m=.08)
    if not result["feasible"]:
        return None
    return {**source, **result, "case": case, "obstacles": [], "detour": None,
            "home_q": home, "target_pose_xyz_xyzw": pose}


def randomized_pose(doc, rng, mode, j, warmup):
    source = doc["case"]
    pose = np.array(source["target_pose_xyz_xyzw"], dtype=float)
    width = .030 if mode == "pose" else .015
    spread = rng.uniform(.007, width, size=2) * rng.choice([-1, 1], 2)
    pose[:2] = np.clip(pose[:2] + spread, [.065, -.515], [.295, -.33])
    yaw = rng.uniform(-13, 13) if mode == "pose" else rng.uniform(-4, 4)
    pose[3:] = (Rotation.from_euler("z", yaw, degrees=True) * Rotation.from_quat(pose[3:])).as_quat()
    home = np.array(source["home_q"], dtype=float)
    if mode == "pose" or (j > 1 and not warmup):
        home += rng.normal(0, .035 if mode == "pose" else .009, 7)
        home[3] = max(home[3], -1.012)
    return pose.tolist(), home.tolist()


def with_obstacle(clear, rng, blocked, pool):
    center = np.array(clear[0]["case"]["target_pose_xyz_xyzw"][:2])
    options = []
    for example in rng.permutation(blocked)[:4]:
        old = example["case"]
        if len(old["fixtures"]) != 1:
            continue
        f = dict(old["fixtures"][0])
        pos = np.asarray(f["pose"], dtype=float)
        pos[:2] += center - np.array(old["target_pose_xyz_xyzw"][:2])
        f["pose"] = pos.tolist()
        options.append(f)
    for i in range(14):
        f = dict(pool[int(rng.integers(len(pool)))])
        dx, dy = ((-.075,-.07),(.075,-.07),(0,-.11),(-.05,-.115),(.05,-.115),
                  (0,.075),(.043,.075),(-.043,.075),(-.11,0),(.11,0))[i % 10]
        try:
            f["pose"] = placement(f["usd"], center[0] + dx, center[1] + dy, float(rng.uniform(-24, 24)))
        except ValueError:
            continue
        options.append(f)
    # Prefer varied obstacles, with verified translated layouts as a fallback.
    for idx in rng.permutation(len(options)):
        f = {**options[idx], "name": "blocker"}
        if np.linalg.norm(np.subtract(f["pose"][:2], center)) < .06:
            continue
        paths = [adapt(doc, {**doc["case"], "fixtures": [f]}, block=True) for doc in clear]
        if all(paths):
            return paths
    return None


def build_group(task, attempt):
    clean, pairs, blocked, pool = bank()
    a, b = task["assets"]
    mode = task["mode"]
    offset = 4000000 + task["id"]*1000 + attempt*2
    rng = np.random.default_rng(offset)
    for trial in range(PLAN_TRIALS):
        use_pair = a == b and bool(pairs[a])
        # Use native part pairs where available. Single-part assets get independent
        # instances, or appear in a two-target scene with a second approved asset.
        if use_pair:
            collection = pairs[a]
            if mode == "pose" and trial < 5:
                yawed = [group for group in collection if abs(group[0]["case"].get("yaw_deg") or 0) >= 45]
                collection = yawed or collection
            refs = collection[int(rng.integers(len(collection)))]
        else:
            refs = [clean[k][int(rng.integers(len(clean[k])))] for k in (a,b)]
        poses = [randomized_pose(ref, rng, mode, trial, task["id"] < 3) for ref in refs]
        if use_pair:
            poses[1] = poses[0]
        if a != b:
            # Side-by-side workstations keep two independently graspable objects
            # separated even when both vetted reference grasps came from one station.
            left, right = poses
            if trial >= 2 or np.linalg.norm(np.subtract(left[0][:2], right[0][:2])) < .105:
                low, high = ([float(rng.uniform(.094,.126)),float(rng.uniform(-.435,-.390))],
                             [float(rng.uniform(.228,.258)),float(rng.uniform(-.454,-.412))])
                slots = (high,low) if trial%2 else (low,high)
                left[0][:2], right[0][:2] = slots
            poses[1] = right[0], left[1]
        paths = [retarget(ref, *pos, offset+i) for i, (ref,pos) in enumerate(zip(refs,poses,strict=True))]
        if not all(paths):
            continue
        if mode == "obstacle":
            if use_pair:
                paths = with_obstacle(paths, rng, blocked[a], pool)
            else:
                choices = [with_obstacle([doc], rng, blocked[a], pool) for doc in paths]
                paths = [q[0] for q in choices] if all(choices) else None
        if a != b:
            fixtures = [fixture(doc["case"], "neighbor") for doc in paths]
            paths = [adapt(doc, {**doc["case"], "fixtures": [fixtures[1-i]]}) for i, doc in enumerate(paths)]
        if paths and all(paths):
            return paths, use_pair or a != b
    return None


def emit(task, attempt, result):
    _, _, approved, known = inputs()
    docs, is_pair = result
    rows = []
    tid = task["id"]
    gid = f"tier2_{task['mode']}_{tid:04}"
    code = {"paired": "A_target", "obstacle": "B_obstacle", "pose": "C_start"}[task["mode"]]
    for i, doc in enumerate(docs):
        case = dict(doc["case"])
        cid = f"{code[0]}_{100000 + tid*1000 + attempt*2 + i}"
        scene = f"{gid}_a{attempt}_{i}"
        case.update(case_id=cid, episode_id=f"tier2_{cid}", scene_id=scene, scene_family=gid,
                    pair_group_id=gid if is_pair else f"{gid}_{i}", split="tier2_new", group=code)
        row = {**case, "layout_signature": layout(case), "scene_signature": scene_signature(case)}
        if case["base_mesh_hash"] not in approved or row["layout_signature"] in known["layouts"]:
            return None
        if any(row[key] in known[key] for key in ("scene_id", "episode_id", "seed", "scene_signature")):
            return None
        path = COLLECT / "plans" / f"{cid}.json"
        save(path, {**doc, "case": case})
        rows.append({**row, "plan": str(path)})
    if not is_pair and len({r["layout_signature"] for r in rows}) != 2:
        return None
    return {"task": task, "attempt": attempt, "is_pair": is_pair, "rows": rows}


def save_progress(receipts):
    rows = [r for p in receipts for r in p["episodes"]]
    counts = Counter(r["asset_family"] for r in rows)
    total = Counter(a for t in schedule() for a in t["assets"])
    state = {"stage": 1, "updated_at": now(), "accepted": len(rows), "expected": TARGET, "combined_total": 460+len(rows),
             "assets": {a: {"accepted": counts[a], "target": n} for a,n in total.items()},
             "modes": {m: {"accepted": sum(len(p["episodes"]) for p in receipts if p["task"]["mode"]==m), "target": n}
                       for m,n in MODE_TARGETS.items()}}
    save(WORK / "stage1_progress.json", state)
    return state



def run_batch(batch, short=False):
    prepared = batch["prepared"]
    rows = [r for p in prepared for r in p["rows"]]
    if not rows:
        return [], []
    step2_dev.WORK = WORK
    stamp = f"collect_{batch['id']:04}"
    alloc = Resources()
    demos = step2_dev.simulate(f"{stamp}_record", rows, "three_view_expert_v3",
                              resume_completed=True, allocator=alloc, max_shard_cases=2 if short else 5,
                              retries=64)
    admitted, failures = {}, []
    for row in rows:
        cid = row["case_id"]
        result, diag, path = demos[cid]
        try:
            if not result.get("dataset_receipt"):
                trial = (diag.get("policy_rollout", {}).get("trials") or [{}])[0]
                raise ValueError(trial.get("failure_category") or result.get("failure") or result["status"])
            admitted[cid] = admit_episode(row, result["dataset_receipt"], diag,
                                          enforce_contact_filter=fixture_or_pregrasp_contact(diag))
        except ValueError as e:
            failures.append({"case": cid, "stage": "record", "reason": str(e), "diagnostic": path})
    eligible = []
    for group in prepared:
        g = group["rows"]
        if not all(r["case_id"] in admitted for r in g):
            continue
        if group["is_pair"] and not admits_pair(g, admitted):
            failures.append({"case": g[0]["case_id"], "stage": "pair", "reason": "paired initial condition failed"})
            continue
        eligible.append(group)
    replays = step2_dev.simulate(f"{stamp}_replay", [r for p in eligible for r in p["rows"]],
                                "three_view_unified_replay", archives=admitted, resume_completed=True,
                                allocator=alloc, max_shard_cases=2 if short else 5, retries=64)
    receipts = []
    for group in eligible:
        proof = []
        for r in group["rows"]:
            cid = r["case_id"]
            _, diag, path = replays[cid]
            if replay_ok(diag):
                proof.append({**admitted[cid], "seed": r["seed"], "layout_signature": r["layout_signature"],
                              "base_mesh_hash": r["base_mesh_hash"], "replay_diagnostic": path,
                              "replay_sha256": sha256(path), "replay_verified": True})
            else:
                trial = (diag.get("policy_rollout", {}).get("trials") or [{}])[0]
                failures.append({"case": cid, "stage": "replay", "reason": trial.get("failure_category"),
                                 "diagnostic": path})
        if len(proof) == 2:
            receipts.append({"task": group["task"], "attempt": group["attempt"], "episodes": proof})
    return receipts, failures


def assert_unique(receipts):
    _, _, approved, pinned = inputs()
    for field in ("scene_id", "episode_id", "seed", "scene_signature", "layouts"):
        owners = {}
        for entry in receipts:
            for row in entry["episodes"]:
                if row["base_mesh_hash"] not in approved or not row["replay_verified"]:
                    raise ValueError("Invalid admission receipt")
                v = row["layout_signature"] if field == "layouts" else row[field]
                owner = entry["task"]["id"]
                if v in pinned[field] or (v in owners and (field != "layouts" or owners[v] != owner)):
                    raise ValueError(f"Reused {field}: {v}")
                owners[v] = owner


def merge(receipts):
    old, _, _, _ = inputs()
    assert_unique(receipts)
    buckets = defaultdict(list)
    for r in receipts:
        key = (r["task"]["mode"], *r["task"]["assets"])
        buckets[key].append(r)
    stable = lambda r: hashlib.sha256(f"tier2_split_{r['task']['id']}".encode()).digest()
    for group in buckets.values():
        group.sort(key=stable)
    budget = {key: len(g)//5 for key,g in buckets.items()}
    extra = len(receipts)//5 - sum(budget.values())
    remainder = sorted(budget, key=lambda k: (-(len(buckets[k]) % 5), k))
    for key in remainder[:extra]:
        budget[key] += 1
    new = [{**row, "split": "validation" if i < budget[key] else "train"}
           for key, g in buckets.items() for i,p in enumerate(g) for row in p["episodes"]]
    merged = [*old["episodes"], *new]
    counts = Counter(r["split"] for r in merged)
    owners = defaultdict(set)
    for r in merged:
        owners[r.get("pair_group_id") or r["scene_family"]].add(r["split"])
    split_cuts = [key for key, values in owners.items() if len(values) != 1]
    if (len(merged) != 1600 or counts != {"train":1271, "validation":329} or split_cuts
            or len({r["case"] for r in merged}) != len(merged)
            or len({r["scene_id"] for r in merged}) != len(merged)
            or len({r["episode_id"] for r in merged}) != len(merged)):
        raise ValueError("Merge or paired split is inconsistent")
    for row in new:
        verify_receipt(row)
    save(WORK / "training_split.json", {"phase": "three_view_v3_split", "episodes": merged,
         "asset_balance": True, "counts": counts, "tier1_split_sha256": digest(TIER1),
         "frozen_sha256": FROZEN_SHA, "cross_split_pair_groups": 0})
    with (WORK / "data_report.md").open("w", encoding="utf-8") as out:
        out.write("# Tier-2 数据盘点\n\nTier-1 460 条原分区保持不变。新增 1,140 条通过 8 cm 原动作回放、Part Hit 和无损 HDF5 验收。\n"
                  "\n训练 1,271 条，验证 329 条。配对跨区数为 0。\n\n"
                  "| 资产 | Tier-1 | 新增训练 | 新增验证 |\n|---|---:|---:|---:|\n")
        for a in ASSET_ORDER:
            line = [sum(r["asset_family"] == a for r in old["episodes"])]
            line.extend(sum(r["asset_family"] == a and r["split"] == s for r in new) for s in ("train","validation"))
            out.write(f"| {a} | {line[0]} | {line[1]} | {line[2]} |\n")
        out.write("\n新增配比为 456 多物／部位配对、400 绕障、284 大偏航／多起姿。")


def collect():
    for folder in ("plans", "batches", "accepted"):
        (COLLECT / folder).mkdir(parents=True, exist_ok=True)
    tasks = schedule()
    schedule_path = COLLECT / "schedule.json"
    if schedule_path.exists() and read(schedule_path) != tasks:
        raise ValueError("Persisted collection schedule changed")
    save(schedule_path, tasks)
    receipts = [read(p) for p in sorted((COLLECT/"accepted").glob("*.json"))]
    assert_unique(receipts)
    done = {p["task"]["id"] for p in receipts}
    if len(done) != len(receipts):
        raise ValueError("Duplicate accepted group")
    save_progress(receipts)
    tries = Counter()
    waiting = []
    logs = sorted((COLLECT / "batches").glob("batch_*.json"))
    for p in logs:
        batch = read(p)
        for task in batch["tasks"]:
            tries[task["id"]] += 1
        if not batch.get("finished"):
            waiting.append((p,batch))
    next_id = len(logs)
    with ProcessPoolExecutor(max_workers=12, mp_context=get_context("spawn")) as pool:
        while len(done) < len(tasks):
            check_pause()
            if waiting:
                path,batch = waiting.pop(0)
            else:
                limit = 3 if next_id == 0 else 12
                remaining = sorted((t for t in tasks if t["id"] not in done), key=lambda t: (tries[t["id"]],t["id"]))
                chosen = remaining[:limit]
                if any(tries[t["id"]] >= PHYSICAL_ATTEMPTS for t in chosen):
                    raise RuntimeError(f"At least one layout remains infeasible after {PHYSICAL_ATTEMPTS} attempts")
                futures = [pool.submit(build_group,t,tries[t["id"]]) for t in chosen]
                prepared = []
                occupied = {r["layout_signature"] for p in receipts for r in p["episodes"]}
                for t,f in zip(chosen,futures,strict=True):
                    outcome = f.result()
                    p = emit(t,tries[t["id"]],outcome) if outcome else None
                    if p and not occupied.intersection(r["layout_signature"] for r in p["rows"]):
                        occupied.update(r["layout_signature"] for r in p["rows"])
                        prepared.append(p)
                    tries[t["id"]] += 1
                path = COLLECT / "batches" / f"batch_{next_id:04}.json"
                batch = {"id": next_id, "tasks": chosen, "prepared": prepared}
                save(path,batch)
                next_id += 1
            print(f"Batch {batch['id']} planned {len(batch['prepared'])}/{len(batch['tasks'])} groups", flush=True)
            check_pause()
            additions, failures = run_batch(batch, short=batch["id"] == 0)
            additions = [p for p in additions if p["task"]["id"] not in done]
            assert_unique([*receipts,*additions])
            for p in additions:
                key = p["task"]["id"]
                save(COLLECT / "accepted" / f"{key:04}.json",p)
                done.add(key)
            receipts.extend(additions)
            save(path,{**batch,"finished":True,"accepted_groups":[p["task"]["id"] for p in additions],
                       "failures":failures})
            status = save_progress(receipts)
            print(f"Batch {batch['id']} complete, {status['accepted']}/{TARGET} new episodes", flush=True)
    check_pause()
    merge(receipts)

