"""Evaluate the repaired gripper on the fixed A40 development suite; no training."""
from collections import Counter, defaultdict
from dataclasses import asdict
import argparse
import fcntl
import os
from pathlib import Path
from threading import Event, Thread
import traceback

import numpy as np

import step2_dev
from background_status import record_complete
from step2_eval import compare, log_case
from step3b_common import A40, DEV, ROOT, RUNS, WORK, Resources, now, read, save
from step3b_eval import binding
from three_view_detector import sha256
from unified_grasp import CONTROL_HZ, EPISODE_STEPS, HOLD_STEPS, LIFT_HEIGHT_M, LiftGateConfig
from v3_prompt_repair import summarize


TAG = "gripperfix_a40_dev28_20261124"
BASE = RUNS / f"{WORK.name}_{TAG}"
ANCHOR = RUNS / "v3_step2_followup_01/a40_last_dev.json"
SOURCES = ("unified_grasp.py", "lab_school_probe.py", "step2_dev.py", "three_view_rollout.py",
           "chunk_model.py", "part_surface.py", "eval_gripper_repair.py")
TERMINALS = {"success", "timeout", "lift_failed"}
LABELS = {"all": "全部", "mug": "杯子", "basket": "篮子", "carton": "长盒",
          "mug_body": "杯身", "mug_handle": "杯柄", "basket_handle": "篮提手",
          "basket_rim": "篮沿", "obstacle": "有障碍", "non_obstacle": "无障碍",
          "positive_end": "长盒正端", "negative_end": "长盒负端"}


def joint(row):
    return row["status"] == "success" and row["part_hit"] is True


def prepare():
    BASE.mkdir(exist_ok=True)
    (BASE / "plans").mkdir(exist_ok=True)
    development, anchor = read(DEV), read(ANCHOR)
    specs = development["cases"]
    old = {r["case"]: r for r in anchor["episodes"]}
    assert development["replay_verified"] and len(specs) == len(old) == 28
    assert anchor["m"] == .01 and anchor["success_and_hit"] == 16
    assert anchor["strata"]["all"]["success"] == 19
    assert CONTROL_HZ == 20 and EPISODE_STEPS == 300 and LIFT_HEIGHT_M == .08 and HOLD_STEPS == 11
    assert set(old) == {r["case_id"] for r in specs}
    policy = binding(A40, "a40_last", "last", receipt_dir=BASE)
    plan_hashes, prepared = {}, []
    for row in specs:
        key = row["case_id"]
        assert row["seed"] == old[key]["seed"] and row["head_prompt"] == old[key]["head_prompt"]
        plan = read(row["plan"])
        assert plan["case"]["seed"] == row["seed"] and plan["feasible"]
        assert plan["case"].get("fixture_physics", "kinematic") == "kinematic"
        plan_hashes[key] = sha256(row["plan"])
        plan["case"]["te_decay"] = .01
        path = BASE / "plans" / f"{key}.json"
        save(path, plan)
        prepared.append({**row, "plan": str(path), "te_decay": .01})
    contract = {"checkpoint": policy, "development_sha256": sha256(DEV),
                "baseline_sha256": sha256(ANCHOR), "plan_sha256": plan_hashes,
                "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
                "gate_config": asdict(LiftGateConfig(gripper_mode="adaptive")),
                "fixture_physics": "kinematic", "control_hz": 20, "max_steps": 300,
                "lift_height_m": .08, "hold_s": .55, "te_exec": 1, "te_m": .01,
                "count": 28, "baseline_pickups": 19, "baseline_correct_pickups": 16,
                "prior_three_case_probe_reused": False}
    prior = read(BASE / "contract.json", {})
    if prior and prior != contract:
        raise ValueError("Pinned inputs changed; use a new run rather than mix protocols")
    save(BASE / "contract.json", contract)
    return prepared, policy, old, contract


def scan():
    found = {}
    for folder in sorted(RUNS.glob(f"{BASE.name}_[0-9][0-9]*")):
        if not folder.is_dir():
            continue
        receipt = folder / "result.json"
        if not receipt.exists():
            receipt = folder / "batch_progress.json"
        try:
            data = read(receipt)
        except (OSError, ValueError):
            continue
        if data.get("gripper_mode") != "adaptive" or data.get("fixture_physics", "kinematic") != "kinematic":
            continue
        for row in data.get("cases", []):
            p = row.get("policy_result", {})
            if p.get("status") in TERMINALS and (folder / row["case"] / "expert_diagnostic.json").exists():
                found.setdefault(row["case"], {"case": row["case"], "status": p["status"],
                                               "steps": p.get("steps"), "part_hit": p.get("part_hit"),
                                               "folder": str(folder)})
    return found


def progress(state="RUNNING", **extra):
    rows = list(scan().values())
    value = {"state": state, "updated_at": now(), "pid": os.getpid(),
             "completed": len(rows), "total": 28, "terminations": dict(Counter(r["status"] for r in rows)),
             "episodes": rows, **extra}
    save(BASE / "progress.json", value)
    return value


def monitor(stop):
    previous = -1
    while not stop.is_set():
        try:
            data = progress()
            if data["completed"] != previous:
                print("PROGRESS", data["completed"], "/28", data["terminations"], flush=True)
                previous = data["completed"]
        except Exception as exc:
            print("Progress monitor retry:", repr(exc), flush=True)
        stop.wait(15)


def report(rows, old, contract):
    new = {r["case"]: r for r in rows}
    comparison = compare({"a": old, "b": new}, "dev")
    stratified = {}
    groups = defaultdict(list)
    for row in rows:
        for key in ("all", row["bucket"], row["part"], "obstacle" if row["obstacle"] else "non_obstacle"):
            groups[key].append(row)
    for key, after in groups.items():
        before = [old[r["case"]] for r in after]
        stratified[key] = {"before": summarize(before), "after": summarize(after),
                           "before_correct": sum(map(joint, before)), "after_correct": sum(map(joint, after))}
    paired = Counter(r["outcome"] for r in comparison["paired"])
    paired_joint = Counter("both" if joint(old[r["case"]]) and joint(r) else
                           "old_only" if joint(old[r["case"]]) else "new_only" if joint(r) else "neither"
                           for r in rows)
    result = {"state": "COMPLETE", "contract": contract, "completed_at": now(),
              "episodes": rows, "strata": stratified, "paired_pickup": dict(paired),
              "paired_correct_pickup": dict(paired_joint), "paired_detail": comparison["paired"]}
    save(BASE / "result.json", result)
    a, b = stratified["all"]["before"], stratified["all"]["after"]
    lines = ["# A40 夹爪修复后 28 开发场景复测", "",
             "同一个 A40 last.pt、同一份开发集、种子、首帧提示与原始障碍设置。无重训。",
             "仅切换夹爪预紧控制与 LiftGate 为 adaptive；使用同步 20 Hz、15 秒/300 步、8 cm 抬升、0.55 秒悬停、TE exec=1/m=0.01。",
             "本轮全部 28 场重新执行，不复用此前三个场景的试跑结果。历史障碍维持固定，避免混入动态障碍变量。", "",
             f"抓起 {a['success']}/28 → {b['success']}/28；抓起且部位正确 {stratified['all']['before_correct']}/28 → {stratified['all']['after_correct']}/28。",
             f"旧失败新成功 {paired['b_wins']} 场；旧成功新失败 {paired['a_wins']} 场；同成 {paired['both_success']}；同败 {paired['both_fail']}。", "",
             "| 分层 | 场数 | 抓起：旧→新 | 抓起且部位正确：旧→新 | lift_failed：旧→新 | timeout：旧→新 |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, s in stratified.items():
        x, y = s["before"], s["after"]
        lines.append(f"| {LABELS.get(key,key)} | {x['count']} | {x['success']}→{y['success']} | {s['before_correct']}→{s['after_correct']} | {x['lift_failed']}→{y['lift_failed']} | {x['timeout']}→{y['timeout']} |")
    lines += ["", "## 逐场结果", "", "| 场景 | 物体/部位 | 障碍 | 旧结果 | 新结果 | 新 Part Hit | 新步数 | 新失败原因 |",
              "|---|---|---|---|---|---|---:|---|"]
    for r in rows:
        lines.append(f"| {r['case']} | {LABELS.get(r['part'],r['part'])} | {'有' if r['obstacle'] else '无'} | {old[r['case']]['status']} | {r['status']} | {r['part_hit']} | {r['steps']} | {r['failure'] or '—'} |")
    lines += ["", "## 诊断与判读", "", f"旧失败分布 `{dict(a['failure_types'])}`。",
              f"新失败分布 `{dict(b['failure_types'])}`。",
              "`lost_grip_during_lift_or_hold` 包含未稳定离桌及中途脱手，是否真正抬起后掉落需结合物理轨迹核对。",
              f"接近期目标位移 ≥10 mm：{a['disturbance']['target_translation_ge_10mm']}→{b['disturbance']['target_translation_ge_10mm']}；倾斜 ≥15°：{a['disturbance']['target_tilt_ge_15deg']}→{b['disturbance']['target_tilt_ge_15deg']}。",
              "本次为一次成对复测，物理仿真存在小幅复跑波动，不能把每个终态翻转均归为代码修复的因果效果。",
              "原始诊断路径在 result.json 的 episodes[].path；逐步物理轨迹位于相同目录下 physics_trace.jsonl。",
              f"权重 SHA-256 `{contract['checkpoint']['sha256']}`。",
              f"开发集 SHA-256 `{contract['development_sha256']}`。", ""]
    (BASE / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def run():
    specs, policy, old, contract = prepare()
    marker = BASE.with_suffix(".complete")
    if marker.exists() and read(marker).get("outcome") == "PASS":
        print("Already COMPLETE PASS:", BASE / "report.md", flush=True)
        return
    stop = Event()
    watcher = Thread(target=monitor, args=(stop,), daemon=True)
    watcher.start()
    try:
        step2_dev.WORK = WORK
        results = step2_dev.simulate(TAG, specs, "three_view_closed_loop", policy=policy,
                                    prompts={r["case_id"]: r["head_prompt"] for r in specs},
                                    allocator=Resources(), resume_completed=True, max_shard_cases=6, retries=8,
                                    fixture_physics="kinematic", gripper_mode="adaptive")
        rows = []
        for spec in specs:
            value = results[spec["case_id"]]
            raw, roll = value[1], value[1]["policy_rollout"]
            assert roll["gate_config"] == contract["gate_config"]
            assert raw["diagnostics"]["policy_execution"] == {"actions_per_decision": 1, "temporal_decay": .01}
            assert all(d["steps_applied"] == 1 and d["temporal_ensemble_chunks"] == min(8, i+1)
                       for i, d in enumerate(roll["decisions"]))
            row = log_case(spec, value, "a")
            row.update(pair_group_id=spec["pair_group_id"], scene_id=spec["scene_id"],
                       wall_s=roll["wall_s"], trials=roll["trials"],
                       stable_max=max(g["jaw_stable_steps"] for g in roll["gate"]))
            rows.append(row)
            print("RESULT", row["case"], row["status"], "Part Hit", row["part_hit"], row["failure"], flush=True)
        assert {name: sha256(ROOT/name) for name in SOURCES} == contract["source_sha256"]
        assert sha256(DEV) == contract["development_sha256"] and sha256(ANCHOR) == contract["baseline_sha256"]
        result = report(rows, old, contract)
    finally:
        stop.set()
        watcher.join()
    progress("COMPLETE PASS", report=str(BASE / "report.md"),
             success=result["strata"]["all"]["after"]["success"],
             success_and_hit=result["strata"]["all"]["after_correct"])
    record_complete(BASE, 0)
    print("COMPLETE PASS", BASE / "report.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        import json
        print(json.dumps(read(BASE / "progress.json", {}), ensure_ascii=False, indent=2))
    elif args.preflight:
        _, _, _, c = prepare()
        print("PREFLIGHT PASS", c["count"], c["checkpoint"]["sha256"], c["gate_config"])
    else:
        BASE.mkdir(exist_ok=True)
        with (BASE / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run()
            except Exception as error:
                traceback.print_exc()
                progress("FAILED", error=repr(error))
                # Retain completed per-case receipts for an explicit resume.
                raise
