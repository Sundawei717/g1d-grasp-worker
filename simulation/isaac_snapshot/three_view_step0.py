import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from shutil import copyfile
import subprocess

import h5py
import numpy as np

from episode_store import jaw_width_m
from three_view_detector import sha256, write_json
from three_view_eval import read, verify_policy
from unified_grasp import CONTROL_HZ, EPISODE_STEPS, LiftGateConfig, ticks


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OLD = RUNS / "three_view_pipeline_03"
ORIGINAL = RUNS / "three_view_time25_01"
ROLES = {"carton": "box", "can": "can"}


def template(role, variant):
    path = ORIGINAL.with_name(f"{ORIGINAL.name}_baseline_{variant}") / "plans" / f"shard_{role}_1.json"
    return read(path)


def check_expert(row, plan):
    h5_path = Path(row["path"])
    if sha256(h5_path) != row["sha256"]:
        raise ValueError(f"Changed expert recording {h5_path}")
    with h5py.File(h5_path, "r") as h5:
        case = json.loads(h5["meta/case_json"].asstr()[()])
        if case != plan["case"] or not json.loads(h5["meta/strict_score_json"].asstr()[()])["success"]:
            raise ValueError(f"Expert recording is unverified or belongs to a different scene: {h5_path}")
        phases = h5["phase"].asstr()[...].tolist()
        recorded_lift = phases.index("lift")
        widths = h5["obs/gripper_width"][...].reshape(-1)
        command = h5["action/native_substep_command"][..., -1, 7:]
        return {"case": row["case"], "recorded_lift_step": recorded_lift,
                "last_width_deltas_m": np.abs(np.diff(widths[recorded_lift - 3:recorded_lift + 1])).tolist(),
                "last_width_targets_m": [jaw_width_m(fingers) for fingers in command[recorded_lift - 5:recorded_lift]],
                "last_pad_contacts": h5["gt/contact_flags"][recorded_lift - 2:recorded_lift + 1, 1:3].tolist()}


def prepare(work):
    config = read(ROOT / "configs/step0_v3.json")
    gate = LiftGateConfig(**config["gate"])
    data = {row["case"]: row for row in read(OLD / "split.json")["episodes"]}
    frozen = ORIGINAL / "validation_38"
    original = read(frozen / "frozen.json")
    policy_cases = {row["case_id"]: row for row in original["cases"] if row["case_id"] in config["policy_cases"]}
    if policy_cases.keys() != set(config["policy_cases"]):
        raise ValueError("Policy development scenes not found in the frozen initial placements")
    fixtures = {}
    expert = {role: [] for role in ROLES}
    offline = []
    for key in config["expert_cases"]:
        row = data[key]
        role_run = Path(row["path"]).parents[1].name.rsplit("_", 1)[0]
        src = RUNS / role_run / "plans" / f"{key}.json"
        plan = read(src)
        offline.append(check_expert(row, plan))
        case = plan["case"]
        fixtures.update({fixture["name"]: fixture["usd"] for fixture in case["fixtures"]})
        expert[case["asset_role"]].append((src, {**case, "plan": src.name, "offline_feasible": True,
                                               "replay_h5": {"path": row["path"], "sha256": row["sha256"]}}))
    work.mkdir()
    definitions = work / "plans"
    definitions.mkdir()
    jobs = []

    def build(name, role, stages, variant="a", repetition=None):
        if not stages:
            return
        shard = template(role, variant)
        if repetition is None:
            shard.pop("policy", None)
            shard.pop("detector", None)
            shard.update(phase="three_view_unified_replay", initial_head_prompts={},
                         fixture_assets={f["name"]: f["usd"] for _, row in stages for f in row["fixtures"]})
        else:
            cfg = verify_policy(shard["policy"])
            if cfg["variant"] != variant.upper():
                raise ValueError("Checkpoint is bound to the wrong policy variant")
            shard["initial_head_prompts"] = {row["case_id"]: original["head_prompts"][row["case_id"]]
                                             for _, row in stages}
        for path, row in stages:
            destination = definitions / row["plan"]
            if not destination.exists():
                copyfile(path, destination)
        path = definitions / f"{name}.json"
        write_json(path, {**shard, "cases": [row for _, row in stages], "total_steps": EPISODE_STEPS,
                          "unified_lift_gate": config["gate"]})
        jobs.append({"name": f"{work.name}_{name}", "role": role, "plan": str(path),
                     "phase": "expert" if repetition is None else "policy",
                     "variant": variant.upper(), "repeat": repetition})

    for role, stages in expert.items():
        build(f"expert_{role}", role, stages)
    for repetition in range(1, config["policy_repeats"] + 1):
        for role in ROLES:
            for variant in "ab":
                stages = [(frozen / row["plan"], row) for row in policy_cases.values() if row["asset_role"] == role]
                build(f"policy_{variant}_r{repetition}_{role}", role, stages, variant, repetition)
    write_json(work / "definition.json", {"config": config, "jobs": jobs, "archived_expert_closure": offline,
                                         "jaw_stable_steps": ticks(gate.jaw_stable_duration_s)})


def run_jobs(work, phase):
    planned = [job for job in read(work / "definition.json")["jobs"] if job["phase"] == phase]
    pending = iter(planned)

    def worker(gpu):
        for job in pending:
            output = RUNS / job["name"]
            if output.with_suffix(".complete").exists() and read(output.with_suffix(".complete"))["exit_code"] == 0:
                continue
            config = ROOT / ("env_resolved.json" if job["role"] == "carton" else "runs/candidate_can_config.json")
            env = {**os.environ, "G1D_WORKER_CONFIG": str(config), "G1D_GROUPED_WALL_SECONDS": "5400"}
            command = ["bash", str(ROOT / "run_lab_school.sh"), str(output), "--grouped-manifest", job["plan"],
                       "--grasp-after", str(RUNS / "lab_school_03/result.json"), "--gpu", str(gpu)]
            with output.with_suffix(".log").open("x") as log:
                with subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT) as process:
                    output.with_suffix(".pid").write_text(str(process.pid))
                    print(f"{job['name']} GPU {gpu} PID {process.pid}", flush=True)
                    if process.wait():
                        raise RuntimeError(f"Simulator failed in {job['name']}")
    with ThreadPoolExecutor(max_workers=2) as workers:
        tasks = [workers.submit(worker, gpu) for gpu in (1, 2)]
        for task in tasks:
            task.result()


def collect(work, phase):
    rows = []
    for job in read(work / "definition.json")["jobs"]:
        if job["phase"] != phase:
            continue
        base = RUNS / job["name"]
        receipt = read(base / "result.json")
        manifest = read(job["plan"])
        expected = {case["case_id"]: case for case in manifest["cases"]}
        if {row["case"] for row in receipt["cases"]} != expected.keys():
            raise ValueError(f"Incomplete evaluation shard {base}")
        for case in receipt["cases"]:
            path = base / case["case"] / "expert_diagnostic.json"
            diagnostic = read(path)
            result = diagnostic.get("policy_rollout", {})
            status = result.get("status")
            if status not in {"success", "lift_failed", "timeout"} or not 0 < result["steps"] <= EPISODE_STEPS:
                raise ValueError(f"Invalid control outcome in {path}: {diagnostic.get('failure')}")
            if status == "timeout" and result["steps"] != EPISODE_STEPS:
                raise ValueError(f"Incomplete timeout horizon in {path}")
            if phase == "policy":
                prompt = manifest["initial_head_prompts"][case["case"]]
                if any(diagnostic["task"][key] != value for key, value in prompt.items()):
                    raise ValueError(f"Frozen prompt changed in {path}")
            attempt = result["trials"][0] if result["trials"] else {}
            gates = ("sustained_close", "arm_stable", "jaw_stable_loaded", "ready")
            counts = {key: sum(g[key] for g in result["gate"]) for key in gates}
            if status == "timeout":
                category = "insufficient_trial_budget" if counts["ready"] else "gates_never_jointly_ready"
                reasons = [category, *(key + "_never_met" for key in gates[:-1] if not counts[key])]
            else:
                reasons = attempt["failure_reasons"]
                category = attempt.get("failure_category")
            decisions = result["decisions"]
            latency = {}
            for key in ("policy_forward_ms", "right_detector_ms", "to_command_ms"):
                values = [d[key] for d in decisions[1:] if d.get(key) is not None]
                if values:
                    latency[key] = dict(zip(("p50", "p95"), np.percentile(values, (50, 95)).tolist()))
            row = {"case": case["case"], "group": expected[case["case"]]["group"], "seed": expected[case["case"]]["seed"],
                   "status": status, "steps": result["steps"], "part_hit": result["part_hit"],
                   "failure_category": category, "reasons": reasons, "trigger": attempt.get("trigger"),
                   "latency_ms": latency, "gate_passed_steps": counts,
                   "failed_gate_at_deadline": [key for key in ("sustained_close", "arm_stable", "jaw_stable_loaded")
                                               if not result["gate"][-1][key]] if status == "timeout" else [],
                   "diagnostics": diagnostic["diagnostics"], "diagnostic_path": str(path)}
            if phase == "expert":
                row["alignment"] = diagnostic["replay_alignment"]
            else:
                row.update(variant=job["variant"], repeat=job["repeat"],
                           settled_target_xyz_xyzw=diagnostic["target_initial_xyz_xyzw"])
            rows.append(row)
    return rows


def expert_report(work):
    cases = collect(work, "expert")
    for row in cases:
        a = row["alignment"]
        delay = a["trigger_delta_s"]
        row["replay_pass"] = (row["status"] == "success" and a["max_timestamp_error_s"] <= 2e-7
                              and a["max_native_setpoint_error"] <= 1e-6 and a["max_arm_error_rad"] <= 0.005
                              and a["max_jaw_error_m"] <= 0.002 and delay is not None and 0 <= delay <= 0.15)
    return {"configured": len(cases), "accepted": sum(r["replay_pass"] for r in cases), "cases": cases}


def report(work):
    definition = read(work / "definition.json")
    cases = collect(work, "policy")
    summaries, differences, diagnostics = {}, {}, {}
    for variant in ("A", "B"):
        records = [row for row in cases if row["variant"] == variant]
        summaries[variant] = {}
        for rep in (1, 2):
            counts = Counter(row["status"] for row in records if row["repeat"] == rep)
            summaries[variant][str(rep)] = {key: counts[key] for key in ("success", "lift_failed", "timeout")}
        by_key = {(row["case"], row["repeat"]): row for row in records}
        differences[variant] = []
        for key in definition["config"]["policy_cases"]:
            first, second = by_key[key, 1], by_key[key, 2]
            differences[variant].append({"case": key, "status": [first["status"], second["status"]],
                                         "step_delta": second["steps"] - first["steps"],
                                         "seed_equal": first["seed"] == second["seed"],
                                         "initial_position_delta_m": float(np.linalg.norm(
                                             np.array(first["settled_target_xyz_xyzw"][:3])
                                             - second["settled_target_xyz_xyzw"][:3]))})
        decisions = [decision for row in records
                     for decision in read(row["diagnostic_path"])["policy_rollout"]["decisions"][1:]]
        latency = {}
        for name in ("policy_forward_ms", "right_detector_ms", "to_command_ms"):
            values = [entry[name] for entry in decisions if entry.get(name) is not None]
            if values:
                latency[name] = {"samples": len(values),
                                 **dict(zip(("p50", "p95"), np.percentile(values, (50, 95)).tolist()))}
        diagnostics[variant] = {
            "latency_ms": latency,
            "part_hits": sum(row["part_hit"] is True for row in records),
            "trial_count": sum(row["trigger"] is not None for row in records),
            "arm_link_collision_onsets": {
                phase: sum(row["diagnostics"]["collision"].get(phase, {}).get("arm_link_collision_onsets", 0)
                           for row in records) for phase in ("approach", "close")},
        }
    failure = Counter(row["failure_category"] for row in cases if row["status"] != "success")
    write_json(work / "result.json", {"phase": "three_view_step0", "passed": True,
                                      "control_hz": CONTROL_HZ, "steps": EPISODE_STEPS,
                                      "gate": definition["config"]["gate"], "expert": expert_report(work),
                                      "policy": summaries, "policy_failure_categories": failure,
                                      "policy_diagnostics": diagnostics,
                                      "policy_repeat_comparison": differences, "cases": cases})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "expert", "policy", "report"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    work = args.output.resolve()
    if work.parent != RUNS or not work.name.startswith("three_view_step0_"):
        raise SystemExit("Use a new three_view_step0 run under runs/")
    if args.stage == "prepare":
        prepare(work)
    elif args.stage == "expert":
        run_jobs(work, "expert")
        write_json(work / "expert_result.json", expert_report(work))
    elif args.stage == "policy":
        run_jobs(work, "policy")
    else:
        report(work)
