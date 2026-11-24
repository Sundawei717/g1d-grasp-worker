"""Evaluate the locked V3 A/B experiment on the training diagnostic and verified development cases."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from background_status import record_complete
from step2_dev import RUNS, TIER, WORK, groups, simulate, verify_isolation
from three_view_detector import sha256, write_json
from three_view_eval import read
from unified_grasp import EPISODE_STEPS

TERMINALS = {"success", "lift_failed", "timeout"}
GATES = ("sustained_close", "arm_stable", "jaw_stable_loaded")
TIMINGS = ("policy_forward_ms", "right_detector_ms", "to_command_ms")


def marker(job):
    return RUNS / f"{WORK.name}_{job}.complete"


def wait(jobs, timeout=4*3600):
    end = time.monotonic()+timeout
    pending = set(jobs)
    while pending and time.monotonic() < end:
        for job in sorted(pending):
            path = marker(job)
            if path.exists():
                result = read(path)
                if result["outcome"] != "PASS":
                    raise RuntimeError(f"Prerequisite failed: {path}")
                pending.remove(job)
        if pending:
            print("Waiting for", ", ".join(sorted(pending)), flush=True)
            time.sleep(30)
    if pending:
        raise RuntimeError(f"Prerequisites timed out: {sorted(pending)}")


def train_rows():
    manifest = read(TIER / "training_split.json")
    if Counter(r["split"] for r in manifest["episodes"]) != {"train": 359, "validation": 101}:
        raise ValueError("Unexpected training split")
    eligible = [r for r in manifest["episodes"] if r["split"] == "train" and r["pair_group_id"]
                and r["bucket"] in {"mug", "basket", "carton"}]
    # Freeze the selection using only layout metadata, before reading any policy scores.
    pairs = groups(eligible)
    strata = (("mug", "C_start"), ("mug", "B_obstacle"), ("basket", "C_start"),
              ("basket", "B_obstacle"), ("carton", "B_obstacle"))
    rows = []
    for bucket, group in strata:
        coll = sorted((pair for pair in pairs if (pair[0]["bucket"], pair[0]["group"]) == (bucket, group)),
                      key=lambda pair: pair[0]["pair_group_id"])
        for pair in coll:
            paths = [TIER / "plans" / r["case"] / "plan.json" for r in pair]
            if not all(path.is_file() and read(path)["feasible"] for path in paths):
                continue
            for r, path in zip(pair, paths, strict=True):
                case = read(path)["case"]
                if case["case_id"] != r["case"] or case["split"] != "train":
                    raise ValueError(f"Wrong training diagnostic plan: {path}")
                rows.append({**case, "bucket": bucket, "plan": str(path),
                             "scene_signature": r["scene_signature"], "initial_q": r["initial_q"],
                             "initial_target": r["initial_target"], "initial_width": r["initial_width"],
                             "head_prompt": {"bbox_0": r["bbox"], "affordance_0": r["point"]}})
            break
        else:
            raise ValueError(f"No training pair in {bucket} {group}")
    write_json(WORK / "diagnostic10.json", {"count": len(rows), "cases": rows})
    return rows


def bindings(scope, checkpoint_name="last.pt", step=2000):
    from v3_detector import cache_key, detector_config
    import torch

    settings, references, norms = {}, {}, {}
    for variant in "ab":
        folder = RUNS / f"{WORK.name}_train_{variant}"
        result = read(folder / "result.json")
        check = folder / checkpoint_name
        pt = torch.load(check, map_location="cpu", weights_only=True)
        config = result["config"]
        if (result["updates"] != 2000 or pt["step"] != step
                or pt["config"] != config or config["variant"] != variant.upper()
                or config["manifest_sha256"] != sha256(TIER / "training_split.json")
                or config["wrist_channels"] != (3 if variant == "a" else 4)
                or config["state_dim"] != 17 or not config["freeze_bn"]):
            raise ValueError(f"Unexpected training receipt for {variant}")
        settings[variant] = {k: v for k,v in config.items()
                             if k not in {"variant", "wrist_channels", "detector_config_sha256"}}
        norms[variant] = pt["normalization"]
        proof = read(folder / "supervision.json")
        if proof["phases"]["close"]["sampled_unique"] != 4733 or proof["phases"]["lift"]["loss_terms"] != 0:
            raise ValueError(f"Invalid supervision for {variant}")
        settings[variant]["sampler_sha256"] = proof["window_draws_sha256"]
        with_detector = detector_config() if variant == "b" else None
        if config["detector_config_sha256"] != (cache_key(with_detector) if with_detector else None):
            raise ValueError(f"Detector does not match training for {variant}")
        source = WORK / f"{scope}_{check.stem}_{variant}_binding.json"
        binding = {"path": str(check), "receipt": str(source), "sha256": sha256(check)}
        write_json(source, {**result, "checkpoint": binding["path"], "checkpoint_sha256": binding["sha256"]})
        references[variant] = binding, with_detector
    if settings["a"] != settings["b"] or norms["a"] != norms["b"]:
        raise ValueError("The experiment differs in shared training settings or normalization")
    return references


def log_case(spec, value, variant):
    raw, diag, path = value
    roll = diag.get("policy_rollout", {})
    status = roll.get("status")
    steps = roll.get("steps", 0)
    if (status not in TERMINALS or not 0 < steps <= EPISODE_STEPS
            or status == "timeout" and steps != EPISODE_STEPS):
        raise ValueError(f"Incomplete evaluator outcome {path}: {raw.get('failure')}")
    task = diag["task"]
    if any(task[k] != v for k, v in spec["head_prompt"].items()):
        raise ValueError(f"Changed fixed first-frame prompt: {path}")
    decisions = roll["decisions"]
    if (not decisions or any(a["image_time_s"] >= b["image_time_s"] for a,b in zip(decisions,decisions[1:]))
            or any((r.get("boxes") is not None) != (variant == "b") for r in decisions)):
        raise ValueError(f"Camera or detector contract was violated: {path}")
    if len(roll["trials"]) != (status != "timeout"):
        raise ValueError(f"Unexpected trial count for {status}: {path}")
    gate = {k: sum(step[k] for step in roll["gate"]) for k in (*GATES,"ready")}
    attempt = roll["trials"][0] if roll["trials"] else {}
    missing = [k for k in GATES if not gate[k]]
    if status == "timeout":
        cause = "insufficient_trial_budget" if gate["ready"] else ("+".join(missing) if missing else "gates_never_jointly_ready")
    else:
        cause = attempt.get("failure_category")
    diagnostics = diag["diagnostics"]
    d = {"case": spec["case_id"], "seed": spec["seed"], "bucket": spec["bucket"],
         "obstacle": bool(spec["fixtures"]), "part": spec["part"], "status": status, "steps": steps,
         "part_hit": roll["part_hit"], "failure": cause, "failure_reasons": attempt.get("failure_reasons"),
         "failure_detail": attempt.get("detail"), "gate_passed_steps": gate, "trigger": attempt.get("trigger"),
         "contact": attempt.get("part_contact"), "diagnostics": diagnostics,
         "settled_target": diag["target_initial_xyz_xyzw"], "head_prompt": spec["head_prompt"], "path": path,
         "decision_count": len(decisions), "wrist_missing": sum(
             bool(d.get("boxes") and d["boxes"][0]["missing"]) for d in decisions)}
    d["latencies"] = {key: [r[key] for r in decisions[1:] if r.get(key) is not None] for key in TIMINGS}
    return d


def stats(rows):
    n = len(rows)
    result = {"count": n, **{status: sum(r["status"] == status for r in rows) for status in sorted(TERMINALS)},
              "part_hit": sum(r["part_hit"] is True for r in rows),
              "part_assessed": sum(r["part_hit"] is not None for r in rows),
              "failure_types": Counter(r["failure"] for r in rows if r["failure"]),
              "collision": {}, "disturbance": {}, "latency_ms": {}}
    # Unified policy runs label all pre-trial commands approach, then lift/hold.
    # Older expert logs still have close; retain it without discarding trial contact.
    for phase in ("approach", "close", "lift", "hold"):
        phases = [r["diagnostics"]["collision"].get(phase, {}) for r in rows]
        result["collision"][phase] = {
            "episodes_with_contact": sum(p.get("contact_physics_steps", 0) > 0 for p in phases),
            "arm_link_onsets": sum(p.get("arm_link_collision_onsets", 0) for p in phases),
            "contact_physics_steps": sum(p.get("contact_physics_steps", 0) for p in phases)}
    targets = [r["diagnostics"]["target"] for r in rows]
    fixtures = [list(r["diagnostics"]["fixtures"].values()) for r in rows]
    result["disturbance"] = {
        "target_translation_ge_10mm": sum(t["max_pre_lift_translation_m"] >= .01 for t in targets),
        "target_tilt_ge_15deg": sum(t["pre_lift_toppled"] for t in targets),
        "fixture_translation_ge_10mm": sum(any(p["max_translation_m"] >= .01 for p in f) for f in fixtures),
        "fixture_tilt_ge_15deg": sum(any(p["toppled"] for p in f) for f in fixtures)}
    for metric in TIMINGS:
        ts = [t for r in rows for t in r["latencies"][metric]]
        if ts:
            result["latency_ms"][metric] = {"n": len(ts), **dict(zip(("p50", "p95"), np.percentile(ts, [50,95]).tolist()))}
    return result


def compare(records, scope):
    variants = "ab"
    keys = sorted(records["a"])
    if set(records["a"]) != set(records["b"]) or len(keys) != {"train": 10, "dev": 28, "best": 4}[scope]:
        raise ValueError("Incomplete paired A/B evaluation")
    paired = []
    for key in keys:
        a,b = (records[v][key] for v in variants)
        if a["head_prompt"] != b["head_prompt"] or a["seed"] != b["seed"]:
            raise ValueError(f"Unpaired policy initial condition: {key}")
        x,y = np.asarray(a["settled_target"]),np.asarray(b["settled_target"])
        mm = float(1000*np.linalg.norm(x[:3]-y[:3]))
        rotation = float(np.rad2deg(2*np.arccos(min(1, abs(np.dot(x[3:],y[3:]) / np.linalg.norm(x[3:]) / np.linalg.norm(y[3:]))))))
        if mm > 1 or rotation > 1:
            raise ValueError(f"Different settled target states in {key}: {mm} mm {rotation} deg")
        successes = a["status"]=="success",b["status"]=="success"
        outcome = {(True,True):"both_success", (True,False):"a_wins",
                   (False,True):"b_wins", (False,False):"both_fail"}[successes]
        paired.append({"case": key, "asset": a["bucket"], "part": a["part"], "obstacle": a["obstacle"],
                       "a": a["status"], "b": b["status"], "outcome": outcome,
                       "settled_translation_delta_mm": mm, "settled_rotation_delta_deg": rotation})
    strata = defaultdict(list)
    for pair in paired:
        labels = ["all", pair["asset"], pair["part"], "obstacle" if pair["obstacle"] else "non_obstacle"]
        for label in labels:
            strata[label].append(pair)
    return {"scope": scope, "passed": True, "paired": paired,
            "strata": {k: {**{v:stats([records[v][p["case"]] for p in rs]) for v in variants},
                           "paired": Counter(p["outcome"] for p in rs)} for k,rs in strata.items()},
            "episodes": {v:list(records[v].values()) for v in variants}}


def evaluate(scope):
    if scope == "train":
        rows = train_rows()
    else:
        wait(("dev",))
        frozen = read(WORK / "development.json")
        rows = frozen["cases"]
        if not frozen["replay_verified"] or len(rows) != 28:
            raise ValueError("Unexpected developer evaluation manifest")
        verify_isolation(rows)
    wait(("train_a", "train_b"))
    prompts = {row["case_id"]: row["head_prompt"] for row in rows}
    references = bindings(scope)
    preferred, exclude = (0, (1,2)) if scope == "train" else (1, (0,2))
    records = {}
    for v, (policy, detector) in references.items():
        simulated = simulate(f"{scope}_{v}", rows, "three_view_closed_loop", policy=policy, detector=detector,
                             prompts=prompts, preferred=preferred, exclude=exclude,
                             lanes=(1,2) if scope == 'dev' else None)
        records[v] = {r["case_id"]: log_case(r, simulated[r["case_id"]], v) for r in rows}
        write_json(WORK / f"{scope}_{v}.json", list(records[v].values()))
    report = compare(records, scope)
    write_json(WORK / f"{scope}_comparison.json", report)
    print(scope, report["strata"]["all"], flush=True)


def evaluate_best():
    wait(("eval_train",))
    names = {"C_7302", "C_7303", "C_8200", "C_8201"}
    rows = [r for r in read(WORK/"diagnostic10.json")["cases"] if r["case_id"] in names]
    if {r["case_id"] for r in rows} != names:
        raise ValueError("The supplementary four cases differ from the locked diagnostic")
    prompts = {r["case_id"]:r["head_prompt"] for r in rows}
    records = {}
    for v,(policy, detector) in bindings("best", checkpoint_name="best_val.pt", step=1500).items():
        result = simulate(f"best_{v}",rows,"three_view_closed_loop",policy=policy,detector=detector,
                          prompts=prompts,preferred=0,exclude=(1,2))
        records[v] = {r["case_id"]:log_case(r,result[r["case_id"]],v) for r in rows}
    data = compare(records,"best")
    data["checkpoint_step"] = 1500
    write_json(WORK/"best_comparison.json", data)
    print("best_val", data["strata"]["all"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("train", "dev", "best", "supervisor"))
    stage = ap.parse_args().stage
    path = marker("eval_"+stage) if stage != "supervisor" else RUNS / f"{WORK.name}.complete"
    try:
        if stage == "supervisor":
            wait(("train_a", "train_b", "dev", "eval_train", "eval_dev", "report"))
            write_json(WORK / "result.json", {"passed": True, "optimization": "last",
                       "train": read(WORK / "train_comparison.json")["strata"],
                       "dev": read(WORK / "dev_comparison.json")["strata"],
                       "dev_verified_count": read(WORK / "development.json")["count"],
                       "early_wrist_review": read(WORK / "early_wrist_review.json")})
        elif stage == "best":
            evaluate_best()
        else:
            evaluate(stage)
    except BaseException:
        record_complete(path.with_suffix(""),1)
        raise
    else:
        record_complete(path.with_suffix(""),0)
