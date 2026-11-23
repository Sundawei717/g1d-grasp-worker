import argparse
import json
from collections import Counter
from pathlib import Path

from school_planner import sha256


EXPERIMENTS = {
    "gt_visible": "fit20_closed_loop_03",
    "gt_full_projection": "fit20_closed_loop_04",
    "hold_last_reliable": "fit20_closed_loop_05",
    "frozen_init": "fit20_closed_loop_06",
}


def summarize(runs):
    groups, per_case = {}, {}
    reference = None
    complete = True
    for mode, name in EXPERIMENTS.items():
        root = runs / name
        if not root.with_suffix(".complete").is_file() or not (root / "result.json").is_file():
            groups[mode] = {"run": name, "complete": False}
            complete = False
            continue
        data = json.loads((root / "result.json").read_text())
        cases = {row["case"]: row for row in data["cases"]}
        bindings = {}
        checksums = set()
        for role in ("carton", "can"):
            manifest = json.loads((root / "plans" / f"shard_{role}.json").read_text())
            if manifest.get("prompt_mode", "gt_visible") != mode:
                raise ValueError(f"Prompt mode differs from the comparison design: {name}")
            checksums.add(manifest["policy"]["sha256"])
            for row in manifest["cases"]:
                bindings[row["case_id"]] = sha256(root / "plans" / row["plan"])
        evidence = (checksums, bindings)
        if reference is None:
            reference = evidence
        if evidence != reference or cases.keys() != bindings.keys() or len(cases) != 20:
            raise ValueError(f"Checkpoint or frozen case mismatch: {name}")
        entered_close, checked_lift = 0, 0
        sources = Counter()
        cases_by_source = Counter()
        safety_reasons = Counter()
        latest_prompt_age = 0.0
        fresh_image_transitions = 0
        for key, row in cases.items():
            policy = row.get("policy_result") or {}
            entered_close += "close_started_step" in policy
            checked_lift += "close_completed_step" in policy
            entry = {
                "status": row["status"], "result": policy.get("status"), "failure": row.get("failure"),
                "executed_control_steps": policy.get("executed_control_steps"),
                "close_started_step": policy.get("close_started_step"),
                "close_completed_step": policy.get("close_completed_step"),
                "strict_score": row.get("strict_score")}
            per_case.setdefault(key, {"case": key, "group": row["group"], "modes": {}})["modes"][mode] = entry
            trace = runs / row["shard"] / key / "expert_diagnostic.json"
            if trace.is_file():
                raw = json.loads(trace.read_text())
                decisions = raw.get("policy_rollout", {}).get("decisions", [])
                if policy.get("status") == "safety_stop":
                    safety_reasons.update([raw.get("failure") or "unknown"])
                if mode != "gt_visible" and decisions:
                    used = Counter(item["source"] for item in decisions)
                    sources.update(used)
                    cases_by_source.update(used.keys())
                    entry["prompt_sources"] = dict(used)
                    entry["maximum_prompt_age_s"] = max(item["prompt_age_s"] for item in decisions)
                    latest_prompt_age = max(latest_prompt_age, entry["maximum_prompt_age_s"])
                    times = [item["image_time_s"] for item in decisions]
                    if any(item is None for item in times) or any(a >= b for a, b in zip(times, times[1:])):
                        raise ValueError(f"Policy did not receive advancing image timestamps: {name} {key}")
                    fresh_image_transitions += len(times) - 1
                    reliable = raw["initial_prompt"]
                    for item in decisions:
                        if mode == "frozen_init" or item["source"] == "last_reliable":
                            if (item["bbox_0"], item["affordance_0"]) != (
                                    reliable["bbox_0"], reliable["affordance_0"]):
                                raise ValueError(f"Expected identical held prompt: {name} {key}")
                        else:
                            reliable = item
        evaluated = bool(data["evaluation_completed"])
        complete &= evaluated
        groups[mode] = {"run": name, "configured": 20, "strict_success": data["strict_success"],
                        "entered_close_wait": entered_close, "entered_lift_verification": checked_lift,
                        "failure_categories": data["failure_categories"], "prompt_sources": dict(sources),
                        "cases_by_prompt_source": dict(cases_by_source), "safety_stop_reasons": dict(safety_reasons),
                        "fresh_image_transitions": fresh_image_transitions,
                        "maximum_prompt_age_s": latest_prompt_age if sources else None,
                        "latency_ms": data["decision_latency_ms"], "complete": evaluated}
    return {"phase": "prompt_mode_comparison", "passed": complete, "groups": groups,
            "cases": list(per_case.values()), "checkpoint_sha256": next(iter(reference[0])) if reference else None,
            "scope": "same checkpoint and 20 in-sample layouts, live RGB and feedback in all modes",
            "full_projection_contract": "use visible truth when available, otherwise project full render mesh and fixed local anchor",
            "holding_contract": "preserve freeze arm speed and cadence; cap the model age input at 0.2 s"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.runs)
    with (args.output / "result.json").open("x") as out:
        json.dump(result, out, indent=2)
    print(json.dumps(result["groups"], indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
