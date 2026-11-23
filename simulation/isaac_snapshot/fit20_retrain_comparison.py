import argparse
import json
from pathlib import Path

from school_planner import sha256


def summary(root):
    marker = json.loads(root.with_suffix(".complete").read_text())
    result = json.loads((root / "result.json").read_text())
    cases = {row["case"]: row for row in result["cases"]}
    checksums, plans = set(), {}
    for path in sorted((root / "plans").glob("shard_*.json")):
        manifest = json.loads(path.read_text())
        checksums.add(manifest["policy"]["sha256"])
        for case in manifest["cases"]:
            plans[case["case_id"]] = sha256(path.parent / case["plan"])
    if (marker["status"] != "COMPLETE" or not result["evaluation_completed"]
            or result["configured"] != 20 or len(cases) != 20 or cases.keys() != plans.keys()
            or len(checksums) != 1):
        raise ValueError(f"Incomplete paired evaluation: {root}")
    report = {
        "run": root.name, "strict_success": result["strict_success"],
        "entered_close_wait": sum("close_started_step" in (row.get("policy_result") or {}) for row in cases.values()),
        "entered_lift_verification": sum("close_completed_step" in (row.get("policy_result") or {}) for row in cases.values()),
        "failure_categories": result["failure_categories"],
    }
    return result["prompt_mode"], checksums.pop(), plans, cases, report


def compare(output, new, old):
    supervision = json.loads((output / "supervision" / "summary.json").read_text())
    close = supervision["phase_totals"]["close"]
    if (close["sampled_unique_actions"] != close["eligible_actions"] or not close["sampled_held_prompt_starts"]
            or supervision["episodes_with_no_close_loss"] or supervision["episodes_with_no_first_close_loss"]):
        raise ValueError("Revised training did not recover the held-prompt closing supervision")
    groups = {}
    for updated, baseline in zip(new, old):
        mode, new_sha, plans, updated_cases, after = summary(output.parent / updated)
        old_mode, old_sha, old_plans, old_cases, before = summary(output.parent / baseline)
        if (mode != old_mode or plans != old_plans or new_sha == old_sha
                or new_sha != supervision["checkpoint_sha256"]):
            raise ValueError(f"Invalid paired checkpoint comparison: {mode}")
        cases = [{
            "case": key, "group": current["group"],
            "old": (old_cases[key].get("policy_result") or {}).get("status", old_cases[key]["status"]),
            "new": (current.get("policy_result") or {}).get("status", current["status"]),
            "old_strict_success": old_cases[key]["status"] == "strict_success",
            "new_strict_success": current["status"] == "strict_success",
        } for key, current in updated_cases.items()]
        groups[mode] = {"old": before, "new": after, "cases": cases}
    result = {"phase": "fit20_occlusion_retrain", "passed": True, "checkpoint_sha256": supervision["checkpoint_sha256"],
              "trained_episodes": supervision["trajectory_count"], "updates": supervision["updates"],
              "close_supervision": close, "groups": groups}
    with (output / "result.json").open("x") as out:
        json.dump(result, out, indent=2)
    print(json.dumps({mode: {k: v for k, v in table.items() if k != "cases"} for mode, table in groups.items()},
                     indent=2), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("output", type=Path)
    cli.add_argument("new", nargs=2)
    cli.add_argument("old", nargs=2)
    args = cli.parse_args()
    compare(args.output, args.new, args.old)
