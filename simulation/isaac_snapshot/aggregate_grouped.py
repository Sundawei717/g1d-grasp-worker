import argparse
import json
from collections import Counter
from pathlib import Path

from audit_grasp_receipt import audit


REQUIRED = {"A_target": 40, "B_obstacle": 35, "C_start": 25}
OUTCOMES = {"strict_success", "physical_fail", "preflight_fail", "offline_reject", "infrastructure_fail", "pending"}
CASE_FIELDS = ("case_id", "target_usd", "target_pose_xyz_xyzw", "home_q", "fixtures", "part", "seed",
               "group", "scene_family")


def audited_success(episode, row, original):
    result = json.loads((episode / "expert_diagnostic.json").read_text())
    if any(result["case"][key] != original[key] for key in CASE_FIELDS):
        raise ValueError(f"Physical case changed after planning: {episode}")
    state = audit(episode)
    if state["strict_success"] is not True or result["strict_score"] != row["strict_score"]:
        raise ValueError(f"Strict success does not reproduce: {episode}")
    pixels = row["prompt"]["visible_fixture_pixels"]
    if len(pixels) != len(original["fixtures"]) or any(value < 80 for value in pixels.values()):
        raise ValueError(f"Left-eye visibility does not reproduce the fixture binding: {episode}")
    return state


def tally(records):
    counts = {name: Counter(row["status"] for row in records if row["group"] == name) for name in REQUIRED}
    successes = sum(value["strict_success"] for value in counts.values())
    by_group = {group: {**({key: counts[group][key] for key in sorted(OUTCOMES)}),
                        "configured": count, "strict_rate": counts[group]["strict_success"] / count}
                for group, count in REQUIRED.items()}
    if len(records) != 100 or any(sum(value.values()) != REQUIRED[group] for group, value in counts.items()):
        raise ValueError("Grouped receipt does not have 40/35/25 cases")
    if any(row["status"] not in OUTCOMES for row in records):
        raise ValueError("Unknown case outcome")
    return {"configured": 100, "strict_success": successes, "strict_rate": successes / 100,
            "groups": by_group, "passes_rate_gate": successes >= 90 and all(
                counts[group]["strict_success"] * 5 >= count * 4 for group, count in REQUIRED.items())}


def report(plan_dir, runs, prefix, *, partial=False):
    plan = json.loads((plan_dir / "result.json").read_text())
    expected = {case["case_id"]: case for case in plan["cases"]}
    if len(expected) != 100:
        raise ValueError("Missing or duplicate frozen cases")
    records = {}
    for shard in plan["shards"]:
        directory = runs / f"{prefix}_{shard['name']}"
        result = directory / "result.json"
        progress = directory / "batch_progress.json"
        if result.is_file():
            observed = json.loads(result.read_text())["cases"]
        elif progress.is_file():
            try:
                observed = json.loads(progress.read_text())["cases"]
            except json.JSONDecodeError:
                observed = []
        else:
            observed = []
        receipt = {row["case"]: row for row in observed}
        if len(receipt) != len(observed) or not receipt.keys() <= set(shard["case_ids"]):
            raise ValueError("Unexpected shard case identities")
        for case_id in shard["case_ids"]:
            if case_id in records:
                raise ValueError("Duplicate case across shards")
            original = expected[case_id]
            row = receipt.get(case_id)
            info = {"case": case_id, "group": original["group"], "scene_family": original["scene_family"]}
            if row is None:
                info["status"] = "pending" if partial else "infrastructure_fail"
                info["failure"] = "no physical case receipt"
            else:
                if row["group"] != original["group"] or row["scene_family"] != original["scene_family"]:
                    raise ValueError("Recorded group differs from the frozen definition")
                info.update(row)
                if info["status"] == "strict_success" and not partial:
                    info["independent_audit"] = audited_success(directory / case_id, row, original)
            info["shard"] = directory.name
            records[case_id] = info
    if records.keys() != expected.keys():
        raise ValueError("The shard inventory does not cover the frozen suite")
    ordered = [records[case_id] for case_id in expected]
    summary = tally(ordered)
    return {"phase": "grouped_expert_100", **summary, "cases": ordered, "frozen_plan": str(plan_dir.resolve()),
            "start_pose_screen": plan.get("start_pose_screen"),
            "coverage_note": "A uses target changes on paired two-object layouts with admitted upper-side regions; not a same-asset part-switch claim",
            "passed": not partial and summary["passes_rate_gate"], "fixture_scope": "fixed kinematic distractors/blockers at reset"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("runs", type=Path)
    parser.add_argument("prefix")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    artifact = report(args.plan, args.runs, args.prefix, partial=args.partial)
    if args.output:
        if not args.partial:
            from grouped_video import representative_runs

            artifact["representative_videos"] = representative_runs(
                artifact["cases"], args.runs, args.output.parent / "videos")
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(artifact, stream, indent=2)
    print(json.dumps({key: artifact[key] for key in ("phase", "configured", "strict_success", "groups", "passed")}, indent=2))
    return 0 if args.partial or artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
