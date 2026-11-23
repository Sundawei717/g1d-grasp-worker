import argparse
import concurrent.futures
import json
from collections import Counter
from pathlib import Path

from aggregate_grouped import tally
from prepare_grouped import TARGETS, generate_layouts, plan_case, sha256, verify_gates, write_json
from repair_grouped import retry_case
from replay_grouped_obstacle import PREVIOUS_DIRECTION, replace_direction


def verified_expert(binding):
    receipt = Path(binding["path"])
    if sha256(receipt) != binding["sha256"]:
        raise ValueError("The completed expert receipt changed")
    result = json.loads(receipt.read_text())
    complete = json.loads(receipt.parent.with_suffix(".complete").read_text())
    if (result.get("phase") != "grouped_expert_100" or result.get("passed") is not True
            or complete.get("status") != "COMPLETE" or complete.get("exit_code") != 0
            or not tally(result["cases"])["passes_rate_gate"]):
        raise ValueError("Dense collection requires the completed 100-case physical expert gate")
    frozen = json.loads((Path(result["frozen_plan"]) / "frozen_cases.json").read_text())
    if {case["case_id"] for case in frozen["cases"]} != {row["case"] for row in result["cases"]}:
        raise ValueError("The expert report does not cover its frozen cases")
    return frozen["cases"]


def verify_collection_inputs(binding, cases):
    gate = verified_expert(binding)
    for field in ("case_id", "seed", "scene_family"):
        if {row[field] for row in cases} & {row[field] for row in gate}:
            raise ValueError(f"Dense collection reuses expert development {field}")
    if any(case.get("split") != "preview" or not case.get("asset_family") or not case.get("config_sha256")
           for case in cases):
        raise ValueError("Each capacity preview must carry frozen family and configuration provenance")


def preview_cases(root, config, evidence, config_hash):
    seed = 228315
    drawn = generate_layouts(root, config, evidence, seed)
    cases = [*drawn[:8], *drawn[40:47], *drawn[75:80]]
    for index, case in enumerate(cases):
        case["case_id"] = f"{case['group'][0]}_{300 + index}"
        case["scene_family"] = f"preview_{case['scene_family']}"
        case["split"] = "preview"
        case["asset_family"] = "/".join(Path(case["target_usd"]).parts[-3:-1])
        case["config_sha256"] = config_hash
        if case["group"] == "B_obstacle":
            # Preserve the blocker geometry but use a different independent physical layout.
            dy = -0.010 - (index - 8) * 0.002
            case["target_pose_xyz_xyzw"][1] += dy
            case["fixtures"][0]["pose"][1] += dy
        if case["group"] == "C_start":
            case["home_q"][1] = -abs(case["home_q"][1])
    return seed, cases


def plan_preview(parameters):
    config, urdf, evidence, case, guide = parameters
    original = plan_case((config, urdf, evidence, case))
    if original.get("offline_failure") == "no_checked_grasp":
        original = retry_case((config, urdf, evidence, original, guide if case["group"] == "B_obstacle" else None))
    if (case["group"] == "B_obstacle" and original["feasible"]
            and original["feasible"][0]["candidate"] == PREVIOUS_DIRECTION):
        return replace_direction(config, urdf, evidence, original)
    return original


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "urdf", "expert", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = json.loads(args.expert.read_text())
    expert = {"path": str(args.expert.resolve()), "sha256": sha256(args.expert)}
    verified_expert(expert)
    prerequisites = verify_gates(root)
    config_path = root / "env_resolved.json"
    config = json.loads(config_path.read_text())
    evidence = json.loads((root / "runs/lab_school_03/result.json").read_text())
    seed, cases = preview_cases(root, config, evidence, sha256(config_path))
    verify_collection_inputs(expert, cases)
    args.output.mkdir()
    write_json(args.output / "frozen_cases.json", {"seed": seed, "split": "preview", "cases": cases,
                                                  "collection_gate": expert})
    reference = Path(result["frozen_plan"]) / "B_040.json"
    guide = json.loads(reference.read_text())["feasible"][0]
    receipts = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(plan_preview, (config, args.urdf, evidence, case, guide)): case for case in cases}
        for future in concurrent.futures.as_completed(futures):
            case = futures[future]
            planned = future.result()
            key = case["case_id"]
            write_json(args.output / f"{key}.json", planned)
            receipts[key] = {**case, "plan": f"{key}.json", "offline_feasible": bool(planned["feasible"]),
                             "offline_failure": planned.get("offline_failure")}
            print(f"{len(receipts)}/20 {key}: {planned.get('offline_failure') or 'planned'}", flush=True)
    ordered = [receipts[case["case_id"]] for case in cases]
    for role in TARGETS:
        shard = [case for case in ordered if case["asset_role"] == role]
        fixtures = {item["name"]: item["usd"] for case in shard for item in case["fixtures"]}
        write_json(args.output / f"shard_{role}.json", {"phase": "collection_preview", "asset": shard[0]["target_usd"],
                   "fixture_assets": fixtures, "cases": shard, "gates": prerequisites, "collection_gate": expert})
    feasible = Counter(row["group"] for row in ordered if row["offline_feasible"])
    print(f"Capacity preview feasibility: {dict(feasible)}", flush=True)
    write_json(args.output / "result.json", {"cases": ordered, "split": "preview", "collection_gate": expert})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
