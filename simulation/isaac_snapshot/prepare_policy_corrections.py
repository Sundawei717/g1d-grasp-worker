import argparse
import concurrent.futures
import json
import time
from collections import Counter
from pathlib import Path

from prepare_collection_preview import plan_preview
from prepare_grouped import TARGETS, write_json
from prepare_train_bootstrap import verify_train_inputs
from school_planner import sha256


def correction_case(row, runs, contents, new_index):
    source = runs / row["shard"] / row["case"] / "expert_diagnostic.json"
    data = json.loads(source.read_text())
    commands = data["commands"]
    steps = min(int((data.get("policy_rollout") or {}).get("close_started_step", len(commands))), len(commands))
    if steps < 8:
        return None
    start = min(steps * 2 // 5, 25)
    q = commands[start - 1]["arm_observed_rad"]
    previous = contents[row["case"]]
    if max(abs(left - right) for left, right in zip(q, previous["home_q"])) < 0.035:
        return None
    trace = source.parent / "physics_trace.jsonl"
    physics = [json.loads(line) for line in trace.read_text().splitlines()[:start * 30 + 1]]
    initial = physics[0]["target_z_m"]
    if (any(point["forbidden_contact"] or abs(point["target_z_m"] - initial) > 0.0005 for point in physics)
            or not physics[-1]["target_supported"]):
        return None
    return {**previous, "case_id": f"{previous['group'][0]}_{910 + new_index}", "home_q": q,
            "seed": 3368190 + new_index, "command_delay_substeps": 0,
            "correction_parent_case": row["case"], "correction_source": str(source),
            "correction_source_sha256": sha256(source), "correction_state_control_step": start}


def main():
    cli = argparse.ArgumentParser()
    for key in ("root", "urdf", "evaluation", "output"):
        cli.add_argument(key, type=Path)
    args = cli.parse_args()
    runs = args.root / "runs"
    marker = args.evaluation.with_suffix(".complete")
    deadline = time.monotonic() + 4800
    while not marker.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError("The paired closed-loop evaluation did not complete")
        time.sleep(3)
    evaluation = json.loads((args.evaluation / "result.json").read_text())
    if not evaluation["evaluation_completed"]:
        raise ValueError("Policy correction selection requires every completed physical rollout")
    initial = json.loads((runs / "train_bootstrap_01/plans/frozen_cases.json").read_text())
    originals = {row["case_id"]: row for row in initial["cases"]}
    failed = [row for row in evaluation["cases"] if row["status"] == "physical_fail"]
    cases, counts = [], Counter()
    order = {"safety_stop": 0, "tracking_lost": 1, "timeout": 2}
    failed.sort(key=lambda row: (order.get((row.get("policy_result") or {}).get("status"), 3), row["case"]))
    for row in failed:
        group = row["group"]
        if counts[group] >= 2:
            continue
        case = correction_case(row, runs, originals, len(cases))
        if case:
            cases.append(case)
            counts[group] += 1
    verify_train_inputs(initial["training_gates"], cases)
    gates = json.loads((runs / "train_bootstrap_01/plans/shard_carton.json").read_text())["gates"]
    output = args.output
    output.mkdir()
    write_json(output / "frozen_cases.json", {"cases": cases, "scope": "fixed earlier policy states in training scenes",
                                              "original_policy_receipt": str(args.evaluation / "result.json")})
    config = json.loads((args.root / "env_resolved.json").read_text())
    evidence = json.loads((runs / "lab_school_03/result.json").read_text())
    guide = json.loads((runs / "grouped_eval_03/plans/B_040.json").read_text())["feasible"][0]
    recorded = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(plan_preview, (config, args.urdf, evidence, case, guide)): case for case in cases}
        for future in concurrent.futures.as_completed(futures):
            case = futures[future]
            key = case["case_id"]
            result = future.result()
            write_json(output / f"{key}.json", result)
            recorded[key] = {**case, "plan": f"{key}.json", "offline_feasible": bool(result["feasible"]),
                             "offline_failure": result.get("offline_failure")}
            print(f"{key} from {case['correction_parent_case']} at step {case['correction_state_control_step']}: "
                  f"{result.get('offline_failure') or 'planned'}", flush=True)
    ordered = [recorded[case["case_id"]] for case in cases]
    for role in TARGETS:
        part = [row for row in ordered if row["asset_role"] == role]
        if part:
            fixtures = {slot["name"]: slot["usd"] for case in part for slot in case["fixtures"]}
            write_json(output / f"shard_{role}.json", {"phase": "collection_train", "asset": part[0]["target_usd"],
                      "cases": part, "fixture_assets": fixtures, "gates": gates, "training_gates": initial["training_gates"]})
    write_json(output / "result.json", {"phase": "policy_correction_plans", "cases": ordered})
    print(f"{sum(row['offline_feasible'] for row in ordered)}/{len(cases)} corrected starts have offline paths", flush=True)
    return 0


def review_capture(plans, output):
    import h5py
    import numpy as np

    fixed = json.loads((plans / "result.json").read_text())["cases"]
    receipts = {}
    for role in TARGETS:
        path = output.parent / f"{output.name}_{role}"
        for name in ("result.json", "batch_progress.json"):
            if (path / name).is_file():
                receipts.update({row["case"]: {**row, "shard": path.name} for row in
                                 json.loads((path / name).read_text())["cases"]})
                break
    episodes, cases = [], []
    for item in fixed:
        key = item["case_id"]
        row = receipts.get(key, {"case": key, "group": item["group"],
                                 "status": "infrastructure_fail" if item["offline_feasible"] else "offline_reject"})
        cases.append(row)
        if row["status"] != "strict_success" or not row.get("dataset_receipt"):
            continue
        episode = output.parent / row["shard"] / key / "episode.h5"
        with h5py.File(episode, "r") as h5:
            case = json.loads(h5["meta/case_json"].asstr()[()])
            for field in ("scene_family", "home_q", "seed", "correction_source_sha256", "correction_parent_case"):
                if case[field] != item[field]:
                    raise ValueError(f"{key} did not reproduce the planned correction state")
            anchors = h5["gt/part_anchor_local"][...]
            if not np.allclose(anchors, anchors[0], atol=1e-8, rtol=0):
                raise ValueError(f"{key} did not preserve its initial local hint")
        episodes.append({"case": key, "parent_case": item["correction_parent_case"],
                         "path": str(episode.resolve()), "sha256": sha256(episode), "group": item["group"],
                         "scene_family": item["scene_family"], "asset": item["asset_family"]})
    report = {"phase": "policy_correction_capture", "configured": len(fixed), "strict_success": len(episodes),
              "passed": bool(episodes), "episodes": episodes, "cases": cases}
    write_json(output / "result.json", report)
    print(f"Correction capture: {len(episodes)}/{len(fixed)} strict", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
