import argparse
import concurrent.futures
import json
from pathlib import Path

from aggregate_grouped import audited_success, tally
from grouped_video import representative_runs
from prepare_grouped import write_json
from school_planner import Planner, sha256


PREVIOUS_DIRECTION = "d0.012_tilt90_skew-15_rev180"
PREFERRED = ("d0.012_tilt90_skew-30_rev180", "d0.012_tilt90_skew-45_rev180",
             "d0.012_tilt90_skew-60_rev180", "d0.018_tilt90_skew-30_rev180",
             "d0.018_tilt90_skew-45_rev180", "d0.012_tilt75_skew-30_rev180")


def replace_direction(config, urdf, evidence, baseline):
    case = baseline["case"]
    prior = baseline["feasible"][0]
    if prior["candidate"] != PREVIOUS_DIRECTION:
        raise ValueError(f"{case['case_id']} did not use the implicated direction")
    config = {**config, "target_usd": case["target_usd"]}
    planner = Planner(config, urdf, evidence, case["target_pose_xyz_xyzw"], obstacles=case["fixtures"],
                      home=case["home_q"], expanded_search=True, guide=prior["pre_grasp_q"], part=case["part"])
    retry = planner.plan(case["seed"], stop_after=1, max_candidates=120,
                         preferred=PREFERRED, excluded=(PREVIOUS_DIRECTION,))
    details = {"previous_candidate": PREVIOUS_DIRECTION, "changed": bool(retry["feasible"]),
               "alternative_search_checks": planner.checks, "alternative_candidate_log": retry["candidate_log"]}
    return {**baseline, "feasible": retry["feasible"] or baseline["feasible"], "direction_recheck": details}


def prepare(root, urdf, prior, output):
    before = json.loads(prior.read_text())
    marker = json.loads(prior.parent.with_suffix(".complete").read_text())
    if marker["status"] != "COMPLETE" or before["passed"] or marker["exit_code"] == 0:
        raise ValueError("Expected a completed grouped evaluation with a failed acceptance gate")
    frozen = Path(before["frozen_plan"])
    plan = json.loads((frozen / "result.json").read_text())
    expected = {record["case_id"]: record for record in plan["cases"]}
    rows = {record["case"]: record for record in before["cases"]}
    if len(rows) != 100 or expected.keys() != rows.keys():
        raise ValueError("Prior evaluation does not cover the unchanged 100-case suite")
    selected = {}
    for case_id, case in expected.items():
        if case["group"] != "B_obstacle":
            continue
        dataset = json.loads((frozen / case["plan"]).read_text())
        row = rows[case_id]
        if dataset["feasible"][0]["candidate"] == PREVIOUS_DIRECTION:
            if row["status"] not in {"strict_success", "physical_fail"}:
                raise ValueError(f"No completed physical comparison for {case_id}")
            selected[case_id] = dataset
        if row["status"] != "strict_success" and (
                case_id not in selected or (row["strict_score"] or {}).get("reasons") != ["invalid_pre_lift_baseline"]):
            raise ValueError(f"Different causal failure in {case_id}")
    config = json.loads((root / "env_resolved.json").read_text())
    evidence = json.loads((root / "runs/lab_school_03/result.json").read_text())
    output.mkdir()
    write_json(output / "selection.json", {
        "original_result": str(prior.resolve()), "original_result_sha256": sha256(prior),
        "frozen_manifest_sha256": sha256(frozen / "frozen_cases.json"), "frozen_plan": str(frozen),
        "selected_ids": list(selected), "selection_rule": f"All B cases whose previous candidate was {PREVIOUS_DIRECTION}"})
    routes = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        work = {pool.submit(replace_direction, config, urdf, evidence, row): key for key, row in selected.items()}
        for future in concurrent.futures.as_completed(work):
            key = work[future]
            route = future.result()
            if route["case"] != selected[key]["case"]:
                raise ValueError(f"Frozen case changed for {key}")
            write_json(output / expected[key]["plan"], route)
            routes[key] = route["feasible"][0]["candidate"]
            print(f"{key} {rows[key]['status']} previous={PREVIOUS_DIRECTION} replay={routes[key]}", flush=True)
    gates = json.loads((frozen / "frozen_cases.json").read_text())["gates"]
    for index, shard in enumerate((list(selected)[::2], list(selected)[1::2])):
        cases = [expected[key] for key in shard]
        fixtures = {item["name"]: item["usd"] for case in cases for item in case["fixtures"]}
        write_json(output / f"shard_{index}.json", {"phase": "grouped_expert_100", "asset": cases[0]["target_usd"],
                   "fixture_assets": fixtures, "cases": cases, "gates": gates})
    print(f"Prepared {len(selected)} replays, with {sum(value != PREVIOUS_DIRECTION for value in routes.values())} new routes",
          flush=True)
    return 0


def summarize(output):
    runs = output.parent
    plan = output / "plans"
    selection = json.loads((plan / "selection.json").read_text())
    prior = Path(selection["original_result"])
    before = json.loads(prior.read_text())
    frozen = Path(selection["frozen_plan"])
    if sha256(prior) != selection["original_result_sha256"] or (
            sha256(frozen / "frozen_cases.json") != selection["frozen_manifest_sha256"]):
        raise ValueError("Original expert receipts changed during obstacle replay")
    expected = {row["case_id"]: row for row in json.loads((frozen / "frozen_cases.json").read_text())["cases"]}
    original_rows = {row["case"]: row for row in before["cases"]}
    affected = selection["selected_ids"]
    updated = {}
    ledger = []
    for index in range(2):
        directory = runs / f"{output.name}_b{index}"
        manifest = json.loads((plan / f"shard_{index}.json").read_text())
        intended = {row["case_id"] for row in manifest["cases"]}
        rows = []
        for name in ("result.json", "batch_progress.json"):
            path = directory / name
            if path.is_file():
                rows = json.loads(path.read_text())["cases"]
                break
        observed = {row["case"]: row for row in rows}
        if len(observed) != len(rows) or not observed.keys() <= intended:
            raise ValueError("Unexpected obstacle replay case IDs")
        for key in sorted(intended):
            if key in updated:
                raise ValueError(f"Duplicate physical replay: {key}")
            candidate = json.loads((plan / f"{key}.json").read_text())["feasible"][0]["candidate"]
            case = expected[key]
            row = observed.get(key, {"case": key, "group": case["group"], "scene_family": case["scene_family"],
                                     "status": "infrastructure_fail", "failure": "no physical replay receipt"})
            if row["group"] != case["group"] or row["scene_family"] != case["scene_family"]:
                raise ValueError(f"Physical replay metadata differs from frozen scene: {key}")
            row["shard"] = directory.name
            if key in observed:
                episode = directory / key
                diagnostic = json.loads((episode / "expert_diagnostic.json").read_text())
                if diagnostic["case"] != case or diagnostic["offline_plan"]["candidate"] != candidate:
                    raise ValueError(f"Physical replay was not bound to its planned direction: {key}")
                if row["status"] == "strict_success":
                    row["independent_audit"] = audited_success(episode, row, case)
            updated[key] = row
            ledger.append({"case": key, "previous_status": original_rows[key]["status"], "new_status": row["status"],
                           "previous_shard": original_rows[key]["shard"], "new_shard": directory.name,
                           "previous_candidate": PREVIOUS_DIRECTION, "new_candidate": candidate})
    if set(updated) != set(affected):
        raise ValueError("Replay shards did not account for all affected scenes")
    chosen = [updated.get(row["case"], row) for row in before["cases"]]
    summary = tally(chosen)
    complete_replay = all(row["status"] in {"strict_success", "physical_fail", "preflight_fail"} for row in updated.values())
    passed = summary["passes_rate_gate"] and complete_replay
    videos = representative_runs(chosen, runs, output / "videos")
    artifact = {**before, **summary, "passed": passed, "cases": chosen, "representative_videos": videos,
                "retained_original_cases": 100 - len(affected), "replayed_cases": ledger,
                "complete_replay_receipts": complete_replay,
                "original_result": {"path": str(prior), "strict_success": before["strict_success"], "groups": before["groups"]}}
    write_json(output / "result.json", artifact)
    print(json.dumps({key: artifact[key] for key in ("strict_success", "groups", "passed", "retained_original_cases")},
                     indent=2), flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser()
    stages = parser.add_subparsers(dest="stage", required=True)
    planning = stages.add_parser("prepare")
    for field in ("root", "urdf", "prior", "output"):
        planning.add_argument(field, type=Path)
    stages.add_parser("summarize").add_argument("output", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        return prepare(args.root, args.urdf, args.prior, args.output)
    return summarize(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
