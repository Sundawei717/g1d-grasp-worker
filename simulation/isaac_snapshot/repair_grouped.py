import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from prepare_grouped import GROUP_QUOTAS, write_json
from school_planner import Planner


def retry_case(args):
    config, urdf, evidence, baseline, guide = args
    case = baseline["case"]
    config = {**config, "target_usd": case["target_usd"]}
    planner = Planner(config, urdf, evidence, case["target_pose_xyz_xyzw"], obstacles=case["fixtures"],
                      home=case["home_q"], expanded_search=True, guide=guide["pre_grasp_q"] if guide else (),
                      part=case["part"])
    preferred = ("d0.012_tilt90_skew-15_rev180", "d0.012_tilt90_skew-30_rev180", "d0.012_tilt90_skew-45_rev180",
                 "d0.012_tilt75_skew-45_rev180", "d0.012_tilt90_skew0_rev180", "d0.012_tilt90_skew-60_rev180")
    changed = planner.plan(case["seed"], stop_after=1, max_candidates=110, preferred=preferred if case["fixtures"] else ())
    result = {**baseline, **changed, "original_offline_failure": baseline["offline_failure"],
              "offline_failure": None if changed["feasible"] else "expanded_search_no_checked_grasp",
              "collision_checks": planner.checks,
              "repair": "same frozen scene; neighboring route guide and bidirectional joint-space search"}
    return result


def try_read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frozen = json.loads((args.source / "frozen_cases.json").read_text())
    config = json.loads((args.root / "env_resolved.json").read_text())
    evidence = json.loads((args.root / "runs/lab_school_03/result.json").read_text())
    guide_receipt = json.loads((args.source / "B_040.json").read_text())
    guide = guide_receipt["feasible"][0]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "frozen_cases.json", frozen)
    cases = {case["case_id"]: case for case in frozen["cases"]}
    pending = set(cases)
    records = {}
    futures = {}

    def receive(key, result):
        if result["case"] != cases[key]:
            raise ValueError(f"Frozen case changed: {key}")
        write_json(args.output / f"{key}.json", result)
        records[key] = {**cases[key], "plan": f"{key}.json", "offline_feasible": bool(result["feasible"]),
                        "offline_failure": result.get("offline_failure")}
        print(f"{len(records):3d}/100 {key}: {result.get('offline_failure') or 'planned'}"
              f" repair={'yes' if result.get('repair') else 'no'}", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        while pending or futures:
            for key in sorted(pending):
                original = try_read(args.source / f"{key}.json")
                if original is None:
                    continue
                pending.remove(key)
                if original.get("offline_failure") == "no_checked_grasp":
                    futures[pool.submit(retry_case, (config, args.urdf, evidence, original,
                                                     guide if cases[key]["group"] == "B_obstacle" else None))] = key
                else:
                    receive(key, original)
            for future in list(futures):
                if future.done():
                    receive(futures.pop(future), future.result())
            marker = try_read(args.source.with_suffix(".complete"))
            if pending and marker and marker["exit_code"] != 0:
                raise RuntimeError("Baseline planning terminated before the frozen sweep completed")
            if pending or futures:
                time.sleep(2)
    while (baseline := try_read(args.source / "result.json")) is None:
        if try_read(args.source.with_suffix(".complete")):
            raise RuntimeError("Missing baseline summary after the baseline planner exited")
        time.sleep(2)
    for shard in baseline["shards"]:
        name = shard["manifest"]
        immutable = json.loads((args.source / name).read_text())
        immutable["cases"] = [records[key] for key in shard["case_ids"]]
        write_json(args.output / name, immutable)
    tally = {group: sum(item["offline_feasible"] and item["group"] == group for item in records.values())
             for group in GROUP_QUOTAS}
    result = {**baseline, "cases": [records[key] for key in cases],
              "original_plan": str(args.source.resolve()), "feasibility_by_group": tally}
    write_json(args.output / "result.json", result)
    print(f"Revised planning on all original cases: {tally}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
