import argparse
import concurrent.futures
import json
from collections import Counter
from pathlib import Path
from shutil import copyfile

import numpy as np

from prepare_grouped import plan_case, write_json
from repair_grouped import retry_case
from school_planner import ARM_NAMES, Planner


def plan_revised(parameters):
    config, urdf, evidence, case = parameters
    result = plan_case(parameters)
    if result.get("offline_failure") == "no_checked_grasp":
        result = retry_case((config, urdf, evidence, result, None))
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "urdf", "source", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    frozen = json.loads((args.source / "frozen_cases.json").read_text())
    previous = json.loads((args.source / "result.json").read_text())
    config = json.loads((args.root / "env_resolved.json").read_text())
    evidence = json.loads((args.root / "runs/lab_school_03/result.json").read_text())
    cases = {}
    rejected_starts = {}
    revised_records = {}
    for case in frozen["cases"]:
        key = case["case_id"]
        cases[key] = case
        old = json.loads((args.source / f"{key}.json").read_text())
        if old["case"] != case:
            raise ValueError(f"Original plan differs from the frozen case: {key}")
        if case["group"] != "C_start" or not old.get("home_contacts"):
            continue
        q = np.asarray(case["home_q"], dtype=float)
        planner = Planner({**config, "target_usd": case["target_usd"]}, args.urdf, evidence,
                          case["target_pose_xyz_xyzw"], home=q)
        blocked = planner.collision(q, 0.002)
        if blocked != ("torso_link", "right_shoulder_yaw_link") or q[1] <= 0:
            raise ValueError(f"Unexpected start-pose failure: {key} {blocked}")
        replacement = q.copy()
        replacement[1] = -q[1]
        if np.any(replacement < planner.lower) or np.any(replacement > planner.upper):
            raise ValueError(f"Corrected starting pose exceeded a joint limit: {key}")
        if planner.collision(replacement, 0.004):
            raise ValueError(f"Prespecified reflection did not clear the start-pose check: {key}")
        rejected_starts[key] = {"original_home_q": q.tolist(), "revised_home_q": replacement.tolist(),
                                "blocking_pair_at_2_mm": blocked, "blocking_pair_at_0_mm": planner.collision(q, 0)}
        cases[key] = {**case, "home_q": replacement.tolist()}
    if not rejected_starts:
        raise ValueError("No unsafe sampled starting poses to correct")

    args.output.mkdir(parents=True, exist_ok=False)
    start_screen = {"source_frozen_cases": str((args.source / "frozen_cases.json").resolve()),
                    "original_configuration_count": len(cases),
                    "original_C_start_count": frozen["quotas"]["C_start"],
                    "rejected_initializations": len(rejected_starts),
                    "replacement_rule": f"Reflect only {ARM_NAMES[1]} away from the torso; one candidate per rejected start",
                    "replacements": rejected_starts}
    write_json(args.output / "start_pose_screen.json", start_screen)
    write_json(args.output / "frozen_cases.json",
               {**frozen, "cases": list(cases.values()), "start_pose_screen": "start_pose_screen.json"})
    for record in previous["cases"]:
        key = record["case_id"]
        if key not in rejected_starts:
            copyfile(args.source / record["plan"], args.output / record["plan"])
            revised_records[key] = record
    print(f"Froze {len(cases)} cases after screening {len(rejected_starts)} invalid original starts", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(plan_revised, (config, args.urdf, evidence, cases[key])): key
                   for key in rejected_starts}
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            result = future.result()
            write_json(args.output / f"{key}.json", result)
            revised_records[key] = {**cases[key], "plan": f"{key}.json", "offline_feasible": bool(result["feasible"]),
                                    "offline_failure": result.get("offline_failure")}
            print(f"{len(revised_records)}/100 {key}: {result.get('offline_failure') or 'planned'}", flush=True)

    for shard in previous["shards"]:
        path = args.source / shard["manifest"]
        manifest = json.loads(path.read_text())
        write_json(args.output / path.name, {**manifest, "cases": [revised_records[key] for key in shard["case_ids"]]})
    records = [revised_records[key] for key in cases]
    tally = Counter(record["group"] for record in records if record["offline_feasible"])
    print(f"Offline feasibility after start screening: {dict(tally)}", flush=True)
    write_json(args.output / "result.json", {**previous, "cases": records, "original_plan": str(args.source.resolve()),
                                            "start_pose_screen": "start_pose_screen.json",
                                            "feasibility_by_group": dict(tally)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
