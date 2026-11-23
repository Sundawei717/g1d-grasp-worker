"""Verify a second distinct layout for the measured round-package grasp."""
import numpy as np

from background_status import record_complete
from step3a_planner import adapt, build_planner, pose_matrix
from step3a_suites import CASE_PLANS, RUNS, WORK, planned_receipts, read, verify_groups

JOB = "step3a_01_round_followup"


def plan_pair(original, obstacle, offset, index, *, label="round", seed_start=2471300, first_id=9920):
    fit = original["feasible"][0]
    q = [*fit["pre_grasp_q"], fit["approach_q"][0]]
    target = np.array(original["case"]["target_pose_xyz_xyzw"])
    target[:2] += offset
    seed = seed_start + 2 * index
    clear = {**original["case"], "target_pose_xyz_xyzw": target.tolist(),
             "seed": seed, "planner_hint": {"guide": q},
             "layout_description": "translated from a physically verified round-package grasp"}
    planner = build_planner(clear)
    options = []
    for height in (0, .004, -.004):
        hover = pose_matrix(fit["hover_xyz_xyzw"])
        wrist = pose_matrix(fit["wrist_xyz_xyzw"])
        for pose in (hover, wrist):
            pose.translation = pose.translation + np.array([*offset, height])
        options.append((f"reference_height_{height:+.3f}", hover, wrist))
    planner.candidate_goals = lambda: iter(options)
    result = planner.plan(seed=seed, stop_after=1, max_candidates=len(options), lift_height_m=.08)
    if not result["feasible"]:
        return None
    case_plan = {**original, **result, "case": clear, "target_pose_xyz_xyzw": target.tolist()}
    fixture = dict(obstacle["case"]["fixtures"][0])
    old_pose = np.array(fixture["pose"])
    old_pose[:2] += offset
    fixture["pose"] = old_pose.tolist()
    blocked = {**clear, "fixtures": [fixture], "group": "B_obstacle", "seed": seed + 1,
               "layout_description": "same translated grasp with a directly blocking panel"}
    detour = adapt(case_plan, blocked, block=True)
    if detour is None:
        return None
    tag = f"tier2_{label}_followup_{index}"
    slots = [{"scene_id": f"{tag}_{i}", "seed": seed + i} for i in (0, 1)]
    return planned_receipts([case_plan, detour], slots, first_id + 2 * index, tag)


def main():
    source = read(WORK / "pilots_guide_round_package_r1_result.json")["cases"]
    assert len(source) == 2 and all(c["replay_verified"] for c in source)
    original, obstacle = (read(CASE_PLANS / f"{r['case_id']}.json") for r in source)
    physical_trials = 0
    for index, offset in enumerate(((.010, .004), (-.010, .003), (.008, -.009), (-.005, -.010))):
        group = plan_pair(original, obstacle, offset, index)
        print("round follow-up", index, offset, "planned" if group else "IK/clearance rejected", flush=True)
        if group:
            result = verify_groups([group], f"pilots_guide_round_followup_r{index}")
            if len(result["cases"]) == 2:
                return
            physical_trials += 1
            if physical_trials >= 2:
                break
    raise RuntimeError("Second round-package layout did not pass recording and native replay")


if __name__ == "__main__":
    code = 1
    try:
        main()
        code = 0
    finally:
        record_complete(RUNS / JOB, code)
