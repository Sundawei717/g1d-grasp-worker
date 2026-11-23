"""Move the grasp into the labeled compact-box end using measured contact points."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from background_status import record_complete
from step3a_planner import adapt, build_planner, pose_matrix
from step3a_suites import CASE_PLANS, RUNS, planned_receipts, read, verify_groups

JOB = "step3a_01_compact_calibration"


def correct(first, index, shift):
    clear, obstacle = (read(CASE_PLANS / f"{kind}_{first + i}.json")
                       for i, kind in enumerate(("C", "B")))
    fit = clear["feasible"][0]
    original = clear["case"]
    seed = 2471400 + index * 2
    row = {**original, "seed": seed,
           "planner_hint": {"guide": [*fit["pre_grasp_q"], fit["approach_q"][0]]},
           "layout_description": "move measured finger contacts from the seam toward the positive end"}
    planner = build_planner(row)
    tangent = pose_matrix(row["target_pose_xyz_xyzw"]).rotation[:, 1]
    for offset in (shift, shift + .003, shift - .003):
        for dz in (0, .004):
            delta = tangent * offset + np.array([0, 0, dz])
            hover = pose_matrix(fit["hover_xyz_xyzw"])
            wrist = pose_matrix(fit["wrist_xyz_xyzw"])
            for pose in (hover, wrist):
                pose.translation = pose.translation + delta
            planner.candidate_goals = lambda: iter([(f"shift_end_{offset:.3f}_{dz:.3f}", hover, wrist)])
            result = planner.plan(seed=seed, stop_after=1, max_candidates=1, lift_height_m=.08)
            if not result["feasible"]:
                continue
            plan = {**clear, **result, "case": row}
            blocked = {**row, "seed": seed + 1, "fixtures": obstacle["case"]["fixtures"], "group": "B_obstacle"}
            detour = adapt(plan, blocked, block=True)
            if detour is None:
                continue
            tag = f"tier2_compact_end_{index}"
            slots = [{"scene_id": f"{tag}_{i}", "seed": seed + i} for i in (0, 1)]
            return planned_receipts([plan, detour], slots, 9940 + 2 * index, tag)
    return None


def main():
    accepted = set()
    for j, shift in enumerate((.013, .019)):
        with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as workers:
            futures = {
                i: workers.submit(correct, first, i + 2 * j, shift)
                for i, first in enumerate((9844, 9846)) if i not in accepted
            }
            groups = {i: group for i, future in futures.items() if (group := future.result())}
        print(f"compact attempt {j}: planned groups {list(groups)}", flush=True)
        if groups:
            result = verify_groups(list(groups.values()), f"pilots_guide_compact_calibration_r{j}")
            passed_ids = {c["pair_group_id"] for c in result["cases"]}
            accepted.update(i for i, group in groups.items() if group[0]["pair_group_id"] in passed_ids)
        if len(accepted) == 2:
            return
    raise RuntimeError(f"Only {len(accepted)}/2 compact-box groups passed after grip-center calibration")


if __name__ == "__main__":
    code = 1
    try:
        main()
        code = 0
    finally:
        record_complete(RUNS / JOB, code)
