"""Screen measured wrist geometries on the remaining independent local substitutes."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from background_status import record_complete
from step3a_planner import ARM_NAMES, CFG, PAD_OFFSET, URDF, asset_case, build_planner, placement
from step3a_suites import CASE_PLANS, RUNS, WORK, paired_obstacle, planned_receipts, read, verify_groups
from step3a_verify import file_hash

JOB = "step3a_01_final_substitutes"
ASSETS = (("matte_soda", ("C_9850", "C_9920")), ("marker_long_box", ("C_9844", "C_9846")))


def candidate(item, source, index, detour_builder=paired_obstacle):
    original = read(CASE_PLANS / f"{source}.json")
    fit = original["feasible"][0]
    old = original["case"]
    marker = item["name"] == "marker_long_box"
    seed = 2471800 + 2 * index
    part = "box_positive_end" if marker else "upper_side_panels"
    row = asset_case(item, part, index, seed)
    x, y = old["target_pose_xyz_xyzw"][:2]
    if index % 2:
        y += .008
    yaw = old["yaw_deg"]
    target = placement(item["asset"], x, y, yaw)
    if marker:
        box_rotation = Rotation.from_euler("z", yaw, degrees=True) * Rotation.from_euler("x", 90, degrees=True)
        target = [x, y, target[2] - item["bounds"][0][1], *box_rotation.as_quat()]
    gripper = Rotation.from_quat(fit["wrist_xyz_xyzw"][3:])
    row.update(target_pose_xyz_xyzw=target, yaw_deg=yaw, home_q=old["home_q"],
               check_alternate_gaps=True,
               planner_hint={"guide": [*fit["pre_grasp_q"], fit["approach_q"][0]]},
               layout_description=f"independent local object, wrist orientation retargeted from {source}")
    planner = build_planner(row)
    poses = []
    centers = ((0., .035, .030), (0., .028, .030), (0., .042, .030)) if marker else (
               (0., 0., .107), (0., 0., .115), (0., 0., .098))
    for center in centers:
        for pitch in (0, 10, -10):
            rotation = gripper * Rotation.from_euler("y", pitch, degrees=True)
            grip = planner.target_pose.act(np.array(center))
            wrist = grip - rotation.apply(PAD_OFFSET)
            hover = wrist - rotation.apply([.068, 0, 0])
            poses.append((f"retarget_{center}_{pitch}", pin.SE3(rotation.as_matrix(), hover),
                          pin.SE3(rotation.as_matrix(), wrist)))
    planner.candidate_goals = lambda: iter(poses)
    result = planner.plan(seed=seed, stop_after=1, max_candidates=len(poses), lift_height_m=.08)
    if not result["feasible"]:
        print(item["name"], source, "no collision-free retargeted grasp", flush=True)
        return None
    row["planned_gripper_gap_m"] = planner.planned_gap_m
    plan = {**result, "case": row, "obstacles": [], "home_q": row["home_q"],
            "right_joint_names": list(ARM_NAMES), "target_pose_xyz_xyzw": target,
            "school_sha256": file_hash(CFG["school_usd"]), "target_sha256": item["asset_sha256"],
            "urdf_sha256": file_hash(URDF)}
    detour = detour_builder(plan)
    if not detour:
        print(item["name"], source, "no compatible forced detour", flush=True)
        return None
    tag = f"tier2_substitute_retarget_{item['name']}_{index}"
    slots = [{"scene_id": f"{tag}_{i}", "seed": seed + i} for i in (0, 1)]
    return planned_receipts([plan, detour], slots, 9970 + 2 * index, tag)


def main():
    inventory = read(WORK / "asset_substitutes.json")["assets"]
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as planner:
        futures = {
            name: [planner.submit(candidate, inventory[name], source, 2 * n + j)
                   for j, source in enumerate(sources)]
            for n, (name, sources) in enumerate(ASSETS)
        }
        with ThreadPoolExecutor(max_workers=len(ASSETS)) as runs:
            def run(name):
                groups = [group for f in futures[name] if (group := f.result()) is not None]
                print(name, "retargeted groups", len(groups), flush=True)
                if groups:
                    return len(verify_groups(groups, f"pilots_guide_{name}_retarget")["cases"])
                return 0
            verified = list(runs.map(run, futures))
    if verified != [4, 4]:
        raise RuntimeError(f"Local substitutes did not both reach two accepted groups: {verified}")


if __name__ == "__main__":
    code = 1
    try:
        main()
        code = 0
    finally:
        record_complete(RUNS / JOB, code)
