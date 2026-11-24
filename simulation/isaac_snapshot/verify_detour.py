import argparse
import json
from pathlib import Path

import numpy as np

from school_planner import CHECK_STEP, Planner, authored_meshes, pose_matrix, sha256


def verify(config, urdf, evidence, plan, obstacle_usd, full_grasp=False):
    obstacle = np.concatenate([points for _, points in authored_meshes(obstacle_usd)])
    table = next(points for name, points in authored_meshes(config["school_usd"]) if name == "/Scene/Table/Top")
    table = table + np.asarray(config["school_position_m"])
    z = float(table[:, 2].max() - obstacle[:, 2].min())
    target = plan["target_pose_xyz_xyzw"]
    receipt = []
    placements = ((0.0, -0.10), (-0.05, -0.08), (0.05, -0.08), (-0.07, -0.03), (0.07, -0.03),
                  (-0.025, -0.075), (0.025, -0.075), (0.0, -0.14), (-0.08, -0.12), (0.08, -0.12))
    if full_grasp:
        fine = [(round(float(x_value), 3), round(float(y_value), 3))
                for x_value in np.arange(-0.105, -0.044, 0.01) for y_value in np.arange(-0.075, 0.026, 0.01)]
        placements = (*placements, *fine)
    planner = Planner(config, urdf, evidence, target,
                      obstacles=[{"usd": str(obstacle_usd), "name": "blocker", "pose": [0, 0, z, 0, 0, 0, 1]}])
    obstacle_indices = [index for index, path in enumerate(planner.fixture_names) if path.startswith("blocker:")]
    for candidate in plan["feasible"]:
        home = np.asarray(plan["home_q"])
        hover = np.asarray(candidate["pre_grasp_q"][-1])
        for offset_x, offset_y in placements:
            placement = [target[0] + offset_x, target[1] + offset_y, z, 0, 0, 0, 1]
            if not np.all((obstacle[:, :2] + placement[:2]).min(0) >= table[:, :2].min(0)) or not np.all(
                    (obstacle[:, :2] + placement[:2]).max(0) <= table[:, :2].max(0)):
                continue
            for index in obstacle_indices:
                planner.fixture_poses[index] = pose_matrix(placement)
            direct = planner.edge(home, hover)
            if not direct or not any(part.startswith("blocker:") for part in direct):
                receipt.append({"candidate": candidate["candidate"], "placement": placement, "reason": "direct_not_blocked"})
                continue
            if planner.collision(home) or planner.collision(hover):
                receipt.append({"candidate": candidate["candidate"], "placement": placement, "reason": "endpoint_blocked"})
                continue
            if full_grasp:
                approach = [np.asarray(point) for point in candidate["approach_q"]]
                lift = [np.asarray(point) for point in candidate["trial_lift_ik_only_q"]]
                blocked = next((hit for first, second in zip(approach, approach[1:])
                                if (hit := planner.edge(first, second))), None)
                blocked = blocked or planner.screen_lift(approach[-1], lift, pose_matrix(candidate["wrist_xyz_xyzw"]))
                if blocked:
                    receipt.append({"candidate": candidate["candidate"], "placement": placement,
                                    "reason": "approach_or_lift_blocked", "pair": blocked})
                    continue
            route, _ = planner.route(home, hover, np.random.default_rng(20261118), budget=320)
            if not route:
                receipt.append({"candidate": candidate["candidate"], "placement": placement, "reason": "rrt_budget_exhausted"})
                continue
            if len(route) <= 2 or any(planner.edge(first, second) for first, second in zip(route, route[1:])):
                raise RuntimeError("Detour did not pass independent edge verification")
            nodes = [point.tolist() for point in route]
            result = {"passed": True, "candidate": candidate["candidate"], "blocking_direct_pair": direct,
                      "asset": str(obstacle_usd), "obstacle_xyz_xyzw": placement, "nodes": nodes,
                      "checks": planner.checks, "joint_interpolation_l1_rad": CHECK_STEP, "attempts": receipt,
                      "scope": "offline pre-grasp motion only with a named fixture on the school table"}
            if full_grasp:
                result.update({key: plan[key] for key in ("urdf_sha256", "target_sha256", "school_sha256",
                                                         "target_pose_xyz_xyzw", "right_joint_names", "home_q")})
                result["scope"] = "offline complete approach and nominal hold clearance, no physical certificate"
                result["feasible"] = [{**candidate, "pre_grasp_q": nodes, "joint_margin_collision_check_m": 0.002}]
                result["obstacles"] = [{"name": "soda_blocker", "usd": str(obstacle_usd.resolve()),
                                        "sha256": sha256(obstacle_usd), "pose": placement}]
            return result
    return {"passed": False, "attempts": receipt, "scope": "offline pre-grasp only"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("obstacle_usd", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--full-grasp", action="store_true")
    args = parser.parse_args()
    result = verify(json.loads(args.config.read_text()), args.urdf, json.loads(args.evidence.read_text()),
                    json.loads(args.plan.read_text()), args.obstacle_usd, args.full_grasp)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print("Detour verified:", result["passed"], "checked placements:", len(result["attempts"]), flush=True)
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
