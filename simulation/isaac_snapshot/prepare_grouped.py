import argparse
import concurrent.futures
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from school_planner import ARM_NAMES, CHECK_STEP, Planner, authored_meshes, sha256


GROUP_QUOTAS = {"A_target": 40, "B_obstacle": 35, "C_start": 25}
TARGETS = {
    "carton": "sand3d_packaging/00152/object.usdz",
    "can": "sand3d_packaging/00156/object.usdz",
}
BLOCKER = "soda_can/00000/object.usdz"
DEFAULT_ORDER = ("d0.012_tilt45_skew30_rev0", "d0.012_tilt45_skew15_rev0", "d0.012_tilt45_skew0_rev0")
BLOCKED_ORDER = ("d0.012_tilt90_skew-30_rev180", "d0.012_tilt90_skew-15_rev180",
                 "d0.012_tilt90_skew0_rev0", "d0.012_tilt90_skew15_rev180")


def verify_gates(root):
    files = {
        "simple": root / "runs/expert20_05/result.json",
        "blocker": root / "runs/obstacle_grasp_02/result.json",
        "carton": root / "reports/asset_admission_04_baseline_review.json",
        "can": root / "runs/can_admission_01/result.json",
    }
    data = {name: json.loads(path.read_text()) for name, path in files.items()}
    if not all(group["passed"] for group in data.values()) or data["simple"]["strict_success"] < 18:
        raise ValueError("Simple expert or asset admission gate is not passed")
    for role, relative in TARGETS.items():
        entry = data[role]
        if entry["strict_success"] != 10 or entry["asset_usd"] != str(root / "assets" / relative):
            raise ValueError(f"Invalid five-pose asset admission for {role}")
    return {name: {"path": str(path), "sha256": sha256(path)} for name, path in files.items()}


def generate_layouts(root, config, evidence, seed):
    rng = np.random.default_rng(seed)
    q_home = np.asarray(evidence["samples"][0]["right_rad"])
    table = next(points for name, points in authored_meshes(config["school_usd"]) if name == "/Scene/Table/Top")
    surface = float(table[:, 2].max() + config["school_position_m"][2])
    shapes = {name: root / "assets" / path for name, path in {**TARGETS, "blocker": BLOCKER}.items()}
    hulls = {name: np.concatenate([points for _, points in authored_meshes(path)]) for name, path in shapes.items()}

    def placement(name, x_value, y_value, yaw):
        orientation = Rotation.from_euler("z", yaw, degrees=True)
        height = surface - float(orientation.apply(hulls[name])[:, 2].min())
        return [float(x_value), float(y_value), height + (0 if name == "blocker" else 0.0001), *orientation.as_quat().tolist()]

    cases = []

    def add(group, family, role, slot, obstacle_pairs=(), start=None):
        fixtures = [{"name": name, "usd": str(shapes[asset]), "sha256": sha256(shapes[asset]), "pose": pose}
                    for name, asset, pose in obstacle_pairs]
        cases.append({"case_id": f"{group[0]}_{len(cases):03d}", "group": group,
                      "scene_family": family, "asset_role": role, "target_usd": str(shapes[role]),
                      "target_pose_xyz_xyzw": slot, "home_q": (q_home if start is None else start).tolist(),
                      "part": "upper_side_panels", "fixtures": fixtures,
                      "seed": seed + len(cases) + 1})

    for index in range(20):
        separation = (0.095, 0.110, 0.125, 0.140)[index % 4]
        x_value = 0.183 + rng.uniform(-0.009, 0.009)
        y_value = -0.418 + rng.uniform(-0.018, 0.019)
        axes = [x_value - separation / 2, x_value + separation / 2]
        if index % 2:
            axes.reverse()
        slots = {role: placement(role, axes[id_role], y_value + rng.uniform(-0.009, 0.009),
                                 rng.uniform(-19, 19)) for id_role, role in enumerate(TARGETS)}
        for role, other in (("carton", "can"), ("can", "carton")):
            add("A_target", f"switch_{index:02d}", role, slots[role], (("neighbor", other, slots[other]),))

    for index, (target_variant, blocker_variant) in enumerate(itertools.product(range(5), range(7))):
        dx, dy, yaw = ((0, 0, 0), (-0.005, 0.002, -9), (0.006, 0.003, 11), (-0.004, -0.005, 13),
                       (0.007, -0.002, -13))[target_variant]
        ox, oy = ((0, 0), (-0.004, -0.007), (0.005, -0.007), (-0.004, 0.006),
                  (0.005, 0.006), (-0.006, -0.002), (0.003, 0.011))[blocker_variant]
        slot = placement("carton", 0.1828 + dx, -0.422 + dy, yaw)
        blocker = placement("blocker", 0.1128 + dx + ox, -0.452 + dy + oy, 0)
        add("B_obstacle", f"block_{index:02d}", "carton", slot, (("blocker", "blocker", blocker),))

    for index in range(25):
        role = "carton" if index % 2 == 0 else "can"
        yaw = rng.uniform(-24, 24)
        slot = placement(role, 0.1828 + rng.uniform(-0.041, 0.041), -0.422 + rng.uniform(-0.036, 0.036), yaw)
        delta = rng.uniform(-1, 1, 7) * np.array([0.15, 0.11, 0.11, 0.12, 0.13, 0.12, 0.13])
        if np.linalg.norm(delta) < 0.14:
            delta[2] += 0.14
        start = q_home + delta
        start[3] = max(-1.018, start[3])
        if start[3] - q_home[3] < 0.02:
            start[3] = q_home[3] + 0.04 + (index % 4) * 0.017
        add("C_start", f"start_{index:02d}", role, slot, start=start)
    return cases


def plan_case(parameters):
    config, urdf, evidence, case = parameters
    config = {**config, "target_usd": case["target_usd"]}
    kwargs = {"home": case["home_q"], "part": case["part"]}
    planner = Planner(config, urdf, evidence, case["target_pose_xyz_xyzw"], obstacles=case["fixtures"], **kwargs)
    result = {"feasible": [], "obstacles": case["fixtures"], "case": case, "baseline_blocking_pair": None}
    if case["group"] == "B_obstacle" and case["fixtures"]:
        baseline = Planner(config, urdf, evidence, case["target_pose_xyz_xyzw"], **kwargs)
        reference = baseline.plan(seed=0, stop_after=1, max_candidates=32, preferred=DEFAULT_ORDER)
        if reference["feasible"]:
            direct = reference["feasible"][0]
            legs = [*direct["pre_grasp_q"], *direct["approach_q"][1:]]
            blocked = next((contact for first, second in zip(legs, legs[1:])
                            if (contact := planner.edge(np.asarray(first), np.asarray(second)))), None)
            if blocked and any(owner.startswith("blocker:") for owner in blocked):
                result["baseline_blocking_pair"] = blocked
                result["baseline_candidate"] = direct["candidate"]
            else:
                result["offline_failure"] = "fixture_does_not_block_baseline"
        else:
            result["offline_failure"] = "no_clear_baseline"
    if "offline_failure" not in result:
        preferred = DEFAULT_ORDER
        if case["asset_role"] == "can":
            preferred = ("d0.012_tilt45_skew-30_rev0", "d0.012_tilt45_skew-15_rev0")
        elif case["group"] == "B_obstacle" and case["fixtures"]:
            preferred = BLOCKED_ORDER
        scored = planner.plan(case["seed"], stop_after=1, max_candidates=86, preferred=preferred)
        result.update(scored)
        if not scored["feasible"]:
            result["offline_failure"] = "no_checked_grasp"
    result.update(target_sha256=sha256(config["target_usd"]), school_sha256=sha256(config["school_usd"]),
                  urdf_sha256=sha256(urdf), target_pose_xyz_xyzw=case["target_pose_xyz_xyzw"],
                  right_joint_names=list(ARM_NAMES), home_q=case["home_q"], collision_checks=planner.checks,
                  max_interpolation_l1_rad=CHECK_STEP)
    return result


def write_json(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    parser.add_argument("--seed", type=int, default=840112)
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "env_resolved.json").read_text())
    evidence = json.loads((root / "runs/lab_school_03/result.json").read_text())
    bindings = verify_gates(root)
    cases = generate_layouts(root, config, evidence, args.seed)
    if len(cases) != 100 or {group: sum(case["group"] == group for case in cases) for group in GROUP_QUOTAS} != GROUP_QUOTAS:
        raise ValueError("Invalid frozen group quotas")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "frozen_cases.json", {"seed": args.seed, "quotas": GROUP_QUOTAS,
                                                  "gates": bindings, "cases": cases})
    records = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(plan_case, (config, args.urdf, evidence, case)): case for case in cases}
        for future in concurrent.futures.as_completed(futures):
            case = futures[future]
            result = future.result()
            case_file = f"{case['case_id']}.json"
            write_json(args.output / case_file, result)
            records[case["case_id"]] = {**case, "plan": case_file, "offline_feasible": bool(result["feasible"]),
                                         "offline_failure": result.get("offline_failure")}
            print(f"{len(records):3d}/100 {case['case_id']} {case['asset_role']}: "
                  f"{result.get('offline_failure', 'planned')} ({result['collision_checks']} checks)", flush=True)
    ordered = [records[case["case_id"]] for case in cases]
    shards = []
    for role in TARGETS:
        role_cases = [case for case in ordered if case["asset_role"] == role]
        for index in range(0, len(role_cases), 17):
            key = f"{role}_{index // 17}"
            part = role_cases[index:index + 17]
            fixture_assets = {item["name"]: item["usd"] for case in part for item in case["fixtures"]}
            write_json(args.output / f"shard_{key}.json", {"phase": "grouped_expert_100",
                       "asset": role_cases[0]["target_usd"], "fixture_assets": fixture_assets, "cases": part,
                       "gates": bindings})
            shards.append({"name": key, "manifest": f"shard_{key}.json", "asset_role": role,
                           "case_ids": [case["case_id"] for case in part]})
    write_json(args.output / "result.json", {"passed": True, "frozen_cases": "frozen_cases.json",
              "cases": ordered, "quotas": GROUP_QUOTAS, "shards": shards})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
