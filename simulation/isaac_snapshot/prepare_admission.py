import argparse
import json
import sys
from pathlib import Path

from scipy.spatial.transform import Rotation

from school_planner import ARM_NAMES, CHECK_STEP, Planner, sha256


POSES = (
    (0.0, 0.0, 0.0),
    (-0.018, -0.012, 12.0),
    (0.016, -0.008, -14.0),
    (-0.013, 0.018, -9.0),
    (0.011, 0.025, 17.0),
)
SIMPLE_POSES = (
    (-0.024, -0.018, -18), (-0.018, -0.018, 3), (-0.012, -0.019, 21), (0.004, -0.020, -9), (0.021, -0.018, 14),
    (-0.023, -0.006, 8), (-0.014, -0.004, -22), (-0.002, -0.005, 12), (0.009, -0.005, -15), (0.019, -0.004, 0),
    (-0.022, 0.011, 16), (-0.017, 0.010, -10), (-0.006, 0.012, 25), (0.007, 0.011, -24), (0.020, 0.010, 9),
    (-0.027, 0.024, 0), (-0.014, 0.025, 19), (-0.004, 0.022, -12), (0.008, 0.026, 5), (0.021, 0.024, -20),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("lab_evidence", type=Path)
    parser.add_argument("target_evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--simple-expert", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    evidence = json.loads(args.lab_evidence.read_text())
    anchor = json.loads(args.target_evidence.read_text())["target_initial_xyz_xyzw"]
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    layouts = SIMPLE_POSES if args.simple_expert else POSES
    repeats = 1 if args.simple_expert else 2
    for index, (offset_x, offset_y, yaw) in enumerate(layouts):
        rotation = Rotation.from_euler("z", yaw, degrees=True) * Rotation.from_quat(anchor[3:])
        target_pose = [anchor[0] + offset_x, anchor[1] + offset_y, anchor[2], *rotation.as_quat()]
        planner = Planner(config, args.urdf, evidence, target_pose)
        seed = (20683 if args.simple_expert else 19711) + index
        result = planner.plan(seed=seed, stop_after=2, max_candidates=100)
        result.update(urdf_sha256=sha256(args.urdf), target_sha256=sha256(config["target_usd"]),
                      school_sha256=sha256(config["school_usd"]), target_pose_xyz_xyzw=target_pose,
                      fk_agreement=planner.fk_errors, right_joint_names=list(ARM_NAMES),
                      home_q=planner.home.tolist(), collision_checks=planner.checks,
                      max_interpolation_l1_rad=CHECK_STEP, seed=seed)
        (args.output / f"pose_{index:02d}.json").write_text(json.dumps(result, indent=2))
        for repetition in range(repeats):
            cases.append({"case_id": f"pose_{index:02d}_rep{repetition}", "plan": f"pose_{index:02d}.json",
                          "feasible_offline": bool(result["feasible"])})
        print(f"pose {index}: {len(result['feasible'])} feasible; {planner.checks} checks", flush=True)
    manifest = {"scope": "one asset, one grip region, frozen target layouts at the known home pose",
                "phase": "simple_expert_20" if args.simple_expert else "asset_admission_10",
                "asset": config["target_usd"], "cases": cases, "offline_feasible": sum(case["feasible_offline"] for case in cases)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
