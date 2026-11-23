import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from episode_store import verify_three_view
from grasp_regions import MUG_BODY, MUG_HANDLE
from school_planner import ARM_NAMES, CHECK_STEP, PAD_OFFSET, Planner, authored_meshes
from three_view_detector import sha256, write_json
from three_view_eval import read


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
URDF = Path("/data/sundawei/vla_rsi_g1d/assets/Assets/Robots/g1d/g1d_dex1.urdf")


class MugPlanner(Planner):
    def candidate_goals(self):
        direction = self.target_pose.rotation[:, 1]
        yaw = math.degrees(math.atan2(direction[1], direction[0]))
        handle = self.part == MUG_HANDLE
        centers = ((0.050, 0, 0.012), (0.051, 0, 0.004), (0.048, 0, 0.018),
                   (0.053, 0, 0.008)) if handle else ((-0.022, 0, 0.028), (-0.020, 0, 0.031))
        for center in centers:
            grip = self.target_pose.act(np.asarray(center))
            for tilt in (75, 60, 90, 45):
                for skew in (0, -15, 15, -30, 30):
                    for reverse in (0, 180):
                        rotation = (Rotation.from_euler("z", yaw + skew + reverse, degrees=True)
                                    * Rotation.from_euler("x", -tilt, degrees=True)
                                    * Rotation.from_euler("z", 90, degrees=True))
                        axes = rotation.as_matrix()
                        wrist = grip - rotation.apply(PAD_OFFSET)
                        hover = wrist - axes[:, 0] * 0.062
                        code = f"p{center[0]:.3f}_{center[2]:.3f}_t{tilt}_s{skew}_r{reverse}"
                        yield code, pin.SE3(axes, hover), pin.SE3(axes, wrist)

    def screen_lift(self, initial, lift, wrist):
        # This is the minimum geometric aperture screened offline; the physical
        # controller measures actual pad force and aperture before locking.
        gap = 0.024 if self.part == MUG_HANDLE else 0.077
        closed = np.full(2, -0.02 + (0.09007 - gap) / 2)
        relative = wrist.inverse() * self.target_pose
        for opening in np.linspace(-0.02, closed, 10):
            if hit := self.collision(initial, allow_pad_target=True, fingers=opening):
                return hit
        for first, second in zip([initial, *lift], lift):
            count = max(1, math.ceil(np.linalg.norm(second - first, ord=1) / CHECK_STEP))
            for fraction in np.linspace(0, 1, count + 1)[1:]:
                q = first + fraction * (second - first)
                held = self.world_frame(q) * relative
                if hit := self.collision(q, allow_pad_target=True, held_pose=held, fingers=closed):
                    return hit
        return None


def plan_case(cfg, evidence, case):
    planner = MugPlanner(cfg, URDF, evidence, case["target_pose_xyz_xyzw"], home=case["home_q"], part=case["part"])
    result = planner.plan(seed=case["seed"], stop_after=1, max_candidates=90)
    return {**result, "case": case, "obstacles": [], "home_q": case["home_q"],
            "right_joint_names": list(ARM_NAMES), "target_pose_xyz_xyzw": case["target_pose_xyz_xyzw"],
            "school_sha256": sha256(cfg["school_usd"]), "target_sha256": sha256(cfg["target_usd"]),
            "urdf_sha256": sha256(URDF), "collision_checks": planner.checks}


def template():
    return read(RUNS / "three_view_step0_01/plans/expert_carton.json")


def prepare(output):
    assets = read(RUNS / "step1_mug_asset/result.json")
    cfg = {**read(ROOT / "env_resolved.json"), "target_usd": assets["asset"]}
    evidence = read(RUNS / "lab_school_03/result.json")
    output.mkdir(parents=True)
    write_json(output / "config.json", cfg)
    table = next(points for name, points in authored_meshes(cfg["school_usd"]) if name == "/Scene/Table/Top")
    top = float(table[:, 2].max() + cfg["school_position_m"][2])
    lowest = min(float(points[:, 2].min()) for _, points in authored_meshes(cfg["target_usd"]))
    cases, planned = [], {}
    layouts = [(0, 0.179, -0.417), (-12, 0.179, -0.407), (12, 0.182, -0.422),
               (-25, 0.176, -0.406), (25, 0.168, -0.419), (-35, 0.181, -0.398),
               (35, 0.173, -0.425), (-5, 0.168, -0.414), (18, 0.186, -0.426), (-18, 0.175, -0.423)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {}
        for index, (yaw, x, y) in enumerate(layouts):
            quat = Rotation.from_euler("z", yaw, degrees=True).as_quat().tolist()
            for part in (MUG_BODY, MUG_HANDLE):
                handle = part == MUG_HANDLE
                key = f"C_{6100 + 2 * index + int(handle)}"
                case = {"case_id": key, "seed": 8910 + index, "group": "C_start", "asset_role": "mug",
                        "asset_family": "mug/00007", "scene_family": f"mug_step1_pair_{index:02}",
                        "split": "step1_dev", "config_sha256": sha256(output / "config.json"), "part": part,
                        "target_usd": cfg["target_usd"], "target_pose_xyz_xyzw": [x, y, top - lowest + 0.00005, *quat],
                        "home_q": list(evidence["samples"][0]["right_rad"]), "fixtures": [], "command_delay_substeps": 0,
                        "yaw_deg": yaw, "part_meshes": {"face_atlas": assets["atlas"]},
                        "grip_goal_m": 0.002 if handle else 0.051}
                cases.append(case)
                futures[pool.submit(plan_case, cfg, evidence, case)] = case
        for job in as_completed(futures):
            case, plan = futures[job], job.result()
            key = case["case_id"]
            write_json(output / f"{key}.json", plan)
            planned[key] = {**case, "plan": f"{key}.json", "offline_feasible": bool(plan["feasible"])}
            print(key, case["part"], case["yaw_deg"], len(plan["feasible"]), plan["collision_checks"], flush=True)
    pairs = [[planned[row["case_id"]] for row in cases[i:i+2]] for i in range(0, len(cases), 2)]
    eligible = [pair for pair in pairs if all(row["offline_feasible"] for row in pair)]
    for i in range(0, len(eligible), 4):
        rows = [row for pair in eligible[i:i+4] for row in pair]
        write_json(output / f"collect_{i // 4}.json", {**template(), "phase": "three_view_expert_v3",
                                                      "asset": cfg["target_usd"], "target_query": "mug",
                                                      "fixture_assets": {}, "cases": rows})
    write_json(output / "result.json", {"passed": bool(eligible), "offline_ready_pairs": len(eligible), "pairs": pairs})


def replays(output, capture_dirs):
    grouped = {}
    for capture in capture_dirs:
        for row in read(capture / "result.json")["cases"]:
            if not (receipt := row.get("dataset_receipt")):
                continue
            with h5py.File(receipt["path"], "r") as h5:
                case = json.loads(h5["meta/case_json"].asstr()[()])
                prompt = json.loads(h5["meta/initial_prompt_json"].asstr()[()])
                assert h5.attrs["schema"] == "g1d_v3_step_aligned"
            row = {"receipt": receipt, "case": case, "prompt": prompt}
            grouped.setdefault(case["scene_family"], []).append(row)
    eligible = []
    for family, pair in sorted(grouped.items()):
        if {row["case"]["part"] for row in pair} != {MUG_BODY, MUG_HANDLE} or len(pair) != 2:
            continue
        a, b = pair
        for key in ("target_pose_xyz_xyzw", "home_q", "seed", "fixtures", "lift_height_m"):
            if a["case"][key] != b["case"][key]:
                raise ValueError(f"Paired initial condition changed for {family}")
        if (a["prompt"]["bbox_0"] != b["prompt"]["bbox_0"]
                or a["prompt"]["affordance_0"] == b["prompt"]["affordance_0"]):
            raise ValueError(f"Expected an identical box and two different affordance points: {family}")
        eligible.extend(pair)
    for i in range(0, len(eligible), 8):
        rows = []
        for row in eligible[i:i+8]:
            case, receipt = row["case"], row["receipt"]
            rows.append({**case, "plan": f"{case['case_id']}.json", "offline_feasible": True,
                         "replay_h5": {"path": receipt["path"], "sha256": receipt["sha256"]}})
        write_json(output / f"replay_{i // 8}.json", {**template(), "phase": "three_view_unified_replay",
                                                      "asset": rows[0]["target_usd"], "target_query": "mug",
                                                      "fixture_assets": {}, "cases": rows})
    write_json(output / "episodes.json", {"paired_layouts": len(eligible)//2,
                                           "episodes": [row["receipt"] for row in eligible]})


def review(output, captures, replays):
    episodes = read(output / "episodes.json")
    capture_rows, replay_rows = [], []

    def row(root, case):
        report = read(root / case / "expert_diagnostic.json")
        policy = report.get("policy_rollout", {})
        trigger = policy.get("trials", [{}])[0].get("trigger") if policy.get("trials") else None
        return {"case": case, "part": report["part"], "status": policy.get("status"),
                "part_hit": policy.get("part_hit"), "failure": report.get("failure"),
                "diagnostic_path": str(root / case / "expert_diagnostic.json"),
                "trigger": trigger, "alignment": report.get("replay_alignment")}

    for directory in captures:
        for case in read(directory / "result.json")["cases"]:
            capture_rows.append({**row(directory, case["case"]), "persisted": bool(case.get("dataset_receipt"))})
    for directory in replays:
        replay_rows.extend(row(directory, case["case"]) for case in read(directory / "result.json")["cases"])
    for episode in episodes["episodes"]:
        if sha256(episode["path"]) != episode["sha256"]:
            raise ValueError("Collected aligned episode SHA-256 changed")
        with h5py.File(episode["path"], "r") as data:
            length = episode["steps"]
            assert verify_three_view(data) == length + 1
            phases = data["phase"].asstr()[...]
            assert np.array_equal(data["obs/control_step_id"][:], np.arange(length + 1))
            assert np.array_equal(data["action/control_step_id"][:], np.arange(length))
            assert np.array_equal(data["action/native_command"][:], data["action/native_substep_command"][:, -1])
            assert np.array_equal(data["train_action_valid"][:], phases != "lift")
            assert set(phases) == {"approach", "close", "lift"}
            assert np.allclose(np.diff(data["obs/timestamp"][:]), 0.05, atol=2e-7, rtol=0)
            assert data["obs/qpos"].shape == (length + 1, 7)
            assert data["obs/gripper_width"].shape == (length + 1,)
            assert data["obs/eef_pose"].shape == (length + 1, 7)
            lift_index = int(np.flatnonzero(phases == "lift")[0])
            finger_targets = data["action/native_substep_command"][lift_index:, :, 7:]
            assert np.allclose(finger_targets, data["action/native_command"][lift_index - 1, 7:],
                               atol=1e-7, rtol=0), "The saved lift did not lock the last commanded fingers"
            for image in ("obs/rgb_left", "obs/left_wrist/rgb", "obs/right_wrist/rgb"):
                assert data[image].shape == (length + 1, 224, 224, 3) and data[image].dtype == np.uint8
                assert data[image].attrs["encoding"] == "raw" and data[image].compression == "gzip"
    by_id = {r["case"]: r for r in replay_rows}
    if by_id.keys() != {ep["case"] for ep in episodes["episodes"]} or len(by_id) != len(replay_rows):
        raise ValueError("Replay and paired collection episode sets differ")
    for r in replay_rows:
        e = r["alignment"]
        r["passed"] = bool(r["status"] == "success" and r["part_hit"] and e
                           and e["max_timestamp_error_s"] <= 2e-7 and e["max_native_setpoint_error"] <= 1e-6
                           and e["max_arm_error_rad"] <= 0.005 and e["max_jaw_error_m"] <= 0.002
                           and e["trigger_delta_s"] is not None and abs(e["trigger_delta_s"]) <= 0.10)
    write_json(output / "result.json", {"phase": "step1_mug_pairs", "configured_pairs": episodes["paired_layouts"],
                                        "collection_attempts": len(capture_rows), "capture": capture_rows,
                                        "replay": replay_rows,
                                        "passed_replays": sum(r["passed"] for r in replay_rows),
                                        "replay_part_hits": dict(Counter(r["part"] for r in replay_rows if r["part_hit"])),
                                        "v3_verified_episodes": len(episodes["episodes"]),
                                        "passed": episodes["paired_layouts"] >= 5 and all(r["passed"] for r in replay_rows)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "replays", "review"))
    parser.add_argument("output", type=Path)
    parser.add_argument("captures", type=Path, nargs="*")
    parser.add_argument("--replay", type=Path, nargs="+")
    args = parser.parse_args()
    if args.stage == "replays":
        replays(args.output.resolve(), args.captures)
    elif args.stage == "review":
        review(args.output.resolve(), args.captures, args.replay)
    else:
        prepare(args.output.resolve())
