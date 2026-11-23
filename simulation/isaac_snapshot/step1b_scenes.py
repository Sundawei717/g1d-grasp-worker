import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import math
import os

import h5py
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from episode_store import verify_three_view
from grasp_regions import BASKET_HANDLE, BASKET_RIM, BOX_NEG_END, BOX_POS_END, MUG_BODY, MUG_HANDLE
from school_planner import ARM_NAMES, CHECK_STEP, PAD_OFFSET, authored_meshes, pose_matrix
from step1_mug_pairs import MugPlanner, ROOT, RUNS, URDF, template
from three_view_detector import sha256, write_json
from three_view_eval import read
from unified_grasp import EPISODE_STEPS, LIFT_HEIGHT_M, LiftGateConfig


PAIR_PARTS = {"basket": (BASKET_HANDLE, BASKET_RIM), "mug": (MUG_BODY, MUG_HANDLE),
              "carton": (BOX_NEG_END, BOX_POS_END)}


class Tier1Planner(MugPlanner):
    def candidate_goals(self):
        if self.part in PAIR_PARTS["mug"]:
            yield from super().candidate_goals()
            return
        direction = self.target_pose.rotation[:, 0]
        yaw = math.degrees(math.atan2(direction[1], direction[0]))
        if self.part == BASKET_HANDLE:
            centers = [(x, 0, z) for z in (.095, .093, .097) for x in (0, .008, -.008)]
            yaw += 90
        elif self.part == BASKET_RIM:
            centers = [(x, y, self.rim_z + dz) for x in (0, .006, -.006) for dz in (.005, .001, .009, -.003)
                       for y in (-.054, .054)]
            yaw += 90
        else:
            direction = -1 if self.part == BOX_NEG_END else 1
            centers = [(0, y*direction, z) for z in (.048, .045, .050) for y in (.041, .046, .036)]
        for center, tilt, skew, rev in product(centers, (90, 75, 60, 45), (0, -15, 15, -30, 30), (0, 180)):
            rotation = (Rotation.from_euler("z", yaw + skew + rev, degrees=True)
                * Rotation.from_euler("x", -tilt, degrees=True) * Rotation.from_euler("z", 90, degrees=True))
            axes = rotation.as_matrix()
            grip = self.target_pose.act(np.asarray(center))
            wrist = grip - rotation.apply(PAD_OFFSET)
            hover = wrist - axes[:, 0] * .065
            code = f"{center[0]:.3f}_{center[1]:.3f}_{center[2]:.3f}_t{tilt}_s{skew}_r{rev}"
            yield code, pin.SE3(axes, hover), pin.SE3(axes, wrist)

    def screen_lift(self, initial, lift, wrist):
        if self.part in PAIR_PARTS["mug"]:
            return super().screen_lift(initial, lift, wrist)
        gap = 0.018 if self.part in PAIR_PARTS["basket"] else 0.043
        fingers = np.full(2, -0.02 + (0.09007 - gap) / 2)
        relative = wrist.inverse() * self.target_pose
        for opening in np.linspace(self.q_ref[self.fingers], fingers, 10):
            if hit := self.collision(initial, allow_pad_target=True, fingers=opening):
                return hit
        for first, second in zip([initial, *lift], lift):
            count = max(1, math.ceil(np.linalg.norm(second - first, ord=1) / CHECK_STEP))
            for amount in np.linspace(0, 1, count+1)[1:]:
                q = first + amount * (second-first)
                if hit := self.collision(q, allow_pad_target=True, held_pose=self.world_frame(q)*relative, fingers=fingers):
                    return hit
        return None

    def route(self, start, goal, rng, margin=.002, budget=135):
        if "blocker" not in self.obstacle_centers:
            return super().route(start, goal, rng, margin, budget)
        direct = self.edge(start, goal, margin)
        if not direct or not any(name.startswith("blocker:") for name in direct):
            return None, direct or ("direct_path_not_blocked",)
        hover = self.world_frame(goal)
        center = self.obstacle_centers["blocker"]
        # The right arm's outboard direction is world +X in this pinned scene.
        for side in (1, -1):
            for sideways, z in product((.095, .125, .075, .155), (.04, .07, .015)):
                points = [center[0] + side * sideways, center[1], hover.translation[2] + z]
                waypoint, _ = self.ik(pin.SE3(hover.rotation, np.asarray(points)), goal)
                if (waypoint is not None and not self.edge(start, waypoint, margin)
                        and not self.edge(waypoint, goal, margin)):
                    self.detour = {"side": "outboard" if side == 1 else "inboard", "waypoint_m": points,
                                   "direct_path_collision": direct}
                    return [start, waypoint, goal], None
        return None, direct


def make_plan(cfg, evidence, case):
    hint = case.get("planner_hint", {})
    planner = Tier1Planner(cfg, URDF, evidence, case["target_pose_xyz_xyzw"], home=case["home_q"],
                           part=case["part"], obstacles=case["fixtures"], expanded_search=True,
                           guide=hint.get("guide", ()))
    planner.obstacle_centers = {f["name"]: f["pose"][:3] for f in case["fixtures"]}
    planner.detour = None
    planner.rim_z = case["part_meshes"].get("rim_center_z_m", .046)
    if "approach_width_m" in case:
        planner.q_ref[planner.fingers] = -0.02 + (0.09007 - case["approach_width_m"]) / 2
    result = planner.plan(seed=case["seed"], stop_after=1, max_candidates=hint.get("max_candidates", 200),
                          preferred=hint.get("preferred", ()), lift_height_m=LIFT_HEIGHT_M)
    return {**result, "case": case, "obstacles": case["fixtures"], "home_q": case["home_q"],
            "right_joint_names": list(ARM_NAMES), "target_pose_xyz_xyzw": case["target_pose_xyz_xyzw"],
            "school_sha256": sha256(cfg["school_usd"]), "target_sha256": sha256(cfg["target_usd"]),
            "urdf_sha256": sha256(URDF), "collision_checks": planner.checks, "detour": planner.detour}


def infos(role):
    if role == "mug":
        src = read(RUNS / "step1_mug_asset/result.json")
        return src["asset"], {"face_atlas": src["atlas"]}, "mug/00007", "mug"
    directory = os.environ.get("STEP1B_BASKET_ASSET", "step1b_asset_lip") if role == "basket" else "step1b_asset"
    src = read(RUNS / directory / f"{role}.json")
    family, query = ("basket/00001", "basket") if role == "basket" else ("sand3d_packaging/00208", "box")
    return src["asset"], src["part_meshes"], family, query


def place(path, table, x, y, yaw):
    floor = min(float(points[:, 2].min()) for _, points in authored_meshes(path))
    return [x, y, table - floor + .00005, *Rotation.from_euler("z", yaw, degrees=True).as_quat().tolist()]


def prepare_known_obstacles(output, role, source):
    asset, _, _, query = infos(role)
    cfg = {**read(ROOT / "env_resolved.json"), "target_usd": asset}
    evidence = read(RUNS / "lab_school_03/result.json")
    output.mkdir(parents=True, exist_ok=True)
    config = output / "config.json"
    write_json(config, cfg)
    table = next(points for name, points in authored_meshes(cfg["school_usd"]) if name == "/Scene/Table/Top")
    table_z = float(table[:, 2].max() + cfg["school_position_m"][2])
    blocker = ROOT / "assets" / {"basket": "handled_paint/00000", "mug": "sand3d_packaging/00156",
                                 "carton": "soda_can/00000"}[role] / "object.usdz"
    custom_blocker = os.environ.get("STEP1B_BASKET_BLOCKER") if role == "basket" else None
    if custom_blocker:
        blocker = Path(custom_blocker).resolve(strict=True)
    blocker_xyz = np.concatenate([points for _, points in authored_meshes(blocker)])
    target_xyz = np.concatenate([points for _, points in authored_meshes(asset)])
    blocker_lo, blocker_hi = blocker_xyz.min(0), blocker_xyz.max(0)
    base = read(source / "plan_result.json")
    originals = [case for pair in base["available"] for case in pair if case["offline_feasible"]]
    selected, attempted = [], []
    parts = Counter()
    for original in originals:
        if parts[original["part"]] >= 2:
            continue
        plan = read(source / original["plan"])
        first = plan["feasible"][0]
        initial = original["target_pose_xyz_xyzw"]
        home = np.array(original["home_q"])
        hover = np.array(first["approach_q"][0])
        approach = np.array(first["approach_q"])
        lift = np.array(first["trial_lift_ik_only_q"])
        target_pose = pose_matrix(initial)
        world = target_xyz @ target_pose.rotation.T + target_pose.translation
        lower, upper = world.min(0), world.max(0)
        qwidth = original.get("approach_width_m", .09007)
        # Place the obstruction around the swept fingers of an already verified unobstructed approach.
        reference = Tier1Planner(cfg, URDF, evidence, initial, home=home, part=original["part"])
        centers = []
        for fraction in (.85, .72, .60, .92, .78):
            wrist = reference.world_frame(home + fraction * (hover - home))
            for jaw in (0, -qwidth/2, qwidth/2):
                point = wrist.act(np.array([.112, jaw, 0.0]))
                if point[2] > table_z + (blocker_hi[2] - blocker_lo[2]) + .012:
                    continue
                for dx, dy in ((0, 0), (-.019, -.018), (.019, -.018), (0, -.025), (-.024, 0), (.024, 0)):
                    x, y = point[:2] + [dx, dy]
                    if not .06 < x < .34 or not -.57 < y < -.41:
                        continue
                    if np.all([x, y] + blocker_hi[:2] + .014 >= lower[:2]) and np.all(
                            [x, y] + blocker_lo[:2] - .014 <= upper[:2]):
                        continue
                    candidate = (round(float(x), 3), round(float(y), 3))
                    if candidate not in centers:
                        centers.append(candidate)
        for x, y in centers:
            pose = [x, y, table_z - float(blocker_lo[2]) + .00005, 0, 0, 0, 1]
            obstacle = {"name": "blocker", "usd": str(blocker), "sha256": sha256(blocker), "pose": pose}
            planner = Tier1Planner(cfg, URDF, evidence, initial, home=home, part=original["part"], obstacles=[obstacle])
            planner.q_ref[planner.fingers] = -0.02 + (0.09007 - qwidth) / 2
            planner.obstacle_centers, planner.detour = {"blocker": pose[:3]}, None
            if planner.collision(home) or planner.collision(hover):
                continue
            direct = planner.edge(home, hover)
            if not direct or not any(name.startswith("blocker:") for name in direct):
                continue
            if any(planner.edge(a, b) for a, b in zip(approach, approach[1:])):
                continue
            if planner.screen_lift(approach[-1], lift, pose_matrix(first["wrist_xyz_xyzw"])):
                continue
            route, _ = planner.route(home, hover, np.random.default_rng(original["seed"]))
            if route is None:
                continue
            num = 6800 + {"mug": 0, "basket": 30, "carton": 60}[role] + (10 if custom_blocker else 0) + len(selected)
            case = {**plan["case"], "case_id": f"B_{num}", "group": "B_obstacle", "fixtures": [obstacle],
                    "config_sha256": sha256(config), "layout_description": "swept finger obstruction",
                    "scene_family": f"step1b_{role}_obstacle_from_{original['case_id']}"}
            goal = {**first, "pre_grasp_q": [q.tolist() for q in route], "joint_margin_collision_check_m": .002}
            file = f"{case['case_id']}.json"
            output_plan = {**plan, "case": case, "obstacles": [obstacle], "feasible": [goal], "detour": planner.detour}
            write_json(output / file, output_plan)
            selected.append({**case, "plan": file, "offline_feasible": True, "detour": planner.detour})
            parts[case["part"]] += 1
            print(case["case_id"], case["part"], planner.detour, flush=True)
            break
        attempted.append({"case": original["case_id"], "positions": len(centers)})
        if len(selected) == 3:
            break
    document = {**template(), "phase": "three_view_expert_v3", "asset": asset, "target_query": query,
                "asset_role": role, "total_steps": EPISODE_STEPS, "unified_lift_gate": asdict(LiftGateConfig()),
                "fixture_assets": {"blocker": str(blocker)}, "cases": selected}
    write_json(output / "collection.json", document)
    write_json(output / "plan_result.json", {"passed": len(selected) >= 2, "selected": selected, "attempted": attempted})
    return 0 if len(selected) >= 2 else 1


def prepare(output, role, workers, mode, only_angle=None):
    asset, part_meshes, family, query = infos(role)
    cfg = {**read(ROOT / "env_resolved.json"), "target_usd": asset}
    evidence = read(RUNS / "lab_school_03/result.json")
    output.mkdir(parents=True, exist_ok=True)
    config = output / "config.json"
    write_json(config, cfg)
    table = next(points for name, points in authored_meshes(cfg["school_usd"]) if name == "/Scene/Table/Top")
    table_z = float(table[:, 2].max() + cfg["school_position_m"][2])
    home = evidence["samples"][0]["right_rad"]
    total = 6200 + {"basket": 0, "mug": 200, "carton": 400}[role] + (100 if mode == "obstacles" else 0)
    if only_angle is not None:
        total += 70 if only_angle == 75 else 50
    rows = []

    def add(yaw, x, y, parts, fixtures=(), description=""):
        idx = len(rows)
        scene = f"step1b_{role}_{mode}_{yaw:+d}_{idx:02}"
        pose = place(asset, table_z, x, y, yaw)
        group = "B_obstacle" if fixtures else "C_start"
        rows.append([{"case_id": f"{group[0]}_{total+2*idx+i}", "seed": 77741+total+idx, "group": group,
                      "asset_role": role, "asset_family": family, "scene_family": scene, "split": "step1_dev",
                      "config_sha256": sha256(config), "part": part, "target_usd": asset, "target_pose_xyz_xyzw": pose,
                      "home_q": home, "fixtures": list(fixtures), "command_delay_substeps": 0, "yaw_deg": yaw,
                      "part_meshes": part_meshes, "grip_goal_m": .002, "layout_description": description,
                      **({"approach_width_m": .044} if part == BASKET_RIM else
                         {"approach_width_m": .052} if part == BASKET_HANDLE else
                         {"approach_width_m": .063} if role == "carton" else {})}
                     for i, part in enumerate(parts)])

    if mode == "obstacles":
        blocker = ROOT / "assets" / {"basket": "marker_box/00000", "mug": "sand3d_packaging/00156",
                                     "carton": "soda_can/00000"}[role] / "object.usdz"
        yaws = (0, -12, 12, 25, -25, 0, -16, 16, 5, -5)
        poses = ((.168, -.494), (.187, -.487), (.170, -.484), (.150, -.481), (.159, -.495),
                 (.170, -.478), (.190, -.498), (.180, -.495), (.151, -.502), (.182, -.477))
        bottom = np.min(np.concatenate([pts for _, pts in authored_meshes(blocker)]), axis=0)
        top = np.max(np.concatenate([pts for _, pts in authored_meshes(blocker)]), axis=0)
        print(role, "blocker dimensions", (top - bottom).tolist(), flush=True)
        for i, ((bx, by), yaw) in enumerate(zip(poses, yaws, strict=True)):
            fixture = [{"name": "blocker", "usd": str(blocker), "sha256": sha256(blocker),
                        "pose": place(blocker, table_z, bx, by, yaw / 2)}]
            part = PAIR_PARTS[role][i % 2]
            add(yaw, .184, -.385, [part], fixture, "verified blocked direct path")
    else:
        yaws = (0, -55, 55, -75, 75, -25, 25) if role == "basket" else (-55, 55, -75, 75) if role == "mug" else (65, -65, 85)
        centers = ((.178, -.416), (.204, -.410), (.149, -.420)) if role != "carton" else ((.180, -.445), (.200, -.442))
        if only_angle is not None:
            yaws = (only_angle,)
            centers = ((.224, -.442), (.228, -.416), (.204, -.444), (.236, -.435), (.165, -.451), (.199, -.475))
        for offset, (x, y) in enumerate(centers):
            for yaw in yaws:
                add(yaw, x, y, PAIR_PARTS[role], description=f"sorted paired center {offset}")

    plans = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(make_plan, cfg, evidence, case): case for pair in rows for case in pair}
        for future in as_completed(pending):
            case = pending[future]
            plan = future.result()
            filename = f"{case['case_id']}.json"
            write_json(output / filename, plan)
            plans[case["case_id"]] = {**case, "plan": filename, "offline_feasible": bool(plan["feasible"]),
                                      "detour": plan["detour"]}
            print(case["case_id"], case["yaw_deg"], case["part"], bool(plan["feasible"]), plan["collision_checks"], flush=True)
    available = [[plans[c["case_id"]] for c in pair] for pair in rows]
    good = [pair for pair in available if all(case["offline_feasible"] for case in pair)]
    selected = []
    if mode == "obstacles":
        selected = good[:3]
    else:
        for angle in yaws:
            if match := next((pair for pair in good if pair[0]["yaw_deg"] == angle), None):
                selected.append(match)
        selected = selected[:5 if role == "basket" else 4 if role == "mug" else 2]
    cases = [case for pair in selected for case in pair]
    document = {**template(), "phase": "three_view_expert_v3", "asset": asset, "target_query": query,
                "asset_role": role, "total_steps": EPISODE_STEPS, "unified_lift_gate": asdict(LiftGateConfig()),
                "fixture_assets": {f["name"]: f["usd"] for case in cases for f in case["fixtures"]}, "cases": cases}
    write_json(output / "collection.json", document)
    write_json(output / "plan_result.json", {"passed": bool(cases), "selected": selected, "available": available})


def replays(work, capture):
    manifest = read(work / "collection.json")
    originals = {row["case"]: row["dataset_receipt"] for row in read(capture / "result.json")["cases"]
                 if row.get("dataset_receipt")}
    cases, records, groups = [], [], defaultdict(list)
    for row in manifest["cases"]:
        case = row["case_id"]
        if case not in originals:
            continue
        receipt = originals[case]
        if sha256(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Accepted v3 trajectory changed")
        with h5py.File(receipt["path"], "r") as h5:
            steps = receipt["steps"]
            assert h5.attrs["schema"] == "g1d_v3_step_aligned"
            assert verify_three_view(h5) == steps + 1
            assert np.array_equal(h5["obs/control_step_id"][:], np.arange(steps+1))
            assert np.array_equal(h5["action/control_step_id"][:], np.arange(steps))
            phase = h5["phase"].asstr()[:]
            assert np.array_equal(phase != "lift", h5["train_action_valid"][:])
            assert np.allclose(h5["action/native_substep_command"][phase == "lift", :, 7:],
                               h5["action/native_command"][np.flatnonzero(phase == "lift")[0]-1, 7:], rtol=0, atol=1e-7)
            assert h5["obs/eef_pose"].shape == (steps+1, 7)
            for image in ("obs/rgb_left", "obs/left_wrist/rgb", "obs/right_wrist/rgb"):
                assert h5[image].shape == (steps+1, 224, 224, 3) and h5[image].compression == "gzip"
            p = json.loads(h5["meta/initial_prompt_json"].asstr()[()])
            initial_state = np.r_[h5["obs/qpos"][0], h5["obs/gripper_width"][0]]
        native_initial = read(capture / case / "expert_diagnostic.json")["target_initial_xyz_xyzw"]
        cases.append({**row, "replay_h5": {"path": receipt["path"], "sha256": receipt["sha256"]}})
        records.append({**receipt, "part": row["part"], "asset_family": row["asset_family"], "yaw_deg": row["yaw_deg"],
                        "scene_family": f"{row['scene_family']}_yaw_{row['yaw_deg']:+d}"})
        groups[(row["scene_family"], row["yaw_deg"])].append((row, p, initial_state, native_initial))
    expected_groups = Counter((row["scene_family"], row["yaw_deg"]) for row in manifest["cases"])
    required_pairs = sum(n == 2 for n in expected_groups.values())
    exact_pairs = 0
    for pair in groups.values():
        if len(pair) != 2:
            continue
        (a, pa, sa, ta), (b, pb, sb, tb) = pair
        translation_error = np.linalg.norm(np.asarray(ta[:3]) - tb[:3])
        rotation_error = (Rotation.from_quat(ta[3:]).inv() * Rotation.from_quat(tb[3:])).magnitude()
        if (pa["bbox_0"] == pb["bbox_0"] and pa["affordance_0"] != pb["affordance_0"]
                and np.allclose(sa, sb, rtol=0, atol=1e-5) and translation_error <= .001 and rotation_error <= .003
                and all(a[key] == b[key] for key in ("target_pose_xyz_xyzw", "home_q", "seed", "fixtures"))):
            exact_pairs += 1
    write_json(work / "episodes.json", {"episodes": records, "required_pairs": required_pairs,
                                        "bbox_and_initial_pose_verified_pairs": exact_pairs})
    write_json(work / "replay.json", {**manifest, "phase": "three_view_unified_replay", "cases": cases})
    if not cases:
        raise RuntimeError("All candidate captures failed physical validation")


def review(work, replay):
    manifest, episodes = read(work / "collection.json"), read(work / "episodes.json")
    rows = []
    for row in read(replay / "result.json")["cases"]:
        case = read(replay / row["case"] / "expert_diagnostic.json")
        alignment = case.get("replay_alignment")
        roll = case.get("policy_rollout", {})
        stability = roll.get("trials", [{}])[0].get("stability") if roll.get("trials") else None
        forbidden = case.get("forbidden_contact_pairs", {})
        passed = bool(roll.get("success") and roll.get("part_hit") and alignment and alignment["trigger_delta_s"] == 0
                      and alignment["max_native_setpoint_error"] <= 1e-6 and alignment["max_timestamp_error_s"] <= 2e-7
                      and alignment["max_arm_error_rad"] <= .005 and alignment["max_jaw_error_m"] <= .002
                      and not any(v for k, v in forbidden.items() if "/World/Obstacle" in k or "Blocker" in k))
        rows.append({"case": row["case"], "part": case.get("part"), "status": roll.get("status"),
                     "part_hit": roll.get("part_hit"), "passed": passed, "alignment": alignment,
                     "stability": stability, "failure": case.get("failure"), "forbidden_contact_pairs": forbidden})
    passed = (len(rows) == len(manifest["cases"]) and all(row["passed"] for row in rows)
              and episodes["bbox_and_initial_pose_verified_pairs"] == episodes["required_pairs"])
    report = {"passed": passed, "attempted": len(manifest["cases"]), "persisted": len(episodes["episodes"]),
              "replay_passes": sum(row["passed"] for row in rows), "part_hits": Counter(row["part"] for row in rows if row["part_hit"]),
              **{k:v for k,v in episodes.items() if k != "episodes"}, "cases": rows}
    write_json(work / "result.json", report)
    print({k: v for k, v in report.items() if k != "cases"}, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "prepare_obstacles", "replays", "review"))
    parser.add_argument("output", type=Path)
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument("--role", choices=tuple(PAIR_PARTS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--mode", default="pairs", choices=("pairs", "obstacles"))
    parser.add_argument("--only-angle", type=int)
    a = parser.parse_args()
    if a.stage == "prepare":
        prepare(a.output.resolve(), a.role, a.workers, a.mode, a.only_angle)
    elif a.stage == "prepare_obstacles":
        raise SystemExit(prepare_known_obstacles(a.output.resolve(), a.role, a.run.resolve()))
    elif a.stage == "replays":
        replays(a.output.resolve(), a.run.resolve())
    else:
        raise SystemExit(review(a.output.resolve(), a.run.resolve()))
