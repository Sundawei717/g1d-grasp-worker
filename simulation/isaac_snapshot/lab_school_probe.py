import argparse
import json
import math
import os
import re
from shutil import copytree
import sys
from tempfile import TemporaryDirectory
import traceback
from pathlib import Path
from types import SimpleNamespace


def run(config, output, samples, grasp=False, plan=None, admission=None):
    import numpy as np
    import omni.usd
    import torch
    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationCfg, SimulationContext
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
    from env.robot_manager.robot_config.g1d import get_robot_config
    from projects.g1d_rsi.camera_capture import capture_owned_cameras, native_reference
    from projects.g1d_rsi.cameras import create_cameras, load_config, robot_asset_root
    from projects.g1d_rsi.native_render import prepare_owned_rendering, render_owned_cameras
    from projects.g1d_rsi.sim60_legacy_bridge import bind_lab3_core_backend
    from projects.g1d_rsi.simulation_clock import read_simulation_clock

    device = f"cuda:{config['gpu']}"
    bind_lab3_core_backend(device)
    sim = SimulationContext(SimulationCfg(device=device, dt=1 / 600, render_interval=30))
    stage = omni.usd.get_context().get_stage()
    if not stage or stage.GetRootLayer() != sim.stage.GetRootLayer():
        raise RuntimeError("Kit and Lab stage differ")
    school = UsdGeom.Xform.Define(stage, "/World/School")
    school.GetPrim().GetReferences().AddReference(config["school_usd"])
    UsdGeom.XformCommonAPI(school).SetTranslate(Gf.Vec3d(*config["school_position_m"]))
    tabletop = stage.GetPrimAtPath("/World/School/Table/Top")
    backdrop = stage.GetPrimAtPath("/World/School/Background/Backdrop")
    if not tabletop or not backdrop:
        raise RuntimeError("Original tabletop or backdrop missing")
    target_root = UsdGeom.Xform.Define(stage, "/World/Target")
    target = UsdGeom.Xform.Define(stage, "/World/Target/Object").GetPrim()
    target.GetReferences().AddReference(config["target_usd"])
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    target_box = bounds.ComputeWorldBound(target).ComputeAlignedRange()
    table_box = bounds.ComputeWorldBound(tabletop).ComputeAlignedRange()
    if target_box.IsEmpty() or table_box.IsEmpty():
        raise RuntimeError("Missing tabletop or target geometry")
    table_center = table_box.GetMidpoint()
    target_center = target_box.GetMidpoint()
    UsdGeom.XformCommonAPI(target_root).SetTranslate(Gf.Vec3d(
        table_center[0] + 0.10 - target_center[0],
        table_center[1] - 0.12 - target_center[1],
        table_box.GetMax()[2] - target_box.GetMin()[2] + 0.002))

    robot_cfg = get_robot_config().replace(prim_path="/World/G1D")
    robot_cfg.init_state.pos = tuple(config["root_position_m"])
    yaw = math.radians(config["root_yaw_deg"]) / 2
    robot_cfg.init_state.rot = (0, 0, math.sin(yaw), math.cos(yaw))
    robot = Articulation(robot_cfg)
    if grasp:
        import carb
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from grasp_regions import UPPER_PART
        from rigid_grasp_probe import ContactObserver, TargetNotVisible, full_projection_prompt, pad_geometry, run_grasp, visible_prompt

        rigid_paths = [str(prim.GetPath()) for prim in Usd.PrimRange(target) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(rigid_paths) != 1:
            raise RuntimeError(f"Diagnostic target must have one rigid body: {rigid_paths}")
        target_asset = RigidObject(RigidObjectCfg(prim_path=rigid_paths[0]))
        obstacle_paths = []
        grouped_fixtures = {}
        if admission and admission.get("fixture_assets"):
            fixtures = [{"name": name, "usd": asset, "pose": [4.0 + 0.6 * index, 0, 0.5, 0, 0, 0, 1]}
                        for index, (name, asset) in enumerate(sorted(admission["fixture_assets"].items()))]
        else:
            fixtures = plan.get("obstacles", []) if plan else ()
        for index, obstacle in enumerate(fixtures):
            placement = UsdGeom.Xform.Define(stage, f"/World/Obstacle{index}")
            actor = UsdGeom.Xform.Define(stage, f"{placement.GetPath()}/Body")
            actor.GetPrim().GetReferences().AddReference(obstacle["usd"])
            x_value, y_value, z_value, quat_x, quat_y, quat_z, quat_w = obstacle["pose"]
            placement.AddTranslateOp().Set(Gf.Vec3d(x_value, y_value, z_value))
            placement.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(quat_w, quat_x, quat_y, quat_z))
            bodies = [prim for prim in Usd.PrimRange(actor.GetPrim()) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(bodies) != 1:
                raise RuntimeError("Expected one physical body for the configured blocker")
            UsdPhysics.RigidBodyAPI(bodies[0]).CreateKinematicEnabledAttr().Set(True)
            obstacle_paths.append(str(bodies[0].GetPath()))
            if admission and admission.get("fixture_assets"):
                grouped_fixtures[obstacle["name"]] = (
                    obstacle_paths[-1], RigidObject(RigidObjectCfg(prim_path=obstacle_paths[-1])), obstacle["pose"])
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        observer = ContactObserver(stage, robot_cfg.prim_path, rigid_paths[0], str(tabletop.GetPath()), obstacle_paths)
        pads = pad_geometry(stage, stage.GetPrimAtPath(robot_cfg.prim_path))
    sim.reset()
    if grasp:
        observer.bind_tensors(sim)
    adapter = SimpleNamespace(sim=SimpleNamespace(sim=sim))
    rendering = prepare_owned_rendering(adapter)
    root, resolution = robot_asset_root(stage, robot)
    camera_config, source = load_config(config["camera_config"])
    parents = camera_config["isaac"]["parent_paths"]
    for name, path in parents.items():
        parents[name] = f"{root}/{path.rsplit('/', 1)[-1]}"
    mounts = create_cameras(stage, root, camera_config)
    camera = mounts["cameras"]["head_left"]
    three_view = admission is not None and admission.get("three_view", False)
    capture_names = ("head_left", "left_wrist", "right_wrist") if three_view else ("head_left",)
    active_cameras = {name: mounts["cameras"][name] for name in capture_names}
    active_audits = {name: mounts["audit"]["cameras"][name] for name in capture_names}
    for name, another in mounts["cameras"].items():
        if name not in active_cameras:
            another.pause()
    left_audit = mounts["audit"]["cameras"]["head_left"]
    if grasp:
        dense = admission is not None and admission["phase"] in {"collection_preview", "collection_train"}
        record_v3 = admission is not None and admission["phase"] == "three_view_expert_v3"
        moving_target = admission is not None and admission["phase"] == "tracking_motion"
        policy = None
        prompt_mode = "gt_visible"
        if admission and admission["phase"] == "fit20_closed_loop":
            from policy_rollout import LoadedPolicy

            policy = LoadedPolicy(admission["policy"]["path"], device)
            prompt_mode = admission["prompt_mode"]
            lift_model = admission["lift_model"]
            lift_evidence = json.loads(Path(lift_model["evidence"]).read_text())
            if prompt_mode == "gt_full_projection":
                geometry = UsdGeom.XformCache(Usd.TimeCode.Default())
                rigid = stage.GetPrimAtPath(rigid_paths[0])
                to_local = geometry.GetLocalToWorldTransform(rigid).GetInverse()
                vertices = []
                for prim in Usd.PrimRange(rigid):
                    if (not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible"
                            or UsdGeom.Imageable(prim).ComputePurpose() not in {"default", "render"}):
                        continue
                    transform = geometry.GetLocalToWorldTransform(prim)
                    points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                    vertices.extend(tuple(to_local.Transform(transform.Transform(Gf.Vec3d(*point)))) for point in points)
                if not vertices:
                    raise RuntimeError("The target has no rendered mesh vertices for a full projection")
                local_vertices = np.array(vertices, dtype=float)
        three_view_steps = None
        unified_lift_gate = admission.get("unified_lift_gate") if admission else None
        if admission and admission["phase"] == "three_view_closed_loop":
            from three_view_rollout import ThreeViewPolicy
            from three_view_detector import WristDetector

            binding = admission["policy"]
            if admission.get("policy_architecture") == "v3_final" and admission.get("detector"):
                from v3_detector import V3WristDetector

                detector = V3WristDetector(device)
                if detector.config != admission["detector"]:
                    raise ValueError("V3 deployment detector differs from its frozen specification")
            else:
                detector = WristDetector(admission["detector"], device) if admission.get("detector") else None
            policy = ThreeViewPolicy(binding["path"], device, detector)
            three_view_steps = admission["total_steps"]
        if three_view_steps is not None or unified_lift_gate is not None:
            lift_model = admission["lift_model"]
            lift_evidence = json.loads(Path(lift_model["evidence"]).read_text())

        def clock():
            return read_simulation_clock(sim)

        def episode(directory, candidate, reset_target, verify_visibility):
            from time import perf_counter

            grasp_part = candidate["part"] if candidate else UPPER_PART
            if unified_lift_gate is not None:
                import random

                seed = candidate["case"]["seed"]
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
            robot.reset()
            joint_start = robot.data.default_joint_pos.clone()
            if candidate:
                arm_indices = [list(robot.joint_names).index(name) for name in candidate["right_joint_names"]]
                if len(set(arm_indices)) != 7:
                    raise RuntimeError("Expected seven right arm start coordinates")
                joint_start[0, arm_indices] = joint_start.new_tensor(candidate["home_q"])
            robot.write_joint_position_to_sim_index(position=joint_start)
            robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.clone())
            if reset_target:
                target_asset.reset()
                settled = candidate["target_pose_xyz_xyzw"].copy()
                settled[2] += 0.002
                target_asset.write_root_link_pose_to_sim_index(root_pose=torch.tensor([settled], device=device, dtype=torch.float32))
                target_asset.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=device))
            active_fixtures = {entry["name"]: entry for entry in candidate.get("obstacles", [])} if candidate else {}
            if grouped_fixtures:
                for fixture_name, (_, rigid, parked) in grouped_fixtures.items():
                    placement = active_fixtures[fixture_name]["pose"] if fixture_name in active_fixtures else parked
                    rigid.reset()
                    rigid.write_root_link_pose_to_sim_index(root_pose=torch.tensor([placement], device=device, dtype=torch.float32))
                    rigid.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=device))
            previous = None
            frame_log = {}
            truth = {}
            label_sources = {}
            anchor_local = None
            policy_frames = 0
            render_product = camera.get_render_product_path()
            if verify_visibility:
                import omni.replicator.core as rep

                label_sources["segmentation"] = rep.AnnotatorRegistry.get_annotator(
                    "instance_id_segmentation", init_params={"colorize": False})
                label_sources["depth"] = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
                for annotator in label_sources.values():
                    annotator.attach([render_product])

            def capture(name=None):
                nonlocal previous
                start = perf_counter()
                for _ in range(4):
                    receipt = capture_owned_cameras(active_cameras, expected_sim_time=clock()["current_time"],
                                                    render=lambda: render_owned_cameras(adapter),
                                                    sim_time=lambda: clock()["current_time"], previous=previous,
                                                    resolutions={key: tuple(value["resolution"])
                                                                 for key, value in active_audits.items()})
                    if (name != "expert_initial.png" and not dense and not moving_target and policy is None) or not label_sources:
                        break
                    stamp = receipt["frames"]["head_left"]["reference_time"]
                    blob = {key: annotator.get_data() for key, annotator in label_sources.items()}
                    seg = blob.get("segmentation")
                    depth = blob.get("depth")
                    if (isinstance(seg, dict) and "info" in seg and np.asarray(seg.get("data")).size > 1
                            and np.asarray(depth.get("data") if isinstance(depth, dict) else depth).size > 1
                            and native_reference(camera)["reference_time"] == stamp):
                        truth.update(blob)
                        break
                else:
                    raise RuntimeError("Same-frame RTX truth annotators not ready")
                previous = receipt["frames"]
                rgb = receipt["images"]["head_left"][..., :3].copy()
                frame = previous["head_left"]
                if name is not None:
                    frame_log[name] = frame
                    Image.fromarray(rgb).save(directory / name)
                    if three_view:
                        for view in capture_names[1:]:
                            wrist_name = f"{Path(name).stem}_{view}.png"
                            frame_log[wrist_name] = previous[view]
                            Image.fromarray(receipt["images"][view][..., :3]).save(directory / wrist_name)
                    (directory / "expert_camera_frames.json").write_text(json.dumps(frame_log, indent=2), encoding="utf-8")
                if three_view:
                    frame = {**frame, "wrist_views": {view: (receipt["images"][view][..., :3], previous[view])
                                                    for view in capture_names[1:]}}
                if unified_lift_gate is not None:
                    frame = {**frame, "acquisition_started_wall_s": start, "acquisition_finished_wall_s": perf_counter()}
                return rgb, frame

            def visibility(initial):
                nonlocal anchor_local
                from PIL import ImageDraw

                index = list(robot.body_names).index("head_link")
                head = robot.data.body_link_pose_w[0, index].tolist()
                fixture_paths = []
                for name, entry in active_fixtures.items():
                    if name not in grouped_fixtures:
                        continue
                    path, rigid, _ = grouped_fixtures[name]
                    rigid.update(sim.get_physics_dt())
                    actual = np.asarray(rigid.data.root_link_pose_w[0].tolist())
                    expected = np.asarray(entry["pose"])
                    angle = (Rotation.from_quat(actual[3:]).inv() * Rotation.from_quat(expected[3:])).magnitude()
                    if np.linalg.norm(actual[:3] - expected[:3]) > 0.001 or angle > 0.003:
                        raise RuntimeError(f"Fixture {name} did not bind to the frozen pose")
                    fixture_paths.append(path)
                prompt = visible_prompt(head, left_audit, initial, target_box, rigid_paths[0], truth["segmentation"],
                                        truth["depth"], visible_fixtures=fixture_paths, grasp_part=grasp_part,
                                        include_masks=dense or moving_target,
                                        part_meshes=candidate["case"].get("part_meshes") if candidate and candidate.get("case") else None)
                if candidate and candidate.get("initial_head_prompt"):
                    prompt.update(candidate["initial_head_prompt"])
                anchor_local = prompt["part_hit_local_xyz_m"]
                initial_image = Image.open(directory / "expert_initial.png").convert("RGB")
                draw = ImageDraw.Draw(initial_image)
                width, height = left_audit["resolution"]
                draw.rectangle([coord * (width - 1 if index % 2 == 0 else height - 1)
                                for index, coord in enumerate(prompt["bbox_0"])], outline="yellow", width=2)
                pixel_x, pixel_y = np.asarray(prompt["affordance_0"]) * [width - 1, height - 1]
                draw.ellipse((pixel_x - 3, pixel_y - 3, pixel_x + 3, pixel_y + 3), fill="red")
                initial_image.save(directory / "expert_prompt_check.png")
                if admission and admission["phase"] in {"grouped_expert_100", "three_view_closed_loop",
                                                        "three_view_unified_replay", "three_view_expert_v3"}:
                    for annotator in label_sources.values():
                        annotator.detach([render_product])
                    label_sources.clear()
                return prompt

            def dense_frame():
                nonlocal policy_frames
                name = None
                if policy is not None:
                    policy_frames += 1
                    if policy_frames % 3 == 0:
                        name = f"policy_observation_{policy_frames:03d}.png"
                rgb, frame = capture(name)
                head_id = list(robot.body_names).index("head_link")
                head = robot.data.body_link_pose_w[0, head_id].tolist()
                target_pose = np.asarray(target_asset.data.root_link_pose_w[0].tolist())
                try:
                    prompt = visible_prompt(head, left_audit, target_pose, target_box, rigid_paths[0],
                                            truth["segmentation"], truth["depth"], grasp_part=grasp_part,
                                            include_masks=dense or moving_target, anchor_local=anchor_local,
                                            part_meshes=candidate["case"].get("part_meshes") if candidate and candidate.get("case") else None)
                except TargetNotVisible:
                    prompt = None
                if prompt_mode == "gt_full_projection" and not (prompt and prompt.get("prompt_valid")):
                    prompt = full_projection_prompt(head, left_audit, target_pose, local_vertices, anchor_local)
                return rgb, frame, prompt

            def lift_guard(q, wrist, fingers):
                from vertical_lift import checked_lift
                from unified_grasp import LIFT_HEIGHT_M

                return checked_lift(config, lift_model["urdf"], lift_evidence, tuple(active_fixtures.values()),
                                    q, wrist, fingers,
                                    height_m=LIFT_HEIGHT_M if unified_lift_gate is not None else 0.030)

            def fixture_feedback():
                poses = {}
                for name in active_fixtures:
                    _, rigid, _ = grouped_fixtures[name]
                    rigid.update(sim.get_physics_dt())
                    poses[name] = np.asarray(rigid.data.root_link_pose_w[0].tolist())
                return poses

            try:
                if moving_target:
                    from tracking_motion import record_motion

                    try:
                        return record_motion(directory, left_audit, candidate["case"], candidate["force_xy"],
                                             robot, target_asset, sim, capture, visibility, dense_frame)
                    except Exception as error:
                        return {"passed": False, "failure": f"{type(error).__name__}: {error}"}
                replay = None
                if candidate and candidate.get("replay_h5"):
                    from dataset_alignment import load_replay

                    replay = load_replay(candidate["replay_h5"], candidate["case"], candidate["right_joint_names"], left_audit)
                return run_grasp(robot=robot, target=target_asset, sim=sim, observer=observer, pads=pads,
                                 bounds=target_box, output=directory, clock=clock, capture=capture, plan=candidate,
                                 asset_path=config["target_usd"], visibility=visibility if verify_visibility else None,
                                 dense_camera=left_audit if dense else None,
                                 dense_frame=dense_frame if dense or policy is not None else None,
                                 replay=replay, policy=policy,
                                 lift_guard=lift_guard if policy is not None or unified_lift_gate is not None else None,
                                 prompt_mode=prompt_mode,
                                 wrist_cameras={name: active_audits[name] for name in capture_names[1:]} if three_view else None,
                                 target_query=admission.get("target_query") if three_view else None,
                                 three_view_steps=three_view_steps,
                                 unified_lift_gate=unified_lift_gate,
                                 record_v3_cameras=active_audits if record_v3 else None,
                                 fixture_feedback=fixture_feedback if unified_lift_gate is not None else None)
            finally:
                for annotator in label_sources.values():
                    annotator.detach([render_product])

        if admission:
            cases = []
            aggregate = {"scope": "single-asset fixed-layout right-arm trial batch", "cases": cases}
            phase = admission["phase"]
            aggregate.update(phase=phase, asset_usd=config["target_usd"], required=admission["required"])
            metadata = admission.get("case_metadata", {})
            for case_id, candidate in admission["cases"]:
                directory = output / case_id
                directory.mkdir()
                if candidate is None:
                    cases.append({"case": case_id, "status": "offline_reject",
                                  "failure": metadata.get(case_id, {}).get("offline_failure")})
                elif moving_target:
                    trial = episode(directory, candidate, reset_target=True, verify_visibility=True)
                    cases.append({"case": case_id, "status": "recorded" if trial["passed"] else "motion_failed",
                                  "motion_receipt": trial})
                else:
                    trial = episode(directory, candidate, reset_target=True, verify_visibility=True)
                    if three_view and dense and candidate.get("replay_h5") and not trial["diagnostic_passed"]:
                        fallback = directory / "expert_rerun"
                        fallback.mkdir()
                        trial = episode(fallback, {key: value for key, value in candidate.items() if key != "replay_h5"},
                                        reset_target=True, verify_visibility=True)
                    outcome = ("strict_success" if trial["diagnostic_passed"] else
                               "physical_fail" if trial["commands"] else "preflight_fail")
                    cases.append({"case": case_id, "status": outcome,
                                  "failure": trial.get("failure"), "strict_score": trial.get("strict_score"),
                                  "prompt": trial.get("initial_prompt")})
                    if dense or record_v3:
                        cases[-1]["dataset_receipt"] = trial.get("dataset_receipt")
                    if phase == "collection_replay":
                        cases[-1]["replay_alignment"] = trial.get("replay_alignment")
                    if phase in {"fit20_closed_loop", "three_view_closed_loop", "three_view_unified_replay",
                                 "three_view_expert_v3"}:
                        rollout = trial.get("policy_rollout", {})
                        cases[-1]["policy_result"] = {key: value for key, value in rollout.items() if key != "decisions"}
                if case_id in metadata:
                    cases[-1].update(group=metadata[case_id]["group"], scene_family=metadata[case_id]["scene_family"])
                print(json.dumps(cases[-1]), flush=True)
                (output / "batch_progress.json").write_text(json.dumps(aggregate, indent=2))
            aggregate["attempted"] = sum(case["status"] in {"physical_fail", "strict_success"} for case in cases)
            aggregate["strict_success"] = sum(case["status"] == "strict_success" for case in cases)
            if moving_target:
                aggregate["passed"] = all(row["status"] == "recorded" for row in cases)
                aggregate["recorded"] = sum(row["status"] == "recorded" for row in cases)
                (output / "result.json").write_text(json.dumps(aggregate, indent=2))
                return
            if phase in {"grouped_expert_100", "collection_preview", "collection_replay", "collection_train",
                          "fit20_closed_loop", "three_view_closed_loop", "three_view_unified_replay",
                          "three_view_expert_v3"}:
                aggregate["passed"] = len(cases) == len(admission["cases"])
                aggregate["scope"] = "completed physical shard; global acceptance is evaluated separately"
            else:
                aggregate["passed"] = aggregate["strict_success"] >= admission["required"]
            (output / "result.json").write_text(json.dumps(aggregate, indent=2))
            if not aggregate["passed"]:
                raise RuntimeError(f"{phase} gate: {aggregate['strict_success']}/{len(cases)} configured cases")
            return
        trial = episode(output, plan, reset_target=plan is not None, verify_visibility=plan is not None)
        (output / "result.json").write_text(json.dumps({"passed": trial["diagnostic_passed"]}), encoding="utf-8")
        if not trial["diagnostic_passed"]:
            raise RuntimeError(trial.get("failure", str(trial.get("strict_score"))))
        return
    names = list(robot.joint_names)
    right_ids = [names.index(f"right_{name}_joint") for name in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
    finger_ids = [names.index(f"right_dex1_finger_joint_{side}") for side in (1, 2)]
    links = list(robot.body_names)
    ee_name = "right_dex1_base_link"
    if ee_name not in links or not robot.is_fixed_base:
        raise RuntimeError("Expected fixed G1D with physical right gripper link")
    rigid_points = [str(prim.GetPath()) for prim in Usd.PrimRange(target) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(rigid_points) != 1:
        raise RuntimeError(f"Diagnostic target must have one rigid body: {rigid_points}")
    joints = robot.data.default_joint_pos.clone()
    robot.write_joint_position_to_sim_index(position=joints)
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.clone())
    robot.set_joint_position_target_index(target=joints)
    left_ids = [index for index, name in enumerate(names) if name.startswith("left_")]
    head_id = links.index("head_link")
    pad_ids = [links.index(f"right_dex1_finger_link_{finger}_tip") for finger in (1, 2)]
    ee_id = links.index(ee_name)
    for _ in range(180):
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())
    head_initial = robot.data.body_link_pose_w[0, head_id].tolist()
    camera_offset = np.asarray(left_audit["T_parent_camera_usd_stage_units"])[:3, 3]
    camera_z = head_initial[2] + Rotation.from_quat(head_initial[3:]).apply(camera_offset)[2]
    if not 1.1 < camera_z < 1.4:
        raise RuntimeError(f"Mounted left eye is not at the historical school height: {camera_z}")
    frames = []
    previous = None
    report = {"camera_source": source, "head_left": left_audit, "rendering": rendering, "robot_root": resolution,
              "school_usd": config["school_usd"], "target_usd": config["target_usd"], "target_rigid_prim": rigid_points[0],
              "joint_names": names, "right_arm": [names[index] for index in right_ids],
              "right_fingers": [names[index] for index in finger_ids], "right_ee_link": ee_name,
              "fixed_base": bool(robot.is_fixed_base), "right_arm_limits_rad": robot.data.joint_pos_limits[0, right_ids].tolist(),
              "right_finger_limits_m": robot.data.joint_pos_limits[0, finger_ids].tolist(), "samples": frames,
              "pad_width_convention": "historical nominal 0.095 m gap at open; measured displacement from two prismatic positions",
              "scope": "bounded single-arm no-grasp camera/command/observation smoke"}

    def clock():
        return read_simulation_clock(sim)

    try:
        for index in range(samples):
            joints[0, right_ids[-1]] = robot.data.default_joint_pos[0, right_ids[-1]] + min(0.15, 0.025 * index)
            joints[0, finger_ids] = min(0.0245, -0.02 + max(0, index - 1) * 0.0032)
            robot.set_joint_position_target_index(target=joints)
            start = clock()
            for _ in range(30):
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(sim.get_physics_dt())
            stamp = clock()
            if stamp["current_time_step_index"] - start["current_time_step_index"] != 30:
                raise RuntimeError(f"Physics substep count mismatch: {start}, {stamp}")
            if not math.isclose(stamp["current_time"] - start["current_time"], 0.05, abs_tol=1e-5):
                raise RuntimeError("Physics did not advance one 20 Hz control interval")
            state = robot.data.joint_pos.detach().cpu()
            fingers = state[0, finger_ids].tolist()
            head_pose = robot.data.body_link_pose_w[0, head_id].tolist()
            pad_pos = [robot.data.body_link_pose_w[0, pad_id, :3].tolist() for pad_id in pad_ids]
            frame = {"clock": stamp, "right_rad": state[0, right_ids].tolist(),
                     "right_command_rad": joints[0, right_ids].tolist(),
                     "right_fingers_m": fingers, "right_fingers_command_m": joints[0, finger_ids].tolist(),
                     "pad_gap_estimate_m": 0.095 - sum(value + 0.02 for value in fingers),
                     "pad_link_origin_distance_m": math.dist(*pad_pos),
                     "right_ee_xyz_xyzw": robot.data.body_link_pose_w[0, ee_id].tolist(),
                     "left_rad_or_m": state[0, left_ids].tolist(),
                     "root_w": robot.data.root_pos_w[0].tolist(),
                     "head_pose_w_xyzw": head_pose}
            camera_world = np.eye(4)
            camera_world[:3, :3] = Rotation.from_quat(head_pose[3:]).as_matrix()
            camera_world[:3, 3] = head_pose[:3]
            frame["T_world_camera_usd"] = (camera_world @ left_audit["T_parent_camera_usd_stage_units"]).tolist()
            frames.append(frame)
            receipt = capture_owned_cameras({"head_left": camera}, expected_sim_time=stamp["current_time"],
                                            render=lambda: render_owned_cameras(adapter),
                                            sim_time=lambda: clock()["current_time"], previous=previous,
                                            resolutions={"head_left": tuple(left_audit["resolution"])})
            previous = receipt["frames"]
            frame["head_left"] = previous["head_left"]
            frame["render_passes"] = receipt["render_passes"]
            rgb = np.ascontiguousarray(receipt["images"]["head_left"][..., :3])
            if float(rgb.std(axis=(0, 1)).mean()) <= 1:
                raise RuntimeError("RGB is spatially blank")
            Image.fromarray(rgb).save(output / f"head_left_{index:02d}.png")
            print(f"aligned frame {index}: {stamp['current_time']:.6f} s, passes={frame['render_passes']}", flush=True)
        report["right_wrist_yaw_displacement_rad"] = frames[-1]["right_rad"][-1] - frames[0]["right_rad"][-1]
        report["base_translation_drift_m"] = max(math.dist(frames[0]["root_w"], sample["root_w"]) for sample in frames)
        first_head = frames[0]["head_pose_w_xyzw"]
        report["head_translation_drift_m"] = max(math.dist(first_head[:3], sample["head_pose_w_xyzw"][:3]) for sample in frames)
        first_rotation = Rotation.from_quat(first_head[3:])
        report["head_rotation_drift_rad"] = max((first_rotation.inv() * Rotation.from_quat(sample["head_pose_w_xyzw"][3:])).magnitude()
                                                for sample in frames)
        report["max_left_joint_drift_rad_or_m"] = max(abs(value - origin) for sample in frames
            for value, origin in zip(sample["left_rad_or_m"], frames[0]["left_rad_or_m"]))
        report["observed_gap_change_m"] = frames[0]["pad_gap_estimate_m"] - frames[-1]["pad_gap_estimate_m"]
        report["pad_link_origin_distance_change_m"] = frames[0]["pad_link_origin_distance_m"] - frames[-1]["pad_link_origin_distance_m"]
        if (report["right_wrist_yaw_displacement_rad"] < 0.02 or report["base_translation_drift_m"] > 0.001
                or report["head_translation_drift_m"] > 0.002 or report["head_rotation_drift_rad"] > 0.005
                or report["max_left_joint_drift_rad_or_m"] > 0.005):
            raise RuntimeError("Right wrist motion or base/head stability failed")
        if (report["observed_gap_change_m"] < 0.01 or
                abs(report["observed_gap_change_m"] - report["pad_link_origin_distance_change_m"]) > 0.002):
            raise RuntimeError("Right jaw width did not match measured pad displacement")
        if not torch.isfinite(robot.data.joint_pos).all():
            raise RuntimeError("Nonfinite robot position")
        report["passed"] = True
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=18, choices=range(6, 21))
    parser.add_argument("--grasp-after", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--admission-manifest", type=Path)
    parser.add_argument("--passed-smoke", type=Path)
    parser.add_argument("--passed-admission", type=Path)
    parser.add_argument("--grouped-manifest", type=Path)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.gpu is not None:
        config["gpu"] = args.gpu
    if args.grasp_after and json.loads(args.grasp_after.read_text(encoding="utf-8")).get("passed") is not True:
        raise SystemExit("Refusing physical grasp diagnostic until the Lab school probe passes")
    if (args.plan or args.admission_manifest or args.grouped_manifest) and not args.grasp_after:
        raise SystemExit("Planned physics command requires the passed Lab receipt")
    plan = load_plan(config, args.plan, args.candidate_index) if args.plan else None
    if args.plan and plan is None:
        raise SystemExit("No accepted offline plan at candidate index")
    admission = None
    if args.grouped_manifest:
        if args.admission_manifest or plan:
            raise SystemExit("Grouped expert batch must run alone")
        admission = load_grouped(config, args.grouped_manifest)
    if args.admission_manifest:
        if plan or not args.passed_smoke:
            raise SystemExit("Asset batch requires one earlier strict smoke success and no standalone plan")
        smoke = json.loads(args.passed_smoke.read_text())
        if not smoke.get("diagnostic_passed") or (smoke.get("strict_score") or {}).get("success") is not True:
            raise SystemExit("No strict single-episode pass; refusing 10-case asset batch")
        manifest = json.loads(args.admission_manifest.read_text())
        phase = manifest.get("phase", "asset_admission_10")
        if phase not in {"asset_admission_10", "simple_expert_20"}:
            raise SystemExit("Unexpected physical batch phase")
        if phase == "simple_expert_20":
            if not args.passed_admission:
                raise SystemExit("20-case expert gate requires the completed 10-case asset admission")
            asset_gate = json.loads(args.passed_admission.read_text())
            if asset_gate.get("passed") is not True or asset_gate.get("strict_success") != 10 or len(asset_gate["cases"]) != 10:
                raise SystemExit("Asset admission receipt is incompatible or unpassed")
            if asset_gate.get("phase") != "asset_admission_10" or asset_gate.get("asset_usd") != config["target_usd"]:
                for case in asset_gate["cases"]:
                    case_path = args.passed_admission.parent / case["case"] / "expert_diagnostic.json"
                    bound = json.loads(case_path.read_text())
                    if (case["status"] != "strict_success" or (bound.get("strict_score") or {}).get("success") is not True
                            or bound.get("asset_usd") != config["target_usd"] or not bound.get("initial_prompt")):
                        raise SystemExit("Unbound per-episode asset admission evidence")
        smoke_asset = smoke.get("asset_usd")
        if smoke_asset is None and smoke.get("offline_plan"):
            previous = smoke["offline_plan"]
            previous_candidate = load_plan(config, Path(previous["path"]), previous["index"])
            if (previous_candidate and previous_candidate["candidate"] == previous["candidate"]
                    and smoke["target_asset"] == "/".join(Path(config["target_usd"]).parts[-3:-1])):
                smoke_asset = config["target_usd"]
        if manifest["asset"] != config["target_usd"] or smoke_asset != config["target_usd"]:
            raise SystemExit("Batch, smoke and simulator asset do not match")
        ids = [case["case_id"] for case in manifest["cases"]]
        count = 20 if phase == "simple_expert_20" else 10
        if (len(ids) != count or len(set(ids)) != count
                or any(not name.startswith("pose_") or not name.replace("_", "").isalnum() for name in ids)):
            raise SystemExit(f"Expected {count} distinct immutable cases")
        if any(Path(case["plan"]).name != case["plan"] for case in manifest["cases"]):
            raise SystemExit("Case path leaves the manifest directory")
        admission = {"phase": phase, "required": 18 if phase == "simple_expert_20" else 10,
                     "cases": [(case["case_id"], load_plan(config, args.admission_manifest.parent / case["plan"],
                                                          case.get("candidate_index", 0)))
                               for case in manifest["cases"]]}
    args.output.mkdir(parents=True, exist_ok=False)
    from projects.g1d_rsi.sim60_legacy_bridge import install_import_bridge
    from isaaclab.app import AppLauncher

    install_import_bridge()
    transient_kit = None
    kit_root = args.output / "kit"
    if os.environ.get("G1D_RUNTIME_DIR") or re.fullmatch(r"step1_[a-z0-9_]+|three_view_step0_[a-z0-9_]+|three_view_scale_[0-9]+_w[0-9]+_(carton|can)|"
                    r"three_view_time25_[0-9]+_(baseline|candidate|test)_[ab]_[a-z_]+_[0-9]+", args.output.name):
        transient_kit = TemporaryDirectory(prefix="g1d_three_view_kit_", dir=os.environ.get("G1D_RUNTIME_DIR", "/dev/shm"))
        kit_root = Path(transient_kit.name)
    graphics = os.environ.get("G1D_RENDER_GPU")
    graphics_arg = f" --/renderer/activeGpu={int(graphics)}" if graphics is not None else ""
    launcher = AppLauncher(headless=True, enable_cameras=True, device=f"cuda:{config['gpu']}",
                           experience=config["experience"],
                           kit_args=f"--portable-root {kit_root} --/renderer/multiGpu/autoEnable=false "
                           "--/renderer/multiGpu/enabled=false --/rtx/hydra/supportMultiTickRate=true "
                           "--/exts/omni.replicator.core/Orchestrator/enabled=false "
                           "--/persistent/omni/replicator/captureOnPlay=true "
                           f"--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera{graphics_arg}")
    status = 0
    try:
        run(config, args.output, args.samples, grasp=bool(args.grasp_after), plan=plan, admission=admission)
    except Exception:
        status = 1
        failure = traceback.format_exc()
        (args.output / "error.txt").write_text(failure, encoding="utf-8")
        print(failure, file=sys.stderr, flush=True)
    finally:
        launcher.app.close(exit_code=status)
        if transient_kit:
            logs = kit_root / "logs"
            if logs.is_dir():
                copytree(logs, args.output / "kit" / "logs")
            if args.output.name.startswith("step3b_01_"):
                cache = kit_root / "cache"
                size = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file() and not p.is_symlink())
                (args.output / "kit_cache_before_cleanup.bytes").write_text(str(size))
            transient_kit.cleanup()
    return status


def load_plan(config, path, index):
    from hashlib import sha256
    from grasp_regions import PARTS, UPPER_PART

    receipt = json.loads(path.read_text())
    for asset in ("school", "target"):
        actual = sha256(Path(config[f"{asset}_usd"]).read_bytes()).hexdigest()
        if receipt[f"{asset}_sha256"] != actual:
            raise SystemExit(f"Planner receipt {asset} asset differs from simulation asset")
    if not 0 <= index < len(receipt["feasible"]):
        return None
    obstacles = receipt.get("obstacles", [])
    if any(sha256(Path(item["usd"]).read_bytes()).hexdigest() != item["sha256"] for item in obstacles):
        raise SystemExit("A planned obstacle changed before replay")
    case = receipt.get("case")
    part = receipt.get("part", case["part"] if case else UPPER_PART)
    if part not in PARTS or (case and case["part"] != part):
        raise SystemExit("Grasp part is inconsistent with the planned scene")
    return {**receipt["feasible"][index], "home_q": receipt["home_q"], "part": part,
            "right_joint_names": receipt["right_joint_names"], "target_pose_xyz_xyzw": receipt["target_pose_xyz_xyzw"],
            "source": str(path.resolve()), "candidate_index": index, "obstacles": obstacles, "case": case}


def load_grouped(config, path):
    from hashlib import sha256

    manifest = json.loads(path.read_text())
    phase = manifest["phase"]
    phases = {"grouped_expert_100", "collection_preview", "collection_replay", "collection_train", "fit20_closed_loop",
              "tracking_motion", "three_view_closed_loop", "three_view_unified_replay", "three_view_expert_v3"}
    if phase not in phases or manifest["asset"] != config["target_usd"]:
        raise SystemExit("Grouped shard has wrong asset or phase")
    if set(manifest["gates"]) != {"simple", "blocker", "carton", "can"}:
        raise SystemExit("Grouped shard lacks prerequisite receipts")
    for binding in manifest["gates"].values():
        gate = Path(binding["path"])
        if sha256(gate.read_bytes()).hexdigest() != binding["sha256"] or json.loads(gate.read_text()).get("passed") is not True:
            raise SystemExit("Grouped prerequisite receipt failed or changed")
    cases = manifest["cases"]
    if phase in {"collection_preview", "collection_replay"}:
        from prepare_collection_preview import verify_collection_inputs

        verify_collection_inputs(manifest["collection_gate"], cases)
    elif phase in {"collection_train", "fit20_closed_loop"}:
        from prepare_train_bootstrap import verify_train_inputs

        verify_train_inputs(manifest["training_gates"], cases)
        if phase == "fit20_closed_loop":
            from fit20_eval import verify_model

            verify_model(manifest["policy"])
    ids = [case["case_id"] for case in cases]
    max_cases = 24 if phase == "three_view_closed_loop" else 17
    tier2 = phase in {"three_view_expert_v3", "three_view_unified_replay"} and all(
        case.get("split") == "tier2_new" for case in cases)
    id_pattern = r"[ABC]_[0-9]{6}" if tier2 else r"[ABC]_[0-9]{3,4}"
    if not 1 <= len(ids) <= max_cases or len(set(ids)) != len(ids) or any(not re.fullmatch(id_pattern, key) for key in ids):
        raise SystemExit("Bad grouped case IDs")
    fixture_assets = manifest["fixture_assets"]
    if not set(fixture_assets) <= {"neighbor", "blocker"}:
        raise SystemExit("Unexpected grouped fixture slot")
    plans = []
    for case in cases:
        if case["plan"] != f"{case['case_id']}.json" or case["target_usd"] != config["target_usd"]:
            raise SystemExit("Grouped plan is not bound to the frozen case")
        candidate = load_plan(config, path.parent / case["plan"], 0)
        if bool(candidate) != case["offline_feasible"]:
            raise SystemExit("Grouped offline feasibility flag mismatches its plan")
        if candidate:
            case_fields = ("case_id", "seed", "target_pose_xyz_xyzw", "home_q", "fixtures", "group", "part")
            if "lift_height_m" in case:
                case_fields += ("lift_height_m",)
            if phase != "grouped_expert_100":
                case_fields += ("asset_family", "scene_family", "split", "config_sha256")
            if phase in {"collection_train", "fit20_closed_loop"}:
                case_fields += ("command_delay_substeps",)
            if any(candidate["case"].get(key) != case[key] for key in case_fields):
                raise SystemExit("Grouped case definition changed after planning")
            if any(fixture_assets.get(entry["name"]) != entry["usd"] for entry in candidate["obstacles"]):
                raise SystemExit("Grouped fixture asset does not match checked plan")
            if phase in {"collection_replay", "three_view_unified_replay"} or (phase == "three_view_expert_v3"
                                                and case.get("replay_h5", {}).get("rerecord")) or (phase == "collection_train" and manifest.get("three_view")
                                                and "replay_h5" in case):
                candidate["replay_h5"] = case["replay_h5"]
            if phase == "tracking_motion":
                force = case["force_xy"]
                if sha256((path.parent / case["plan"]).read_bytes()).hexdigest() != case["source_plan_sha256"]:
                    raise SystemExit("The motion scenario changed after the offline freeze")
                if force not in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                    raise SystemExit("Invalid tabletop force direction")
                candidate["force_xy"] = force
            if phase == "three_view_closed_loop":
                candidate["initial_head_prompt"] = manifest.get("initial_head_prompts", {}).get(case["case_id"])
        plans.append((case["case_id"], candidate))
    result = {"phase": phase, "required": 0, "cases": plans,
              "fixture_assets": fixture_assets, "case_metadata": {case["case_id"]: case for case in cases}}
    if manifest.get("three_view"):
        if not manifest.get("target_query"):
            raise ValueError("Three-view grouped tests need a target query")
        result.update(three_view=True, target_query=manifest["target_query"])
    policy_architecture = None
    if phase == "three_view_closed_loop":
        from three_view_eval import verify_policy

        policy_cfg = verify_policy(manifest["policy"])
        policy_architecture = policy_cfg.get("architecture")
        if not isinstance(manifest.get("total_steps"), int) or manifest["total_steps"] <= 0:
            raise ValueError("Three-view evaluation requires a positive shared total time limit")
        result.update(policy=manifest["policy"], policy_architecture=policy_architecture, lift_model=manifest["lift_model"],
                      detector=manifest.get("detector"), total_steps=manifest["total_steps"])
    if phase == "three_view_unified_replay" or "unified_lift_gate" in manifest or policy_architecture == "v3_final":
        from dataclasses import asdict
        from unified_grasp import EPISODE_STEPS, LiftGateConfig

        if manifest["total_steps"] != EPISODE_STEPS:
            raise ValueError("Unified execution reserves exactly 300 total control steps")
        result.update(unified_lift_gate=asdict(LiftGateConfig()), lift_model=manifest["lift_model"])
    if phase == "fit20_closed_loop":
        from policy_rollout import PROMPT_MODES

        prompt_mode = manifest.get("prompt_mode", "gt_visible")
        if prompt_mode not in PROMPT_MODES:
            raise SystemExit(f"Unknown fit20 prompt mode: {prompt_mode}")
        result.update(policy=manifest["policy"], lift_model=manifest["lift_model"], prompt_mode=prompt_mode)
    return result


if __name__ == "__main__":
    sys.exit(main())
