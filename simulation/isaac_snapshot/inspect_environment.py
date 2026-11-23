import argparse
import importlib.metadata
import json
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--physics-readback", action="store_true")
    parser.add_argument("--replicator-fence", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.argv = [sys.argv[0], "--portable-root", str(output / "kit"), "--/renderer/multiGpu/autoEnable=false"]
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "active_gpu": config["gpu"], "multi_gpu": False},
                        experience=config["experience"])
    failed = False
    try:
        inspect(app, config, output, args.physics_readback, args.replicator_fence)
    except Exception:
        failed = True
        failure = traceback.format_exc()
        (output / "error.txt").write_text(failure, encoding="utf-8")
        print(failure, file=sys.stderr, flush=True)
    finally:
        app.close(exit_code=int(failed))
    return int(failed)


def inspect(app, config, output, physics_readback, replicator_fence):
    import numpy as np
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from PIL import Image, ImageDraw
    from projects.g1d_rsi.cameras import create_cameras, load_config
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    context = omni.usd.get_context()
    if context.get_stage() is None:
        context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    school = UsdGeom.Xform.Define(stage, "/World/School")
    school.GetPrim().GetReferences().AddReference(config["school_usd"])
    UsdGeom.XformCommonAPI(school).SetTranslate(Gf.Vec3d(*config["school_position_m"]))
    for name in ("Table", "Background/Backdrop", "Lights"):
        if not stage.GetPrimAtPath(f"{school.GetPath()}/{name}"):
            raise RuntimeError(f"School scene missing {name}")
    stand = UsdGeom.Xform.Define(stage, "/World/G1D")
    UsdGeom.XformCommonAPI(stand).SetTranslate(Gf.Vec3d(*config["root_position_m"]))
    UsdGeom.XformCommonAPI(stand).SetRotate(Gf.Vec3f(0, 0, config["root_yaw_deg"]))
    robot = UsdGeom.Xform.Define(stage, "/World/G1D/Robot")
    robot.GetPrim().GetReferences().AddReference(config["robot_usd"])
    camera_config, camera_source = load_config(config["camera_config"])
    parents = camera_config["isaac"]["parent_paths"]
    for name, path in parents.items():
        parents[name] = f"{robot.GetPath()}/{path.rsplit('/', 1)[-1]}"
    installed = create_cameras(stage, str(robot.GetPath()), camera_config)
    evidence = installed["audit"]["cameras"]["head_left"]
    (output / "camera_install.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    placement = UsdGeom.Xform.Define(stage, "/World/DiagnosticTarget")
    target = UsdGeom.Xform.Define(stage, "/World/DiagnosticTarget/Object").GetPrim()
    target.GetReferences().AddReference(config["target_usd"])
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    target_box = bounds.ComputeWorldBound(target).ComputeAlignedRange()
    table_box = bounds.ComputeWorldBound(stage.GetPrimAtPath("/World/School/Table/Top")).ComputeAlignedRange()
    if target_box.IsEmpty() or table_box.IsEmpty():
        raise RuntimeError("Target or tabletop has no geometric bound")
    table_centre = table_box.GetMidpoint()
    target_centre = target_box.GetMidpoint()
    UsdGeom.XformCommonAPI(placement).SetTranslate(Gf.Vec3d(
        table_centre[0] - target_centre[0], table_centre[1] - 0.12 - target_centre[1],
        table_box.GetMax()[2] - target_box.GetMin()[2]))
    resolution = evidence["resolution"]
    width, height = resolution
    product = rep.create.render_product(evidence["prim_path"], tuple(resolution))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(product)
    for _ in range(20):
        app.update()
    for attempt in range(3):
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
        data = annotator.get_data()
        pixels = np.asarray(data["data"] if isinstance(data, dict) else data)
        if pixels.ndim == 3 and pixels.shape[:2] == (height, width) and pixels.shape[-1] >= 3:
            rgb = pixels[..., :3]
            if rgb.dtype == np.uint8 and rgb.std(axis=(0, 1)).mean() > 1:
                break
    else:
        raise RuntimeError(f"Head-left RTX returned no spatial RGB: {pixels.shape}, {pixels.dtype}")
    Image.fromarray(rgb).save(output / "head_left_raw.png")
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    bounds.Clear()
    transform = UsdGeom.XformCache()
    camera_world = np.asarray(evidence["T_world_camera_usd_stage_units_at_install"])
    right_jaw = robot.GetPrim().GetChild("right_dex1_base_link")
    if not right_jaw or not right_jaw.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("Expected physical right gripper link")
    points = {"target centre": bounds.ComputeWorldBound(target).ComputeAlignedRange().GetMidpoint(),
              "right gripper base": transform.GetLocalToWorldTransform(right_jaw).ExtractTranslation()}
    projected = {}
    for label, point in points.items():
        pixel = project(point, camera_world, evidence)
        projected[label] = pixel
        if pixel is not None and 0 <= pixel[0] < width and 0 <= pixel[1] < height:
            pixel_x, pixel_y = pixel
            draw.ellipse((pixel_x - 5, pixel_y - 5, pixel_x + 5, pixel_y + 5), outline="yellow", width=2)
            draw.text((pixel_x + 7, pixel_y - 8), label, fill="yellow", stroke_width=1, stroke_fill="black")
    image.save(output / "head_left_projection.png")
    joints = []
    for prim in Usd.PrimRange(robot.GetPrim()):
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            angular = prim.IsA(UsdPhysics.RevoluteJoint)
            joint = UsdPhysics.RevoluteJoint(prim) if angular else UsdPhysics.PrismaticJoint(prim)
            joints.append({"path": str(prim.GetPath()), "unit": "degree" if angular else "metre",
                           "lower": str(joint.GetLowerLimitAttr().Get()), "upper": str(joint.GetUpperLimitAttr().Get())})
    result = {"isaacsim": importlib.metadata.version("isaacsim"), "robot_usd": config["robot_usd"],
              "school_usd": config["school_usd"], "head_left": evidence, "frame_attempt": attempt,
              "camera_source": camera_source, "sim_time_s": omni.timeline.get_timeline_interface().get_current_time(),
              "target_usd": config["target_usd"], "projected_landmarks_px": projected,
              "static_usd_joints": joints, "scope": "static diagnostic; no physics, visibility occlusion or action-clock validation"}
    (output / "environment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if physics_readback:
        product_path = product.path if replicator_fence else None
        read_physics(robot.GetPrim(), installed["cameras"]["head_left"], output, product_path)


def read_physics(robot, camera, output, product_path):
    import carb
    import numpy as np
    import omni.kit.app
    import omni.replicator.core as rep
    from projects.g1d_rsi.camera_capture import capture_owned_cameras
    from pxr import Usd, UsdPhysics

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.core.api", True)
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import Articulation
    from isaacsim.sensors.camera import Camera

    roots = [str(prim.GetPath()) for prim in Usd.PrimRange(robot) if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise RuntimeError(f"Expected exactly one PhysX articulation root: {roots}")
    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/hydra/supportMultiTickRate", True)
    settings.set_bool("/persistent/omni/replicator/captureOnPlay", True)
    rep.orchestrator.set_capture_on_play(True)
    simulation = SimulationContext(physics_dt=1 / 120, rendering_dt=1 / 30, backend="numpy", device="cpu")
    simulation.reset()
    if product_path:
        camera = Camera(camera.prim_path, name="g1d_readback_left", resolution=camera.get_resolution(),
                        render_product_path=product_path)
        camera.initialize()

    def render_fence():
        if not product_path:
            simulation.render()
            return
        previous_simulation_flag = settings.get_as_bool("/app/player/playSimulations")
        settings.set_bool("/app/player/playSimulations", False)
        try:
            rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
        finally:
            settings.set_bool("/app/player/playSimulations", previous_simulation_flag)
    articulation = Articulation(prim_paths_expr=roots[0], name="g1d_readback")
    articulation.initialize()
    names = list(articulation.dof_names)
    limits = np.asarray(articulation.get_dof_limits())
    frames = []
    previous = None
    report = {"articulation_root": roots[0], "dof_names": names, "dof_limits_native": limits.tolist(), "samples": frames,
              "fence": "replicator_paused_physics" if product_path else "simulation_context_render",
              "scope": "uncommanded physics readback; not a controller or grasp test"}
    try:
        for _ in range(8):
            simulation.step(render=True)
            time_s = float(simulation.current_time)
            state = np.asarray(articulation.get_joint_positions()).copy()
            frame = {"physics_time_s": time_s, "joint_position_native": state.tolist()}
            frames.append(frame)
            try:
                receipt = capture_owned_cameras({"head_left": camera}, expected_sim_time=time_s,
                                                render=render_fence, sim_time=lambda: simulation.current_time,
                                                previous=previous, resolutions={"head_left": camera.get_resolution()},
                                                max_passes=3)
                previous = receipt["frames"]
                frame["capture"] = previous["head_left"]
            except Exception as error:
                frame["capture_error"] = str(error)
        report["aligned_frames"] = sum("capture" in frame for frame in frames)
        if report["aligned_frames"] < 3 or len({frame["physics_time_s"] for frame in frames if "capture" in frame}) < 3:
            raise RuntimeError("Fewer than three distinct left-eye frames aligned to native physics readback")
    finally:
        (output / "physics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        simulation.stop()


def project(point, camera_world, evidence):
    import numpy as np

    camera_x, camera_y, camera_z, _ = np.linalg.solve(camera_world, [*point, 1.0])
    if camera_z >= 0:
        return None
    norm_x, norm_y = -camera_x / camera_z, camera_y / camera_z
    radius_sq = norm_x * norm_x + norm_y * norm_y
    radial_first, radial_second, tangent_first, tangent_second, radial_third = evidence["dist"]
    radial = 1 + radius_sq * (radial_first + radius_sq * (radial_second + radius_sq * radial_third))
    distorted_x = norm_x * radial + 2 * tangent_first * norm_x * norm_y + tangent_second * (radius_sq + 2 * norm_x**2)
    distorted_y = norm_y * radial + tangent_first * (radius_sq + 2 * norm_y**2) + 2 * tangent_second * norm_x * norm_y
    intrinsic = evidence["K"]
    return [float(intrinsic[0][0] * distorted_x + intrinsic[0][2]), float(intrinsic[1][1] * distorted_y + intrinsic[1][2])]


if __name__ == "__main__":
    sys.exit(main())
