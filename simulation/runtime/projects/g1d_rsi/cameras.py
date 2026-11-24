"""Four physical G1-D cameras using Isaac Sim 5.1 native OpenCV lenses.

Call after SimulationApp startup with the active stage and the starter JSON (or
its decoded dictionary). Returns ``{"cameras": name_to_Camera, "audit": dict}``.
Call after World.reset(), or set config["isaac"]["initialize"] = False and call
each camera.initialize() after reset. This function does not step physics, hide
robot meshes, save the stage, choose policy camera order, or assert visibility.

Verified API references (5.1.0, not the latest-version aliases):
https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html
https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.sensors.camera/docs/index.html

``isaac`` options: parent_paths (camera name -> exact USD parent path),
parent_frames (name -> "primary" or "alternate"), frequency, initialize,
clipping_range_m. Parent paths must remain below robot_prim_path. Their coordinate
origins must match the selected source URDF frames; matching names alone does not
validate physical calibration. Defaults retain the original 640x480 K/D pairs.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np


CAMERA_NAMES = ("head_left", "head_right", "left_wrist", "right_wrist")
# p_opencv = OPENCV_FROM_USD @ p_usd. USD looks down -Z with +Y up.
OPENCV_FROM_USD = np.diag([1.0, -1.0, -1.0, 1.0])
FLU_FROM_OPENCV = np.array(
    [[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0],
     [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def _matrix(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label}: expected finite {shape}, got {result.shape}")
    return result


def _rigid_transform(value, label):
    result = _matrix(value, (4, 4), label)
    rotation = result[:3, :3]
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7, rtol=0)):
        raise ValueError(f"{label}: expected a rigid right-handed transform")
    return result


def load_config(config):
    """Validate the starter package without importing Isaac or modifying input."""
    if isinstance(config, (str, Path)):
        raw = Path(config).read_bytes()
        doc = json.loads(raw.decode("utf-8-sig"))
        source = {"path": str(Path(config).resolve()),
                  "sha256": hashlib.sha256(raw).hexdigest()}
    else:
        doc = copy.deepcopy(config)
        raw = json.dumps(doc, sort_keys=True, allow_nan=False).encode("utf-8")
        source = {"path": None, "canonical_json_sha256": hashlib.sha256(raw).hexdigest()}
    if doc.get("schema") != "g1d.camera_starter.v1":
        raise ValueError("Expected g1d.camera_starter.v1; do not mix donor schemas")
    if set(doc.get("cameras", {})) != set(CAMERA_NAMES):
        raise ValueError("Exactly four physical cameras are required; policy mapping is separate")
    conventions = doc.get("conventions", {})
    if conventions.get("translation_unit") != "m":
        raise ValueError("Starter extrinsics must use meters")
    if conventions.get("distortion_order") != ["k1", "k2", "p1", "p2", "k3"]:
        raise ValueError("Expected paired OpenCV Brown coefficients k1,k2,p1,p2,k3")
    for name in CAMERA_NAMES:
        spec = doc["cameras"][name]
        size = _matrix(spec["resolution"], (2,), f"{name}.resolution")
        if (size <= 0).any() or not np.equal(size, np.floor(size)).all():
            raise ValueError(f"{name}: resolution must contain positive integers")
        k = _matrix(spec["K"], (3, 3), f"{name}.K")
        if (k[0, 0] <= 0 or k[1, 1] <= 0 or k[0, 1] != 0 or k[1, 0] != 0
                or not np.array_equal(k[2], [0, 0, 1])):
            raise ValueError(f"{name}: invalid or skewed OpenCV K")
        _matrix(spec["dist"], (5,), f"{name}.dist")
        for frame in ("primary", "alternate"):
            parent_to_usd(spec, frame=frame)
    return doc, source


def parent_to_usd(spec, frame="primary", meters_per_unit=1.0):
    """Return T_parent_USDcamera (column-vector convention, stage units).

    The supplied optical transform maps OpenCV camera coordinates into parent
    coordinates. Only its translation is rescaled; orientation is independent
    of stage up-axis because it is a local robot-link transform.
    """
    if frame not in ("primary", "alternate"):
        raise ValueError("parent frame must be primary or alternate")
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("meters_per_unit must be positive and finite")
    prefix = "T_parent_camera_" if frame == "primary" else "T_alternate_parent_camera_"
    optical = _rigid_transform(spec[prefix + "opencv"], prefix + "opencv")
    flu = _rigid_transform(spec[prefix + "flu"], prefix + "flu")
    if not np.allclose(flu @ FLU_FROM_OPENCV, optical, atol=1e-8, rtol=0):
        raise ValueError("FLU/OpenCV transforms disagree; refusing ambiguous axes")
    local = optical @ OPENCV_FROM_USD
    local[:3, 3] /= meters_per_unit
    return local


def _resolve_parent(stage, robot, name, spec, options, Usd, UsdGeom, Sdf):
    frame = options.get("parent_frames", {}).get(name, "primary")
    link = spec["parent_link" if frame == "primary" else "alternate_parent_link"]
    explicit = options.get("parent_paths", {}).get(name)
    if explicit:
        path = Sdf.Path(explicit)
        if not path.IsAbsolutePath() or not path.HasPrefix(robot.GetPath()):
            raise ValueError(f"{name}: parent path must be below {robot.GetPath()}")
        parent = stage.GetPrimAtPath(path)
        candidates = [parent] if parent and parent.IsActive() else []
    else:
        candidates = [p for p in Usd.PrimRange(robot)
                      if p.GetName() == link and p.IsActive() and p.IsA(UsdGeom.Xformable)]
    if len(candidates) != 1:
        raise ValueError(f"{name}: expected one active {link}, found {len(candidates)}; "
                         "supply isaac.parent_paths after auditing the frame")
    parent = candidates[0]
    if not parent.IsA(UsdGeom.Xformable) or parent.IsInstanceProxy() or parent.IsInstance():
        raise ValueError(f"{name}: parent is not an editable transform: {parent.GetPath()}")
    # A scale/shear in the parent chain changes the metric local calibration.
    world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(parent)).T
    _rigid_transform(world, f"{name} parent world transform (scaled rigs require explicit repair)")
    return parent, frame, link


def robot_asset_root(stage, articulation):
    """Resolve a joint-level PhysX articulation root to its owning asset Xform.

    Lab 3 can report the fixed root joint instead of the enclosing robot asset.
    Only ascend an actual USD Joint, and verify every available rigid-body path
    stays under that asset. The joint and its authored transforms are untouched.
    """
    from pxr import Sdf, UsdGeom, UsdPhysics

    paths = list(articulation.root_physx_view.prim_paths)
    if len(paths) != 1:
        raise ValueError(f"Expected one robot articulation, found {paths}")
    physical_path = Sdf.Path(paths[0])
    prim = stage.GetPrimAtPath(physical_path)
    if not prim or not prim.IsActive():
        raise ValueError(f"Missing active articulation root: {physical_path}")
    joint_root = prim.IsA(UsdPhysics.Joint)
    asset = prim.GetParent() if joint_root else prim
    if not asset or not asset.IsA(UsdGeom.Xformable) or asset.IsA(UsdPhysics.Joint):
        raise ValueError(f"Articulation does not identify a robot asset Xform: {physical_path}")
    body_view = getattr(articulation, "body_physx_view", None)
    bodies = list(getattr(body_view, "prim_paths", ()))
    body_verification = "physx_body_view"
    if not bodies:
        # A reported root joint must also have real body descendants. Read the
        # stage schemas instead of guessing a robot root from its name.
        from pxr import Usd
        bodies = [str(p.GetPath()) for p in Usd.PrimRange(asset) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        body_verification = "stage_rigid_body_schemas"
    if not bodies or any(not Sdf.Path(path).HasPrefix(asset.GetPath())
                         or not stage.GetPrimAtPath(path) for path in bodies):
        raise ValueError("Robot body paths do not confirm a single enclosing asset")
    return str(asset.GetPath()), {"physics_articulation_path": str(physical_path),
        "asset_root_path": str(asset.GetPath()), "joint_level_root": joint_root,
        "body_count_verified": len(bodies), "body_prim_paths": bodies,
        "body_verification": body_verification,
        "selection": "parent_of_authored_joint" if joint_root else "articulation_asset_root"}


def create_cameras(stage, robot_prim_path, config):
    """Attach four dynamic cameras in the session layer; return handles + audit.

    USD and Isaac imports are deliberately delayed until this call. All parent
    paths are checked before creation. A second invocation refuses to overwrite
    existing mounts. Native lens readback is checked, but rendering and physical
    visibility must be validated by the caller in the actual simulation.
    """
    doc, source = load_config(config)
    options = doc.get("isaac", {})
    from pxr import Gf, Sdf, Usd, UsdGeom
    import omni.usd
    from isaacsim.sensors.camera import Camera

    active_stage = omni.usd.get_context().get_stage()
    if active_stage is None or active_stage.GetRootLayer().identifier != stage.GetRootLayer().identifier:
        raise ValueError("Camera API requires the supplied stage to be the active Isaac stage")
    robot = stage.GetPrimAtPath(robot_prim_path)
    if not robot or not robot.IsActive():
        raise ValueError(f"Missing active robot prim: {robot_prim_path}")
    units = float(UsdGeom.GetStageMetersPerUnit(stage))
    frequency = options.get("frequency", 30)
    if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency <= 0:
        raise ValueError("isaac.frequency must be a positive integer Hz")
    clipping = _matrix(options.get("clipping_range_m", [0.01, 20.0]), (2,), "clipping_range_m")
    if not 0 < clipping[0] < clipping[1]:
        raise ValueError("clipping_range_m must satisfy 0 < near < far")
    planned = []
    for name in CAMERA_NAMES:
        spec = doc["cameras"][name]
        parent, frame, link = _resolve_parent(stage, robot, name, spec, options, Usd, UsdGeom, Sdf)
        mount_path = parent.GetPath().AppendChild("g1d_rsi_" + name)
        if stage.GetPrimAtPath(mount_path):
            raise ValueError(f"Refusing to overwrite existing camera mount: {mount_path}")
        planned.append((name, spec, parent, frame, link, mount_path,
                        parent_to_usd(spec, frame, units)))

    audit = {"schema": "g1d.isaac_camera_install.v1", "source": source,
             "source_status": doc.get("status", "unknown"),
             "current_robot_calibration": "unvalidated_transferred_parameters",
             "stage_meters_per_unit": units, "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
             "edit_layer": "session", "mount_mode": "USD_parent_link_local",
             "render_validation": "not_performed", "visibility_validation": "not_performed",
             "pixel_channels": "RGBA from Camera.get_rgba(); use [:,:,:3] for RGB",
             "policy_camera_mapping": "not_selected", "cameras": {}}
    handles = {}
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for name, spec, parent, frame, link, mount_path, local in planned:
            mount = UsdGeom.Xform.Define(stage, mount_path)
            # Gf matrices act on row vectors, hence transpose from config convention.
            mount.AddTransformOp().Set(Gf.Matrix4d(*local.T.ravel().tolist()))
            camera_path = mount_path.AppendChild("Camera")
            camera = Camera(prim_path=str(camera_path), name="g1d_rsi_" + name,
                            frequency=frequency, resolution=tuple(spec["resolution"]))
            camera.set_local_pose(translation=np.zeros(3), orientation=np.array([1., 0., 0., 0.]),
                                  camera_axes="usd")
            camera_xform = UsdGeom.Xformable(stage.GetPrimAtPath(camera_path))
            camera_xform.SetResetXformStack(False)
            if options.get("initialize", True):
                camera.initialize()
            k = np.asarray(spec["K"], dtype=float)
            width, height = spec["resolution"]
            # Physical pixel pitch is a rendering convention, not a measured sensor
            # dimension. Depth of field is disabled. Native lens K/D is authoritative.
            pitch_m = 3e-6
            camera.set_focal_length(pitch_m * (k[0, 0] + k[1, 1]) / (2 * units))
            camera.set_horizontal_aperture(pitch_m * width / units)
            camera.set_vertical_aperture(pitch_m * height / units)
            camera.set_lens_aperture(0.0)
            camera.set_clipping_range(*(clipping / units).tolist())
            distortion = list(spec["dist"]) + [0.0] * 7
            camera.set_opencv_pinhole_properties(cx=float(k[0, 2]), cy=float(k[1, 2]),
                                                 fx=float(k[0, 0]), fy=float(k[1, 1]),
                                                 pinhole=distortion)
            cx, cy, fx, fy, actual_distortion = camera.get_opencv_pinhole_properties()
            if (not np.allclose([cx, cy, fx, fy], [k[0, 2], k[1, 2], k[0, 0], k[1, 1]],
                                atol=1e-4, rtol=1e-6)
                    or not np.allclose(actual_distortion, distortion, atol=1e-7, rtol=1e-6)):
                raise RuntimeError(f"{name}: native OpenCV lens readback mismatch")
            actual_world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(
                stage.GetPrimAtPath(camera_path))).T
            parent_world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(parent)).T
            if not np.allclose(actual_world, parent_world @ local, atol=1e-6, rtol=0):
                raise RuntimeError(f"{name}: camera does not inherit expected parent transform")
            handles[name] = camera
            audit["cameras"][name] = {
                "prim_path": str(camera_path), "parent_path": str(parent.GetPath()),
                "source_parent_link": link, "parent_frame": frame,
                "extrinsics_status": spec["extrinsics_status"],
                "intrinsics_profile": spec["intrinsics_profile"],
                "resolution": spec["resolution"], "frequency_hz": frequency,
                "K": spec["K"], "dist": spec["dist"],
                "T_parent_camera_usd_stage_units": local.tolist(),
                "T_world_camera_usd_stage_units_at_install": actual_world.tolist(),
                "native_lens_readback": {"cx": float(cx), "cy": float(cy), "fx": float(fx),
                                         "fy": float(fy), "dist_12": list(map(float, actual_distortion))},
                "initialized": bool(options.get("initialize", True)),
                "native_lens_configuration": "verified_by_readback",
                "mount_configuration": "verified_at_install",
                "dynamic_motion_validation": "not_performed",
                "depth_of_field": "disabled", "physical_pixel_pitch": "render_convention_3um"}
    return {"cameras": handles, "audit": audit}
