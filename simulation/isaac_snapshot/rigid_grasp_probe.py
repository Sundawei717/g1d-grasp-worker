import json
import math
from collections import Counter
from dataclasses import asdict
from time import perf_counter

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation, Slerp

from grasp_regions import (BASKET_HANDLE, BASKET_RIM, BOX_NEG_END, BOX_POS_END, MUG_BODY, MUG_HANDLE,
                           UPPER_PART, anchor_height, contact_band)
from strict_grasp import PhysicsSample, evaluate


class ContactObserver:
    def __init__(self, stage, robot_path, target_path, table_path, obstacle_paths=()):
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema, PhysicsSchemaTools, Usd, UsdPhysics

        self.decode = PhysicsSchemaTools.intToSdfPath
        self.robot_path = robot_path
        self.target_path = target_path
        self.table_path = table_path
        self.obstacle_paths = tuple(obstacle_paths)
        self.events = []
        self.views = []
        self.stage = stage
        self.events_seen = 0
        for path in (robot_path, target_path):
            bodies = [prim for prim in Usd.PrimRange(stage.GetPrimAtPath(path)) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
            if not bodies:
                raise RuntimeError(f"No contact-report rigid bodies below {path}")
            for body in bodies:
                api = PhysxSchema.PhysxContactReportAPI.Apply(body)
                api.CreateThresholdAttr().Set(0.0)
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self.on_contact)

    def bind_tensors(self, sim):
        from pxr import Usd, UsdPhysics

        simulation_view = sim.physics_manager.get_physics_sim_view()
        school = self.stage.GetPrimAtPath("/World/School")
        static_colliders = sorted(str(prim.GetPath()) for prim in Usd.PrimRange(school)
                                  if prim.HasAPI(UsdPhysics.CollisionAPI))
        if self.table_path not in static_colliders:
            raise RuntimeError("Tabletop is not a PhysX collider")
        robot = self.stage.GetPrimAtPath(self.robot_path)
        robot_paths = sorted(str(prim.GetPath()) for prim in Usd.PrimRange(robot)
                             if prim.HasAPI(UsdPhysics.RigidBodyAPI))
        filters = [self.target_path, *static_colliders, *self.obstacle_paths, *robot_paths]
        main = simulation_view.create_rigid_contact_view(robot_paths, [filters] * len(robot_paths),
                                                         max_contact_data_count=4096)
        support_filters = [self.table_path, *self.obstacle_paths]
        support = simulation_view.create_rigid_contact_view(self.target_path, support_filters, max_contact_data_count=128)
        if main.sensor_count != len(robot_paths) or main.filter_count != len(filters):
            raise RuntimeError(f"PhysX tensor filter count mismatch: {main.sensor_count}, {main.filter_count}")
        if support.sensor_count != 1 or support.filter_count != len(support_filters):
            raise RuntimeError("PhysX tensor support view did not bind to the expected fixture pairs")
        self.views = [main, support]
        self.dt = sim.get_physics_dt()
        self.robot_paths = list(map(str, main.sensor_paths))
        self.filters = filters
        self.support_filters = support_filters
        if set(self.robot_paths) != set(robot_paths) or len(self.robot_paths) != len(robot_paths):
            raise RuntimeError(f"Unexpected PhysX tensor sensor paths: {self.robot_paths}")
        reported_filters = main.filter_paths
        if reported_filters and len(reported_filters) == len(robot_paths) and all(
                isinstance(row, (list, tuple)) for row in reported_filters):
            if any(list(map(str, row)) != filters for row in reported_filters):
                raise RuntimeError("PhysX tensor filter path order differs from the explicit request")
        self.tensor_bind = {"robot_body_count": len(self.robot_paths), "filter_count": len(filters),
                            "support_pair": [self.target_path, self.table_path], "obstacle_bodies": self.obstacle_paths}

    def tensor_events(self):
        for view_index, view in enumerate(self.views):
            forces, points, _, separations, counts, offsets = (value.numpy() for value in view.get_contact_data(self.dt))
            counts = counts.reshape(view.sensor_count, view.filter_count)
            offsets = offsets.reshape(view.sensor_count, view.filter_count)
            if counts.sum() >= view.max_contact_data_count:
                raise RuntimeError("PhysX contact data buffer exhausted")
            for sensor_id, filter_id in np.argwhere(counts):
                first = int(offsets[sensor_id, filter_id])
                last = first + int(counts[sensor_id, filter_id])
                if first < 0 or last > view.max_contact_data_count:
                    raise RuntimeError("PhysX contact data exceeds view capacity")
                pair = ((self.robot_paths[sensor_id], self.filters[filter_id]) if view_index == 0
                        else (self.target_path, self.support_filters[filter_id]))
                for index in range(first, last):
                    strength = float(forces[index].item())
                    if abs(strength) > 0.005:
                        yield pair, pair, points[index], float(separations[index].item()), abs(strength)

    def on_contact(self, headers, points):
        for header in headers:
            self.events_seen += 1
            actors = tuple(str(self.decode(value)) for value in (header.actor0, header.actor1))
            colliders = tuple(str(self.decode(value)) for value in (header.collider0, header.collider1))
            for index in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                item = points[index]
                impulse = np.linalg.norm(tuple(item.impulse))
                if impulse > 1e-8:
                    self.events.append((actors, colliders, np.array(tuple(item.position)), float(item.separation), float(impulse)))

    def sample_contacts(self, phase, target_pose, wrist_pose, pad_planes, part_band, *, allow_target_inner_contact=False):
        target_supported = False
        forbidden = False
        pads = set()
        problems = set()
        wrist_rotation = Rotation.from_quat(wrist_pose[3:])
        target_rotation = Rotation.from_quat(target_pose[3:])
        events = list(self.tensor_events()) if self.views else self.events
        self.last_pairs = sorted(set(" | ".join(pair) for pair, _, _, _, _ in events))
        self.last_pad_forces_n = {}
        self.last_tactile_pad_forces_n = {}
        self.last_inner_contact_points = []
        self.last_target_inner_pads = set()
        self.last_target_contacts = []
        for actors, colliders, point, separation, strength in events:
            robot = [index for index, actor in enumerate(actors) if actor.startswith(self.robot_path + "/")]
            target = [index for index, actor in enumerate(actors) if actor == self.target_path]
            support = any(path.startswith(self.table_path + "/") or path == self.table_path
                          or path in self.obstacle_paths for path in colliders)
            target_supported |= bool(target and support)
            if target and any(path in self.obstacle_paths for path in colliders):
                forbidden = True
                problems.add(" | ".join(colliders))
            if not robot:
                continue
            pad = actors[robot[0]].rsplit("/", 1)[-1]
            if (len(robot) == 1 and "_Wheel_Link" in pad
                    and "/World/School/Background/Floor" in colliders):
                continue
            allowed = False
            if len(robot) == 1 and pad in pad_planes:
                sign, plane = pad_planes[pad]
                jaw_y = wrist_rotation.inv().apply(point - wrist_pose[:3])[1]
                inner_face = abs(jaw_y - plane) <= 0.004 and sign * jaw_y >= -0.002 and separation >= -0.003
                key = f"pad_{pad.split('_')[-2]}"
                if inner_face:
                    self.last_tactile_pad_forces_n[key] = self.last_tactile_pad_forces_n.get(key, 0.0) + strength
                if target:
                    local_point = target_rotation.inv().apply(point - target_pose[:3])
                    object_z = local_point[2]
                    self.last_target_contacts.append((pad, float(object_z), bool(inner_face), float(jaw_y - plane)))
                    if inner_face:
                        self.last_target_inner_pads.add(key)
                        self.last_inner_contact_points.append({"pad": key, "target_local_xyz_m": local_point.tolist(),
                                                               "normal_force_n": float(strength)})
                        if phase in {"close", "lift", "hold"}:
                            part_contact = part_band[0] <= object_z <= part_band[1]
                            allowed = part_contact or allow_target_inner_contact
                            if part_contact:
                                pads.add(key)
                                self.last_pad_forces_n[key] = self.last_pad_forces_n.get(key, 0.0) + strength
            if not allowed:
                forbidden = True
                problems.add(" | ".join(colliders))
        return target_supported, frozenset(pads), forbidden, problems


def pad_geometry(stage, robot_prim):
    from pxr import Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    wrist = robot_prim.GetChild("right_dex1_base_link")
    inverse = cache.GetLocalToWorldTransform(wrist).GetInverse()
    geometry = {}
    for finger in (1, 2):
        name = f"right_dex1_finger_link_{finger}_tip"
        pad = robot_prim.GetChild(name)
        link_y = (cache.GetLocalToWorldTransform(pad) * inverse).ExtractTranslation()[1]
        vertices = []
        for prim in Usd.PrimRange(pad.GetChild("collisions"), Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or not prim.GetParent().HasAPI(UsdPhysics.CollisionAPI):
                continue
            link_transform = cache.GetLocalToWorldTransform(prim) * inverse
            vertices.extend(link_transform.Transform(point)[1] for point in UsdGeom.Mesh(prim).GetPointsAttr().Get())
        if not vertices:
            raise RuntimeError(f"Missing physical pad geometry: {name}")
        sign = 1 if link_y > 0 else -1
        geometry[name] = (sign, (min(vertices) if sign > 0 else max(vertices)) - link_y)
    if sum(side for side, _ in geometry.values()) != 0:
        raise RuntimeError("Both pad links are on the same side of the jaw")
    return geometry


class TargetNotVisible(RuntimeError):
    pass


def project_points(camera_world, camera, world):
    import cv2

    local = (world - camera_world[:3, 3]) @ camera_world[:3, :3]
    optical = local * [1, -1, -1]
    if np.any(optical[:, 2] <= 0):
        return None
    pixels = cv2.projectPoints(optical, np.zeros(3), np.zeros(3), np.asarray(camera["K"]),
                               np.asarray(camera["dist"]))[0].reshape(-1, 2)
    if not np.isfinite(pixels).all():
        return None
    return pixels, optical[:, 2]


def project_anchor(camera_world, camera, target_pose, object_rotation, anchor, mask, depth):
    world = object_rotation.apply(anchor) + target_pose[:3]
    result = project_points(camera_world, camera, world[None])
    if result is None:
        return None, False
    pixels, distance = result[0][0], result[1][0]
    limits = np.asarray(camera["resolution"]) - 1
    projected = pixels / limits
    if np.any(pixels < 0) or np.any(pixels > limits):
        return projected.tolist(), False
    x, y = np.rint(pixels).astype(int)
    visible = bool(mask[y, x] and abs(float(depth[y, x]) - distance) <= 0.003)
    return projected.tolist(), visible


def camera_world_pose(head_pose, camera):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(head_pose[3:]).as_matrix()
    transform[:3, 3] = head_pose[:3]
    return transform @ camera["T_parent_camera_usd_stage_units"]


def full_projection_prompt(head_pose, camera, target_pose, local_vertices, anchor_local):
    camera_world = camera_world_pose(head_pose, camera)
    object_rotation = Rotation.from_quat(target_pose[3:])
    projected = project_points(camera_world, camera,
                               object_rotation.apply(np.vstack((local_vertices, anchor_local))) + target_pose[:3])
    if projected is None:
        return {"prompt_valid": False}
    pixels = projected[0] / (np.asarray(camera["resolution"]) - 1)
    point = pixels[-1]
    vertices = pixels[:-1]
    lo, hi = np.clip(vertices.min(axis=0), 0, 1), np.clip(vertices.max(axis=0), 0, 1)
    if np.any(point < 0) or np.any(point > 1) or np.any(hi <= lo):
        return {"prompt_valid": False}
    return {"prompt_valid": True, "bbox_0": [*lo.tolist(), *hi.tolist()], "affordance_0": point.tolist(),
            "prompt_source": "gt_full_projection"}


def visible_prompt(head_pose, camera, target_pose, bounds, target_path, segmentation, depth, *, visible_fixtures=(),
                   grasp_part=UPPER_PART, include_masks=False, anchor_local=None, part_meshes=None):
    import cv2

    camera_world = camera_world_pose(head_pose, camera)
    object_rotation = Rotation.from_quat(target_pose[3:])
    raw = np.asarray(segmentation["data"]).squeeze()
    labels = segmentation["info"]["idToLabels"]
    if isinstance(labels, str):
        labels = json.loads(labels)
    ids = [int(key) for key, path in labels.items() if isinstance(path, str) and (
        path == target_path or path.startswith(target_path + "/"))]
    depth = np.asarray(depth["data"] if isinstance(depth, dict) else depth).squeeze()
    width, height = camera["resolution"]
    if raw.shape != (height, width) or depth.shape != raw.shape:
        raise RuntimeError(f"Truth render dimensions mismatch: {raw.shape}/{depth.shape}")
    fixture_pixels = {}
    for fixture in visible_fixtures:
        visible_ids = [int(key) for key, path in labels.items() if isinstance(path, str) and (
            path == fixture or path.startswith(fixture + "/"))]
        fixture_pixels[fixture] = int(np.count_nonzero(np.isin(raw, visible_ids)))
        if fixture_pixels[fixture] < 80:
            raise RuntimeError(f"Expected visible fixture in the first left frame: {fixture} has {fixture_pixels[fixture]} pixels")
    mask = np.isin(raw, ids).astype(np.uint8)
    edges_y, edges_x = np.nonzero(mask)
    projected = None
    if anchor_local is not None:
        projected, anchor_visible = project_anchor(camera_world, camera, target_pose, object_rotation,
                                                    anchor_local, mask, depth)

    def absent():
        return {"prompt_valid": False, "part_hit_local_xyz_m": list(anchor_local), "projected_anchor": projected,
                "target_mask_packed": np.packbits(mask, axis=None)}

    if len(edges_y) < 100:
        if anchor_local is not None:
            return absent()
        raise TargetNotVisible(f"Too few visible target pixels; matched {ids}, labels={list(labels.values())[:24]}")
    limits = np.asarray(camera["resolution"]) - 1
    lo = np.array([edges_x.min(), edges_y.min()])
    hi = np.array([edges_x.max(), edges_y.max()])
    if np.any(hi - lo < 4):
        if anchor_local is not None:
            return absent()
        raise TargetNotVisible("Target does not occupy a usable image area")
    part_anchors = (part_meshes or {}).get("part_anchors")
    kernel = 3 if part_anchors or grasp_part in {BASKET_HANDLE, BASKET_RIM} else 5
    interior = cv2.erode(mask, np.ones((kernel, kernel), dtype=np.uint8))
    ys, xs = np.nonzero((interior != 0) & np.isfinite(depth) & (depth > 0))
    if not len(ys):
        if anchor_local is not None:
            return absent()
        raise TargetNotVisible("No finite depth on the interior of the visible target")
    pixels = np.c_[xs, ys].astype(np.float64)
    optical = cv2.undistortPoints(pixels[:, None, :], np.asarray(camera["K"]), np.asarray(camera["dist"]))[:, 0]
    local = np.c_[optical[:, 0], -optical[:, 1], -np.ones(len(optical))] * depth[ys, xs, None]
    hits = local @ camera_world[:3, :3].T + camera_world[:3, 3]
    part = object_rotation.inv().apply(hits - target_pose[:3])
    lower, upper = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
    valid = np.all((part >= lower - 0.008) & (part <= upper + 0.008), axis=1)
    if part_anchors and anchor_local is None:
        from part_surface import load_face_atlas
        from trimesh.proximity import closest_point

        atlas, face_labels = load_face_atlas(part_meshes["face_atlas"])
        name = {part:int(k) for k,part in part_meshes["face_names"].items()}
        possible = np.flatnonzero(valid)
        delta = np.linalg.norm(part[possible]-part_anchors[grasp_part],axis=1)
        order = possible[np.argsort(delta)[:384]]
        if not len(order):
            raise TargetNotVisible(f"No visible depth in the bounds for {grasp_part}")
        part_id = name[grasp_part]
        _, distance, face = closest_point(atlas,part[order])
        matches = order[(face_labels[face]==part_id) & (distance<=.006)]
        if not len(matches):
            # A grasp may pinch a far face of a broad region. Find the nearest visible
            # surface of the same annotated region before checking exact face identity.
            from scipy.spatial import cKDTree
            bins = cKDTree(atlas.triangles_center)
            _, nearby = bins.query(part[possible],k=min(6,len(face_labels)))
            subset = possible[np.any(face_labels[nearby]==part_id,axis=1)]
            order = subset[np.argsort(np.linalg.norm(part[subset]-part_anchors[grasp_part],axis=1))[:384]]
            if len(order):
                _, distance, face = closest_point(atlas,part[order])
                matches = order[(face_labels[face]==part_id) & (distance<=.006)]
        if not len(matches):
            raise TargetNotVisible(f"No visible depth bound to the requested triangle part {grasp_part}")
        selected=int(matches[0])
        anchor_local,projected,anchor_visible = part[selected],(pixels[selected]/limits).tolist(),True
    elif not part_anchors:
        low_z, high_z = contact_band(grasp_part, lower[2], upper[2])
        valid &= part[:, 2] >= low_z
        if grasp_part != UPPER_PART:
            valid &= part[:, 2] <= high_z
        if grasp_part == MUG_HANDLE:
            valid &= part[:, 0] >= 0.019
        elif grasp_part == MUG_BODY:
            valid &= part[:, 0] <= 0.012
        elif grasp_part == BASKET_HANDLE:
            valid &= (part[:, 2] >= 0.080) & (np.abs(part[:, 0]) < 0.028)
        elif grasp_part == BASKET_RIM:
            rim_z = (part_meshes or {}).get("rim_center_z_m", .046)
            valid &= (np.abs(part[:, 1]) >= 0.046) & (np.abs(part[:, 2] - rim_z) < 0.008)
        elif grasp_part in {BOX_NEG_END, BOX_POS_END}:
            direction = 1 if grasp_part == BOX_POS_END else -1
            valid &= direction * part[:, 1] >= 0.029
        if not valid.any() and anchor_local is None:
            raise TargetNotVisible(f"No measured {grasp_part} depth inside asset bounds; z={part[:, 2].min():.3f}..{part[:, 2].max():.3f}")
        if anchor_local is None:
            indices = np.flatnonzero(valid)
            error = np.abs(part[indices, 2] - anchor_height(grasp_part, lower[2], upper[2]))
            if grasp_part == MUG_HANDLE:
                error = np.linalg.norm(part[indices] - [0.052, 0, 0.013], axis=1)
            elif grasp_part == MUG_BODY:
                error = np.linalg.norm(part[indices] - [-0.022, 0, 0.021], axis=1)
            elif grasp_part == BASKET_HANDLE:
                error = np.linalg.norm(part[indices] - [0, 0, 0.097], axis=1)
            elif grasp_part == BASKET_RIM:
                error = np.linalg.norm(part[indices] - [0, -0.054, rim_z], axis=1)
            elif grasp_part in {BOX_NEG_END, BOX_POS_END}:
                error = np.linalg.norm(part[indices] - [0, direction * 0.044, upper[2] - 0.012], axis=1)
            else:
                error += 0.1 * np.linalg.norm(part[indices, :2], axis=1)
            selected = indices[int(np.argmin(error))]
            anchor_local, projected, anchor_visible = part[selected], (pixels[selected] / limits).tolist(), True
    result = {"bbox_0": [*(lo / limits).tolist(), *(hi / limits).tolist()],
              "affordance_0": projected, "arm_side": "right", "prompt_valid": anchor_visible,
              "projected_anchor": projected, "part_hit_local_xyz_m": list(anchor_local),
              "visible_pixels": int(len(edges_y)),
              "visible_fixture_pixels": fixture_pixels,
              "visibility_check": "same-frame RTX instance mask and calibrated depth, expert labels only"}
    if include_masks:
        part_mask = np.zeros_like(mask)
        part_mask[ys[valid], xs[valid]] = 1
        result["target_mask_packed"] = np.packbits(mask, axis=None)
        result["part_mask_packed"] = np.packbits(part_mask, axis=None)
    return result


def run_grasp(*, robot, target, sim, observer, pads, bounds, output, clock, capture, plan=None, asset_path=None,
              visibility=None, dense_camera=None, dense_frame=None, replay=None, policy=None, lift_guard=None,
              prompt_mode="gt_visible", wrist_cameras=None, target_query=None, three_view_steps=None,
              unified_lift_gate=None, fixture_feedback=None, record_v3_cameras=None):
    wrist_id = robot.body_names.index("right_dex1_base_link")
    names = list(robot.joint_names)
    arm_ids = [names.index(f"right_{joint}_joint") for joint in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
    finger_ids = [names.index(f"right_dex1_finger_joint_{index}") for index in (1, 2)]
    pad_ids = {name: robot.body_names.index(name) for name in pads}
    lower, upper = np.array(robot.data.joint_pos_limits[0, arm_ids].tolist()).T
    joints = robot.data.default_joint_pos.clone()
    if plan:
        home = np.asarray(plan["home_q"], dtype=float)
        if home.shape != (7,) or not np.isfinite(home).all() or np.any(home < lower) or np.any(home > upper):
            raise RuntimeError("Planned arm start is outside the live G1D limit envelope")
        joints[0, arm_ids] = joints.new_tensor(home)
    trace, commands = [], []
    previous_arm = joints[0, arm_ids].clone()
    delay = (plan.get("case") or {}).get("command_delay_substeps", 0) if plan else 0
    if delay not in {0, 6}:
        raise ValueError("Unsupported collection command delay")
    grasp_part = plan["part"] if plan else UPPER_PART
    dense = None
    if dense_camera is not None:
        from episode_store import EpisodeStore

        dense = EpisodeStore(dense_camera, [names[index] for index in arm_ids], lower, upper,
                             wrist_cameras=wrist_cameras, target_query=target_query)
    unexpected_pairs = Counter()
    collision_by_phase = {}
    previous_collision_pairs = set()
    trial = {"scope": "one straight-path diagnostic; not asset admission or obstacle planning",
             "target_asset": "sand3d_packaging/00152", "part": grasp_part,
             "pad_inner_surface_offsets_m": pads, "commands": commands,
             "contact_tensor_binding": getattr(observer, "tensor_bind", None)}
    if asset_path:
        from pathlib import Path

        trial["asset_usd"] = asset_path
        trial["target_asset"] = "/".join(Path(asset_path).parts[-3:-1])
    if plan:
        trial["scope"] = "one offline planned and PhysX-monitored right-arm diagnostic; not asset admission"
        trial["offline_plan"] = {"path": plan["source"], "candidate": plan["candidate"], "index": plan["candidate_index"]}
        if plan.get("case"):
            trial["case"] = plan["case"]
            trial["target_asset"] = plan["case"].get("asset_family", trial["target_asset"])
        trial["grasp_region_id"] = grasp_part
    dt = sim.get_physics_dt()
    part_band = (float(bounds.GetMin()[2]),float(bounds.GetMax()[2]))
    if not ((plan or {}).get("case",{}).get("part_meshes") or {}).get("part_anchors"):
        part_band = contact_band(grasp_part,*part_band)
    if grasp_part in {"top_side_panels", "middle_side_panels"}:
        trial["part_contact_diagnostics"] = {"band_m": list(part_band), "samples": []}

    def pose():
        return np.array(robot.data.body_link_pose_w[0, wrist_id].tolist())

    def observe(phase):
        nonlocal previous_collision_pairs
        wrist = pose()
        obj = np.array(target.data.root_link_pose_w[0].tolist())
        wrist_inverse = Rotation.from_quat(wrist[3:]).inv()
        planes = {}
        for name, (sign, offset) in pads.items():
            local = wrist_inverse.apply(np.array(robot.data.body_link_pose_w[0, pad_ids[name], :3].tolist()) - wrist[:3])
            planes[name] = (sign, local[1] + offset)
        support, contact_pads, collision, others = observer.sample_contacts(
            phase, obj, wrist, planes, part_band,
            allow_target_inner_contact=unified_lift_gate is not None or "part_contact_diagnostics" in trial)
        if unified_lift_gate is not None:
            # Policy commands before the trial can narrow the jaw while still approaching.
            # Keep the contact observer and expert training phases intact.
            pretrial = phase in {"pre_grasp", "align", "approach", "close"}
            if policy is not None and pretrial:
                label = "approach"
            else:
                label = "approach" if phase in {"pre_grasp", "align"} else phase
            phase_report = collision_by_phase.setdefault(
                label, {"pair_onsets": Counter(), "arm_link_collision_onsets": 0, "contact_physics_steps": 0})
            onsets = others - previous_collision_pairs
            phase_report["pair_onsets"].update(onsets)
            phase_report["arm_link_collision_onsets"] += sum(
                observer.robot_path + "/" in pair for pair in onsets)
            phase_report["contact_physics_steps"] += bool(others)
            if others and "first_contact" not in phase_report:
                phase_report["first_contact"] = {"sim_time_s":clock()["current_time"],
                                                "control_step_1based":len(commands)+1 if label!="settle" else 0,
                                                "pairs":sorted(others)}
            previous_collision_pairs = others
        if phase == "close" and "part_contact_diagnostics" in trial:
            samples = trial["part_contact_diagnostics"]["samples"]
            if observer.last_target_contacts and len(samples) < 120:
                samples.append({"t": clock()["current_time"], "contacts": observer.last_target_contacts})
        unexpected_pairs.update(others)
        trace.append(PhysicsSample(clock()["current_time"], float(target.data.root_com_pose_w[0, 2]), float(wrist[2]),
                                   tuple(wrist_inverse.apply(obj[:3] - wrist[:3])),
                                   tuple((wrist_inverse * Rotation.from_quat(obj[3:])).as_quat()),
                                   support, contact_pads, collision, False, frozenset(observer.last_target_inner_pads)))

    last_observed_rgb = None
    scene_observer = None
    dispatch_wall_s = None
    v3_transition = None

    def record_observation(rgb, frame, prompt):
        nonlocal last_observed_rgb
        last_observed_rgb = rgb
        stamp = frame["reference_time"]
        contact = trace[-1]
        dense.observation(rgb=rgb, timestamp=frame["sensor_sim_time_s"],
                          frame_id=f"{stamp['numerator']}/{stamp['denominator']}",
                          arm=robot.data.joint_pos[0, arm_ids].tolist(), fingers=robot.data.joint_pos[0, finger_ids].tolist(),
                          wrist=pose().tolist(), target=target.data.root_link_pose_w[0].tolist(), prompt=prompt,
                          contacts=[contact.target_supported, "pad_1" in contact.target_part_pads,
                                    "pad_2" in contact.target_part_pads, contact.forbidden_contact],
                          extra_views=frame.get("wrist_views"))

    def block(phase, record=True, blend_from=None, commands_override=None, hold_grip=False):
        nonlocal previous_arm, dispatch_wall_s
        if commands_override is not None:
            joints[0, arm_ids + finger_ids] = joints.new_tensor(commands_override[-1])
        elif phase in {"close", "lift", "hold"} and not hold_grip:
            measured = robot.data.joint_pos[0, finger_ids].tolist()
            joints[0, finger_ids] = min(0.0245, min(measured) + 0.002)
        dispatch_wall_s = perf_counter()
        robot.set_joint_position_target_index(target=joints)
        begin = len(trace)
        native_steps = [] if (dense or v3_transition is not None) and record else None
        native_error = 0.0 if unified_lift_gate is not None and commands_override is not None else None
        lag = delay if phase == "pre_grasp" and len(commands) == 5 else 0
        for substep in range(30):
            observer.events.clear()
            target_joints = joints
            if commands_override is not None:
                target_joints = joints.clone()
                target_joints[0, arm_ids + finger_ids] = joints.new_tensor(commands_override[substep])
                robot.set_joint_position_target_index(target=target_joints)
            elif lag:
                target_joints = joints.clone()
                if substep < lag:
                    target_joints[0, arm_ids] = previous_arm
                robot.set_joint_position_target_index(target=target_joints)
            elif blend_from is not None:
                target_joints = joints.clone()
                fraction = (substep + 1) / 30
                target_joints[0, arm_ids] = (1 - fraction) * blend_from + fraction * joints[0, arm_ids]
                robot.set_joint_position_target_index(target=target_joints)
            if native_steps is not None:
                native_steps.append(target_joints[0, arm_ids + finger_ids].tolist())
            if native_error is not None:
                native_error = max(native_error, float(np.max(np.abs(
                    np.asarray(target_joints[0, arm_ids + finger_ids].tolist()) - commands_override[substep]))))
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            target.update(dt)
            if record:
                observe(phase)
        previous_arm = joints[0, arm_ids].clone()
        if record:
            commands.append({"phase": phase, "time_s": trace[-1].time_s,
                             "arm_target_rad": joints[0, arm_ids].tolist(),
                             "finger_target_m": joints[0, finger_ids].tolist(),
                             "arm_observed_rad": robot.data.joint_pos[0, arm_ids].tolist(),
                             "finger_observed_m": robot.data.joint_pos[0, finger_ids].tolist(),
                             "wrist_xyz_xyzw": pose().tolist(),
                             "pad_contacts": sorted(trace[-1].target_part_pads),
                             "pad_normal_force_n": observer.last_pad_forces_n,
                             "target_z_m": trace[-1].target_z_m, "wrist_z_m": trace[-1].wrist_z_m})
            if scene_observer is not None:
                scene_observer(phase)
            if native_error is not None:
                commands[-1]["max_native_setpoint_error"] = native_error
            if lag:
                trial["command_delay"] = {"transition": len(commands) - 1, "substeps": lag, "seconds": lag * dt}
            if v3_transition is not None:
                v3_transition(phase, native_steps)
            elif native_steps is not None:
                dense.command(phase, joints[0, arm_ids].tolist(), joints[0, finger_ids].tolist(), native_steps)
                record_observation(*dense_frame())
            elif replay is None and policy is None and len(commands) % 6 == 1:
                capture(f"expert_{len(commands):03d}.png")
            if three_view_steps is None and unified_lift_gate is None and any(sample.forbidden_contact for sample in trace[begin:]):
                raise RuntimeError(f"Forbidden physical contact during {phase}")

    def drive(goal, rotation, phase, budget):
        for _ in range(budget):
            wrist = pose()
            position = goal - wrist[:3]
            angle = (rotation * Rotation.from_quat(wrist[3:]).inv()).as_rotvec()
            if np.linalg.norm(position) < 0.004 and np.linalg.norm(angle) < math.radians(5):
                return
            jacobian = np.array(robot.data.body_link_jacobian_w[0, wrist_id - 1].tolist())[:, arm_ids]
            jacobian[3:] *= 0.3
            error = np.r_[position, 0.3 * angle]
            current = np.array(robot.data.joint_pos[0, arm_ids].tolist())
            margin = 0.025
            inside = np.clip(current, lower + margin, upper - margin)
            min_delta = np.maximum(-0.04, lower + margin - inside)
            max_delta = np.minimum(0.04, upper - margin - inside)
            solution = lsq_linear(np.vstack([jacobian, 0.02 * np.eye(7)]), np.r_[error, np.zeros(7)],
                                  bounds=(min_delta, max_delta), method="bvls")
            if not solution.success:
                raise RuntimeError("Bounded Jacobian solve failed")
            target_q = inside + solution.x
            joints[0, arm_ids] = joints.new_tensor(target_q)
            block(phase)
        raise RuntimeError(f"IK did not reach {phase}: position error={np.linalg.norm(position):.5f} m, "
                           f"orientation error={np.degrees(np.linalg.norm(angle)):.2f} deg")

    def joint_path(path, phase):
        if not path:
            raise RuntimeError(f"Missing planned phase {phase}")
        for segment in path:
            endpoint = np.asarray(segment, dtype=float)
            if endpoint.shape != (7,) or not np.isfinite(endpoint).all() or np.any(endpoint < lower) or np.any(endpoint > upper):
                raise RuntimeError(f"Planned {phase} joint target violates live pinned limits")
            origin = np.asarray(joints[0, arm_ids].tolist())
            steps = max(1, math.ceil(np.max(np.abs(endpoint - origin)) / 0.035))
            for fraction in np.linspace(0, 1, steps + 1)[1:]:
                request = origin + fraction * (endpoint - origin)
                joints[0, arm_ids] = joints.new_tensor(request)
                block(phase)
                actual = np.asarray(robot.data.joint_pos[0, arm_ids].tolist())
                if np.max(np.abs(actual - request)) > 0.045:
                    raise RuntimeError(f"PhysX tracking deviated from the collision-checked {phase} path")

    def fixed_lift(path, hold_grip=False):
        if np.asarray(path).shape != (4, 7):
            raise RuntimeError("Fixed trial lift requires four checked elevation waypoints")
        current = np.asarray(joints[0, arm_ids].tolist())
        samples = np.vstack([current, path])
        if not np.isfinite(samples).all() or np.any(samples < lower) or np.any(samples > upper):
            raise RuntimeError("Fixed trial lift waypoint violates the guarded arm limits")
        trial["trial_lift_profile"] = {"duration_s": 0.8, "outer_steps": 16, "inner_servo_hz": 600,
                                      "shape": "quintic phase with substep interpolation"}
        for step in range(1, 17):
            ratio = step / 16
            phase = ratio**3 * (10 + ratio * (-15 + 6 * ratio))
            indexed = 4 * phase
            segment = min(3, int(indexed))
            weight = indexed - segment
            request = (1 - weight) * samples[segment] + weight * samples[segment + 1]
            origin = joints[0, arm_ids].clone()
            joints[0, arm_ids] = joints.new_tensor(request)
            block("lift", blend_from=origin, hold_grip=hold_grip)

    def policy_grasp(first_frame, first_prompt):
        from time import perf_counter

        from episode_store import jaw_width_m
        from policy_rollout import run_approach

        start = perf_counter()
        trial.pop("offline_plan", None)
        receipt = {"phase": "INIT", "prompt_mode": prompt_mode, "verification_source": "sim_gt"}
        trial.update(scope="policy closed-loop simulation", policy_rollout=receipt)

        def policy_state():
            arm = robot.data.joint_pos[0, arm_ids].tolist()
            width = jaw_width_m(robot.data.joint_pos[0, finger_ids].tolist())
            return np.asarray([*arm, width])

        def apply(arm, width, phase, arm_step):
            state = policy_state()
            clipped_q = np.clip(np.clip(arm, state[:7] - arm_step, state[:7] + arm_step), lower + 0.025, upper - 0.025)
            width_step = 0.002 if 0 < arm_step < 0.035 else 0.004
            clipped_width = float(np.clip(np.clip(width, state[7] - width_step, state[7] + width_step), 0.00107, 0.09007))
            joints[0, arm_ids] = joints.new_tensor(clipped_q)
            joints[0, finger_ids] = -0.020 + (0.09007 - clipped_width) / 2
            block(phase, hold_grip=True)
            return clipped_width

        try:
            run_approach(policy, first_frame, first_prompt, dense_frame, policy_state, apply, receipt, prompt_mode)
            arm = policy_state()[:7]
            fingers = robot.data.joint_pos[0, finger_ids].tolist()
            points = lift_guard(arm, pose(), fingers)
            joints[0, arm_ids] = joints.new_tensor(arm)
            lift = len(trace) - 1
            fixed_lift(points, hold_grip=True)
            hold = len(trace) - 1
            for _ in range(11):
                block("hold", hold_grip=True)
            trial.update(lift_start_sample=lift, hold_start_sample=hold, baseline_contract="stable_support_v2")
            score = evaluate(trace, lift_start=lift, hold_start=hold, dt=dt, baseline_contract="stable_support_v2")
            trial["strict_score"] = score
            receipt.update(phase="RETURN", status="success" if score["success"] else "grasp_failed")
            capture("policy_final.png")
            return score["success"]
        finally:
            receipt["elapsed_wall_s"] = perf_counter() - start
            receipt["elapsed_sim_s"] = len(commands) * 0.05

    def policy_grasp_three_view(frame, prompt):
        from episode_store import jaw_width_m
        from policy_rollout import PolicyStop
        from three_view_rollout import rollout

        reference = frame[1]["reference_time"]
        task = {"frame_id": f'{reference["numerator"]}/{reference["denominator"]}', "bbox_0": prompt["bbox_0"],
                "affordance_0": prompt["affordance_0"], "target_query": target_query, "arm_side": "right"}
        required = {"pad_1", "pad_2"}
        retry_after = 0

        def state():
            arm = robot.data.joint_pos[0, arm_ids].tolist()
            return np.asarray([*arm, jaw_width_m(robot.data.joint_pos[0, finger_ids].tolist())])

        def apply(command):
            actual = state()
            arm = np.clip(np.clip(command[:7], actual[:7] - 0.035, actual[:7] + 0.035), lower + 0.025, upper - 0.025)
            width = float(np.clip(np.clip(command[7], actual[7] - 0.004, actual[7] + 0.004), 0.00107, 0.09007))
            joints[0, arm_ids] = joints.new_tensor(arm)
            joints[0, finger_ids] = -0.020 + (0.09007 - width) / 2
            block("close" if width < actual[7] else "approach", hold_grip=True)

        def verify(remaining, step):
            nonlocal retry_after
            if remaining < 26 or step < retry_after or observer.last_target_inner_pads != required:
                return 0, None
            grip = robot.data.joint_pos[0, finger_ids].tolist()
            before = trace[-1]
            try:
                path = lift_guard(state()[:7], pose(), grip)
            except PolicyStop as error:
                retry_after = step + 10
                return 0, {"success": False, "step": step, "lift_geometry": str(error)}
            fixed_lift(path, hold_grip=True)
            for _ in range(10):
                block("hold", hold_grip=True)
            final = trace[-1]
            rise = final.target_z_m - before.target_z_m
            held = observer.last_target_inner_pads == required
            success = bool(held and not final.target_supported and rise > 0)
            retry_after = step + 36
            return 26, {"step": step, "success": success, "target_held": held, "target_rise_m": rise,
                        "part_hit": final.target_part_pads == required}

        trial["task"] = task
        trial["policy_rollout"] = rollout(policy, frame, task, capture, state, apply, verify, three_view_steps)
        if trial["policy_rollout"]["status"] == "success":
            return True
        return False

    def replay_commands(first_frame):
        from episode_store import jaw_width_m

        native = replay["commands"]
        if (np.any(native[:, :, :7] < lower) or np.any(native[:, :, :7] > upper)
                or np.any(native[:, :, 7:] < -0.020001) or np.any(native[:, :, 7:] > 0.024501)):
            raise ValueError("Recorded native setpoints leave the pinned arm limits")
        comparisons = []
        saved_images = replay["images"]
        for index, (phase, native) in enumerate(zip(replay["phases"], replay["commands"])):
            if index == replay["lift_index"]:
                trial["lift_start_sample"] = len(trace) - 1
            if index == replay["hold_index"]:
                trial["hold_start_sample"] = len(trace) - 1
            block(phase, commands_override=native)
            row = commands[-1]
            width = jaw_width_m(row["finger_observed_m"])
            comparisons.append((float(np.max(np.abs(np.asarray(row["arm_observed_rad"]) - replay["arm"][index+1]))),
                                abs(width - replay["width"][index+1])))
        end = last_observed_rgb if dense is not None else capture("replay_end.png")[0]
        images = {0: first_frame[0], len(comparisons): end}
        image_errors = [float(np.abs(images[i].astype(np.int16) - saved_images[i]).mean() / 255) for i in images]
        max_arm, max_width = np.max(comparisons, axis=0).tolist()
        score = evaluate(trace, lift_start=trial["lift_start_sample"], hold_start=trial["hold_start_sample"],
                         dt=dt, baseline_contract="stable_support_v2")
        trial.update(baseline_contract="stable_support_v2", strict_score=score,
                     replay_alignment={"source": replay["source"], "transitions": len(comparisons),
                                       "max_arm_error_rad": max_arm, "max_jaw_error_m": max_width,
                                       "left_rgb_mae": image_errors})
        return score["success"] and max_arm <= 0.005 and max_width <= 0.002 and max(image_errors) <= 0.08

    def grasp_unified(frame, prompt):
        nonlocal scene_observer, v3_transition

        from episode_store import jaw_width_m
        from part_surface import attribute_contacts
        from policy_rollout import PolicyStop
        from unified_grasp import CONTROL_HZ, HOLD_STEPS, LIFT_HEIGHT_M, LiftGateConfig, execute, guarded_gripper_width
        from three_view_inputs import proprioception

        cfg = LiftGateConfig(**unified_lift_gate)
        reference = frame[1]["reference_time"]
        task = {"frame_id": f'{reference["numerator"]}/{reference["denominator"]}', "bbox_0": prompt["bbox_0"],
                "affordance_0": prompt["affordance_0"], "target_query": target_query, "arm_side": "right"}
        trial["task"] = task
        safety = Counter()
        initial_target_pose = np.asarray(target.data.root_link_pose_w[0].tolist())
        initial_fixtures = fixture_feedback() if fixture_feedback else {}
        target_motion = {"max_pre_lift_tilt_deg": 0.0, "max_pre_lift_translation_m": 0.0}
        fixture_motion = {name: {"max_translation_m": 0.0, "max_tilt_deg": 0.0} for name in initial_fixtures}
        replay_errors = []
        native = replay["commands"] if replay else None
        te_decay = plan["case"].get("te_decay") if policy is not None else None
        temporal = None
        if te_decay is not None:
            from temporal_actions import TemporalActions

            temporal = TemporalActions(te_decay)
        cached = [frame]
        route = []
        writer = None
        approach_width = 0.09007
        if record_v3_cameras is not None:
            from step1_episode import AlignedEpisode

            if not prompt["prompt_valid"]:
                raise ValueError("The requested part has no visible prompt")
            writer = AlignedEpisode(plan["case"], [names[i] for i in arm_ids], record_v3_cameras, task)
            if replay is None:
                previous = np.asarray(joints[0, arm_ids].tolist())
                approach_width = plan["case"].get("approach_width_m", approach_width)
                if approach_width < 0.09007:
                    route.extend([previous.copy()] * (math.ceil((0.09007 - approach_width) / 0.004) + 3))
                for endpoint in [*plan["pre_grasp_q"], *plan["approach_q"][1:]]:
                    endpoint = np.asarray(endpoint)
                    count = max(1, math.ceil(np.max(np.abs(endpoint - previous)) / 0.030))
                    route.extend(previous + ratio * (endpoint - previous) for ratio in np.linspace(0, 1, count + 1)[1:])
                    previous = endpoint

            def observe_v3():
                width = jaw_width_m(robot.data.joint_pos[0, finger_ids].tolist())
                writer.observe(cached[0], robot.data.joint_pos[0, arm_ids].tolist(), width, pose().tolist())

            def record_v3(phase, applied):
                writer.action(phase, applied)
                cached[0] = capture()
                observe_v3()
                if len(writer.actions) % 10 == 0:
                    print(plan["case"]["case_id"], "control_step", len(writer.actions), phase,
                          "width_m", round(writer.observations[-1]["width"], 4),
                          "forces", observer.last_tactile_pad_forces_n, flush=True)

            observe_v3()
            v3_transition = record_v3
        if replay and (np.any(native[:, :, :7] < lower) or np.any(native[:, :, :7] > upper)
                       or np.any(native[:, :, 7:] < -0.020001) or np.any(native[:, :, 7:] > 0.024501)):
            raise ValueError("Recorded native setpoints leave the pinned arm limits")

        def pose_change(before, after):
            prior_axis = Rotation.from_quat(before[3:]).apply([0, 0, 1])
            current_axis = Rotation.from_quat(after[3:]).apply([0, 0, 1])
            tilt = math.degrees(math.acos(np.clip(prior_axis @ current_axis, -1, 1)))
            return float(np.linalg.norm(after[:3] - before[:3])), tilt

        def watch_scene(phase):
            if phase not in {"lift", "hold"}:
                distance, tilt = pose_change(initial_target_pose, np.asarray(target.data.root_link_pose_w[0].tolist()))
                target_motion["max_pre_lift_translation_m"] = max(target_motion["max_pre_lift_translation_m"], distance)
                target_motion["max_pre_lift_tilt_deg"] = max(target_motion["max_pre_lift_tilt_deg"], tilt)
            for name, live in (fixture_feedback() if fixture_feedback else {}).items():
                distance, tilt = pose_change(initial_fixtures[name], live)
                row = fixture_motion[name]
                row["max_translation_m"] = max(row["max_translation_m"], distance)
                row["max_tilt_deg"] = max(row["max_tilt_deg"], tilt)

        scene_observer = watch_scene

        def feedback():
            arm = robot.data.joint_pos[0, arm_ids].tolist()
            width = jaw_width_m(robot.data.joint_pos[0, finger_ids].tolist())
            wrist = pose().tolist()
            return {"sim_time_s": clock()["current_time"], "eef_xyz_m": wrist[:3], "eef_pose": wrist,
                    "arm": arm, "width_m": width,
                    "max_joint_speed_rad_s": float(robot.data.joint_vel[0, arm_ids].abs().max()),
                    "pad_normal_force_n": observer.last_tactile_pad_forces_n}

        def source(step, rgb, metadata, live):
            if replay:
                lift_index = replay["lift_index"]
                if step < lift_index:
                    phase, values = replay["phases"][step], native[step]
                    if writer is not None:
                        phase = "close" if phase in {"close", "close_hold"} else "approach"
                else:
                    phase, values = "close", np.tile(native[lift_index - 1, -1], (30, 1))
                    recorded_width = jaw_width_m(values[0, 7:])
                    if recorded_width <= cfg.close_threshold_m:
                        previous = jaw_width_m(joints[0, finger_ids].tolist())
                        width = guarded_gripper_width(0.00107, live["width_m"], previous,
                                                      live["pad_normal_force_n"], cfg)
                        values[:, 7:] = -0.020 + (0.09007 - width) / 2
                return [(phase, values)], {}
            if writer is not None:
                if step < len(route):
                    return [np.r_[route[step], approach_width]], {}
                return [np.r_[route[-1], plan["case"].get("grip_goal_m", 0.002)]], {}
            if getattr(policy,"v3_final",False):
                prev = live["previous_control"]
                current = proprioception(live["arm"], live["width_m"], live["eef_pose"])
                previous = proprioception(prev["arm"], prev["width_m"], prev["eef_pose"])
                chunk, timings = policy.predict(rgb,metadata,task,current,previous_state=previous,diagnostics=True,
                                                 full_chunk=temporal is not None)
                if temporal is not None:
                    action, overlapping = temporal.step(step,chunk)
                    timings["temporal_ensemble_chunks"] = overlapping
                    return [action], timings
                return chunk, timings
            return policy.predict(rgb, metadata, task, np.asarray([*live["arm"], live["width_m"]]), diagnostics=True)

        def apply(command):
            if replay:
                phase, values = command
                index = len(commands)
                block(phase, commands_override=values)
                report = commands[-1]
                width = jaw_width_m(report["finger_observed_m"])
                if index < replay["lift_index"]:
                    live = np.r_[report["arm_observed_rad"], width]
                    expected = np.r_[replay["arm"][index + 1], replay["width"][index + 1]]
                    errors = np.abs(live - expected)
                    replay_errors.append((float(errors[:7].max()), float(errors[-1]), report["max_native_setpoint_error"]))
                return jaw_width_m(report["finger_target_m"]), dispatch_wall_s
            actual = feedback()
            q = np.asarray(actual["arm"])
            limited = np.clip(command[:7], lower + 0.025, upper - 0.025)
            slew = np.clip(command[:7], q - 0.035, q + 0.035)
            safety["joint_limit_command_steps"] += bool(np.any(limited != command[:7]))
            safety["arm_slew_limited_steps"] += bool(np.any(slew != command[:7]))
            safety["jaw_limit_command_steps"] += not 0.00107 <= command[7] <= 0.09007
            arm = np.clip(slew, lower + 0.025, upper - 0.025)
            width = guarded_gripper_width(command[7], actual["width_m"], jaw_width_m(joints[0, finger_ids].tolist()),
                                          actual["pad_normal_force_n"], cfg)
            joints[0, arm_ids] = joints.new_tensor(arm)
            joints[0, finger_ids] = -0.020 + (0.09007 - width) / 2
            if writer is not None:
                phase = "approach" if len(commands) < len(route) else "close"
            else:
                phase = "close" if width <= cfg.close_threshold_m or width < actual["width_m"] - 0.0005 else "approach"
            block(phase, hold_grip=True)
            return width, dispatch_wall_s

        def lift(trigger):
            contacts = list(observer.last_inner_contact_points)
            try:
                diagnostic = attribute_contacts(asset_path, contacts, grasp_part, part_band, plan["case"].get("part_meshes"))
            except (RuntimeError, ValueError) as error:
                diagnostic = {"hit": None, "reason": str(error), "contact_points": contacts}
            current = np.asarray(feedback()["arm"])
            fingers = robot.data.joint_pos[0, finger_ids].tolist()
            attempt = {"trigger": trigger, "part_hit": diagnostic["hit"], "part_contact": diagnostic,
                       "lock_finger_target_m": joints[0, finger_ids].tolist()}
            try:
                path = lift_guard(current, pose(), fingers)
            except PolicyStop as error:
                safety["lift_path_rejected"] += 1
                return {**attempt, "success": False, "steps_used": 0,
                        "failure_category": "lift_path_rejected",
                        "failure_reasons": ["lift_path_rejected"], "detail": str(error)}
            begin = len(commands)
            joints[0, arm_ids] = joints.new_tensor(current)
            lift_start = len(trace) - 1
            fixed_lift(path, hold_grip=True)
            hold_start = len(trace) - 1
            for _ in range(HOLD_STEPS):
                block("hold", hold_grip=True)
            score = evaluate(trace, lift_start=lift_start, hold_start=hold_start, dt=dt,
                             baseline_contract="stable_support_v2", unified=True,
                             expected_lift_m=LIFT_HEIGHT_M)
            trial.update(lift_start_sample=lift_start, hold_start_sample=hold_start, strict_score=score)
            category = None
            if not score["success"]:
                pads_at_trigger = {point["pad"] for point in contacts}
                if pads_at_trigger < {"pad_1", "pad_2"}:
                    category = "empty_or_single_pad_at_trigger"
                elif set(score["reasons"]) & {"no_bilateral_target_hold", "target_slipped"}:
                    category = "lost_grip_during_lift_or_hold"
                else:
                    category = "insufficient_clearance_or_hold"
            return {**attempt, "success": score["success"], "steps_used": len(commands) - begin,
                    "failure_category": category, "failure_reasons": score["reasons"], "stability": score}

        capture_next = capture
        if writer is not None:
            def capture_next():
                return cached[0]
        elif replay and replay.get("v3_step_aligned"):
            # The recorded commands use no live RGB. Check the real physics clock every control step.
            def capture_next():
                return None, {"sensor_sim_time_s": clock()["current_time"]}
        rollout = execute(initial_frame=frame, capture=capture_next, source=source, feedback=feedback, apply=apply,
                          lift=lift, config=cfg)
        trial["policy_rollout"] = rollout
        if writer is not None and rollout["success"] and rollout["part_hit"]:
            trial["dataset_receipt"] = writer.save(output / "episode.h5", trial["strict_score"])
        target_motion["pre_lift_toppled"] = target_motion["max_pre_lift_tilt_deg"] >= 15.0
        for row in fixture_motion.values():
            row["toppled"] = row["max_tilt_deg"] >= 15.0
        trial["diagnostics"] = {"collision": collision_by_phase, "target": target_motion,
                                "fixtures": fixture_motion, "safety": dict(safety)}
        if policy is not None:
            trial["diagnostics"]["collision_phase_contract"] = "pretrial_approach_then_lift_and_hold"
            trial["diagnostics"]["policy_execution"] = {"actions_per_decision":1 if temporal is not None else 2,
                                                        "temporal_decay":te_decay}
        if replay:
            trigger_step = rollout["trials"][0]["trigger"]["step"] if rollout["trials"] else None
            max_arm, max_width, command_error = np.max(replay_errors, axis=0).tolist()
            relative_times = [row["image_time_s"] - rollout["decisions"][0]["image_time_s"]
                              for row in rollout["decisions"][:len(replay_errors)]]
            recorded_times = replay["times_s"][:len(relative_times)] - replay["times_s"][0]
            trial["replay_alignment"] = {"source": replay["source"], "transitions": len(replay_errors),
                                         "recorded_lift_step": replay["lift_index"], "trigger_step": trigger_step,
                                         "trigger_delta_s": (trigger_step - replay["lift_index"]) / CONTROL_HZ
                                         if trigger_step is not None else None,
                                         "max_timestamp_error_s": float(np.max(np.abs(relative_times - recorded_times))),
                                         "max_arm_error_rad": max_arm, "max_jaw_error_m": max_width,
                                         "max_native_setpoint_error": command_error}
        return rollout["success"]

    try:
        for _ in range(12):
            block("settle", record=False)
        observe("settle")
        if trace[-1].forbidden_contact or not trace[-1].target_supported:
            raise RuntimeError(f"Contact monitor baseline failed: target_supported={trace[-1].target_supported}, "
                               f"forbidden={trace[-1].forbidden_contact}, pairs={observer.last_pairs}")
        frame = capture("expert_initial.png")
        initial = np.array(target.data.root_link_pose_w[0].tolist())
        trial["target_initial_xyz_xyzw"] = initial.tolist()
        if visibility:
            prompt = visibility(initial)
            trial["initial_prompt"] = {key: value for key, value in prompt.items() if not key.endswith("_mask_packed")}
        if dense:
            record_observation(*frame, prompt)
        if plan:
            planned_target = np.asarray(plan["target_pose_xyz_xyzw"])
            target_error = np.linalg.norm(initial[:3] - planned_target[:3])
            angle_error = (Rotation.from_quat(initial[3:]).inv() * Rotation.from_quat(planned_target[3:])).magnitude()
            q_error = np.max(np.abs(np.asarray(robot.data.joint_pos[0, arm_ids].tolist()) - plan["home_q"]))
            if target_error > 0.0025 or angle_error > math.radians(2.5) or q_error > 0.012:
                raise RuntimeError(f"Settled scene differs from offline plan: object {target_error:.4f} m / {angle_error:.4f} rad, q {q_error:.4f} rad")
            if [names[index] for index in arm_ids] != plan["right_joint_names"]:
                raise RuntimeError("Planner and PhysX right arm order differ")
            if unified_lift_gate is not None:
                trial["diagnostic_passed"] = grasp_unified(frame, prompt)
                return trial
            if policy is not None:
                trial["diagnostic_passed"] = (policy_grasp_three_view(frame, prompt) if three_view_steps is not None
                                              else policy_grasp(frame, prompt))
                return trial
            trial["grasp_wrist_xyz"] = plan["wrist_xyz_xyzw"][:3]
            if replay is not None:
                trial["diagnostic_passed"] = replay_commands(frame)
                if dense and trial["diagnostic_passed"]:
                    trial["dataset_receipt"] = dense.write_success(output / "episode.h5", trial["case"], trial["strict_score"])
                return trial
            joint_path(plan["pre_grasp_q"], "pre_grasp")
            joint_path(plan["approach_q"][1:], "approach")
        else:
            orientation = Rotation.from_euler("x", -60, degrees=True) * Rotation.from_euler("z", 90, degrees=True)
            grip_point = initial[:3] + np.array([0, 0, bounds.GetMax()[2] - 0.025])
            goal = grip_point - orientation.apply([0.12, 0, 0])
            trial["grasp_wrist_xyz"] = goal.tolist()
            hover = goal + [0, -0.05, 0.075]
            start = pose()
            interpolate = Slerp([0, 1], Rotation.from_quat([start[3:], orientation.as_quat()]))
            for fraction in (0.25, 0.5, 0.75, 1.0):
                drive(start[:3] * (1 - fraction) + hover * fraction, interpolate(fraction), "pre_grasp", 28)
            drive(goal, orientation, "approach", 40)
        stable_contact_blocks = 0
        for _ in range(20):
            block("close")
            bilateral = (all(sample.target_part_pads == {"pad_1", "pad_2"} for sample in trace[-12:])
                         and min(observer.last_pad_forces_n.get(key, 0.0) for key in ("pad_1", "pad_2")) >= 1.0)
            stable_contact_blocks = stable_contact_blocks + 1 if bilateral else 0
            if stable_contact_blocks >= 2:
                break
        else:
            raise RuntimeError("No sustained bilateral target-part pressure during closure")
        lift_start = len(trace) - 1
        trial["lift_start_sample"] = lift_start
        if plan:
            fixed_lift(plan["trial_lift_ik_only_q"])
        else:
            lift_goal = pose()[:3] + [0, 0, 0.03]
            drive(lift_goal, orientation, "lift", 30)
        hold_start = len(trace) - 1
        trial["hold_start_sample"] = hold_start
        for _ in range(11):
            block("hold")
        trial["baseline_contract"] = "stable_support_v2"
        trial["strict_score"] = evaluate(trace, lift_start=lift_start, hold_start=hold_start, dt=dt,
                                         baseline_contract=trial["baseline_contract"])
        trial["diagnostic_passed"] = trial["strict_score"]["success"]
        if dense and trial["diagnostic_passed"]:
            trial["dataset_receipt"] = dense.write_success(output / "episode.h5", trial["case"], trial["strict_score"])
    except Exception as error:
        trial["diagnostic_passed"] = False
        trial["failure"] = f"{type(error).__name__}: {error}"
        if "policy_rollout" in trial:
            status = getattr(error, "status", "safety_stop" if "Forbidden physical contact" in str(error) else "unverified")
            trial["policy_rollout"].update(status=status, reason=str(error))
    finally:
        trial["sample_count"] = len(trace)
        trial["callback_headers_seen"] = observer.events_seen
        trial["forbidden_contact_pairs"] = dict(unexpected_pairs)
        trial["any_support_evidence"] = any(sample.target_supported for sample in trace)
        (output / "expert_diagnostic.json").write_text(json.dumps(trial, indent=2), encoding="utf-8")
        with (output / "physics_trace.jsonl").open("x", encoding="utf-8") as receipt:
            for sample in trace:
                fields = asdict(sample)
                fields["target_part_pads"] = sorted(sample.target_part_pads)
                fields["target_inner_pads"] = sorted(sample.target_inner_pads)
                receipt.write(json.dumps(fields) + "\n")
    return trial
