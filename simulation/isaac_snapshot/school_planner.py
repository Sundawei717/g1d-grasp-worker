import argparse
import hashlib
import itertools
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import coal
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from grasp_regions import PARTS, UPPER_PART, candidate_depths


ARM_NAMES = tuple(f"right_{joint}_joint" for joint in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"))
PAD_NAMES = {"right_dex1_finger_link_1_tip", "right_dex1_finger_link_2_tip"}
PAD_OFFSET = np.array([0.12, 0, 0])
CHECK_STEP = 0.007


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose_matrix(pose):
    return pin.SE3(Rotation.from_quat(pose[3:]).as_matrix(), np.asarray(pose[:3], dtype=float))


def serial_pose(pose):
    return [*pose.translation.tolist(), *Rotation.from_matrix(pose.rotation).as_quat().tolist()]


def as_coal(pose):
    return coal.Transform3s(pose.rotation, pose.translation)


def mesh_hull(vertices):
    points = coal.StdVec_Vec3s()
    for vertex in vertices:
        points.append(np.asarray(vertex, dtype=float))
    return coal.ConvexBase.convexHull(points, False, "")


def authored_meshes(path):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if not stage or UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError(f"Expected metre/Z-up USD: {path}")
    cache = UsdGeom.XformCache()
    meshes = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if not prim.IsA(UsdGeom.Mesh):
                raise ValueError(f"Unmodelled collider: {prim.GetPath()}")
            transform = cache.GetLocalToWorldTransform(prim)
            points = np.array([tuple(transform.Transform(point)) for point in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
            meshes.append((str(prim.GetPath()), points))
    if not meshes:
        raise ValueError(f"No authored collision meshes: {path}")
    return meshes


class Planner:
    def __init__(self, config, urdf, evidence, target_pose, *, obstacles=(), home=None, expanded_search=False, guide=(),
                 part=UPPER_PART):
        self.config = config
        self.expanded_search = expanded_search
        self.part = part
        self.guide = [np.asarray(point, dtype=float) for point in guide]
        self.urdf = Path(urdf)
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.geometry = pin.buildGeomFromUrdf(self.model, str(urdf), pin.GeometryType.COLLISION,
                                              package_dirs=[str(self.urdf.parent)])
        self.geom_data = self.geometry.createData()
        self.root = pin.SE3(Rotation.from_euler("z", config["root_yaw_deg"], degrees=True).as_matrix(),
                            np.asarray(config["root_position_m"], dtype=float))
        self.arm_indices = np.array([self.idx(name) for name in ARM_NAMES])
        self.fingers = np.array([self.idx(f"right_dex1_finger_joint_{index}") for index in (1, 2)])
        self.left = [name for name in evidence["joint_names"] if name.startswith("left_")]
        if len(self.left) != 9 or tuple(evidence["right_arm"]) != ARM_NAMES or self.model.nq != 22:
            raise RuntimeError("Pinned G1D joint order is unsupported")
        self.lower = self.model.lowerPositionLimit[self.arm_indices] + 0.025
        self.upper = self.model.upperPositionLimit[self.arm_indices] - 0.025
        self.arm_id = self.model.getFrameId("right_dex1_base_link")
        self.q_ref = pin.neutral(self.model)
        self.row = evidence["samples"][0]
        for name, value in zip(self.left, self.row["left_rad_or_m"]):
            self.q_ref[self.idx(name)] = value
        self.q_ref[self.fingers] = self.row["right_fingers_m"]
        self.home = np.array(self.row["right_rad"] if home is None else home)
        errors = []
        for name, key in (("right_dex1_base_link", "right_ee_xyz_xyzw"), ("head_link", "head_pose_w_xyzw")):
            predicted = self.world_frame(self.row["right_rad"], self.model.getFrameId(name))
            truth = pose_matrix(self.row[key])
            position = np.linalg.norm(predicted.translation - truth.translation)
            rotation = Rotation.from_matrix(predicted.rotation @ truth.rotation.T).magnitude()
            errors.append({"frame": name, "position_m": float(position), "rotation_rad": float(rotation)})
            if position > 0.002 or rotation > 0.003:
                raise RuntimeError(f"URDF FK disagrees with native Isaac pose: {errors[-1]}")
        self.fk_errors = errors
        school_pose = pin.SE3(np.eye(3), np.array(config["school_position_m"], dtype=float))
        self.fixture_names = []
        self.fixtures = []
        for name, points in authored_meshes(config["school_usd"]):
            self.add_fixture(name, mesh_hull(points), school_pose)
        if target_pose is not None:
            self.target_pose = pose_matrix(target_pose)
            meshes = authored_meshes(config["target_usd"])
            self.target_points = np.concatenate([mesh[1] for mesh in meshes])
            self.target_lower, self.target_upper = self.target_points.min(axis=0), self.target_points.max(axis=0)
            for name, points in meshes:
                self.add_fixture("target:" + name, mesh_hull(points), self.target_pose)
        for obstacle in obstacles:
            for name, points in authored_meshes(obstacle["usd"]):
                self.add_fixture(obstacle["name"] + ":" + name, mesh_hull(points), pose_matrix(obstacle["pose"]))
        self.make_collision_pairs()
        self.checks = 0

    def idx(self, name):
        joint = self.model.getJointId(name)
        if joint == 0 or joint >= self.model.njoints:
            raise ValueError(f"Missing pinned URDF joint: {name}")
        return self.model.joints[joint].idx_q

    def full_q(self, arm):
        joints = self.q_ref.copy()
        joints[self.arm_indices] = arm
        return joints

    def world_frame(self, arm, frame_id=None):
        pin.framesForwardKinematics(self.model, self.data, self.full_q(arm))
        return self.root * self.data.oMf[self.arm_id if frame_id is None else frame_id]

    def add_fixture(self, name, geometry, pose):
        geometry.computeLocalAABB()
        self.fixture_names.append(name)
        self.fixtures.append((geometry, pose))

    def make_collision_pairs(self):
        robot_urdf = ET.parse(self.urdf).getroot()
        ignored = {frozenset((joint.find("parent").get("link"), joint.find("child").get("link")))
                   for joint in robot_urdf.findall("joint")}
        contract = json.loads(self.urdf.with_name("physics_contract.json").read_text())
        ignored.update(frozenset((first, other)) for first, others in contract["filtered_pairs"].items() for other in others)
        ignored.update(frozenset(pair) for pair in contract["additional_internal_slider_carrier_pairs"])
        arm_joints = {self.model.getJointId(name) for name in ARM_NAMES}

        def moves(joint):
            while joint:
                if joint in arm_joints:
                    return True
                joint = self.model.parents[joint]
            return False

        owners = [self.model.frames[shape.parentFrame].name for shape in self.geometry.geometryObjects]
        robot_count = len(owners)
        moving = [index for index, shape in enumerate(self.geometry.geometryObjects) if moves(shape.parentJoint)]
        robot_shapes = []
        for shape in self.geometry.geometryObjects:
            geom = shape.geometry
            if isinstance(geom, coal.BVHModelBase):
                geom.buildConvexHull(False, "")
                geom = geom.convex
            geom.computeLocalAABB()
            robot_shapes.append(geom)
        self.robot_count = robot_count
        self.shape_names = owners + self.fixture_names
        self.shapes = robot_shapes + [shape for shape, _ in self.fixtures]
        self.fixture_poses = [pose for _, pose in self.fixtures]
        lo = np.array([shape.aabb_local.min_ for shape in self.shapes])
        hi = np.array([shape.aabb_local.max_ for shape in self.shapes])
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi < lo):
            raise RuntimeError("Collision hull has invalid AABB")
        self.centers = (hi + lo) / 2
        self.extents = (hi - lo) / 2
        pairs = {tuple(sorted((first, second))) for first in moving for second in range(len(self.shapes))
                 if first != second and self.shape_names[first] != self.shape_names[second]
                 and frozenset((self.shape_names[first], self.shape_names[second])) not in ignored}
        self.pairs = np.array(sorted(pairs), dtype=int)
        self.first, self.second = self.pairs.T
        self.is_pad_target = np.array([self.shape_names[first] in PAD_NAMES and self.shape_names[second].startswith("target:")
                                       for first, second in self.pairs])
        self.request = coal.CollisionRequest()

    def collision(self, arm, margin=0.002, allow_pad_target=False, held_pose=None, fingers=None):
        self.checks += 1
        joints = self.full_q(arm)
        if fingers is not None:
            joints[self.fingers] = fingers
        pin.updateGeometryPlacements(self.model, self.data, self.geometry, self.geom_data, joints)
        poses = [self.root * pose for pose in self.geom_data.oMg] + [
            held_pose if held_pose is not None and name.startswith("target:") else pose
            for name, pose in zip(self.fixture_names, self.fixture_poses)]
        rotations = np.array([pose.rotation for pose in poses])
        translations = np.array([pose.translation for pose in poses])
        centers = translations + (rotations @ self.centers[..., None])[..., 0]
        extents = (abs(rotations) @ self.extents[..., None])[..., 0]
        keep = np.all(abs(centers[self.first] - centers[self.second]) <= extents[self.first] + extents[self.second] + margin, axis=1)
        if allow_pad_target:
            keep &= ~self.is_pad_target
        self.request.security_margin = margin
        for index in np.flatnonzero(keep):
            first, second = self.pairs[index]
            close_finger_neighbor = (allow_pad_target and self.shape_names[first].startswith("right_dex1_finger_link_")
                                     and self.shape_names[first].endswith("_mid")
                                     and self.shape_names[second].startswith("target:"))
            self.request.security_margin = 0.0002 if close_finger_neighbor else margin
            if coal.collide(self.shapes[first], as_coal(poses[first]), self.shapes[second], as_coal(poses[second]),
                            self.request, coal.CollisionResult()):
                return (self.shape_names[first], self.shape_names[second])
        if held_pose is not None:
            self.request.security_margin = 0.0
            targets = [index for index in range(self.robot_count, len(self.shapes))
                       if self.shape_names[index].startswith("target:")]
            other = [index for index in range(self.robot_count, len(self.shapes)) if index not in targets]
            for first in targets:
                for second in other:
                    if self.shape_names[second] == "/Scene/Table/Top" and held_pose.translation[2] - self.target_pose.translation[2] < 0.003:
                        continue
                    if np.any(abs(centers[first] - centers[second]) > extents[first] + extents[second]):
                        continue
                    if coal.collide(self.shapes[first], as_coal(poses[first]), self.shapes[second], as_coal(poses[second]),
                                    self.request, coal.CollisionResult()):
                        return (self.shape_names[first], self.shape_names[second])
        return None

    def screen_lift(self, initial, lift, wrist):
        relative_target = wrist.inverse() * self.target_pose
        jaw_y = (self.target_points @ relative_target.rotation.T + relative_target.translation)[:, 1]
        left_edge, right_edge = float(jaw_y.min()), float(jaw_y.max())
        closing = np.clip(-0.02 + np.array([0.045034 + left_edge, 0.045034 - right_edge]),
                          -0.02, 0.023)
        for stroke in np.linspace(-0.02, closing, 12):
            collision = self.collision(initial, allow_pad_target=True, fingers=stroke)
            if collision:
                return collision
        for first, second in zip([initial, *lift], lift):
            intervals = max(1, math.ceil(np.linalg.norm(second - first, ord=1) / CHECK_STEP))
            for fraction in np.linspace(0, 1, intervals + 1)[1:]:
                joint = first + fraction * (second - first)
                held = self.world_frame(joint) * relative_target
                collision = self.collision(joint, allow_pad_target=True, held_pose=held, fingers=closing)
                if collision:
                    return collision
        return None

    def edge(self, start, stop, margin=0.002):
        count = max(1, math.ceil(np.linalg.norm(stop - start, ord=1) / CHECK_STEP))
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            contact = self.collision(start + fraction * (stop - start), margin)
            if contact:
                return contact
        return None

    def ik(self, goal, seed):
        def residual(angles):
            actual = self.world_frame(angles)
            rotation = Rotation.from_matrix(goal.rotation.T @ actual.rotation).as_rotvec()
            return np.r_[actual.translation - goal.translation, 0.18 * rotation]

        result = least_squares(residual, np.clip(seed, self.lower, self.upper), bounds=(self.lower, self.upper),
                               max_nfev=85, xtol=1e-7, ftol=1e-7, gtol=2e-7)
        error = residual(result.x)
        if np.linalg.norm(error[:3]) > 0.0025 or np.linalg.norm(error[3:]) > 0.18 * math.radians(2):
            return None, [float(np.linalg.norm(error[:3])), float(np.degrees(np.linalg.norm(error[3:]) / 0.18))]
        return result.x, None

    def route(self, start, goal, rng, margin=0.002, budget=135):
        direct = self.edge(start, goal, margin)
        if direct is None:
            return [start, goal], None
        if self.expanded_search:
            guided = self.route_via_guide(start, goal, margin)
            if guided:
                return guided, None
            alternative = self.route_bidirectional(start, goal, rng, margin, budget=185)
            return (alternative, None) if alternative else (None, direct)
        nodes = [start]
        parents = [-1]
        for trial in range(budget):
            sample = goal if trial % 5 == 0 else (
                np.clip((start + goal) / 2 + rng.normal(size=7) * 0.6, self.lower, self.upper)
                if trial < budget * 0.8 else rng.uniform(self.lower, self.upper))
            closest = int(np.argmin(np.linalg.norm(np.asarray(nodes) - sample, axis=1)))
            delta = sample - nodes[closest]
            distance = np.linalg.norm(delta)
            if distance < 1e-6:
                continue
            node = nodes[closest] + delta * min(1, 0.16 / distance)
            if self.edge(nodes[closest], node, margin):
                continue
            nodes.append(node)
            parents.append(closest)
            if np.linalg.norm(node - goal) < 0.55 and not self.edge(node, goal, margin):
                path = [goal, node]
                while closest >= 0:
                    path.append(nodes[closest])
                    closest = parents[closest]
                path.reverse()
                return path, None
        return None, direct

    def route_via_guide(self, start, goal, margin):
        if not self.guide:
            return None
        for guide_endpoint in range(len(self.guide) - 1, -1, -1):
            route = [start, *self.guide[:guide_endpoint + 1], goal]
            if any(np.any(point < self.lower) or np.any(point > self.upper) for point in route):
                continue
            if any(self.edge(first, second, margin) for first, second in zip(route, route[1:])):
                continue
            return route
        return None

    def route_bidirectional(self, start, goal, rng, margin, budget):
        trees = [([start], [-1]), ([goal], [-1])]

        def advance(tree, destination):
            values, parents = tree
            nearest = int(np.argmin(np.linalg.norm(np.asarray(values) - destination, axis=1)))
            delta = destination - values[nearest]
            distance = np.linalg.norm(delta)
            if distance < 1e-9:
                return nearest, True
            ratio = min(1.0, 0.18 / distance)
            endpoint = values[nearest] + delta * ratio
            if self.edge(values[nearest], endpoint, margin):
                return None, False
            values.append(endpoint)
            parents.append(nearest)
            return len(values) - 1, ratio == 1.0

        def trace(tree, index):
            values, parents = tree
            result = []
            while index >= 0:
                result.append(values[index])
                index = parents[index]
            return list(reversed(result))

        for attempt in range(budget):
            direction = attempt % 2
            current, other = trees[direction], trees[1 - direction]
            if attempt % 5 == 0:
                sample = other[0][0]
            elif attempt % 4:
                sample = np.clip((start + goal) / 2 + rng.normal(size=7) * 0.45, self.lower, self.upper)
            else:
                sample = rng.uniform(self.lower, self.upper)
            current_index, _ = advance(current, sample)
            if current_index is None:
                continue
            endpoint = current[0][current_index]
            for _ in range(12):
                other_index, connected = advance(other, endpoint)
                if other_index is None:
                    break
                if connected:
                    route = trace(current, current_index) + trace(other, other_index)[-2::-1]
                    if direction:
                        route.reverse()
                    return route
        return None

    def candidate_goals(self):
        center = (self.target_lower + self.target_upper) / 2
        narrow = int(np.argmin((self.target_upper - self.target_lower)[:2]))
        axis = self.target_pose.rotation[:, narrow]
        jaw_yaw = math.degrees(math.atan2(axis[1], axis[0]))
        tilts = (90, 75, 60, 45, 30) if self.expanded_search else (45, 60, 75, 30, 90)
        skews = (0, -15, -30, 15, 30, -45, -60, 45, 60, 90) if self.expanded_search else (0, 15, -15, 30, -30)
        depths = candidate_depths(self.part, self.target_lower[2], self.target_upper[2])
        for depth, tilt, skew, reverse in itertools.product(depths, tilts, skews, (0, 180)):
            center[2] = self.target_upper[2] - depth
            grip = self.target_pose.act(center)
            angles = (Rotation.from_euler("z", jaw_yaw + skew + reverse, degrees=True)
                      * Rotation.from_euler("x", -tilt, degrees=True) * Rotation.from_euler("z", 90, degrees=True))
            orient = angles.as_matrix()
            wrist = grip - angles.apply(PAD_OFFSET)
            hover = wrist - orient[:, 0] * 0.072
            if hover[2] < wrist[2] + 0.025:
                continue
            yield f"d{depth:.3f}_tilt{tilt}_skew{skew}_rev{reverse}", pin.SE3(orient, hover), pin.SE3(orient, wrist)

    def plan(self, seed=0, stop_after=4, max_candidates=None, preferred=(), excluded=(), lift_height_m=0.030):
        rng = np.random.default_rng(seed)
        result = {"candidate_log": [], "feasible": [], "home_contacts": self.collision(self.home)}
        if result["home_contacts"]:
            return result
        goals = [goal for goal in self.candidate_goals() if goal[0] not in excluded]
        if preferred:
            rank = {code: index for index, code in enumerate(preferred)}
            goals.sort(key=lambda item: rank.get(item[0], len(rank)))
        for code, hover, wrist in itertools.islice(goals, max_candidates):
            receipt = {"candidate": code, "hover_xyz_xyzw": serial_pose(hover), "wrist_xyz_xyzw": serial_pose(wrist)}
            result["candidate_log"].append(receipt)
            starts = (self.home, np.clip(self.home + [0.2, -0.4, 0.1, 0.6, 0.2, -0.2, 0.1], self.lower, self.upper),
                      self.lower + 0.65 * (self.upper - self.lower), rng.uniform(self.lower, self.upper))
            if self.guide:
                starts = (self.guide[-1], *starts)
            for start in starts:
                hover_q, error = self.ik(hover, start)
                if hover_q is None:
                    receipt["last_hover_error_m_deg"] = error
                    continue
                collision = self.collision(hover_q)
                if collision:
                    receipt["last_collision"] = collision
                    continue
                previous = hover_q
                approach_q = [hover_q]
                for fraction in np.linspace(0, 1, 13)[1:]:
                    step = pin.SE3(hover.rotation, hover.translation + fraction * (wrist.translation - hover.translation))
                    following, error = self.ik(step, previous)
                    if error:
                        receipt["last_approach_error_m_deg"] = error
                        break
                    collision = self.edge(previous, following)
                    if collision:
                        receipt["last_collision"] = collision
                        break
                    approach_q.append(following)
                    previous = following
                if len(approach_q) < 13:
                    continue
                trial_lift = []
                for height in np.linspace(lift_height_m / 4, lift_height_m, 4):
                    goal = pin.SE3(wrist.rotation, wrist.translation + [0.0, 0.0, height])
                    following, error = self.ik(goal, previous)
                    if error:
                        receipt["last_lift_error_m_deg"] = error
                        break
                    trial_lift.append(following)
                    previous = following
                if len(trial_lift) < 4:
                    continue
                collision = self.screen_lift(approach_q[-1], trial_lift, wrist)
                if collision:
                    receipt["last_offline_lift_collision"] = collision
                    continue
                route, reason = self.route(self.home, hover_q, rng)
                if route is None:
                    receipt["last_route_collision"] = reason
                    continue
                path = [*route, *approach_q[1:]]
                preferred_margin = 0.004 if all(not self.edge(first, second, 0.004)
                                                for first, second in zip(path, path[1:])) else 0.002
                length = float(sum(np.linalg.norm(second - first) for first, second in zip(path, path[1:])))
                record = {**receipt, "joint_margin_collision_check_m": preferred_margin, "path_length_joint_rad": length,
                          "pre_grasp_q": [point.tolist() for point in route],
                          "approach_q": [point.tolist() for point in approach_q],
                          "trial_lift_ik_only_q": [point.tolist() for point in trial_lift],
                          "trial_lift_offline_clearance_checked": True,
                          "joint_motion_rad": float(np.linalg.norm(path[-1] - self.home))}
                result["feasible"].append(record)
                receipt["valid"] = True
                break
            if len(result["feasible"]) >= stop_after:
                break
        result["feasible"].sort(key=lambda item: (-item["joint_margin_collision_check_m"], item["path_length_joint_rad"],
                                                   item["joint_motion_rad"], item["candidate"]))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("lab_evidence", type=Path)
    parser.add_argument("target_evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-after", type=int, default=4)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--settled-height-from-geometry", action="store_true")
    parser.add_argument("--obstacle-receipt", type=Path)
    parser.add_argument("--part", choices=PARTS, default=UPPER_PART)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    evidence = json.loads(args.lab_evidence.read_text())
    target_pose = json.loads(args.target_evidence.read_text())["target_initial_xyz_xyzw"]
    if args.settled_height_from_geometry:
        tabletop = next(points for name, points in authored_meshes(config["school_usd"]) if name == "/Scene/Table/Top")
        target_points = np.concatenate([points for _, points in authored_meshes(config["target_usd"])])
        rotated = Rotation.from_quat(target_pose[3:]).apply(target_points)
        target_pose[2] = float(np.max(tabletop[:, 2]) + config["school_position_m"][2] - np.min(rotated[:, 2]) + 0.0001)
    obstacles = []
    if args.obstacle_receipt:
        prior = json.loads(args.obstacle_receipt.read_text())
        if not prior["passed"] or not prior.get("blocking_direct_pair"):
            raise SystemExit("Expected a verified blocked baseline pre-grasp path")
        obstacle_usd = str(Path(prior["asset"]).resolve())
        obstacles.append({"name": "blocker", "pose": prior["obstacle_xyz_xyzw"],
                          "usd": obstacle_usd, "sha256": sha256(obstacle_usd)})
    planner = Planner(config, args.urdf, evidence, target_pose, obstacles=obstacles, part=args.part)
    result = planner.plan(args.seed, args.stop_after, args.max_candidates)
    result.update(urdf_sha256=sha256(args.urdf), target_sha256=sha256(config["target_usd"]),
                  school_sha256=sha256(config["school_usd"]), target_pose_xyz_xyzw=target_pose, fk_agreement=planner.fk_errors,
                  right_joint_names=list(ARM_NAMES), home_q=planner.home.tolist(), collision_checks=planner.checks,
                  max_interpolation_l1_rad=CHECK_STEP, seed=args.seed, part=args.part,
                  scope="offline URDF convex-hull planning with nominal held-object clearance; no physical certification")
    if obstacles:
        result["obstacles"] = obstacles
        result["blocked_baseline_receipt"] = str(args.obstacle_receipt.resolve())
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(f"{len(result['feasible'])} feasible paths, {planner.checks} collision states; home={result['home_contacts']}", flush=True)
    return int(not result["feasible"])


if __name__ == "__main__":
    sys.exit(main())
