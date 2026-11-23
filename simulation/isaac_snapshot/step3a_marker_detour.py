"""Search safe, genuinely blocking layouts for the upright marker package."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import trimesh

from background_status import record_complete
from step3a_final_substitutes import candidate
from step3a_planner import adapt, placement
from step3a_suites import RUNS, WORK, read, verify_groups
from step3a_verify import file_hash

JOB = "step3a_01_marker_detour"


def obstacle_asset():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    path = WORK / "tall_blue_panel.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/root").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.145)
    mesh = trimesh.creation.box((.063, .033, .145)).apply_translation([0, 0, .0725])
    geom = UsdGeom.Mesh.Define(stage, "/root/Panel")
    geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertices.astype(np.float32)))
    geom.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    geom.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
    geom.CreateDisplayColorAttr([Gf.Vec3f(.04, .24, .53)])
    geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexHull")
    stage.GetRootLayer().Save()
    return str(path)


def block(plan):
    source = plan["case"]
    path = str(WORK / "tall_blue_panel.usda")
    digest = file_hash(path)
    x, y = source["target_pose_xyz_xyzw"][:2]
    for radius in (.11, .135, .16):
        for heading in (-120, -90, -60, -150, -30, 0, -180, 60, 120, 90):
            dx, dy = radius * np.array([np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))])
            if not (.014 <= x + dx <= .355 and -.565 <= y + dy <= -.28):
                continue
            pose = placement(path, x + dx, y + dy, heading + 90)
            fixture = {"name": "blocker", "usd": path, "sha256": digest, "pose": pose,
                       "asset_family": "step3a/tall_blue_panel"}
            row = {**source, "fixtures": [fixture], "group": "B_obstacle",
                   "layout_description": "upright marker package with a verified blocking panel"}
            if (result := adapt(plan, row, block=True)) is not None:
                return result
    return None


def main():
    if not (WORK / "tall_blue_panel.usda").exists():
        obstacle_asset()
    marker = read(WORK / "asset_substitutes.json")["assets"]["marker_long_box"]
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as planning:
        futures = [planning.submit(candidate, marker, s, 4 + i, block)
                   for i, s in enumerate(("C_9844", "C_9846"))]
        groups = [group for f in futures if (group := f.result()) is not None]
    print("marker package", len(groups), "forced detour groups", flush=True)
    if not groups:
        raise RuntimeError("No layout had both a blocked direct path and an admissible detour")
    result = verify_groups(groups, "pilots_guide_marker_detour")
    if len(result["cases"]) != 4:
        raise RuntimeError(f"Only {len(result['cases'])}/4 marker-package episodes passed native replay")


if __name__ == "__main__":
    code = 1
    try:
        main()
        code = 0
    finally:
        record_complete(RUNS / JOB, code)
