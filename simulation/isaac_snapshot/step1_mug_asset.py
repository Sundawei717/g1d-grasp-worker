import argparse
import json
from pathlib import Path

import numpy as np

from three_view_detector import sha256, write_json


PROBE_XZ_M = ((0.033, 0.0), (0.036, 0.0), (0.039, 0.0), (0.036, 0.006), (0.036, -0.006))


def prepare(source, output):
    import coacd
    import trimesh
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    output.mkdir(parents=True)
    stage = Usd.Stage.Open(str(source))
    visual = UsdGeom.Mesh(stage.GetPrimAtPath("/root/visual/model"))
    collision = UsdGeom.Mesh(stage.GetPrimAtPath("/root/collision/model"))
    vertices = np.array(collision.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.array(collision.GetFaceVertexCountsAttr().Get())
    if np.any(counts != 3):
        raise ValueError("Expected original triangle geometry")
    triangles = np.array(collision.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    # Leave the two body/handle joins unlabeled for Part Hit.
    face_x = vertices[triangles].mean(axis=1)[:, 0]
    labels = np.where(face_x >= 0.019, 2, np.where(face_x <= 0.012, 1, 0)).astype(np.uint8)
    atlas = output / "part_faces.npz"
    np.savez_compressed(atlas, vertices=vertices, faces=triangles, labels=labels)
    source_mesh = trimesh.Trimesh(vertices, triangles, process=False)
    if not visual or not np.allclose(np.array(visual.GetPointsAttr().Get()).min(0), vertices.min(0), atol=5e-4):
        raise ValueError("Original visual and physical meshes use different coordinates")

    # Millimetre-scale simplification for the decomposition. Contact labels remain on the original triangles.
    keys = np.round(vertices / 0.0005).astype(np.int32)
    _, indices, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    reduced = inverse[triangles]
    keep = (reduced[:, 0] != reduced[:, 1]) & (reduced[:, 0] != reduced[:, 2]) & (reduced[:, 1] != reduced[:, 2])
    simplified = trimesh.Trimesh(vertices[indices], reduced[keep], process=True)
    simplified.update_faces(simplified.unique_faces())
    simplified.remove_unreferenced_vertices()
    options = dict(threshold=0.025, preprocess_mode="auto", preprocess_resolution=50, resolution=1400,
                   mcts_nodes=12, mcts_iterations=60, mcts_max_depth=3, merge=True, seed=19)
    hulls = coacd.run_coacd(coacd.Mesh(simplified.vertices, simplified.faces), **options)
    origins = [[x, -0.08, z] for x, z in PROBE_XZ_M]
    directions = [[0, 1, 0]] * len(origins)
    raw_blocks = source_mesh.ray.intersects_any(origins, directions)
    blocked = np.zeros(len(origins), dtype=bool)
    for verts, faces in hulls:
        poly = trimesh.Trimesh(verts, faces, process=False)
        blocked |= poly.ray.intersects_any(origins, directions)
    edited = output / "mug_collision_v3.usda"
    target = Usd.Stage.CreateNew(str(edited))
    UsdGeom.SetStageUpAxis(target, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(target, 1)
    root = UsdGeom.Xform.Define(target, "/root").GetPrim()
    root.GetReferences().AddReference(str(source), Sdf.Path("/root"))
    target.SetDefaultPrim(root)
    target.OverridePrim("/root/collision/model").SetActive(False)
    for i, (verts, faces) in enumerate(hulls):
        shape = UsdGeom.Mesh.Define(target, f"/root/collision/hull_{i:03}")
        shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(verts, dtype=np.float32)))
        shape.CreateFaceVertexCountsAttr([3] * len(faces))
        shape.CreateFaceVertexIndicesAttr(np.asarray(faces).reshape(-1).tolist())
        shape.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        shape.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr("convexHull")
    target.GetRootLayer().Save()
    write_json(output / "result.json", {"source": str(source), "source_sha256": sha256(source), "asset": str(edited),
                                        "atlas": str(atlas), "atlas_sha256": sha256(atlas), "parts": {
                                            "mug_body": int((labels == 1).sum()), "mug_handle": int((labels == 2).sum()),
                                            "unlabeled_joins": int((labels == 0).sum())}, "hulls": len(hulls),
                                        "probe_xz_m": PROBE_XZ_M, "original_mesh_blocked": raw_blocks.tolist(),
                                        "decomposed_geometry_blocked": blocked.tolist(), "coacd_cfg": options})
    print("CoACD hulls", len(hulls), "open rays", int((~blocked).sum()), flush=True)


def check_physx(source, output, device):
    from tempfile import TemporaryDirectory

    from isaaclab.app import AppLauncher
    from projects.g1d_rsi.sim60_legacy_bridge import install_import_bridge

    install_import_bridge()
    root = Path(__file__).resolve().parent
    cfg = json.loads((root / "env_resolved.json").read_text())
    geometry = json.loads((output / "result.json").read_text())
    with TemporaryDirectory(prefix="mug_hole_", dir="/dev/shm") as kit:
        app = AppLauncher(headless=True, enable_cameras=False, device=device, experience=cfg["experience"],
                          kit_args=f"--portable-root {kit} --/renderer/multiGpu/enabled=false").app
        try:
            import omni.usd
            import torch
            import carb
            from isaaclab.assets import RigidObject, RigidObjectCfg
            from isaaclab.sim import SimulationCfg, SimulationContext
            from omni.physx import get_physx_scene_query_interface
            from projects.g1d_rsi.sim60_legacy_bridge import bind_lab3_core_backend
            from pxr import Gf, UsdGeom, UsdPhysics

            bind_lab3_core_backend(device)
            sim = SimulationContext(SimulationCfg(device=device, dt=1 / 600, enable_scene_query_support=True))
            carb.settings.get_settings().set_bool("/physics/enableSceneQuerySupport", True)
            stage = omni.usd.get_context().get_stage()
            targets = {}
            for name, x, asset in (("original", 0, str(source)), ("coacd", 1, geometry["asset"])):
                placement = UsdGeom.Xform.Define(stage, f"/World/{name}")
                UsdGeom.XformCommonAPI(placement).SetTranslate(Gf.Vec3d(x, 0, 1))
                body = UsdGeom.Xform.Define(stage, f"/World/{name}/Body")
                body.GetPrim().GetReferences().AddReference(asset)
                UsdPhysics.RigidBodyAPI(body.GetPrim()).CreateKinematicEnabledAttr().Set(True)
                targets[name] = RigidObject(RigidObjectCfg(prim_path=str(body.GetPath())))
            block = UsdGeom.Cube.Define(stage, "/World/PositiveControl")
            block.CreateSizeAttr(0.04)
            UsdGeom.XformCommonAPI(block).SetTranslate(Gf.Vec3d(2, 0, 1))
            UsdPhysics.CollisionAPI.Apply(block.GetPrim())
            sim.reset()
            for _ in range(4):
                sim.step(render=False)
            for actor in targets.values():
                actor.update(sim.get_physics_dt())
            query = get_physx_scene_query_interface()
            results = {}
            def convert(hit):
                return str(hit.get("collision")) if hit.get("hit") else None

            for name, actor in targets.items():
                position = tuple(actor.data.root_link_pos_w[0].tolist())
                def shot(x, z):
                    hit = query.raycast_closest((position[0] + x, position[1] - 0.08, position[2] + z),
                                                (0, 1, 0), 0.16)
                    return convert(hit)

                results[name] = {"hole_rays": [shot(x, z) for x, z in PROBE_XZ_M],
                                 "material_rays": [shot(0.054, 0), shot(-0.020, 0)], "position": position}
            control = convert(query.raycast_closest((2, -0.08, 1), (0, 1, 0), 0.16))
            passed = (bool(control) and all(all(row["material_rays"]) for row in results.values())
                      and any(hit is None for hit in results["coacd"]["hole_rays"]))
            write_json(output / "physx.json", {**results, "positive_control": control, "passed": passed})
            print(results, flush=True)
        finally:
            app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--physx", action="store_true")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.physx:
        check_physx(args.source.resolve(), args.output.resolve(), args.device)
    else:
        prepare(args.source.resolve(), args.output.resolve())
