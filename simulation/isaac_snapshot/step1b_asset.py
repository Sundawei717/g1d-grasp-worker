import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from grasp_regions import BASKET_HANDLE, BASKET_RIM, BOX_NEG_END, BOX_POS_END
from three_view_detector import sha256, write_json


ROOT = Path(__file__).resolve().parent
FACE_NAMES = {"0": "other_surface", "1": BASKET_HANDLE, "2": BASKET_RIM}


def basket_parts(short_rim=False):
    import trimesh

    def box(size, center):
        return trimesh.creation.box(extents=size).apply_translation(center)

    # A watertight rectangular cross-section swept along an open arch, with a flat crown.
    arch = np.array([[-.047, .032], [-.047, .062], [-.034, .084], [-.020, .097],
                     [.020, .097], [.034, .084], [.047, .062], [.047, .032]])
    tangent = np.gradient(arch, axis=0)
    normals = tangent[:, ::-1] * [-1, 1] / np.linalg.norm(tangent, axis=1)[:, None]
    vertices = [[x + side*n[0]*.0045, depth*.0060, z + side*n[1]*.0045]
                for (x, z), n in zip(arch, normals, strict=True)
                for side, depth in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
    faces = [(0, 2, 1), (0, 3, 2)]
    for index in range(len(arch) - 1):
        for edge in range(4):
            a, b = 4*index + edge, 4*index + (edge+1) % 4
            faces.extend(((a, b, b+4), (a, b+4, a+4)))
    last = 4 * (len(arch) - 1)
    faces.extend(((last, last+1, last+2), (last, last+2, last+3)))
    handle = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    handle.fix_normals()
    if not handle.is_volume:
        raise ValueError("Basket handle sweep must be a closed positive volume")

    yield "base", box((.108, .080, .006), (0, 0, .003)), 0
    for side in (-1, 1):
        end = "neg" if side < 0 else "pos"
        yield f"side_{end}", box((.106, .004, .032), (0, side*.038, .019)), 0
        yield f"end_{end}", box((.010, .072, .039), (side*.049, 0, .0225)), 0
        rim_z = .068 if short_rim else .046
        yield f"rim_{end}", box((.069 if short_rim else .108, .011 if short_rim else .008, .008),
                                (0, side*.054, rim_z)), 2
        for bridge in (-1, 1):
            label = f"{end}_{'f' if bridge > 0 else 'b'}"
            x = bridge * (.031 if short_rim else .046)
            yield f"rim_bridge_{label}", box((.014 if short_rim else .018, .024, .008),
                                            (x, side*.044, .037 if short_rim else .0425)), 2
            if short_rim:
                yield f"rim_post_{label}", box((.013, .011, .029), (x, side*.054, .0525)), 2
    yield "handle", handle, 1


def prepare_basket(output, short_rim):
    import coacd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    output.mkdir(parents=True, exist_ok=True)
    asset = output / "basket.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/root").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.085)
    material = UsdShade.Material.Define(stage, "/root/BluePolymer")
    shader = UsdShade.Shader.Define(stage, "/root/BluePolymer/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    from pxr import Sdf
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.015, .38, .68))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.66)
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    UsdGeom.Xform.Define(stage, "/root/visual")
    UsdGeom.Xform.Define(stage, "/root/collision")

    def mesh(path, vertices, faces):
        prim = UsdGeom.Mesh.Define(stage, path)
        prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, dtype=np.float32)))
        prim.CreateFaceVertexCountsAttr([3] * len(faces))
        prim.CreateFaceVertexIndicesAttr(np.asarray(faces).reshape(-1).tolist())
        prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        return prim

    vertices, faces, labels, offset = [], [], [], 0
    decompositions = {}
    options = {"threshold": .035, "preprocess_mode": "auto", "preprocess_resolution": 45,
               "resolution": 1600, "mcts_nodes": 14, "mcts_iterations": 80, "seed": 19}
    for name, base, part in basket_parts(short_rim):
        visual = mesh(f"/root/visual/{name}", base.vertices, base.faces)
        UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)
        hulls = coacd.run_coacd(coacd.Mesh(base.vertices, base.faces), **options)
        decompositions[name] = len(hulls)
        for i, (xyz, triangles) in enumerate(hulls):
            collider = mesh(f"/root/collision/{name}_{i:02}", xyz, triangles)
            collider.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr("convexHull")
        vertices.extend(base.vertices)
        faces.extend(base.faces + offset)
        labels.extend([part] * len(base.faces))
        offset += len(base.vertices)
    stage.GetRootLayer().Save()
    atlas = output / "basket_faces.npz"
    np.savez_compressed(atlas, vertices=vertices, faces=faces, labels=np.asarray(labels, dtype=np.uint8))
    write_json(output / "basket.json", {"asset": str(asset), "asset_sha256": sha256(asset), "atlas_sha256": sha256(atlas),
                                        "part_meshes": {"face_atlas": str(atlas), "face_names": FACE_NAMES,
                                                        "rim_center_z_m": .068 if short_rim else .046},
                                        "source": "parametric metre-scale basket created in step1b_asset.py",
                                        "rim_style": "short reinforced center rim" if short_rim else "continuous side rim",
                                        "coacd": options, "hulls_by_component": decompositions,
                                        "parts": {v: labels.count(int(k)) for k, v in FACE_NAMES.items()}})


def prepare_carton(output):
    from part_surface import load_collision_surfaces
    import trimesh

    asset = ROOT / "assets/sand3d_packaging/00208/object.usdz"
    original = trimesh.util.concatenate(list(load_collision_surfaces(str(asset)).values()))
    # This mesh has large coplanar triangles across the long face; subdivide before assigning regions.
    vertices, faces = trimesh.remesh.subdivide_to_size(original.vertices, original.faces, max_edge=.008, max_iter=8)
    centers = vertices[faces].mean(axis=1)[:, 1]
    labels = np.where(centers <= -.028, 1, np.where(centers >= .028, 2, 0)).astype(np.uint8)
    atlas = output / "carton_faces.npz"
    np.savez_compressed(atlas, vertices=vertices, faces=faces, labels=labels)
    write_json(output / "carton.json", {"asset": str(asset), "asset_sha256": sha256(asset), "atlas_sha256": sha256(atlas),
                                        "part_meshes": {"face_atlas": str(atlas),
                                                        "face_names": {"0": "center", "1": BOX_NEG_END, "2": BOX_POS_END}}})


def check_physx(output, device):
    from isaaclab.app import AppLauncher
    from projects.g1d_rsi.sim60_legacy_bridge import install_import_bridge

    install_import_bridge()
    cfg = json.loads((ROOT / "env_resolved.json").read_text())
    geometry = json.loads((output / "basket.json").read_text())
    with TemporaryDirectory(prefix="basket_physics_", dir="/dev/shm") as kit:
        app = AppLauncher(headless=True, enable_cameras=False, device=device, experience=cfg["experience"],
                          kit_args=f"--portable-root {kit} --/renderer/multiGpu/enabled=false").app
        try:
            import carb
            import omni.usd
            from isaaclab.assets import RigidObject, RigidObjectCfg
            from isaaclab.sim import SimulationCfg, SimulationContext
            from omni.physx import get_physx_scene_query_interface
            from projects.g1d_rsi.sim60_legacy_bridge import bind_lab3_core_backend
            from pxr import Gf, UsdGeom, UsdPhysics

            bind_lab3_core_backend(device)
            sim = SimulationContext(SimulationCfg(device=device, dt=1/600, enable_scene_query_support=True))
            carb.settings.get_settings().set_bool("/physics/enableSceneQuerySupport", True)
            scene = omni.usd.get_context().get_stage()
            frame = UsdGeom.Xform.Define(scene, "/World/BasketMount")
            UsdGeom.XformCommonAPI(frame).SetTranslate(Gf.Vec3d(0, 0, 1))
            body = UsdGeom.Xform.Define(scene, "/World/BasketMount/Body")
            body.GetPrim().GetReferences().AddReference(geometry["asset"])
            UsdPhysics.RigidBodyAPI(body.GetPrim()).CreateKinematicEnabledAttr().Set(True)
            actor = RigidObject(RigidObjectCfg(prim_path=str(body.GetPath())))
            control = UsdGeom.Cube.Define(scene, "/World/PositiveControl")
            control.CreateSizeAttr(.04)
            UsdGeom.XformCommonAPI(control).SetTranslate(Gf.Vec3d(1, 0, 1))
            UsdPhysics.CollisionAPI.Apply(control.GetPrim())
            sim.reset()
            for _ in range(4):
                sim.step(render=False)
            actor.update(sim.get_physics_dt())
            origin = np.asarray(actor.data.root_link_pos_w[0].tolist())
            query = get_physx_scene_query_interface()

            def ray(start, direction, length):
                result = query.raycast_closest(tuple(origin + start), direction, length)
                return (str(result["collision"]), result["distance"]) if result.get("hit") else None

            # The raised rim crosses the lower sightline. Probe the handle aperture above it.
            checks = {"hole_rays": [ray((x, -.1, z), (0, 1, 0), .2)
                                    for x, z in ((0, .075), (.020, .078), (-.020, .078), (0, .082))],
                      "low_rim_rays": [ray((x, -.1, .068), (0, 1, 0), .2) for x in (-.020, .020)],
                      "interior_rays": [ray((x, y, .145), (0, 0, -1), .16)
                                        for x, y in ((0, .021), (.020, -.021), (-.020, .020))],
                      "handle_control": ray((0, -.1, .097), (0, 1, 0), .2),
                      "rim_control": ray((0, .054, .145), (0, 0, -1), .16),
                      "cube_control": ray((1, -.1, 0), (0, 1, 0), .2)}
            passed = (all(hit is None for hit in checks["hole_rays"])
                      and all(hit and "/base_" in hit[0] and abs(hit[1] - .139) < .003 for hit in checks["interior_rays"])
                      and all(checks[k] for k in ("handle_control", "rim_control", "cube_control")))
            write_json(output / "physx.json", {**checks, "passed": passed})
            print(checks, "passed", passed, flush=True)
            if not passed:
                raise RuntimeError("Basket collider failed geometry acceptance")
        finally:
            app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--physx", action="store_true")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--short-rim", action="store_true")
    args = parser.parse_args()
    if args.physx:
        check_physx(args.output.resolve(), args.device)
    else:
        prepare_basket(args.output.resolve(), args.short_rim)
        prepare_carton(args.output.resolve())
