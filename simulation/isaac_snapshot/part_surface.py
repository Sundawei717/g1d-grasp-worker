from functools import lru_cache

import numpy as np


@lru_cache(maxsize=4)
def load_face_atlas(path):
    import trimesh

    with np.load(path, allow_pickle=False) as data:
        return trimesh.Trimesh(data["vertices"], data["faces"], process=False), data["labels"]


@lru_cache(maxsize=8)
def load_collision_surfaces(asset):
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(asset)
    cache = UsdGeom.XformCache()
    result = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mesh = UsdGeom.Mesh(prim)
        transform = cache.GetLocalToWorldTransform(prim)
        vertices = [tuple(transform.Transform(point)) for point in mesh.GetPointsAttr().Get()]
        counts, indices = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
        offset, faces = 0, []
        for count in counts:
            face = indices[offset:offset + count]
            faces.extend((face[0], face[i], face[i + 1]) for i in range(1, count - 1))
            offset += count
        if faces:
            result[str(prim.GetPath())] = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not result:
        raise ValueError(f"No collision triangle mesh for {asset}")
    return result


def attribute_contacts(asset, contact_points, requested, band, named_meshes=None):
    from trimesh.proximity import closest_point_naive

    if named_meshes and "face_atlas" in named_meshes:
        from trimesh.proximity import closest_point

        atlas, labels = load_face_atlas(named_meshes["face_atlas"])
        name = named_meshes.get("face_names", {"0": "unlabeled_join", "1": "mug_body", "2": "mug_handle"})
        details = []
        if contact_points:
            points = [point["target_local_xyz_m"] for point in contact_points]
            nearest, distances, faces = closest_point(atlas, points)
            for point, xyz, distance, face in zip(contact_points, nearest, distances, faces, strict=True):
                part = name[str(labels[face])] if distance <= 0.006 else "unmatched_surface"
                details.append({**point, "mesh_face": int(face), "mesh_part": part,
                                "mesh_distance_m": float(distance), "nearest_local_xyz_m": xyz.tolist()})
        hit = {d["pad"] for d in details if d["mesh_part"] == requested} == {"pad_1", "pad_2"}
        return {"hit": hit, "contact_points": details, "requested_part": requested}

    meshes = load_collision_surfaces(asset)
    labels = {}
    if named_meshes:
        if requested not in named_meshes:
            raise ValueError("Requested contact part lacks a bound collision surface")
        for part, membership in named_meshes.items():
            # A part can bind whole collider meshes or explicit faces within a shared mesh.
            sources = membership if isinstance(membership, dict) else dict.fromkeys(membership)
            for path, indices in sources.items():
                if path not in meshes:
                    raise ValueError(f"Contact label references missing collider {path}")
                count = len(meshes[path].faces)
                for face in range(count) if indices is None else indices:
                    if not isinstance(face, int) or not 0 <= face < count or (path, face) in labels:
                        raise ValueError(f"Invalid or overlapping contact face {path}#{face}")
                    labels[path, face] = part
    hits, details = set(), []
    for contact in contact_points:
        point = contact["target_local_xyz_m"]
        nearest = []
        for path, mesh in meshes.items():
            surface, distance, face = closest_point_naive(mesh, [point])
            nearest.append((float(distance[0]), path, int(face[0]), surface[0], float(mesh.face_normals[face[0], 2])))
        distance, path, face, surface, normal_z = min(nearest, key=lambda row: row[0])
        part = labels.get((path, face))
        if not named_meshes:
            if abs(normal_z) < 0.5 and band[0] <= surface[2] <= band[1]:
                part = requested
            else:
                part = "other_surface"
        if part == requested:
            hits.add(contact["pad"])
        details.append({**contact, "mesh": path, "mesh_face": face, "mesh_part": part, "mesh_distance_m": distance,
                        "nearest_local_xyz_m": surface.tolist()})
    return {"hit": hits == {"pad_1", "pad_2"}, "contact_points": details, "requested_part": requested}
