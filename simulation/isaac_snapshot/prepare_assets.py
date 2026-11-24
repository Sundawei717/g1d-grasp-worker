import argparse
import importlib.metadata
import json
import math
from pathlib import Path


def inspect_asset(path, runtime_material):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

    metadata = json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    stage = Usd.Stage.Open(str(path))
    if not stage or not stage.GetDefaultPrim():
        raise RuntimeError(f"No USD default prim: {path}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bounds = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    size = None if bounds.IsEmpty() else list(bounds.GetSize())
    valid_size = size is not None and all(math.isfinite(axis) and axis > 0 for axis in size)
    bodies = sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in stage.Traverse())
    colliders = sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in stage.Traverse())
    mass = metadata.get("physics", {}).get("mass")
    valid_mass = isinstance(mass, (float, int)) and math.isfinite(mass) and mass > 0
    problems = []
    runtime_dependencies = [name for name in unresolved if str(name) == "OmniPBR.mdl" and runtime_material.is_file()]
    if set(unresolved) - set(runtime_dependencies):
        problems.append("unresolved_dependencies")
    if not valid_size:
        problems.append("missing_or_invalid_geometry")
    if bodies != 1 or colliders < 1:
        problems.append("missing_single_rigid_body_or_collision")
    if not valid_mass:
        problems.append("missing_or_invalid_documented_mass")
    return {"path": str(path), "dimensions_stage_units": size, "metres_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "documented_mass_kg": mass, "rigid_bodies": bodies, "colliders": colliders,
            "source_metadata": metadata.get("source"),
            "source_catalog_id": metadata.get("uuid"),
            "family_review": "unassigned; exclude from all splits until physical admission and family review",
            "resolved_layers": len(layers), "resolved_other_assets": len(assets), "unresolved": list(map(str, unresolved)),
            "installed_runtime_material": str(runtime_material) if runtime_dependencies else None,
            "problems": problems, "static_status": "reject" if problems else "candidate_only",
            "opacity_visibility_and_physical_admission": "not_tested"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.assets.resolve().glob("*/*/object.usdz"))
    if not paths:
        raise SystemExit("No candidate USDZ assets")
    runtime_material = Path(importlib.metadata.distribution("isaacsim").locate_file("isaacsim/kit/mdl/core/Base/OmniPBR.mdl"))
    result = {"scope": "static dependency and geometry check only",
              "source_root": "21197:/workspace/Assets/Object/RoboDojo/Rigid",
              "assets": [inspect_asset(path, runtime_material) for path in paths]}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
