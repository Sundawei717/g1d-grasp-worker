import argparse
import hashlib
import json
from pathlib import Path


def physical_receipts(root, native_review=None):
    groups = {}
    for report in sorted(root.glob("*admission_*/result.json")):
        data = json.loads(report.read_text())
        rows = data.get("cases", [])
        if len(rows) != 10:
            continue
        asset = data.get("asset_usd")
        trials = []
        for row in rows:
            artifact = report.parent / row["case"] / "expert_diagnostic.json"
            if artifact.is_file():
                trial = json.loads(artifact.read_text())
                if asset is None:
                    asset = trial.get("asset_usd")
                if trial.get("asset_usd") != asset:
                    raise ValueError(f"Asset admission mixed assets: {report}")
                trials.append({"case_id": row["case"], "receipt": str(artifact.resolve()), "status": row["status"],
                               "strict_score": trial.get("strict_score"), "failure": trial.get("failure"),
                               "pose_xyz_xyzw": trial.get("target_initial_xyz_xyzw"), "initial_prompt": trial.get("initial_prompt")})
        if not asset:
            continue
        success = [trial for trial in trials if trial["status"] == "strict_success"
                   and (trial["strict_score"] or {}).get("success") is True and trial["initial_prompt"]]
        distinct_poses = {trial["case_id"].split("_rep")[0] for trial in success}
        groups.setdefault(str(Path(asset).resolve()), []).append({
            "report": str(report.resolve()), "physical_trials": sum(trial["status"] in {"physical_fail", "strict_success"}
                                                                  for trial in trials), "strict_successes": len(success),
            "covered_poses": len(distinct_poses), "cases": trials,
            "candidate_part_gate_passed": data.get("passed") is True and len(success) == 10 and len(distinct_poses) == 5,
        })
    if native_review:
        from review_admission import review

        supplied = json.loads(native_review.read_text())
        source = Path(supplied["source_directory"]).resolve()
        if root.resolve() not in source.parents:
            raise ValueError("Asset baseline review leaves the run directory")
        checked = review(source)
        if checked != supplied:
            raise ValueError("A provided support review does not reproduce")
        groups.setdefault(checked["asset_usd"], []).append({
            "report": str(native_review.resolve()), "source_report": checked["original_batch"],
            "physical_trials": checked["attempted"], "strict_successes": checked["strict_success"],
            "covered_poses": len({row["case"].split("_rep")[0] for row in checked["cases"] if row["status"] == "strict_success"}),
            "baseline_contract": checked["baseline_contract"], "candidate_part_gate_passed": checked["passed"],
        })
    return groups


def properties(path):
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    body_records, materials, colliders = [], [], []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass = UsdPhysics.MassAPI(prim)
            inertia = mass.GetDiagonalInertiaAttr().Get()
            body_records.append({"prim": str(prim.GetPath()), "mass_kg_authored": mass.GetMassAttr().Get(),
                                 "diagonal_inertia_authored": list(inertia) if inertia is not None else None})
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            material = UsdPhysics.MaterialAPI(prim)
            materials.append({"prim": str(prim.GetPath()), "static_friction": material.GetStaticFrictionAttr().Get(),
                              "dynamic_friction": material.GetDynamicFrictionAttr().Get()})
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            mesh = UsdPhysics.MeshCollisionAPI(prim)
            colliders.append({"prim": str(prim.GetPath()),
                              "approximation": mesh.GetApproximationAttr().Get() if mesh else str(prim.GetTypeName())})
    return {"body_properties": body_records, "material_coefficients": materials, "colliders": colliders,
            "values_are_simulation_settings": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    parser.add_argument("runs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    admissions = physical_receipts(args.runs, args.review)
    records = []
    for candidate in inventory["assets"]:
        path = Path(candidate["path"]).resolve()
        previous = admissions.get(str(path), [])
        asset = "/".join(path.parts[-3:-1])
        source = candidate.get("source_metadata") or {}
        part = None
        if asset in {"sand3d_packaging/00152", "sand3d_packaging/00156"}:
            part = {"id": "upper_side_panels", "anchor_definition": "valid calibrated RTX depth on actual target pixels",
                    "acceptance_region": "side contact on both pads in upper local band: bottom+0.04 m to top+0.003 m",
                    "candidate_settings": "depth, yaw, skew, tilt and offline plan per physical receipt"}
        records.append({"asset_id": asset, "family_id": None, "source_catalog_id": source.get("asset_id") or candidate.get("source_catalog_id"),
                        "category": path.parent.parent.name, "source_root": inventory["source_root"],
                        "source_relative_path": "/".join(path.parts[-3:]), "source_metadata": source or None,
                        "license": source.get("license"), "usd_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "usd": str(path), "nominal_dimensions_m": (
                            [dimension * candidate["metres_per_unit"] for dimension in candidate["dimensions_stage_units"]]
                            if candidate["dimensions_stage_units"] else None),
                        "scale_range_validated": None, "nominal_scale_tested": bool(previous),
                        "static_status": candidate["static_status"], "static_problems": candidate["problems"],
                        "physics": properties(path), "evaluated_part": part, "admission_batches": previous,
                        "part_passed": any(group["candidate_part_gate_passed"] for group in previous),
                        "split": None})
    artifact = {"scope": "candidate and physical-part evidence; unverified family IDs and splits are intentionally null",
                "asset_count": len(records), "physically_admitted_parts": sum(item["part_passed"] for item in records),
                "assets": records}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(artifact, stream, indent=2)
    print(f"{len(records)} assets, {artifact['physically_admitted_parts']} passed parts")


if __name__ == "__main__":
    main()
