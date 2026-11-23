import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from episode_store import verify_three_view
from prepare_collection_preview import plan_preview
from school_planner import authored_meshes
from three_view_detector import sha256, write_json


TRAIN_ASSETS = {
    "mug_train": ("mug/00007", "mug", "sand3d_packaging/00152"),
    "flat_carton": ("sand3d_packaging/00208", "box", "sand3d_packaging/00156"),
    "tall_can": ("soda_can/00000", "can", "sand3d_packaging/00152"),
}
HOLDOUT = {"mug/00004", "sand3d_packaging/00225"}


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root, output):
    plans = output / "plans"
    plans.mkdir()
    write_json(plans / "reused.json", {"cases": [], "episodes": []})
    cfg_path = root / "env_resolved.json"
    cfg = read(cfg_path)
    evidence = read(root / "runs/lab_school_03/result.json")
    urdf = Path("/data/sundawei/vla_rsi_g1d/assets/Assets/Robots/g1d/g1d_dex1.urdf")
    template = read(root / "runs/train_bootstrap_01/plans/shard_carton.json")
    table = next(points for name, points in authored_meshes(cfg["school_usd"]) if name == "/Scene/Table/Top")
    surface = float(table[:, 2].max() + cfg["school_position_m"][2])
    rng = np.random.default_rng(2387551)

    def asset(family):
        path = root / "assets" / family / "object.usdz"
        return path, np.concatenate([points for _, points in authored_meshes(path)])

    def place(points, x, y, yaw):
        rotation = Rotation.from_euler("z", yaw, degrees=True)
        z = surface - float(rotation.apply(points)[:, 2].min()) + 0.0001
        return [float(x), float(y), z, *rotation.as_quat().tolist()]

    jobs, results, descriptors = [], {}, {}
    home = np.asarray(evidence["samples"][0]["right_rad"])
    for role, (family, query, neighbor_family) in TRAIN_ASSETS.items():
        usd, points = asset(family)
        neighbor, neighbor_points = asset(neighbor_family)
        config = {**cfg, "target_usd": str(usd)}
        write_json(plans / f"config_{role}.json", config)
        descriptors[role] = (str(usd), query)
        for sample in range(8):
            x, y, yaw = 0.183 + rng.uniform(-0.022, 0.022), -0.419 + rng.uniform(-0.022, 0.021), rng.uniform(-30, 30)
            target_pose = place(points, x, y, yaw)
            fixtures = []
            if sample % 2:
                fixture_pose = place(neighbor_points, x + 0.13, y + rng.uniform(-0.012, 0.012), yaw / 2)
                fixtures = [{"name": "neighbor", "usd": str(neighbor), "sha256": sha256(neighbor), "pose": fixture_pose}]
            start = home.copy()
            if not fixtures:
                start += rng.uniform(-1, 1, 7) * [0.08, 0.06, 0.08, 0.06, 0.06, 0.07, 0.06]
                start[1] = -abs(start[1])
                start[3] = max(start[3], -1.01)
            num = 5100 + len(jobs)
            group = "A_target" if fixtures else "C_start"
            case = {"case_id": f"{group[0]}_{num}", "group": group, "asset_role": role, "target_usd": str(usd),
                    "target_pose_xyz_xyzw": target_pose, "home_q": start.tolist(), "fixtures": fixtures,
                    "part": "upper_side_panels", "config_sha256": sha256(cfg_path), "split": "train",
                    "scene_family": f"three_view_diverse_{role}_{sample}", "asset_family": family,
                    "seed": 2387551 + num, "command_delay_substeps": 0}
            jobs.append(case)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(plan_preview, (cfg, urdf, evidence, row, None)): row for row in jobs}
        for future in as_completed(futures):
            row, plan = futures[future], future.result()
            identifier = row["case_id"]
            write_json(plans / f"{identifier}.json", plan)
            results[identifier] = {**row, "plan": f"{identifier}.json", "offline_feasible": bool(plan["feasible"]),
                                   "offline_failure": plan.get("offline_failure")}
            print(identifier, row["asset_role"], "planned" if plan["feasible"] else plan.get("offline_failure"), flush=True)
    physical_parts = root / "runs/three_view_part_plans_02"
    parts = [row for row in read(physical_parts / "result.json")["cases"]
             if row["asset_role"] == "carton" and row["scene_family"] != "three_view_extra_carton_0"]
    pair_count = Counter(row["scene_family"] for row in parts if row["offline_feasible"])
    for row in parts:
        if pair_count[row["scene_family"]] != 2:
            continue
        plan = read(physical_parts / row["plan"])
        row = {**row, "asset_role": "part_carton", "split": "train", "command_delay_substeps": 0}
        plan["case"].update(asset_role="part_carton", split="train", command_delay_substeps=0)
        write_json(plans / row["plan"], plan)
        results[row["case_id"]] = row
    descriptors["part_carton"] = (cfg["target_usd"], "box")
    write_json(plans / "config_part_carton.json", cfg)
    for role, (usd, query) in descriptors.items():
        rows = [row for _, row in sorted(results.items()) if row["asset_role"] == role]
        if not rows:
            continue
        fixtures = {fixture["name"]: fixture["usd"] for row in rows for fixture in row["fixtures"]}
        write_json(plans / f"shard_{role}.json", {**template, "asset": usd, "fixture_assets": fixtures,
                                                "three_view": True, "target_query": query, "cases": rows})
    write_json(output / "plan_result.json", {"cases": list(results.values()), "holdout_asset_families": sorted(HOLDOUT)})


def merge(root, output):
    baseline = read(root / "runs/three_view_scale_01/result.json")
    addition = read(root / "runs/three_view_diversity_01/result.json")
    if baseline["configured"] != 250 or len(baseline["cases"]) != 250:
        raise ValueError("Expected the finished 250-layout base collection")
    extra = addition["episodes"]
    extras_by_asset = Counter(row["asset_family"] for row in extra)
    part_labels = {row["part"] for row in extra if row["asset_family"] == "sand3d_packaging/00152"}
    episodes = baseline["episodes"] + extra
    known_cases = [row["case"] for row in episodes]
    heldout_scenes = {row["scene_family"] for path in ("three_view_pipeline_01", "three_view_generalization_01", "three_view_parts_01")
                      for row in read(root / "runs" / path / "frozen/frozen.json")["cases"]}
    if (len(set(known_cases)) != len(known_cases) or {row["asset_family"] for row in episodes} & HOLDOUT
            or {row["scene_family"] for row in episodes} & heldout_scenes):
        raise ValueError("An added target or scene overlaps the fixed test split")
    for row in extra:
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError("An added expert recording changed")
        with h5py.File(row["path"], "r") as h5:
            verify_three_view(h5)
    passed = bool(extra)
    write_json(output / "result.json", {"phase": "three_view_diverse_training_data", "passed": passed,
                                       "accepted": len(episodes), "base_episodes": len(baseline["episodes"]),
                                       "added_episodes": len(extra), "added_carton_parts": sorted(part_labels),
                                       "episodes": episodes, "added_by_asset": extras_by_asset,
                                       "asset_counts": Counter(row["asset_family"] for row in episodes),
                                       "asset_balance": True, "holdout_asset_families": sorted(HOLDOUT)})
    print("Added physical episodes", dict(extras_by_asset), "requested box parts", sorted(part_labels))
    return 0 if passed else 1


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("command", choices=("prepare", "merge"))
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    root = Path(__file__).resolve().parent
    if args.command == "prepare":
        prepare(root, args.output)
    else:
        raise SystemExit(merge(root, args.output))
