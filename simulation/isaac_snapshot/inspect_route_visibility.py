import argparse
import concurrent.futures
import json
import math
from itertools import groupby
from pathlib import Path

import coal
import h5py
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from prepare_grouped import write_json
from school_planner import Planner, as_coal


def projection_ray(camera, target, anchor):
    endpoint = Rotation.from_quat(target[3:]).apply(anchor) + target[:3]
    vector = endpoint - camera
    length = np.linalg.norm(vector)
    z = vector / length
    x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    frame = pin.SE3(np.column_stack((x, np.cross(z, x), z)), camera + vector / 2)
    return coal.Capsule(0.002, length - 0.004), as_coal(frame)


def model_occlusion(planner, positions, ray, ray_pose):
    request = coal.CollisionRequest()
    right_shapes = [index for index, name in enumerate(planner.shape_names[:planner.robot_count])
                    if name.startswith("right_")]
    rows = []
    for arm in positions:
        q = planner.full_q(arm)
        pin.updateGeometryPlacements(planner.model, planner.data, planner.geometry, planner.geom_data, q)
        blocker = next((planner.shape_names[index] for index in right_shapes if coal.collide(
            ray, ray_pose, planner.shapes[index], as_coal(planner.root * planner.geom_data.oMg[index]),
            request, coal.CollisionResult())), None)
        rows.append(blocker)
    return rows


def longest_gap(flags):
    return max((len(list(run)) for blocked, run in groupby(flags, key=bool) if blocked), default=0)


def path_samples(candidate, origin):
    samples = [origin]
    for endpoint in (*candidate["pre_grasp_q"], *candidate["approach_q"][1:]):
        endpoint = np.asarray(endpoint)
        start = samples[-1]
        steps = max(1, math.ceil(np.max(np.abs(endpoint - start)) / 0.035))
        samples.extend(start + (endpoint - start) * fraction for fraction in np.linspace(0, 1, steps + 1)[1:])
    return samples


def compare(parameters):
    root, config, urdf, evidence, episode = parameters
    with h5py.File(episode, "r") as h5:
        case = json.loads(h5["meta/case_json"].asstr()[()])
        camera = json.loads(h5["meta/camera_json"].asstr()[()])
        anchor, target = h5["gt/part_anchor_local"][0], h5["gt/target_pose_xyz_xyzw"][0]
        count = h5["phase"].asstr()[...].tolist().index("close")
        measured_arm = h5["obs/qpos"][:count]
        visibility = h5["policy_prompt/valid"][:count]
    head = np.asarray(evidence["samples"][0]["head_pose_w_xyzw"])
    sensor = head[:3] + Rotation.from_quat(head[3:]).apply(np.asarray(camera["T_parent_camera_usd_stage_units"])[:3, 3])
    ray = projection_ray(sensor, target, anchor)
    plan = json.loads((root / "runs/train_bootstrap_01/plans" / (case["case_id"] + ".json")).read_text())
    planner = Planner({**config, "target_usd": case["target_usd"]}, urdf, evidence, case["target_pose_xyz_xyzw"],
                      obstacles=case["fixtures"], home=case["home_q"], expanded_search=True,
                      guide=plan["feasible"][0]["pre_grasp_q"], part=case["part"])
    baseline = model_occlusion(planner, measured_arm, *ray)
    alternatives = planner.plan(case["seed"], stop_after=6, max_candidates=95,
                                excluded=(plan["feasible"][0]["candidate"],))
    rows = []
    for candidate in alternatives["feasible"]:
        sampled = model_occlusion(planner, path_samples(candidate, planner.home), *ray)
        rows.append({"candidate": candidate["candidate"], "predicted_occlusion_s": longest_gap(sampled) * 0.05,
                     "xray_links": sorted({name for name in sampled if name}), "plan": candidate})
    return {"case": case["case_id"], "original": plan["feasible"][0]["candidate"],
            "measured_longest_occlusion_s": longest_gap(~visibility) * 0.05,
            "modeled_baseline_longest_occlusion_s": longest_gap(baseline) * 0.05,
            "modeled_baseline_links": sorted({name for name in baseline if name}),
            "alternatives": sorted(rows, key=lambda row: row["predicted_occlusion_s"])}


def replay_manifest(root, visibility, output):
    report = json.loads(visibility.read_text())
    chosen = next(row for row in report["cases"] if row["case"] == "C_422")
    alternative = chosen["alternatives"][0]
    if alternative["predicted_occlusion_s"] > 0.200001:
        raise ValueError("C_422 has no alternative pose path within the visibility window")
    source = root / "runs/train_bootstrap_01/plans"
    manifest = json.loads((source / "shard_carton.json").read_text())
    key = chosen["case"]
    case = next(row for row in manifest["cases"] if row["case_id"] == key)
    plan = json.loads((source / case["plan"]).read_text())
    output.mkdir()
    write_json(output / case["plan"], {**plan, "feasible": [alternative["plan"]]})
    write_json(output / "manifest.json", {**manifest, "cases": [case], "fixture_assets": {}})
    print(f"{key}: {chosen['original']} -> {alternative['candidate']}", flush=True)


def review_replay(root, replay, output):
    key = "C_422"
    attempts = json.loads((replay / "result.json").read_text())["cases"]
    path = replay / key / "episode.h5"
    baseline = root / "runs/train_bootstrap_01_carton" / key / "episode.h5"
    result = {"case": key, "strict_success": 0, "configured": 1, "passed": False, "cases": attempts}
    if path.is_file() and attempts[0]["status"] == "strict_success":
        intervals = []
        anchors = []
        for sample in (baseline, path):
            with h5py.File(sample, "r") as h5:
                anchors.append(h5["gt/part_anchor_local"][0])
                stop = h5["phase"].asstr()[...].tolist().index("close")
                intervals.append(longest_gap(~h5["policy_prompt/valid"][:stop]) * 0.05)
        difference = float(np.linalg.norm(anchors[1] - anchors[0]))
        result.update(strict_success=1, original_hidden_s=intervals[0], new_hidden_s=intervals[1],
                      anchor_displacement_m=difference, passed=intervals[1] <= 0.200001 and difference <= 0.001)
    write_json(output, result)
    print({key: value for key, value in result.items() if key != "cases"}, flush=True)
    return 0 if result["passed"] else 1


def main():
    cli = argparse.ArgumentParser()
    for name in ("root", "urdf", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    config = json.loads((args.root / "env_resolved.json").read_text())
    runs = args.root / "runs"
    evidence = json.loads((runs / "lab_school_03/result.json").read_text())
    bootstrap = json.loads((runs / "train_bootstrap_01/result.json").read_text())
    affected = {"A_401", "B_410", "B_412", "C_422"}
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(compare, (args.root, config, args.urdf, evidence, row["path"]))
                   for row in bootstrap["episodes"] if row["case"] in affected]
        rows = []
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            rows.append(item)
            alternatives = item["alternatives"]
            print(f"{item['case']} actual={item['measured_longest_occlusion_s']:.2f}s"
                  f" model={item['modeled_baseline_longest_occlusion_s']:.2f}s"
                  f" best_alternative={alternatives[0]['predicted_occlusion_s'] if alternatives else None}",
                  flush=True)
    args.output.mkdir()
    write_json(args.output / "result.json", {"scope": "offline visibility screening of the existing fixed anchors",
                                            "cases": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
