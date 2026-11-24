import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from episode_store import jaw_width_m
from dataset_contract import read_rgb
from policy_rollout import PolicyStop
from three_view_detector import write_json
from vertical_lift import checked_lift


def read(path):
    return json.loads(path.read_text())


def stats(values):
    return {key: float(value) for key, value in zip(("avg", "p50", "p95"), [np.mean(values), *np.percentile(values, [50, 95])])}


def main(root):
    import cv2

    runs = root / "runs"
    work = runs / "three_view_step0_01"
    report = read(work / "result.json")
    cfg = read(root / "env_resolved.json")
    evidence = read(runs / "lab_school_03/result.json")
    urdf = Path("/data/sundawei/vla_rsi_g1d/assets/Assets/Robots/g1d/g1d_dex1.urdf")
    latency = {}
    lift_audit = []
    for variant in ("A", "B"):
        steps = []
        for row in (row for row in report["cases"] if row["variant"] == variant):
            diagnostic = read(Path(row["diagnostic_path"]))
            rollout = diagnostic["policy_rollout"]
            steps.extend(rollout["decisions"][1:])
            if row["failure_category"] != "lift_path_rejected":
                continue
            trial = rollout["trials"][0]
            trigger = trial["trigger"]["step"]
            command = diagnostic["commands"][trigger - 1]
            fixtures = read(work / "plans" / f"{row['case']}.json")["obstacles"]
            comparisons = {}
            for clearance in (0.002, 0.001, 0.0005, 0.0):
                try:
                    checked_lift(cfg, urdf, evidence, fixtures, np.asarray(command["arm_observed_rad"]),
                                 np.asarray(command["wrist_xyz_xyzw"]), command["finger_observed_m"],
                                 margin=clearance)
                    comparisons[str(clearance)] = "clear"
                except PolicyStop as error:
                    comparisons[str(clearance)] = str(error)
            final_physics = json.loads(Path(row["diagnostic_path"]).with_name("physics_trace.jsonl").read_text().splitlines()[-1])
            lift_audit.append({"case": row["case"], "variant": variant, "repeat": row["repeat"],
                               "original": trial["detail"], "margins_m": comparisons,
                               "trigger": trial["trigger"], "collision_in_actual_final_step": final_physics["forbidden_contact"],
                               "actual_forbidden_contact_pairs": diagnostic["forbidden_contact_pairs"]})
        by_stage = {"capture_ms": [], "forward_ms": [], "left_detector_ms": [], "right_detector_ms": [],
                    "unprofiled_pre_post_copy_dispatch_ms": [], "end_to_end_ms": []}
        for s in steps:
            by_stage["capture_ms"].append(s["capture_ms"])
            by_stage["forward_ms"].append(s["policy_forward_ms"])
            for k in ("left_detector_ms", "right_detector_ms"):
                by_stage[k].append(s.get(k) or 0)
            remainder = s["to_command_ms"] - sum(s.get(k) or 0 for k in
                                               ("capture_ms", "policy_forward_ms", "left_detector_ms", "right_detector_ms"))
            by_stage["unprofiled_pre_post_copy_dispatch_ms"].append(remainder)
            by_stage["end_to_end_ms"].append(s["to_command_ms"])
        latency[variant] = {"samples": len(steps), **{key: stats(values) for key, values in by_stage.items()}}
    row = next(row for row in report["expert"]["cases"] if row["case"] == "A_400")
    d = read(Path(row["diagnostic_path"]))
    trigger = d["policy_rollout"]["trials"][0]
    idx = trigger["trigger"]["step"]
    with h5py.File(d["replay_alignment"]["source"], "r") as h5:
        lift = d["replay_alignment"]["recorded_lift_step"]
        native_targets = h5["action/native_command"][lift:lift + 16, 7:9]
        old_widths = [jaw_width_m(target) for target in native_targets]
    widths = [jaw_width_m(command["finger_target_m"]) for command in d["commands"][idx:]]
    initial_images = {}
    for a in (x for x in report["cases"] if x["repeat"] == 1):
        b = next(x for x in report["cases"] if x["repeat"] == 2
                 and (x["variant"], x["case"]) == (a["variant"], a["case"]))
        first, second = (Path(x["diagnostic_path"]).parent for x in (a, b))
        initial_images[f'{a["variant"]}_{a["case"]}'] = {
            p.name: float(np.abs(cv2.imread(str(p)).astype(float) - cv2.imread(str(second / p.name))).mean())
            for p in first.glob("expert_initial*.png")}
    return {"A_400": {"trigger": trigger["trigger"], "locked_target_width_m": jaw_width_m(trigger["lock_finger_target_m"]),
                      "actual_width_m": trigger["trigger"]["width_actual_m"],
                      "lift_command_min_max_m": [min(widths), max(widths)],
                      "old_expert_lift_width_m": old_widths, "score": trigger["stability"]},
            "latency_ms": latency, "lift_audit": lift_audit, "repeat_initial_rgb_mae_0_to_255": initial_images,
            "stepping": "synchronous capture/predict/dispatch, 30 physics substeps per 20 Hz action; no inference-time simulated advance"}


def profile_inputs(root, device):
    import torch
    from three_view_detector import WristDetector
    from three_view_rollout import ThreeViewPolicy

    split = read(root / "runs/three_view_pipeline_03/split.json")
    cases = {"A_402": range(53, 58), "C_5102": range(66, 71), "C_5104": range(71, 76)}
    examples = []
    for row in split["episodes"]:
        if row["case"] not in cases:
            continue
        with h5py.File(row["path"], "r") as f:
            task = json.loads(f["meta/initial_prompt_json"].asstr()[()])
            for i in cases[row["case"]]:
                frame = {"wrist_views": {name: (read_rgb(f, i, f"obs/{name}/rgb"), {})
                                         for name in ("left_wrist", "right_wrist")}}
                state = np.r_[f["obs/qpos"][i], f["obs/gripper_width"][i]]
                examples.append((read_rgb(f, i), frame, task, state))
    rec = {}
    for variant in "ab":
        source = root / f"runs/three_view_pipeline_03_fit_{variant}/checkpoint.pt"
        detector = WristDetector(read(root/"three_view_detector.json"), device) if variant == "b" else None
        policy = ThreeViewPolicy(source, device, detector)
        for _ in range(3):
            policy.predict(*examples[0], diagnostics=True)
        values = [policy.predict(*sample, diagnostics=True)[1] for sample in examples]
        rec[variant.upper()] = {field: stats([r[field] for r in values]) for field in
                                ("cpu_preprocess_ms", "input_h2d_ms", "output_d2h_post_ms", "policy_forward_ms")}
        del policy, detector
        torch.cuda.empty_cache()
    return {"samples_per_variant": len(examples), "device": device, "profile": rec}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile-device")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    write_json(args.output, profile_inputs(root, args.profile_device) if args.profile_device else main(root))
