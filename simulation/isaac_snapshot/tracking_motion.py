import argparse
import json
from pathlib import Path
from shutil import copyfile
from tempfile import NamedTemporaryFile

import cv2
import h5py
import numpy as np

from prepare_grouped import verify_gates, write_json
from school_planner import sha256


def prepare(root, output):
    source = root / "runs/stage3_splits_01"
    summary = json.loads((source / "result.json").read_text())
    result = json.loads(source.with_suffix(".complete").read_text())
    if result["exit_code"] != 0 or summary["validation_offline_feasible"] != 30:
        raise ValueError("The independent layout preparation has not passed")
    layout_groups = summary["cases"]
    selected = [row for row in layout_groups if row["group"] == "C_start"]
    selected.append(next(row for row in layout_groups if row["group"] == "A_target" and row["asset_role"] == "carton"))
    if len(selected) != 9:
        raise ValueError("Expected nine frozen development placements")
    output.mkdir()
    gates = verify_gates(root)
    for index, row in enumerate(selected):
        key = row["case_id"]
        direction = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[index % 4]
        selected[index] = {**row, "force_xy": direction, "source_plan_sha256": sha256(source / row["plan"])}
        copyfile(source / row["plan"], output / row["plan"])
        if key[0] == "A" and not row["fixtures"]:
            raise ValueError("The selected multi-object recording must have a distractor")
    for role in ("carton", "can"):
        cases = [row for row in selected if row["asset_role"] == role]
        fixtures = {slot["name"]: slot["usd"] for row in cases for slot in row["fixtures"]}
        write_json(output / f"shard_{role}.json", {"phase": "tracking_motion", "cases": cases,
                   "asset": cases[0]["target_usd"], "gates": gates, "fixture_assets": fixtures})
    write_json(output / "frozen_cases.json", {"cases": selected, "scope": "physical tabletop motion, fixed arm"})


def record_motion(output, camera, case, direction, robot, target, simulation, capture, initial_prompt, next_prompt):
    import torch

    dt = simulation.get_physics_dt()
    if not np.isclose(dt, 1 / 600, atol=1e-9):
        raise RuntimeError("The motion recording requires the pinned 600 Hz physics clock")
    names = list(robot.joint_names)
    arm = [names.index(name) for name in (
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")]
    hold = robot.data.default_joint_pos.clone()
    hold[0, arm] = hold.new_tensor(case["home_q"])
    robot.set_joint_position_target_index(target=hold)
    composer = target.permanent_wrench_composer
    composer.reset()
    mass = {"carton": 0.0398, "can": 0.0775}[case["asset_role"]]
    direction = np.asarray(direction, dtype=float)
    force = torch.zeros((1, 1, 3), device=hold.device, dtype=hold.dtype)

    def physics():
        robot.write_data_to_sim()
        target.write_data_to_sim()
        simulation.step(render=False)
        robot.update(dt)
        target.update(dt)

    for _ in range(360):
        physics()
    initial = np.array(target.data.root_link_pose_w[0].tolist())
    first_rgb, frame = capture("expert_initial.png")
    frames = [(first_rgb, frame, initial_prompt(initial), initial)]
    previous = initial[:2].copy()
    pulse_steps = 0
    peak_force = 0.0
    active = True
    arm_drift = 0.0
    try:
        for control in range(1, 43):
            pushing = 8 <= control < 24 and active
            for _ in range(30):
                position = np.asarray(target.data.root_link_pose_w[0, :2].tolist())
                if pushing and float((position - initial[:2]) @ direction) >= 0.007:
                    active = pushing = False
                    composer.reset()
                if pushing:
                    speed = float(((position - previous) @ direction) / dt)
                    strength = float(np.clip(mass * (11 - 70 * speed), 0.0, 1.1))
                    force[0, 0, :2] = force.new_tensor(direction * strength)
                    composer.set_forces_and_torques_index(forces=force, is_global=True)
                    pulse_steps += 1
                    peak_force = max(peak_force, strength)
                else:
                    composer.reset()
                physics()
                previous = position
            rgb, observation, prompt = next_prompt()
            current = np.asarray(target.data.root_link_pose_w[0].tolist())
            frames.append((rgb, observation, prompt, current))
            measured = np.asarray(robot.data.joint_pos[0, arm].tolist())
            arm_drift = max(arm_drift, float(np.max(np.abs(measured - case["home_q"]))))
    finally:
        composer.reset()
    positions = np.asarray([item[3] for item in frames])
    movement = np.linalg.norm(positions[:, :2] - initial[:2], axis=1)
    times = np.asarray([item[1]["sensor_sim_time_s"] for item in frames])
    if not np.allclose(np.diff(times), 0.05, atol=1e-6, rtol=0):
        raise RuntimeError("Moving-target RGB clock lost alignment")
    passed = bool(0.005 <= movement[-1] <= 0.015 and movement.max() <= 0.015 and arm_drift <= 0.005)
    path = output / "episode.h5"
    with NamedTemporaryFile(dir=output, suffix=".tmp", delete=False) as temporary:
        stage = Path(temporary.name)
    try:
        with h5py.File(stage, "w") as h5:
            jpeg = h5.create_dataset("obs/rgb_left", (len(frames),), dtype=h5py.vlen_dtype(np.uint8))
            jpeg.attrs.update(encoding="jpeg", quality=95)
            for index, entry in enumerate(frames):
                image = cv2.cvtColor(entry[0], cv2.COLOR_RGB2BGR)
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError("Motion frame JPEG encoding failed")
                jpeg[index] = encoded
            h5["obs/timestamp"] = times
            h5["meta/camera_json"] = json.dumps(camera)
            h5["meta/case_json"] = json.dumps({**case, "split": "tracking_development_motion"})
            h5["meta"].attrs["motion_start_frame"] = 8
            prompts = [item[2] or {} for item in frames]
            h5["policy_prompt/bbox"] = [prompt.get("bbox_0", [0.0] * 4) for prompt in prompts]
            h5["policy_prompt/point"] = [prompt.get("affordance_0") or [0.0] * 2 for prompt in prompts]
            h5["gt/part_visibility"] = [bool(prompt.get("prompt_valid")) for prompt in prompts]
            h5["gt/part_anchor_projected"] = [prompt.get("projected_anchor") or [0.0] * 2 for prompt in prompts]
            h5["gt/target_pose_xyz_xyzw"] = positions
            width, height = camera["resolution"]
            empty_mask = np.zeros((width * height + 7) // 8, dtype=np.uint8)
            for name in ("target_mask_packed", "part_mask_packed"):
                h5.create_dataset(f"gt/{name}", data=[prompt.get(name, empty_mask) for prompt in prompts],
                                  compression="gzip")
        stage.replace(path)
    finally:
        stage.unlink(missing_ok=True)
    result = {"passed": passed, "recording_frames": len(frames), "target_final_m": float(movement[-1]),
              "target_peak_m": float(movement.max()), "max_arm_error_rad": arm_drift, "force_steps": pulse_steps,
              "peak_force_n": peak_force, "bytes": path.stat().st_size}
    write_json(output / "motion_diagnostic.json", result)
    return result


def finish(output):
    planned = json.loads((output / "plans/frozen_cases.json").read_text())["cases"]
    received = {}
    for role in ("carton", "can"):
        shard = output.parent / f"{output.name}_{role}"
        result = shard / "result.json"
        if result.is_file():
            for row in json.loads(result.read_text())["cases"]:
                received[row["case"]] = {**row, "shard": shard.name}
    rows = [received.get(row["case_id"], {"case": row["case_id"], "status": "infrastructure_fail"}) for row in planned]
    count = sum(row["status"] == "recorded" for row in rows)
    write_json(output / "result.json", {"phase": "tracking_motion", "configured": 9, "recorded": count,
                                        "passed": count == 9, "cases": rows})
    print(f"Recorded {count}/9 moving-target videos with observed shifts in the planned range", flush=True)
    return 0 if count == 9 else 1


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="phase", required=True)
    prep = modes.add_parser("prepare")
    prep.add_argument("root", type=Path)
    prep.add_argument("output", type=Path)
    modes.add_parser("finish").add_argument("output", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.root, args.output)
        return 0
    return finish(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
