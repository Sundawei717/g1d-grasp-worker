import argparse
import json
from collections import Counter
from pathlib import Path
from shutil import copyfile

import h5py
import numpy as np

from dataset_contract import TRAIN_PHASES, read_rgb
from prepare_collection_preview import verify_collection_inputs
from prepare_grouped import write_json
from school_planner import sha256


def load_replay(binding, case, joint_names, camera):
    path = Path(binding["path"])
    if sha256(path) != binding["sha256"]:
        raise ValueError(f"Original HDF5 changed: {path}")
    with h5py.File(path, "r") as h5:
        expected = ({k: v for k, v in case.items() if k not in {"replay_preload_m", "gate_override"}}
                    if "replay_preload_m" in case else case)
        stored = json.loads(h5["meta/case_json"].asstr()[()])
        if binding.get("rerecord"):
            # A re-record gets a new dataset identity while preserving all physical initial conditions.
            identity = ("target_usd", "target_pose_xyz_xyzw", "home_q", "fixtures", "part", "seed")
            matches = (stored["case_id"] == binding["source_case"]
                       and all(stored[key] == case[key] for key in identity))
        else:
            matches = stored == expected
        if not matches:
            raise ValueError(f"HDF5 episode does not correspond to replay placement: {path}")
        if (tuple(h5["meta/joint_names"].asstr()[...]) != tuple(joint_names)
                or h5["meta"].attrs["camera_path"] != camera["prim_path"]
                or not json.loads(h5["meta/strict_score_json"].asstr()[()])["success"]):
            raise ValueError(f"Wrong arm, camera, or score in preview episode: {path}")
        phases = h5["action/native_phase" if "action/native_phase" in h5 else "phase"].asstr()[...].tolist()
        lift, hold = phases.index("lift"), phases.index("hold")
        if (not all(phase in TRAIN_PHASES for phase in phases[:lift]) or not 0 < lift < hold < len(phases)
                or set(phases[lift:hold]) != {"lift"} or set(phases[hold:]) != {"hold"}):
            raise ValueError("Training, trial lift, and hold phases are out of order")
        command = h5["action/native_substep_command"][...]
        names = len(joint_names)
        if (command.shape != (len(phases), 30, names + 2) or not np.isfinite(command).all()
                or not np.allclose(command[:, -1], h5["action/native_command"][...], atol=1e-6, rtol=0)):
            raise ValueError("Native 600 Hz command stream disagrees with its transition endpoints")
        image = {i: read_rgb(h5, i) for i in (0, len(phases))}
        times = h5["obs/timestamp"][...]
        if len(times) != len(phases) + 1 or not np.allclose(np.diff(times), 0.05, atol=1e-7, rtol=0):
            raise ValueError("Recorded observations do not advance at 20 Hz")
        return {"source": str(path), "phases": phases, "commands": command, "images": image,
                "arm": h5["obs/qpos"][...], "width": h5["obs/gripper_width"][...],
                "times_s": times, "lift_index": lift, "hold_index": hold,
                "v3_step_aligned": h5.attrs.get("schema") == "g1d_v3_step_aligned"}


def prepare(preview, destination):
    result = json.loads((preview / "result.json").read_text())
    marker = json.loads(preview.with_suffix(".complete").read_text())
    if not result["passed"] or marker["exit_code"] != 0 or result["strict_success"] < 20:
        raise ValueError("The 20-episode capacity preview must pass before alignment replay")
    records = {item["case"]: item for item in result["cases"]}
    quota = {"A_target": 4, "B_obstacle": 4, "C_start": 2}
    selected = Counter()
    count = 0
    destination.mkdir()
    for role in ("carton", "can"):
        manifest = json.loads((preview / "plans" / f"shard_{role}.json").read_text())
        verify_collection_inputs(manifest["collection_gate"], manifest["cases"])
        cases = []
        # Reserve the two start cases for both assets when possible.
        for row in manifest["cases"]:
            key = row["case_id"]
            group = row["group"]
            remaining = quota[group] - selected[group]
            cap = {"A_target": 2, "B_obstacle": 4, "C_start": 1}[group]
            if remaining <= 0 or sum(item["group"] == group for item in cases) >= cap:
                continue
            result_row = records[key]
            if result_row["status"] != "strict_success" or not result_row.get("dataset_receipt"):
                continue
            h5 = preview.parent / result_row["shard"] / key / "episode.h5"
            if h5.stat().st_size != result_row["dataset_receipt"]["bytes"]:
                raise ValueError(f"Preview HDF5 bytes changed: {h5}")
            cases.append({**row, "replay_h5": {"path": str(h5.resolve()), "sha256": sha256(h5)}})
            copyfile(preview / "plans" / row["plan"], destination / row["plan"])
            selected[group] += 1
        count += len(cases)
        write_json(destination / f"shard_{role}.json", {**manifest, "phase": "collection_replay", "cases": cases})
    if selected != quota or count != 10:
        raise ValueError(f"Expected 10 fixed replays over both assets and three groups: {dict(selected)}")
    print(f"Prepared {count} native-command replays: {dict(selected)}", flush=True)


def finish(output):
    attempts = []
    for role in ("carton", "can"):
        run = output.parent / f"{output.name}_{role}"
        for name in ("result.json", "batch_progress.json"):
            path = run / name
            if path.is_file():
                attempts.extend(json.loads(path.read_text())["cases"])
                break
    strict = sum(row["status"] == "strict_success" and bool(row.get("replay_alignment")) for row in attempts)
    passed = strict == len(attempts) == 10
    write_json(output / "result.json", {"phase": "recorded_command_alignment", "configured": 10,
                                       "strict_success": strict, "passed": passed, "cases": attempts})
    print(f"Alignment replay: {strict}/10", flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser()
    stages = parser.add_subparsers(dest="command", required=True)
    prep = stages.add_parser("prepare")
    prep.add_argument("preview", type=Path)
    prep.add_argument("output", type=Path)
    stages.add_parser("finish").add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.preview, args.output)
        return 0
    return finish(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
