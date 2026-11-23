import argparse
import json
import math
from pathlib import Path

from strict_grasp import PhysicsSample, evaluate


def audit(directory, baseline_contract=None):
    report = json.loads((directory / "expert_diagnostic.json").read_text())
    raw = [json.loads(line) for line in (directory / "physics_trace.jsonl").read_text().splitlines()]
    if report["sample_count"] != len(raw) or not raw:
        raise ValueError("Native physics trace is missing or truncated")
    trace = [PhysicsSample(**{**row, "target_part_pads": frozenset(row["target_part_pads"])}) for row in raw]
    commands = report["commands"]
    for previous, current in zip(commands, commands[1:]):
        if not math.isclose(current["time_s"] - previous["time_s"], 0.05, abs_tol=1e-7):
            raise ValueError("Expert commands are not on the 20 Hz clock")
    frames = json.loads((directory / "expert_camera_frames.json").read_text())
    stamps = sorted(frame["sensor_sim_time_s"] for frame in frames.values())
    if any(after <= before for before, after in zip(stamps, stamps[1:])):
        raise ValueError("Expert RGB frame clock is stale")
    errors = [abs(frame["sensor_sim_time_s"] - frame["requested_sim_time_s"]) for frame in frames.values()]
    if not errors or max(errors) > 1e-5:
        raise ValueError("Native timestamps lost camera/physics alignment")
    if "strict_score" not in report:
        return {"strict_success": False, "failure": report.get("failure"), "trace_samples": len(trace)}
    recorded_contract = report.get("baseline_contract", "instantaneous_v1")
    arguments = {"lift_start": report["lift_start_sample"], "hold_start": report["hold_start_sample"], "dt": 1 / 600}
    score = evaluate(trace, **arguments, baseline_contract=recorded_contract)
    stored = report["strict_score"]
    matching = score.keys() == stored.keys() and all(
        math.isclose(float(value), float(stored[key]), rel_tol=1e-8, abs_tol=1e-10)
        if isinstance(value, float) else value == stored[key] for key, value in score.items())
    if not matching:
        raise ValueError("Stored score disagrees with independent trace replay")
    if baseline_contract:
        score = evaluate(trace, **arguments, baseline_contract=baseline_contract)
    return {"strict_success": score["success"], "score": score, "baseline_contract": baseline_contract or recorded_contract,
            "trace_samples": len(trace), "rgb_frames": len(frames),
            "max_rgb_clock_error_s": max(errors), "control_steps": len(commands)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--baseline-contract", choices=("instantaneous_v1", "stable_support_v2"))
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.baseline_contract), indent=2))


if __name__ == "__main__":
    main()
