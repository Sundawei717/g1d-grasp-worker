import argparse
import json
import shutil
import time
from collections import Counter
from itertools import groupby
from pathlib import Path

import h5py
import numpy as np

from collect_training_overnight import QUOTAS, select
from school_planner import sha256


def scan(episode):
    path = Path(episode["path"])
    if sha256(path) != episode["sha256"]:
        raise ValueError(f"Episode content changed: {path}")
    with h5py.File(path, "r") as h5:
        camera = json.loads(h5["meta/camera_json"].asstr()[()])
        case = json.loads(h5["meta/case_json"].asstr()[()])
        success = json.loads(h5["meta/strict_score_json"].asstr()[()])["success"]
        if case["case_id"] != episode["case"] or case["split"] != "train" or not success:
            raise ValueError(f"Training provenance or strict score changed: {path}")
        if camera["resolution"] != [640, 480] or h5["meta"].attrs["arm_side"] != "right":
            raise ValueError(f"Wrong left-eye resolution or arm: {path}")
        steps = len(h5["phase"])
        visible = h5["policy_prompt/valid"][...]
        native = h5["action/native_substep_command"]
        if (len(visible) != steps + 1 or not visible[0] or native.shape != (steps, 30, 9)
                or not np.allclose(np.diff(h5["obs/timestamp"][...]), .05, rtol=0, atol=1e-6)):
            raise ValueError(f"Frame, step, or native actuator alignment failed: {path}")
        anchor = h5["gt/part_anchor_local"][...]
        if not np.allclose(anchor, anchor[0], rtol=0, atol=1e-8):
            raise ValueError(f"Object-local anchor is not fixed: {path}")
        phases = h5["phase"].asstr()[...].tolist()
        close = phases.index("close")
        largest = max((sum(1 for _ in span) for valid, span in groupby(visible[:close]) if not valid), default=0)
        eligible = int(np.count_nonzero(visible[:-1] & h5["train_action_valid"][...]))
        stored = sum(item.stat().st_size for item in path.parent.iterdir() if item.is_file())
    return {**episode, "hdf5_bytes": path.stat().st_size, "episode_artifact_bytes": stored,
            "eligible_visible_training_starts": eligible, "largest_preclose_anchor_gap_s": largest * 0.05,
            "preclose_anchor_gap_at_most_0p2_s": largest <= 4, "status": "verified"}


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("run", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    out.mkdir()
    print(f"Waiting for {run.with_suffix('.complete')}", flush=True)
    deadline = time.monotonic() + 32400
    while not run.with_suffix(".complete").is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("Overnight recorder completion did not arrive")
        time.sleep(15)
    complete_result = run / "result.json"
    if complete_result.is_file():
        source = json.loads(complete_result.read_text())
        episodes = source["episodes"]
    else:
        journal = json.loads((run / "journal.json").read_text())
        episodes, _ = select(journal["successful_hdf5"], QUOTAS)
    if len(episodes) != len({row["case"] for row in episodes}):
        raise ValueError("Duplicate case in the selected overnight dataset")
    findings = []
    for episode in episodes:
        row = scan(episode)
        findings.append(row)
        if len(findings) % 25 == 0:
            print(f"Verified {len(findings)}/{len(episodes)} HDF5 artifacts", flush=True)
    summary = {"phase": "overnight_data_audit", "configured": len(findings), "verified": len(findings),
               "passed": bool(findings), "cases": findings,
               "groups": dict(Counter(row["group"] for row in findings)),
               "hdf5_bytes": sum(row["hdf5_bytes"] for row in findings),
               "episode_artifact_bytes": sum(row["episode_artifact_bytes"] for row in findings),
               "free_bytes": shutil.disk_usage(run).free,
               "long_preclose_occlusion_episodes": sum(not row["preclose_anchor_gap_at_most_0p2_s"] for row in findings),
               "scope": "physical successes and independent visibility eligibility; does not mark the policy gate"}
    (out / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print({key: value for key, value in summary.items() if key != "cases"}, flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
