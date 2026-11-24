"""Bind cached scores to the actual executor, scene and input contract."""
import hashlib
import json
from pathlib import Path


EVALUATOR_VERSION = "adaptive_gripper_v2"
EXECUTION_SOURCES = (
    "unified_grasp.py", "rigid_grasp_probe.py", "vertical_lift.py", "strict_grasp.py",
    "school_planner.py", "part_surface.py", "lab_school_probe.py", "step2_dev.py",
    "three_view_rollout.py", "three_view_inputs.py", "chunk_model.py", "temporal_actions.py",
    "step2_eval.py", "step3b_eval.py", "evaluation_contract.py", "run_lab_school.sh",
)


def fingerprint(value):
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_contract(rows, checkpoint_sha, *, source_root, gate_config, timing,
                        fixture_physics="kinematic", sources=EXECUTION_SOURCES, runtime_files=()):
    cases = []
    for row in rows:
        # Bind the whole planned scene as well as its separately supplied prompt/seed.
        cases.append({"case": row["case_id"], "scene_id": row["scene_id"], "seed": row["seed"],
                      "head_prompt": row["head_prompt"], "plan_sha256": file_sha(row["plan"])})
    if len({r["case"] for r in cases}) != len(cases):
        raise ValueError("Duplicate evaluation cases")
    return {"version": EVALUATOR_VERSION, "checkpoint_sha256": checkpoint_sha,
            "fixture_physics": fixture_physics, "gate": gate_config, "timing": timing,
            "cases": cases, "sources": {name: file_sha(Path(source_root) / name) for name in sources},
            "runtime_files": {str(path): file_sha(path) for path in sorted(set(map(str,runtime_files)))}}


def validate_cached_result(data, contract):
    if (data.get("evaluation_contract") != contract
            or data.get("evaluation_contract_sha256") != fingerprint(contract)
            or data.get("checkpoint_sha256") != contract["checkpoint_sha256"]
            or [r["case"] for r in data.get("episodes", [])] != [r["case"] for r in contract["cases"]]):
        raise ValueError("Cached evaluation has a different or unversioned executor/scene contract")
    return data["episodes"]


def execution_diagnostics(roll):
    """Separate a rejected plan, an executed lift and a late gate without changing scores."""
    gates = roll["gate"]
    trial = (roll["trials"] or [{}])[0]
    ready = [g["step"] for g in gates if g["ready"]]
    return {"trial_requested": bool(roll["trials"]),
            "trial_control_steps": trial.get("steps_used", 0),
            "trial_motion_executed": trial.get("steps_used", 0) > 0,
            "precheck_rejected": trial.get("failure_category") == "lift_path_rejected",
            "first_gate_ready_step": min(ready) if ready else None,
            "empty_at_limit_steps": sum(g.get("empty_at_mechanical_limit", False) for g in gates),
            "loaded_control_steps": sum(g.get("target_load_ready", False) for g in gates),
            "max_jaw_stable_steps": max((g["jaw_stable_steps"] for g in gates), default=0)}
