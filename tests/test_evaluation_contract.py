from copy import deepcopy

import pytest

from simulation.isaac_snapshot.evaluation_contract import (
    evaluation_contract, execution_diagnostics, fingerprint, validate_cached_result,
)


def contract(tmp_path):
    (tmp_path / "plan.json").write_text('{"initial_state": 1}')
    (tmp_path / "executor.py").write_text("version = 1\n")
    rows = [{"case_id":"case1", "scene_id":"scene1", "seed":13,
             "head_prompt":{"bbox_0":[0,0,1,1], "affordance_0":[.3,.5]},
             "plan":str(tmp_path / "plan.json")}]
    value = evaluation_contract(rows, "weights1", source_root=tmp_path,
        gate_config={"gripper_mode":"adaptive"}, timing={"episode_steps":300},
        sources=("executor.py",))
    return rows, value


def saved(value):
    return {"checkpoint_sha256":value["checkpoint_sha256"],
            "evaluation_contract":deepcopy(value),"evaluation_contract_sha256":fingerprint(value),
            "episodes":[{"case":"case1","status":"success"}]}


@pytest.mark.parametrize("field", ("gate", "timing", "sources", "runtime_files", "cases", "fixture_physics", "checkpoint_sha256"))
def test_changed_execution_or_inputs_cannot_reuse_scores(tmp_path, field):
    _, current = contract(tmp_path)
    cache = saved(current)
    current[field] = "changed"
    with pytest.raises(ValueError, match="executor/scene"):
        validate_cached_result(cache, current)


def test_identical_contract_resumes_but_unversioned_or_incomplete_result_fails(tmp_path):
    _, value = contract(tmp_path)
    cache = saved(value)
    assert validate_cached_result(cache,value) == cache["episodes"]
    del cache["evaluation_contract"]
    with pytest.raises(ValueError):
        validate_cached_result(cache,value)
    cache = saved(value)
    cache["episodes"] = []
    with pytest.raises(ValueError):
        validate_cached_result(cache,value)


def test_same_case_id_with_changed_plan_changes_contract(tmp_path):
    rows, value = contract(tmp_path)
    (tmp_path / "plan.json").write_text('{"initial_state": 2}')
    other = evaluation_contract(rows,"weights1",source_root=tmp_path,
        gate_config=value["gate"],timing=value["timing"],sources=("executor.py",))
    assert fingerprint(value) != fingerprint(other)


def test_changed_calibration_invalidates_cached_evaluation(tmp_path):
    rows,value = contract(tmp_path)
    camera = tmp_path / "camera.json"
    camera.write_text('{"focal_length": 10}')
    arguments = dict(source_root=tmp_path,gate_config=value["gate"],timing=value["timing"],
                     sources=("executor.py",),runtime_files=[camera])
    before = evaluation_contract(rows,"weights1",**arguments)
    camera.write_text('{"focal_length": 11}')
    after = evaluation_contract(rows,"weights1",**arguments)
    assert fingerprint(before) != fingerprint(after)


def test_rejected_precheck_is_not_reported_as_an_executed_lift():
    roll = {"gate":[{"step":120,"ready":True,"jaw_stable_steps":3}],
            "trials":[{"steps_used":0,"failure_category":"lift_path_rejected"}]}
    summary = execution_diagnostics(roll)
    assert summary["precheck_rejected"] and summary["trial_requested"]
    assert not summary["trial_motion_executed"]
    roll["trials"] = [{"steps_used":27,"failure_category":"lost_grip_during_lift_or_hold"}]
    summary = execution_diagnostics(roll)
    assert summary["trial_motion_executed"] and not summary["precheck_rejected"]
