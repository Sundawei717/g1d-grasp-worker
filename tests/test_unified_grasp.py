import pytest

from simulation.isaac_snapshot.unified_grasp import (
    JAW_MIN_M,
    LiftGate,
    LiftGateConfig,
    guarded_gripper_width,
    ticks,
)


LOADED = {"pad_1": 5.0, "pad_2": 5.0}


def feedback(width, forces=LOADED):
    return {
        "width_m": width,
        "pad_normal_force_n": forces,
        "eef_xyz_m": [0.0, 0.0, 0.0],
        "max_joint_speed_rad_s": 0.0,
    }


def steady_gate(simulated_feedback, commands, config):
    gate = LiftGate(config, simulated_feedback)
    return [gate.update(command, simulated_feedback) for command in commands]


def test_jitter_does_not_release_preload():
    config = LiftGateConfig()
    previous = guarded_gripper_width(0.0410, 0.050, 0.045, LOADED, config)
    assert previous == pytest.approx(0.0410)
    assert guarded_gripper_width(0.0412, 0.050, previous, LOADED, config) == pytest.approx(0.0412)
    legacy = LiftGateConfig(gripper_mode="legacy")
    assert guarded_gripper_width(0.0412, 0.050, previous, LOADED, legacy) == pytest.approx(0.046)


def test_force_dropout_holds_preload_and_opening_is_rate_limited():
    config = LiftGateConfig()
    assert guarded_gripper_width(0.0412, 0.050, 0.041, {}, config) == pytest.approx(0.0412)
    assert guarded_gripper_width(0.0300, 0.050, 0.041, {}, config) == pytest.approx(0.041)
    previous = 0.041
    for expected in (0.045, 0.049, 0.053, 0.054):
        previous = guarded_gripper_width(0.060, 0.050, previous, LOADED, config)
        assert previous == pytest.approx(expected)


@pytest.mark.parametrize(
    ("observed", "command", "required_reserve"),
    ((0.082, 0.080, 0.002), (0.005, 0.003, 0.001965),
     (0.0015, JAW_MIN_M, 0.000215), (JAW_MIN_M, JAW_MIN_M, 0.0)),
)
def test_width_aware_gate_can_reach_both_force_and_reserve(observed, command, required_reserve):
    config = LiftGateConfig()
    count = ticks(config.close_duration_s)
    states = steady_gate(feedback(observed), [command] * count, config)
    assert all(not state["ready"] for state in states[:-1])
    assert states[-1]["ready"]
    assert states[-1]["required_closing_reserve_m"] == pytest.approx(required_reserve)


@pytest.mark.parametrize("observed,command", ((0.082, 0.080), (0.005, 0.003)))
def test_legacy_gates_reproduce_the_wide_and_thin_dead_zones(observed, command):
    config = LiftGateConfig(gripper_mode="legacy")
    assert not steady_gate(feedback(observed), [command] * 8, config)[-1]["ready"]


def test_empty_stop_and_opening_do_not_authorize_a_lift():
    config = LiftGateConfig()
    state = steady_gate(feedback(JAW_MIN_M, {}), [JAW_MIN_M] * 8, config)[-1]
    assert state["empty_at_mechanical_limit"]
    assert not state["ready"]
    assert not steady_gate(feedback(0.050), [0.052] * 8, config)[-1]["ready"]
    assert not steady_gate(feedback(0.050, {"pad_1": 10.0}), [0.041] * 8, config)[-1]["ready"]


def test_a_force_dropout_breaks_the_three_step_streak():
    gate = LiftGate(LiftGateConfig(), feedback(0.050))
    for forces in (LOADED, LOADED, LOADED, {}, LOADED, LOADED):
        assert not gate.update(0.041, feedback(0.050, forces))["ready"]
    assert gate.update(0.041, feedback(0.050))["ready"]
