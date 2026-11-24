from dataclasses import replace

import pytest

from simulation.isaac_snapshot.strict_grasp import PhysicsSample, evaluate
from simulation.isaac_snapshot.unified_grasp import LiftGateConfig, execute


def lift_trace():
    dt = 1 / 600
    samples = []
    for i in range(811):
        height = .08 * min(i / 480, 1)
        samples.append(PhysicsSample(i*dt, .7+height, 1+height, (0,0,-.3), (0,0,0,1),
                                     i == 0, frozenset(), False, False, frozenset(("pad_1","pad_2"))))
    return samples


def test_contact_diagnostics_and_wrong_part_do_not_override_physical_pickup():
    trace = [replace(s, forbidden_contact=True) for s in lift_trace()]
    result = evaluate(trace,lift_start=0,hold_start=480,dt=1/600,unified=True,expected_lift_m=.08)
    assert result["success"]  # Part attribution is an independent diagnostic.


def test_supported_or_dropped_object_cannot_be_counted_as_success():
    trace = lift_trace()
    trace[-1] = replace(trace[-1],target_supported=True,target_z_m=.7,target_inner_pads=frozenset())
    result = evaluate(trace,lift_start=0,hold_start=480,dt=1/600,unified=True,expected_lift_m=.08)
    assert not result["success"] and "target_not_lifted_clear" in result["reasons"]


@pytest.mark.parametrize("first_loaded,expected", ((271,"success"),(272,"timeout")))
def test_lift_and_hold_are_inside_the_300_step_budget(first_loaded,expected):
    tick = [0]

    def feedback():
        return {"sim_time_s":tick[0]/20,"width_m":.05,"eef_xyz_m":[0,0,0],
                "max_joint_speed_rad_s":0.,"pad_normal_force_n":
                    {"pad_1":5.,"pad_2":5.} if tick[0]>=first_loaded else {}}

    def capture():
        return None,{"sensor_sim_time_s":tick[0]/20}

    def apply(command):
        tick[0] += 1
        return .041,0.

    def lift(state):
        assert tick[0]<=273
        tick[0] += 27
        return {"success":True,"steps_used":27,"part_hit":True}

    roll = execute(initial_frame=capture(),capture=capture,feedback=feedback,
                   source=lambda *args:([0],{}),apply=apply,lift=lift,config=LiftGateConfig())
    assert roll["status"] == expected
    assert roll["steps"] == tick[0] == 300
    assert roll["sim_duration_s"] == 15
