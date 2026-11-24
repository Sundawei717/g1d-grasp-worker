"""Read authoritative runtime clocks without synthesizing time from actions."""

import importlib
import math


def _native_simulation_manager_clock():
    module = importlib.import_module("isaacsim.core.simulation_manager.impl.simulation_manager")
    binding = importlib.import_module("isaacsim.core.simulation_manager.bindings._simulation_manager")
    interface = module.SimulationManager._simulation_manager_interface
    if not isinstance(interface, binding.ISimulationManager):
        raise RuntimeError("Original core manager does not hold the native ISimulationManager interface")
    return interface.get_simulation_time(), interface.get_num_physics_steps()


def read_simulation_clock(sim):
    """Return measured time, counter semantics and timestep for Sim 5.1 / Lab 3.

    Lab 3's base PhysicsManager clock is not advanced by its PhysX step method.
    Its compatibility stub also calls a removed IPhysx method in Sim 6. Use
    the original core manager's native ISimulationManager binding instead of
    either Lab alias. Context step calls are a separate diagnostic counter.
    """
    current = getattr(sim, "current_time", None)
    if current is not None:
        value = current
        index = sim.current_time_step_index
        source = "sim.current_time"
        index_source = "sim.current_time_step_index"
        index_semantics = "legacy_context_physics_step_index"
        native_counter = False
        context_steps = int(index)
    else:
        manager = getattr(sim, "physics_manager", None)
        if getattr(manager, "__module__", "") != "isaaclab_physx.physics.physx_manager":
            raise RuntimeError("No audited authoritative simulation clock for this runtime")
        value, index = _native_simulation_manager_clock()
        context_steps = sim.get_physics_step_count()
        source = "original_core_SimulationManager._simulation_manager_interface.get_simulation_time()"
        index_source = "original_core_SimulationManager._simulation_manager_interface.get_num_physics_steps()"
        index_semantics = "native_ISimulationManager_physics_step_counter"
        native_counter = True
    dt = sim.get_physics_dt()
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError("Native simulation time is invalid")
    if isinstance(index, bool) or int(index) != index or int(index) < 0:
        raise ValueError("Actual simulation step counter is invalid")
    if not math.isfinite(float(dt)) or float(dt) <= 0:
        raise ValueError("Actual physics timestep is invalid")
    return {"current_time": float(value), "current_time_step_index": int(index), "physics_dt": float(dt),
            "time_source": source, "step_index_source": index_source, "step_index_semantics": index_semantics,
            "step_index_is_native_physx_counter": native_counter,
            "context_step_calls": int(context_steps),
            "context_step_calls_source": "sim.get_physics_step_count()" if native_counter else index_source,
            "native_time_minus_context_steps_dt": float(value) - int(context_steps) * float(dt),
            "derived_from_action_count": False, "absolute_clock_writeback_supported": False}


def clock_restore_contract(reference, observed):
    """Keep a measured new clock origin; never claim native time was restored."""
    return {"mode": "relative_rollout_from_measured_native_origin", "absolute_clock_restored": False,
            "writeback_performed": False, "reference_time_origin": float(reference["current_time"]),
            "observed_time_origin": float(observed["current_time"]),
            "observed_time_source": observed.get("time_source", "legacy_snapshot_unspecified"),
            "reason": "No audited native absolute-clock writeback API; compare elapsed rollout time."}
