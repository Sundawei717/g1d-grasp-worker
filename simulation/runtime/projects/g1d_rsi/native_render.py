"""Drive the pinned Lab3 RTX app loop for owned deprecated Camera products."""
from .simulation_clock import read_simulation_clock


def prepare_owned_rendering(env):
    import carb
    import omni.usd
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_rtx_hydra_engine_attached

    sim = env.sim.sim
    settings = carb.settings.get_settings()
    settings.set("/isaaclab/render/rtx_sensors", True)
    ensure_rtx_hydra_engine_attached()
    engines = list(omni.usd.get_context().get_attached_hydra_engine_names())
    if "rtx" not in engines or not sim.is_rendering:
        raise RuntimeError(f"Owned Camera rendering is inactive: engines={engines}, rendering={sim.is_rendering}")
    return {"schema": "g1d.lab3_owned_render_setup.v1", "attached_hydra_engines": engines,
            "is_rendering": bool(sim.is_rendering),
            "offscreen": settings.get("/isaaclab/render/offscreen"),
            "rtx_sensors": settings.get("/isaaclab/render/rtx_sensors"),
            "multitick": settings.get("/rtx/hydra/supportMultiTickRate"),
            "orchestrator_enabled": settings.get("/exts/omni.replicator.core/Orchestrator/enabled"),
            "capture_on_play": settings.get("/persistent/omni/replicator/captureOnPlay"),
            "visualizers": [type(v).__name__ for v in sim.visualizers],
            "render_entry": "isaaclab_physx.renderers.isaac_rtx_renderer_utils.ensure_isaac_rtx_render_update"}


def render_owned_cameras(env):
    """Render one generation and require native time and native steps unchanged."""
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

    sim = env.sim.sim
    before = read_simulation_clock(sim)
    previous = sim.get_setting("/app/player/playSimulations")
    if type(previous) is not bool:
        raise RuntimeError(f"Expected an explicit Kit player simulation flag, got {previous!r}")
    try:
        # Lab3 render() increments render_generation and dispatches render hooks.
        # With no visualizer, its official RTX helper supplies the app.update.
        sim.render()
        ensure_isaac_rtx_render_update()
    finally:
        sim.set_setting("/app/player/playSimulations", previous)
    after = read_simulation_clock(sim)
    if after["current_time"] != before["current_time"] \
            or after["current_time_step_index"] != before["current_time_step_index"]:
        raise RuntimeError(f"Owned Camera render advanced native physics: before={before}, after={after}")
