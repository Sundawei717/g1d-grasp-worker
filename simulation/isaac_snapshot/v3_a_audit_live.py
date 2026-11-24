"""Tap the production V3-A model call at actual simulator observations 0, 2 and 4."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from background_status import record_complete
from three_view_detector import write_json
from three_view_eval import read


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
WORK = RUNS / "v3_a_root_audit_01"
TRAIN_A = RUNS / "step2_v3_01_train_a"
NAMES = ("head_5ch", "left_3ch", "state_2x17", "prompt_meta", "right_3ch")
SPECS = {"C_7302": ("01", "mug"), "C_8200": ("03", "basket")}


def tap(case):
    import lab_school_probe
    import unified_grasp
    from three_view_rollout import ThreeViewPolicy
    from three_view_inputs import rgb224

    context = {}
    begin = unified_grasp.execute
    predict = ThreeViewPolicy.predict
    best = None
    captures = []

    def execute(**params):
        source = params["source"]

        def record_step(step, rgb, info, feedback):
            context.update(step=step, feedback=feedback)
            return source(step, rgb, info, feedback)

        with patch.object(unified_grasp, "EPISODE_STEPS", 5):
            return begin(**{**params, "source": record_step})

    def forward_call(policy, name, rgb, frame, task, state, options, arrays):
        def capture(module, inputs, result):
            for key, value in zip(NAMES, inputs, strict=True):
                arrays[f"{name}_{key}"] = value[0].detach().cpu().numpy().copy()
            arrays[f"{name}_pred_action_8x8"] = result[0].detach().cpu().numpy().copy()

        hook = policy.model.register_forward_hook(capture)
        try:
            returned = predict(policy, rgb, frame, task, state, **options)
        finally:
            hook.remove()
        arrays[f"{name}_executed_prefix"] = returned[0]
        return returned

    def observe(policy, rgb, frame, task, state, **options):
        nonlocal best
        if not context:
            raise ValueError("Online prediction did not originate in the unified executor")
        step, feedback = context["step"], context["feedback"]
        prev = feedback["previous_control"]
        images = {"head": rgb, **{k: v[0] for k, v in frame["wrist_views"].items()}}
        data = {f"raw_{k}": rgb224(v) for k, v in images.items()}
        data["state_current_physical"] = state
        data["state_previous_physical"] = options["previous_state"]
        result = forward_call(policy, "last", rgb, frame, task, state, options, data)
        if step == 0:
            best = ThreeViewPolicy(TRAIN_A / "best_val.pt", policy.device)
            forward_call(best, "best_val", rgb, frame, task, state, options, data)
        write_json(WORK / f"{case}_step{step}.json", {
            "step": step, "current": {k: feedback[k] for k in ("eef_pose","arm","width_m","sim_time_s")},
            "previous": {k: prev[k] for k in ("eef_pose","arm","width_m","sim_time_s")},
            "task": task})
        np.savez_compressed(WORK / f"{case}_step{step}.npz", **data)
        captures.append(step)
        print(f"{case} live input captured at step {step}", flush=True)
        return result

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *original_argv[3:]]
        with patch.object(unified_grasp, "execute", execute), patch.object(ThreeViewPolicy, "predict", observe):
            lab_school_probe.main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    finally:
        sys.argv = original_argv
    if captures != [0, 2, 4]:
        raise ValueError(f"Expected real rollout calls at [0, 2, 4], captured {captures}")


def run_live():
    import os
    import subprocess

    from step2_dev import wait_gpu

    WORK.mkdir(exist_ok=True)
    gpu = wait_gpu(2, exclude=(0,1), timeout=600)
    for case, (index, _) in SPECS.items():
        source = RUNS / f"step2_v3_01/sim/step2_v3_01_train_a_{index}"
        manifest = read(source / "manifest.json")
        chosen = [r for r in manifest["cases"] if r["case_id"] == case]
        if len(chosen) != 1 or manifest["detector"] is not None or not manifest["policy"]["path"].endswith("/last.pt"):
            raise ValueError(f"Wrong Step 2 frozen scenario: {case}")
        tag = f"{WORK.name}_sim_{case}"
        cfg = WORK / tag
        cfg.mkdir(exist_ok=True)
        for row in chosen:
            write_json(cfg / row["plan"], read(source / row["plan"]))
        manifest.update(cases=chosen, initial_head_prompts={case: manifest["initial_head_prompts"][case]})
        path = cfg / "manifest.json"
        write_json(path, manifest)
        runtime = Path("/dev/shm") / f"g1d-{os.getuid()}-{tag}"
        runtime.mkdir(exist_ok=True)
        env = {**os.environ, "G1D_GPU_UUID": gpu["uuid"], "G1D_RENDER_GPU": str(gpu["gpu"]),
               "G1D_RUNTIME_DIR": str(runtime)}
        command = ["bash", str(ROOT/"run_v3_a_audit_sim.sh"), case, str(source/"config.json"), str(RUNS/tag),
                   "--grouped-manifest", str(path), "--grasp-after", str(RUNS/"lab_school_03/result.json"),
                   "--gpu", "0"]
        with (RUNS / f"{tag}.log").open("w") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            (RUNS / f"{tag}.pid").write_text(str(proc.pid))
            print(f"{tag} GPU {gpu['gpu']} PID {proc.pid}", flush=True)
            code = proc.wait()
        record_complete(RUNS / tag, code)
        if code:
            raise RuntimeError(f"Simulator tap failed for {case}; see runs/{tag}.log")


if __name__ == "__main__":
    if len(sys.argv)>2 and sys.argv[1] == "tap":
        tap(sys.argv[2])
    else:
        job = RUNS / f"{WORK.name}_live"
        try:
            run_live()
        except BaseException:
            record_complete(job, 1)
            raise
        else:
            record_complete(job, 0)
