import argparse
import json
from collections import Counter
from pathlib import Path
from shutil import copyfile

import numpy as np

from audit_grasp_receipt import audit
from grouped_video import representative_runs
from prepare_grouped import write_json
from school_planner import sha256


def verify_model(binding):
    fit = Path(binding["fit_result"])
    model = Path(binding["path"])
    result = json.loads(fit.read_text())
    marker = json.loads(fit.parent.with_suffix(".complete").read_text())
    if (sha256(fit) != binding["fit_sha256"] or sha256(model) != binding["sha256"]
            or marker["exit_code"] != 0 or not result["complete_optimization"]
            or result.get("base_episodes", result["episodes"]) != 20 or model != Path(result["checkpoint"])):
        raise ValueError("The completed fit20 checkpoint is missing or changed")
    return result


def prepare(bootstrap, fit, urdf, evidence, output):
    bootstrap_result = bootstrap / "result.json"
    source = json.loads(bootstrap_result.read_text())
    if json.loads(bootstrap.with_suffix(".complete").read_text())["exit_code"] != 0 or not source["passed"]:
        raise ValueError("Expected the completed independent training collection")
    fit_receipt = fit / "result.json"
    model = Path(json.loads(fit_receipt.read_text())["checkpoint"])
    policy = {"fit_result": str(fit_receipt.resolve()), "fit_sha256": sha256(fit_receipt),
              "path": str(model.resolve()), "sha256": sha256(model)}
    training = verify_model(policy)
    if training["config"]["dataset_sha256"] != sha256(bootstrap_result):
        raise ValueError("Fit20 used a different bootstrap manifest")
    selected = {row["case"] for row in source["episodes"] if row["use_for_fit20"]}
    if len(selected) != 20:
        raise ValueError("Expected exactly 20 fitting layouts")
    output.mkdir()
    for role in ("carton", "can"):
        origin = json.loads((bootstrap / "plans" / f"shard_{role}.json").read_text())
        rows = [row for row in origin["cases"] if row["case_id"] in selected]
        for row in rows:
            copyfile(bootstrap / "plans" / row["plan"], output / row["plan"])
        write_json(output / f"shard_{role}.json",
                   {**origin, "phase": "fit20_closed_loop", "cases": rows, "policy": policy,
                    "lift_model": {"urdf": str(urdf.resolve()), "evidence": str(evidence.resolve())}})
    print(f"Prepared {len(selected)} in-sample policy episodes from the frozen fit20 manifest", flush=True)


def prepare_paired(reference, fit, mode, output, replace_checkpoint=False):
    from policy_rollout import PROMPT_MODES

    if mode not in PROMPT_MODES[1:]:
        raise ValueError(f"Unsupported comparison mode {mode}")
    result = json.loads((reference / "result.json").read_text())
    if result.get("evaluation_completed") is not True or result["configured"] != 20:
        raise ValueError("The baseline requires 20 completed paired evaluations")
    if replace_checkpoint:
        if result.get("prompt_mode") != mode:
            raise ValueError("A checkpoint comparison must keep the reference test prompt mode")
        fit_receipt = fit / "result.json"
        model = Path(json.loads(fit_receipt.read_text())["checkpoint"])
        policy = {"fit_result": str(fit_receipt.resolve()), "fit_sha256": sha256(fit_receipt),
                  "path": str(model.resolve()), "sha256": sha256(model)}
        training = verify_model(policy)
    expected = {row["case"] for row in result["cases"]}
    output.mkdir()
    observed = set()
    for role in ("carton", "can"):
        source = reference / "plans" / f"shard_{role}.json"
        manifest = json.loads(source.read_text())
        baseline_training = verify_model(manifest["policy"])
        if replace_checkpoint:
            keys = ("dataset_sha256", "visibility_replay_sha256", "corrections_sha256", "seed",
                    "effective_batch", "microbatch", "action_reference", "weights_sha256")
            if (training["updates"] != baseline_training["updates"]
                    or any(training["config"].get(key) != baseline_training["config"].get(key) for key in keys)):
                raise ValueError("The checkpoint comparison changed the training data sources or update budget")
            manifest["policy"] = policy
        elif Path(manifest["policy"]["fit_result"]) != fit / "result.json":
            raise ValueError("The requested checkpoint differs from the baseline")
        for row in manifest["cases"]:
            observed.add(row["case_id"])
            copyfile(reference / "plans" / row["plan"], output / row["plan"])
        write_json(output / f"shard_{role}.json", {**manifest, "prompt_mode": mode})
    if expected != observed or len(observed) != 20:
        raise ValueError("The paired run did not preserve the frozen 20-case manifest")
    print(f"Prepared {len(observed)} paired cases with prompt mode {mode}", flush=True)


def finish(output):
    cases, timing = [], []
    modes = set()
    for role in ("carton", "can"):
        run = output.parent / f"{output.name}_{role}"
        shard = json.loads((output / "plans" / f"shard_{role}.json").read_text())
        mode = shard.get("prompt_mode", "gt_visible")
        modes.add(mode)
        expected = {row["case_id"]: row for row in shard["cases"]}
        observed = {}
        for name in ("result.json", "batch_progress.json"):
            if (run / name).is_file():
                observed = {row["case"]: row for row in json.loads((run / name).read_text())["cases"]}
                break
        for key, original in expected.items():
            row = {**observed.get(key, {"case": key, "group": original["group"], "status": "infrastructure_fail"}),
                   "shard": run.name}
            diagnostic = run / key / "expert_diagnostic.json"
            if diagnostic.is_file():
                item = json.loads(diagnostic.read_text())
                for field in ("home_q", "target_pose_xyz_xyzw", "fixtures", "seed", "command_delay_substeps"):
                    if item["case"][field] != original[field]:
                        raise ValueError(f"Policy replay changed its initial scene: {key}")
                policy = item.get("policy_rollout", {})
                if policy and policy.get("prompt_mode") != mode:
                    raise ValueError(f"Runtime used a different prompt mode: {key}")
                timing.extend(policy.get("decisions", []))
                if row["status"] == "strict_success" and not audit(diagnostic.parent)["strict_success"]:
                    raise ValueError(f"Strict success did not reproduce: {key}")
            cases.append(row)
    if len(cases) != len({row["case"] for row in cases}) or len(cases) != 20 or len(modes) != 1:
        raise ValueError("Missing or duplicate policy test identities")
    counts = Counter(row["status"] for row in cases)
    failures = Counter((row.get("policy_result") or {}).get("status") or row["status"]
                       for row in cases if row["status"] != "strict_success")
    timing_fields = ("capture_and_prompt_ms", "crop_and_inference_ms", "total_ms")
    latency = {key: dict(zip(("p50", "p95"), np.percentile([row[key] for row in timing], [50, 95]).tolist()))
               for key in timing_fields} if timing else {}
    evaluated = counts["infrastructure_fail"] == 0 and bool(timing)
    passed = evaluated and counts["strict_success"] > 0
    mode = modes.pop()
    result = {"phase": "fit20_closed_loop", "scope": f"in-sample controller diagnostic with {mode} prompts",
              "prompt_mode": mode,
              "configured": len(cases), "strict_success": counts["strict_success"], "outcomes": dict(counts),
              "failure_categories": dict(failures), "evaluation_completed": evaluated,
              "passed": passed, "decision_latency_ms": latency,
              "cases": cases, "representative_videos": representative_runs(cases, output.parent, output / "videos")}
    write_json(output / "result.json", result)
    print(json.dumps({key: result[key] for key in ("configured", "strict_success", "outcomes", "failure_categories",
                                                   "decision_latency_ms", "passed")}, indent=2), flush=True)
    return 0 if passed else 1


def main():
    cli = argparse.ArgumentParser()
    modes = cli.add_subparsers(required=True, dest="command")
    prep = modes.add_parser("prepare")
    for name in ("bootstrap", "fit", "urdf", "evidence", "output"):
        prep.add_argument(name, type=Path)
    paired = modes.add_parser("prepare-paired")
    paired.add_argument("reference", type=Path)
    paired.add_argument("fit", type=Path)
    paired.add_argument("mode")
    paired.add_argument("output", type=Path)
    paired.add_argument("--replace-checkpoint", action="store_true")
    modes.add_parser("finish").add_argument("output", type=Path)
    args = cli.parse_args()
    if args.command == "prepare":
        prepare(args.bootstrap, args.fit, args.urdf, args.evidence, args.output)
        return 0
    if args.command == "prepare-paired":
        prepare_paired(args.reference, args.fit, args.mode, args.output, args.replace_checkpoint)
        return 0
    return finish(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
