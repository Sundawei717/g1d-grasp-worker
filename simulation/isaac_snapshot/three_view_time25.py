import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
from shutil import copyfile
import subprocess

import h5py

from three_view_detector import sha256, write_json
from three_view_eval import compare, finish, prepare, read


ROLE_QUERY = {"carton": "box", "can": "can", "mug": "mug", "unseen_can": "can"}
FORMER_TEST = ("three_view_pipeline_03", "three_view_generalization_03", "three_view_parts_03")
LIMIT_STEPS = 25 * 20


def completed(root, name):
    p = root / "runs" / name
    if read(p.with_suffix(".complete"))["exit_code"] != 0:
        raise ValueError(f"Required run did not complete successfully: {name}")
    return p


def geometry(row):
    def actor(asset, pose):
        return (Path(asset).resolve().as_posix(), tuple(round(float(v), 6) for v in pose))
    locations = [actor(p["usd"], p["pose"]) for p in row["fixtures"]]
    locations.append(actor(row["target_usd"], row["target_pose_xyz_xyzw"]))
    return tuple(sorted(locations)), tuple(round(float(v), 6) for v in row["home_q"])


def freeze_validation(root, work):
    output = work / "validation_38"
    output.mkdir()
    original_results, rows, prompts = {"a": {}, "b": {}}, [], {}
    for name in FORMER_TEST:
        original = completed(root, name)
        frozen = read(original / "frozen/frozen.json")
        for variant in original_results:
            rec = read(completed(root, f"{name}_eval_{variant}") / "result.json")
            original_results[variant].update({x["case"]: x for x in rec["cases"]})
        for row in frozen["cases"]:
            key = row["case_id"]
            a, b = (original_results[variant][key]["task"] for variant in ("a", "b"))
            fields = ("bbox_0", "affordance_0", "target_query", "arm_side")
            if any(a[f] != b[f] for f in fields):
                raise ValueError(f"Former test prompts differ: {key}")
            prompts[key] = {f: a[f] for f in ("bbox_0", "affordance_0")}
            rows.append(row)
            copyfile(original / "frozen" / row["plan"], output / row["plan"])
    counts = Counter(r["group"] if r["part"] == "upper_side_panels" and r["asset_role"] in {"carton", "can"}
                     else "part" if r["part"] != "upper_side_panels" else "unseen" for r in rows)
    if len(prompts) != 38 or counts != {"A_target": 12, "B_obstacle": 10, "C_start": 8, "unseen": 6, "part": 2}:
        raise ValueError(f"Former test inventory does not match the frozen 38: {counts}")
    write_json(output / "frozen.json", {"cases": rows, "head_prompts": prompts, "roles": ROLE_QUERY,
                                     "scope": "closed-loop development validation, former test 38"})
    write_json(work / "former_10s.json", {v: {key: {f: row[f] for f in ("outcome", "steps", "part_hit")}
                                           for key, row in values.items()} for v, values in original_results.items()})
    return rows


def freeze_test(root, work, original_validation):
    source = completed(root, "grouped_eval_03")
    receipt = read(source / "result.json")
    plans = Path(receipt["frozen_plan"])
    scope = set()
    excluded_ids = {row["case_id"] for row in original_validation}
    excluded_families = {row["scene_family"] for row in original_validation}
    for row in original_validation:
        scope.add(geometry(row))
    original_split = read(root / "runs/three_view_pipeline_03/split.json")["episodes"]
    for row in original_split:
        with h5py.File(row["path"], "r") as h:
            case = json.loads(h["meta/case_json"].asstr()[()])
        scope.add(geometry(case))
        excluded_ids.add(case["case_id"])
        excluded_families.add(case["scene_family"])
    eligible = []
    for result in receipt["cases"]:
        key = result["case"]
        if result["status"] != "strict_success" or key in excluded_ids:
            continue
        file = source / "plans" / f"{key}.json"
        plan = read(file if file.is_file() else plans / f"{key}.json")
        row = plan["case"]
        if row["scene_family"] not in excluded_families and geometry(row) not in scope:
            eligible.append((row, plan))

    def sort_key(entry):
        return hashlib.sha256(("time25:test:" + entry[0]["scene_family"]).encode()).digest()

    families = Counter(row["scene_family"] for row, _ in eligible if row["group"] == "A_target")
    targets = sorted((e for e in eligible if e[0]["group"] == "A_target" and families[e[0]["scene_family"]] == 2),
                     key=sort_key)[:12]
    blocks = sorted((e for e in eligible if e[0]["group"] == "B_obstacle"), key=sort_key)[:10]
    starts = [item for role in ("carton", "can") for item in sorted(
              (e for e in eligible if e[0]["group"] == "C_start" and e[0]["asset_role"] == role), key=sort_key)[:4]]
    if (len(targets), len(blocks), len(starts)) != (12, 10, 8):
        raise ValueError("Could not freeze 30 physically verified independent cases")
    output = work / "independent_test_30"
    output.mkdir()
    rows = []
    for row, plan in targets + blocks + starts:
        case = {**row, "asset_family": "/".join(Path(row["target_usd"]).parts[-3:-1]),
                "split": "independent_test",
                "config_sha256": sha256(root / ("env_resolved.json" if row["asset_role"] == "carton"
                                              else "runs/candidate_can_config.json"))}
        path = row["case_id"] + ".json"
        write_json(output / path, {**plan, "case": case})
        rows.append({**case, "plan": path, "offline_feasible": True})
    write_json(output / "frozen.json", {"cases": rows, "scope": "independent seen-asset test, previously expert-verified"})
    return rows


def inventory(work, validation, test):
    keys = ("suite", "case_id", "group", "asset_role", "part", "scene_family", "fixture_types")
    with (work / "scene_inventory.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for name, rows in (("closed_loop_validation", validation), ("independent_test", test)):
            for row in rows:
                writer.writerow({"suite": name, **{k: row[k] for k in keys[1:-1]},
                                 "fixture_types": ";".join(item["name"] for item in row["fixtures"])})


def write_marker(root, name, code):
    subprocess.run(["python3", str(root / "background_status.py"), "--record-exit", str(code), name],
                   cwd=root, check=True)


def make_batches(root, frozen, fit, directory):
    directory.mkdir()
    planned = directory / "plans"
    prepare(root, frozen, fit, root / "three_view_detector.json", LIMIT_STEPS, planned)
    source_plans = directory / "original_shards"
    source_plans.mkdir()
    jobs = []
    for shard in sorted(planned.glob("shard_*.json")):
        manifest = read(shard)
        role = shard.stem.removeprefix("shard_")
        shard.rename(source_plans / shard.name)
        cases = manifest["cases"]
        for offset in range(0, len(cases), 6):
            name = f"{directory.name}_{role}_{offset // 6 + 1}"
            plan = planned / f"shard_{role}_{offset // 6 + 1}.json"
            write_json(plan, {**manifest, "asset_role": role, "cases": cases[offset:offset+6]})
            config = root / ("env_resolved.json" if role == "carton" else "runs/candidate_can_config.json")
            if role not in {"carton", "can"}:
                config = planned / f"config_{role}.json"
            jobs.append((name, plan, config, directory))
    return jobs


def run_batches(root, jobs):
    queue = iter(jobs)

    def work(gpu):
        for name, plan, config, directory in queue:
            run = root / "runs" / name
            env = {**os.environ, "G1D_WORKER_CONFIG": str(config), "G1D_GROUPED_WALL_SECONDS": "10800"}
            command = ["bash", str(root / "run_lab_school.sh"), str(run), "--grouped-manifest", str(plan),
                       "--grasp-after", str(root / "runs/lab_school_03/result.json"), "--gpu", str(gpu)]
            with run.with_suffix(".log").open("x") as stream:
                with subprocess.Popen(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT) as child:
                    run.with_suffix(".pid").write_text(str(child.pid))
                    with (directory / "worker_pids").open("a") as workers:
                        workers.write(f"{child.pid}\n")
                    print(f"{name} GPU {gpu} PID {child.pid}", flush=True)
                    if child.wait():
                        raise RuntimeError(f"Simulator shard did not complete: {name}")
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in (pool.submit(work, 1), pool.submit(work, 2)):
            future.result()


def evaluate(root, work, label, frozen, fits):
    dirs = [work.with_name(f"{work.name}_{label}_{v}") for v in ("a", "b")]
    lists = [make_batches(root, frozen, fits[0], dirs[0])]
    reused = "head_prompts" in read(frozen / "frozen.json")
    if reused:
        lists.append(make_batches(root, frozen, fits[1], dirs[1]))
        interleaved = [item for pair in zip(*lists, strict=True) for item in pair]
        run_batches(root, interleaved)
    else:
        run_batches(root, lists[0])
        if finish(dirs[0]):
            raise RuntimeError("Independent A test had incomplete cases")
        write_marker(root, dirs[0].name, 0)
        run_batches(root, make_batches(root, frozen, fits[1], dirs[1]))
    for directory in dirs:
        if not (directory / "result.json").exists():
            code = finish(directory)
            write_marker(root, directory.name, code)
            if code:
                raise RuntimeError(f"Incomplete 25-second evaluation: {directory.name}")
    report = work / label
    report.mkdir()
    compare(dirs, report)
    return dirs, report


def summarize_time(work, baseline):
    history = read(work / "former_10s.json")
    result = {}
    for v, directory in zip(("a", "b"), baseline):
        stats = Counter()
        for row in read(directory / "result.json")["cases"]:
            before = history[v][row["case"]]["outcome"]
            if row["outcome"] == "success":
                stats["success_at_25s"] += 1
                if before == "timeout":
                    stats["recovered_after_10s" if row["steps"] > 200 else "recovered_within_10s_on_rerun"] += 1
            if before == "success":
                stats["success_at_10s"] += 1
                stats["old_success_lost"] += row["outcome"] != "success"
        result[v] = stats
    write_json(work / "time_limit_comparison.json", result)


def summarize_seen_baseline(root, work, original_fits):
    frozen = work / "validation_38"
    source = read(frozen / "frozen.json")
    cases = {row["case_id"]: row for row in source["cases"]
             if row["asset_role"] in {"carton", "can"} and row["part"] == "upper_side_panels"}
    if Counter(row["group"] for row in cases.values()) != {"A_target": 12, "B_obstacle": 10, "C_start": 8}:
        raise ValueError("Expected exactly 30 regular seen-asset tasks from the original evaluation")
    history = read(work / "former_10s.json")
    comparisons = {}
    for variant, fit in zip(("a", "b"), original_fits, strict=True):
        receipt = read(fit / "result.json")
        binding = {"receipt": str(fit / "result.json"), "path": receipt["checkpoint"],
                   "sha256": receipt["checkpoint_sha256"]}
        if receipt["selected_step"] != 6000 or sha256(binding["path"]) != binding["sha256"]:
            raise ValueError(f"Original 6,000-step checkpoint changed: {variant}")
        run = work.with_name(f"{work.name}_baseline_{variant}")
        rows = {}
        for path in sorted((run / "plans").glob("shard_*.json")):
            plan = read(path)
            selected = {row["case_id"]: row for row in plan["cases"] if row["case_id"] in cases}
            if not selected:
                continue
            if plan["policy"] != binding or plan["total_steps"] != LIMIT_STEPS:
                raise ValueError(f"Unexpected baseline configuration in {path}")
            shard = run.with_name(f"{run.name}_{path.stem.removeprefix('shard_')}")
            for result in read(shard / "batch_progress.json")["cases"]:
                key = result["case"]
                if key not in selected:
                    continue
                if key in rows or sha256(run / "plans" / selected[key]["plan"]) != sha256(frozen / cases[key]["plan"]):
                    raise ValueError(f"Duplicate or changed frozen scene: {variant} {key}")
                diagnostic = read(shard / key / "expert_diagnostic.json")
                task, rollout = diagnostic["task"], diagnostic["policy_rollout"]
                if any(task[name] != value for name, value in source["head_prompts"][key].items()):
                    raise ValueError(f"Head prompt changed: {variant} {key}")
                steps, outcome = rollout["steps"], rollout["status"]
                success = outcome == "success"
                if success:
                    valid = 0 < steps <= LIMIT_STEPS and rollout["trials"][-1]["success"]
                else:
                    valid = outcome == "timeout" and steps == LIMIT_STEPS
                if not valid or success != (result["status"] == "strict_success"):
                    raise ValueError(f"Unverified rollout: {variant} {key}")
                previous = history[variant][key]
                rows[key] = {"case": key, "group": cases[key]["group"], "status_10s": previous["outcome"],
                             "steps_10s": previous["steps"], "status_25s": outcome, "steps_25s": steps,
                             "source": str(shard / key)}
        if rows.keys() != cases.keys():
            raise ValueError(f"Missing baseline outcomes for {variant}: {sorted(cases.keys() - rows.keys())}")
        comparisons[variant.upper()] = [rows[key] for key in sorted(rows)]
    summary = {variant: count_time_outcomes(rows) for variant, rows in comparisons.items()}
    for variant, rows in comparisons.items():
        summary[variant]["groups"] = {group: count_time_outcomes([row for row in rows if row["group"] == group])
                                      for group in ("A_target", "B_obstacle", "C_start")}
    summary["combined"] = count_time_outcomes(comparisons["A"] + comparisons["B"])
    output = work.with_name(f"{work.name}_seen_baseline")
    output.mkdir(exist_ok=True)
    write_json(output / "result.json", {"phase": "three_view_seen_time_limit_baseline", "passed": True,
                                         "task_count_per_variant": len(cases),
                                         "control_hz": 20, "limit_seconds": [10, 25],
                                         "reference_10s": str(work / "former_10s.json"),
                                         "summary": summary, "cases": comparisons})
    return output


def count_time_outcomes(rows):
    successes = [row for row in rows if row["status_25s"] == "success"]
    gained = [row for row in successes if row["status_10s"] == "timeout"]
    return {"episodes": len(rows),
            "success_10s": sum(row["status_10s"] == "success" for row in rows),
            "success_25s": len(successes),
            "success_by_10s_in_25s_run": sum(row["steps_25s"] <= 200 for row in successes),
            "recovered_after_10s": [row["case"] for row in gained if row["steps_25s"] > 200],
            "recovered_within_10s_on_rerun": [row["case"] for row in gained if row["steps_25s"] <= 200],
            "old_success_lost": [row["case"] for row in rows
                                 if row["status_10s"] == "success" and row["status_25s"] != "success"]}


def select_model(baseline, continuation):
    def score(report):
        d = read(report / "result.json")["variants"].values()
        return (sum(v["counts"].get("success", 0) for v in d),
                sum(group["part_hit"] for v in d for group in v["groups"].values()))
    scores = {"baseline_6000": score(baseline), "continued_8000": score(continuation)}
    winner = max(scores, key=scores.get)
    return winner, scores


def main(root, work, stage):
    old = [completed(root, f"three_view_pipeline_03_fit_{v}") for v in ("a", "b")]
    if stage == "freeze":
        validation = freeze_validation(root, work)
        test = freeze_test(root, work, validation)
        inventory(work, validation, test)
        print("Frozen: 38 former test cases for validation, 30 independent seen-asset test cases.", flush=True)
    elif stage == "baseline":
        dirs, _ = evaluate(root, work, "baseline", work / "validation_38", old)
        summarize_time(work, dirs)
    elif stage == "seen-baseline":
        print(summarize_seen_baseline(root, work, old), flush=True)
    elif stage == "candidate":
        new = [completed(root, f"{work.name}_fit_{v}") for v in ("a", "b")]
        _, report = evaluate(root, work, "candidate", work / "validation_38", new)
        winner, scores = select_model(work / "baseline", report)
        fits = old if winner == "baseline_6000" else new
        write_json(work / "selection.json", {"selected": winner, "scores": scores, "criteria":
                   ["total validation successes across A and B", "total validation part hits", "baseline wins ties"],
                   "checkpoints": {v: str(f) for v, f in zip(("a", "b"), fits)}})
        _, final = evaluate(root, work, "test", work / "independent_test_30", fits)
        write_json(work / "result.json", {"phase": "three_view_time25", "passed": True, "limit_sim_s": 25,
                                          "validation_cases": 38, "test_cases": 30, "selected": winner,
                                          "time_limit_comparison": read(work / "time_limit_comparison.json"),
                                          "independent_test": read(final / "result.json")})


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("stage", choices=("freeze", "baseline", "seen-baseline", "candidate"))
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    main(Path(__file__).resolve().parent, args.output.resolve(), args.stage)
