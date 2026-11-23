import argparse
import json
import os
import re
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


RUNS = Path(__file__).resolve().parent / "runs"


def run_base(runs, job):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", job):
        raise ValueError(f"Invalid run name: {job!r}")
    return runs / job


def read_exit(base):
    path = base.with_suffix(".exit")
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return int(text) if re.fullmatch(r"[0-9]{1,3}", text) and int(text) <= 255 else None


def record_complete(base, code, completed_at=None):
    if not 0 <= code <= 255:
        raise ValueError(f"Invalid exit code: {code}")
    marker = base.with_suffix(".complete")
    data = {
        "status": "COMPLETE",
        "outcome": "PASS" if code == 0 else "FAILED",
        "exit_code": code,
        "completed_at_utc": datetime.fromtimestamp(
            completed_at if completed_at is not None else time.time(), timezone.utc
        ).isoformat(timespec="seconds"),
    }
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=base.parent, delete=False) as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        try:
            os.link(temporary, marker)
        except FileExistsError:
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if existing.get("status") != "COMPLETE" or existing.get("exit_code") != code:
                raise ValueError(f"Conflicting completion record: {marker}")
            data = existing
    finally:
        temporary.unlink()
    return data


def complete_from_exit(base):
    code = read_exit(base)
    if code is None:
        return None
    return record_complete(base, code, base.with_suffix(".exit").stat().st_mtime)


def live_workers(base):
    workers = []
    supervisors = {"run_grouped_expert.sh", "run_collection_preview.sh", "run_grouped_obstacle_replay.sh",
                   "run_dataset_alignment.sh", "run_train_bootstrap.sh", "run_fit20.sh",
                   "run_fit20_closed_loop.sh", "run_fit20_residual.sh", "run_visibility_replay.sh",
                   "run_policy_correction_capture.sh", "run_fit20_corrected.sh", "setup_sam2_tracking.sh",
                   "run_tracking_benchmark.sh", "run_tracking_motion.sh", "run_collect_training_overnight.sh",
                   "run_prompt_mode_comparison.sh", "run_fit20_occlusion_retrain.sh", "run_three_view_collection.sh",
                   "run_three_view_pipeline.sh", "run_three_view_extra_physics.sh", "run_three_view_generalization.sh",
                   "run_three_view_parts_capture.sh", "run_three_view_scale.sh", "run_lab_school.sh",
                   "run_three_view_diversity.sh", "run_three_view_training_data.sh", "run_three_view_time25.sh",
                   "run_three_view_step0.sh"}
    entrypoints = {"lab_school_probe.py", "inspect_environment.py", "prepare_grouped.py", "prepare_stage3_splits.py",
                     "inspect_route_visibility.py", "prepare_policy_corrections.py",
                     "three_view_train.py", "three_view_detector.py", "three_view_extra_plans.py",
                     "three_view_time25.py", "three_view_step0.py", "step1_detector.py",
                     "step1b_detector.py", "step1b_asset.py", "step1b_scenes.py"}
    plan_revisions = {"repair_grouped.py", "screen_grouped_starts.py", "prepare_policy_corrections.py",
                      "audit_overnight_data.py"}
    try:
        shard_pids = set((base / "worker_pids").read_text().split())
    except (FileNotFoundError, NotADirectoryError):
        shard_pids = set()
    for process in Path("/proc").glob("[0-9]*"):
        try:
            if process.stat().st_uid != os.getuid():
                continue
            args = (process / "cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        names = [Path(arg).name for arg in args]
        tier = re.fullmatch(r"(tier1_[A-Za-z0-9_]+?)(?:_(plan|model|cache|cache_smoke))?",base.name)
        tier_scripts = {"plan":"plan.sh", "model":"model_validation.sh", "cache_smoke":"cache_smoke.sh", None:"run.sh"}
        tier_path = f"runs/{tier[1]}/{tier_scripts.get(tier[2], 'run.sh')}" if tier else None
        if (str(base / "run.sh") in args or f"runs/{base.name}/run.sh" in args or tier_path in args
                or tier_path and str(base.parent.parent / tier_path) in args):
            workers.append(int(process.name))
            continue
        if ("step2_dev.py" in names and base.name == "step2_v3_01_dev"
                or "step2_report.py" in names and base.name == "step2_v3_01_report"
                or "step2_teacher_diagnostic.py" in names and base.name == "step2_v3_01_teacher"):
            workers.append(int(process.name))
            continue
        if (base.name in {"step3b_01","step3b_01_stage1","step3b_01_stage3"}
                and "step3b_control.py" in names and "run" in args):
            try:
                state = json.loads((base.parent/"step3b_01/controller.json").read_text())
            except (FileNotFoundError,json.JSONDecodeError):
                state = {}
            phase = str(state.get("stage",0))
            if state.get("state") == "RUNNING" and base.name in {"step3b_01", f"step3b_01_stage{phase}"}:
                workers.append(int(process.name))
                continue
        if base.name == "step3b_01_stage2" and "step3b_train.py" in names:
            workers.append(int(process.name))
            continue
        step3a = {"step3a_fetch_ycb.sh":"ycb", "step3a_assets.py":"assets", "step3a_l1.py":"l1",
                  "step3a_rays.py":"rays", "step3a_memory_watch.py":"memory_guard", "step3a_switch_gpu.py":"gpu_switch", "step3a_preplan.py":"preplan", "step3a_pilots_parallel.py":"pilots", "step3a_guide_diagnostics.py":"shape_guide", "step3a_local_substitutes.py":"substitutes", "step3a_local_guides.py":"local_guides", "step3a_sugar_calibration.py":"sugar_calibration", "step3a_basket_calibration.py":"basket_calibration", "step3a_round_followup.py":"round_followup", "step3a_compact_calibration.py":"compact_calibration", "step3a_final_substitutes.py":"final_substitutes", "step3a_marker_detour.py":"marker_detour", "step3a_marker_followup.py":"marker_followup", "resume_step3a_pilots.py":"pilot_reload", "run_step3a_report_watch.sh":"report", "step3a_freeze.py":"freeze", "step3a_probe_plan.py":"cpu_probe"}
        suite = args[-1] if "step3a_suites.py" in names else None
        if (any(script in names and base.name == f"step3a_01_{stage}" for script,stage in step3a.items())
                or "resume_step3a_pilots.py" in names and "--parallel" in args and base.name == "step3a_01_parallel_pilots"
                or "step3a_rays.py" in names and "--local-substitutes" in args and base.name=="step3a_01_substitute_rays"
                or suite and base.name == f"step3a_01_{suite}"
                or "run_v3_b40_fit.sh" in names and base.name == "v3_b40_step3a_01_fit"
                or {"v3_b40_eval.py","reserve_b40_eval_gpu.sh"}.intersection(names) and base.name == "v3_b40_step3a_01"):
            workers.append(int(process.name))
            continue
        if "step2_eval.py" in names:
            start = names.index("step2_eval.py") + 1
            if len(args) > start:
                stage = args[start]
                job = "step2_v3_01" if stage == "supervisor" else f"step2_v3_01_eval_{stage}"
                if base.name == job:
                    workers.append(int(process.name))
            continue
        if "step1b_accept.sh" in names:
            argument = names.index("step1b_accept.sh") + 1
            if argument < len(args) and base.name == f"step1b_accept_{args[argument]}":
                workers.append(int(process.name))
            continue
        if "run_three_view_step0.sh" in names:
            start = names.index("run_three_view_step0.sh") + 1
            if len(args) < start + 2:
                continue
            work, stage = args[start:start + 2]
            args = [work, f"{work}_{stage}"]
        elif "three_view_step0.py" in names:
            start = names.index("three_view_step0.py") + 1
            if len(args) < start + 2:
                continue
            stage, work = args[start:start + 2]
            args = [work, f"{work}_{stage}"]
        elif plan_revisions.intersection(names):
            args = args[-1:]
        elif supervisor := supervisors.intersection(names):
            first_argument = names.index(supervisor.pop()) + 1
            args = args[first_argument:first_argument + 1]
        elif not entrypoints.intersection(names):
            continue
        if str(base) in args or f"runs/{base.name}" in args:
            workers.append(int(process.name))
        elif process.name in shard_pids and any(
                str(arg).startswith(f"{base}_") or str(arg).startswith(f"runs/{base.name}_") for arg in args):
            workers.append(int(process.name))
    return sorted(workers)


def step2_progress(base):
    work = base.with_name("step2_v3_01")

    def query(path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    if base.name.endswith(("_train_a", "_train_b")):
        curve = query(base / "loss_curves.json")
        values = curve.get("train", [])
        validation = curve.get("validation", [])
        recent = f" update={values[-1]['step']}/2000" if values else ""
        if validation:
            recent += f" val@{validation[-1]['step']}={validation[-1]['total']:.6f}"
        return recent
    if base.name.endswith("_dev") and base.name != "step2_v3_01_eval_dev":
        if dev := query(work / "development.json"):
            return f" replay_verified={dev['count']}/28"
        completed = [query(path) for path in work.glob("expert_round_*.json")]
        observed = sum(row.get("accepted", 0) for row in completed)
        recorded = {}
        for batch in base.parent.glob("step2_v3_01_demo*/batch_progress.json"):
            for row in query(batch).get("cases", []):
                recorded[row["case"]] = row["status"]
        return f" demo_attempts={len(recorded)} replay_verified={observed}/28"
    if base.name.endswith(("_eval_train", "_eval_dev", "_eval_best")):
        scope = base.name.rsplit("_", 1)[-1]
        den = {"train":10,"dev":28,"best":4}[scope]
        results = {v: {} for v in "ab"}
        for v in results:
            for batch in base.parent.glob(f"step2_v3_01_{scope}_{v}_*/batch_progress.json"):
                for row in query(batch).get("cases", []):
                    results[v][row["case"]] = row.get("policy_result", {}).get("status")
        return "".join(f" {v.upper()}={sum(val in {'success', 'lift_failed', 'timeout'} for val in rows.values())}/{den}"
                       for v, rows in results.items())
    return ""


def progress_suffix(base):
    if base.name in {"step3b_01","step3b_01_stage1","step3b_01_stage2","step3b_01_stage3"}:
        folder = base.parent/"step3b_01"
        found = []
        for name in ("controller","stage1_progress","stage2_progress","stage3_progress"):
            try:
                found.append(json.loads((folder/f"{name}.json").read_text()))
            except (FileNotFoundError,json.JSONDecodeError):
                found.append({})
        state, capture, fit, metrics = found
        stage = state.get("stage")
        if base.name.endswith("stage1"):
            return f" new={capture.get('accepted',0)}/1140 total={capture.get('combined_total',460)}/1600"
        if base.name.endswith("stage2"):
            return f" epoch={fit.get('epoch',0)}/80 updates={fit.get('updates',0)}/{fit.get('expected_updates','?')}"
        if base.name.endswith("stage3"):
            return f" {metrics.get('phase','pending')} levels={metrics.get('completed_levels',0)}/4"
        return f" stage={stage or 1} state={state.get('state','PENDING')}"
    if base.name == "v3_b40_step3a_01":
        results={"train":{},"dev":{}}
        for scope,cases in results.items():
            for file in base.parent.glob(f"{base.name}_b40_last_{scope}_*/batch_progress.json"):
                try:
                    data=json.loads(file.read_text(encoding="utf-8"))
                    for row in data["cases"]:
                        status=(row.get("policy_result") or {}).get("status")
                        if status in {"success","timeout","lift_failed"}:
                            cases[row["case"]]=status
                except (FileNotFoundError,KeyError,json.JSONDecodeError):
                    continue
        return f" closed_loop_train={len(results['train'])}/10 closed_loop_dev={len(results['dev'])}/28"
    if base.name.startswith("step3a_01_"):
        stage = base.name.removeprefix("step3a_01_")
        if stage in {"l1","l2","l3","l4","pilots","substitutes"}:
            progress = base.parent / "step3a_01" / f"{stage}_progress.json"
            try:
                data = json.loads(progress.read_text(encoding="utf-8"))
                return f" replay_verified={data['accepted']}/{data.get('expected',20)}"
            except (FileNotFoundError, json.JSONDecodeError):
                return ""
    if base.name == "v3_b40_step3a_01_fit":
        try:
            with base.with_suffix(".log").open("rb") as stream:
                stream.seek(max(0, stream.seek(0, os.SEEK_END) - 4096))
                epochs = re.findall(rb"(?m)^epoch ([0-9]+/[0-9]+), updates ([0-9]+/[0-9]+)", stream.read())
            if epochs:
                e, u = (v.decode() for v in epochs[-1])
                return f" epoch={e} updates={u}"
        except FileNotFoundError:
            return ""
    if base.name.startswith("step2_v3_01"):
        if details := step2_progress(base):
            return details
    if base.name.startswith("tier1_"):
        suffix = "_cache" if base.name.endswith("_cache") else "_collection" if base.name.endswith("_collection") else ""
        root = base.with_name(base.name.removesuffix(suffix)) if suffix else base
        filename = "cache_progress.json" if suffix == "_cache" else "progress.json"
        path = root / filename
        if path.is_file():
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if suffix == "_cache":
                    return f" cached={row['episodes']}/{row['target']} detected={row['detected']}/{row['frames']}"
                batches=[]
                for job in row["active"]:
                    current=root.parent/job["job"]/"batch_progress.json"
                    try:
                        results=json.loads(current.read_text(encoding="utf-8"))["cases"]
                    except (FileNotFoundError,json.JSONDecodeError):
                        results=[]
                    batches.append(f"gpu{job['gpu']}:{sum(r['status']=='strict_success' for r in results)}/{len(results)}")
                return f" accepted={row['total']}/{sum(row['target'].values())} active={','.join(batches)}"
            except (FileNotFoundError, json.JSONDecodeError):
                pass
    for filename in ("result.json", "batch_progress.json"):
        try:
            summary = json.loads((base / filename).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if summary.get("phase") == "three_view_scale":
            return f" accepted={summary['accepted']}/{summary['configured']} evaluated={len(summary['cases'])}"
        if summary.get("phase") == "overnight_expert_data":
            checked = {row["case"] for row in summary["cases"]}
            line = f" accepted={summary['accepted']}/{summary['configured']} attempts={len(checked)}"
            if filename == "result.json":
                return line
            batch = {}
            for shard in base.parent.glob(f"{base.name}_w*"):
                for child_file in ("result.json", "batch_progress.json"):
                    try:
                        child = json.loads((shard / child_file).read_text(encoding="utf-8"))
                    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
                        continue
                    batch.update({row["case"]: row["status"] for row in child["cases"] if row["case"] not in checked})
                    break
            if batch:
                line += f" current_batch_results={len(batch)} current_batch_strict={sum(value == 'strict_success' for value in batch.values())}"
            return line
        if summary.get("phase") == "overnight_data_audit":
            return f" verified={summary['verified']}/{summary['configured']}"
        if summary.get("phase") == "three_view_seen_time_limit_baseline":
            counts = summary["summary"]["combined"]
            total = counts["episodes"]
            return f" success_10s={counts['success_10s']}/{total} success_25s={counts['success_25s']}/{total}"
        if summary.get("phase") == "three_view_step0":
            expert = summary["expert"]
            return f" expert_aligned={expert['accepted']}/{expert['configured']} policy_episodes={len(summary['cases'])}"
        if summary.get("phase") == "step1_right_wrist_detectors" or (
                base.name.startswith(("step1_detector_", "step1b_detector_")) and isinstance(summary.get("cases"), dict)):
            return f" visible_frames={next(iter(summary['summary'].values()))['frames']}"
        if cases := summary.get("cases"):
            if isinstance(cases, int):
                return f" paired_cases={cases}"
            if summary.get("phase") == "tracking_benchmark":
                return f" recordings={len(cases)}/{summary['configured']}"
            if summary.get("phase") == "tracking_motion":
                return f" recorded={sum(row['status'] == 'recorded' for row in cases)}/{len(cases)}"
            if "offline_feasible" in cases[0] and "status" not in cases[0]:
                feasible = sum(bool(row["offline_feasible"]) for row in cases)
                return f" cases={len(cases)} offline_feasible={feasible}"
            if "status" not in cases[0]:
                return f" cases={len(cases)}"
            counts = Counter(row["status"] for row in cases)
            result = f" cases={len(cases)} strict_success={counts['strict_success']}"
            if (filename == "result.json" and re.fullmatch(r"fit20_closed_loop_[0-9]+", base.name)
                    and not counts["strict_success"]):
                result += " controller_validation=FAILED"
            return result
    if (base / "frozen_cases.json").is_file():
        checked = sum(bool(re.fullmatch(r"[ABC]_[0-9]{3}\.json", path.name)) for path in base.glob("*.json"))
        return f" planned_cases_checked={checked}"
    if match := re.fullmatch(r"(three_view_step0_[a-z0-9]+)_(expert|policy)", base.name):
        work, phase = match.groups()
        definition = base.with_name(work) / "definition.json"
        if definition.is_file():
            jobs = [row for row in json.loads(definition.read_text())["jobs"] if row["phase"] == phase]
            planned = sum(len(json.loads(Path(row["plan"]).read_text())["cases"]) for row in jobs)
            completed = 0
            for row in jobs:
                try:
                    progress = json.loads((base.with_name(row["name"]) / "batch_progress.json").read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                completed += len(progress["cases"])
            return f" cases_reported={completed}/{planned}"
    preview = base / "plans/frozen_cases.json"
    if (base / "worker_pids").is_file() or preview.is_file():
        totals = Counter()
        for shard in base.parent.glob(f"{base.name}_*"):
            for filename in ("result.json", "batch_progress.json"):
                try:
                    receipt = json.loads((shard / filename).read_text(encoding="utf-8"))
                except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
                    continue
                totals.update(row["status"] for row in receipt["cases"])
                break
        selection = base / "plans/selection.json"
        if selection.is_file():
            selected = json.loads(selection.read_text(encoding="utf-8"))["selected_ids"]
            return f" replay_reported={sum(totals.values())}/{len(selected)} strict_success={totals['strict_success']}"
        if preview.is_file():
            try:
                cases = json.loads(preview.read_text(encoding="utf-8"))["cases"]
            except json.JSONDecodeError:
                return ""
            if not totals:
                checked = sum((preview.parent / f"{case['case_id']}.json").is_file() for case in cases)
                return f" planned={checked}/{len(cases)}"
            return f" cases_reported={sum(totals.values())}/{len(cases)} strict_success={totals['strict_success']}"
        if base.name.startswith("dataset_alignment_"):
            return f" replay_reported={sum(totals.values())}/10 strict_success={totals['strict_success']}"
        if base.name.startswith("fit20_closed_loop_"):
            return f" cases_reported={sum(totals.values())}/20 strict_success={totals['strict_success']}"
        if base.name.startswith(("three_view_expert_", "three_view_parts_capture_")) or re.fullmatch(
                r"three_view_(pipeline|generalization|parts)_[0-9]+_eval_[ab]|three_view_scale_[0-9]+_w[0-9]+|"
                r"three_view_time25_[0-9]+_(baseline|candidate|test)_[ab]", base.name):
            configured = sum(len(json.loads(path.read_text())["cases"]) for path in (base / "plans").glob("shard_*.json"))
            return f" cases_reported={sum(totals.values())}/{configured} strict_success={totals['strict_success']}"
        return f" cases_reported={sum(totals.values())}/100 strict_success={totals['strict_success']}"
    if re.fullmatch(r"fit20_[0-9]+|three_view_(pipeline|time25)_[0-9]+_fit_[ab]", base.name):
        try:
            with base.with_suffix(".log").open("rb") as log:
                log.seek(max(0, log.seek(0, os.SEEK_END) - 4096))
                matches = re.findall(rb"(?m)^([0-9]+/[0-9]+) loss=", log.read())
        except FileNotFoundError:
            return ""
        if matches:
            return f" updates={matches[-1].decode()}"
    return ""


def status(base):
    marker = base.with_suffix(".complete")
    replacement = base.with_suffix(".superseded")
    if replacement.is_file():
        return f"{base.name}: COMPLETE SUPERSEDED replacement={replacement.read_text().strip()} marker={marker}"
    if marker.is_file():
        data = json.loads(marker.read_text(encoding="utf-8"))
        line = f"COMPLETE {data['outcome']} exit={data['exit_code']} marker={marker}"
    elif (code := read_exit(base)) is not None:
        line = f"COMPLETE {'PASS' if code == 0 else 'FAILED'} exit={code} marker=pending"
    elif workers := live_workers(base):
        line = f"RUNNING pid={','.join(map(str, workers))}"
    elif base.exists() or base.with_suffix(".log").exists():
        line = "NOT COMPLETE (no live simulator found; check log or queue)"
    else:
        line = "PENDING (not started or unknown run)"
    return f"{base.name}: {line}{progress_suffix(base)}"


def main():
    parser = argparse.ArgumentParser(description="Show each run's explicit background completion status.")
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--record-exit", type=int)
    options.add_argument("--wait-exit", type=int, metavar="SECONDS")
    options.add_argument("--sync-exits", action="store_true")
    parser.add_argument("jobs", nargs="+")
    args = parser.parse_args()
    bases = [run_base(RUNS, job) for job in args.jobs]
    if args.record_exit is not None:
        if len(bases) != 1:
            parser.error("--record-exit requires one job")
        record_complete(bases[0], args.record_exit)
    elif args.sync_exits:
        for base in bases:
            complete_from_exit(base)
    elif args.wait_exit is not None:
        if args.wait_exit <= 0:
            parser.error("--wait-exit must be positive")
        deadline = time.monotonic() + args.wait_exit
        pending = bases.copy()
        while pending:
            for base in pending.copy():
                if base.with_suffix(".complete").is_file() or complete_from_exit(base):
                    print(status(base), flush=True)
                    pending.remove(base)
            if not pending:
                return 0
            if time.monotonic() >= deadline:
                for base in pending:
                    print(status(base), flush=True)
                return 124
            time.sleep(2)
    for base in bases:
        print(status(base), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
