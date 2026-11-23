import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from shutil import copyfile
from time import sleep

import numpy as np
import h5py

from three_view_collection import ROLES
from three_view_detector import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def verify_policy(binding):
    config = read(binding["receipt"])
    if (not config["complete_optimization"] or binding["path"] != config["checkpoint"]
            or sha256(binding["path"]) != binding["sha256"] or config["checkpoint_sha256"] != binding["sha256"]):
        raise ValueError("Checkpoint is unavailable or changed")
    return config["config"]


def freeze(root, split, output, per_group=4):
    source = root / "runs/stage3_splits_01"
    data = read(source / "result.json")["cases"]
    train = read(split)["episodes"]
    if {row["scene_family"] for row in data} & {row["scene_family"] for row in train}:
        raise ValueError("Test layout family shares training or validation data")
    selected = []
    for group in ("A_target", "B_obstacle", "C_start"):
        rows = [row for row in data if row["group"] == group and row["offline_feasible"]]
        families = list(dict.fromkeys(row["scene_family"] for row in rows))
        for family in families:
            selected.extend(row for row in rows if row["scene_family"] == family)
            if sum(row["group"] == group for row in selected) >= per_group:
                break
    output.mkdir()
    for row in selected:
        copyfile(source / row["plan"], output / row["plan"])
    write_json(output / "frozen.json", {"cases": selected, "scope": "seen assets in independent layouts"})


def freeze_extra(root, split, output):
    physical = root / "runs/three_view_extra_physics_01"
    receipt = read(physical / "result.json")
    succeeded = {row["case"] for row in receipt["cases"] if row["status"] == "strict_success"}
    trained = read(split)["episodes"]
    train_families = {row["asset_family"] for row in trained}
    rows, roles = [], {}
    output.mkdir()
    for role, query in (("mug", "mug"), ("unseen_can", "can")):
        directory = physical / role
        manifest = read(directory / "manifest.json")
        eligible = [row for row in manifest["cases"] if row["case_id"] in succeeded
                    and row["asset_family"] not in train_families]
        if eligible:
            roles[role] = query
        for row in eligible:
            copyfile(directory / row["plan"], output / row["plan"])
            rows.append(row)
    if not rows:
        raise ValueError("No unseen asset passed the independent expert trials")
    write_json(output / "frozen.json", {"cases": rows, "roles": roles, "scope": "unseen asset generalization"})


def freeze_parts(root, split, output):
    source_runs = sorted((root / "runs").glob("three_view_parts_capture_[0-9][0-9]"))
    for pending in source_runs:
        if not pending.is_dir() or pending.with_suffix(".complete").is_file():
            continue
        print(f"Waiting for {pending.name}.complete", flush=True)
        while not pending.with_suffix(".complete").is_file():
            sleep(10)
    trained = {row["scene_family"] for row in read(split)["episodes"]}
    rows, roles, prompts, common, families = [], {}, {}, {}, set()
    output.mkdir()
    for filename in sorted((root / "runs").glob("three_view_parts_capture_[0-9][0-9]/result.json"), reverse=True):
        source = filename.parent
        entries = {row["case"]: row for row in read(filename)["episodes"] if row["scene_family"] not in trained | families}
        pairs = Counter(row["scene_family"] for row in entries.values())
        for role, query in ROLES.items():
            shard = source / "plans" / f"shard_{role}.json"
            if not shard.is_file():
                continue
            for row in read(shard)["cases"]:
                key = row["case_id"]
                if key not in entries or pairs[row["scene_family"]] != 2:
                    continue
                with h5py.File(entries[key]["path"], "r") as h5:
                    prompt = json.loads(h5["meta/initial_prompt_json"].asstr()[()])
                bbox = common.setdefault(row["scene_family"], prompt["bbox_0"])
                prompts[key] = {"bbox_0": bbox, "affordance_0": prompt["affordance_0"]}
                copyfile(shard.parent / row["plan"], output / row["plan"])
                rows.append(row)
                roles[role] = query
        families.update(common)
    if not rows:
        raise ValueError("Neither target has a physically verified pair of grasp parts in the same layout")
    write_json(output / "frozen.json", {"cases": rows, "roles": roles, "head_prompts": prompts,
                                       "scope": "matched part selection on the same asset and layout"})


def prepare_v3_dev(root, frozen, binding, cfg, detector, output):
    from dataclasses import asdict
    from step1_mug_pairs import template
    from unified_grasp import EPISODE_STEPS, LiftGateConfig

    pilot = read(frozen / "development.json")
    groups = defaultdict(list)
    for row in pilot["cases"]:
        fixtures = tuple(sorted((f["name"], f["usd"]) for f in row["fixtures"]))
        groups[row["asset_role"], row["target_usd"], fixtures].append(row)
    fixed = {}
    if cfg["variant"] == "B":
        reference = output.parent.with_name(output.parent.name.removesuffix("_b") + "_a")
        fixed = {row["case"]: row["task"] for row in read(reference / "result.json")["cases"]}
    output.mkdir()
    config = read(root / "env_resolved.json")
    for index, ((role, usd, fixtures), rows) in enumerate(sorted(groups.items())):
        name = f"{role}_{index:02}"
        for row in rows:
            copyfile(row["plan"], output / f"{row['case_id']}.json")
        prompts = {r["case_id"]: {name: fixed[r["case_id"]][name] for name in ("bbox_0","affordance_0")}
                   for r in rows if fixed and fixed[r["case_id"]]}
        cases = [{**r, "plan": f"{r['case_id']}.json", "offline_feasible": True} for r in rows]
        write_json(output / f"config_{name}.json", dict(config,target_usd=usd))
        write_json(output / f"shard_{name}.json", {**template(), "phase":"three_view_closed_loop",
                   "asset":usd,"asset_role":role,"fixture_assets":dict(fixtures),"cases":cases,"policy":binding,
                   "target_query": {"mug":"mug","basket":"basket","carton":"box"}[role], "three_view":True,
                   "detector":detector,"total_steps":EPISODE_STEPS,"initial_head_prompts":prompts,
                   "unified_lift_gate":asdict(LiftGateConfig())})


def prepare(root, frozen, fit, detector_file, steps, output):
    receipt = fit / "result.json"
    result = read(receipt)
    binding = {"receipt": str(receipt.resolve()), "path": result["checkpoint"], "sha256": result["checkpoint_sha256"]}
    cfg = verify_policy(binding)
    if cfg.get("architecture") == "v3_final":
        from unified_grasp import EPISODE_STEPS
        from v3_detector import cache_key, detector_config

        if steps != EPISODE_STEPS:
            raise ValueError("V3-Final evaluates exactly 300 control steps")
        detector = detector_config() if cfg["variant"] == "B" else None
        if detector and cfg["detector_config_sha256"] != cache_key(detector):
            raise ValueError("Live Swin-T detector differs from training")
        prepare_v3_dev(root, frozen, binding, cfg, detector, output)
        return
    detector = read(detector_file) if cfg["variant"] == "B" else None
    if detector and cfg["detector_config_sha256"] != sha256(detector_file):
        raise ValueError("Live detector differs from training")
    output.mkdir()
    data = read(frozen / "frozen.json")
    rows = data["cases"]
    for role, query in data.get("roles", ROLES).items():
        file = f"shard_{role}.json"
        if role in ROLES:
            template = read(root / "runs/train_bootstrap_01/plans" / file)
        else:
            physical = root / "runs/three_view_extra_physics_01" / role
            template = read(physical / "manifest.json")
            copyfile(physical / "config.json", output / f"config_{role}.json")
        cases = [row for row in rows if row["asset_role"] == role]
        if not cases:
            raise ValueError(f"No frozen test scenes for {role}")
        for row in cases:
            copyfile(frozen / row["plan"], output / row["plan"])
        prompts = {row["case_id"]: data["head_prompts"][row["case_id"]]
                   for row in cases if "head_prompts" in data}
        if cfg["variant"] == "B" and len(prompts) < len(cases):
            first_run = output.parent.with_name(output.parent.name.removesuffix("_b") + "_a")
            reference = {row["case"]: row["task"] for row in read(first_run / "result.json")["cases"]}
            for row in cases:
                task = reference[row["case_id"]]
                prompts[row["case_id"]] = {key: task[key] for key in ("bbox_0", "affordance_0")}
        write_json(output / file, {**template, "phase": "three_view_closed_loop", "three_view": True,
                                   "target_query": query, "cases": cases, "policy": binding,
                                   "detector": detector, "total_steps": steps, "initial_head_prompts": prompts, "lift_model": {
                                       "urdf": "/data/sundawei/vla_rsi_g1d/assets/Assets/Robots/g1d/g1d_dex1.urdf",
                                       "evidence": str(root / "runs/lab_school_03/result.json")}})


def finish(output):
    rows = []
    for shard in sorted((output / "plans").glob("shard_*.json")):
        suffix = shard.stem.removeprefix("shard_")
        name = f"{output.name}_{suffix}"
        root = output.with_name(name)
        cases = read(root / "result.json")["cases"]
        manifest = read(shard)
        role = manifest.get("asset_role", suffix)
        specs = {case["case_id"]: case for case in manifest["cases"]}
        if set(specs) != {row["case"] for row in cases}:
            raise ValueError(f"Incomplete physical evaluations in {root}")
        for row in cases:
            diagnostic = read(root / row["case"] / "expert_diagnostic.json")
            rollout = diagnostic.get("policy_rollout", {})
            timings = rollout.get("decisions", [])
            status = rollout.get("status", "infrastructure_fail")
            limit = manifest["total_steps"]
            if status == "timeout" and rollout["steps"] != limit:
                raise ValueError(f"Timeout occurred before the shared simulation deadline: {row['case']}")
            if status == "success" and (rollout["steps"] > limit or not rollout["trials"][-1]["success"]):
                raise ValueError(f"Missing verified grip and lift evidence: {row['case']}")
            if status in {"timeout", "success", "lift_failed"}:
                if not timings or any(a["image_time_s"] >= b["image_time_s"] for a, b in zip(timings, timings[1:])):
                    raise ValueError(f"Camera frames stopped advancing: {row['case']}")
                if any(bool(manifest["detector"]) != (item["boxes"] is not None) for item in timings):
                    raise ValueError(f"Wrist detector input differed from the checkpoint: {row['case']}")
            part = specs[row["case"]]["part"]
            group = "unseen_" + manifest["target_query"] if role not in ROLES else row["group"]
            if part.startswith(("top_", "middle_")):
                group = part
            wrists = ("right_wrist",) if manifest.get("detector", {}) and "detector_weights" in manifest["detector"] else ("left_wrist", "right_wrist")
            rows.append({"case": row["case"], "group": group, "role": role, "shard": name,
                         "part": part,
                         "outcome": status, "part_hit": rollout.get("part_hit"),
                         "steps": rollout.get("steps"), "wall_s": rollout.get("wall_s"),
                         "contact_record": diagnostic["forbidden_contact_pairs"], "task": diagnostic.get("task"),
                         "wrist_missing": {wrist: sum(1 for item in timings if item["boxes"] and item["boxes"][i]["missing"])
                                           for i, wrist in enumerate(wrists)},
                         "timings": timings})
    valid = all(row["outcome"] in {"success", "timeout", "lift_failed"} for row in rows)
    counts = Counter(row["outcome"] for row in rows)
    write_json(output / "result.json", {"phase": "three_view_eval", "passed": valid, "cases": rows,
                                       "configured": len(rows), "strict_success": counts["success"], "counts": counts})
    print(dict(counts), flush=True)
    return 0 if valid else 1


def compare(variants, output):
    tables, same = {}, None
    fixed_prompts = None
    training_signature = None
    for variant, run in zip(("A", "B"), variants):
        receipt = read(run / "result.json")
        if not receipt["passed"]:
            raise ValueError(f"Incomplete evaluation for {variant}")
        scenes = {}
        for path in sorted((run / "plans").glob("shard_*.json")):
            shard = read(path)
            fit = read(shard["policy"]["receipt"])
            cfg = fit["config"]
            signature = (cfg["manifest_sha256"], cfg["seed"], cfg["effective_batch"], fit["updates"],
                         fit["sampler_sha256"], cfg.get("architecture"), cfg.get("freeze_bn"), cfg.get("weights_sha256"))
            if training_signature is not None and signature != training_signature:
                raise ValueError("Comparison models used different training data or draws")
            training_signature = signature
            if cfg["variant"] != variant:
                raise ValueError("Physical test received the wrong A/B checkpoint")
            scenes.update({row["case_id"]: sha256(run / "plans" / row["plan"]) for row in shard["cases"]})
        if same is not None and same != scenes:
            raise ValueError("Paired scene plans differ")
        same = scenes
        cases = receipt["cases"]
        if {row["case"] for row in cases} != scenes.keys():
            raise ValueError("Incomplete outcome identities")
        prompts = {row["case"]: tuple(row["task"][key] for key in ("bbox_0", "affordance_0", "target_query", "arm_side"))
                   for row in cases}
        if fixed_prompts is not None and fixed_prompts != prompts:
            raise ValueError("Paired tests used different head prompts")
        fixed_prompts = prompts
        groups = {}
        for group in sorted({row["group"] for row in cases}):
            current = [row for row in cases if row["group"] == group]
            groups[group] = {"cases": len(current), "success": sum(row["outcome"] == "success" for row in current),
                             "timeout": sum(row["outcome"] == "timeout" for row in current),
                             "part_hit": sum(bool(row["part_hit"]) for row in current)}
        all_decisions = [step for row in cases for step in row["timings"]]
        latency = {key: dict(zip(("p50", "p95"), np.percentile([row[key] for row in all_decisions], [50, 95]).tolist()))
                   for key in ("capture_ms", "detector_ms", "inference_ms", "to_command_ms")}
        tables[variant] = {"run": str(run), "counts": receipt["counts"], "groups": groups, "latency_ms": latency,
                           "wall_s": sum(row["wall_s"] for row in cases),
                           "wall_replans_hz": len(all_decisions) / sum(row["wall_s"] for row in cases)}
    write_json(output / "result.json", {"phase": "three_view_ab", "passed": True, "cases": len(same), "variants": tables})


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cmds = cli.add_subparsers(dest="command", required=True)
    fr = cmds.add_parser("freeze")
    extra = cmds.add_parser("freeze-extra")
    parts = cmds.add_parser("freeze-parts")
    prep = cmds.add_parser("prepare")
    fin = cmds.add_parser("finish")
    comp = cmds.add_parser("compare")
    for parser, fields in [(fr, ("root", "split", "output")), (extra, ("root", "split", "output")),
                           (parts, ("root", "split", "output")),
                           (prep, ("root", "frozen", "fit", "detector", "output")),
                           (fin, ("output",)), (comp, ("a", "b", "output"))]:
        for key in fields:
            parser.add_argument(key, type=Path)
    prep.add_argument("--steps", type=int, default=200)
    fr.add_argument("--per-group", type=int, default=4)
    args = cli.parse_args()
    if args.command == "freeze":
        freeze(args.root, args.split, args.output, args.per_group)
    if args.command == "freeze-extra":
        freeze_extra(args.root, args.split, args.output)
    if args.command == "freeze-parts":
        freeze_parts(args.root, args.split, args.output)
    if args.command == "prepare":
        prepare(args.root, args.frozen, args.fit, args.detector, args.steps, args.output)
    if args.command == "finish":
        raise SystemExit(finish(args.output))
    if args.command == "compare":
        compare((args.a, args.b), args.output)
