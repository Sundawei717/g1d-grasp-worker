import argparse
from collections import Counter
import json
from pathlib import Path
from shutil import copyfile

import h5py

from episode_store import verify_three_view
from three_view_collection import ROLES
from three_view_detector import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root, output):
    runs = root / "runs"
    source = runs / "overnight_expert_01"
    milestone = read(source / "milestone_250.json")
    existing = read(runs / "three_view_expert_03/result.json")["episodes"]
    selected = milestone["episodes"]
    if len(selected) != 250 or len({row["case"] for row in selected}) != 250:
        raise ValueError("Expected the verified 250-episode expert milestone")
    keys = {row["case"] for row in selected}
    reused = [row for row in existing if row["case"] in keys]
    finished = {row["case"] for row in reused}
    selected_rows = {row["case"]: row for row in selected}
    for row in reused:
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError("Reused three-view episode changed")
        with h5py.File(row["path"], "r") as h5:
            verify_three_view(h5)
    batches = {}
    for role, query in ROLES.items():
        name = f"shard_{role}.json"
        template = read(runs / "train_bootstrap_01/plans" / name)
        rows = []
        for key in sorted(keys - finished):
            path = source / "plans" / f"{key}.json"
            if not path.is_file():
                path = runs / "train_bootstrap_01/plans" / f"{key}.json"
            prepared = read(path)
            case = prepared["case"]
            if case["asset_role"] != role:
                continue
            reference = selected_rows[key]
            if (not prepared["feasible"] or case["scene_family"] != reference["scene_family"]
                    or sha256(reference["path"]) != reference["sha256"]):
                raise ValueError(f"Source layout or expert recording changed: {key}")
            rows.append(({**case, "plan": path.name, "offline_feasible": True, "replay_h5": {
                "path": reference["path"], "sha256": reference["sha256"]}}, path))
        batches[role] = (template, query, [rows[i:i+12] for i in range(0, len(rows), 12)])
    batch_count = max(len(value[2]) for value in batches.values())
    for index in range(batch_count):
        wave = output.with_name(f"{output.name}_w{index+1:03d}")
        plans = wave / "plans"
        plans.mkdir(parents=True)
        write_json(plans / "reused.json", {"episodes": [], "cases": []})
        for role, (template, query, chunks) in batches.items():
            if index >= len(chunks):
                continue
            selected_chunk = chunks[index]
            for row, path in selected_chunk:
                copyfile(path, plans / row["plan"])
            write_json(plans / f"shard_{role}.json", {**template, "three_view": True, "target_query": query,
                                                    "cases": [row for row, _ in selected_chunk]})
    write_json(output / "expanded_manifest.json", {"source_sha256": sha256(source / "milestone_250.json"),
                                                  "configured": 250, "reused": reused,
                                                  "waves": batch_count, "groups": Counter(row["group"] for row in selected),
                                                  "pending": len(keys - finished)})
    print(f"Frozen {len(reused)} reusable and {len(keys - finished)} replay episodes in {batch_count} waves", flush=True)


def gather(output, stage):
    manifest = read(output / "expanded_manifest.json")
    episodes = list(manifest["reused"])
    cases = [{"case": row["case"], "status": "strict_success"} for row in episodes]
    waves = [path for path in output.parent.glob(f"{output.name}_w*") if path.name.rsplit("_w", 1)[-1].isdigit()]
    for path in sorted(wave / "result.json" for wave in waves if (wave / "result.json").is_file()):
        result = read(path)
        episodes.extend(result["episodes"])
        cases.extend(result["cases"])
    if len({row["case"] for row in episodes}) != len(episodes):
        raise ValueError("An expanded episode entered the data split twice")
    result = {"phase": "three_view_scale", "configured": manifest["configured"], "strict_success": len(episodes),
              "accepted": len(episodes), "passed": len(episodes) == manifest["configured"], "cases": cases,
              "episodes": episodes, "groups": Counter(row["group"] for row in episodes)}
    write_json(output / ("result.json" if stage == "finish" else "batch_progress.json"), result)
    print(f"Expanded three-view data: {len(episodes)}/{manifest['configured']}", flush=True)
    return 0 if stage == "progress" or result["passed"] else 1


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("command", choices=("prepare", "progress", "finish"))
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.command == "prepare":
        prepare(Path(__file__).resolve().parent, args.output)
    else:
        raise SystemExit(gather(args.output, args.command))
