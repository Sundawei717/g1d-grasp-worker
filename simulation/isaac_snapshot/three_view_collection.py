import argparse
import json
from pathlib import Path
from shutil import copyfile

import h5py

from episode_store import verify_three_view
from prepare_grouped import write_json
from school_planner import sha256


ROLES = {"carton": "box", "can": "can"}


def prepare(bootstrap, output, limit, reuse=None):
    source = json.loads((bootstrap / "result.json").read_text())
    records = {row["case"]: row for row in source["episodes"]}
    output.mkdir()
    reused = json.loads(reuse.read_text()) if reuse else {"episodes": [], "cases": []}
    reused_ids = {row["case"] for row in reused["episodes"]}
    write_json(output / "reused.json", reused)
    for role, query in ROLES.items():
        file = f"shard_{role}.json"
        manifest = json.loads((bootstrap / "plans" / file).read_text())
        selected = [row for row in manifest["cases"] if row["case_id"] in records and row["case_id"] not in reused_ids]
        if limit:
            selected = selected[:limit]
        cases = []
        for case in selected:
            record = records[case["case_id"]]
            cases.append({**case, "replay_h5": {"path": record["path"], "sha256": record["sha256"]}})
            copyfile(bootstrap / "plans" / case["plan"], output / case["plan"])
        if not cases:
            raise ValueError(f"Missing accepted source trajectories for {role}")
        write_json(output / file, {**manifest, "cases": cases, "three_view": True, "target_query": query})


def finish(output):
    reused = json.loads((output / "plans/reused.json").read_text())
    episodes, attempts = list(reused["episodes"]), list(reused["cases"])
    configured = len(episodes)
    for row in episodes:
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError(f"Reused synchronized episode changed: {row['case']}")
    for plan in sorted((output / "plans").glob("shard_*.json")):
        role = plan.stem.removeprefix("shard_")
        manifest = json.loads(plan.read_text())
        configured += len(manifest["cases"])
        directory = output.with_name(f"{output.name}_{role}")
        receipt = json.loads((directory / "result.json").read_text())
        attempts.extend(receipt["cases"])
        for row in receipt["cases"]:
            if row["status"] != "strict_success":
                continue
            path = directory / row["case"] / "episode.h5"
            if not path.is_file():
                path = path.parent / "expert_rerun" / "episode.h5"
            with h5py.File(path, "r") as h5:
                frames = verify_three_view(h5)
                case = json.loads(h5["meta/case_json"].asstr()[()])
            diagnostic = json.loads((path.parent / "expert_diagnostic.json").read_text())
            episodes.append({"case": row["case"], "group": row["group"], "scene_family": case["scene_family"],
                             "asset_family": case["asset_family"], "split": case["split"],
                             "source": diagnostic.get("replay_alignment", {}).get("source"),
                             "path": str(path.resolve()), "sha256": sha256(path), "frames_per_camera": frames,
                             "part": case["part"],
                             "target_query": manifest["target_query"], "replay_alignment": diagnostic.get("replay_alignment")})
    passed = configured == len(episodes)
    result = {"phase": "three_view_collection", "configured": configured, "passed": passed,
              "strict_success": len(episodes), "cases": attempts, "episodes": episodes}
    write_json(output / "result.json", result)
    print(f"Synchronized three-view expert data: {len(episodes)}/{configured}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    commands = cli.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("bootstrap", type=Path)
    prep.add_argument("output", type=Path)
    prep.add_argument("--limit", type=int, default=0)
    prep.add_argument("--reuse", type=Path)
    commands.add_parser("finish").add_argument("output", type=Path)
    args = cli.parse_args()
    if args.command == "prepare":
        prepare(args.bootstrap, args.output, args.limit, args.reuse)
    else:
        raise SystemExit(finish(args.output))
