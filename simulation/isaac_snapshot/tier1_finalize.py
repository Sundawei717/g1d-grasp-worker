import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
from background_status import record_complete
from three_view_detector import sha256, write_json
from tier1_collect import QUOTAS, ROOT, RUNS, SIM, gpu_manifest, read, verify_receipt
from v3_detector import V3WristDetector, cache_key, detector_config, save_episode_cache, verify_episode_cache


def cache_worker(work, index):
    rows = read(work/"cache_batches"/f"{index}.json")
    detector = V3WristDetector("cuda:0")
    receipts = work/"cache_receipts"
    for row in rows:
        path = receipts / f"{row['case']}.json"
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError(f"Changed archive RGB for {row['case']}")
        with h5py.File(row["path"], "r") as data:
            query = json.loads(data["meta/initial_prompt_json"].asstr()[()])["target_query"]
            ids = data["obs/right_wrist/frame_id"].asstr()[:]
            if path.exists():
                try:
                    previous = read(path)
                    if previous["target_query"] != query:
                        raise ValueError("Cached target prompt was changed")
                    verify_episode_cache(row, previous, ids,detector.config)
                    continue
                except (KeyError, ValueError, OSError):
                    pass
            rgb = data["obs/right_wrist/rgb"]
            frames = [{"frame_id": frame_id, "control_step_id": i, **detector.detect(rgb[i], query)}
                      for i, frame_id in enumerate(ids)]
        entry = save_episode_cache(work/"right_wrist_cache", detector, row, query, frames)
        verify_episode_cache(row, entry, ids, detector.config)
        write_json(path, entry)
        print(row["case"], entry["detected"], "/", entry["frames"], flush=True)


def audit(work, collection):
    rows = collection["episodes"]
    if Counter(row["bucket"] for row in rows) != Counter(QUOTAS):
        raise ValueError("Missing full Tier-1 quota")
    for key in ("scene_id", "episode_id", "scene_signature", "sha256"):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate admitted {key}")
    groups = Counter(row["pair_group_id"] for row in rows if row["pair_group_id"])
    if any(count!=2 for count in groups.values()):
        raise ValueError("Tier-1 contains an incomplete part pair")
    while not (work/"development.json").exists():
        if (work/"planning_exhausted.json").exists():
            raise RuntimeError("Independent development quota not met")
        time.sleep(10)
    development = read(work/"development.json")["cases"]
    reserved = read(work/"final_test_reservation.json")
    if (len(development) != 28 or {row["scene_signature"] for row in rows} & (
            {row["scene_signature"] for row in development} | set(reserved["existing_frozen_signatures"]))
            or {row["asset_family"] for row in rows} & set(reserved["heldout_asset_families"])
            or set(reserved["training_base_mesh_signatures"].values()) & set(reserved["reserved_base_mesh_signatures"].values())
            or not all(any(c["bucket"]==bucket and c["fixtures"] for c in development)
                       for bucket in ("mug","basket","carton"))):
        raise ValueError("Development or test layout leakage")
    for row in rows:
        verify_receipt(row)
    write_json(work/"audit.json", {"passed": True, "verified":len(rows), "paired_groups":len(groups),
                                  "development_scenes":len(development), "frozen_test_slots":reserved["count"],
                                  "artifact_sha256":sha256(work/"collection.json")})


def print_final_inventory(work, collection, cache):
    rows=collection["episodes"]
    histories=defaultdict(dict)
    for p in sorted((work/"batches").glob("*/launch.json")):
        run=read(p)
        for row in run["rows"]:
            if not row.get("source_case"):
                continue
            path=RUNS/run["job"]/row["case_id"]/"expert_diagnostic.json"
            if not path.exists():
                continue
            diag=read(path)
            roll=diag.get("policy_rollout",{})
            trial=next(iter(roll.get("trials",[])),{})
            histories[row["bucket"]][row["case_id"]]=("accepted" if (work/"accepted"/f"{row['case_id']}.json").exists()
                    else trial.get("failure_category") or roll.get("status") or "preflight_rejected")
    rerecord={}
    for bucket,results in histories.items():
        counts=Counter(results.values())
        excluded=len(results)-counts["accepted"]
        rerecord[bucket]={"attempted":len(results),"accepted":counts["accepted"],"excluded":excluded,
                           "excluded_rate":excluded/len(results),
                           "failure_reasons":{k:v for k,v in counts.items() if k!="accepted"}}
    profile=work/"model_forward.json"
    report={"passed":True,"collection_count":len(rows),"counts":collection["counts"],
            "storage_bytes":sum(r["bytes"] for r in rows),"paired_groups":read(work/"audit.json")["paired_groups"],
            "obstacle_counts":Counter(r["bucket"] for r in rows if r["group"]=="B_obstacle"),
            "obstacle_assets":{bucket:sorted({p for r in rows if r["bucket"]==bucket for p in r["fixture_assets"]})
                               for bucket in ("mug","basket","carton")},
            "rerecord":rerecord,"cache_frames":cache["frames"],"cache_detected":cache["detected"],
            "cache_detected_rate":cache["detected"]/cache["frames"],"cache_by_asset":cache["by_asset"],
            "development_scenes":len(read(work/"development.json")["cases"]),
            "frozen_test_reservation":str(work/"final_test_reservation.json"),
            "profile":read(profile) if profile.is_file() else None,
            "training_started":False}
    write_json(work/"final_inventory.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="profile"},ensure_ascii=False),flush=True)


def finalize(work):
    from three_view_train import make_split

    collection = read(work/"collection.json")
    audit(work,collection)
    record_complete(RUNS/(work.name+"_audit"),0)
    make_split(work/"collection.json", work/"training_split.json")
    config = detector_config()
    for name in ("cache_receipts","cache_batches"):
        (work/name).mkdir(exist_ok=True)
    rows = collection["episodes"]
    active = []
    while not active:
        active = [g for g in gpu_manifest() if g["free_mib"] >= 8192]
        if not active:
            time.sleep(30)
    def launch(index,gpu,attempt):
        name = f"{work.name}_cache_g{gpu['gpu']}"
        env = {**os.environ, "CUDA_DEVICE_ORDER":"PCI_BUS_ID","CUDA_VISIBLE_DEVICES":gpu["uuid"],
               "PYTHONPATH":os.pathsep.join((str(ROOT/"cache/three_view_detector/deps"),os.environ.get("PYTHONPATH","")))}
        with (RUNS/f"{name}.log").open("a") as log:
            p = subprocess.Popen([str(SIM),"-Bu",str(ROOT/"tier1_finalize.py"),str(work),"--worker",str(index)],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,env=env)
        print(name,"PID",p.pid,"GPU",gpu["gpu"],"attempt",attempt,flush=True)
        return {"gpu":gpu,"process":p,"attempt":attempt}

    jobs = {}
    for index,gpu in enumerate(active):
        if rows[index::len(active)]:
            write_json(work/"cache_batches"/f"{index}.json", rows[index::len(active)])
            jobs[index]=launch(index,gpu,1)
    while True:
        busy={job["gpu"]["gpu"] for job in jobs.values() if job["process"].poll() is None}
        for index,job in jobs.items():
            code=job["process"].poll()
            if not code:
                continue
            if job["attempt"]>=2:
                raise RuntimeError(f"Cache shard {index} failed twice, exit {code}")
            retry=next((g for g in gpu_manifest() if g["gpu"] not in busy and g["free_mib"]>=8192),None)
            if retry:
                jobs[index]=launch(index,retry,2)
                busy.add(retry["gpu"])
        receipts=[read(path) for path in (work/"cache_receipts").glob("*.json")]
        write_json(work/"cache_progress.json",{"episodes":len(receipts),"target":len(rows),
                                               "frames":sum(r["frames"] for r in receipts),
                                               "detected":sum(r["detected"] for r in receipts),
                                               "workers":[{"gpu":j["gpu"]["gpu"],"pid":j["process"].pid}
                                                          for j in jobs.values() if j["process"].poll() is None]})
        if all(job["process"].poll()==0 for job in jobs.values()):
            break
        time.sleep(10)
    indexed={r["case"]:r for r in receipts}
    by_asset=defaultdict(Counter)
    for row in rows:
        entry=indexed[row["case"]]
        with h5py.File(row["path"],"r") as data:
            ids=data["obs/right_wrist/frame_id"].asstr()[:]
            verify_episode_cache(row,entry,ids,config)
        by_asset[row["bucket"]].update({"frames": entry["frames"], "detected":entry["detected"]})
    cache={"passed":True,"config":config,"config_sha256":cache_key(config),
              "collection_sha256":sha256(work/"collection.json"),"episodes":receipts,"by_asset":by_asset,
              "frames":sum(r["frames"] for r in receipts),"detected":sum(r["detected"] for r in receipts)}
    write_json(work/"detection_cache.json",cache)
    print_final_inventory(work,collection,cache)
    record_complete(RUNS/(work.name+"_cache"),0)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    p.add_argument("--worker",type=int)
    a=p.parse_args()
    if a.worker is not None:
        cache_worker(a.output.resolve(),a.worker)
    else:
        try:
            finalize(a.output.resolve())
        except Exception:
            record_complete(RUNS/(a.output.name+"_cache"),1)
            raise
