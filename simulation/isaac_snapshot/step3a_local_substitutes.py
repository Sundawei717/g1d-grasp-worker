"""Qualify independent local substitutes for the Tier-2 asset pool."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,as_completed
from multiprocessing import get_context
from threading import Lock
import numpy as np

from background_status import record_complete
from step3a_assets import SUBSTITUTES, prepare
from step3a_suites import (RUNS,WORK,planned_receipts,verify_groups,profile_pair,read,write_json)
from step3a_verify import isolate_from_previous,layout

JOB="step3a_01_substitutes"
PREP="step3a_01_substitute_assets"


def inventory():
    frozen=read(WORK/"l4_base_mesh_fingerprints.json")
    held={x["base_mesh_hash"] for x in frozen["heldout"].values()}
    excluded=held|set(frozen["training_assets_excluded"].values())
    source=read(WORK/"asset_inventory.json")
    held.update(a.get("raw_visual_base_mesh_hash",a["base_mesh_hash"])
                for a in source["l4_heldout"].values())
    added={}
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context("spawn")) as pool:
        for task in as_completed([pool.submit(prepare,name) for name in SUBSTITUTES]):
            item=task.result()
            hashes={item["base_mesh_hash"],item.get("raw_visual_base_mesh_hash",item["base_mesh_hash"])}
            if hashes&(excluded|held):
                print(item["name"],"excluded because the base mesh is already used",flush=True)
                continue
            excluded.update(hashes)
            added[item["name"]]=item
    write_json(WORK/"asset_substitutes.json",{"assets":added,"heldout_overlap":0})
    record_complete(RUNS/PREP,0)
    return added


def main():
    items=inventory()
    if not items:
        raise RuntimeError("No independent local substitute mesh was found")
    prior,_,ids,seeds=isolate_from_previous()
    frozen=read(WORK/"frozen.json")["cases"]
    prior.update(layout(r) for r in frozen)
    ids.update(r["scene_id"] for r in frozen)
    seeds.update(r["seed"] for r in frozen)
    verified=defaultdict(list)
    audit=[]
    update=Lock()
    names=sorted(items)

    def plan_asset(name,planner):
        number=names.index(name)*4
        accepted={}
        for attempt in range(6):
            pending=[i for i in (0,1) if i not in accepted]
            if not pending:
                return
            tasks={planner.submit(profile_pair,items[name], (number//2+i)+(attempt+1)*13, 0,
                                  2411000+100*number+7*attempt+i):i for i in pending}
            valid={}
            for f in as_completed(tasks):
                i=tasks[f]
                docs=f.result()
                if not docs:
                    continue
                footprints={layout(x["case"]) for x in docs}
                with update:
                    if footprints & prior:
                        continue
                    prior.update(footprints)
                slots=[{"scene_id":f"tier2_local_{name}_{i}_{p}",
                        "seed":2411000+number+i*2+p} for p in (0,1)]
                assert not {r["scene_id"] for r in slots}&ids
                assert not {r["seed"] for r in slots}&seeds
                pairs=planned_receipts(docs,slots,9500+attempt*40+number+i*2,
                                       f"tier2_local_{name}_{i}_r{attempt}")
                valid[i]=pairs
            if not valid:
                print(name,"no pair solved",attempt,flush=True)
                continue
            result=verify_groups(list(valid.values()),f"pilots_local_{name}_r{attempt}")
            grouped=defaultdict(list)
            slot={r["scene_id"]:i for i,rows in valid.items() for r in rows}
            for r in result["cases"]:
                grouped[slot[r["scene_id"]]].append(r)
            accepted.update(grouped)
            with update:
                audit.extend(result["attempts"])
                verified[name]=[r for rs in accepted.values() for r in rs]
                rows=[r for rs in verified.values() for r in rs]
                write_json(WORK/"substitutes_progress.json",{"accepted":len(rows),"expected":4*len(items),
                           "attempts":audit,"cases":rows,"ready":[k for k,v in verified.items() if len(v)==4]})
            print(name,"accepted",len(verified[name]),"/4",flush=True)

    with ProcessPoolExecutor(max_workers=5,mp_context=get_context("spawn")) as planner:
        with ThreadPoolExecutor(max_workers=5) as pool:
            for task in as_completed([pool.submit(plan_asset,name,planner) for name in names]):
                task.result()
    if any(len(verified[name])!=4 for name in names):
        raise RuntimeError("One or more substitute candidates lack two full accepted pairs")


if __name__=="__main__":
    code=1
    try:
        main()
        code=0
    finally:
        if not (RUNS/f"{PREP}.complete").is_file():
            record_complete(RUNS/PREP,code)
        record_complete(RUNS/JOB,code)
