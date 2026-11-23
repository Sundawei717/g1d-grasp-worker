"""Independently admit two expert pairs per new Tier-2 asset."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,as_completed
from multiprocessing import get_context
from threading import Lock

from background_status import record_complete
from step3a_suites import (RUNS,WORK,ID_BASE,assignments,existing_plans,layout,planned_receipts,
                          previous_layouts,profile_pair,read,verify_groups,write_json)

JOB="step3a_01_pilots"


def main():
    assets=read(WORK/"asset_inventory.json")["tier2_proposed"]
    assigned=assignments("pilots",assets)
    used,old_ids,old_seeds=previous_layouts()
    assert not old_ids & {r["scene_id"] for _,_,slots,_ in assigned for r in slots}
    assert not old_seeds & {r["seed"] for _,_,slots,_ in assigned for r in slots}
    owner_slots={r["scene_id"]:key for key,_,slots,_ in assigned for r in slots}
    verified=defaultdict(list)
    audit=[]
    legacy=sorted(WORK.glob("pilots_r[0-9]*_result.json"))
    for p in legacy:
        d=read(p)
        audit.extend(d["attempts"])
        for row in d["cases"]:
            verified[owner_slots[row["scene_id"]]].append(row)
    first=max((int(f.name.split("_")[1][1:]) for f in legacy),default=-1)+1
    existing={attempt:existing_plans("pilots",attempt) for attempt in range(first,13)}
    assigned_by_asset=defaultdict(list)
    for key,n,slots,name in assigned:
        assert len(verified[key]) in (0,len(slots))
        if not verified[key]:
            assigned_by_asset[name].append((key,n,slots))
    lock=Lock()

    def progress():
        rows=[r for key,_,_,_ in assigned for r in verified[key]]
        write_json(WORK/"pilots_progress.json",{"stage":"pilots","accepted":len(rows),"expected":2*len(assigned),
            "cases":rows,"attempts":audit,"groups":sum(bool(v) for v in verified.values()),
            "pending":[key for key,_,_,_ in assigned if not verified[key]]})

    def work_asset(name,groups,planner):
        for attempt in range(first,13):
            batch=[]
            current=[(key,n,slots) for key,n,slots in groups if not verified[key]]
            if not current:
                return
            queue={planner.submit(profile_pair,assets[name],n//2,attempt,slots[0]["seed"]+attempt*701):(key,n,slots)
                   for key,n,slots in current if key not in existing[attempt]}
            prepared=[(key,existing[attempt][key]) for key,_,_ in current if key in existing[attempt]]
            for future in as_completed(queue):
                key,n,slots=queue[future]
                docs=future.result()
                if not docs or len(docs)!=len(slots):
                    print(name,key,"no collision-free pair",attempt,flush=True)
                    continue
                physical={layout(p["case"]) for p in docs}
                with lock:
                    if used&physical:
                        continue
                    used.update(physical)
                work_group=f"step3a_pilots_{key}_attempt{attempt}"
                rows=planned_receipts(docs,slots,ID_BASE["pilots"]+40*attempt+n,work_group)
                prepared.append((key,rows))
            batch=[r for _,r in sorted(prepared)]
            if not batch:
                continue
            receipt=verify_groups(batch,f"pilots_{name}_r{attempt}")
            with lock:
                audit.extend(receipt["attempts"])
                for r in receipt["cases"]:
                    verified[owner_slots[r["scene_id"]]].append(r)
                progress()
            print(name,"accepted",sum(len(verified[k]) for k,_,_ in groups),"/",2*len(groups),flush=True)

    with lock:
        progress()
    with ProcessPoolExecutor(max_workers=6,mp_context=get_context("spawn")) as planner:
        with ThreadPoolExecutor(max_workers=8) as workers:
            tasks=[workers.submit(work_asset,name,groups,planner) for name,groups in assigned_by_asset.items()]
            for f in as_completed(tasks):
                f.result()
    bad=[key for key,_,_,_ in assigned if len(verified[key])!=2]
    if bad:
        raise RuntimeError(f"Unresolved pilot groups after physical probes and candidate screening: {bad}")


if __name__=="__main__":
    code=1
    try:
        main()
        code=0
    finally:
        record_complete(RUNS/JOB,code)
