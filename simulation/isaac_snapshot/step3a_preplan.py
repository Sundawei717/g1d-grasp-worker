"""Compute measured collision-screened proposals while the GPU lanes replay."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context

from background_status import record_complete
from step3a_suites import WORK,RUNS,assignments,read,write_json,profile_receipt,compute_profile_pair

JOB="step3a_01_preplan"

def build(item,n,attempt,seed,parts=None):
    path,contract=profile_receipt(item,n,attempt,seed,parts)
    if path.is_file():
        return item["name"],n,attempt,bool(read(path)["plans"])
    plans=compute_profile_pair(item,n,attempt,seed,parts)
    if profile_receipt(item,n,attempt,seed,parts)[1]==contract:
        write_json(path,{"contract":contract,"plans":plans})
    return item["name"],n,attempt,bool(plans)


def main():
    inv=read(WORK/"asset_inventory.json")
    (WORK/"offline_candidates").mkdir(exist_ok=True)
    args=[]
    for stage,extent in (("l4","l4_heldout"),("pilots","tier2_proposed")):
        assets=inv[extent]
        for key,n,slots,name in assignments(stage,assets):
            if stage=="l4" and name!="007_tuna_fish_can":
                continue
            for attempt in (2,3):
                args.append((assets[name],n//2,attempt,slots[0]["seed"]+attempt*701))
    with ProcessPoolExecutor(max_workers=6,mp_context=get_context("spawn")) as pool:
        pending=[pool.submit(build,*row) for row in args]
        for f in as_completed(pending):
            print(*f.result(),flush=True)


if __name__=="__main__":
    code=1
    try:
        main()
        code=0
    finally:
        record_complete(RUNS/JOB,code)
