from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import sys
import time

from step3a_suites import clone_new,assignments,read,WORK
from step3a_planner import build_planner

def test(key,number,slots,item):
    seed=slots[0]["seed"]
    for row in clone_new(item,seed,number//2):
        start=time.monotonic()
        planner=build_planner(row)
        result=planner.plan(seed=seed,stop_after=1,max_candidates=28,lift_height_m=.08)
        stages=Counter()
        for log in result["candidate_log"]:
            for k,v in log.items():
                if k.startswith("last_"):
                    stages[str((k,v[:2] if "collision" in k and isinstance(v,(list,tuple)) else ""))]+=1
        print(key,row["part"],"OK" if result["feasible"] else "NO",
              "home",result["home_contacts"],"tried",len(result["candidate_log"]),
              "elapsed",round(time.monotonic()-start,1),stages.most_common(5),flush=True)

if __name__=="__main__":
    inv=read(WORK/"asset_inventory.json")
    stage=sys.argv[1]
    selected=inv["tier2_proposed" if stage=="pilots" else "l4_heldout"]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(test,key,n,slots,selected[name])
                 for key,n,slots,name in assignments(stage,selected) if key.endswith("_0")]
        for f in futures:
            f.result()
