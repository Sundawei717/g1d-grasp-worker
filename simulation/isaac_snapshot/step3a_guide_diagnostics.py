from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from step3a_suites import WORK,RUNS,read
from step3a_probe_plan import test
from background_status import record_complete

JOB="step3a_01_shape_guide"

def run():
    data=read(WORK/"asset_inventory.json")["tier2_proposed"]
    names=("arched_basket","011_banana","019_pitcher_base","035_power_drill","006_mustard_bottle","024_bowl")
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context("spawn")) as p:
        pending=[]
        for id,name in enumerate(names):
            for index in (40,41):
                slots=[{"seed":2195400+id*11+index}]
                pending.append(p.submit(test,name,2*index,slots,data[name]))
        for f in as_completed(pending): f.result()
if __name__=="__main__":
    code=1
    try:
        run()
        code=0
    finally:
        record_complete(RUNS/JOB,code)
