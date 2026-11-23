"""Freeze a replay-verified L1 from previously unused offline plans."""
from collections import defaultdict
import hashlib

from background_status import record_complete
from step3a_verify import *
from step3a_planner import placement,solve


def main():
    exclude,signatures,ids,seeds=isolate_from_previous()
    reserve1=reserve()[1]
    assert not {r["scene_id"] for r in reserve1}&ids
    assert not {r["seed"] for r in reserve1}&seeds
    pools=defaultdict(list)
    for ready in sorted((TIER/"plans").glob("*/ready.json")):
        case=read(ready)
        bucket=case["bucket"]
        if case["fixtures"] or bucket not in ("mug","basket","carton","white_box","pepsi"):
            continue
        if layout(case) in exclude or scene_signature(case) in signatures:
            continue
        if read(ready.parent/"plan.json").get("feasible"):
            pools[bucket].append(ready.parent/"plan.json")
    for bucket,paths in pools.items():
        paths.sort(key=lambda p:(abs(read(p)["case"].get("yaw_deg",0))<50,hashlib.sha256(p.name.encode()+p.parent.name.encode()).hexdigest()))
        print(bucket,"unused candidates",len(paths),flush=True)
    slots=dict(zip((r["scene_id"] for r in reserve1),reserve1))
    assigned={r["scene_id"]:b for i,b in enumerate(("mug","basket","carton","white_box","pepsi"))
              for r in reserve1[4*i:4*i+4]}
    completed={}
    for attempt in range(12):
        batch=[]
        for i,(slot,bucket) in enumerate(assigned.items()):
            if slot in completed:
                continue
            number=3000+100*attempt+i
            key=f"step3a_{slot}_r{attempt}"
            stored=CASE_PLANS/f"C_{number}.json"
            if stored.is_file():
                doc=read(stored)
                case=doc["case"]
                if case["scene_id"]!=slot or case["scene_family"]!=key or case["seed"]!=slots[slot]["seed"]:
                    raise ValueError(f"L1 retained plan does not match {slot}: {stored}")
                physical=layout(case)
                if physical in exclude:
                    raise ValueError(f"L1 retained a duplicate physical layout: {stored}")
                exclude.add(physical)
                batch.append([{**case,"scene_signature":scene_signature(case),
                               "layout_signature":physical,"plan":str(stored)}])
                continue
            paths=pools[bucket]
            if not paths:
                continue
            while paths:
                chosen=read(paths.pop(0))
                physical=layout(chosen["case"])
                if physical in exclude:
                    continue
                if attempt and "pre_grasp_q" not in chosen["feasible"][0]:
                    c=chosen["case"]
                    yaw=(55,-60,75,-75)[i%4]
                    x,y=c["target_pose_xyz_xyzw"][:2]
                    c={**c,"target_pose_xyz_xyzw":placement(c["target_usd"],x,y,yaw),
                       "yaw_deg":yaw,"planner_hint":{}}
                    regenerated=solve(c,max_candidates=110)
                    if not regenerated:
                        continue
                    chosen=regenerated
                    physical=layout(c)
                if physical not in exclude:
                    break
            else:
                continue
            exclude.add(physical)
            group=planned_receipts([chosen],[slots[slot]],number,key)
            batch.append(group)
        if not batch:
            break
        r=verify_groups(batch,f"l1_r{attempt}")
        completed.update({r["scene_id"]:r for r in r["cases"]})
        write_json(WORK/"l1_progress.json",{"cases":list(completed.values()),"accepted":len(completed)})
        if len(completed)==20:
            return
    if len(completed)!=20:
        raise RuntimeError(f"Only {len(completed)}/20 L1 cases accepted")


if __name__=="__main__":
    code=1
    try:
        main()
        code=0
    finally:
        record_complete(RUNS/"step3a_01_l1",code)
