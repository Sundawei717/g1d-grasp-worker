"""Retarget previously valid approach guides onto the new training tray."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context

from background_status import record_complete
from step3a_suites import WORK,RUNS,TIER,read,write_json
from step3a_planner import placement,solve

JOB="step3a_01_basket_calibration"


def run_group(key,rows,item):
    docs=[]
    for known in rows:
        p=read(TIER/"plans"/known["case"]/"plan.json")
        source=p["case"]
        goal=p["feasible"][0]
        x,y=source["target_pose_xyz_xyzw"][:2]
        pose=placement(item["asset"],x,y,source["yaw_deg"])
        candidate={**source,"asset_family":"basket/arched_train","target_usd":item["asset"],
             "target_sha256":item["asset_sha256"],"base_mesh_hash":item["base_mesh_hash"],
             "target_pose_xyz_xyzw":pose,"part_meshes":{**item["part_meshes"],"rim_center_z_m":.047},
             "target_query":"basket","layout_description":"retargeted guide for the independent arched training basket",
             "planner_hint":{"guide":[*goal["pre_grasp_q"],goal["approach_q"][0]],
                             "preferred":[goal["candidate"]]},
             "source_guide_case":known["case"]}
        fitted=solve(candidate,max_candidates=64)
        if not fitted:
            return key,False,known['case'],bool(source["fixtures"]),[]
        docs.append(fitted)
    return key,True,None,bool(rows[0]["fixture_assets"]),docs


def initialize():
    groups=defaultdict(list)
    item=read(WORK/"asset_inventory.json")["tier2_proposed"]["arched_basket"]
    for row in read(TIER/"collection.json")["episodes"]:
        if row["asset_family"]=="basket/00001" and row.get("pair_group_id"):
            groups[row["pair_group_id"]].append(row)
    selected={True:[],False:[]}
    for key,rows in sorted(groups.items()):
        if len(rows)==2 and all((TIER/"plans"/r["case"]/"plan.json").is_file() for r in rows):
            selected[bool(rows[0]["fixture_assets"])].append((key,rows))
    target=WORK/"arched_calibrated_plans"
    target.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context("spawn")) as pool:
        queue=[pool.submit(run_group,k,v,item) for grouped in selected.values() for k,v in grouped[:6]]
        for task in as_completed(queue):
            group,ok,why,obstacle,docs=task.result()
            print(group,"pass" if ok else "rejected",why,"obstacle",obstacle,flush=True)
            if ok:
                write_json(target/f"{group}.json",{"plans":docs,"obstacle":obstacle})


if __name__=="__main__":
    code=1
    try:
        initialize()
        code=0
    finally:
        record_complete(RUNS/JOB,code)
