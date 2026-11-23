"""Replan two already screened grasp pairs around measured sugar-box support poses."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
import sys

from background_status import record_complete
from step3a_suites import WORK,RUNS,CASE_PLANS,planned_receipts,verify_groups,read,write_json
from step3a_planner import Step3Planner,build_planner,pose_matrix,ARM_NAMES
from step3a_verify import file_hash
from school_planner import serial_pose

JOB="step3a_01_sugar_calibration"



def retarget(first,group,retry):
    base=RUNS/"step3a_01_pilots_r2_demo_00"
    scene=np.array(read(base/f"C_{first}"/"expert_diagnostic.json")["target_initial_xyz_xyzw"])
    known=[read(CASE_PLANS/f"C_{first+i}.json") for i in (0,1)]
    for variant,dz in enumerate((0,.003,-.003,.006,-.006,.009)):
        docs=[]
        for i,doc in enumerate(known):
            c=doc["case"]
            fit=doc["feasible"][0]
            calibrated={**c,"target_pose_xyz_xyzw":scene.tolist(),
                          "seed":2393100+group*31+i,"support_pose_source":f"C_{first}",
                          "layout_description":"YCB sugar box replanned in its measured settled pose",
                          "planner_hint":{"guide":[*fit["pre_grasp_q"],fit["approach_q"][0]]}}
            planner=build_planner(calibrated)
            nominal=pose_matrix(c["target_pose_xyz_xyzw"])
            transform=pose_matrix(calibrated["target_pose_xyz_xyzw"])*nominal.inverse()
            lateral=nominal.rotation @ np.array([0,-.010 if retry and i==0 else 0,0])
            axes=pose_matrix(fit["wrist_xyz_xyzw"])
            hover=pose_matrix(fit["hover_xyz_xyzw"])
            for t in (axes,hover):
                t.translation=t.translation+lateral+np.array([0,0,dz])
            options=[(f"settled_{variant}_carry_orientation",transform*hover,transform*axes)]
            keep_hover=pose_matrix(fit["hover_xyz_xyzw"])
            keep_wrist=pose_matrix(fit["wrist_xyz_xyzw"])
            for t in (keep_wrist,keep_hover):
                t.translation=transform.act(t.translation+lateral)+np.array([0,0,dz])
            options.append((f"settled_{variant}_keep_orientation",keep_hover,keep_wrist))
            planner.candidate_goals=lambda:iter(options)
            result=planner.plan(seed=calibrated["seed"],stop_after=1,max_candidates=2,lift_height_m=.08)
            if not result["feasible"]:
                print(group,i,dz,[{k:v for k,v in x.items() if k.startswith("last_")} for x in result["candidate_log"]],flush=True)
                break
            docs.append({**doc,**result,"case":calibrated,"target_pose_xyz_xyzw":scene.tolist(),
                         "preflight_settle_offset_m":scene[:3].tolist()})
        if len(docs)==2:
            
            slots=[{"scene_id":f"tier2_sugar_cal_{retry}_{group}_{i}",
                    "seed":2393100+group*31+i} for i in (0,1)]
            return planned_receipts(docs,slots,9300+40*retry+2*group,f"tier2_sugar_cal_{retry}_{group}")
    return None


def main():
    retry=int("--shift-negative" in sys.argv)
    groups=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context("spawn")) as pool:
        tasks=[pool.submit(retarget,reference,i,retry) for i,reference in enumerate((8580,8582))]
        for task in as_completed(tasks):
            cases=task.result()
            if cases:
                print("Planned",[(r["case_id"],r["part"]) for r in cases],flush=True)
                groups.append(cases)
    if not groups:
        raise RuntimeError("The measured support pose did not yield a collision-free pair")
    result=verify_groups(groups,"pilots_sugar_calibration_r1" if retry else "pilots_sugar_calibration")
    if len(result["cases"])!=4:
        raise RuntimeError(f"Only {len(result['cases'])}/4 calibrated sugar-box episodes passed")


if __name__=="__main__":
    code=1
    try:
        main()
        code=0
    finally:
        record_complete(RUNS/JOB,code)
