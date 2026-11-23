"""Use physically successful Tier-1 wrist orientations to guide local substitute grasps."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,as_completed
from multiprocessing import get_context
from pathlib import Path
import json
import math
import sys

import h5py
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from background_status import record_complete
from step3a_planner import asset_case,placement,build_planner,PAD_OFFSET,ARM_NAMES,CFG,URDF
from step3a_suites import WORK,RUNS,TIER,read,planned_receipts,verify_groups,paired_obstacle
from step3a_verify import file_hash

JOB="step3a_01_local_guides"
ASSETS=("marker_long_box","compact_box","round_package","matte_soda")


def read_reference(path):
    with h5py.File(path) as f:
        case=json.loads(f["meta/case_json"].asstr()[()])
        upto=int(np.flatnonzero(f["phase"].asstr()[()]=="lift")[0])
        frame=f["obs/eef_pose"][upto]
        return case,frame,f["obs/qpos"][0],f["obs/qpos"][max(0,upto-10)]


def make_pair(asset,index,group,references,detour,gap_search):
    for ref in references[index::2]:
        source,frame,home,guide=read_reference(ref["path"])
        orient=Rotation.from_quat(frame[3:])
        heading=orient.apply([0,1,0])
        yaw=math.degrees(math.atan2(heading[1],heading[0]))
        x,y=source["target_pose_xyz_xyzw"][:2]
        x+=.005 if group==0 else -.007
        y+=.009 if group==0 else .006
        docs=[]
        seed=2420000+10*ASSETS.index(asset["name"])+group*2
        target=placement(asset["asset"],x,y,yaw)
        if detour and asset["name"]=="marker_long_box":
            # Stand the narrow template on one end so the upper grasp remains clear of the table.
            upright=Rotation.from_euler("z",yaw,degrees=True)*Rotation.from_euler("x",90,degrees=True)
            target=[x,y,target[2]-asset["bounds"][0][1],*upright.as_quat().tolist()]
        parts=[next(reversed(asset["grips"]))] if detour else asset["grips"]
        for j,part in enumerate(parts):
            row=asset_case(asset,part,group,seed)
            row.update(target_pose_xyz_xyzw=target,yaw_deg=yaw,home_q=home.tolist(),
                       planner_hint={"guide":[guide.tolist()]},source_guide=ref["case"],
                       check_alternate_gaps=gap_search,
                       layout_description="guided by a verified Tier-1 wrist orientation")
            planner=build_planner(row)
            center=np.array(row["grip_geometry"][0])
            if detour and asset["name"] in {"round_package","matte_soda"}:
                verified_grip=frame[:3]+orient.apply(PAD_OFFSET)
                center[2]=np.clip(verified_grip[2]-target[2],.69*asset["bounds"][1][2],asset["bounds"][1][2]-.006)
            opts=[]
            offsets=((0,0,0),(.005,0,0),(-.004,0,0),(0,0,-12),(0,0,12),(.007,0,-12),(.007,0,12))
            if detour:
                offsets+=((0,18,0),(0,-18,0),(.005,-22,12),(.005,22,-12))
            for dz,tilt,twist in offsets:
                local=center+np.array([0,0,dz])
                r=(Rotation.from_euler("z",twist,degrees=True)*orient
                   *Rotation.from_euler("y",tilt,degrees=True))
                grip=planner.target_pose.act(local)
                wrist=grip-r.apply(PAD_OFFSET)
                hover=wrist-r.apply([.068,0,0])
                opts.append((f"guided_z{dz:.3f}_tilt{tilt}_yaw{twist}",pin.SE3(r.as_matrix(),hover),pin.SE3(r.as_matrix(),wrist)))
            planner.candidate_goals=lambda: iter(opts)
            result=planner.plan(seed=seed,stop_after=1,max_candidates=len(opts),lift_height_m=.08)
            if not result["feasible"]:
                print(asset["name"],group,ref["case"],part,
                      result["home_contacts"],
                      [k for log in result["candidate_log"] for k in log if k.startswith("last_")][-4:],
                      flush=True)
                break
            row["planned_gripper_gap_m"]=getattr(planner,"planned_gap_m",row["grip_geometry"][-1])
            docs.append({**result,"case":row,"obstacles":[],
                         "home_q":row["home_q"],"right_joint_names":list(ARM_NAMES),
                         "target_pose_xyz_xyzw":target,"school_sha256":file_hash(CFG["school_usd"]),
                         "target_sha256":asset["asset_sha256"],"urdf_sha256":file_hash(URDF)})
        if detour and len(docs)==1:
            alternate=paired_obstacle(docs[0])
            if alternate:
                docs.append(alternate)
        if len(docs)==2:
            tag=f"tier2_guided_{asset['name']}_{group}_{int(detour)}"
            slots=[{"scene_id":f"{tag}_{n}","seed":seed+n+100*int(detour)} for n in (0,1)]
            return planned_receipts(docs,slots,9800+40*int(detour)+4*ASSETS.index(asset["name"])+2*group,tag)
    return None


def main():
    detour="--with-detour" in sys.argv
    gap_search="--check-gaps" in sys.argv
    inventory=read(WORK/"asset_substitutes.json")["assets"]
    selected=read(TIER/"training_split.json")["episodes"]
    refs={k:[r for r in selected if r["split"]=="train" and r["bucket"]==k][:12] for k in ("pepsi","white_box")}
    done=[]
    with ProcessPoolExecutor(max_workers=6,mp_context=get_context("spawn")) as planning:
        with ThreadPoolExecutor(max_workers=4) as physics:
            for name in ASSETS:
                if name not in inventory:
                    continue
                code="pepsi" if name in {"round_package","matte_soda"} else "white_box"
                planned=[planning.submit(make_pair,inventory[name],i,i,refs[code],detour,gap_search) for i in (0,1)]
                def run_groups(futures,label):
                    groups=[g for f in futures if (g:=f.result()) is not None]
                    print(label,"guided groups",len(groups),flush=True)
                    if not groups:
                        return []
                    return verify_groups(groups,f"pilots_guide_{label}_r{int(detour)}")["cases"]
                done.append(physics.submit(run_groups,planned,name))
            summary=[f.result() for f in as_completed(done)]
    if sum(len(x) for x in summary)<4*len(ASSETS):
        raise RuntimeError("At least one local asset lacks two verified retargeted groups")


if __name__=="__main__":
    status=1
    try:
        main()
        status=0
    finally:
        record_complete(RUNS/JOB,status)
