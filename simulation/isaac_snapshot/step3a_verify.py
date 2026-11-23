"""Verify small Step 3-A and benchmark groups through recording and native replay."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import h5py
import numpy as np

import step2_dev
from step3a_resources import SimAllocator
from step3a_assets import WORK,ROOT
from three_view_detector import write_json,sha256 as file_hash
from three_view_eval import read
from tier1_collect import admit_episode,paired,scene_signature
from unified_grasp import EPISODE_STEPS

RUNS=ROOT/"runs"
TIER=RUNS/"tier1_v3_01"
DEV=RUNS/"step2_v3_01/development.json"
CASE_PLANS=WORK/"plans"
CASE_PLANS.mkdir(exist_ok=True)


def layout(case):
    """Exclude physical level layouts even when the requested contact part changes."""
    objects=[(case["asset_family"], tuple(np.round(case["target_pose_xyz_xyzw"],5)))]
    for f in case["fixtures"]:
        objects.append((f.get("asset_family",f["sha256"]),tuple(np.round(f["pose"],5))))
    p=[sorted(objects),tuple(np.round(case["home_q"],5))]
    return sha256(json.dumps(p,separators=(",",":")).encode()).hexdigest()


def original_cases():
    old=[]
    for r in read(TIER/"collection.json")["episodes"]:
        with h5py.File(r["path"]) as h:
            old.append(json.loads(h["meta/case_json"].asstr()[()]))
    old+=read(DEV)["cases"]
    return old


def isolate_from_previous():
    old=original_cases()
    return ({layout(r) for r in old},{scene_signature(r) for r in old},
            {r["scene_id"] for r in old if "scene_id" in r},
            {r["seed"] for r in old if "seed" in r})


def reserve():
    return {layer:[r for r in read(TIER/"final_test_reservation.json")["slots"] if r["tier"]==layer]
            for layer in range(1,5)}


def planned_receipts(docs,slots,number,group_key):
    assert len(docs)==len(slots) and all(d["feasible"] for d in docs)
    records=[]
    for i,(doc,slot) in enumerate(zip(docs,slots,strict=True)):
        row=dict(doc["case"])
        cid=f"{row['group'][0]}_{number+i}"
        names={k:slot[k] for k in ("scene_id","seed","tier") if k in slot}
        row.update(**names,case_id=cid,scene_family=group_key,
                   pair_group_id=group_key if len(docs)>1 else None,episode_id=f"step3a_{cid}",
                   split="frozen_test" if "tier" in slot else "tier2_pilot")
        row["bucket"]=row.get("bucket",row["asset_role"])
        asset=CASE_PLANS/f"{cid}.json"
        fixed={**doc,"case":row}
        if asset.is_file() and read(asset)!=fixed:
            raise ValueError(f"Changed deterministic replay plan {asset}")
        write_json(asset,fixed)
        records.append(dict(row,scene_signature=scene_signature(row),layout_signature=layout(row),plan=str(asset)))
    return records


def admits_pair(group,demos):
    if len(group)==1:
        return True
    first,second=group
    a,b=[demos[r["case_id"]] for r in group]
    if first["asset_family"]==second["asset_family"]:
        if first["part"]==second["part"] and bool(first["fixtures"])!=bool(second["fixtures"]):
            return (np.allclose(first["target_pose_xyz_xyzw"],second["target_pose_xyz_xyzw"],atol=1e-8)
                    and np.allclose(a["initial_q"],b["initial_q"],atol=1e-5,rtol=0)
                    and abs(a["initial_width"]-b["initial_width"])<=1e-5
                    and np.linalg.norm(np.subtract(a["initial_target"][:3],b["initial_target"][:3]))<.001
                    and 1-abs(np.dot(a["initial_target"][3:],b["initial_target"][3:]))<1e-6)
        return paired(a,b)
    return (np.allclose(a["initial_q"],b["initial_q"],atol=1e-5,rtol=0)
            and len(first["fixtures"])==len(second["fixtures"])==1
            and np.allclose(first["fixtures"][0]["pose"],second["target_pose_xyz_xyzw"],atol=1e-8)
            and np.allclose(second["fixtures"][0]["pose"],first["target_pose_xyz_xyzw"],atol=1e-8))


def replay_ok(d):
    roll=d.get("policy_rollout",{})
    a=d.get("replay_alignment",{})
    return (roll.get("status")=="success" and roll.get("part_hit") is True and roll["steps"]<=EPISODE_STEPS
            and a.get("max_timestamp_error_s",1)<=2e-7
            and a.get("max_native_setpoint_error",1)<=1e-6
            and abs(a.get("trigger_delta_s",1))<=.10000001)


def fixture_or_pregrasp_contact(diagnostic):
    """Keep transport contact diagnostics distinct from an unsafe expert approach."""
    forbidden=set(diagnostic.get("forbidden_contact_pairs",{}))
    if not forbidden:
        return False
    pads={f"/World/G1D/right_dex1_finger_link_{n}_mid" for n in (1,2)}
    target="/World/Target/Object"
    for pair in forbidden:
        links=set(pair.split(" | "))
        if len(links)!=2 or target not in links or len(links & pads)!=1:
            return True
    collisions=diagnostic.get("diagnostics",{}).get("collision",{})
    transport=set()
    for phase,row in collisions.items():
        seen=set(row.get("pair_onsets",{})) & forbidden
        if phase not in {"lift","hold"} and seen:
            return True
        transport.update(seen)
    return forbidden!=transport


def verify_groups(groups, stage):
    step2_dev.WORK=WORK
    receipt=WORK/f"{stage}_result.json"
    rows=[r for group in groups for r in group]
    if receipt.is_file():
        previous=read(receipt)
        from_collection_filter=(any(r.get("split")=="frozen_test" for r in rows)
            and any(a.get("record_failure")=="A new demonstration reported forbidden contact"
                    for a in previous["attempts"]))
        after_close_rule_changed=(stage.startswith("pilots_") and "contact_policy" not in previous
            and any(a.get("record_failure")=="A new demonstration reported forbidden contact"
                    for a in previous["attempts"]))
        if not from_collection_filter and not after_close_rule_changed:
            return previous
        print(f"{stage}: rechecking the saved recordings with the benchmark contact contract.",flush=True)
    demos=step2_dev.simulate(f"{stage}_demo",rows,"three_view_expert_v3",
                            allocator=SimAllocator(),max_shard_cases=4,resume_completed=True)
    admitted={}
    attempts=[]
    for r in rows:
        key=r["case_id"]
        result,diag,path=demos[key]
        a={"case":key,"slot":r["scene_id"],"asset":r["asset_family"],
           "record_diagnostic":path,"record_status":result["status"],"record_failure":result.get("failure")}
        if result.get("dataset_receipt"):
            try:
                enforce=(r.get("split")!="frozen_test" and fixture_or_pregrasp_contact(diag))
                admitted[key]=admit_episode(r,result["dataset_receipt"],diag,enforce_contact_filter=enforce)
            except ValueError as exc:
                a["record_failure"]=str(exc)
        if "policy_rollout" in diag:
            p=diag["policy_rollout"]
            a["capture_status"]=p.get("status")
            a["capture_failure"]=p.get("trials",[{}])[0].get("failure_category") if p.get("trials") else None
            a["capture_steps"]=p.get("steps")
        attempts.append(a)
    print(f"{stage}: recorded {len(admitted)}/{len(rows)}",flush=True)
    qualifying=[g for g in groups if all(r["case_id"] in admitted for r in g) and admits_pair(g,admitted)]
    candidates=[r for g in qualifying for r in g]
    results=step2_dev.simulate(f"{stage}_replay",candidates,"three_view_unified_replay",archives=admitted,
                               allocator=SimAllocator(),max_shard_cases=4,resume_completed=True)
    accepted=[]
    audit={r["case"]:r for r in attempts}
    for group in qualifying:
        checks=[]
        for r in group:
            key=r["case_id"]
            summary,diag,path=results[key]
            roll=diag.get("policy_rollout",{})
            audit[key].update(replay_diagnostic=path,replay_status=roll.get("status"),part_hit=roll.get("part_hit"),
                              replay_pass=bool(replay_ok(diag)),replay_wall_s=roll.get("wall_s"))
            d=admitted[key]
            checks.append({**r,"expert_path":d["path"],"expert_sha256":d["sha256"],"expert_diagnostic":path,
                          "expert_alignment":diag.get("replay_alignment"),"replay_verified":bool(replay_ok(diag)),
                          "head_prompt":{"bbox_0":d["bbox"],"affordance_0":d["point"]},
                          "initial_q":d["initial_q"],"initial_width":d["initial_width"],
                          "initial_target":d["initial_target"],"capture_wall_s":d["capture_wall_s"]})
        if all(r["replay_verified"] for r in checks):
            accepted.extend(checks)
    result={"cases":accepted,"attempts":attempts,"group_count":len(qualifying),
            "contact_policy":"fixture_and_pregrasp_filter_v2",
            "replay_pass_counts":dict(Counter(r["asset_family"] for r in accepted))}
    write_json(receipt,result)
    print(f"{stage}: accepted {len(accepted)}/{len(rows)} after replay",flush=True)
    return result
