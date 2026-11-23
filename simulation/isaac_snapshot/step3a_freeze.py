"""Create the irreversible content-addressed benchmark only after exact expert admission."""
from collections import Counter,defaultdict
from pathlib import Path
import time

from background_status import record_complete
from step3a_verify import *
from tier1_plan import base_mesh_signature


def main():
    inventory=read(WORK/"asset_inventory.json")
    prior=read(TIER/"final_test_reservation.json")
    meshes=inventory["l4_heldout"]
    assert len(meshes)==5
    assert meshes["white_mug"]["raw_visual_base_mesh_hash"]==prior["reserved_base_mesh_signatures"]["mug/00004"]
    held={k:{"base_mesh_hash":v["base_mesh_hash"],"source_sha256":v["source_sha256"],
             "asset":v["asset"],"asset_sha256":v["asset_sha256"],"atlas_sha256":v["atlas_sha256"]} for k,v in meshes.items()}
    training={**inventory["tier1_training_base_hashes"],
              **{k:v["base_mesh_hash"] for k,v in inventory["tier2_proposed"].items()}}
    assert not set(v["base_mesh_hash"] for v in held.values())&set(training.values())
    assert len({v["base_mesh_hash"] for v in held.values()})==5
    fingerprint=WORK/"l4_base_mesh_fingerprints.json"
    certificate={"status":"permanently_reserved","heldout":held,
                 "training_assets_excluded":training,"shared_base_meshes":0}
    if fingerprint.is_file() and read(fingerprint)!=certificate:
        raise ValueError("Permanent L4 mesh registry changed")
    write_json(fingerprint,certificate)
    for stage in ("l1","l2","l3","l4"):
        flag=RUNS/f"step3a_01_{stage}.complete"
        while not flag.is_file():
            time.sleep(30)
        if read(flag)["outcome"]!="PASS":
            raise RuntimeError(f"{stage} is incomplete")
    ray_result=WORK/"physx_rays.json"
    while not ray_result.is_file():
        marker=RUNS/"step3a_01_rays.complete"
        if marker.is_file() and read(marker)["outcome"]!="PASS":
            raise RuntimeError("PhysX aperture audit incomplete")
        time.sleep(20)
    rays=read(ray_result)["assets"]
    assert all(rays[k]["passed"] for k in ("white_mug","odd_handle_mug","round_basket"))
    layout_original,case_original,old_ids,old_seeds=isolate_from_previous()
    accepted=[]
    for layer in range(1,5):
        d=read(WORK/f"l{layer}_progress.json")
        assert len(d["cases"])==20 and d["accepted"]==20
        accepted.extend(d["cases"])
    slot_map={r["scene_id"]:r for r in prior["slots"]}
    assert len({r["scene_id"] for r in accepted})==len({r["case_id"] for r in accepted})==len(slot_map)==80
    assert len({r["episode_id"] for r in accepted})==80
    assert {r["seed"] for r in accepted}=={slot["seed"] for slot in prior["slots"]}
    layouts=defaultdict(set)
    for case in accepted:
        slot=slot_map[case["scene_id"]]
        assert case["tier"]==slot["tier"] and case["seed"]==slot["seed"]
        assert case["scene_signature"]==scene_signature(case) and case["scene_signature"] not in case_original
        initial=layout(case)
        assert initial==case["layout_signature"] and initial not in layout_original
        assert case["seed"] not in old_seeds and case["scene_id"] not in old_ids
        layouts[initial].add(case["scene_family"])
        assert file_hash(case["expert_path"])==case["expert_sha256"]
        assert case["replay_verified"] and replay_ok(read(case["expert_diagnostic"]))
        case["plan_sha256"]=file_hash(case["plan"])
        case["expert_diagnostic_sha256"]=file_hash(case["expert_diagnostic"])
        if case["tier"]==4:
            assert case["base_mesh_hash"] in {v["base_mesh_hash"] for v in held.values()}
            assert case["base_mesh_hash"]==base_mesh_signature(case["target_usd"])
    assert all(len(k)==1 for k in layouts.values())
    for k,p in (("l3",3),("l4",4)):
        label="layout_type" if p==3 else "base_mesh_hash"
        counts=Counter(r[label] for r in accepted if r["tier"]==p)
        if p==3:
            assert counts=={"front_panel":8,"dual_corridor":6,"adjacent_distractor":6}
        else:
            assert len(counts)==5 and set(counts.values())=={4}
    frozen=WORK/"frozen.json"
    if frozen.exists():
        raise ValueError("Refusing to replace a permanent freeze")
    receipt={"status":"frozen","count":80,"replay_verified":True,"part_hit_passes":80,
             "training_overlap":0,"development_overlap":0,"levels":{"L1":20,"L2":20,"L3":20,"L4":20},
             "authoritative_fingerprints":{"path":str(fingerprint),"sha256":file_hash(fingerprint)},
             "sources":{"tier1_sha256":file_hash(TIER/"collection.json"),"development_sha256":file_hash(DEV),
                        "reservation_sha256":file_hash(TIER/"final_test_reservation.json")},
             "cases":sorted(accepted,key=lambda r:r["scene_id"])}
    write_json(frozen,receipt)
    write_json(WORK/"frozen_sha256.json",{"path":str(frozen),"sha256":file_hash(frozen)})
    print("80/80 expert-verified and Part Hit passed; frozen",file_hash(frozen),flush=True)


if __name__=="__main__":
    code=1
    try:
        main()
        code=0
    finally:
        record_complete(RUNS/"step3a_01_freeze",code)
