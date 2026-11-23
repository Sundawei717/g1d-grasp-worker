"""Build and verify the independent L2?L4 benchmark and small Tier-2 pilots."""
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import hashlib
import sys
import time

from background_status import record_complete
from step3a_assets import PREPARED
from step3a_planner import *
from step3a_verify import *

L2_PARTS=("mug","basket")*3
L3_LAYOUTS=("front_panel","dual_corridor","adjacent_distractor","front_panel","dual_corridor",
            "adjacent_distractor","front_panel","dual_corridor","adjacent_distractor","front_panel")
L3_BUCKETS=("carton","basket","mug","mug","carton","basket","basket","mug","carton","carton")
L4_NAMES=("white_mug","odd_handle_mug","round_basket","008_pudding_box","007_tuna_fish_can")
ID_BASE={"l2":4600,"l3":5800,"l4":7200,"pilots":8500}


def wait_marker(job):
    f=RUNS/f"step3a_01_{job}.complete"
    while not f.is_file():
        time.sleep(20)
    if read(f)["outcome"]!="PASS":
        raise RuntimeError(f"{job} did not pass")


def plan_bank():
    singles=defaultdict(list)
    by_pair=defaultdict(list)
    for f in sorted((TIER/"plans").glob("*/ready.json")):
        r=read(f)
        p=f.parent/"plan.json"
        if not p.is_file():
            continue
        row=(r,p)
        singles[r["bucket"]].append(row)
        if r.get("pair_group_id") and r["bucket"] in ("mug","basket","carton"):
            by_pair[r["pair_group_id"]].append(row)
    pairs=defaultdict(list)
    for group,rs in by_pair.items():
        if len(rs)==2 and len({r["part"] for r,_ in rs})==2 and len({layout(r) for r,_ in rs})==1:
            r=rs[0][0]
            pairs[r["bucket"],bool(r["fixtures"])].append((group,rs))
    for records in pairs.values():
        records.sort(key=lambda r:hashlib.sha256(r[0].encode()).hexdigest())
    return singles,pairs


def old_pair(pairs,bucket,has_obstacle,used):
    for group,rs in pairs.get((bucket,has_obstacle),[]):
        if layout(rs[0][0]) in used:
            continue
        docs=[read(p) for _,p in rs]
        if all(d.get("feasible") for d in docs):
            yield docs


def clone_new(item,seed,index,parts=None):
    rows=[]
    for part in list(item["grips"]) if parts is None else parts:
        c=asset_case(item,part,index,seed)
        c["layout_description"]="fresh orientations on pinned atlas"
        rows.append(c)
    return rows


def compute_profile_pair(item,number,attempt,seed,parts=None):
    for retry in range(2):
        n=number+attempt*13+retry
        rows=clone_new(item,seed+retry*997,n,parts)
        if "basket" in item["name"] and number%4==3 and (attempt<3 or attempt%3==0):
            # The alternate pilot uses the same part pair through a verified detour.
            r=rows[0]
            xx,yy=r["target_pose_xyz_xyzw"][:2]
            source=authored_panel()
            block={"name":"blocker","usd":source,"sha256":file_hash(source),
                   "pose":placement(source,xx-.05,yy-.125,0),"asset_family":"step3a/front_panel"}
            for c in rows:
                c["fixtures"]=[block]
                c["group"]="B_obstacle"
                c["layout_description"]="pillar in basket approach, 8 cm clearance screened"
        docs=[]
        for row in rows:
            doc=solve(row,max_candidates=56)
            if doc is None:
                break
            docs.append(doc)
        if len(docs)==len(rows):
            return docs
        if attempt>=4 and item["name"] in {"006_mustard_bottle","011_banana","019_pitcher_base","035_power_drill"}:
            for row in rows[len(docs):]:
                doc=solve(row,max_candidates=40)
                if doc is not None:
                    docs.append(doc)
            for doc in docs:
                alternate=paired_obstacle(doc)
                if alternate is not None:
                    return [doc,alternate]
    return None


def paired_obstacle(doc):
    """For shapes with one readily grasped region, validate it with and without a blocker."""
    original=doc["case"]
    target_x,target_y=original["target_pose_xyz_xyzw"][:2]
    usd=authored_panel()
    digest=file_hash(usd)
    for dx,dy in ((-.050,-.115),(0,-.110),(.050,-.115),(-.075,-.07),(.075,-.07),
                  (0,.075),(.043,.075),(-.043,.075)):
        fixture_pose=placement(usd,target_x+dx,target_y+dy,10 if dx>=0 else -10)
        block={"name":"blocker","usd":usd,"sha256":digest,"pose":fixture_pose,"asset_family":"step3a/front_panel"}
        trial={**original,"fixtures":[block],"group":"B_obstacle",
               "layout_description":"same grasp region and initial pose, detour around a blocking panel"}
        result=adapt(doc,trial,block=True)
        if result is not None:
            return result
    return None


def profile_receipt(item,number,attempt,seed,parts):
    files=(ROOT/"step3a_planner.py",ROOT/"step3a_suites.py")
    contract={"asset_sha256":item["asset_sha256"],"atlas_sha256":item["atlas_sha256"],
              "number":number,"attempt":attempt,"seed":seed,"parts":parts,
              "code":[file_hash(p) for p in files]}
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    return WORK/"offline_candidates"/f"{item['name']}_{digest}.json",contract


def profile_pair(item,number,attempt,seed,parts=None):
    receipt,contract=profile_receipt(item,number,attempt,seed,parts)
    if receipt.is_file():
        saved=read(receipt)
        assert saved["contract"]==contract
        return saved["plans"]
    return compute_profile_pair(item,number,attempt,seed,parts)


def multi_pair(singles,index,attempt,seed):
    rng=np.random.default_rng(seed+attempt*71)
    targets=(("mug","pepsi"),("carton","mug"),("basket","pepsi"),("white_box","mug"))[index%4]
    for retry in range(28):
        rows=[]
        yaw=(0,12,-15,25,-30)[(attempt+retry)%5]
        for j,bucket in enumerate(targets):
            ready,path=next((r,p) for r,p in singles[bucket] if not r["fixtures"] and read(p).get("feasible"))
            source=read(path)["case"]
            asset=source["target_usd"]
            x=(.105,.228)[(j+retry)%2]+rng.uniform(-.012,.012)
            y=(-.464,-.398)[(j+retry)%2]+rng.uniform(-.013,.013)
            current={**source,"target_pose_xyz_xyzw":placement(asset,x,y,yaw if bucket!="pepsi" else 0),
                     "seed":seed,"yaw_deg":yaw if bucket!="pepsi" else 0,"planner_hint":{},
                     "home_q":EVIDENCE["samples"][0]["right_rad"],"fixtures":[],"bucket":bucket,
                     "layout_description":"same tabletop objects, different requested target"}
            rows.append(current)
        for c,d in (rows,rows[::-1]):
            c["fixtures"]=[fixture(d,"neighbor")]
        docs=[solve(row,max_candidates=75) for row in rows]
        if all(docs):
            return docs
    return None


def l3_pair(pairs,bucket,kind,used,attempt):
    marker=ROOT/"assets/soda_can/00000/object.usdz"
    fresh_panel=authored_panel()
    for docs in old_pair(pairs,bucket,True,used):
        src=docs[0]["case"]
        used.add(layout(src))
        old=src["fixtures"][0]
        xx,yy=src["target_pose_xyz_xyzw"][:2]
        ox,oy=old["pose"][:2]
        for sign in (1,-1):
            for jitter in (0,.014,-.014):
                if kind=="front_panel":
                    x=ox+jitter
                    obs=[{"name":"blocker","usd":fresh_panel,"sha256":file_hash(fresh_panel),
                          "pose":placement(fresh_panel,x,oy+sign*.008,sign*12),"asset_family":"step3a/front_panel"}]
                else:
                    side=-1 if ox>=xx else 1
                    x=float(np.clip(xx+side*(.105+jitter),.067,.285))
                    y=yy-.064 if kind=="dual_corridor" else yy+.012+sign*.017
                    obs=[old,{"name":"neighbor","usd":str(marker),"sha256":file_hash(marker),
                              "pose":placement(str(marker),x,y,0),"asset_family":"soda_can/00000"}]
                rows=[{**d["case"],"fixtures":obs,"layout_description":kind,"layout_type":kind} for d in docs]
                altered=[adapt(d,row,block=False) for d,row in zip(docs,rows,strict=True)]
                if all(altered):
                    check=build_planner(rows[0]).edge(np.asarray(rows[0]["home_q"]),
                                                    np.asarray(docs[0]["feasible"][0]["approach_q"][0]))
                    if kind=="front_panel" and not any(x.startswith("blocker:") for x in (check or [])):
                        continue
                    return altered
    return None


def previous_layouts():
    static,_,ids,seeds=isolate_from_previous()
    for file in ("l1_progress.json","l2_progress.json","l3_progress.json","l4_progress.json","pilots_progress.json"):
        f=WORK/file
        if f.is_file():
            static.update(layout(r) for r in read(f)["cases"])
    return static,ids,seeds


def assignments(stage,assets):
    slots=reserve()
    if stage=="l2":
        r=slots[2]
        return [(f"part_{i}",2*i,r[2*i:2*i+2],b) for i,b in enumerate(L2_PARTS)]+[
                 (f"multi_{i}",12+2*i,r[12+2*i:14+2*i],i) for i in range(4)]
    if stage=="l3":
        r=slots[3]
        return [(f"{k}_{i}",2*i,r[2*i:2*i+2],(b,k)) for i,(b,k) in enumerate(zip(L3_BUCKETS,L3_LAYOUTS))]
    if stage=="l4":
        r=slots[4]
        result=[]
        for i,name in enumerate(L4_NAMES):
            if name=="007_tuna_fish_can":
                result.extend((f"{name}_{j}",4*i+j,r[4*i+j:4*i+j+1],name) for j in range(4))
            else:
                result.extend((f"{name}_{j}",4*i+2*j,r[4*i+2*j:4*i+2*j+2],name) for j in range(2))
        return result
    result=[]
    for i,name in enumerate(sorted(assets)):
        for j in range(2):
            at=4*i+2*j
            pair_slots=[{"scene_id":f"tier2_pilot_{name}_{j}_{k}","seed":2301000+at+k} for k in (0,1)]
            result.append((f"{name}_{j}",at,pair_slots,name))
    return result


def candidates(stage,payload,key,number,attempt,seed,pools,used,assets):
    if stage in {"pilots","l4"}:
        # For broad baskets in L4 use the compact hand-position candidates in the new atlas.
        return profile_pair(assets[payload],number//2,attempt,seed)
    singles,pairs=pools
    if stage=="l2":
        if key.startswith("multi"):
            return multi_pair(singles,payload,attempt,seed)
        return next(old_pair(pairs,payload,False,used),None)
    bucket,kind=payload
    return l3_pair(pairs,bucket,kind,used,attempt)


def plan_groups(stage,assigns,verified,attempt,pools,used,assets):
    todo=[a for a in assigns if a[0] not in verified]
    if stage in {"pilots","l4"}:
        with ProcessPoolExecutor(max_workers=4,mp_context=get_context("spawn")) as pool:
            futures={pool.submit(profile_pair,assets[payload],number//2,attempt,
                                 rp[0]["seed"]+attempt*701):(key,number,rp)
                     for key,number,rp,payload in todo}
            for task in as_completed(futures):
                yield *futures[task],task.result()
    else:
        for key,number,rp,payload in todo:
            yield key,number,rp,candidates(stage,payload,key,number,attempt,rp[0]["seed"]+attempt*701,
                                         pools,used,assets)


def existing_plans(stage,attempt):
    by_key=defaultdict(list)
    prefix=f"step3a_{stage}_"
    suffix=f"_attempt{attempt}"
    for path in sorted(CASE_PLANS.glob("*.json")):
        num=int(path.stem.rsplit("_",1)[1])
        if not ID_BASE[stage]+40*attempt <= num < ID_BASE[stage]+40*(attempt+1):
            continue
        row=read(path)["case"]
        group=row.get("scene_family","")
        if group.startswith(prefix) and group.endswith(suffix):
            key=group.removeprefix(prefix).removesuffix(suffix)
            by_key[key].append({**row,"scene_signature":scene_signature(row),
                                "layout_signature":layout(row),"plan":str(path)})
    return by_key


def main(stage):
    print(f"{stage}: loading",flush=True)
    if stage == "l2":
        # Start once all three source families of the L2 part pairs are admitted.
        receipt=WORK/"l1_progress.json"
        while not receipt.is_file() or read(receipt)["accepted"]<12:
            time.sleep(15)
    if stage in ("l4","pilots"):
        wait_marker("assets")
        inventory=read(WORK/"asset_inventory.json")
        assets={**inventory["l4_heldout"],**inventory["tier2_proposed"]}
        if stage=="l4":
            assets={k:v for k,v in assets.items() if v["heldout"]}
        else:
            assets={k:v for k,v in assets.items() if not v["heldout"]}
    else:
        assets={}
    pools=plan_bank() if stage in {"l2","l3"} else None
    used,old_ids,old_seeds=previous_layouts()
    assigns=assignments(stage,assets)
    for _,_,rp,_ in assigns:
        assert not {r["scene_id"] for r in rp}&old_ids
        assert not {r["seed"] for r in rp}&old_seeds
    verified={}
    history=[]
    for attempt in range(13):
        previous=existing_plans(stage,attempt)
        batch=list(previous.values())
        group_for={group[0]["scene_family"]:key for key,group in previous.items()}
        used.update(layout(r) for group in batch for r in group)
        preexisting_shards=any((WORK/"sim").glob(f"step3a_01_{stage}_r{attempt}_demo_*"))
        choices=[] if preexisting_shards else [a for a in assigns if a[0] not in previous]
        for key,number,rp,docs in plan_groups(stage,choices,verified,attempt,pools,used,assets):
            if not docs or len(docs)!=len(rp):
                print(stage,key,"not solved attempt",attempt,flush=True)
                continue
            locations={layout(d["case"]) for d in docs}
            if locations & used:
                continue
            used.update(locations)
            work_group=f"step3a_{stage}_{key}_attempt{attempt}"
            group_for[work_group]=key
            members=planned_receipts(docs,rp,ID_BASE[stage]+40*attempt+number,work_group)
            batch.append(members)
            print(stage,key,"prepared",[(r["case_id"],r["part"]) for r in members],flush=True)
        if not batch:
            continue
        result=verify_groups(batch,f"{stage}_r{attempt}")
        grouped=defaultdict(list)
        for row in result["cases"]:
            grouped[group_for[row["scene_family"]]].append(row)
        verified.update(grouped)
        tried=[a for a in result["attempts"]]
        history.extend(tried)
        all_rows=[row for key,_,_,_ in assigns for row in verified.get(key,[])]
        evidence={"stage":stage,"accepted":len(all_rows),"cases":all_rows,"attempts":history,
                  "expected":sum(len(r[2]) for r in assigns),"groups":len(verified),
                  "pending":[r[0] for r in assigns if r[0] not in verified]}
        write_json(WORK/f"{stage}_progress.json",evidence)
        if len(verified)==len(assigns):
            return
    raise RuntimeError(f"{stage} has unresolved groups")


if __name__=="__main__":
    stage=sys.argv[1]
    code=1
    try:
        main(stage)
        code=0
    finally:
        record_complete(RUNS/f"step3a_01_{stage}",code)
