import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from school_planner import authored_meshes, pose_matrix
from step1b_scenes import PAIR_PARTS, URDF, Tier1Planner, make_plan
from three_view_detector import sha256, write_json
from tier1_collect import QUOTAS, ROOT, RUNS, read, scene_signature, write_candidate


PILOTS = {"mug": ("step1_mug_plan_lift8", "step1b_plan_mug_pairs", "step1b_plan_mug_plus55", "step1b_plan_mug_plus75"),
          "basket": ("step1b_basket_verified_plan",), "carton": ("step1b_plan_carton_pairs_v2",)}
START = {"mug": 7300, "basket": 8200, "carton": 9000}
LIMITS = {"mug": 450, "basket": 400, "carton": 400}
ALLOC = {"mug": 5, "basket": 5, "carton": 4}
DEV_OBSTACLE_PAIRS = {"mug": 2, "basket": 2, "carton": 1}


def references(bucket):
    bank = {part: [] for part in PAIR_PARTS[bucket]}
    for directory in PILOTS[bucket]:
        root = RUNS / directory
        for path in sorted(root.glob("C_*.json")):
            doc = read(path)
            if not doc.get("feasible"):
                continue
            case = doc["case"]
            if case["part"] in bank and not case["fixtures"]:
                bank[case["part"]].append((case, doc["feasible"][0]))
    if any(not refs for refs in bank.values()):
        raise ValueError(f"No pinned calibration plan for {bucket}")
    return bank


def base_mesh_signature(path):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage=Usd.Stage.Open(str(path))
    transforms=UsdGeom.XformCache()
    vertices=[]
    for prim in stage.Traverse():
        if (prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI)
                and UsdGeom.Imageable(prim).ComputeVisibility()!=UsdGeom.Tokens.invisible):
            matrix=transforms.GetLocalToWorldTransform(prim)
            points=UsdGeom.Mesh(prim).GetPointsAttr().Get()
            vertices.extend(tuple(matrix.Transform(p)) for p in points)
    if not vertices:
        raise ValueError(f"Missing original visual mesh: {path}")
    world=np.asarray(vertices)
    xyz=np.rint((world-world.min(0))/np.maximum(np.ptp(world,axis=0),1e-9)*100000).astype("<i4")
    return hashlib.sha256(np.unique(xyz,axis=0).tobytes()).hexdigest()


def forbidden(work):
    signatures = set()
    for path in RUNS.glob("**/frozen.json"):
        doc = read(path)
        for row in doc.get("cases", []):
            try:
                signatures.add(scene_signature(row))
            except KeyError:
                source = path.parent / (row.get("plan") or f"{row.get('case_id')}.json")
                if source.is_file():
                    p = read(source)
                    if "case" in p:
                        signatures.add(scene_signature(p["case"]))
    base_assets = {role:ROOT/"assets"/role/"object.usdz" for role in
                   ("mug/00007", "sand3d_packaging/00152", "sand3d_packaging/00156", "sand3d_packaging/00208")}
    base_assets["basket/00001"] = Path(read(RUNS/"step1b_asset_highrail/basket.json")["asset"])
    heldout_paths = {"mug/00004":ROOT/"assets/mug/00004/object.usdz"}
    training_meshes = {key:base_mesh_signature(path) for key,path in base_assets.items()}
    heldout_meshes = {key:base_mesh_signature(path) for key,path in heldout_paths.items() if path.exists()}
    if set(training_meshes.values()) & set(heldout_meshes.values()):
        raise ValueError("A reserved L4 base mesh is already in the Tier-1 pool")
    freeze = {"status": "reserved", "count": 80, "heldout_asset_families":
              ["mug/00004", "unseen_mug_base", "unseen_basket_base", "008_pudding_box", "007_tuna_fish_can"],
              "training_base_mesh_signatures":training_meshes,"reserved_base_mesh_signatures":heldout_meshes,
              "unseen_basket_mesh_selected": False,
              "slots": [{"scene_id": f"final_L{layer}_{i:02}", "seed": 2200000+100*layer+i,
                         "tier": layer} for layer in range(1,5) for i in range(20)],
              "existing_frozen_signatures": sorted(signatures)}
    write_json(work / "final_test_reservation.json", freeze)
    return signatures


def fixture_bank(config):
    candidates = [ROOT / "assets/sand3d_packaging/00156/object.usdz",
                  ROOT / "assets/soda_can/00000/object.usdz", RUNS / "step1b_asset_highrail/green_blocker.usda",
                  ROOT / "assets/handled_paint/00000/object.usdz"]
    table = next(points for name, points in authored_meshes(config["school_usd"]) if name == "/Scene/Table/Top")
    surface = float(table[:, 2].max() + config["school_position_m"][2])
    result = []
    for asset in candidates:
        points = np.concatenate([v for _, v in authored_meshes(asset)])
        result.append((str(asset), sha256(asset), surface-float(points[:,2].min())+.00005,
                       (points.max(0)-points.min(0))[:2]/2))
    return result


def candidate(bucket, index, catalog, fixtures, config, *, dev=False, dev_index=0):
    role_index = list(START).index(bucket)
    lane = (index+role_index) % 3
    seed = (1850000 if dev else 1000000) + lane*150000 + role_index*10000 + index
    if dev:
        seed = 1800000+dev_index*100+index
    generator = np.random.default_rng(seed)
    parts = tuple(catalog)
    angles = (-75, -55, -35, -15, 0, 15, 35, 55, 75) if bucket != "carton" else (-85, 85, -65, 65, -50, 50)
    yaw = int(angles[(index+2*role_index)%len(angles)])
    if index >= len(angles):
        jitter = int(generator.integers(-8, 9))
        yaw = int(np.clip(yaw+jitter, -75 if bucket != "carton" else -90, 75 if bucket != "carton" else 90))
    sources = {part: min(catalog[part], key=lambda item: abs(item[0]["yaw_deg"]-yaw)) for part in parts}
    baseline = sources[parts[0]][0]
    nearby = [r for r,_ in catalog[parts[0]] if abs(r["yaw_deg"]-yaw) <= 12]
    if index >= len(angles) and nearby:
        baseline = nearby[int(generator.integers(len(nearby)))];
    x, y = np.asarray(baseline["target_pose_xyz_xyzw"][:2])+generator.uniform(-.013,.013,2)
    x, y = float(x), float(y)
    pose = [x, y, baseline["target_pose_xyz_xyzw"][2], *Rotation.from_euler("z", yaw, degrees=True).as_quat()]
    home = np.array(baseline["home_q"])+generator.uniform(-1,1,7)*[.10,.06,.08,.06,.065,.06,.06]
    home[1] = -abs(home[1])
    home[3] = max(-1.02, home[3])
    scene = f"tier1_{'dev' if dev else 'train'}_{bucket}_{index:04}"
    obstacle = []
    if index % 3 == (1 if dev else 2):
        usd, digest, z, half_width = fixtures[(index//3+role_index) % len(fixtures)]
        # The unchecked straight segment must hit a fixture in each selected obstruction plan.
        # Sample the fore-edge corridor independently of yaw sign and check the exact geometry below.
        relx = float(generator.uniform(-.125,-.060) if index%2 else generator.uniform(-.060,.035))
        rely = float(generator.uniform(-.143,-.104))
        if bucket == "carton":
            rely -= .012
        bx = float(np.clip(x+relx, .040+half_width[0], .31-half_width[0]))
        by = max(-.55, y+rely)
        rz = int(generator.integers(-35, 36))
        obstacle = [{"name": "blocker", "usd": usd, "sha256": digest,
                     "pose": [bx, by, z, *Rotation.from_euler("z", rz, degrees=True).as_quat()]}]
    group = "B_obstacle" if obstacle else "C_start"
    group_id = f"{scene}_pair"
    for j, part in enumerate(parts):
        source, fit = sources[part]
        number = (9800+2*dev_index if dev else START[bucket]+2*index)+j
        key = f"{group[0]}_{number}"
        tips = {"preferred": [fit["candidate"]], "guide": [fit["approach_q"][0]], "max_candidates": 72}
        definition = dict(source, case_id=key, scene_id=f"{scene}_{part}", episode_id=f"tier1_{key}",
                          scene_family=group_id, pair_group_id=group_id, split="dev_eval" if dev else "train",
                          seed=seed, seed_lane=lane, group=group, target_pose_xyz_xyzw=pose,
                          home_q=home.tolist(), fixtures=obstacle, yaw_deg=yaw, order=index, planner_hint=tips,
                          config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())
        if bucket == "basket":
            definition["approach_width_m"] = .052 if j==0 else .044
        yield definition


def pair_with_obstacle(cfg, evidence, definitions, clean):
    block = definitions[0]["fixtures"][0]
    seed = definitions[0]["seed"]
    rng = np.random.default_rng(seed)
    mesh = np.concatenate([points for _, points in authored_meshes(block["usd"])])
    xy_radius = np.max(np.abs(mesh[:, :2]), axis=0)
    max_z = block["pose"][2] + mesh[:,2].max()
    target = np.concatenate([points for _, points in authored_meshes(cfg["target_usd"])])
    location = pose_matrix(definitions[0]["target_pose_xyz_xyzw"])
    world = target @ location.rotation.T + location.translation
    target_lo, target_hi = world[:,:2].min(0), world[:,:2].max(0)
    home = np.array(definitions[0]["home_q"])
    checked = []
    locations = []
    for case, doc in zip(definitions, clean, strict=True):
        fit = doc["feasible"][0]
        hover = np.array(fit["approach_q"][0])
        g = Tier1Planner(cfg,URDF,evidence,case["target_pose_xyz_xyzw"],home=home,part=case["part"],obstacles=[block])
        width = case.get("approach_width_m",.09007)
        g.q_ref[g.fingers] = -.02 + (.09007-width)/2
        g.rim_z = case.get("part_meshes",{}).get("rim_center_z_m",.046)
        indices = [i for i,k in enumerate(g.fixture_names) if k.startswith("blocker:")]
        if not indices:
            raise ValueError("Obstacle has no physical collider")
        checked.append((g,fit,hover,indices))
        for fraction in (.81,.72,.87,.64,.92):
            wrist = g.world_frame(home+fraction*(hover-home))
            for offset in (0,-width/2,width/2):
                point = wrist.act(np.array([.124,offset,0.]))
                if point[2] > max_z + .01:
                    continue
                for shift in ((0,0),(-.012,-.015),(.012,-.015),(0,-.024)):
                    xy=point[:2]+shift
                    if not .035<xy[0]<.32 or not -.56<xy[1]<-.40:
                        continue
                    if np.all(xy+xy_radius+.008>=target_lo) and np.all(xy-xy_radius-.008<=target_hi):
                        continue
                    rounded=(round(float(xy[0]),3),round(float(xy[1]),3))
                    if rounded not in locations:
                        locations.append(rounded)
    for x,y in locations[:30]:
        rz=int(rng.integers(-22,23))
        fixture_pose=[x,y,block["pose"][2],*Rotation.from_euler("z",rz,degrees=True).as_quat()]
        transform=pose_matrix(fixture_pose)
        revised=[]
        for g,fit,hover,indices in checked:
            for i in indices:
                g.fixture_poses[i]=transform
            g.obstacle_centers={"blocker":fixture_pose[:3]}
            g.detour=None
            if g.collision(home) or g.collision(hover):
                break
            direct=g.edge(home,hover)
            if direct and not any(k.startswith("blocker:") for k in direct):
                break
            approach=np.asarray(fit["approach_q"])
            if any(g.edge(a,b) for a,b in zip(approach,approach[1:])):
                break
            if g.screen_lift(approach[-1],np.asarray(fit["trial_lift_ik_only_q"]),pose_matrix(fit["wrist_xyz_xyzw"])):
                break
            route=[home,hover] if not direct else g.route(home,hover,rng)[0]
            if route is None:
                break
            revised.append(({**fit,"pre_grasp_q":[q.tolist() for q in route]},g.detour))
        if len(revised)==2 and any(detour is not None for _,detour in revised):
            obstacle={**block,"pose":fixture_pose}
            return [{**doc,"case":dict(case,fixtures=[obstacle]),"obstacles":[obstacle],"feasible":[fit],"detour":detour}
                    for doc,case,(fit,detour) in zip(clean,definitions,revised,strict=True)]
    return []


def solve_pair(cfg, evidence, definitions):
    result = []
    for row in definitions:
        doc = make_plan(cfg, evidence, {**row,"fixtures":[]})
        if not doc["feasible"]:
            return []
        result.append(doc)
    return pair_with_obstacle(cfg,evidence,definitions,result) if definitions[0]["fixtures"] else result


def stream(pool, pending, tasks, limit):
    tasks = iter(tasks)
    while True:
        while len(pending) < limit:
            task = next(tasks, None)
            if task is None:
                break
            key, cfg, e, definitions = task
            pending[pool.submit(solve_pair, cfg, e, definitions)] = key
        if not pending:
            return
        finished, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in finished:
            yield pending.pop(future), future.result()


def plan(work, workers):
    for name in ("plans", "dev_plans", "planning"):
        (work / name).mkdir(parents=True, exist_ok=True)
    config = read(ROOT / "env_resolved.json")
    evidence = read(RUNS / "lab_school_03/result.json")
    holdout = forbidden(work)
    models = {bucket: references(bucket) for bucket in START}
    versions = {bucket: {**config, "target_usd": next(iter(bank.values()))[0][0]["target_usd"]} for bucket,bank in models.items()}
    fixtures = fixture_bank(config)
    done = set()
    for p in (work/"planning").glob("*.json"):
        row = read(p)
        dev = p.stem.startswith("dev_")
        index = int(p.stem.rsplit("_",1)[-1])
        if row["accepted"] or row.get("revision")==2 or index%3 != (1 if dev else 2):
            done.add(p.stem)
    reserved = {row["scene_signature"] for path in (work / "dev_plans").glob("*/ready.json") for row in [read(path)]}
    planned = {row["scene_signature"] for path in (work / "plans").glob("*/ready.json") for row in [read(path)]}
    dev_counts = Counter(read(path)["bucket"] for path in (work / "dev_plans").glob("*/ready.json"))
    dev_blocks = Counter(read(path)["bucket"] for path in (work / "dev_plans").glob("*/ready.json")
                         if read(path)["fixtures"])
    dev_rows = defaultdict(list)
    for path in sorted((work/"dev_plans").glob("*/ready.json")):
        row=read(path)
        dev_rows[row["bucket"]].append(row)

    def freeze_dev():
        selected=[]
        for bucket,rows in dev_rows.items():
            blocks=[row for row in rows if row["fixtures"]]
            neutral=[row for row in rows if not row["fixtures"]]
            selected.extend(blocks[:2*DEV_OBSTACLE_PAIRS[bucket]]+neutral[:2*(ALLOC[bucket]-DEV_OBSTACLE_PAIRS[bucket])])
        if len(selected)==28 and all(dev_blocks[b]>=2*n for b,n in DEV_OBSTACLE_PAIRS.items()):
            write_json(work/"development.json",{"phase":"step2_dev_frozen","cases":selected,"count":28,
                       "asset_counts":Counter(row["bucket"] for row in selected),"replay_verified":False})
            return True
        return False

    def tasks():
        for index in range(max(LIMITS.values())):
            progress = read(work / "progress.json") if (work / "progress.json").exists() else {}
            if (work / "collection.json").exists():
                break
            for bucket in START:
                want_dev = (dev_blocks[bucket] < 2*DEV_OBSTACLE_PAIRS[bucket] if index%3==1 else dev_counts[bucket] < 2*ALLOC[bucket])
                if index < 90 and want_dev:
                    # Dev tasks draw separate physical placements and seeds. IDs are assigned at admission.
                    name = f"dev_{bucket}_{index:04}"
                    if name not in done:
                        rows = list(candidate(bucket, index, models[bucket], fixtures, versions[bucket],
                                              dev=True, dev_index=28+list(START).index(bucket)*94+index))
                        yield (name,bucket,True),versions[bucket],evidence,rows
                if index >= LIMITS[bucket] or progress.get("accepted",{}).get(bucket,0) >= QUOTAS[bucket]:
                    continue
                name = f"train_{bucket}_{index:04}"
                if name not in done:
                    rows = list(candidate(bucket, index, models[bucket], fixtures, versions[bucket]))
                    yield (name,bucket,False),versions[bucket],evidence,rows

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (name,bucket,dev), docs in stream(pool, {}, tasks(), 2*workers):
            signatures = {scene_signature(d["case"]) for d in docs}
            ok = len(docs)==2 and not (signatures & (holdout|reserved|planned))
            if ok and dev:
                limit = 2*DEV_OBSTACLE_PAIRS[bucket] if docs[0]["obstacles"] else 2*ALLOC[bucket]
                used = dev_blocks[bucket] if docs[0]["obstacles"] else dev_counts[bucket]-dev_blocks[bucket]
                ok = used < limit
            if ok and dev:
                after = sum(dev_counts.values())
                for i,d in enumerate(docs):
                    row = d["case"]
                    key = f"{row['group'][0]}_{9800+after+i}"
                    row.update(case_id=key, episode_id=f"tier1_dev_{key}")
                    place = work/"dev_plans"/key
                    place.mkdir(exist_ok=True)
                    write_json(place/"plan.json",d)
                    entry=dict(row,bucket=bucket,scene_signature=scene_signature(row),plan=str((place/"plan.json").resolve()))
                    write_json(place/"ready.json",entry)
                    dev_rows[bucket].append(entry)
                dev_counts[bucket]+=2
                if docs[0]["obstacles"]:
                    dev_blocks[bucket]+=2
                reserved.update(signatures)
            elif ok:
                for d in docs:
                    write_candidate(work, bucket, d)
                planned.update(signatures)
            write_json(work/"planning"/f"{name}.json",{"accepted":ok,"bucket":bucket,"dev":dev,"revision":2})
            if ok and dev:
                freeze_dev()
            if ok:
                print(name, [d["case"]["case_id"] for d in docs], flush=True)
    write_json(work/"planning_exhausted.json",{"passed":freeze_dev(),"dev":dict(dev_counts),"dev_obstacles":dict(dev_blocks)})


if __name__ == "__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("output",type=Path)
    cli.add_argument("--workers",type=int,default=24)
    args=cli.parse_args()
    plan(args.output.resolve(),args.workers)
