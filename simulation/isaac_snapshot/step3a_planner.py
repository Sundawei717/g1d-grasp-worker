"""Create checked plans on new part atlases and new multi-object layouts."""
from functools import lru_cache
import itertools
import math

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from grasp_regions import *
from school_planner import ARM_NAMES,CHECK_STEP,PAD_OFFSET,Planner,authored_meshes,pose_matrix
from step1b_scenes import Tier1Planner,URDF
from step3a_assets import ROOT,WORK
from part_surface import load_face_atlas as load_atlas
from three_view_detector import sha256,write_json
from three_view_eval import read

CFG=read(ROOT/"env_resolved.json")
EVIDENCE=read(ROOT/"runs/lab_school_03/result.json")
TABLE=float(next(p[:,2].max() for name,p in authored_meshes(CFG["school_usd"]) if name=="/Scene/Table/Top")
            +CFG["school_position_m"][2])


@lru_cache(maxsize=140)
def bounds(path):
    points=np.concatenate([p for _,p in authored_meshes(path)])
    return points.min(0),points.max(0)


def placement(asset,x,y,yaw):
    lo,_=bounds(asset)
    return [float(x),float(y),TABLE-float(lo[2])+.00005,*Rotation.from_euler("z",yaw,degrees=True).as_quat().tolist()]


def fixture(row,tag):
    return {"name":tag,"usd":row["target_usd"],"sha256":sha256(row["target_usd"]),
            "pose":row["target_pose_xyz_xyzw"],"asset_family":row["asset_family"]}


class Step3Planner(Tier1Planner):
    grip=None

    def candidate_goals(self):
        if self.grip is None:
            if self.part in (UPPER_PART,LOWER_PART,MID_PART,TOP_PART):
                yield from Planner.candidate_goals(self)
            else:
                yield from super().candidate_goals()
            return
        anchor,yaw0,_=self.grip
        self.candidate_centers={}
        world_yaw=math.degrees(math.atan2(self.target_pose.rotation[1,0],self.target_pose.rotation[0,0]))
        if self.closing_axis is not None:
            axis=self.target_pose.rotation @ self.closing_axis
            world_yaw=math.degrees(math.atan2(axis[1],axis[0]))-yaw0
        offsets=((0,0,0),(0,0,.007),(0,0,.003),(0,0,-.004),(.012,0,0),(-.012,0,0),
                 (0,.004,.004),(0,-.004,.004),(0,.010,.006),(0,-.010,.006))
        primary=((90,0,0),(90,0,180),(75,0,0),(75,0,180),(90,15,0),(90,-15,0))
        if self.part==BASKET_HANDLE:
            primary=((48,0,0),(48,0,180),(60,0,0),(75,0,0),*primary)
        if self.angle_search=="wide":
            tilts=(48,60,75,90) if self.grip_asset in {"019_pitcher_base","arched_basket"} else (90,75,60,48)
            basics=tuple(itertools.product(tilts,(0,-15,15),(0,180)))
            angles=tuple(itertools.product(tilts,(-30,30,-60,60),(0,180)))
            orientations=itertools.chain(
                itertools.product(basics,offsets[:1]),
                itertools.product(primary,offsets[1:4]),
                itertools.product(angles,offsets[:2]),
                itertools.product(primary,offsets[4:]))
        else:
            orientations=itertools.chain(itertools.product(primary,offsets),
                itertools.product(itertools.product((90,75,60,48),(30,-30,60,-60,90),(0,180)),offsets))
        anchors=(anchor, [anchor[0],-anchor[1],anchor[2]]) if self.part==BASKET_RIM else (anchor,)
        for ((tilt,skew,back),delta),center in itertools.product(orientations,anchors):
            local=np.array(center)+delta
            rot=(Rotation.from_euler("z",world_yaw+yaw0+skew+back,degrees=True)
                 * Rotation.from_euler("x",-tilt,degrees=True)*Rotation.from_euler("z",90,degrees=True))
            orientation=rot.as_matrix()
            wrist=self.target_pose.act(local)-rot.apply(PAD_OFFSET)
            pre=wrist-orientation[:,0]*.065
            code="p"+",".join(f"{c:.4f}" for c in local)+f"_t{tilt}_s{skew}_r{back}"
            self.candidate_centers[code]=local.tolist()
            yield code,pin.SE3(orientation,pre),pin.SE3(orientation,wrist)

    def screen_lift(self, initial, lift, wrist):
        if self.grip is None:
            if self.part in (UPPER_PART,LOWER_PART,MID_PART,TOP_PART):
                return Planner.screen_lift(self,initial,lift,wrist)
            return super().screen_lift(initial,lift,wrist)
        gap=float(np.clip(self.grip[2],.010,.083))
        self.planned_gap_m=gap
        problem=self.clearance_with_gap(initial,lift,wrist,gap)
        # A guessed closing width can place the projected middle link into the target.
        # Test a wider plausible contact before the measured-finger runtime guard.
        if problem and self.alternate_gaps and any(k.startswith("target:") for k in problem) and any(
                k.startswith("right_dex1_finger_link_") for k in problem):
            for width in (min(gap+.004,.087),min(gap+.008,.087)):
                if self.clearance_with_gap(initial,lift,wrist,width) is None:
                    self.planned_gap_m=width
                    return None
        return problem

    def clearance_with_gap(self,initial,lift,wrist,gap):
        closed=np.full(2,-.02+(.09007-gap)/2)
        relative=wrist.inverse()*self.target_pose
        for opening in np.linspace(self.q_ref[self.fingers],closed,12):
            if hit:=self.collision(initial,allow_pad_target=True,fingers=opening):
                return hit
        for first,second in zip([initial,*lift],lift):
            n=max(1,math.ceil(np.linalg.norm(second-first,ord=1)/CHECK_STEP))
            for amount in np.linspace(0,1,n+1)[1:]:
                q=first+amount*(second-first)
                if hit:=self.collision(q,allow_pad_target=True,held_pose=self.world_frame(q)*relative,fingers=closed):
                    return hit
        return None


def build_planner(row):
    cfg={**CFG,"target_usd":row["target_usd"]}
    hint=row.get("planner_hint",{})
    p=Step3Planner(cfg,URDF,EVIDENCE,row["target_pose_xyz_xyzw"],home=row["home_q"],part=row["part"],
                   obstacles=row["fixtures"],expanded_search=True,guide=hint.get("guide",()))
    p.grip=row.get("grip_geometry")
    p.closing_axis=row.get("closing_axis")
    p.angle_search=row.get("angle_search")
    p.grip_asset=row.get("grip_asset")
    p.alternate_gaps=row.get("check_alternate_gaps",False)
    p.rim_z=row.get("part_meshes",{}).get("rim_center_z_m",.046)
    p.detour=None
    p.obstacle_centers={f["name"]:f["pose"][:3] for f in row["fixtures"]}
    if row.get("approach_width_m") is not None:
        p.q_ref[p.fingers]=-.02+(.09007-row["approach_width_m"])/2
    return p


def solve(row,max_candidates=110):
    p=build_planner(row)
    hint=row.get("planner_hint",{})
    fit=p.plan(seed=row["seed"],stop_after=1,max_candidates=max_candidates,
               preferred=hint.get("preferred",()),lift_height_m=.08)
    if not fit["feasible"]:
        return None
    if p.grip is not None:
        mesh,names=load_atlas(row["part_meshes"]["face_atlas"])
        part_id=next(int(i) for i,name in row["part_meshes"]["face_names"].items() if name==row["part"])
        surface=mesh.triangles_center[names==part_id]
        center=p.candidate_centers[fit["feasible"][0]["candidate"]]
        anchors={**row["part_meshes"]["part_anchors"],
                 row["part"]:surface[np.argmin(np.linalg.norm(surface-center,axis=1))].tolist()}
        row={**row,"part_meshes":{**row["part_meshes"],"part_anchors":anchors}}
    return {**fit,"case":row,"obstacles":row["fixtures"],"home_q":row["home_q"],
            "right_joint_names":list(ARM_NAMES),"target_pose_xyz_xyzw":row["target_pose_xyz_xyzw"],
            "school_sha256":sha256(CFG["school_usd"]),"target_sha256":sha256(row["target_usd"]),
            "urdf_sha256":sha256(URDF),"collision_checks":p.checks,"detour":p.detour}


def adapt(doc,row,*,block=False):
    """Validate a stored grasp and solve a new route when the transformed fixtures obstruct it."""
    row={**row,"planner_hint":{}}
    p=build_planner(row)
    rowfit=doc["feasible"][0]
    start=np.asarray(row["home_q"])
    approach=np.asarray(rowfit["approach_q"])
    hover=approach[0]
    pre=np.asarray(rowfit["pre_grasp_q"])
    if p.collision(start) or p.collision(hover):
        return None
    direct=p.edge(start,hover)
    if block and (not direct or not any(name.startswith("blocker:") for name in direct)):
        return None
    if any(p.edge(a,b) for a,b in zip(approach,approach[1:])):
        return None
    if p.screen_lift(approach[-1],np.asarray(rowfit["trial_lift_ik_only_q"]),pose_matrix(rowfit["wrist_xyz_xyzw"])):
        return None
    cached=np.concatenate((start[None],pre[1:]))
    if not block and not any(p.edge(a,b) for a,b in zip(cached,cached[1:])):
        route=cached
    elif not direct:
        route=(start,hover)
    else:
        route,_=p.route(start,hover,np.random.default_rng(row["seed"]))
        if route is None:
            return None
    updated={**rowfit,"pre_grasp_q":[q.tolist() for q in route]}
    return {**doc,"home_q":row["home_q"],"case":row,"feasible":[updated],"detour":p.detour or doc.get("detour"),
            "obstacles":row["fixtures"]}


def authored_panel():
    from pxr import Gf,Usd,UsdGeom,UsdPhysics,Vt
    import trimesh
    p=WORK/"new_red_panel.usda"
    if not p.is_file():
        s=Usd.Stage.CreateNew(str(p))
        UsdGeom.SetStageMetersPerUnit(s,1)
        UsdGeom.SetStageUpAxis(s,UsdGeom.Tokens.z)
        root=UsdGeom.Xform.Define(s,"/root").GetPrim()
        s.SetDefaultPrim(root)
        UsdPhysics.RigidBodyAPI.Apply(root)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.126)
        m=trimesh.creation.box((.075,.023,.11)).apply_translation([0,0,.055])
        geom=UsdGeom.Mesh.Define(s,"/root/Panel")
        geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(m.vertices.astype(np.float32)))
        geom.CreateFaceVertexCountsAttr([3]*len(m.faces))
        geom.CreateFaceVertexIndicesAttr(m.faces.ravel().tolist())
        geom.CreateDisplayColorAttr([Gf.Vec3f(.55,.085,.054)])
        geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexHull")
        s.GetRootLayer().Save()
    return str(p)


def asset_case(item,part,index,seed,*,obstacle=False):
    asset=item["asset"]
    name=item["name"]
    family={"white_mug":"mug/00004","odd_handle_mug":"l4/odd_handle_mug",
            "round_basket":"l4/round_basket","arched_basket":"basket/arched_train",
            "raised_rim_basket":"basket/raised_rim_train","marker_long_box":"marker_box/00003",
            "compact_box":"sand3d_packaging/00000","round_package":"sand3d_packaging/00225",
            "matte_soda":"soda_can/00000"}.get(name,"ycb/"+name)
    role=("mug" if "mug" in name else "basket" if "basket" in name else
          "can" if name in {"round_package","matte_soda"} else "carton")
    yaw=(0,-22,20,-42,40,-55,62,-13,12,-70,71)[index%11]
    rng=np.random.default_rng(seed)
    xx=.181 + rng.uniform(-.027,.027)
    yy=-.443 + rng.uniform(-.027,.022)
    if name=="arched_basket" and index>=39:
        yaw=(-24,-12,0,12,24)[index%5]
        xx=.211+rng.uniform(-.019,.019)
        yy=-.411+rng.uniform(-.018,.018)
    if name in {"004_sugar_box","006_mustard_bottle","011_banana"} and index>=26:
        # Reach both long-axis ends with a reachable gripper yaw and neutral elbow.
        yaw=(62,69,76,84,92)[index%5]
        xx=.183 + rng.uniform(-.027,.027)
        yy=-.416 + rng.uniform(-.021,.019)
    if name in {"marker_long_box","compact_box","round_package","matte_soda","raised_rim_basket"}:
        yaw=((-18,0,18,29)[index%4] if name=="raised_rim_basket" else (63,72,81,89)[index%4])
        xx=.189+rng.uniform(-.025,.025)
        yy=-.421+rng.uniform(-.025,.020)
    home=np.array(EVIDENCE["samples"][0]["right_rad"])+rng.uniform(-.015,.015,7)
    home[3]=max(-1.018,home[3])
    obj={"target_usd":asset,"target_sha256":item["asset_sha256"],"asset_family":family,"asset_role":role,"bucket":role,
         "target_pose_xyz_xyzw":placement(asset,xx,yy,yaw),"part":part,"group":"C_start","seed":seed,
         "yaw_deg":yaw,"command_delay_substeps":0,"home_q":home.tolist(),"fixtures":[],"part_meshes":item["part_meshes"],
         "grip_geometry":item["grips"][part],"grip_goal_m":.002,"base_mesh_hash":item["base_mesh_hash"],
         "config_sha256":sha256(ROOT/"env_resolved.json"),
         "target_query":("can" if name=="007_tuna_fish_can" else "bowl" if name=="024_bowl" else
                         "bottle" if name=="006_mustard_bottle" else "banana" if name=="011_banana" else
                         "pitcher" if name=="019_pitcher_base" else "drill" if name=="035_power_drill" else
                         "box" if role=="carton" else role)}
    center,yaw_axis,gap=item["grips"][part]
    lo,hi=np.array(item["bounds"])
    if name=="008_pudding_box":
        # Stand the box on a long, narrow edge. The two pads enclose its package thickness.
        rotation=Rotation.from_euler("z",yaw,degrees=True)*Rotation.from_euler("y",90,degrees=True)
        obj["target_pose_xyz_xyzw"]=[xx,yy,TABLE+hi[0]+.00005,*rotation.as_quat().tolist()]
        center=[lo[0]+.014,center[1],float((lo[2]+hi[2])/2)]
        gap=float(hi[2]-lo[2])
        obj["closing_axis"]=[0,0,1]
    elif name=="007_tuna_fish_can":
        # Use the packaging's thickness with the cylinder set on its edge.
        # Benchmarks keep the original YCB mesh and verify its final settled pose.
        yaw=67+(index%6)*4
        rotation=Rotation.from_euler("z",yaw,degrees=True)*Rotation.from_euler("y",90,degrees=True)
        xx=(.13,.226,.176,.243)[index%4]+rng.uniform(-.014,.014)
        yy=-.423+rng.uniform(-.018,.023)
        obj["target_pose_xyz_xyzw"]=[xx,yy,TABLE+hi[0]+.00005,*rotation.as_quat().tolist()]
        obj["yaw_deg"]=yaw
        center=[lo[0]+.009,0,float((lo[2]+hi[2])/2)]
        gap=float(hi[2]-lo[2])
        obj["closing_axis"]=[0,0,1]
    elif name=="004_sugar_box":
        side=-1 if part==BOX_NEG_END else 1
        center=[0,side*.043,.080]
        receipt=WORK/"sugar_support_pose.json"
        if receipt.is_file():
            stable=read(receipt)
            if stable["asset_sha256"]!=item["asset_sha256"]:
                raise ValueError("Sugar box support calibration belongs to another mesh")
            origin=obj["target_pose_xyz_xyzw"]
            yaw_rotation=Rotation.from_quat(origin[3:])
            xyz=np.array(origin[:3])+yaw_rotation.apply(stable["local_delta_m"])
            orientation=yaw_rotation*Rotation.from_quat(stable["local_delta_quat"])
            obj["target_pose_xyz_xyzw"]=[*xyz.tolist(),*orientation.as_quat().tolist()]
            obj["support_pose_source"]=stable["source"]
    elif name=="006_mustard_bottle":
        # The shoulder tapers. Keep both pads on the straighter belt below it.
        side=-1 if part==BOX_NEG_END else 1
        center,yaw_axis,gap=[0,side*.029,.083],0,.052
    elif name=="011_banana":
        center=[center[0],center[1],.027]
    elif name=="025_mug" and part==MUG_BODY:
        side=(-1,1)[index%2]
        center,yaw_axis,gap=[-.028,side*.042,.078],90,.012
    elif name in {"round_package","matte_soda"}:
        local=Rotation.from_euler("z",-yaw,degrees=True)
        heading=math.radians((72,79,86)[index%3])
        obj["closing_axis"]=local.apply([math.cos(heading),math.sin(heading),0]).tolist()
    elif name=="024_bowl":
        side=-1 if part==BOX_NEG_END else 1
        center,yaw_axis,gap=[(-1 if index%2==0 else 1)*.008,side*.070,.046],90,.018
    elif name=="019_pitcher_base":
        side=-1 if part==BOX_NEG_END else 1
        center,yaw_axis,gap=[-.008,side*.060,.233],90,.014
    elif name=="035_power_drill":
        side=-1 if part==BOX_NEG_END else 1
        center,yaw_axis,gap=[.022,side*.066,.037],90,(.039 if side<0 else .058)
    elif name=="round_basket" and part==BASKET_RIM:
        center=[0,-.046,.047]
    elif name=="arched_basket" and part==BASKET_RIM:
        center=[(-1 if index%2==0 else 1)*.024,-.050,.052]
    obj["grip_geometry"]=[center,yaw_axis,gap]
    if index>=39 and name in {"019_pitcher_base","011_banana","035_power_drill",
                               "006_mustard_bottle","arched_basket","024_bowl"}:
        obj.update(angle_search="wide",grip_asset=name)
    obj["approach_width_m"]=.036 if name=="arched_basket" and part==BASKET_RIM else float(np.clip(gap+.025,.044,.09007))
    return obj
