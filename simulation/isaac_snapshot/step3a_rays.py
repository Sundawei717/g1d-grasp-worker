"""Certify apertures against original mesh silhouettes and PhysX cooked colliders."""
import os
import resource
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.ndimage import binary_fill_holes

from background_status import record_complete
from part_surface import load_face_atlas
from step3a_resources import SimAllocator
from three_view_detector import write_json
from three_view_eval import read

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/"runs"
WORK=RUNS/"step3a_01"
LOCAL="--local-substitutes" in sys.argv
JOB="step3a_01_substitute_rays" if LOCAL else "step3a_01_rays"
REQUIRED=(("raised_rim_basket",) if LOCAL else
          ("white_mug","odd_handle_mug","round_basket","arched_basket","025_mug","019_pitcher_base"))


def ray_grid(asset):
    mesh,_=load_face_atlas(asset["part_meshes"]["face_atlas"])
    import trimesh

    # Simplification is only used to choose candidate aperture rays from the original surface silhouette.
    if len(mesh.faces)>70000:
        _,idx,inv=np.unique(np.rint(mesh.vertices/.00045).astype(np.int32),axis=0,return_index=True,return_inverse=True)
        f=inv[mesh.faces]
        keep=(f[:,0]!=f[:,1])&(f[:,1]!=f[:,2])&(f[:,0]!=f[:,2])
        mesh=trimesh.Trimesh(mesh.vertices[idx],f[keep],process=True)
    hole,solid=[],[]
    from scipy.spatial.transform import Rotation
    degrees=(0,45,90,135) if asset["name"]=="019_pitcher_base" else (0,90)
    for angle in degrees:
        rotation=Rotation.from_euler("z",angle,degrees=True)
        projected=trimesh.Trimesh(rotation.inv().apply(mesh.vertices),mesh.faces,process=False)
        lo,hi=projected.bounds
        dimension,column=0,1
        coords=np.meshgrid(np.linspace(lo[column]-.001,hi[column]+.001,31),
                           np.linspace(lo[2]-.001,hi[2]+.001,35))
        points=np.zeros((31*35,3))
        points[:,dimension]=lo[dimension]-.02
        points[:,column]=coords[0].ravel()
        points[:,2]=coords[1].ravel()
        direction=np.eye(3)[dimension]
        hits=projected.ray.intersects_any(points,np.tile(direction,(len(points),1)))
        interior=binary_fill_holes(hits.reshape(coords[0].shape)).ravel() & ~hits
        for kind,new,mask in (("hole",hole,interior),("solid",solid,hits)):
            ids=np.flatnonzero(mask)
            if len(ids):
                ids=ids[np.linspace(0,len(ids)-1,min(32,len(ids)),dtype=int)]
            new.extend((rotation.apply(points[i]).tolist(),rotation.apply(direction).tolist(),
                        float(hi[dimension]-lo[dimension]+.04)) for i in ids)
    return hole,solid


def check():
    from tempfile import TemporaryDirectory
    from isaaclab.app import AppLauncher
    from projects.g1d_rsi.sim60_legacy_bridge import install_import_bridge
    install_import_bridge()
    cfg=read(ROOT/"env_resolved.json")
    inv=read(WORK/"asset_inventory.json")
    assets=(read(WORK/"asset_substitutes.json")["assets"] if LOCAL else
            {**inv["l4_heldout"],**inv["tier2_proposed"]})
    with TemporaryDirectory(prefix="step3a_rays_",dir="/dev/shm") as kit:
        active=os.environ.get("G1D_RENDER_GPU","0")
        app=AppLauncher(headless=True,enable_cameras=True,device="cuda:0",experience=cfg["experience"],
                        kit_args=f"--portable-root {kit} --/renderer/multiGpu/autoEnable=false "
                        f"--/renderer/multiGpu/enabled=false --/renderer/activeGpu={active}").app
        try:
            import omni.usd
            import carb
            from isaaclab.assets import RigidObject,RigidObjectCfg
            from isaaclab.sim import SimulationCfg,SimulationContext
            from omni.physx import get_physx_scene_query_interface
            from projects.g1d_rsi.sim60_legacy_bridge import bind_lab3_core_backend
            from pxr import UsdGeom,Gf,UsdPhysics

            probes={k:ray_grid(assets[k]) for k in REQUIRED}

            bind_lab3_core_backend("cuda:0")
            sim=SimulationContext(SimulationCfg(device="cuda:0",dt=1/600,enable_scene_query_support=True))
            carb.settings.get_settings().set_bool("/physics/enableSceneQuerySupport",True)
            stage=omni.usd.get_context().get_stage()
            targets={}
            for i,k in enumerate(REQUIRED):
                prefix=f"/World/Probe{i:02}"
                placement=UsdGeom.Xform.Define(stage,prefix)
                UsdGeom.XformCommonAPI(placement).SetTranslate(Gf.Vec3d(float(i)*.65,0,1))
                body=UsdGeom.Xform.Define(stage,prefix+"/Body")
                body.GetPrim().GetReferences().AddReference(assets[k]["asset"])
                UsdPhysics.RigidBodyAPI(body.GetPrim()).CreateKinematicEnabledAttr().Set(True)
                targets[k]=(prefix,RigidObject(RigidObjectCfg(prim_path=str(body.GetPath()))))
            sim.reset()
            for _ in range(4):
                sim.step(render=False)
            query=get_physx_scene_query_interface()
            verified={}
            for k,(prefix,actor) in targets.items():
                actor.update(sim.get_physics_dt())
                origin=np.array(actor.data.root_link_pos_w[0].tolist())
                def shoot(position,direction,distance):
                    hit=query.raycast_closest(tuple(origin+position),tuple(direction),distance)
                    collider=str(hit.get("collision",""))
                    return {"hit":bool(hit.get("hit") and collider.startswith(prefix)),
                            "distance_m":float(hit["distance"]) if hit.get("hit") else None,"collider":collider}
                empty,rim=probes[k]
                holes=[shoot(*ray) for ray in empty]
                solids=[shoot(*ray) for ray in rim]
                interior=[]
                if "basket" in k:
                    for x,y in ((.010,.020),(-.012,.026),(.015,-.026)):
                        interior.append(shoot([x,y,.140],[0,0,-1],.160))
                passed=(sum(not p["hit"] for p in holes)>=2 and sum(p["hit"] for p in solids)>=2
                        and all(p["hit"] and p["distance_m"]>.12 for p in interior))
                verified[k]={"passed":passed,"base_mesh_hash":assets[k]["base_mesh_hash"],
                    "silhouette_hole_samples":len(empty),"physical_clear_hole_rays":sum(not row["hit"] for row in holes),
                    "physical_solid_controls":sum(row["hit"] for row in solids),"interior_clear_to_floor":interior,
                    "hole_rays":[{"ray":ray,**result} for ray,result in zip(empty,holes,strict=True)]}
                print(k,passed,verified[k]["physical_clear_hole_rays"],"of",len(empty),flush=True)
            passed=all(v["passed"] for v in verified.values())
            write_json(WORK/("physx_rays_substitutes.json" if LOCAL else "physx_rays.json"),{"passed":passed,"assets":verified})
            if not passed:
                raise RuntimeError(f'Aperture rays rejected {[k for k,v in verified.items() if not v["passed"]]}')
        finally:
            app.close()


def main():
    marker=RUNS/("step3a_01_substitute_assets.complete" if LOCAL else "step3a_01_assets.complete")
    while not marker.exists():
        time.sleep(20)
    assert read(marker)["outcome"]=="PASS"
    with SimAllocator().claim(JOB) as gpu:
        env={**os.environ,"CUDA_VISIBLE_DEVICES":gpu["uuid"],"CUDA_DEVICE_ORDER":"PCI_BUS_ID",
             "G1D_RENDER_GPU":str(gpu["index"]),
             "LD_LIBRARY_PATH":"/data/sundawei/vla_rsi_g1d/tools/native_lib60_os_deps",
             "OMNI_KIT_ACCEPT_EULA":"YES"}
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        start='source /data/sundawei/vla_rsi_g1d/sim60/environment.sh && exec "$@"'
        options=["--local-substitutes"] if LOCAL else []
        subprocess.run(["bash","-c",start,"step3a-rays",sys.executable,"-Bu",__file__,"--worker",*options],
                       check=True,env=env,timeout=900)


if __name__=="__main__":
    if "--worker" in sys.argv:
        check()
    else:
        status=1
        try:
            main()
            status=0 if read(WORK/"physx_rays.json")["passed"] else 1
        finally:
            record_complete(RUNS/JOB,status)
