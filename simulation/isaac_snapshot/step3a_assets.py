"""Prepare opaque original meshes, CoACD collision, and the spatial part atlases."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import tarfile

import coacd
import numpy as np
import trimesh
from pxr import Gf,Usd,UsdGeom,UsdPhysics,UsdShade,Vt
from scipy.spatial.transform import Rotation

from grasp_regions import *
from part_surface import load_collision_surfaces
from three_view_detector import sha256 as file_hash,write_json
from three_view_eval import read
from tier1_plan import base_mesh_signature

ROOT=Path(__file__).resolve().parent
WORK=ROOT/"runs/step3a_01"
SOURCE=WORK/"source"
PREPARED=WORK/"assets"
YCB=("004_sugar_box","006_mustard_bottle","007_tuna_fish_can","008_pudding_box","011_banana",
     "019_pitcher_base","024_bowl","025_mug","035_power_drill")
HOLD={"white_mug","odd_handle_mug","round_basket","007_tuna_fish_can","008_pudding_box"}
NEW=("white_mug","odd_handle_mug","round_basket","arched_basket",*YCB)
LOCAL_TRAIN={"marker_long_box":"marker_box/00003","compact_box":"sand3d_packaging/00000",
             "round_package":"sand3d_packaging/00225","matte_soda":"soda_can/00000"}
SUBSTITUTES=(*LOCAL_TRAIN,"raised_rim_basket")


def tube(points,radius,sides=10):
    points=np.asarray(points,dtype=float)
    tang=np.gradient(points,axis=0)
    normals=np.c_[tang[:,2],np.zeros(len(points)),-tang[:,0]]
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    angles=np.linspace(0,2*np.pi,sides,endpoint=False)
    vert=(points[:,None,:]+radius*(np.cos(angles)[None,:,None]*normals[:,None,:]
            +np.sin(angles)[None,:,None]*np.array([0.,1.,0.])[None,None,:])).reshape(-1,3)
    faces=[]
    for i in range(len(points)-1):
        for j in range(sides):
            a,b=i*sides+j,i*sides+(j+1)%sides
            faces.extend(((a,b,b+sides),(a,b+sides,a+sides)))
    end=(len(points)-1)*sides
    faces.extend((0,j+1,j) for j in range(1,sides-1))
    faces.extend((end,end+j,end+j+1) for j in range(1,sides-1))
    m=trimesh.Trimesh(vert,faces,process=True)
    m.fix_normals()
    return m


def original(name):
    if name in LOCAL_TRAIN:
        source=ROOT/"assets"/LOCAL_TRAIN[name]/"object.usdz"
        stage=Usd.Stage.Open(str(source))
        cache=UsdGeom.XformCache()
        meshes=[]
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            geom=UsdGeom.Mesh(prim)
            if geom.ComputeVisibility()==UsdGeom.Tokens.invisible:
                continue
            points=np.array([tuple(cache.GetLocalToWorldTransform(prim).Transform(v))
                             for v in geom.GetPointsAttr().Get()])
            counts=np.array(geom.GetFaceVertexCountsAttr().Get(),dtype=int)
            indices=np.array(geom.GetFaceVertexIndicesAttr().Get(),dtype=int)
            if len(counts)==0:
                continue
            triangles=[]
            start=0
            for size in counts:
                f=indices[start:start+size]
                triangles.extend((f[0],f[k],f[k+1]) for k in range(1,size-1))
                start+=size
            meshes.append(trimesh.Trimesh(points,triangles,process=True))
        if not meshes:
            meshes=list(load_collision_surfaces(str(source)).values())
        mesh=trimesh.util.concatenate(meshes)
        if name=="marker_long_box":
            mesh.vertices=mesh.vertices[:,[2,0,1]]
        mesh.vertices[:,:2]-=mesh.bounds.mean(0)[:2]
        mesh.vertices[:,2]-=mesh.bounds[0,2]
        return [(mesh,None)],str(source),file_hash(source)
    if name=="white_mug":
        src=ROOT/"assets/mug/00004/object.usdz"
        mesh=trimesh.util.concatenate(list(load_collision_surfaces(str(src)).values()))
        return [(mesh,None)],str(src),file_hash(src)
    if name in {"round_basket","arched_basket","raised_rim_basket","odd_handle_mug"}:
        if name=="odd_handle_mug":
            bottom=trimesh.creation.cylinder(.031,.005,sections=36).apply_translation([0,0,.0025])
            wall=trimesh.creation.annulus(.027,.032,height=.065,sections=36).apply_translation([0,0,.035])
            bar=tube([(x,0,z) for x,z in ((.028,.025),(.050,.021),(.060,.031),(.058,.047),(.042,.058),(.030,.057))],.0045,8)
            return [(bottom,MUG_BODY),(wall,MUG_BODY),(bar,MUG_HANDLE)],"authored angled handle mug",None
        if name=="round_basket":
            bottom=trimesh.creation.cylinder(.043,.006,sections=12).apply_translation([0,0,.003])
            side=trimesh.creation.annulus(.040,.047,height=.037,sections=12).apply_translation([0,0,.023])
            rim=trimesh.creation.annulus(.041,.054,height=.009,sections=12).apply_translation([0,0,.045])
            arch=[(-.044,.040),(-.049,.058),(-.040,.081),(-.023,.097),(.017,.102),(.040,.085),(.046,.061),(.042,.041)]
            handle=tube([(x,0,z) for x,z in arch],.0055,8)
            return [(bottom,None),(side,None),(rim,BASKET_RIM),(handle,BASKET_HANDLE)],"authored 12-sided round basket",None
        def box(ext,pos):
            return trimesh.creation.box(extents=ext).apply_translation(pos)
        floor=[(box((.107,.080,.008),(0,0,.004)),None)]
        for sign in (-1,1):
            floor.extend(((box((.108,.007,.044),(0,sign*.04,.021)),None),
                          (box((.008,.076,.045),(sign*.051,0,.021)),None)))
            if name=="raised_rim_basket":
                for end in (-1,1):
                    floor.append((box((.012,.009,.028),(end*.042,sign*.047,.055)),None))
                floor.append((box((.095,.017,.013),(0,sign*.052,.072)),BASKET_RIM))
            else:
                floor.append((box((.086,.014,.009),(0,sign*.050,.048)),BASKET_RIM))
        arch=[(-.048,.04),(-.046,.068),(-.034,.090),(-.013,.098),(.021,.098),(.046,.084),(.051,.060),(.049,.040)]
        floor.append((tube([(x,0,z) for x,z in arch],.0055,12),BASKET_HANDLE))
        provenance=("authored tray with a raised lateral rim and independent contact clearance"
                    if name=="raised_rim_basket" else "authored tray with round-section reinforced handle")
        return floor,provenance,None
    file=SOURCE/f"{name}.tgz"
    with tarfile.open(file) as tar:
        member=next(m for m in tar.getmembers() if m.name.endswith("/nontextured.ply"))
        raw=tar.extractfile(member).read()
    m=trimesh.load(BytesIO(raw),file_type="ply",force="mesh",process=False)
    if name=="004_sugar_box":
        m.vertices=m.vertices[:,[0,2,1]]*[1,1,-1]
    elif name in {"008_pudding_box","011_banana","035_power_drill","006_mustard_bottle"}:
        transform,_=trimesh.bounds.oriented_bounds_2D(m.vertices[:,:2])
        xy=trimesh.transform_points(m.vertices[:,:2],transform)
        if np.ptp(xy[:,0])>np.ptp(xy[:,1]):
            xy=xy[:,::-1]*[-1,1]
        m.vertices=np.c_[xy,m.vertices[:,2]]
    elif name=="025_mug":
        base=np.median(m.vertices[m.vertices[:,2]<.012,:2],axis=0)
        large=m.vertices[:,:2]-base
        polar=large[np.linalg.norm(large,axis=1)>.9*np.linalg.norm(large,axis=1).max()].mean(axis=0)
        angle=np.arctan2(polar[1],polar[0])
        rotated=Rotation.from_euler("z",-angle).apply(m.vertices-np.r_[base,0])
        rotated[:,0]-=.015
        m.vertices=rotated
    if name!="025_mug":
        m.vertices[:,:2]-=m.bounds.mean(axis=0)[:2]
    m.vertices[:,2]-=m.bounds[0,2]
    return [(m,None)],f"https://ycb-benchmarks.s3.amazonaws.com/data/google/{name}_google_16k.tgz",sha256(raw).hexdigest()


def profiles(name,mesh):
    low,high=mesh.bounds
    center=mesh.triangles_center
    extent=high-low
    cap=low[2]+.78*extent[2]
    grips={}
    probes=[]
    if name in {"white_mug","odd_handle_mug","025_mug"}:
        if name=="white_mug":
            split=.018
            grips={MUG_BODY:([-.021,0,.027],90,.065),MUG_HANDLE:([.039,0,.007],90,.017)}
            probes=[[x,z] for x in (.023,.027,.031,.034) for z in (-.006,.0,.006,.011)]
        elif name=="odd_handle_mug":
            split=.030
            grips={MUG_BODY:([-.012,0,.057],90,.062),MUG_HANDLE:([.056,0,.039],90,.015)}
            probes=[[x,z] for x in (.037,.042,.048) for z in (.026,.032,.038,.044)]
        else:
            split=.023
            grips={MUG_BODY:([-.024,0,cap],90,.076),MUG_HANDLE:([high[0]-.006,0,.47*high[2]],90,.019)}
            probes=[[x,z] for x in np.linspace(split+.003,high[0]-.006,4) for z in np.linspace(.23*high[2],.70*high[2],5)]
        marks=np.where(center[:,0]>=split,2,np.where(center[:,0]<=split-.007,1,0))
        names=[None,MUG_BODY,MUG_HANDLE]
    elif name in {"round_basket","arched_basket","raised_rim_basket"}:
        rim=[0,-.052,.072] if name=="raised_rim_basket" else [0,-.050,.052]
        grips={BASKET_HANDLE:([0,0,.096],90,.016),BASKET_RIM:(rim,90,.017)}
        probes=[[x,z] for x in (-.02,-.01,0,.01,.02) for z in (.065,.076,.085)]
        marks=np.zeros(len(center),dtype=np.uint8)
        names=[None,BASKET_HANDLE,BASKET_RIM]
    elif name=="024_bowl":
        marks=np.where((center[:,2]>high[2]-.013)&(center[:,1]>.048),2,
                 np.where((center[:,2]>high[2]-.013)&(center[:,1]<-.048),1,0))
        grips={BOX_NEG_END:([0,-.074,high[2]-.003],90,.016),BOX_POS_END:([0,.074,high[2]-.003],90,.016)}
        names=[None,BOX_NEG_END,BOX_POS_END]
    elif name in {"round_package","matte_soda"}:
        from grasp_regions import MID_PART
        border=.59*high[2]
        marks=np.where(center[:,2]>border,2,1)
        names=[None,MID_PART,UPPER_PART]
        grips={MID_PART:([0,0,.48*high[2]],90,min(extent[1]-.002,.07)),
               UPPER_PART:([0,0,.77*high[2]],90,min(extent[1]-.002,.07))}
    elif name in {"004_sugar_box","008_pudding_box","011_banana","035_power_drill","006_mustard_bottle",
                  "019_pitcher_base","marker_long_box","compact_box"}:
        cut=.16*extent[1]
        marks=np.where(center[:,1]>cut,2,np.where(center[:,1]<-cut,1,0))
        x=(low[0]+high[0])/2
        grips={BOX_NEG_END:([x,-.32*extent[1],cap],0,min(extent[0]-.004,.072)),
               BOX_POS_END:([x,.32*extent[1],cap],0,min(extent[0]-.004,.072))}
        names=[None,BOX_NEG_END,BOX_POS_END]
        # A curved or tapered object's useful opposing surfaces are evaluated by the planner.
        if name=="011_banana":
            for part,(loc,_,_) in grips.items():
                subset=center[np.abs(center[:,1]-loc[1])<.010]
                if len(subset): loc[0]=float(np.median(subset[:,0]))
                grips[part]=loc,0,.032
    else:
        marks=np.ones(len(center),dtype=np.uint8)
        names=[None,UPPER_PART]
        grips={UPPER_PART:([0,.013,cap],0,.075)}
    return np.asarray(marks,dtype=np.uint8),names,grips,[[float(x),float(z)] for x,z in probes]


def prepare(name):
    output=PREPARED/name
    output.mkdir(parents=True,exist_ok=True)
    receipt=output/"result.json"
    if receipt.is_file():
        return read(receipt)
    pieces,source,digest=original(name)
    base=trimesh.util.concatenate([m for m,_ in pieces])
    marks,names,grips,probes=profiles(name,base)
    stage=Usd.Stage.CreateNew(str(output/"object.usda"))
    UsdGeom.SetStageMetersPerUnit(stage,1)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    root=UsdGeom.Xform.Define(stage,"/root").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.085 if "basket" in name else .09)
    UsdGeom.Xform.Define(stage,"/root/visual")
    UsdGeom.Xform.Define(stage,"/root/collision")
    mat=UsdShade.Material.Define(stage,"/root/Opaque")
    shader=UsdShade.Shader.Define(stage,"/root/Opaque/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    from pxr import Sdf
    colors=((.70,.68,.62),(.14,.40,.57),(.69,.45,.18),(.27,.56,.34),(.65,.23,.12))
    color=colors[(NEW+SUBSTITUTES).index(name)%len(colors)]
    shader.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.64)
    mat.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface",Sdf.ValueTypeNames.Token))
    def mesh(path,v,f):
        prim=UsdGeom.Mesh.Define(stage,path)
        prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(v,dtype=np.float32)))
        prim.CreateFaceVertexCountsAttr([3]*len(f))
        prim.CreateFaceVertexIndicesAttr(np.asarray(f).ravel().tolist())
        prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        return prim
    options=dict(threshold=.022,preprocess_mode="auto",preprocess_resolution=50,resolution=1600,
                 mcts_nodes=12,mcts_iterations=55,mcts_max_depth=3,seed=19)
    hull_count=0
    offset=0
    for i,(part,label) in enumerate(pieces):
        visible=mesh(f"/root/visual/surface_{i:02}",part.vertices,part.faces)
        UsdShade.MaterialBindingAPI.Apply(visible.GetPrim()).Bind(mat)
        if any(p is not None for _,p in pieces):
            marks[offset:offset+len(part.faces)]=names.index(label)
        offset+=len(part.faces)
        keys=np.rint(part.vertices/.0006).astype(np.int32)
        _,indices,inverse=np.unique(keys,axis=0,return_index=True,return_inverse=True)
        faces=inverse[part.faces]
        keep=(faces[:,0]!=faces[:,1])&(faces[:,0]!=faces[:,2])&(faces[:,1]!=faces[:,2])
        clean=trimesh.Trimesh(part.vertices[indices],faces[keep],process=True)
        clean.update_faces(clean.unique_faces())
        clean.remove_unreferenced_vertices()
        hulls=coacd.run_coacd(coacd.Mesh(clean.vertices,clean.faces),**options)
        for xyz,tri in hulls:
            c=mesh(f"/root/collision/hull_{hull_count:03}",xyz,tri)
            c.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(c.GetPrim()).CreateApproximationAttr("convexHull")
            hull_count+=1
    stage.GetRootLayer().Save()
    m=output/"object.usda"
    atlas=output/"part_faces.npz"
    np.savez_compressed(atlas,vertices=base.vertices,faces=base.faces,labels=marks)
    anchors={}
    for part,(center,yaw,gap) in grips.items():
        surface=base.triangles_center[marks==names.index(part)]
        if len(surface)<2:
            raise ValueError(f"No mapped faces for {name}/{part}")
        anchors[part]=surface[np.argmin(np.linalg.norm(surface-center,axis=1))].tolist()
    result={"name":name,"asset":str(m),"source":source,"source_sha256":digest,
            "asset_sha256":file_hash(m),"atlas_sha256":file_hash(atlas),"base_mesh_hash":base_mesh_signature(m),
            "heldout":name in HOLD,"hulls":hull_count,"coacd_options":options,"dims_m":(base.bounds[1]-base.bounds[0]).tolist(),
            "bounds":base.bounds.tolist(),"grips":grips,"hole_probes_xz":probes,
            "part_meshes":{"face_atlas":str(atlas),"face_names":{str(i):p or "other_surface" for i,p in enumerate(names)},
                "part_anchors":anchors},"triangle_counts":{str(p):(int((marks==i).sum())) for i,p in enumerate(names)}}
    if name=="white_mug" or name in LOCAL_TRAIN:
        result["raw_visual_base_mesh_hash"]=base_mesh_signature(source)
    write_json(receipt,result)
    print("asset",name,"hulls",hull_count,"base",result["base_mesh_hash"],flush=True)
    return result


def run():
    from background_status import record_complete
    code=1
    try:
        prepared=[]
        with ProcessPoolExecutor(max_workers=2) as pool:
            pending=[pool.submit(prepare,k) for k in NEW]
            for f in as_completed(pending):
                prepared.append(f.result())
                write_json(WORK/"asset_progress.json",{"assets":prepared})
        hashes=[a["base_mesh_hash"] for a in prepared]
        old=read(ROOT/"runs/tier1_v3_01/final_test_reservation.json")
        base=old["training_base_mesh_signatures"]
        assert len(set(hashes))==len(NEW) and not set(hashes).intersection(base.values())
        train={a["name"]:a for a in prepared if not a["heldout"]}
        hold={a["name"]:a for a in prepared if a["heldout"]}
        write_json(WORK/"asset_inventory.json",{"passed":True,"tier1_training_base_hashes":base,
                  "tier2_proposed":train,"l4_heldout":hold,"base_overlap":0})
        code=0
    finally:
        record_complete(ROOT/"runs/step3a_01_assets",code)


if __name__=="__main__":
    run()
