import json

import h5py
import numpy as np
import pytest

from g1d_grasp.data import WindowDataset, load_manifest, make_portable, protected_check, split_groups
from g1d_grasp.data.cli import audit
from g1d_grasp.data.episode import RGB_PATHS, file_sha256, read_episode


def episode(root, number, *, split=None, family=None):
    case = {"case_id":f"C_{100+number}","episode_id":f"episode-{number}","scene_id":f"scene-{number}",
            "asset_family":"fixture/mug","part":"mug_body"}
    path = root/f"{number}.h5"
    ts = np.arange(5)*.05
    ids = [f"f{i}" for i in range(5)]
    phase = ["approach","close","close","lift"]
    with h5py.File(path,"w") as h:
        h.attrs["schema"] = "g1d_v3_step_aligned"
        h["meta/case_json"] = json.dumps(case)
        h["meta/initial_prompt_json"] = json.dumps({"frame_id":ids[0],"bbox_0":[.2,.3,.7,.8],
                                                  "affordance_0":[.6,.6],"target_query":"mug"})
        h["meta/strict_score_json"] = json.dumps({"success":True})
        h["meta/joint_names"] = [f"joint_{i}" for i in range(7)]
        h["obs/timestamp"] = ts
        h["obs/frame_id"] = ids
        h["obs/control_step_id"] = np.arange(5)
        h["obs/qpos"] = np.arange(35).reshape(5,7)/100
        h["obs/gripper_width"] = np.full(5,.04)
        h["obs/eef_pose"] = np.tile([0.,0,0,0,0,0,1],(5,1))
        for camera,key in RGB_PATHS.items():
            pixel = np.arange(5,dtype=np.uint8)[:,None,None,None]
            h.create_dataset(key,data=np.broadcast_to(pixel,(5,224,224,3)),compression="gzip")
            if camera!="head":
                h[f"obs/{camera}/timestamp"] = ts
                h[f"obs/{camera}/frame_id"] = ids
        h["action/control_step_id"] = np.arange(4)
        h["action/qpos_target"] = np.arange(28).reshape(4,7)/100
        h["action/gripper_width_target"] = np.full(4,.03)
        h["action/native_command"] = np.zeros((4,9))
        h["action/native_substep_command"] = np.zeros((4,30,9))
        h["phase"] = phase
        h["train_action_valid"] = [True,True,True,False]
    return {**case,"case":case["case_id"],"scene_family":family or f"family-{number}",
            "steps":4,"sha256":file_sha256(path),"path":str(path),"split":split}


def make(root, rows):
    source=root/"original.json"
    source.write_text(json.dumps({"episodes":rows}),encoding="utf-8")
    return make_portable(source,root)


def test_windows_history_and_lift_mask(tmp_path):
    row=episode(tmp_path,1,split="train")
    manifest=make(tmp_path,[row])
    manifest.write(tmp_path/"portable.json")
    ds=WindowDataset(load_manifest(tmp_path/"portable.json"),tmp_path,horizon=4,history=3,verify_sha256=True)
    assert len(ds)==3 and ds[0]["history"]["steps"].tolist()==[0,0,0]
    assert ds[1]["history"]["steps"].tolist()==[0,0,1]
    assert ds[-1]["observation_step"]==2
    assert ds[0]["action_mask"].tolist()==[True,True,True,False]
    assert ds[0]["close_mask"].tolist()==[False,True,True,False]
    assert ds[-1]["action_mask"].tolist()==[True,False,False,False]
    assert all(int(x[0,0,0])==2 for x in ds[-1]["rgb"].values())
    assert not ds[-1]["actions"][1:].any()
    assert audit(manifest,tmp_path)["trainable_windows"]=={"train":3}


def test_split_preserves_existing_and_protects_pairs(tmp_path):
    paired=[episode(tmp_path,n,family="pair",split="train" if n==1 else None) for n in (1,2)]
    assigned = [*paired,episode(tmp_path,3,split="validation"),episode(tmp_path,4),episode(tmp_path,5)]
    manifest=make(tmp_path,assigned)
    split=split_groups(manifest,validation_fraction=.5)
    assert [row["split"] for row in split.episodes[:3]]==["train","train","validation"]
    assert split.episodes==split_groups(manifest,validation_fraction=.5).episodes
    assigned[1]["split"]="validation"
    with pytest.raises(ValueError,match="split leakage"):
        make(tmp_path,assigned)


def test_protected_geometry_and_checksum(tmp_path):
    row={**episode(tmp_path,1,split="train"),"layout_signature":"layout-1","base_mesh_hash":"mesh-a"}
    manifest=make(tmp_path,[row])
    forbidden=tmp_path/"reserved.json"
    for case, field in (({"layout_signature":"layout-1"},"layout_signature"),
                        ({"tier":4,"base_mesh_hash":"mesh-a"},"L4 mesh")):
        forbidden.write_text(json.dumps({"cases":[case]}))
        with pytest.raises(ValueError,match=field):
            protected_check(manifest,[forbidden])
    with h5py.File(row["path"],"r+") as h:
        h["train_action_valid"][-1]=True
    with pytest.raises(ValueError,match="SHA-256"):
        read_episode(row["path"],sha256=row["sha256"])
    with pytest.raises(ValueError,match="train_action_valid"):
        read_episode(row["path"])


def test_portable_containment(tmp_path):
    row=episode(tmp_path,1)
    with pytest.raises(ValueError,match="outside"):
        make(tmp_path,[{**row,"path":str(tmp_path.parent/"other.h5")}])

def test_official_ids_override_rebound_archive_aliases(tmp_path):
    row = episode(tmp_path,1,split="train")
    row.update(episode_id="tier1-episode-1",scene_id="tier1-scene-1")
    manifest = make(tmp_path,[row])
    assert WindowDataset(manifest,tmp_path)[0]["episode_id"] == "tier1-episode-1"
    assert audit(manifest,tmp_path)["archive_id_aliases"] == 1
