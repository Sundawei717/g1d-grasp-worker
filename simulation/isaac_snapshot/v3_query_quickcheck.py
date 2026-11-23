"""Pre-fit checks using original unified executor traces and frozen model predictions."""
from collections import defaultdict
from pathlib import Path
import json
import math

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from temporal_actions import TemporalActions
from three_view_detector import sha256, write_json
from three_view_eval import read
from three_view_rollout import ThreeViewPolicy
from three_view_train import ThreeViewWindows
from v3_prompt_train import ROOT, TIER, STEP2
from v3_a_micro import alignment


RUNS = ROOT/"runs"
WORK = RUNS/"v3_a_query_gate_01"
PAIRS = (("torso_link","right_shoulder_yaw_link"),("logo_link","right_dex1_model_base_link"))
NEAR = (20,30,40,50,53,60,80)


def checked_experts(selected):
    rows = []
    for row in selected:
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Changed expert trajectory {path}")
        raw = read(path.parent/"expert_diagnostic.json")
        roll = raw["policy_rollout"]
        with (path.parent/"physics_trace.jsonl").open() as file:
            samples = [json.loads(line) for line in file]
        named = raw.get("forbidden_contact_pairs",{})
        matches = [name for name in named if any(all(token in name for token in pair) for pair in PAIRS)]
        rows.append({"case":row["case"],"path":str(path.parent),
                     "status":roll["status"],"part_hit":roll["part_hit"],"forbidden_steps":sum(x["forbidden_contact"] for x in samples),
                     "pairs":named,"focus_pairs":matches,"native_replay":bool(raw.get("replay_alignment")),
                     "samples":len(samples)})
    return rows


@torch.inference_mode()
def inspect_fit():
    fit = RUNS/"v3_a_micro_01_unpaired"
    model = ThreeViewPolicy(fit/"best_fit.pt","cpu")
    manifest = read(fit/"pilot_split.json")
    manifest["episodes"] = [r for r in manifest["episodes"] if r["case"] in {"C_7302","B_8343"}]
    data = ThreeViewWindows(manifest,"train",None,model.norm,phase_metrics=True,point_geometry=True)
    stats, targets, forecasts = defaultdict(list),defaultdict(list),defaultdict(list)
    scale = np.asarray(model.norm["action"]["scale"])
    centre = np.asarray(model.norm["action"]["mean"])
    cursor=0
    for *inputs,y,mask,close in DataLoader(data,batch_size=12,num_workers=2):
        predicted = model.model(*inputs)
        err = (predicted-y).abs()
        for j,(yhat,c,ok) in enumerate(zip(predicted,close,mask,strict=True)):
            idx,step,_ = data.windows[cursor]
            stats[idx].append((err[j,:,:7].mean(-1)[ok].numpy(),err[j,:,7][ok].numpy(),c[ok].numpy()))
            targets[idx].append(y[j,0].numpy()*scale+centre)
            forecasts[idx].append(yhat.numpy()*scale+centre)
            cursor+=1
    def distance(x,y):
        x = np.abs(np.asarray(x)-np.asarray(y))
        return np.column_stack((np.rad2deg(x[:,:7]).mean(-1),x[:,7]*1000))
    rollouts = read(RUNS/"v3_a_micro_01/unpaired_result.json")["modes"]
    result = {}
    for i,row in enumerate(manifest["episodes"]):
        j,g,c = [np.concatenate(x) for x in zip(*stats[i],strict=True)]
        valid = np.asarray(targets[i])
        prediction = np.asarray(forecasts[i])
        ensemble = TemporalActions(.01)
        fused = np.stack([ensemble.step(t,x)[0] for t,x in enumerate(prediction)])
        unfolded = np.stack([prediction[t-t%2,t%2] for t in range(len(valid))])
        open_err = {"first_action":distance(prediction[:,0],valid),
                    "teacher_exec2":distance(unfolded,valid),
                    "teacher_te":distance(fused,valid)}
        rows = {}
        for mode,data in rollouts.items():
            run = next(x for x in data if x["case"]==row["case"])
            raw = read(run["path"])
            with h5py.File(row["path"]) as h:
                ref=h["obs/qpos"][:]
            e = []
            for k,q in enumerate(raw["commands"][:len(raw["policy_rollout"]["gate"])]):
                err=np.rad2deg(np.abs(ref-np.asarray(q["arm_observed_rad"]))).mean(axis=-1)
                e.append({"step":k+1,"space_deg":float(err.min()),"match":int(err.argmin())})
            max_band = next((i["step"] for i in e if i["space_deg"]>=10),None)
            rows[mode] = {"status":run["status"],"first_contact":run["first_contact"],"alignment":run["alignment"],
                          "first_space_ge_10deg":max_band,"near":{str(t):e[t-1] for t in NEAR if t<=len(e)},
                          "failure":run["failure"],"trigger":run["trigger"]}
        result[row["case"]] = {"joint_l1":float(j.mean()),"jaw_l1":float(g.mean()),
                               "joint_close_l1":float(j[c].mean()),"jaw_close_l1":float(g[c].mean()),"action_terms":len(j),
                               "open_loop":{k:{"mean":v.mean(0).tolist(),"p50":np.median(v,0).tolist(),
                                             "at":{str(t):v[t].tolist() for t in NEAR if t<len(v)}} for k,v in open_err.items()},
                               "runs":rows}
    return result


if __name__=="__main__":
    ids = {r["case_id"] for r in read(STEP2/"diagnostic10.json")["cases"]}
    source = [r for r in read(TIER/"training_split.json")["episodes"] if r["case"] in ids]
    evidence=checked_experts(source)
    write_json(WORK/"quickcheck.json",{"experts":evidence,"allowlisted_pairs":[],"fit":inspect_fit()})
    print("Quick check complete",flush=True)
