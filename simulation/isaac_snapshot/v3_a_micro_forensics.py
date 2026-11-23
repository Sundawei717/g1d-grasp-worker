"""Forensics on the ten original V3-A paired rollout failures and point cues."""
from collections import defaultdict
import math
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

from background_status import record_complete
from three_view_detector import write_json
from three_view_eval import read
from three_view_inputs import head_input
from three_view_rollout import ThreeViewPolicy
from three_view_train import ThreeViewWindows
from v3_a_micro import RUNS, WORK, alignment

OLD = RUNS/"v3_a_prompt_repair_01"
OLD_FIT = RUNS/f"{OLD.name}_pilot"
EVAL_TICKS = (0,5,10,20,30,40)
CUT_DEG = .5
PERSIST = 3


def pair_plans():
    manifest = read(OLD_FIT/"pilot_split.json")
    groups = defaultdict(list)
    for r in manifest["episodes"]:
        groups[r["pair_group_id"]].append(r)
    branched = []
    for pid,rows in groups.items():
        a,b = sorted(rows,key=lambda r:r["part"])
        x,y = [],[]
        for r in (a,b):
            with h5py.File(r["path"]) as h:
                n = int(np.flatnonzero(h["phase"].asstr()[:]=="lift")[0])
                x.append(h["action/qpos_target"][:n])
        n = min(map(len,x))
        degrees = np.rad2deg(np.abs(x[0][:n]-x[1][:n])).mean(axis=1)
        idx = np.flatnonzero(degrees>1e-4)
        real = next((i for i in range(n-PERSIST+1) if np.all(degrees[i:i+PERSIST]>=CUT_DEG)),None)
        branched.append({"pair_group_id":pid,"cases":[r["case"] for r in (a,b)],"bucket":a["bucket"],
                         "branch_step":real,"first_numerically_different":int(idx[0]) if len(idx) else None,
                         "difference_at_branch":float(degrees[real]) if real is not None else None,
                         "at":{str(t):float(degrees[t]) for t in EVAL_TICKS},"mae_deg":degrees.tolist()})
    return manifest,branched


def directional_delta(policy,ds,lookup,src,other,step):
    path,prompt,*_ = ds.files[src]
    alt = ds.files[other][1]
    if alt["bbox_0"] != prompt["bbox_0"]:
        raise ValueError("Point swap requires a fixed bounding box")
    inputs = ds[lookup[src,step]][:5]
    with h5py.File(path) as h:
        head,meta = head_input(h["obs/rgb_left"][step],prompt["bbox_0"],alt["affordance_0"],
                               point_geometry=policy.prompt_fusion)
    np.testing.assert_array_equal(inputs[0][:4],head[:4])
    np.testing.assert_array_equal(inputs[3][:6],meta[:6])
    alternate = {"point":(head,inputs[1],inputs[2],meta,inputs[4]),
                 "heatmap":(head,*inputs[1:]),
                 "geometry":(*inputs[:3],meta,inputs[4])}
    batch = [torch.as_tensor(np.stack(x)) for x in zip(inputs,*alternate.values(),strict=True)]
    act = policy.model(*batch).numpy()
    scale = np.asarray(policy.norm["action"]["scale"][:7])
    return {k: np.rad2deg(np.abs(v[:,:7]-act[0,:,:7])*scale) for k,v in zip(alternate,act[1:],strict=True)}


@torch.inference_mode()
def swap(checkpoint,manifest,pairs, *, boxes=None, detector_contract=None):
    policy = ThreeViewPolicy(checkpoint,"cpu",detector=detector_contract)
    ds = ThreeViewWindows(manifest,"train",boxes,policy.norm,point_geometry=policy.prompt_fusion)
    lookup = {(fi,t):k for k,(fi,t,_) in enumerate(ds.windows)}
    index={r["case"]:i for i,r in enumerate(manifest["episodes"]) if r["split"]=="train"}
    records = []
    for pair in pairs:
        if not all(c in index for c in pair["cases"]):
            continue
        ticks = sorted(set(t for t in (0,pair["branch_step"]) if t is not None))
        for t in ticks:
            all_values = defaultdict(list)
            directions = {}
            for src,other in (pair["cases"],pair["cases"][::-1]):
                vals = directional_delta(policy,ds,lookup,index[src],index[other],t)
                for key,v in vals.items():
                    all_values[key].extend(v.ravel())
                directions[src] = {k:{"median_deg":float(np.median(v)),"max_deg":float(v.max())} for k,v in vals.items()}
            records.append({"pair_group_id":pair["pair_group_id"],"cases":pair["cases"],"step":t,"directions":directions,
                            "modes":{k:{"median_deg":float(np.median(v)),"max_deg":float(np.max(v))}
                                     for k,v in all_values.items()}})
    return records


def first_contact(row,raw):
    path = Path(row["path"]).with_name("physics_trace.jsonl")
    with path.open() as f:
        import json
        initial = json.loads(next(f))
        base = initial["time_s"]
        end = base+len(raw["policy_rollout"]["gate"])*.05
        contact = next((record for line in f if (record:=json.loads(line))["forbidden_contact"]
                        and record["time_s"]<=end+1e-8),None)
    if contact is None:
        return None
    rel = contact["time_s"]-base
    approach = raw["diagnostics"]["collision"]["approach"]
    settle = raw["diagnostics"]["collision"].get("settle",{})
    cases = approach["pair_onsets"]
    # Dict insertion order in the historical JSON identifies one of the first
    # onsetting pairs, provided no forbidden contact persisted from settling.
    pair = next(iter(cases),None) if not settle.get("contact_physics_steps") else None
    return {"step_1based":int(math.ceil(rel/.05-1e-7)),"relative_s":rel,
            "first_recorded_pair":pair,"all_approach_pairs":list(cases),
            "simultaneous_pairs_known":False}


def boundary_jitter(gate):
    widths = np.asarray([r["width_command_m"] for r in gate])*1000
    ds = abs(np.diff(widths))
    return {key:{"median_mm":float(np.median(v)),"p95_mm":float(np.percentile(v,95))}
            for key,v in {"within_chunk":ds[::2],"between_chunks":ds[1::2]}.items() if len(v)}


def historical(manifest):
    entries = {r["case"]:r for r in manifest["episodes"]}
    comp = read(OLD/"pilot_eval_comparison.json")
    output = []
    for r in comp["episodes"]:
        raw = read(r["path"])
        gate = raw["policy_rollout"]["gate"]
        motion = alignment(raw,entries[r["case"]]["path"])
        collision = first_contact(r,raw)
        if bool(collision) != bool(raw["diagnostics"]["collision"]["approach"]["contact_physics_steps"]):
            raise ValueError(f"Trace disagrees with phase collision aggregate: {r['case']}")
        ready = [x for x in gate if x["target_load_ready"]]
        trigger = raw["policy_rollout"]["trials"]
        output.append({"case":r["case"],"status":r["status"],"part_hit":r["part_hit"],"steps":r["steps"],
                       "geometry":motion,"first_collision":collision,"boundary_jitter":boundary_jitter(gate),
                       "stable_max":max(x["jaw_stable_steps"] for x in gate),
                       "first_loaded_step":ready[0]["step"] if ready else None,
                       "max_consecutive_stable_loaded":max((x["jaw_stable_steps"] for x in ready),default=0),
                       "loaded_steps":[x["step"] for x in ready],
                       "trigger":trigger[0]["trigger"]["step"] if trigger else None,"path":r["path"]})
    return sorted(output,key=lambda r:r["case"])


def run(scope):
    WORK.mkdir(exist_ok=True)
    manifest, pairs = pair_plans()
    if scope == "historical":
        value = {"branch":{"minimum_deg":CUT_DEG,"consecutive_steps":PERSIST,"pairs":pairs},
                 "historical":historical(manifest),"prompt_swap":swap(OLD_FIT/"best_fit.pt",manifest,pairs)}
        write_json(WORK/"forensics.json",value)
    else:
        fit = RUNS/f"{WORK.name}_query_pair"
        value = swap(fit/"best_fit.pt",read(fit/"pilot_split.json"),pairs)
        write_json(WORK/"query_pair_swap.json",value)


if __name__=="__main__":
    scope = sys.argv[1]
    if scope not in {"historical","query_pair"}:
        raise ValueError(scope)
    job = RUNS/f"{WORK.name}_{'forensics' if scope=='historical' else 'query_pair_swap'}"
    try:
        run(scope)
    except BaseException:
        record_complete(job,1)
        raise
    else:
        record_complete(job,0)
