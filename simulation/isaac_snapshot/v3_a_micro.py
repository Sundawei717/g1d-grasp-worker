"""V3-A micro-experiment evaluation under paired exec2 and ACT TE profiles."""
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path

import h5py
import numpy as np

from background_status import record_complete
import step2_dev
from step2_eval import log_case
from three_view_detector import write_json
from three_view_eval import read
from v3_prompt_repair import bind, summarize, wait_pass
from v3_prompt_train import ROOT, TIER, STEP2, MICRO_CASES

RUNS = ROOT/"runs"
WORK = RUNS/"v3_a_micro_01"
TE_CASES = ("C_7302","B_7322","B_7323")
DECAY = .01
TARGET_TICKS = (20,40,80)


def alignment(raw, expert_path):
    with h5py.File(expert_path) as h:
        reference = h["obs/qpos"][:]
    steps = raw["commands"]
    out = {}
    for t in TARGET_TICKS:
        if len(steps)<t:
            out[str(t)] = None
            continue
        actual = np.asarray(steps[t-1]["arm_observed_rad"])
        errors = np.rad2deg(np.abs(reference-actual)).mean(-1)
        nearest = int(np.argmin(errors))
        out[str(t)] = {"space_deg":float(errors[nearest]),"matched_expert":nearest,
                       "lag_steps":t-nearest,"same_time_deg":float(errors[t]) if t<len(reference) else None}
    return out


def collision_now(raw):
    return raw["diagnostics"]["collision"].get("approach",{}).get("first_contact")


def clone_specs(names, modes):
    source = {r["case_id"]:r for r in read(STEP2/"diagnostic10.json")["cases"]}
    plans = WORK/"eval_plans"
    plans.mkdir(exist_ok=True)
    results = []
    for i,name in enumerate(names):
        for mode in modes:
            row = dict(source[name])
            plan = read(row["plan"])
            alias = f"{name[0]}_{5100+i*2+(mode=='te')}"
            decay = DECAY if mode=="te" else None
            plan["case"].update(case_id=alias,te_decay=decay)
            destination = plans/f"{names[0]}_{len(names)}_{alias}_{mode}.json"
            write_json(destination,plan)
            results.append({**row,"case_id":alias,"plan":str(destination),"te_decay":decay,
                           "real_case":name,"execution":mode})
    return results


def collect(kind, binding, names, modes, *, preferred, lanes=None, exclude=()):
    step2_dev.WORK = WORK
    rows = clone_specs(names,modes)
    prompts = {r["case_id"]:r["head_prompt"] for r in rows}
    out = step2_dev.simulate(kind,rows,"three_view_closed_loop",policy=binding,prompts=prompts,
                             preferred=preferred,lanes=lanes,exclude=exclude)
    archived = {r["case"]:r for r in read(TIER/"training_split.json")["episodes"]}
    run = defaultdict(list)
    for row in rows:
        response = out[row["case_id"]]
        r = log_case(row,response,"a")
        raw = response[1]
        roll = raw["policy_rollout"]
        gate = roll["gate"]
        mode = row["execution"]
        setup = r["diagnostics"]["policy_execution"]
        if setup["temporal_decay"] != row["te_decay"]:
            raise ValueError(f"Wrong temporal mode for {row['real_case']}")
        if mode=="te" and any(d["steps_applied"]!=1 or d.get("temporal_ensemble_chunks")!=min(8,i+1)
                              for i,d in enumerate(roll["decisions"])):
            raise ValueError("TE did not accumulate overlapping chunks at 20 Hz")
        r.update(case=row["real_case"],eval_case=row["case_id"],mode=mode,wall_s=roll["wall_s"],
                 alignment=alignment(raw,archived[row["real_case"]]["path"]),
                 first_contact=collision_now(raw),
                 stable_max=max(g.get("jaw_stable_steps",0) for g in gate),
                 loaded_steps=[g["step"] for g in gate if g["target_load_ready"]],
                 stable_at_loaded_steps=[g["jaw_stable_steps"] for g in gate if g["target_load_ready"]],
                 first_sustained_close=next((g["step"] for g in gate if g["sustained_close"]),None))
        run[mode].append(r)
        print(kind,mode,r["case"],r["status"],r["part_hit"],flush=True)
    if len(modes)==2:
        for a,b in zip(run["exec2"],run["te"],strict=True):
            if a["seed"]!=b["seed"] or a["head_prompt"]!=b["head_prompt"] or a["case"]!=b["case"]:
                raise ValueError("Mode comparison uses different scene input")
            if np.linalg.norm(np.subtract(a["settled_target"][:3],b["settled_target"][:3])) > .001:
                raise ValueError("Mode comparison did not reproduce initial scene within 1 mm")
    write_json(WORK/f"{kind}_result.json",{"modes":run,"summary":{k:summarize(v) for k,v in run.items()}})


def launch(stage):
    WORK.mkdir(exist_ok=True)
    if stage=="old_te":
        base = RUNS/"v3_a_prompt_repair_01"
        binding = bind(base.with_name(base.name+"_pilot"),"best_fit","micro_legacy")
        collect(stage,binding,TE_CASES,("te",),preferred=2,exclude=(0,1))
    else:
        job = f"{WORK.name}_{stage}"
        wait_pass(job)
        fit = RUNS/job
        binding = bind(fit,"best_fit",f"micro_{stage}")
        names = MICRO_CASES[stage]
        if stage=="unpaired":
            original_wait = step2_dev.wait_gpu
            owned_until = {1:f"{WORK.name}_query_pair_eval", 2:f"{WORK.name}_old_te"}

            def available(preferred, exclude=(), timeout=3600):
                if preferred in owned_until:
                    wait_pass(owned_until[preferred])
                return original_wait(preferred,exclude,timeout)

            step2_dev.wait_gpu = available
            try:
                collect(stage,binding,names,("exec2","te"),preferred=0,lanes=(0,2,1))
            finally:
                step2_dev.wait_gpu = original_wait
        else:
            collect(stage,binding,names,("exec2","te"),preferred=1,exclude=(0,2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("old_te",*MICRO_CASES))
    args=p.parse_args()
    stem=f"{WORK.name}_{args.stage}" if args.stage=="old_te" else f"{WORK.name}_{args.stage}_eval"
    try:
        launch(args.stage)
    except BaseException:
        record_complete(RUNS/stem,1)
        raise
    else:
        record_complete(RUNS/stem,0)
