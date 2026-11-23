"""Run the fixed V3-A paired TE gate and only launch the full fit when it passes."""
from collections import defaultdict
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from background_status import record_complete
import step2_dev
from step2_eval import log_case
from three_view_detector import sha256, write_json
from three_view_eval import read
from v3_a_micro import alignment
from v3_a_micro_forensics import pair_plans, swap
from v3_prompt_repair import summarize, wait_pass
from v3_prompt_train import ROOT, STEP2, TIER
from v3_query_quickcheck import PAIRS


RUNS = ROOT/"runs"
WORK = RUNS/"v3_a_query_gate_01"
FIT_GATE = RUNS/f"{WORK.name}_gate_fit"
FIT_FULL = RUNS/f"{WORK.name}_full_fit"
ASSETS = {"mug","basket","carton"}
TICKS = (20,40,80)


def cases():
    return read(STEP2/"diagnostic10.json")["cases"]


def archive_map():
    return {r["case"]:r for r in read(TIER/"training_split.json")["episodes"]}


def bind(fit, pick, name):
    report = read(fit/"result.json")
    config = report["config"]
    if (not report["complete_optimization"] or config["variant"]!="A" or not config["query_bbox"]
            or not config["point_queries"] or config["warm_start"] is not None):
        raise ValueError("This gate requires a cold-start box-and-point Query checkpoint")
    path = fit/f"{pick}.pt"
    digest = sha256(path)
    wanted = report["last_sha256"] if pick=="last" else report["checkpoint_sha256"]
    if digest!=wanted:
        raise ValueError(f"Checkpoint changed: {path}")
    receipt = WORK/f"{name}_binding.json"
    write_json(receipt,{**report,"checkpoint":str(path),"checkpoint_sha256":digest})
    return {"receipt":str(receipt),"path":str(path),"sha256":digest}


def replay_experts():
    step2_dev.WORK = WORK
    specs = cases()
    archive = archive_map()
    outputs = step2_dev.simulate("expert_replay",specs,"three_view_unified_replay",
                                 archives=archive,lanes=(2,0))
    audits = []
    for r in specs:
        case = r["case_id"]
        result, diag, path = outputs[case]
        roll = diag.get("policy_rollout",{})
        contacts = diag.get("forbidden_contact_pairs",{})
        flagged = [name for name in contacts if any(all(x in name for x in pair) for pair in PAIRS)]
        audits.append({"case":case,"status":roll.get("status",result["status"]),"part_hit":roll.get("part_hit"),
                       "alignment":diag.get("replay_alignment"),"target_self_contacts":flagged,
                       "forbidden":contacts,"path":path})
        print("native expert",case,audits[-1]["status"],flagged,flush=True)
    write_json(WORK/"expert_replay.json",{"episodes":audits,"count":len(audits),
               "exact_replay_passes":sum(r["status"]=="success" and r["part_hit"] is True for r in audits),
               "selected_pairs_observed":sum(bool(r["target_self_contacts"]) for r in audits),"allowlisted":[]})


def clone(specs, decay, name):
    path = WORK/"prepared"
    path.mkdir(exist_ok=True)
    output = []
    for i, row in enumerate(specs):
        use = dict(row)
        old = row["case_id"]
        alias = f"{old[0]}_{5300+i}"
        plan = read(row["plan"])
        plan["case"].update(case_id=alias,te_decay=decay)
        p = path/f"{name}_{alias}.json"
        write_json(p,plan)
        use.update(case_id=alias,plan=str(p),original_case=old,te_decay=decay)
        output.append(use)
    return output


def evaluate(name,specs,binding,decay,training, *, anchor=None, detector=None, gpu_pool=None):
    step2_dev.WORK = WORK
    todo = clone(specs,decay,name)
    prompts = {r["case_id"]:r["head_prompt"] for r in todo}
    claim_gpu = step2_dev.wait_gpu
    lanes = tuple(int(gpu) for gpu in os.environ.get(
        "V3_QUERY_EVAL_GPUS", "1,2,0").split(","))
    minimum = int(os.environ.get("V3_QUERY_MIN_MEM_GIB", "0"))
    if not lanes or len(set(lanes)) != len(lanes) or any(gpu not in (0, 1, 2) for gpu in lanes):
        raise ValueError("Invalid evaluator GPU lanes")


    def free_gpu(preferred,exclude=(),timeout=3600):
        if preferred in (0,2):
            replay_groups = {(r["target_usd"],tuple(sorted((f["name"],f["usd"]) for f in r["fixtures"])))
                             for r in cases()}
            lane = (2,0).index(preferred)
            for i in range(lane,len(replay_groups),2):
                shard = f"{WORK.name}_expert_replay_{i:02}"
                while True:
                    markers=[RUNS/f"{shard}.complete", *RUNS.glob(f"{shard}_retry*.complete")]
                    if any(p.exists() and read(p)["outcome"]=="PASS" for p in markers):
                        break
                    final = RUNS/f"{WORK.name}_expert_replay.complete"
                    if final.exists():
                        if read(final)["outcome"]!="PASS":
                            raise RuntimeError("Expert replay exited before the assigned GPU could be released")
                        break
                    time.sleep(15)
        while minimum:
            available = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:")) / (1024*1024)
            if available >= minimum:
                break
            print(f"Waiting for host memory before {name}: {available:.1f} GiB available, need {minimum} GiB",
                  flush=True)
            time.sleep(30)
        return claim_gpu(preferred,exclude,timeout)

    step2_dev.wait_gpu = free_gpu
    try:
        got = step2_dev.simulate(name,todo,"three_view_closed_loop",policy=binding,detector=detector,
                                prompts=prompts,lanes=None if gpu_pool else lanes,allocator=gpu_pool,
                                resume_completed=gpu_pool is not None or name.startswith("full_"))
    finally:
        step2_dev.wait_gpu = claim_gpu
    labels = archive_map() if training else {
        r["case_id"]:{"path":r["expert_path"]} for r in specs}
    baseline = {r["case"]:r for r in anchor["episodes"]} if anchor else {}
    episodes, groups, part_pairs = [], defaultdict(list), defaultdict(list)
    for spec in todo:
        key = spec["original_case"]
        value = got[spec["case_id"]]
        log = log_case(spec,value,"b" if detector is not None else "a")
        raw, roll = value[1],value[1]["policy_rollout"]
        execution = log["diagnostics"]["policy_execution"]
        if execution!={"actions_per_decision":1,"temporal_decay":decay} or any(
                d["steps_applied"]!=1 or d["temporal_ensemble_chunks"]!=min(8,i+1)
                for i,d in enumerate(roll["decisions"])):
            raise ValueError(f"TE contract broken for {key}")
        if anchor and (baseline[key]["seed"]!=log["seed"] or baseline[key]["head_prompt"]!=log["head_prompt"]
                       or np.linalg.norm(np.subtract(baseline[key]["settled_target"][:3],log["settled_target"][:3]))>.001):
            raise ValueError(f"Comparison changed initial scenario: {key}")
        gate = roll["gate"]
        log.update(case=key,eval_case=spec["case_id"],pair_group_id=spec["pair_group_id"],
                   alignment=alignment(raw,labels[key]["path"]),wall_s=roll["wall_s"],
                   first_collision=log["diagnostics"]["collision"].get("approach",{}).get("first_contact"),
                   loaded_first=next((g["step"] for g in gate if g["target_load_ready"]),None),
                   stable_max=max(g["jaw_stable_steps"] for g in gate))
        episodes.append(log)
        part_pairs[(spec["bucket"],spec["pair_group_id"])].append(log)
        for phase in ("all",log["bucket"],log["part"],"obstacle" if log["obstacle"] else "non_obstacle"):
            groups[phase].append(log)
        print(name,key,log["status"],"Part Hit",log["part_hit"],"steps",log["steps"],flush=True)
    pairs = [{"asset":asset,"group_id":group,"cases":[r["case"] for r in items],
              "joint_pass":len(items)==2 and all(r["status"]=="success" and r["part_hit"] is True for r in items)}
             for (asset,group),items in part_pairs.items()]
    wins = sum(r["status"]=="success" and r["part_hit"] is True for r in episodes)
    successful_assets = sorted({r["asset"] for r in pairs if r["joint_pass"]})
    cutting = [r["case"] for r in episodes if r["obstacle"] and (
        (r["alignment"]["40"] is not None and r["alignment"]["40"]["space_deg"] >=10)
        or any("/World/Obstacle" in label for label in (r["first_collision"] or {}).get("pairs",[])))]
    result = {"m":decay,"episodes":episodes,"pairs":pairs,"strata":{x:summarize(y) for x,y in groups.items()},
              "success_and_hit":wins,"pair_assets":successful_assets,"possible_detour_deviation":cutting,
              "passed":len(episodes)==10 and wins>=7 and set(successful_assets)==ASSETS}
    write_json(WORK/f"{name}.json",result)
    record_complete(RUNS/f"{WORK.name}_{name}",0)
    return result


def choose(results):
    def priority(r):
        s = r["strata"]["all"]
        return (r["passed"],r["success_and_hit"],len(r["pair_assets"]),s["success"],-s["collision"]["approach"]["episodes"])
    return max(results,key=priority)


def start_full_fit(job):
    candidates = sorted(step2_dev.gpu_manifest(),key=lambda x:-x["free_mib"])
    while candidates[0]["free_mib"]<16*1024:
        print("Awaiting available GPU memory for the full fit",flush=True)
        time.sleep(30)
        candidates = sorted(step2_dev.gpu_manifest(),key=lambda x:-x["free_mib"])
    gpu = candidates[0]
    with (RUNS/f"{job}.log").open("w") as file:
        proc = subprocess.Popen(["bash",str(ROOT/"run_v3_query_fit.sh"),"full",gpu["uuid"]],
                                cwd=ROOT,stdout=file,stderr=subprocess.STDOUT)
        (RUNS/f"{job}.pid").write_text(str(proc.pid))
        print(f"Full V3-A training launched on GPU {gpu['gpu']}",flush=True)
        if proc.wait():
            raise RuntimeError("Conditional full training failed")


def run_full(locked,train_specs, *, reuse_fit=False):
    job = FIT_FULL.name
    if not reuse_fit:
        start_full_fit(job)
    wait_pass(job)
    fit = read(FIT_FULL/"result.json")
    curve = read(FIT_FULL/"loss_curves.json")
    if not (len(curve)==20 and fit["updates"]==21500 and fit["config"]["unique_close_windows"]==4733
            and fit["validation_windows"]==9826 and all(r["windows"]==r["seen_unique"]==34373 for r in curve)):
        raise ValueError("Full training supervision contract changed")
    frozen = read(STEP2/"development.json")
    if frozen["count"]!=28 or not frozen["replay_verified"]:
        raise ValueError("The 28-case frozen dev suite is unavailable")
    report = {}
    _, pairs = pair_plans()
    probe_manifest = read(FIT_GATE/"pilot_split.json")
    for checkpoint in ("best_val","last"):
        file = bind(FIT_FULL,checkpoint,f"full_{checkpoint}")
        probe = swap(file["path"],probe_manifest,pairs)
        write_json(WORK/f"full_{checkpoint}_swap.json",probe)
        train = evaluate(f"full_{checkpoint}_train",train_specs,file,locked,True)
        dev = evaluate(f"full_{checkpoint}_dev",frozen["cases"],file,locked,False)
        report[checkpoint] = {"train10":train["strata"],"dev28":dev["strata"],"prompt_swap":probe}
    write_json(WORK/"full_result.json",{"fit":fit,"m":locked,"results":report})


def run():
    wait_pass(FIT_GATE.name)
    fitted = read(FIT_GATE/"result.json")
    curve = read(FIT_GATE/"loss_curves.json")
    if (fitted["training_windows"]!=1055 or fitted["updates"]!=4620 or len(curve)!=140
            or any(x["seen_unique"]!=1055 for x in curve)):
        raise ValueError("Gate must cover 1,055 windows for 140 complete epochs")
    spec = cases()
    file = bind(FIT_GATE,"best_fit","gate")
    _,pairs = pair_plans()
    probe = swap(file["path"],read(FIT_GATE/"pilot_split.json"),pairs)
    write_json(WORK/"gate_prompt_swap.json",probe)
    print("Five paired steering probes complete; starting m=0.01 gate",flush=True)
    first = evaluate("gate_m001",spec,file,.01,True)
    candidates = [first]
    if not first["passed"] or first["possible_detour_deviation"]:
        print("Starting the same-checkpoint m=0.10 supplemental gate",flush=True)
        candidates.append(evaluate("gate_m010",spec,file,.10,True,anchor=first))
    winner = choose(candidates)
    verdict = {"passed":winner["passed"],"locked_m":winner["m"] if winner["passed"] else None,
               "best_m":winner["m"],"checkpoint":file["path"],
               "tested":[{"m":r["m"],"success_and_hit":r["success_and_hit"],"pair_assets":r["pair_assets"],
                          "passed":r["passed"],"summary":r["strata"]["all"]} for r in candidates]}
    write_json(WORK/"gate_verdict.json",verdict)
    print("GATE_VERDICT",verdict,flush=True)
    record_complete(RUNS/f"{WORK.name}_gate",0)
    wait_pass(f"{WORK.name}_expert_replay")
    if winner["passed"]:
        run_full(winner["m"],spec)
    write_json(WORK/"result.json",{"complete":True,"gate_passed":winner["passed"],
                                   "full_started":winner["passed"],"status":"full_completed" if winner["passed"] else "paused_after_gate"})


def resume_full():
    verdict = read(WORK/"gate_verdict.json")
    if not verdict["passed"] or verdict["locked_m"] not in (.01,.10):
        raise ValueError("A passing fixed TE gate is required before resuming the full evaluation")
    wait_pass(f"{WORK.name}_expert_replay")
    wait_pass(FIT_FULL.name)
    print("Resuming closed-loop evaluation with the original checkpoints and locked TE",flush=True)
    run_full(verdict["locked_m"],cases(),reuse_fit=True)
    write_json(WORK/"result.json",{"complete":True,"gate_passed":True,
                                   "full_started":True,"status":"full_completed"})


if __name__=="__main__":
    jobs={"expert":replay_experts,"gate":run,"resume":resume_full}
    task=sys.argv[1]
    stem=f"{WORK.name}_expert_replay" if task=="expert" else WORK.name
    try:
        jobs[task]()
    except BaseException:
        record_complete(RUNS/stem,1)
        raise
    else:
        record_complete(RUNS/stem,0)
