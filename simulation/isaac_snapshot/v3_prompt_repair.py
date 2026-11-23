"""Run and audit the V3-A overfit gate before permitting the full 20-epoch fit."""
from collections import Counter, defaultdict
import os
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
import torch

from background_status import record_complete
from episode_store import jaw_width_m
import step2_dev
from step2_eval import log_case, stats as rollout_stats
from three_view_detector import sha256, write_json
from three_view_eval import read
from three_view_inputs import head_input
from three_view_rollout import ThreeViewPolicy
from three_view_train import ThreeViewWindows
from v3_prompt_train import ROOT, TIER, STEP2


RUNS = ROOT/"runs"
WORK = RUNS/"v3_a_prompt_repair_01"
TRAIN_A = RUNS/"step2_v3_01_train_a"
SPECS = STEP2/"diagnostic10.json"
SCOPES = {"mug":2,"basket":2,"carton":1}
STEP_20_MIN_RESPONSE_DEG = .16
STEP_20_MIN_GAIN = 10


def wait_pass(job):
    marker = RUNS/f"{job}.complete"
    while not marker.exists():
        pid = RUNS/f"{job}.pid"
        if pid.exists():
            proc = Path("/proc")/pid.read_text().strip()/"cmdline"
            if not proc.exists() or not proc.read_bytes():
                raise RuntimeError(f"{job} exited without a completion marker")
        time.sleep(20)
    if read(marker)["outcome"] != "PASS":
        raise RuntimeError(f"Failed prerequisite {marker}")


def bind(fit, selection, alias):
    r = read(fit/"result.json")
    if (r["config"]["variant"] != "A" or not r["config"]["prompt_fusion"]
            or r["config"]["wrist_channels"] != 3 or not r["complete_optimization"]):
        raise ValueError("Only finished V3-A prompt-fusion fits may be evaluated")
    path = fit/f"{selection}.pt"
    digest = sha256(path)
    expected = r["last_sha256"] if selection == "last" else r["checkpoint_sha256"]
    if digest != expected:
        raise ValueError(f"Unexpected checkpoint SHA256: {path}")
    receipt = WORK/f"{alias}_binding.json"
    write_json(receipt,{**r,"checkpoint":str(path),"checkpoint_sha256":digest})
    return {"receipt":str(receipt),"path":str(path),"sha256":digest}


def buckets(manifest):
    rows = [r for r in manifest["episodes"] if r["split"] == "train"]
    id_by_group = defaultdict(list)
    for i,row in enumerate(rows):
        id_by_group[row["pair_group_id"]].append((i,row))
    if (len(id_by_group) != 5 or any(k is None or len(v)!=2 for k,v in id_by_group.items())
            or Counter(pair[0][1]["bucket"] for pair in id_by_group.values()) != SCOPES):
        raise ValueError("Expected two mug, two basket, and one carton pair")
    return id_by_group


@torch.inference_mode()
def prompt_swap(checkpoint, manifest):
    policy = ThreeViewPolicy(checkpoint,"cuda:0")
    ds = ThreeViewWindows(manifest,"train",None,policy.norm,point_geometry=policy.prompt_fusion)
    lookup = {(i,t):n for n,(i,t,_) in enumerate(ds.windows) if t in (0,20)}
    groups = buckets(manifest)
    summary = defaultdict(list)
    detail = []
    for key,pair in groups.items():
        pair = sorted(pair,key=lambda item:item[1]["part"])
        # Evaluate both directions while fixing all RGB and state tensors within a direction.
        for (i,row),(alt_i,other) in (pair, pair[::-1]):
            path,prompt,*_ = ds.files[i]
            other_prompt = ds.files[alt_i][1]
            if prompt["bbox_0"] != other_prompt["bbox_0"] or prompt["affordance_0"] == other_prompt["affordance_0"]:
                raise ValueError(f"Bad prompt pair {key}")
            for t in (0,20):
                rgb = ds[lookup[i,t]][:5]
                with h5py.File(path) as h:
                    head,meta = head_input(h["obs/rgb_left"][t],prompt["bbox_0"],other_prompt["affordance_0"],
                                           point_geometry=policy.prompt_fusion)
                np.testing.assert_array_equal(rgb[0][:4],head[:4])
                np.testing.assert_array_equal(rgb[3][:6],meta[:6])
                options = {"base":rgb, "heatmap":(head,*rgb[1:]),
                           "coherent":(head,rgb[1],rgb[2],meta,rgb[4])}
                if policy.prompt_fusion:
                    options["geometry"] = (rgb[0],rgb[1],rgb[2],meta,rgb[4])
                batch = zip(*options.values(),strict=True)
                inputs = [torch.as_tensor(np.stack(axis),device=policy.device) for axis in batch]
                output = policy.model(*inputs).cpu().numpy()
                scale = np.asarray(policy.norm["action"]["scale"])[:7]
                for mode,pred in zip(list(options)[1:],output[1:],strict=True):
                    error = np.rad2deg(np.abs(pred[:,:7]-output[0,:,:7])*scale)
                    summary[(mode,t)].extend(error.ravel())
                    detail.append({"pair_group_id":key,"bucket":row["bucket"],"source":row["case"],
                                   "alternate":other["case"],"step":t,"mode":mode,
                                   "mean_deg":float(error.mean()),"max_deg":float(error.max())})
    return {"checkpoint":str(checkpoint), "pairs":len(groups), "details":detail,
            "summary":{f"{mode}_{step}":{"n":len(x),"median_deg":float(np.median(x)),
                                      "max_deg":float(np.max(x))} for (mode,step),x in summary.items()}}


def drift(r, row):
    log = read(r["path"])
    commands = np.asarray([[*cmd["arm_target_rad"], jaw_width_m(cmd["finger_target_m"])]
                           for cmd in log["commands"]])
    with h5py.File(row["path"]) as h:
        expert = np.column_stack((h["action/qpos_target"][:],h["action/gripper_width_target"][:]))
        phases = h["phase"].asstr()[:]
    applied_by_policy = len(log["policy_rollout"]["gate"])
    steps = min(applied_by_policy,len(commands),int(np.flatnonzero(phases=="lift")[0]))
    e = np.abs(commands[:steps]-expert[:steps])
    joint = np.rad2deg(e[:,:7]).mean(axis=1)
    jaw = e[:,7]*1000
    def snapshot(t):
        return {"joint_mae_deg":float(joint[t]),"jaw_error_mm":float(jaw[t])} if t<steps else None
    return {"compared_steps":steps,"at":{"0":snapshot(0),"20":snapshot(20),"40":snapshot(40)},
            "max_joint_mae_deg":float(joint.max()),
            "mae_series_deg":joint.tolist(),"jaw_abs_error_series_mm":jaw.tolist(),
            "forbidden_contacts":log.get("forbidden_contact_pairs",{}),
            "first_target_load_step":next((g["step"] for g in log["policy_rollout"]["gate"]
                                           if g["target_load_ready"]),None)}


def summarize(rows):
    base = rollout_stats(rows)
    base.pop("collision")
    base["collision"] = {}
    for phase in ("approach","lift","hold"):
        records = [r["diagnostics"]["collision"].get(phase,{}) for r in rows]
        base["collision"][phase] = {"episodes":sum(r.get("contact_physics_steps",0)>0 for r in records),
                                   "arm_link_onsets":sum(r.get("arm_link_collision_onsets",0) for r in records),
                                   "contact_physics_steps":sum(r.get("contact_physics_steps",0) for r in records)}
    return base


def evaluate(tag, specs, binding, training=False):
    step2_dev.WORK = WORK
    prompts = {r["case_id"]:r["head_prompt"] for r in specs}
    outputs = step2_dev.simulate(tag,specs,"three_view_closed_loop",policy=binding,prompts=prompts,lanes=(0,1,2))
    data = []
    archived = {r["case"]:r for r in read(TIER/"training_split.json")["episodes"]} if training else {}
    original = {r["case_id"]:r for r in specs}
    subsets = defaultdict(list)
    for case,response in outputs.items():
        s = original[case]
        r = log_case(s,response,"a")
        if r["diagnostics"].get("collision_phase_contract") != "pretrial_approach_then_lift_and_hold":
            raise ValueError(f"Collision phases were not upgraded for {case}")
        r["pair_group_id"] = s["pair_group_id"]
        if training:
            r["deviation"] = drift(r,archived[case])
        for label in ("all",r["bucket"],r["part"],"obstacle" if r["obstacle"] else "non_obstacle"):
            subsets[label].append(r)
        data.append(r)
        print(f"{tag} {case}: {r['status']}, Part Hit={r['part_hit']}",flush=True)
    all_pairs = defaultdict(list)
    for row in data:
        all_pairs[(row["bucket"],row["pair_group_id"])].append(row)
    pairs = [{"bucket":bucket,"pair_group_id":key,"cases":[r["case"] for r in pair],
              "both_success_and_hit":len(pair)==2 and all(r["status"]=="success" and r["part_hit"] is True for r in pair)}
             for (bucket,key),pair in all_pairs.items()]
    report = {"episodes":data,"strata":{k:summarize(v) for k,v in subsets.items()},"pairs":pairs}
    write_json(WORK/f"{tag}_comparison.json",report)
    return report


def verify_gate(specs, fit):
    binding = bind(fit,"best_fit","pilot")
    manifest = read(fit/"pilot_split.json")
    reference = prompt_swap(TRAIN_A/"last.pt",manifest)
    measured = prompt_swap(binding["path"],manifest)
    write_json(WORK/"pilot_prompt_swap.json",{"baseline":reference,"repaired":measured})
    print("Prompt swap",measured["summary"],flush=True)
    closed = evaluate("pilot_eval",specs,binding,training=True)
    measure = measured["summary"]["coherent_20"]["median_deg"]
    baseline = reference["summary"]["coherent_20"]["median_deg"]
    part_pass = all(any(p["bucket"]==bucket and p["both_success_and_hit"] for p in closed["pairs"])
                    for bucket in ("mug","basket"))
    wrong = [r["case"] for r in closed["episodes"] if r["status"]=="success"
             and r["bucket"] in {"mug","basket"} and r["part_hit"] is not True]
    gates = {"point_response":measure>=max(STEP_20_MIN_RESPONSE_DEG,STEP_20_MIN_GAIN*baseline),
             "at_least_7_successes":closed["strata"]["all"]["success"]>=7,
             "mug_and_basket_pairs":part_pass,"no_success_on_wrong_mug_or_basket_part":not wrong}
    result = {"passed":all(gates.values()),"criteria":gates,"wrong_part_successes":wrong,
              "baseline_t20_deg":baseline,"repaired_t20_deg":measure,
              "threshold_t20_deg":max(STEP_20_MIN_RESPONSE_DEG,STEP_20_MIN_GAIN*baseline),
              "closed_loop":closed["strata"]["all"],"pairs":closed["pairs"],
              "pilot_checkpoint":binding["path"]}
    write_json(WORK/"gate.json",result)
    print("Pilot gate",result,flush=True)
    record_complete(RUNS/f"{WORK.name}_gate",0)
    return result


def choose_training_gpu():
    while True:
        for gpu in sorted(step2_dev.gpu_manifest(),key=lambda gpu:-gpu["free_mib"]):
            if gpu["free_mib"] >= 16*1024:
                return gpu
        print("Waiting for a training GPU with 16 GiB available",flush=True)
        time.sleep(30)


def run_full(specs):
    if not read(WORK/"gate.json")["passed"]:
        raise RuntimeError("The full fit is gated by the pilot results")
    gpu = choose_training_gpu()
    job = f"{WORK.name}_full"
    fit = RUNS/job
    with (RUNS/f"{job}.log").open("w") as log:
        p = subprocess.Popen(["bash",str(ROOT/"run_v3_prompt_train.sh"),"full",gpu["uuid"]],stdout=log,
                             stderr=subprocess.STDOUT,cwd=ROOT)
        (RUNS/f"{job}.pid").write_text(str(p.pid))
        print(f"Full 20-epoch fit started on GPU {gpu['gpu']} PID {p.pid}",flush=True)
        if p.wait():
            raise RuntimeError("Full fit failed")
    wait_pass(job)
    curve = read(fit/"loss_curves.json")
    if (len(curve)!=20 or any(r["windows"]!=34373 or r["seen_unique"]!=34373 for r in curve)
            or curve[-1]["step"]!=21500):
        raise ValueError("Expected 20 complete epochs on the full Tier-1 split")
    dev = read(STEP2/"development.json")
    if dev["count"]!=28 or not dev["replay_verified"]:
        raise ValueError("Step 2 frozen, replay-verified development list required")
    results = {}
    manifest = read(RUNS/f"{WORK.name}_pilot/pilot_split.json")
    for name in ("best_val","last"):
        binding = bind(fit,name,f"full_{name}")
        result = prompt_swap(binding["path"],manifest)
        write_json(WORK/f"full_{name}_prompt_swap.json",result)
        results[name] = {"swap":result["summary"]}
        results[name]["train10"] = evaluate(f"full_{name}_train10",specs,binding,training=True)["strata"]
        results[name]["dev28"] = evaluate(f"full_{name}_dev28",dev["cases"],binding)["strata"]
    write_json(WORK/"full_result.json",{"passed":True,"fit":read(fit/"result.json"),"results":results})
    record_complete(RUNS/f"{WORK.name}_full_eval",0)


def run():
    WORK.mkdir(exist_ok=True)
    pilot = RUNS/f"{WORK.name}_pilot"
    wait_pass(pilot.name)
    specs = read(SPECS)["cases"]
    if len(specs)!=10:
        raise ValueError("The frozen pilot requires ten cases")
    gate = verify_gate(specs,pilot)
    if gate["passed"]:
        print("Pilot passed; proceeding to the full V3-A fit",flush=True)
        run_full(specs)
    else:
        print("Pilot did not pass; paused before the full fit",flush=True)
    write_json(WORK/"result.json",{"complete":True,"gate_passed":gate["passed"],"full_started":gate["passed"],
                                   "status":"full_completed" if gate["passed"] else "paused_after_gate"})


if __name__=="__main__":
    try:
        run()
    except BaseException:
        record_complete(RUNS/WORK.name,1)
        raise
    else:
        record_complete(RUNS/WORK.name,0)
