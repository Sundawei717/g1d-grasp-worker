"""Run the approved strict B20 and A40 comparison with resource-gated simulation."""
import fcntl
import time
from types import SimpleNamespace

import torch

from background_status import record_complete
from step2_eval import compare
from three_view_detector import sha256, write_json
from three_view_eval import read
from three_view_train import read_boxes
from v3_a_micro_forensics import pair_plans, swap
from v3_detector import cache_key, detector_config
from v3_followup_fit_watch import NAME, RUNS
from v3_followup_gpu import EvaluatorPool
from v3_prompt_train import STEP2, TIER
import v3_query_gate


WORK = RUNS/NAME
BASELINE = RUNS/"v3_a_query_gate_01"
BASE_FIT = RUNS/"v3_a_query_gate_01_full_fit"
CACHE = TIER/"detection_cache.json"


def await_training(task):
    marker = RUNS/f"{NAME}_{task}_fit.complete"
    while not marker.exists():
        all_done = RUNS/f"{NAME}_fit.complete"
        if all_done.exists() and read(all_done)["outcome"] != "PASS":
            raise RuntimeError(f"Training supervisor stopped with {task} pending")
        time.sleep(30)
    if read(marker)["outcome"] != "PASS":
        raise RuntimeError(f"{task} failed")
    return RUNS/f"{NAME}_{task}_fit"


def setup(task, folder):
    original = read(BASE_FIT/"result.json")
    result = read(folder/"result.json")
    cfg = result["config"]
    is_b = task == "b20"
    epochs = 20 if is_b else 40
    expected = dict(original["config"])
    if is_b:
        expected.update(variant="B",wrist_channels=4,detector_config_sha256=cache_key(detector_config()),
                        split_wrist_box=True)
        expected["low_lr_parameter_names"] = [
            "right_wrist_stem.rgb_weight" if name=="right_wrist_stem.weight" else name
            for name in expected["low_lr_parameter_names"]]
        added = "right_wrist_stem.prompt_weight"
        high = cfg["high_lr_parameter_names"]
        if high.count(added)!=1 or [name for name in high if name!=added]!=expected["high_lr_parameter_names"]:
            raise ValueError("Only the new box kernel may be added to B's high learning-rate group")
        expected["high_lr_parameter_names"] = high
    else:
        expected.update(epochs=40,steps=43000,warmup_updates=2150)
    if cfg != expected:
        mismatch = {k:(expected.get(k),cfg.get(k)) for k in expected.keys() | cfg.keys() if expected.get(k)!=cfg.get(k)}
        raise ValueError(f"Shared training contract mismatch: {mismatch}")
    if (not result["complete_optimization"] or result["updates"]!=1075*epochs
            or result["training_windows"]!=34373 or result["validation_windows"]!=9826):
        raise ValueError("Training stopped before a complete fixed epoch traversal")
    old_curve = read(BASE_FIT/"loss_curves.json")
    curve = read(folder/"loss_curves.json")
    if (len(curve)!=epochs
            or any(e["seen_unique"]!=34373 or e["windows"]!=34373 or e["train"]["close_terms"]!=37864 for e in curve)
            or any(a["order_sha256"]!=b["order_sha256"] for a,b in zip(old_curve,curve))):
        raise ValueError("Epoch order or supervision differs")
    if is_b and any(a["lr_scale"]!=b["lr_scale"] for a,b in zip(old_curve,curve,strict=True)):
        raise ValueError("B20 learning-rate trajectory differs")
    stride = 5 if is_b else 10
    if {f.name for f in folder.glob("epoch_*.pt")} != {f"epoch_{i:02}.pt" for i in range(stride,epochs+1,stride)}:
        raise ValueError("Missing scheduled checkpoints")
    base = torch.load(BASE_FIT/"last.pt",map_location="cpu",weights_only=True)
    candidate = torch.load(folder/"last.pt",map_location="cpu",weights_only=True)
    if base["normalization"]!=candidate["normalization"]:
        raise ValueError("The same observations must use identical state and action scaling")
    picks = ("best_val","last") if is_b else (("best_val","last") if result["best_epoch"]<35 else ("last",))
    write_json(WORK/f"{task}_protocol.json",
               {"same_sampler_orders_first_20":True,"same_proprioception_and_action_scaling":True,
                "only_config_differences":{k:v for k,v in cfg.items() if original["config"].get(k)!=v},
                "picks":picks,"training_receipt":result})
    return result,picks


def bind(task, folder, result, pick):
    path = folder/f"{pick}.pt"
    digest = sha256(path)
    cp = torch.load(path,map_location="cpu",weights_only=True)
    if (digest!=(result["last_sha256"] if pick=="last" else result["checkpoint_sha256"])
            or cp["config"]!=result["config"]
            or cp["epoch"]!=(cp["config"]["epochs"] if pick=="last" else result["best_epoch"])):
        raise ValueError(f"Changed checkpoint or training receipt: {path}")
    receipt = WORK/f"{task}_{pick}_binding.json"
    write_json(receipt,{**result,"checkpoint":str(path),"checkpoint_sha256":digest})
    return {"path":str(path),"receipt":str(receipt),"sha256":digest}


def collect_pairs(anchor, comparison, scope):
    baseline = {r["case"]:r for r in anchor["episodes"]}
    candidate = {r["case"]:r for r in comparison["episodes"]}
    r = compare({"a":baseline,"b":candidate},scope)
    for row in r["paired"]:
        k = row["case"]
        a,b = baseline[k],candidate[k]
        row["a_success_and_hit"] = a["status"]=="success" and a["part_hit"] is True
        row["b_success_and_hit"] = b["status"]=="success" and b["part_hit"] is True
    return r


def run():
    v3_query_gate.WORK = WORK
    decay = read(BASELINE/"gate_verdict.json")["locked_m"]
    if decay != .01 or read(BASELINE/"result.json")["status"]!="full_completed":
        raise ValueError("The A20 baseline and fixed TE are required")
    training = v3_query_gate.cases()
    development = read(STEP2/"development.json")
    if len(training)!=10 or len(development["cases"])!=28 or not development["replay_verified"]:
        raise ValueError("Frozen diagnostic scenes must match the previous run")
    probe_manifest = read(RUNS/"v3_a_query_gate_01_gate_fit"/"pilot_split.json")
    _, pairs = pair_plans()
    completed = {}
    for task in ("b20","a40"):
        folder = await_training(task)
        result,picks = setup(task,folder)
        detector = detector_config() if task=="b20" else None
        if detector is not None:
            boxes, observed = read_boxes(probe_manifest,CACHE)
            if observed!=cache_key(detector):
                raise ValueError("Offline swap must use the same detector cache as the policy")
            contract = SimpleNamespace(config_sha256=observed)
        else:
            boxes = contract = None
        completed[task]={}
        for ckpt in picks:
            name = f"{task}_{ckpt}"
            binding = bind(task,folder,result,ckpt)
            proof = swap(binding["path"],probe_manifest,pairs,boxes=boxes,detector_contract=contract)
            if len(proof)!=10:
                raise ValueError("Conditioned swap must include all five paired starts and branch steps")
            write_json(WORK/f"{name}_swap.json",proof)
            for scope,specs in (("train",training),("dev",development["cases"])):
                stage = f"{name}_{scope}"
                baseline = read(BASELINE/f"full_{ckpt}_{scope}.json")
                marker = RUNS/f"{NAME}_{stage}.complete"
                cached = WORK/f"{stage}.json"
                if marker.exists() and read(marker)["outcome"]=="PASS" and cached.is_file():
                    candidate = read(cached)
                    if candidate["m"]!=decay or len(candidate["episodes"])!=len(specs):
                        raise ValueError(f"Bad evaluator continuation {stage}")
                else:
                    with EvaluatorPool(with_detector=detector is not None) as pool:
                        candidate = v3_query_gate.evaluate(stage,specs,binding,decay,scope=="train",
                                                           anchor=baseline,detector=detector,gpu_pool=pool)
                matchup = collect_pairs(baseline,candidate,scope)
                write_json(WORK/f"{stage}_paired.json",matchup)
                completed[task].setdefault(ckpt,{})[scope] = {
                    "baseline":baseline["strata"],"candidate":candidate["strata"],
                    "candidate_success_and_hit":candidate["success_and_hit"],
                    "a20_success_and_hit":baseline["success_and_hit"],
                    "pair_result":str(WORK/f"{stage}_paired.json")}
                write_json(WORK/"evaluation_progress.json",completed)
        record_complete(RUNS/f"{NAME}_{task}_eval",0)
    write_json(WORK/"result.json",{"complete":True,"m":decay,"results":completed})


if __name__=="__main__":
    code=1
    with (WORK/"evaluation.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run()
            code=0
        finally:
            record_complete(RUNS/NAME,code)
