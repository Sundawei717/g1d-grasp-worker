"""Verify and evaluate the sole right-wrist variable against the frozen A40 reference."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
from threading import Lock
import time
from types import SimpleNamespace

import torch

from background_status import record_complete
from step3a_resources import FIT, PRIVATE_STARTUP_GIB, SimAllocator, devices, ram_gib
from three_view_detector import write_json
from three_view_eval import read
from three_view_train import read_boxes
from v3_a_micro_forensics import pair_plans,swap
from v3_detector import cache_key,detector_config
from v3_followup_eval import bind,collect_pairs,CACHE
from v3_followup_fit_watch import RUNS
from v3_prompt_train import STEP2
import v3_query_gate


JOB = "v3_b40_step3a_01"
WORK = RUNS/JOB
A40 = RUNS/"v3_step2_followup_01"
A40_FIT=RUNS/"v3_step2_followup_01_a40_fit"


class ComparisonAllocator(SimAllocator):
    """Use one reserved lane and allow extra cards when available."""

    def __init__(self):
        super().__init__(with_detector=True)
        self.primary=int(os.environ["B40_RESERVED_GPU"])
        self.primary_use=Lock()

    def primary_card(self):
        active=0
        for path in RUNS.glob("step3a_01_gpu_*.lock"):
            if path.name==f"step3a_01_gpu_{self.primary}.lock":
                continue
            with path.open("a") as lock:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    active+=1
        if ram_gib() < 34+PRIVATE_STARTUP_GIB*active:
            return None
        return next(({**gpu,"gpu":self.primary} for gpu in devices()
                     if gpu["index"]==self.primary and gpu["free_mib"]>=22*1024),None)

    @contextmanager
    def claim(self,job):
        while True:
            if self.primary_use.acquire(blocking=False):
                try:
                    gpu=self.primary_card()
                    if gpu:
                        print(f"Primary B40 lane GPU {self.primary} for {job}",flush=True)
                        yield gpu
                        return
                finally:
                    self.primary_use.release()
            allocation,ram,needed,count=self.reserve(job)
            if allocation:
                gpu,lock=allocation
                with lock:
                    print(f"Additional B40 lane GPU {gpu['index']} for {job}",flush=True)
                    yield gpu
                return
            print(f"Waiting for {job}: host {ram:.1f}/{needed} GiB, "
                  f"active reservations {count}",flush=True)
            time.sleep(20)


def verify(fit, detector):
    original=read(A40_FIT/"result.json")
    candidate=read(fit/"result.json")
    base,cfg=original["config"],candidate["config"]
    expected={**base,"variant":"B","wrist_channels":4,"split_wrist_box":True,
              "detector_config_sha256":cache_key(detector)}
    expected["low_lr_parameter_names"]=["right_wrist_stem.rgb_weight" if n=="right_wrist_stem.weight" else n
                                        for n in base["low_lr_parameter_names"]]
    extra="right_wrist_stem.prompt_weight"
    high=cfg["high_lr_parameter_names"]
    if high.count(extra)!=1 or [n for n in high if n!=extra]!=base["high_lr_parameter_names"]:
        raise ValueError("B40 requires the separate high-LR fourth kernel")
    expected["high_lr_parameter_names"]=high
    if cfg!=expected:
        raise ValueError({k:(expected.get(k),cfg.get(k)) for k in expected.keys()|cfg.keys() if expected.get(k)!=cfg.get(k)})
    prev=read(A40_FIT/"loss_curves.json")
    new=read(fit/"loss_curves.json")
    assert len(new)==len(prev)==40 and candidate["updates"]==43000 and candidate["complete_optimization"]
    assert candidate["training_windows"]==34373 and candidate["validation_windows"]==9826
    for a,b in zip(prev,new,strict=True):
        assert (a["lr_scale"],a["order_sha256"],a["seen_unique"])==(b["lr_scale"],b["order_sha256"],34373)
        assert a["train"]["close_terms"]==b["train"]["close_terms"]==37864
    assert {p.name for p in fit.glob("epoch_*.pt")}=={f"epoch_{n:02}.pt" for n in (10,20,30,40)}
    def norm(p):
        return torch.load(p/"last.pt",map_location="cpu",weights_only=True)["normalization"]
    assert norm(fit)==norm(A40_FIT)
    write_json(WORK/"strict_protocol.json",{"epochs":40,"same_epoch_orders":True,"same_lr_cycles":True,
        "same_state_action_normalization":True,"baseline":str(A40_FIT),"detector":detector,
        "allowable_config_differences":[k for k in cfg if cfg[k]!=base.get(k)]})
    return candidate


def run():
    import v3_followup_eval
    marker=RUNS/f"{FIT}.complete"
    while not marker.exists():
        time.sleep(30)
    if read(marker)["outcome"]!="PASS":
        raise RuntimeError("The B40 fit failed")
    fit=RUNS/FIT
    detector=detector_config()
    result=verify(fit,detector)
    v3_query_gate.WORK=WORK
    v3_followup_eval.WORK=WORK
    probe=read(RUNS/"v3_a_query_gate_01_gate_fit/pilot_split.json")
    cache,digest=read_boxes(probe,CACHE)
    assert digest==cache_key(detector)
    _,pairs=pair_plans()
    dev=read(STEP2/"development.json")
    assert dev["replay_verified"] and len(dev["cases"])==28
    cases=v3_query_gate.cases()
    assert len(cases)==10
    completed={}
    gpu_pool=ComparisonAllocator() if "B40_RESERVED_GPU" in os.environ else SimAllocator(with_detector=True)
    for checkpoint in ("last","best_val") if result["best_epoch"]<35 else ("last",):
        bound=bind("b40",fit,result,checkpoint)
        response=swap(bound["path"],probe,pairs,boxes=cache,detector_contract=SimpleNamespace(config_sha256=digest))
        assert len(response)==10
        write_json(WORK/f"b40_{checkpoint}_swap.json",response)
        completed[checkpoint]={}
        for scope,rows in (("train",cases),("dev",dev["cases"])):
            stage=f"b40_{checkpoint}_{scope}"
            baseline=read(A40/f"a40_last_{scope}.json")
            output=WORK/f"{stage}.json"
            if output.is_file() and (RUNS/f"{JOB}_{stage}.complete").is_file():
                candidate=read(output)
            else:
                candidate=v3_query_gate.evaluate(stage,rows,bound,.01,scope=="train",anchor=baseline,
                    detector=detector,gpu_pool=gpu_pool)
            match=collect_pairs(baseline,candidate,scope)
            write_json(WORK/f"{stage}_paired.json",match)
            completed[checkpoint][scope]={"A40":baseline["strata"],"B40":candidate["strata"],"pairs":match["strata"]["all"]["paired"],
                                           "A40_success_and_hit":baseline["success_and_hit"],
                                           "B40_success_and_hit":candidate["success_and_hit"]}
            write_json(WORK/"evaluation_progress.json",completed)
    write_json(WORK/"result.json",{"complete":True,"best_epoch":result["best_epoch"],"results":completed})


if __name__=="__main__":
    WORK.mkdir(exist_ok=True)
    code=1
    try:
        run()
        code=0
    finally:
        record_complete(RUNS/JOB,code)
