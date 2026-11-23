"""Drain the active legacy shard and migrate the experiment to GPU leases."""
import os
from pathlib import Path
import signal
import subprocess
import time

from background_status import record_complete

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/"runs"
JOB="step3a_01_gpu_switch"
PY="/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python"
SPECS={
    "step3a_01_l1":["step3a_l1.py"],
    **{f"step3a_01_{stage}":["step3a_suites.py",stage] for stage in ("l2","l3","l4","pilots")},
    "v3_b40_step3a_01":["v3_b40_eval.py"],
}


def argv(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    except (FileNotFoundError,ProcessLookupError):
        return []


def kids(pid):
    try:
        children=list({int(p) for task in Path(f"/proc/{pid}/task").glob("*/children")
                       for p in task.read_text().split()})
    except (FileNotFoundError,ProcessLookupError):
        return []
    return children+[p for c in children for p in kids(c)]


def interrupt_legacy_helper():
    marker=RUNS/"step3a_01_l3_resume_contract_v2.pid"
    if marker.is_file():
        pid=int(marker.read_text().strip())
        if argv(pid)[-2:]==[b"resume_step3a_suite.sh",b"l3"]:
            for child in [*reversed(kids(pid)),pid]:
                try:
                    os.kill(child,signal.SIGTERM)
                except ProcessLookupError:
                    pass


def active_shards(old):
    active=[]
    for parent in old:
        for child in kids(parent):
            args=argv(child)
            if any(a.endswith(b"/run_lab_school.sh") for a in args):
                active.append(child)
    return active


def run():
    interrupt_legacy_helper()
    old={}
    for job,args in SPECS.items():
        if (RUNS/f"{job}.complete").is_file():
            continue
        pid=int((RUNS/f"{job}.pid").read_text())
        if argv(pid)[-len(args):]!=[arg.encode() for arg in args]:
            raise RuntimeError(f"Cannot verify current coordinator for {job}")
        old[pid]=(job,args)
    paused=[]
    try:
        for pid in old:
            os.kill(pid,signal.SIGSTOP)
            paused.append(pid)
        print("Waiting for the current Isaac shard to finish",active_shards(old),flush=True)
        deadline=time.monotonic()+5400
        while active_shards(old):
            if time.monotonic()>deadline:
                raise RuntimeError("An Isaac shard outlasted its time limit")
            time.sleep(10)
        for pid in old:
            os.kill(pid,signal.SIGTERM)
            os.kill(pid,signal.SIGCONT)
        limit=time.monotonic()+45
        while any(argv(pid) for pid in old):
            if time.monotonic()>limit:
                raise RuntimeError("Previous simulator coordinators are still present")
            time.sleep(1)
        paused.clear()
        for job,args in old.values():
            with (RUNS/f"{job}.log").open("a") as log:
                log.write("Resuming from saved results with per-GPU leases and a shared RAM budget.\n")
                log.flush()
                worker=subprocess.Popen([PY,"-Bu",*args],cwd=ROOT,
                    env={**os.environ,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"},
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (RUNS/f"{job}.pid").write_text(f"{worker.pid}\n")
            print(job,worker.pid,flush=True)
    finally:
        for pid in paused:
            try:
                os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:
                pass


if __name__=="__main__":
    code=1
    try:
        run()
        code=0
    finally:
        record_complete(RUNS/JOB,code)
