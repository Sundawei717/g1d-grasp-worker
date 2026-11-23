"""Resume the pilot coordinator after keeping any in-progress recording."""
import os
from pathlib import Path
import signal
import sys
import subprocess
import time

from background_status import record_complete
from step3a_switch_gpu import active_shards,argv,ROOT,RUNS,PY

BATCH="step3a_01_pilots"
PARALLEL="--parallel" in sys.argv
JOB="step3a_01_parallel_pilots" if PARALLEL else "step3a_01_pilot_reload"


def main():
    if PARALLEL:
        while not (RUNS/"step3a_01/pilots_r3_result.json").is_file():
            if (RUNS/f"{BATCH}.complete").is_file():
                raise RuntimeError("Pilot coordinator stopped before finishing the active round")
            time.sleep(10)
    pidfile=RUNS/f"{BATCH}.pid"
    pid=int(pidfile.read_text())
    if argv(pid)[-2:]!=[b"step3a_suites.py",b"pilots"]:
        raise RuntimeError("The pilot coordinator changed")
    os.kill(pid,signal.SIGSTOP)
    try:
        print("Keeping active pilot shards",active_shards([pid]),flush=True)
        deadline=time.monotonic()+3600
        while active_shards([pid]):
            if time.monotonic()>deadline:
                raise RuntimeError("An active pilot shard exceeded its wall-clock guard")
            time.sleep(10)
        os.kill(pid,signal.SIGTERM)
    finally:
        try:
            os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:
            pass
    with (RUNS/f"{BATCH}.log").open("a") as log:
        log.write("Readopting finished pilot recordings with measured post-grasp contact attribution.\n")
        log.flush()
        args=["step3a_pilots_parallel.py"] if PARALLEL else ["step3a_suites.py","pilots"]
        p=subprocess.Popen([PY,"-Bu",*args],cwd=ROOT,
            env={**os.environ,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"},
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pidfile.write_text(f"{p.pid}\n")
    print("Pilot coordinator",p.pid,flush=True)


if __name__=="__main__":
    code=1
    try:
        main()
        code=0
    finally:
        record_complete(RUNS/JOB,code)
