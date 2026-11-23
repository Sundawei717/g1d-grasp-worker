"""Start the two frozen V3-Final fits within the shared server memory budget."""
import os
from pathlib import Path
import subprocess
import sys
import time

from background_status import record_complete
from three_view_detector import write_json
from three_view_eval import read


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
NAME = "v3_step2_followup_01"
TASKS = ("b20", "a40")


def ram_gib():
    line = next(row for row in Path("/proc/meminfo").read_text().splitlines() if row.startswith("MemAvailable:"))
    return int(line.split()[1]) / (1024 * 1024)


def devices():
    lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True).splitlines()
    result = []
    for row in lines:
        index, uuid, free, busy = (v.strip() for v in row.split(","))
        result.append({"index":int(index), "uuid":uuid, "free_mib":int(free), "utilization":int(busy)})
    return sorted(result, key=lambda gpu:(gpu["utilization"], -gpu["free_mib"]))


def start(task, gpu, workers):
    job = f"{NAME}_{task}_fit"
    assert not (RUNS/job).exists() and not (RUNS/f"{job}.complete").exists()
    with (RUNS/f"{job}.log").open("x") as log:
        proc = subprocess.Popen(["bash", str(ROOT/"run_v3_followup_fit.sh"),task,gpu["uuid"],str(workers)],
                                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    (RUNS/f"{job}.pid").write_text(str(proc.pid))
    print(f"{task} launched on GPU {gpu['index']}, PID {proc.pid}, {workers} loaders per split", flush=True)
    return proc


class AdoptedFit:
    def __init__(self, task, pid):
        self.job = f"{NAME}_{task}_fit"
        self.pid = pid

    def poll(self):
        marker = RUNS/f"{self.job}.complete"
        if marker.exists():
            return read(marker)["exit_code"]
        command = Path(f"/proc/{self.pid}/cmdline")
        try:
            args = command.read_bytes().split(b"\0")
        except FileNotFoundError:
            return 1
        return None if b"run_v3_followup_fit.sh" in [Path(os.fsdecode(x)).name.encode() for x in args] else 1


def main(resume=False):
    work = RUNS/NAME
    pending = list(TASKS)
    live, finished, launches = {}, {}, []
    if resume:
        state = read(work/"fit_status.json")
        for record in state["launches"]:
            task = record["task"]
            if task not in pending:
                continue
            pid = state["live"].get(task)
            job = f"{NAME}_{task}_fit"
            if (RUNS/f"{job}.complete").exists():
                finished[task] = read(RUNS/f"{job}.complete")["exit_code"]
            elif pid and (RUNS/job).exists():
                live[task] = AdoptedFit(task,pid),record["gpu"]
            else:
                continue
            pending.remove(task)
            launches.append(record)
    else:
        work.mkdir(exist_ok=False)
    while pending or live:
        for task,(proc,gpu) in list(live.items()):
            code = proc.poll()
            if code is None:
                continue
            finished[task] = code
            del live[task]
            print(f"{task} training exited {code}", flush=True)
        free = ram_gib()
        gpus = devices()
        usable = [gpu for gpu in gpus if gpu["free_mib"]>=18*1024
                  and gpu["index"] not in {g["index"] for _,g in live.values()}]
        before = {"ram_available_gib":free,"gpus":gpus}
        if pending and not launches:
            write_json(work/"initial_resources.json",before)
        if pending and usable and free >= (36 if live else 23):
            size = 2 if not live and len(pending)>=2 and len(usable)>=2 and free>=36 else 1
            for gpu in usable[:size]:
                task = pending.pop(0)
                workers = 2
                live[task] = start(task,gpu,workers),gpu
                launches.append({"task":task,"gpu":gpu,"loader_workers":workers,"start_unix_s":time.time(),
                                 "ram_available_gib":free})
        write_json(work/"fit_status.json",{"pending":pending,"finished":finished,"launches":launches,
                 "live":{task:proc.pid for task,(proc,_) in live.items()},**before})
        if pending or live:
            time.sleep(25)
    for task in TASKS:
        if finished[task] != 0 or read(RUNS/f"{NAME}_{task}_fit.complete")["outcome"]!="PASS":
            raise RuntimeError(f"Training failed for {task}")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main(resume="resume" in sys.argv[1:])
    finally:
        record_complete(RUNS/f"{NAME}_fit", code)
