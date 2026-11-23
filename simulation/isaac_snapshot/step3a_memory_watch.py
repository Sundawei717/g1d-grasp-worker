"""Protect completed work if shared host RAM falls during a batch."""
import os
from pathlib import Path
import re
import signal
import time

from background_status import record_complete

RUNS=Path(__file__).resolve().parent/"runs"
JOB="step3a_01_memory_guard"
SHARD=re.compile(
    r"(?:step3a_01_(?:l[1-4]_r\d+|pilots_[A-Za-z0-9_]+)_(?:demo|replay)_\d+"
    r"|v3_b40_step3a_01_b40_(?:last|best_val)_(?:train|dev)_\d+)(?:_retry\d+)?")

def children(pid):
    try:
        return [int(x) for x in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []

def command(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []

def ours():
    for path in RUNS.glob("*.pid"):
        if not SHARD.fullmatch(path.stem) or path.with_suffix(".complete").is_file():
            continue
        try:
            parent=int(path.read_text().strip())
        except (ValueError,OSError):
            continue
        run_dir=str(path.with_suffix("")).encode()
        if run_dir not in command(parent):
            continue
        for wrapper in children(parent):
            for pid in children(wrapper):
                argv=command(pid)
                if run_dir in argv and any(arg.endswith(b"/lab_school_probe.py") for arg in argv):
                    yield path.stem,pid

def ram_gib():
    fields=next(line.split() for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:"))
    return int(fields[1])/(1024*1024)

def run():
    low=0
    requests={}
    while not (RUNS/"step3a_01_report.complete").is_file():
        ram=ram_gib()
        low=low+1 if ram<8 else 0
        for name,pid in ours():
            key=name,pid
            now=time.monotonic()
            if low>=2 and key not in requests:
                try:
                    os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:
                    continue
                requests[key]=now
                print(f"Terminated {name} PID {pid} at {ram:.2f} GiB RAM",flush=True)
            elif low>=2 and key in requests and requests[key] is not None and now-requests[key]>45:
                try:
                    os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:
                    continue
                requests[key]=None
                print(f"Stopped stalled {name} PID {pid} at {ram:.2f} GiB RAM",flush=True)
        time.sleep(15)

if __name__=="__main__":
    code=1
    try:
        run()
        code=0
    finally:
        record_complete(RUNS/JOB,code)
