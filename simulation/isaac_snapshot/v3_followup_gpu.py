"""Admit physical shards according to live VRAM and host-memory reservations."""
from contextlib import contextmanager
import os
from pathlib import Path
import signal
from threading import Event, Lock, Thread
import time

from three_view_detector import write_json
from three_view_eval import read
from v3_followup_fit_watch import NAME, RUNS, devices, ram_gib


WORK = RUNS / NAME


def process_args(pid):
    proc = Path("/proc") / str(pid)
    try:
        if proc.stat().st_uid == os.getuid():
            return (proc/"cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return []


def children(pid):
    try:
        return [int(p) for p in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except (FileNotFoundError, ProcessLookupError):
        return []


def resident_gib(pid):
    if pid is None:
        return 0
    try:
        rows = Path(f"/proc/{pid}/status").read_text().splitlines()
        own = next((int(line.split()[1]) / 2**20 for line in rows if line.startswith("VmRSS:")),0)
    except (FileNotFoundError, ProcessLookupError):
        return 0
    return own + sum(resident_gib(child) for child in children(pid))


def owned_shards():
    active = {}
    for f in RUNS.glob(f"{NAME}_*_[0-9][0-9]*.pid"):
        job = f.stem
        if (RUNS/f"{job}.complete").exists():
            continue
        try:
            pid = int(f.read_text().strip())
        except (OSError, ValueError):
            continue
        if str(RUNS/job).encode() not in process_args(pid):
            continue
        try:
            env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            gpu = next(int(item.split(b"=")[1]) for item in env if item.startswith(b"G1D_RENDER_GPU="))
        except (OSError, StopIteration, ValueError):
            continue
        active[job] = (gpu,pid)
    return active


def required_ram(active_rss, with_detector, training):
    extra = 6 if training else 0
    if not active_rss:
        return (30 if with_detector else 28) + extra
    each = 26 if with_detector else 23
    pending = sum(max(0, each - .8*rss) for rss in active_rss)
    return 22 + each + pending + extra


class EvaluatorPool:
    def __init__(self, with_detector):
        self.with_detector = with_detector
        self.leases = {}
        self.lock = Lock()
        self.stop = Event()
        self.serial_marker = WORK/"simulation_serial_fallback.json"
        self.monitor = Thread(target=self.watch,daemon=True)

    def __enter__(self):
        self.monitor.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.monitor.join()

    def active(self):
        return {**{job:(gpu,None) for job,gpu in self.leases.items()},**owned_shards()}

    def training_gpus(self):
        state = read(WORK/"fit_status.json")
        return {run["gpu"]["index"] for run in state["launches"] if run["task"] in state["live"]}

    @contextmanager
    def claim(self, job):
        last_notice = -60.
        while True:
            with self.lock:
                active = self.active()
                training = self.training_gpus()
                occupied = training | {gpu for gpu,_ in active.values()}
                available = ram_gib()
                need = required_ram([resident_gib(pid) for _,pid in active.values()],self.with_detector,bool(training))
                cap = 1 if self.serial_marker.exists() else 3
                min_vram = (22 if self.with_detector else 18)*1024
                candidates = [g for g in devices() if g["index"] not in occupied and g["free_mib"]>=min_vram]
                if len(active)<cap and candidates and available>=need:
                    gpu = candidates[0]
                    self.leases[job] = gpu["index"]
                    print(f"Evaluator admission {job}: GPU {gpu['index']}, {gpu['free_mib']} MiB free, "
                          f"host {available:.1f}/{need:.1f} GiB, active slot {len(active)+1}/{cap}",flush=True)
                    break
                now=time.monotonic()
                if now-last_notice>=90:
                    print(f"Waiting evaluator {job}: host {available:.1f}/{need:.1f} GiB, "
                          f"active {len(active)}/{cap}, free GPU candidates {len(candidates)}",flush=True)
                    last_notice=now
            time.sleep(20)
        try:
            yield {**gpu,"gpu":gpu["index"]}
        finally:
            with self.lock:
                del self.leases[job]

    def watch(self):
        while not self.stop.wait(8):
            serial = self.serial_marker.exists()
            if not serial and ram_gib()>=13:
                continue
            with self.lock:
                active = [(pid,job) for job,(_,pid) in self.active().items() if pid is not None]
                if len(active)<=1:
                    continue
                ordered=sorted(active)
                if not serial:
                    free = ram_gib()
                    if free>=13:
                        continue
                    write_json(self.serial_marker,{"host_available_gib":free,"time_unix_s":time.time(),
                              "retained":ordered[0][1],"draining":[name for _,name in ordered[1:]]})
                    print(f"Host memory {free:.1f} GiB: retaining {ordered[0][1]} and draining extra owned shards",flush=True)
                for pid,job in ordered[1:]:
                    for child in children(pid):
                        args = process_args(child)
                        if args and args[0].endswith(b"timeout") and str(RUNS/job).encode() in args:
                            try:
                                os.kill(child,signal.SIGTERM)
                            except ProcessLookupError:
                                pass
