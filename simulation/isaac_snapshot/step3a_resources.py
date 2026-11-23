"""Reserve independent GPUs and budget shared RAM for Isaac Sim startup."""
from contextlib import contextmanager
import fcntl
import time

from v3_followup_fit_watch import RUNS,devices,ram_gib


FIT = "v3_b40_step3a_01_fit"
PRIVATE_STARTUP_GIB = 22


class SimAllocator:
    def __init__(self,with_detector=False):
        self.with_detector=with_detector

    def reserve(self,job):
        with (RUNS/"step3a_01_allocation.lock").open("a") as scheduler:
            fcntl.flock(scheduler,fcntl.LOCK_EX)
            fit_gpu=RUNS/f"{FIT}.gpu"
            training=fit_gpu.is_file() and not (RUNS/f"{FIT}.complete").is_file()
            exclude=fit_gpu.read_text().strip() if training else None
            headroom=(34 if self.with_detector else 32)+(7 if training else 0)
            need_vram=(22 if self.with_detector else 18)*1024
            ram=ram_gib()
            occupancy=0
            available=[]
            for gpu in devices():
                lock=(RUNS/f"step3a_01_gpu_{gpu['index']}.lock").open("a+")
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    occupancy+=1
                    lock.close()
                    continue
                if gpu["uuid"]!=exclude and gpu["free_mib"]>=need_vram:
                    available.append((gpu,lock))
                else:
                    lock.close()
            needed=headroom+PRIVATE_STARTUP_GIB*occupancy
            allocation=None
            if available and ram>=needed:
                gpu,lock=available.pop(0)
                lock.seek(0)
                lock.truncate()
                lock.write(f"{job}\n")
                lock.flush()
                allocation={**gpu,"gpu":gpu["index"]},lock
            for _,lock in available:
                lock.close()
            return allocation,ram,needed,occupancy

    @contextmanager
    def claim(self,job):
        while True:
            allocation,ram,needed,occupancy=self.reserve(job)
            if allocation:
                gpu,lock=allocation
                with lock:
                    print(f"Allocated {job} GPU {gpu['index']} with {ram:.1f} GiB RAM, "
                          f"active simulators {occupancy+1}",flush=True)
                    yield gpu
                return
            print(f"Waiting for {job}: host {ram:.1f}/{needed} GiB, "
                  f"{occupancy} simulator reservations",flush=True)
            time.sleep(20)
