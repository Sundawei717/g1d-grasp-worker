"""Paths, resource leases and durable state shared by the Tier-2 stages."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
NAME = "step3b_01"
WORK = RUNS / NAME
PREFIX = f"g1d-{os.getuid()}-{NAME}_"
TIER1 = RUNS / "tier1_v3_01" / "training_split.json"
DEV = RUNS / "step2_v3_01" / "development.json"
FROZEN = RUNS / "step3a_01" / "frozen.json"
FROZEN_SHA = "33db506caba8d9e11fb990fb98d9acbdee0532f1ed3ff52310a3c142c02d2661"
ASSETS = RUNS / "step3a_01" / "tier2_ready_assets.json"
A40 = RUNS / "v3_step2_followup_01_a40_fit"
PAUSE = WORK / "pause.request"
PYTHON = Path("/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python")
ERR_PAUSED = 75


class Paused(Exception):
    pass


def check_pause():
    if PAUSE.exists():
        raise Paused("Pause requested; pending work is preserved")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        if default is not None:
            return default
        raise


def save(path, data):
    path = Path(path)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def ram_mib():
    return int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:"))) // 1024


def gpu_status():
    cmd = ["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu",
           "--format=csv,noheader,nounits"]
    rows = subprocess.check_output(cmd, text=True, timeout=15).splitlines()
    return [{"gpu": int(i), "uuid": uuid, "free_mib": int(free), "total_mib": int(total), "util_pct": int(util)}
            for i, uuid, free, total, util in (map(str.strip, row.split(",")) for row in rows)]


def shard_live(job):
    pid_file = RUNS / f"{job}.pid"
    try:
        pid = int(pid_file.read_text())
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        return os.getuid() == (Path("/proc") / str(pid)).stat().st_uid and os.fsencode(RUNS / job) in raw.split(b"\0")
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return False


def cache_root(job):
    if not job.startswith(f"{NAME}_") or not job.replace("_", "").isalnum():
        raise ValueError(f"Foreign Kit run: {job}")
    return Path("/dev/shm") / f"g1d-{os.getuid()}-{job}"


def cleanup(job):
    root = cache_root(job)
    if not root.exists() or root.is_symlink() or root.resolve() != root or shard_live(job):
        return
    # The complete child tree is disposable; admission data and durable logs live on /data.
    shutil.rmtree(root)


@contextmanager
def nonblocking_lock(path):
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
        else:
            yield lock


class Resources:
    """Reserve the same GPU locks as Step 3-A; reconsider headroom for every batch."""

    def claim_available(self, job, *, training=False):
        minimum_vram = (16 if training else 18) * 1024
        with (RUNS / "step3a_01_allocation.lock").open("a") as scheduler:
            fcntl.flock(scheduler, fcntl.LOCK_EX)
            locks, ready, occupied = [], [], 0
            try:
                for row in sorted(gpu_status(), key=lambda r: (-r["free_mib"], r["gpu"])):
                    lock = (RUNS / f"step3a_01_gpu_{row['gpu']}.lock").open("a+")
                    locks.append(lock)
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        occupied += 1
                        continue
                    if row["free_mib"] >= minimum_vram:
                        ready.append((row, lock))
                ram = ram_mib()
                if training:
                    if ram < 36 * 1024 or occupied or not ready:
                        return []
                    ready.sort(key=lambda p: (p[0]["util_pct"] > 20, -p[0]["free_mib"]))
                    idle = [p for p in ready if p[0]["util_pct"] <= 20]
                    chosen = idle[:2] if len(idle) >= 2 and ram >= 56 * 1024 else ready[:1]
                else:
                    if not ready or ram < (42 + occupied * 22) * 1024:
                        return []
                    chosen = ready[:1]
                for row, lock in chosen:
                    lock.seek(0)
                    lock.truncate()
                    lock.write(f"{job}\n")
                    lock.flush()
                    row["lease_fd"] = lock.fileno()
                    locks.remove(lock)
                return chosen
            finally:
                for lock in locks:
                    lock.close()

    @contextmanager
    def lease(self, job, *, training=False):
        while True:
            check_pause()
            if shutil.disk_usage(WORK).free < 50 * 2**30:
                raise RuntimeError("Less than 50 GiB free on /data. Preserve in-flight receipts and free space before resuming.")
            try:
                chosen = self.claim_available(job, training=training)
            except subprocess.SubprocessError as error:
                print(f"{job}: GPU query failed ({error}); retrying in 7 s", flush=True)
                time.sleep(7)
                continue
            if chosen:
                try:
                    rows = [row for row, _ in chosen]
                    print(f"{job}: GPU {[r['gpu'] for r in rows]}, RAM {ram_mib()/1024:.1f} GiB",
                          flush=True)
                    yield rows
                finally:
                    if not training:
                        cleanup(job)
                    for _, lock in chosen:
                        lock.close()
                return
            time.sleep(7)

    @contextmanager
    def claim(self, job):
        with self.lease(job) as rows:
            yield rows[0]


def runtime_snapshot():
    gpus = gpu_status()
    locks = []
    for row in gpus:
        gpu = row["gpu"]
        path = RUNS / f"step3a_01_gpu_{gpu}.lock"
        with nonblocking_lock(path) as lock:
            if lock is None:
                tag = path.read_text().strip()
                if tag.startswith(f"{NAME}_"):
                    locks.append({"gpu": gpu, "job": tag})
    last = []
    for path in sorted(WORK.glob("cache_events/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:12]:
        event = read(path)
        if not shard_live(event["job"]):
            cleanup(event["job"])
        last.append(event)
    free = shutil.disk_usage(WORK).free
    return {"updated_at": now(), "available_ram_mib": ram_mib(),
            "gpus": gpus, "active_leases": locks, "data_free_gib": round(free/2**30, 1),
            "kit_cleanup_total": len(list(WORK.glob("cache_events/*.json"))),
            "last_kit_cleanup": last}
