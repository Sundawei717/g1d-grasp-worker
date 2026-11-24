"""Release shared RAM consumed by this project's completed simulator caches."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil

from background_status import record_complete

RUNS=Path(__file__).resolve().parent/"runs"
RAM_ROOT=Path("/dev/shm").resolve(strict=True)
UID=os.getuid()
PREFIX=f"g1d-{UID}-"


def active_directories():
    values=[]
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if proc.stat().st_uid==UID:
                values.append((proc/"cmdline").read_bytes())
                values.append((proc/"environ").read_bytes())
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue
    return values


def eligible():
    busy=active_directories()
    for run_dir in sorted(RAM_ROOT.glob(f"{PREFIX}*")):
        name=run_dir.name.removeprefix(PREFIX)
        if not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]+",name):
            continue
        marker=RUNS/f"{name}.complete"
        if not marker.is_file() or json.loads(marker.read_text()).get("status")!="COMPLETE":
            continue
        if run_dir.is_symlink() or not run_dir.is_dir() or run_dir.stat().st_uid!=UID:
            continue
        if any(str(run_dir).encode() in active or str(RUNS/name).encode() in active for active in busy):
            continue
        for kit in run_dir.glob("g1d_three_view_kit_*"):
            cache=kit/"cache"
            if kit.is_symlink() or cache.is_symlink() or not cache.is_dir():
                continue
            if cache.resolve(strict=True)!=cache or cache.stat().st_uid!=UID:
                continue
            if run_dir not in cache.parents or run_dir.parent!=RAM_ROOT:
                raise RuntimeError(f"Unexpected cache path {cache}")
            yield name,cache


def main(apply):
    entries=[]
    for name,cache in eligible():
        size=sum(p.lstat().st_blocks*512 for p in cache.rglob("*") if p.is_file() and not p.is_symlink())
        entries.append({"job":name,"cache":str(cache),"bytes":size})
        if apply:
            shutil.rmtree(cache)
    print(f"{'Removed' if apply else 'Eligible'} {len(entries)} caches, "
          f"{sum(x['bytes'] for x in entries)/2**30:.2f} GiB",flush=True)
    if apply:
        rec=RUNS/"step3a_01/runtime_cache_cleanup.json"
        rec.write_text(json.dumps({"count":len(entries),"bytes":sum(x["bytes"] for x in entries),"entries":entries},indent=2))
        record_complete(RUNS/"step3a_01_runtime_cleanup",0)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--apply",action="store_true")
    main(parser.parse_args().apply)
