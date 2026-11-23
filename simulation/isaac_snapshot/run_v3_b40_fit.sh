#!/usr/bin/env bash
set -euo pipefail
cd /data/sundawei/g1d_grasp_worker
job=v3_b40_step3a_01_fit
trap 'status=$?; python3 background_status.py --record-exit "$status" "$job"' EXIT
gpu="$(python3 - <<'PY'
from pathlib import Path
import subprocess
ram=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
assert ram>=23*1024**2, 'Insufficient host RAM for B40'
rows=[x.split(', ') for x in subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
ready=[r for r in rows if int(r[1])>=18*1024]
assert ready, 'No suitable training GPU'
print(min(ready,key=lambda r:(int(r[2]),-int(r[1])))[0])
PY
)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
printf 'Training on %s\n' "$gpu"
printf '%s\n' "$gpu" > "runs/$job.gpu"
 /data/sundawei/vla_rsi_g1d/envs/sim60/bin/python -Bu v3_prompt_train.py "runs/$job" \
    --query-protocol full --variant B --epochs 40 --seed 33889 --loader-workers 2 \
    --boxes runs/tier1_v3_01/detection_cache.json
