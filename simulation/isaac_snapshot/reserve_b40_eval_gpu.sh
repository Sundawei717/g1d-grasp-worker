#!/usr/bin/env bash
set -euo pipefail
cd /data/sundawei/g1d_grasp_worker
gpu=2
exec 9>"runs/step3a_01_gpu_${gpu}.lock"
flock -x 9
printf 'v3_b40_step3a_01_reserved\n' >&9
printf 'Reserved GPU %s for the remaining B40 comparison.\n' "$gpu"
export B40_RESERVED_GPU="$gpu"
exec /data/sundawei/vla_rsi_g1d/envs/sim60/bin/python -Bu v3_b40_eval.py
