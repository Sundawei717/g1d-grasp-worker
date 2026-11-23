#!/usr/bin/env bash
set -euo pipefail
cd /data/sundawei/g1d_grasp_worker
mode=$1
uuid=$2
name="v3_a_prompt_repair_01_$mode"
trap 'exit_code=$?; python3 background_status.py --record-exit "$exit_code" "$name"' EXIT
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
args=()
case "$mode" in
  pilot) args=(--pilot --epochs 80) ;;
  full) args=(--epochs 20) ;;
  *) exit 2 ;;
esac
"$python" -Bu v3_prompt_train.py "runs/$name" "${args[@]}"
