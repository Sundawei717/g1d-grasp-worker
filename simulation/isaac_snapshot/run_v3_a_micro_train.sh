#!/usr/bin/env bash
set -euo pipefail
cd /data/sundawei/g1d_grasp_worker
mode=$1
uuid=$2
epochs=$3
job="v3_a_micro_01_$mode"
trap 'exit_code=$?; python3 background_status.py --record-exit "$exit_code" "$job"' EXIT
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python -Bu v3_prompt_train.py "runs/$job" --micro "$mode" --epochs "$epochs"
