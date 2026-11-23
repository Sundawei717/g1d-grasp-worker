#!/usr/bin/env bash
set -euo pipefail
cd /data/sundawei/g1d_grasp_worker
mode=$1
uuid=$2
case "$mode" in
  gate) epochs=140 ;;
  full)
    epochs=20
    python3 -c 'import json; r=json.load(open("runs/v3_a_query_gate_01/gate_verdict.json")); assert r["passed"] and r["locked_m"] in (0.01,0.10)'
    ;;
  *) exit 2 ;;
esac
name="v3_a_query_gate_01_${mode}_fit"
trap 'exit_code=$?; python3 background_status.py --record-exit "$exit_code" "$name"' EXIT
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python -Bu v3_prompt_train.py "runs/$name" --query-protocol "$mode" --epochs "$epochs"
