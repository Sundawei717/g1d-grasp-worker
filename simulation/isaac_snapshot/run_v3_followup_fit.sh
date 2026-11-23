#!/usr/bin/env bash
set -euo pipefail
cd /data/sundawei/g1d_grasp_worker
case "$1" in
  b20)
    job=v3_step2_followup_01_b20_fit
    variant=B
    epochs=20
    boxes=(--boxes runs/tier1_v3_01/detection_cache.json)
    ;;
  a40)
    job=v3_step2_followup_01_a40_fit
    variant=A
    epochs=40
    boxes=()
    ;;
  *) exit 2 ;;
esac
trap 'status=$?; python3 background_status.py --record-exit "$status" "$job"' EXIT
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$2" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python -Bu v3_prompt_train.py "runs/$job" \
    --query-protocol full --variant "$variant" --epochs "$epochs" --seed 33889 \
    --loader-workers "$3" "${boxes[@]}"
