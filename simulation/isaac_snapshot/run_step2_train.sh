#!/usr/bin/env bash
set -euo pipefail
cd /data/sundawei/g1d_grasp_worker
variant=$1
uuid=$2
job="step2_v3_01_train_${variant,,}"
trap 'code=$?; python3 background_status.py --record-exit "$code" "$job"' EXIT
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$uuid"
export PYTHONPATH="$PWD/cache/three_view_detector/deps${PYTHONPATH:+:$PYTHONPATH}"
tier=runs/tier1_v3_01
extra=()
if [[ $variant == B ]]; then extra+=(--boxes "$tier/detection_cache.json"); fi
"$python" -Bu three_view_train.py train "$tier/training_split.json" cache/weights/resnet18-f37072fd.pth "runs/$job" \
    --variant "$variant" --device cuda:0 --steps 2000 --seed 33889 --freeze-bn --select-final "${extra[@]}"
