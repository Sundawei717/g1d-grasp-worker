#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
work=$(realpath -e -- "$1")
stage=$2
[[ $work == "$project/runs/"three_view_step0_* && $stage =~ ^(expert|policy)$ ]]
name=${work##*/}
job="${name}_$stage"
trap 'code=$?; python3 background_status.py --record-exit "$code" "$job"' EXIT
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$project/cache/three_view_detector/deps${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="$project/cache/three_view_detector/config"
export YOLO_AUTOINSTALL=false
"$python" -Bu three_view_step0.py "$stage" "$work"
if [[ $stage == policy ]]; then
    "$python" -Bu three_view_step0.py report "$work"
    python3 background_status.py --record-exit 0 "$name"
fi
