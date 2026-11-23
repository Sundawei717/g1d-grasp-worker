#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^three_view_training_data_[0-9]+$ ]]
mkdir "$output"
trap 'code=$?; python3 "$project/background_status.py" --record-exit "$code" "$job"; printf "%s\n" "$code" > "$output.exit"' EXIT
for dependency in three_view_scale_01 three_view_diversity_01; do
    printf 'Waiting for %s.complete\n' "$dependency"
    while [[ ! -s "runs/$dependency.complete" ]]; do sleep 15; done
done
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 "$python" -B three_view_diversity.py merge "$output"
