#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^three_view_diversity_[0-9]+$ ]]
mkdir "$output"
trap 'code=$?; python3 "$project/background_status.py" --record-exit "$code" "$job"; printf "%s\n" "$code" > "$output.exit"' EXIT
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
"$python" -Bu three_view_diversity.py prepare "$output"
can_plans=(runs/three_view_scale_01_w[0-9][0-9][0-9]/plans/shard_can.json)
last=${can_plans[-1]%%/plans/*}
printf 'Waiting for %s_can.complete before moving GPU 2 to the varied assets\n' "$last"
while [[ ! -s "${last}_can.complete" && ! -s runs/three_view_scale_01.complete ]]; do sleep 15; done
for role in mug_train flat_carton tall_can part_carton; do
    plan="$output/plans/shard_$role.json"
    if [[ ! -s "$plan" ]]; then continue; fi
    free=$(df -PB1 "$project" | awk 'NR==2 {print $4}')
    ((free >= 30 * 1024 * 1024 * 1024))
    name="${job}_$role"
    G1D_WORKER_CONFIG="$output/plans/config_$role.json" bash "$project/run_lab_school.sh" "$project/runs/$name" --grouped-manifest "$plan" --grasp-after "$project/runs/lab_school_03/result.json" --gpu 2 >"runs/$name.log" 2>&1 &
    worker=$!
    echo "$worker" >"runs/$name.pid"
    printf 'Capturing %s PID %s\n' "$role" "$worker"
    wait "$worker" || echo "$role simulator exit $?"
done
code=0
"$python" -B three_view_collection.py finish "$output" || code=$?
test -s "$output/result.json"
exit "$code"
