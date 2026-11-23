#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^three_view_expert_[0-9]+$ ]]
mkdir "$output"
trap 'code=$?; python3 "$project/background_status.py" --record-exit "$code" "$job"; printf "%s\n" "$code" > "$output.exit"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
limit=${2:-0}
flags=(--limit "$limit")
if [[ $limit == 0 ]]; then
    flags+=(--reuse "$project/runs/three_view_expert_02/result.json")
fi
"$python" -B "$project/three_view_collection.py" prepare "$project/runs/train_bootstrap_01" "$output/plans" "${flags[@]}"

run_shard() {
    local role=$1 gpu=$2 config=$3
    G1D_WORKER_CONFIG="$config" bash "$project/run_lab_school.sh" "$project/runs/${job}_$role" --grouped-manifest "$output/plans/shard_$role.json" --grasp-after "$project/runs/lab_school_03/result.json" --gpu "$gpu"
}
run_shard carton 1 "$project/env_resolved.json" > "runs/${job}_carton.log" 2>&1 &
carton=$!
run_shard can 2 "$project/runs/candidate_can_config.json" > "runs/${job}_can.log" 2>&1 &
can=$!
printf '%s\n%s\n' "$carton" "$can" > "$output/worker_pids"
printf 'Collecting carton PID %s and can PID %s\n' "$carton" "$can"
wait "$carton" || printf 'Carton worker exit %s\n' "$?"
wait "$can" || printf 'Can worker exit %s\n' "$?"
"$python" -B "$project/three_view_collection.py" finish "$output"
