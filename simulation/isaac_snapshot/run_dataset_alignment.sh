#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
preview=$(realpath -m -- "$2")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^dataset_alignment_[0-9]+$ ]]
[[ ${preview%/*} == "$project/runs" && ${preview##*/} =~ ^collection_preview_[0-9]+$ ]]
mkdir "$output"
on_exit() {
    local code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$job"
    printf '%s\n' "$code" > "$output.exit"
}
trap on_exit EXIT
printf 'Waiting for the capacity preview at %s\n' "$preview.complete"
for tick in $(seq 1 2400); do
    [[ ! -s "$preview.complete" ]] || break
    sleep 3
done
[[ -s "$preview.complete" ]]
"$python" -B "$project/dataset_alignment.py" prepare "$preview" "$output/plans"
replay() {
    local role=$1 gpu=$2 config=$3 memory
    memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    if ((memory < 20480)); then
        printf 'GPU %s below 20 GiB free for %s alignment\n' "$gpu" "$role"
        return 75
    fi
    printf 'Launching %s_%s on GPU %s\n' "$job" "$role" "$gpu"
    G1D_WORKER_CONFIG="$config" bash "$project/run_lab_school.sh" "$project/runs/${job}_$role" --grouped-manifest "$output/plans/shard_$role.json" --grasp-after "$project/runs/lab_school_03/result.json" --gpu "$gpu" > "runs/${job}_$role.log" 2>&1
}
replay carton 1 "$project/env_resolved.json" &
first=$!
replay can 2 "$project/runs/candidate_can_config.json" &
second=$!
printf '%s\n%s\n' "$first" "$second" > "$output/worker_pids"
wait "$first" || printf 'Carton replay command exited %s\n' "$?"
wait "$second" || printf 'Can replay command exited %s\n' "$?"
"$python" -B "$project/dataset_alignment.py" finish "$output"
