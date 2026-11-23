#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
data=/data/sundawei/vla_rsi_g1d
python="$data/envs/sim60/bin/python"
cd "$project"
output=$(realpath -m -- "$1")
previous=$(realpath -e -- "$2")
prefix=${output##*/}
[[ $output == "$project/runs/$prefix" && $prefix =~ ^grouped_eval_[0-9]+$ ]]
[[ ${previous%/*} == "$project/runs" && ${previous##*/} =~ ^grouped_eval_[0-9]+$ ]]
mkdir "$output"
on_exit() {
    local code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$prefix"
    printf '%s\n' "$code" > "$output.exit"
}
trap on_exit EXIT
export OPENBLAS_NUM_THREADS=1
urdf="$data/assets/Assets/Robots/g1d/g1d_dex1.urdf"
timeout -k 15 1800 "$python" -Bu "$project/replay_grouped_obstacle.py" prepare "$project" "$urdf" "$previous/result.json" "$output/plans"

replay() {
    local index=$1 gpu=$2 memory
    local -a flags
    memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    if ((memory < 20480)); then
        printf 'GPU %s below 20 GiB free; replay shard %s could not start\n' "$gpu" "$index"
        return 75
    fi
    flags=(--grouped-manifest "$output/plans/shard_$index.json" --grasp-after "$project/runs/lab_school_03/result.json" --gpu "$gpu")
    printf 'Launching %s_b%s on GPU %s\n' "$prefix" "$index" "$gpu"
    bash "$project/run_lab_school.sh" "$project/runs/${prefix}_b$index" "${flags[@]}" > "runs/${prefix}_b$index.log" 2>&1
}
replay 0 1 &
first=$!
replay 1 2 &
second=$!
printf '%s\n%s\n' "$first" "$second" > "$output/worker_pids"
wait "$first" || printf 'Replay shard b0 exit=%s\n' "$?"
wait "$second" || printf 'Replay shard b1 exit=%s\n' "$?"
"$python" -Bu "$project/replay_grouped_obstacle.py" summarize "$output"
