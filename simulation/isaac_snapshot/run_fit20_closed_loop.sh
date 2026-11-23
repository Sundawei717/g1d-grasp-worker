#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
data=/data/sundawei/vla_rsi_g1d
python="$data/envs/sim60/bin/python"
cd "$project"
output=$(realpath -m -- "$1")
fit=$(realpath -m -- "${2:-$project/runs/fit20_01}")
previous=$(realpath -m -- "${3:-$fit}")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^fit20_closed_loop_[0-9]+$ ]]
[[ ${fit%/*} == "$project/runs" && ${fit##*/} =~ ^fit20_[0-9]+$ ]]
[[ ${previous%/*} == "$project/runs" && ${previous##*/} =~ ^fit20_([0-9]+|closed_loop_[0-9]+)$ ]]
mkdir "$output"
on_exit() {
    local code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$job"
    printf '%s\n' "$code" > "$output.exit"
}
trap on_exit EXIT
for dependency in "$fit" "$previous"; do
    if [[ ! -s "$dependency.complete" ]]; then
        printf 'Waiting for %s.complete\n' "$dependency"
    fi
    for tick in $(seq 1 4800); do
        [[ ! -s "$dependency.complete" ]] || break
        sleep 3
    done
    [[ -s "$dependency.complete" ]]
done
urdf="$data/assets/Assets/Robots/g1d/g1d_dex1.urdf"
evidence="$project/runs/lab_school_03/result.json"
if (( $# >= 4 )); then
    "$python" -B "$project/fit20_eval.py" prepare-paired "$previous" "$fit" "$4" "$output/plans" "${@:5}"
else
    "$python" -B "$project/fit20_eval.py" prepare "$project/runs/train_bootstrap_01" "$fit" "$urdf" "$evidence" "$output/plans"
fi
run_asset() {
    local role=$1 gpu=$2 config=$3 memory
    memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    if ((memory < 20480)); then
        printf 'GPU %s has less than 20 GiB free\n' "$gpu"
        return 75
    fi
    printf 'Launching %s_%s on GPU %s\n' "$job" "$role" "$gpu"
    G1D_WORKER_CONFIG="$config" bash "$project/run_lab_school.sh" "$project/runs/${job}_$role" --grouped-manifest "$output/plans/shard_$role.json" --grasp-after "$evidence" --gpu "$gpu" > "runs/${job}_$role.log" 2>&1
}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
run_asset carton 1 "$project/env_resolved.json" &
first=$!
run_asset can 2 "$project/runs/candidate_can_config.json" &
second=$!
printf '%s\n%s\n' "$first" "$second" > "$output/worker_pids"
wait "$first" || printf 'Carton policy job exited %s\n' "$?"
wait "$second" || printf 'Can policy job exited %s\n' "$?"
"$python" -B "$project/fit20_eval.py" finish "$output"
