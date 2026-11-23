#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
data=/data/sundawei/vla_rsi_g1d
python="$data/envs/sim60/bin/python"
cd "$project"
output=$(realpath -m -- "$1")
alignment=$(realpath -m -- "$2")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^train_bootstrap_[0-9]+$ ]]
[[ ${alignment%/*} == "$project/runs" && ${alignment##*/} =~ ^dataset_alignment_[0-9]+$ ]]
mkdir "$output"
on_exit() {
    local code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$job"
    printf '%s\n' "$code" > "$output.exit"
}
trap on_exit EXIT
printf 'Waiting for the recorded-command alignment at %s\n' "$alignment.complete"
for tick in $(seq 1 3600); do
    [[ ! -s "$alignment.complete" ]] || break
    sleep 3
done
[[ -s "$alignment.complete" ]]
urdf="$data/assets/Assets/Robots/g1d/g1d_dex1.urdf"
export OPENBLAS_NUM_THREADS=1
timeout -k 15 2400 "$python" -Bu "$project/prepare_train_bootstrap.py" prepare "$project" "$urdf" "$project/runs/grouped_eval_03/result.json" "$project/runs/collection_preview_02/result.json" "$alignment/result.json" "$output/plans"
capture_asset() {
    local role=$1 gpu=$2 config=$3 memory
    memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    if ((memory < 20480)); then
        printf 'GPU %s below 20 GiB free for %s collection\n' "$gpu" "$role"
        return 75
    fi
    printf 'Launching %s_%s on GPU %s\n' "$job" "$role" "$gpu"
    G1D_WORKER_CONFIG="$config" bash "$project/run_lab_school.sh" "$project/runs/${job}_$role" --grouped-manifest "$output/plans/shard_$role.json" --grasp-after "$project/runs/lab_school_03/result.json" --gpu "$gpu" > "runs/${job}_$role.log" 2>&1
}
capture_asset carton 1 "$project/env_resolved.json" &
first=$!
capture_asset can 2 "$project/runs/candidate_can_config.json" &
second=$!
printf '%s\n%s\n' "$first" "$second" > "$output/worker_pids"
wait "$first" || printf 'Carton collection command exited %s\n' "$?"
wait "$second" || printf 'Can collection command exited %s\n' "$?"
"$python" -B "$project/prepare_train_bootstrap.py" finish "$output"
