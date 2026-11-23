#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^tracking_motion_[0-9]+$ ]]
mkdir "$output"
on_exit() {
    local code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$job"
    printf '%s\n' "$code" > "$output.exit"
}
trap on_exit EXIT
"$python" -B "$project/tracking_motion.py" prepare "$project" "$output/plans"
record() {
    local role=$1 gpu=$2 config=$3 memory
    memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    if ((memory < 20480)); then
        printf 'GPU %s has less than 20 GiB free for motion capture\n' "$gpu"
        return 75
    fi
    printf 'Launching %s_%s on GPU %s\n' "$job" "$role" "$gpu"
    G1D_WORKER_CONFIG="$config" bash "$project/run_lab_school.sh" "$project/runs/${job}_$role" --grouped-manifest "$output/plans/shard_$role.json" --grasp-after "$project/runs/lab_school_03/result.json" --gpu "$gpu" > "runs/${job}_$role.log" 2>&1
}
record carton 2 "$project/env_resolved.json" &
carton=$!
(
    for tick in $(seq 1 1200); do
        [[ ! -s runs/tracking_benchmark_01.complete ]] || break
        sleep 3
    done
    [[ -s runs/tracking_benchmark_01.complete ]]
    record can 1 "$project/runs/candidate_can_config.json"
) &
can=$!
wait "$carton" || printf 'Carton motion capture exited %s\n' "$?"
wait "$can" || printf 'Can motion capture exited %s\n' "$?"
"$python" -B "$project/tracking_motion.py" finish "$output"
