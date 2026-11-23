#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/fit20_occlusion_retrain_01" ]]
mkdir "$output"
on_exit() {
    local code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$job"
    printf '%s\n' "$code" > "$output.exit"
}
trap on_exit EXIT
export G1D_TRAIN_GPU=${G1D_TRAIN_GPU:-0}
fit=fit20_04
evals=(fit20_closed_loop_07 fit20_closed_loop_08)
baselines=(fit20_closed_loop_05 fit20_closed_loop_04)
prompts=(hold_last_reliable gt_full_projection)

run_job() {
    local name=$1 task=$2 child
    shift 2
    bash "$project/$task" "$project/runs/$name" "$@" > "runs/$name.log" 2>&1 &
    child=$!
    echo "$child" > "runs/$name.pid"
    printf 'Launching %s PID %s\n' "$name" "$child"
    wait "$child"
}

run_job "$fit" run_fit20_corrected.sh hold_last_reliable
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 "$python" -B "$project/audit_fit20_supervision.py" "$output/supervision" --fit "$fit"
for i in "${!evals[@]}"; do
    run_job "${evals[$i]}" run_fit20_closed_loop.sh "$project/runs/$fit" "$project/runs/${baselines[$i]}" "${prompts[$i]}" --replace-checkpoint ||
        printf 'Evaluation %s exit=%s. Checking receipt after both modes.\n' "${evals[$i]}" "$?"
done
"$python" -B "$project/fit20_retrain_comparison.py" "$output" "${evals[@]}" "${baselines[@]}"
