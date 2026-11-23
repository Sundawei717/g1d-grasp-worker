#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^prompt_mode_comparison_[0-9]+$ ]]
mkdir "$output"
on_exit() {
    local code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$job"
    printf '%s\n' "$code" > "$output.exit"
}
trap on_exit EXIT
baseline="$project/runs/fit20_closed_loop_03"
fit="$project/runs/fit20_03"
experimental=(fit20_closed_loop_04 fit20_closed_loop_05 fit20_closed_loop_06)
modes=(gt_full_projection hold_last_reliable frozen_init)
printf 'Reference %s and checkpoint %s\n' "$baseline" "$fit"
for index in "${!experimental[@]}"; do
    name="${experimental[$index]}"
    test ! -e "runs/$name.complete"
    bash "$project/run_fit20_closed_loop.sh" "$project/runs/$name" "$fit" "$baseline" "${modes[$index]}" > "runs/$name.log" 2>&1 &
    child=$!
    echo "$child" > "runs/$name.pid"
    printf 'Launching %s (%s) PID %s\n' "$name" "${modes[$index]}" "$child"
    wait "$child" || printf '%s exit=%s. Review its completed physical result.\n' "$name" "$?"
done
"$python" -B "$project/prompt_mode_comparison.py" "$project/runs" "$output"
