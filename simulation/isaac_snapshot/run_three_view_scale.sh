#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^three_view_scale_[0-9]+$ ]]
mkdir "$output"
trap 'code=$?; python3 "$project/background_status.py" --record-exit "$code" "$job"; printf "%s\n" "$code" > "$output.exit"' EXIT
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
"$python" -B three_view_scale.py prepare "$output"
for dependency in three_view_pipeline_01 three_view_generalization_01 three_view_parts_01; do
    printf 'Waiting for %s.complete\n' "$dependency"
    while [[ ! -s "runs/$dependency.complete" ]]; do sleep 15; done
done
"$python" -B three_view_scale.py progress "$output"
for wave in "$project/runs/${job}_w"[0-9][0-9][0-9]; do
    free=$(df -PB1 "$project" | awk 'NR==2 {print $4}')
    if (( free < 30 * 1024 * 1024 * 1024 )); then
        printf 'Filesystem free space is below 30 GiB before %s\n' "${wave##*/}"
        break
    fi
    workers=()
    for role in carton can; do
        if [[ ! -f "$wave/plans/shard_$role.json" ]]; then continue; fi
        gpu=1
        config="$project/env_resolved.json"
        if [[ $role == can ]]; then
            gpu=2
            config="$project/runs/candidate_can_config.json"
        fi
        G1D_WORKER_CONFIG="$config" bash "$project/run_lab_school.sh" "${wave}_$role" --grouped-manifest "$wave/plans/shard_$role.json" --grasp-after "$project/runs/lab_school_03/result.json" --gpu "$gpu" >"${wave}_$role.log" 2>&1 &
        workers+=("$!")
        printf 'Launching %s_%s PID %s\n' "${wave##*/}" "$role" "$!"
    done
    printf '%s\n' "${workers[@]}" > "$wave/worker_pids"
    for pid in "${workers[@]}"; do wait "$pid" || printf 'Simulator PID %s exit %s\n' "$pid" "$?"; done
    code=0
    "$python" -B three_view_collection.py finish "$wave" || code=$?
    python3 "$project/background_status.py" --record-exit "$code" "${wave##*/}"
    [[ -s "$wave/result.json" ]]
    "$python" -B three_view_scale.py progress "$output"
done
"$python" -B three_view_scale.py finish "$output"
