#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^policy_correction_capture_[0-9]+$ ]]
mkdir "$output"
on_exit() {
    local code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$job"
    printf '%s\n' "$code" > "$output.exit"
}
trap on_exit EXIT
plans="$project/runs/policy_corrections_01"
printf 'Waiting for %s.complete\n' "$plans"
for tick in $(seq 1 4800); do
    [[ ! -s "$plans.complete" ]] || break
    sleep 3
done
[[ -s "$plans.complete" && -s "$plans/result.json" ]]
memory=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
if ((memory < 20480)); then
    printf 'GPU 1 has less than 20 GiB free\n'
    exit 75
fi
for role in carton can; do
    manifest="$plans/shard_$role.json"
    [[ -s "$manifest" ]] || continue
    if ! "$python" -B -c 'import json,sys; raise SystemExit(not any(row["offline_feasible"] for row in json.load(open(sys.argv[1]))["cases"]))' "$manifest"; then
        printf 'No feasible %s correction waypoints\n' "$role"
        continue
    fi
    config="$project/env_resolved.json"
    if [[ $role == can ]]; then
        config="$project/runs/candidate_can_config.json"
    fi
    printf 'Launching %s_%s on GPU 1\n' "$job" "$role"
    G1D_WORKER_CONFIG="$config" bash "$project/run_lab_school.sh" "$project/runs/${job}_$role" --grouped-manifest "$manifest" --grasp-after "$project/runs/lab_school_03/result.json" --gpu 1 > "runs/${job}_$role.log" 2>&1 || printf 'Correction shard %s exited %s\n' "$role" "$?"
done
"$python" -B -c 'import sys; from pathlib import Path; from prepare_policy_corrections import review_capture; raise SystemExit(review_capture(*map(Path,sys.argv[1:])))' "$plans" "$output"
