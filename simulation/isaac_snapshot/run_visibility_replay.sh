#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^visibility_replay_[0-9]+$ ]]
mkdir "$output"
on_exit() {
    local code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$job"
    printf '%s\n' "$code" > "$output.exit"
}
trap on_exit EXIT
for dependency in route_visibility_01 fit20_closed_loop_02_can; do
    printf 'Waiting for %s.complete\n' "$dependency"
    for tick in $(seq 1 4800); do
        [[ ! -s "runs/$dependency.complete" ]] || break
        sleep 3
    done
    [[ -s "runs/$dependency.complete" ]]
done
memory=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
if ((memory < 20480)); then
    printf 'GPU 2 has less than 20 GiB free\n'
    exit 75
fi
"$python" -B -c 'import sys; from pathlib import Path; from inspect_route_visibility import replay_manifest; replay_manifest(*map(Path,sys.argv[1:]))' "$project" "$project/runs/route_visibility_01/result.json" "$output/plans"
printf 'Launching %s_carton on GPU 2\n' "$job"
bash "$project/run_lab_school.sh" "$project/runs/${job}_carton" --grouped-manifest "$output/plans/manifest.json" --grasp-after "$project/runs/lab_school_03/result.json" --gpu 2 > "runs/${job}_carton.log" 2>&1
"$python" -B -c 'import sys; from pathlib import Path; from inspect_route_visibility import review_replay; raise SystemExit(review_replay(*map(Path,sys.argv[1:])))' "$project" "$project/runs/${job}_carton" "$output/result.json"
