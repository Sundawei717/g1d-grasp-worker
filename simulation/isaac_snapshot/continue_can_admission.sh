#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
cd "$project"
for step in $(seq 1 2600); do
    if [[ -s runs/asset_pose04_support_confirmation.exit ]]; then
        if [[ $(cat runs/asset_pose04_support_confirmation.exit) != 0 ]]; then
            break
        fi
        if [[ -s runs/expert20_05.exit ]]; then
            if [[ $(cat runs/expert20_05.exit) != 0 || $(cat runs/obstacle_replan_01.exit) != 0 || -s runs/obstacle_grasp_02.exit ]]; then
                break
            fi
        fi
    fi
    sleep 2
done
if ((step == 2600)); then
    exit 124
fi
[[ -s runs/can_admission_plans_01.exit && $(cat runs/can_admission_plans_01.exit) == 0 ]]
[[ -s runs/asset_can_smoke_02.exit && $(cat runs/asset_can_smoke_02.exit) == 0 ]]
available=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
((available > 20480)) || exit 75
args=(
    --grasp-after "$project/runs/lab_school_03/result.json"
    --admission-manifest "$project/runs/can_admission_plans_01/manifest.json"
    --passed-smoke "$project/runs/asset_can_smoke_02/expert_diagnostic.json"
)
status=0
G1D_WORKER_CONFIG="$project/runs/candidate_can_config.json" bash ./run_lab_school.sh "$project/runs/can_admission_01" "${args[@]}" > runs/can_admission_01.log 2>&1 || status=$?
printf '%s\n' "$status" > runs/can_admission_01.exit
exit "$status"
