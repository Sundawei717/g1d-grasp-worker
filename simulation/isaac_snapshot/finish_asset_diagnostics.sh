#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
cd "$project"
for step in $(seq 1 2400); do
    if [[ -s runs/asset_admission_04.exit ]]; then
        if [[ $(cat runs/asset_admission_04.exit) != 0 || -s runs/expert20_04.exit ]]; then
            break
        fi
    fi
    sleep 2
done
[[ -s runs/asset_admission_04.exit ]]
if [[ $(cat runs/asset_admission_04.exit) == 0 && ! -s runs/expert20_04.exit ]]; then
    exit 124
fi
for asset in can bandage; do
    [[ -s runs/offline_asset_${asset}_02.exit ]] || exit 1
    [[ $(cat "runs/offline_asset_${asset}_02.exit") == 0 ]] || continue
    available=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
    ((available > 20480)) || exit 75
    status=0
    G1D_WORKER_CONFIG="$project/runs/candidate_${asset}_config.json" bash ./run_lab_school.sh "$project/runs/asset_${asset}_smoke_02" --grasp-after "$project/runs/lab_school_03/result.json" --plan "$project/runs/offline_asset_${asset}_02.json" > "runs/asset_${asset}_smoke_02.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "runs/asset_${asset}_smoke_02.exit"
done
if [[ -s runs/expert20_04.exit && $(cat runs/expert20_04.exit) == 0
    && -s runs/obstacle_replan_01.exit && $(cat runs/obstacle_replan_01.exit) == 0 ]]; then
    status=0
    bash ./run_lab_school.sh "$project/runs/obstacle_grasp_01" --grasp-after "$project/runs/lab_school_03/result.json" --plan "$project/runs/obstacle_replan_01.json" > runs/obstacle_grasp_01.log 2>&1 || status=$?
    printf '%s\n' "$status" > runs/obstacle_grasp_01.exit
fi
