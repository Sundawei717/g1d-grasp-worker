#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
cd "$project"
for step in $(seq 1 540); do
    [[ ! -s runs/asset_bandage_smoke_02.exit ]] || break
    sleep 2
done
[[ -s runs/asset_bandage_smoke_02.exit ]] || exit 124
review="$project/reports/asset_admission_04_baseline_review.json"
python3 -c 'import json,sys; evidence=json.load(open(sys.argv[1])); assert evidence["passed"] and evidence["strict_success"]==10' "$review"
args=(--grasp-after "$project/runs/lab_school_03/result.json")
status=0
bash ./run_lab_school.sh "$project/runs/asset_pose04_support_confirmation" "${args[@]}" --plan "$project/runs/asset_admission_plans_02/pose_04.json" --candidate-index 1 > runs/asset_pose04_support_confirmation.log 2>&1 || status=$?
printf '%s\n' "$status" > runs/asset_pose04_support_confirmation.exit
((status == 0)) || exit "$status"
bash ./run_lab_school.sh "$project/runs/expert20_05" "${args[@]}" --admission-manifest "$project/runs/expert20_plans_01/manifest.json" --passed-smoke "$project/runs/asset_pose04_support_confirmation/expert_diagnostic.json" --passed-admission "$review" > runs/expert20_05.log 2>&1 || status=$?
printf '%s\n' "$status" > runs/expert20_05.exit
((status == 0)) || exit "$status"
if [[ -s runs/obstacle_replan_01.exit && $(cat runs/obstacle_replan_01.exit) == 0 ]]; then
    bash ./run_lab_school.sh "$project/runs/obstacle_grasp_02" "${args[@]}" --plan "$project/runs/obstacle_replan_01.json" > runs/obstacle_grasp_02.log 2>&1 || status=$?
    printf '%s\n' "$status" > runs/obstacle_grasp_02.exit
fi
exit "$status"
