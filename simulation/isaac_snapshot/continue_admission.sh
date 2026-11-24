#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
cd "$project"
existing=runs/asset_admission_02.exit
for step in $(seq 1 900); do
    [[ ! -s $existing ]] || break
    sleep 2
done
[[ -s $existing ]] || exit 124
if [[ $(cat "$existing") == 0 ]]; then
    printf '%s\n' 'Existing asset batch passed; smooth-lift retry not required'
    exit 0
fi
[[ -s runs/asset_admission_02/result.json ]]
common=(
    --grasp-after "$project/runs/lab_school_03/result.json"
    --passed-smoke "$project/runs/planned_grasp_03/expert_diagnostic.json"
)
status=0
bash ./run_lab_school.sh "$project/runs/asset_admission_04" "${common[@]}" --admission-manifest "$project/runs/asset_admission_plans_02/revised_manifest.json" > runs/asset_admission_04.log 2>&1 || status=$?
printf '%s\n' "$status" > runs/asset_admission_04.exit
if ((status != 0)); then
    printf '%s\n' 'Smooth-lift asset admission did not pass; expert batch not launched'
    exit "$status"
fi
[[ -s runs/expert20_plans_01/manifest.json ]]
bash ./run_lab_school.sh "$project/runs/expert20_04" "${common[@]}" --admission-manifest "$project/runs/expert20_plans_01/manifest.json" --passed-admission "$project/runs/asset_admission_04/result.json" > runs/expert20_04.log 2>&1 || status=$?
printf '%s\n' "$status" > runs/expert20_04.exit
exit "$status"
