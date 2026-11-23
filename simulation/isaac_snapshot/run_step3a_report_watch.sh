#!/usr/bin/env bash
set -euo pipefail
cd /data/sundawei/g1d_grasp_worker
job=step3a_01_report
py=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
tasks=(v3_b40_step3a_01_fit v3_b40_step3a_01 step3a_01_assets step3a_01_rays step3a_01_pilots step3a_01_freeze
       step3a_01_sugar_calibration step3a_01_substitutes step3a_01_substitute_rays step3a_01_local_guides step3a_01_round_followup step3a_01_compact_calibration step3a_01_final_substitutes step3a_01_marker_detour step3a_01_marker_followup)
while true; do
    rc=0
    "$py" -Bu step3a_report.py || rc=$?
    waiting=0
    for item in "${tasks[@]}"; do
        if [[ ! -f runs/$item.complete ]]; then waiting=1; fi
    done
    if [[ $waiting == 0 ]]; then
        "$py" background_status.py --record-exit "$rc" "$job"
        exit "$rc"
    fi
    sleep 300
done
