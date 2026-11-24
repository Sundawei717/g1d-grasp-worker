#!/usr/bin/env bash
set -euo pipefail
cd /data/sundawei/g1d_grasp_worker
(
  flock -x 9
  until [[ -f runs/step3a_01_l3_r0_demo_00.complete ]]; do
    sleep 5
  done
) 9>runs/step3a_01_gpu_1.lock
python3 background_status.py --record-exit 0 step3a_01_legacy_gpu_hold
