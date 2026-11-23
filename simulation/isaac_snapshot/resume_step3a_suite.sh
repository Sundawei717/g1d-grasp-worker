#!/usr/bin/env bash
set -euo pipefail
cd /data/sundawei/g1d_grasp_worker
stage=$1
[[ $stage =~ ^(l2|l3|l4)$ ]]
job=step3a_01_$stage
(
 flock -x 9
 if [[ -f "runs/$job.complete" ]]; then exit 0; fi
 pid="$(cat "runs/$job.pid")"
 cmd="$(tr '\000' ' ' < "/proc/$pid/cmdline")"
 [[ "$cmd" == *" step3a_suites.py $stage "* ]]
 kill -TERM "$pid"
 printf '%s\n' 'Resuming from planned receipts and retaining contact diagnostics under the unified acceptance contract.' >> "runs/$job.log"
 nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/sundawei/vla_rsi_g1d/envs/sim60/bin/python -Bu step3a_suites.py "$stage" >> "runs/$job.log" 2>&1 </dev/null 9>&- &
 printf '%s\n' "$!" > "runs/$job.pid"
) 9>runs/step3a_01_sim.lock
