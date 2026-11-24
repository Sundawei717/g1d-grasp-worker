#!/usr/bin/env bash
set -euo pipefail
cd /data/sundawei/g1d_grasp_worker
job=step3a_01_l1
(
  flock -x 9
  pid="$(cat "runs/$job.pid")"
  if [[ -f "runs/$job.complete" ]]; then
     exit 1
  fi
  case "$(tr '\000' ' ' < "/proc/$pid/cmdline")" in
    *" step3a_l1.py "*) kill -TERM "$pid" ;;
    *) exit 1 ;;
  esac
) 9>runs/step3a_01_sim.lock
printf 'Resuming L1 with regenerated 8 cm plans for legacy meshes.\n' >> "runs/$job.log"
nohup env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
 /data/sundawei/vla_rsi_g1d/envs/sim60/bin/python -Bu step3a_l1.py >> "runs/$job.log" 2>&1 </dev/null &
printf '%s\n' "$!" > "runs/$job.pid"
