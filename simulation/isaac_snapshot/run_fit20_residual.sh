#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^fit20_[0-9]+$ ]]
mkdir "$output"
on_exit() {
    local code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$job"
    printf '%s\n' "$code" > "$output.exit"
}
trap on_exit EXIT
prior="$project/runs/fit20_closed_loop_01_can"
printf 'Waiting for the first closing-loop GPU 2 shard at %s.complete\n' "$prior"
for tick in $(seq 1 3600); do
    [[ ! -s "$prior.complete" ]] || break
    sleep 3
done
[[ -s "$prior.complete" ]]
memory=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
if ((memory < 20480)); then
    printf 'GPU 2 has less than 20 GiB free\n'
    exit 75
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
timeout -k 30 14400 "$python" -Bu "$project/train_policy.py" "$project/runs/train_bootstrap_01/result.json" "$project/cache/weights/resnet18-f37072fd.pth" "$output" --device cuda:2 --steps 2000 --action-reference measured
