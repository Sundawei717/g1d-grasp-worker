#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
bootstrap=$(realpath -m -- "$2")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^fit20_[0-9]+$ ]]
[[ ${bootstrap%/*} == "$project/runs" && ${bootstrap##*/} =~ ^train_bootstrap_[0-9]+$ ]]
mkdir "$output"
on_exit() {
    local code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$job"
    printf '%s\n' "$code" > "$output.exit"
}
trap on_exit EXIT
if [[ $# == 3 ]]; then
    previous=$(realpath -e -- "$3")
    [[ ${previous%/*} == "$project/runs" && ${previous##*/} =~ ^fit20_[0-9]+$ && -s "$previous/checkpoint.pt" ]]
    cp --reflink=auto "$previous/checkpoint.pt" "$output/checkpoint.pt"
fi
printf 'Waiting for the 20 frozen independent training episodes at %s\n' "$bootstrap.complete"
for tick in $(seq 1 4800); do
    [[ ! -s "$bootstrap.complete" ]] || break
    sleep 3
done
[[ -s "$bootstrap.complete" ]]
memory=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
if ((memory < 20480)); then
    printf 'GPU 2 below 20 GiB free for fit20\n'
    exit 75
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
timeout -k 30 14400 "$python" -Bu "$project/train_policy.py" "$bootstrap/result.json" "$project/cache/weights/resnet18-f37072fd.pth" "$output" --device cuda:2 --steps 2000
