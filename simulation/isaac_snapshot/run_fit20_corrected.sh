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
replay="$project/runs/visibility_replay_01"
corrections="$project/runs/policy_correction_capture_01"
for dependency in "$replay" "$corrections"; do
    printf 'Waiting for %s.complete\n' "$dependency"
    for tick in $(seq 1 4800); do
        [[ ! -s "$dependency.complete" ]] || break
        sleep 3
    done
    [[ -s "$dependency.complete" ]]
done
gpu=${G1D_TRAIN_GPU:-2}
memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
if ((memory < 20480)); then
    printf 'GPU %s has less than 20 GiB free\n' "$gpu"
    exit 75
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
flags=(--device "cuda:$gpu" --steps 2000 --action-reference measured --visibility-replay "$replay/result.json" --prompt-training "${2:-visible_only}")
if [[ -s "$corrections/result.json" ]]; then
    flags+=(--corrections "$corrections/result.json")
fi
timeout -k 30 14400 "$python" -Bu "$project/train_policy.py" "$project/runs/train_bootstrap_01/result.json" "$project/cache/weights/resnet18-f37072fd.pth" "$output" "${flags[@]}"
