#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
data=/data/sundawei/vla_rsi_g1d
python="$data/envs/sim60/bin/python"
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^overnight_expert_[0-9]+$ && ! -e "$output" ]]
mkdir "$output"
on_exit() {
    local code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$job"
    printf '%s\n' "$code" > "$output.exit"
}
trap on_exit EXIT
urdf="$data/assets/Assets/Robots/g1d/g1d_dex1.urdf"
flags=(--hours 7 --reserve-gib 50)
if [[ $# == 2 ]]; then
    previous=$(realpath -e -- "$2")
    [[ ${previous%/*} == "$project/runs" && ${previous##*/} =~ ^overnight_expert_[0-9]+$ ]]
    flags+=(--resume-from "$previous")
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
timeout -k 45 28800 "$python" -Bu "$project/collect_training_overnight.py" "$project" "$urdf" "$output" "${flags[@]}"
