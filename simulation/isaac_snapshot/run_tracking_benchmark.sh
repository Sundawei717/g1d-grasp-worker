#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^tracking_benchmark_[0-9]+$ && ! -e "$output" ]]
on_exit() {
    local code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$job"
    printf '%s\n' "$code" > "$output.exit"
}
trap on_exit EXIT
[[ -s runs/sam2_setup_01.complete && -s runs/sam2_setup_01/result.json ]]
install="$project/cache/sam2_2b90b9f"
export PYTHONPATH="$install/source:$install/dependencies${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
gpu=1
flags=()
if [[ $# == 2 ]]; then
    motion=$(realpath -m -- "$2")
    [[ ${motion%/*} == "$project/runs" && ${motion##*/} =~ ^tracking_motion_[0-9]+$ ]]
    printf 'Waiting for %s.complete\n' "$motion"
    for tick in $(seq 1 1800); do
        [[ ! -s "$motion.complete" ]] || break
        sleep 3
    done
    [[ -s "$motion.complete" && -s "$motion/result.json" ]]
    gpu=2
    flags=(--motion-run "$motion")
fi
memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
if ((memory < 16000)); then
    printf 'GPU %s has less than 16 GB free for the SAM2 video comparison\n' "$gpu"
    exit 75
fi
timeout -k 15 3600 "$python" -Bu "$project/benchmark_tracking.py" "$project" "$output" --device "cuda:$gpu" "${flags[@]}"
