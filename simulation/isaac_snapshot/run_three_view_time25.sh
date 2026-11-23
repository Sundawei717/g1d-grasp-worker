#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^three_view_time25_[0-9]+$ ]]
mkdir "$output"
trap 'code=$?; python3 "$project/background_status.py" --record-exit "$code" "$job"; printf "%s\n" "$code" > "$output.exit"' EXIT
trap 'exit 143' TERM
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$project/cache/three_view_detector/deps${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="$project/cache/three_view_detector/config"
export YOLO_AUTOINSTALL=false
old="$project/runs/three_view_pipeline_03"
"$python" -B three_view_time25.py freeze "$output"

run_stage() {
    local stage=$1 code=0
    "$python" -Bu three_view_time25.py "$stage" "$output" || code=$?
    python3 "$project/background_status.py" --record-exit "$code" "${job}_$stage"
    return "$code"
}

run_training() {
    local variant=$1 name code=0
    name="${job}_fit_$variant"
    upper=${variant^^}
    flags=()
    if [[ $variant == b ]]; then
        flags+=(--boxes "$old"_boxes/result.json)
    fi
    "$python" -Bu three_view_train.py train "$old/split.json" "$project/cache/weights/resnet18-f37072fd.pth" "$project/runs/$name" --variant "$upper" --device cuda:0 --steps 2000 --seed 33890 --initialize-from "${old}_fit_$variant/checkpoint.pt" --select-final "${flags[@]}" || code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$name"
    return "$code"
}

run_stage baseline >"runs/${job}_baseline.log" 2>&1 &
baseline=$!
echo "$baseline" >"runs/${job}_baseline.pid"
printf 'Existing checkpoints on 38 validation scenes with the 25-second horizon PID %s\n' "$baseline"
for variant in a b; do
    name="${job}_fit_$variant"
    run_training "$variant" >"runs/$name.log" 2>&1 &
    pid=$!
    echo "$pid" >"runs/$name.pid"
    printf 'Continuation %s PID %s\n' "$variant" "$pid"
    wait "$pid"
done
wait "$baseline"
run_stage candidate >"runs/${job}_candidate.log" 2>&1 &
candidate=$!
echo "$candidate" >"runs/${job}_candidate.pid"
printf 'Candidate validation and subsequent independent test PID %s\n' "$candidate"
wait "$candidate"
