#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^three_view_(generalization|parts)_[0-9]+$ ]]
mkdir "$output"
trap 'code=$?; python3 "$project/background_status.py" --record-exit "$code" "$job"; printf "%s\n" "$code" > "$output.exit"' EXIT
trap 'exit 143' TERM
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$project/cache/three_view_detector/deps${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="$project/cache/three_view_detector/config"
export YOLO_AUTOINSTALL=false
main=${2:-three_view_pipeline_01}
preflight=three_view_extra_physics_01
freeze=freeze-extra
roles=(mug unseen_can)
after_main=()
if [[ $job == three_view_parts_* ]]; then
    preflight=three_view_parts_capture_01
    freeze=freeze-parts
    roles=(carton can)
    after_main=("three_view_generalization_${job##*_}")
fi
for dependency in "$preflight" "$main" "${after_main[@]}"; do
    printf 'Waiting for %s.complete\n' "$dependency"
    while [[ ! -s "runs/$dependency.complete" ]]; do sleep 10; done
done
"$python" -B three_view_eval.py "$freeze" "$project" "runs/$main/split.json" "$output/frozen"
evaluate() {
    local variant=$1 directory=$2 role pid
    local -a workers=()
    mkdir "$directory"
    "$python" -B three_view_eval.py prepare "$project" "$output/frozen" "$project/runs/${main}_fit_$variant" "$project/three_view_detector.json" "$directory/plans"
    for role in "${roles[@]}"; do
        if [[ -f "$directory/plans/shard_$role.json" ]]; then
            gpu=1
            [[ $role != unseen_can && $role != can ]] || gpu=2
            config="$directory/plans/config_$role.json"
            if [[ $role == carton ]]; then config="$project/env_resolved.json"; fi
            if [[ $role == can ]]; then config="$project/runs/candidate_can_config.json"; fi
            G1D_WORKER_CONFIG="$config" bash "$project/run_lab_school.sh" "${directory}_$role" --grouped-manifest "$directory/plans/shard_$role.json" --grasp-after "$project/runs/lab_school_03/result.json" --gpu "$gpu" >"${directory}_$role.log" 2>&1 &
            pid=$!
            workers+=("$pid")
            echo "$role PID $pid"
        fi
    done
    printf '%s\n' "${workers[@]}" > "$directory/worker_pids"
    for pid in "${workers[@]}"; do wait "$pid" || echo "Simulator exit $?"; done
    "$python" -B three_view_eval.py finish "$directory"
}
for variant in a b; do
    name="${job}_eval_$variant"
    set +e
    (set -e; evaluate "$variant" "$project/runs/$name") >"runs/$name.log" 2>&1 &
    worker=$!
    printf '%s\n' "$worker" > "runs/$name.pid"
    echo "Evaluating $variant PID $worker"
    wait "$worker"
    code=$?
    set -e
    python3 background_status.py --record-exit "$code" "$name"
    ((code == 0))
done
"$python" -B three_view_eval.py compare "$project/runs/${job}_eval_a" "$project/runs/${job}_eval_b" "$output"
