#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^three_view_pipeline_[0-9]+$ ]]
mkdir "$output"
trap 'code=$?; python3 "$project/background_status.py" --record-exit "$code" "$job"; printf "%s\n" "$code" > "$output.exit"' EXIT
trap 'exit 143' TERM
collection="$project/runs/${2:-three_view_expert_03}"
updates=${3:-2000}
test_per_group=${4:-4}
detector_config="$project/three_view_detector.json"
weights="$project/cache/weights/resnet18-f37072fd.pth"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$project/cache/three_view_detector/deps${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="$project/cache/three_view_detector/config"
export YOLO_AUTOINSTALL=false

printf 'Waiting for %s.complete\n' "$collection"
while [[ ! -s "$collection.complete" ]]; do
    sleep 10
done
test -s "$collection/result.json"
if [[ ${collection##*/} == three_view_scale_* ]]; then
    python3 -B - "$collection/result.json" <<'PY'
import json, sys
receipt=json.load(open(sys.argv[1]))
assert receipt["passed"] and receipt["accepted"]==250, "The 250-episode collection did not complete"
PY
fi
if [[ ${collection##*/} == three_view_training_data_* ]]; then
    python3 -B - "$collection/result.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d.get("asset_balance") and d.get("passed") and d["added_episodes"]>0, "Diversified training data have not passed"
PY
fi
"$python" -B "$project/three_view_train.py" split "$collection/result.json" "$output/split.json"
"$python" -B "$project/three_view_eval.py" freeze "$project" "$output/split.json" "$output/frozen" --per-group "$test_per_group"

run_child() {
    local name=$1 code=0
    shift
    "$@" || code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$name"
    return "$code"
}

cache="${job}_boxes"
fit_a="${job}_fit_a"
fit_b="${job}_fit_b"
run_child "$cache" "$python" -Bu "$project/three_view_detector.py" "$collection/result.json" "$detector_config" "$project/runs/$cache" --device cuda:2 > "runs/$cache.log" 2>&1 &
cache_pid=$!
echo "$cache_pid" > "runs/$cache.pid"
printf 'Detecting wrists PID %s\n' "$cache_pid"
run_child "$fit_a" "$python" -Bu "$project/three_view_train.py" train "$output/split.json" "$weights" "$project/runs/$fit_a" --variant A --device cuda:1 --steps "$updates" > "runs/$fit_a.log" 2>&1 &
a=$!
echo "$a" > "runs/$fit_a.pid"
printf 'Training A PID %s\n' "$a"
wait "$cache_pid"
run_child "$fit_b" "$python" -Bu "$project/three_view_train.py" train "$output/split.json" "$weights" "$project/runs/$fit_b" --variant B --device cuda:2 --steps "$updates" --boxes "$project/runs/$cache/result.json" > "runs/$fit_b.log" 2>&1 &
b=$!
echo "$b" > "runs/$fit_b.pid"
printf 'Training B PID %s\n' "$b"
wait "$a"
wait "$b"

evaluate() {
    local variant=$1 fit=$2 name directory a b
    name="${job}_eval_$variant"
    directory="$project/runs/$name"
    if ((test_per_group > 4)); then export G1D_GROUPED_WALL_SECONDS=5400; fi
    mkdir "$directory"
    "$python" -B "$project/three_view_eval.py" prepare "$project" "$output/frozen" "$project/runs/$fit" "$detector_config" "$directory/plans"
    G1D_WORKER_CONFIG="$project/env_resolved.json" bash "$project/run_lab_school.sh" "${directory}_carton" --grouped-manifest "$directory/plans/shard_carton.json" --grasp-after "$project/runs/lab_school_03/result.json" --gpu 1 > "${directory}_carton.log" 2>&1 &
    a=$!
    G1D_WORKER_CONFIG="$project/runs/candidate_can_config.json" bash "$project/run_lab_school.sh" "${directory}_can" --grouped-manifest "$directory/plans/shard_can.json" --grasp-after "$project/runs/lab_school_03/result.json" --gpu 2 > "${directory}_can.log" 2>&1 &
    b=$!
    printf '%s\n%s\n' "$a" "$b" > "$directory/worker_pids"
    wait "$a" || printf 'Carton worker exit %s\n' "$?"
    wait "$b" || printf 'Can worker exit %s\n' "$?"
    "$python" -B "$project/three_view_eval.py" finish "$directory"
}
for pair in "a:$fit_a" "b:$fit_b"; do
    variant=${pair%%:*}
    name="${job}_eval_$variant"
    set +e
    (set -e; evaluate "$variant" "${pair#*:}") > "runs/$name.log" 2>&1 &
    worker=$!
    echo "$worker" > "runs/$name.pid"
    printf 'Evaluating %s PID %s\n' "$variant" "$worker"
    wait "$worker"
    code=$?
    set -e
    python3 "$project/background_status.py" --record-exit "$code" "$name"
    ((code == 0))
done
"$python" -B "$project/three_view_eval.py" compare "$project/runs/${job}_eval_a" "$project/runs/${job}_eval_b" "$output"
