#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
prefix=${output##*/}
[[ $output == "$project/runs/$prefix" && $prefix =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]
mkdir "$output"
on_exit() {
    local code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$prefix"
    printf '%s\n' "$code" > "$output.exit"
}
trap on_exit EXIT
plan=$(realpath -m -- "$2")
[[ ${plan%/*} == "$project/runs" && ${plan##*/} =~ ^grouped_plan_[0-9]+$ ]]
pilot="$project/runs/grouped_probe_01"
printf '%s\n' "Waiting for checked plans and the two-case scene probe"
for tick in $(seq 1 3000); do
    if [[ -s "$plan.complete" && -s "$pilot.complete" ]]; then
        break
    fi
    sleep 2
done
"$python" -B - "$plan" "$pilot" "$output" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path
plan, pilot, output = map(Path, sys.argv[1:])
for stage in (plan, pilot):
    marker = json.loads(stage.with_suffix(".complete").read_text())
    if marker["exit_code"] != 0:
        raise SystemExit(f"Prerequisite failed: {stage}")
checked = json.loads((plan / "result.json").read_text())
physical = json.loads((pilot / "result.json").read_text())
if any(row["status"] != "strict_success" for row in physical["cases"]) or len(physical["cases"]) != 2:
    raise SystemExit("Two-case physical probe did not pass; leave the 100-case suite unchanged")
feasible = Counter(row["group"] for row in checked["cases"] if row["offline_feasible"])
if sum(feasible.values()) < 90 or any(feasible[group] * 5 < count * 4 for group, count in checked["quotas"].items()):
    (output / "offline_gate.json").write_text(json.dumps({
        "configured": 100, "physical_attempts": 0, "offline_feasible": dict(feasible),
        "offline_rejects": [case["case_id"] for case in checked["cases"] if not case["offline_feasible"]],
        "quotas": checked["quotas"], "passed": False,
    }, indent=2))
    raise SystemExit(f"Offline rejects already make the physical gate impossible: {dict(feasible)}")
print("Offline feasibility:", dict(feasible), flush=True)
PY

worker() {
    local gpu=$1 key role config memory code
    shift
    for key in "$@"; do
        role=${key%_*}
        config="$project/env_resolved.json"
        if [[ $role == can ]]; then
            config="$project/runs/candidate_can_config.json"
        fi
        memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
        if ((memory < 20480)); then
            printf 'GPU %s below 20 GiB; did not launch %s\n' "$gpu" "$key"
            return 75
        fi
        code=0
        printf 'Launching %s_%s on GPU %s\n' "$prefix" "$key" "$gpu"
        G1D_WORKER_CONFIG="$config" bash "$project/run_lab_school.sh" "$project/runs/${prefix}_${key}" --grouped-manifest "$plan/shard_${key}.json" --grasp-after "$project/runs/lab_school_03/result.json" --gpu "$gpu" > "runs/${prefix}_${key}.log" 2>&1 || code=$?
        printf '%s: exit=%s\n' "$key" "$code"
    done
}

worker 1 carton_0 carton_2 can_1 &
first=$!
worker 2 carton_1 carton_3 can_0 &
second=$!
printf '%s\n%s\n' "$first" "$second" > "$output/worker_pids"
wait "$first" || printf 'GPU 1 worker exited %s\n' "$?"
wait "$second" || printf 'GPU 2 worker exited %s\n' "$?"
"$python" -B "$project/aggregate_grouped.py" "$plan" "$project/runs" "$prefix" --output "$output/result.json"
