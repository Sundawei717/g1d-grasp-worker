#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
data=/data/sundawei/vla_rsi_g1d
python="$data/envs/sim60/bin/python"
cd "$project"
output=$(realpath -m -- "$1")
prefix=${output##*/}
[[ $output == "$project/runs/$prefix" && $prefix =~ ^collection_preview_[0-9]+$ ]]
mkdir "$output"
on_exit() {
    local code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$prefix"
    printf '%s\n' "$code" > "$output.exit"
}
trap on_exit EXIT
expert=$(realpath -m -- "$2")
[[ ${expert%/*} == "$project/runs" && ${expert##*/} =~ ^grouped_eval_[0-9]+$ ]]
printf 'Waiting for the 100-case physical expert gate at %s\n' "$expert.complete"
for tick in $(seq 1 4800); do
    [[ ! -s "$expert.complete" ]] || break
    sleep 3
done
[[ -s "$expert.complete" ]]
if [[ $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["exit_code"])' "$expert.complete") != 0 ]]; then
    printf 'The expert gate ended without acceptance. Collection did not start.\n'
    exit 1
fi
urdf="$data/assets/Assets/Robots/g1d/g1d_dex1.urdf"
timeout -k 15 2400 "$python" -Bu "$project/prepare_collection_preview.py" "$project" "$urdf" "$expert/result.json" "$output/plans"

shard() {
    local role=$1 gpu=$2 config=$3 memory
    local -a flags
    memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    ((memory >= 20480))
    printf 'Launching %s_%s on GPU %s\n' "$prefix" "$role" "$gpu"
    flags=(--grouped-manifest "$output/plans/shard_$role.json" --grasp-after "$project/runs/lab_school_03/result.json" --gpu "$gpu")
    G1D_WORKER_CONFIG="$config" bash "$project/run_lab_school.sh" "$project/runs/${prefix}_$role" "${flags[@]}" > "runs/${prefix}_$role.log" 2>&1
}
shard carton 1 "$project/env_resolved.json" &
first=$!
shard can 2 "$project/runs/candidate_can_config.json" &
second=$!
wait "$first" || printf 'Capacity preview carton shard ended with exit %s\n' "$?"
wait "$second" || printf 'Capacity preview can shard ended with exit %s\n' "$?"

"$python" -B - "$output" <<'PY'
import json
import math
import shutil
import sys
from pathlib import Path

output = Path(sys.argv[1])
planned = json.loads((output / "plans/result.json").read_text())["cases"]
observed = {}
for role in ("carton", "can"):
    shard = output.parent / f"{output.name}_{role}"
    for name in ("result.json", "batch_progress.json"):
        path = shard / name
        if path.is_file():
            for row in json.loads(path.read_text())["cases"]:
                observed[row["case"]] = {**row, "shard": shard.name}
            break
rows = [observed.get(case["case_id"], {"case": case["case_id"], "status": "infrastructure_fail"}) for case in planned]
sizes = []
for row in rows:
    if row["status"] == "strict_success" and row.get("dataset_receipt"):
        path = output.parent / row["shard"] / row["case"] / "episode.h5"
        size = path.stat().st_size
        if size != row["dataset_receipt"]["bytes"]:
            raise ValueError(f"Episode has changed since validation: {path}")
        sizes.append(size)
free_bytes = shutil.disk_usage(output).free
total = sorted(sizes)[math.ceil(0.95 * len(sizes)) - 1] * 2100 if sizes else None
enough_examples = len(sizes) >= 20
passed = enough_examples and free_bytes > total
report = {"phase": "collection_capacity_preview", "configured": 20, "strict_success": len(sizes), "cases": rows,
          "measured_h5_file_sizes_bytes": sizes, "free_bytes": free_bytes, "projected_2100_h5_bytes": total,
          "measured_at_least_20_successful_episodes": enough_examples, "passed": passed,
          "scope": "capacity and clock alignment preview excluded from train, validation and test"}
(output / "result.json").write_text(json.dumps(report, indent=2))
print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
raise SystemExit(0 if passed else 1)
PY
