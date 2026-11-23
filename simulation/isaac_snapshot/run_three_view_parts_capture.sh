#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^three_view_parts_capture_[0-9]+$ ]]
mkdir "$output"
trap 'code=$?; python3 "$project/background_status.py" --record-exit "$code" "$job"; printf "%s\n" "$code" > "$output.exit"' EXIT
part_plans=${3:-three_view_part_plans_01}
wait_for=()
if [[ $# -ge 4 ]]; then wait_for+=("$4"); fi
for dependency in "$part_plans" three_view_expert_03 three_view_extra_physics_01 "${wait_for[@]}"; do
    printf 'Waiting for %s.complete\n' "$dependency"
    while [[ ! -s "runs/$dependency.complete" ]]; do sleep 10; done
done
python3 -B - "$output/plans" "${2:-0}" "runs/$part_plans" <<'PY'
import json,sys
from pathlib import Path
from collections import Counter
out=Path(sys.argv[1])
out.mkdir()
base=Path(sys.argv[3])
src=json.loads((base/"result.json").read_text())["cases"]
(out/"reused.json").write_text('{"cases":[],"episodes":[]}')
for role,query in (("carton","box"),("can","can")):
    name=f"shard_{role}.json"
    manifest=json.loads((Path("runs/train_bootstrap_01/plans")/name).read_text())
    candidates=[row for row in src if row["asset_role"]==role and row["offline_feasible"]]
    pairs=Counter(row["scene_family"] for row in candidates)
    paired_families=sorted(key for key,count in pairs.items() if count==2)
    if int(sys.argv[2]):
        paired_families=paired_families[:int(sys.argv[2])]
    rows=[]
    for row in candidates:
        if row["scene_family"] not in paired_families:
            continue
        row={**row,"split":"train","command_delay_substeps":0}
        original=json.loads((base/row["plan"]).read_text())
        original["case"].update(split="train",command_delay_substeps=0)
        (out/row["plan"]).write_text(json.dumps(original))
        rows.append(row)
    if rows:
        (out/name).write_text(json.dumps({**manifest,"cases":rows,"three_view":True,"target_query":query,"fixture_assets":{}}))
PY
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
run_role() {
    local role=$1 gpu=$2 config=$3
    G1D_WORKER_CONFIG="$config" bash "$project/run_lab_school.sh" "$project/runs/${job}_$role" --grouped-manifest "$output/plans/shard_$role.json" --grasp-after "$project/runs/lab_school_03/result.json" --gpu "$gpu"
}
workers=()
if [[ -f "$output/plans/shard_carton.json" ]]; then
    run_role carton 1 "$project/env_resolved.json" > "runs/${job}_carton.log" 2>&1 &
    workers+=("$!")
fi
if [[ -f "$output/plans/shard_can.json" ]]; then
    run_role can 2 "$project/runs/candidate_can_config.json" > "runs/${job}_can.log" 2>&1 &
    workers+=("$!")
fi
[[ ${#workers[@]} -gt 0 ]]
printf '%s\n' "${workers[@]}" | tee "$output/worker_pids"
for pid in "${workers[@]}"; do wait "$pid" || echo "Simulator $pid exit $?"; done
"$python" -B "$project/three_view_collection.py" finish "$output"
