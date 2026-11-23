#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^three_view_extra_physics_[0-9]+$ ]]
mkdir "$output"
trap 'code=$?; python3 "$project/background_status.py" --record-exit "$code" "$job"; printf "%s\n" "$code" > "$output.exit"' EXIT
python3 -B - "$output" <<'PY'
import json,sys
from pathlib import Path
from shutil import copyfile
out=Path(sys.argv[1])
def read(p):
    return json.loads(Path(p).read_text())
base=Path("runs/three_view_extra_plans_01")
src=read(base/"result.json")["cases"]
template=read("runs/train_bootstrap_01/plans/shard_carton.json")
for role in ("mug","unseen_can"):
    rows=[row for row in src if row["asset_role"]==role and row["offline_feasible"]]
    if not rows:
        continue
    usd=rows[0]["target_usd"]
    cfg={**read("env_resolved.json"), "target_usd":usd}
    manifest={**template, "phase":"grouped_expert_100", "asset":usd, "fixture_assets":{}, "cases":rows}
    p=out/role
    p.mkdir()
    (p/"config.json").write_text(json.dumps(cfg))
    (p/"manifest.json").write_text(json.dumps(manifest))
    for row in rows:
        copyfile(base/row["plan"],p/row["plan"])
PY
for role in mug unseen_can; do
    if [[ -f "$output/$role/manifest.json" ]]; then
        name="${job}_$role"
        G1D_WORKER_CONFIG="$output/$role/config.json" bash "$project/run_lab_school.sh" "$project/runs/$name" --grouped-manifest "$output/$role/manifest.json" --grasp-after "$project/runs/lab_school_03/result.json" --gpu 2 > "runs/$name.log" 2>&1 &
        pid=$!
        echo "$pid" > "runs/$name.pid"
        echo "Validating $role PID $pid"
        wait "$pid" || echo "$role launcher exit $?"
    fi
done
python3 -B - "$output" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
rows=[]
for role in ("mug","unseen_can"):
    path=out.with_name(f"{out.name}_{role}")/"result.json"
    if path.is_file():
        rows.extend({**row,"asset_role":role} for row in json.loads(path.read_text())["cases"])
passed=bool(rows) and all(any(row["asset_role"]==role and row["status"]=="strict_success" for row in rows)
                         for role in ("mug","unseen_can"))
(out/"result.json").write_text(json.dumps({"phase":"three_view_extra_physics","passed":passed,"cases":rows},indent=2))
print("Expert physics",[(row["case"],row["status"]) for row in rows])
raise SystemExit(0 if passed else 1)
PY
