#!/usr/bin/env bash
set -euo pipefail
cd /data/sundawei/g1d_grasp_worker
job=step3a_01_ycb
trap 'c=$?; python3 background_status.py --record-exit "$c" "$job"' EXIT
for asset in 004_sugar_box 006_mustard_bottle 007_tuna_fish_can 008_pudding_box 011_banana 019_pitcher_base 024_bowl 025_mug 035_power_drill; do
    file="runs/step3a_01/source/${asset}.tgz"
    if [[ ! -s "$file" ]]; then
        curl --fail --silent --show-error --location --max-time 90 --retry 2 \
          "https://ycb-benchmarks.s3.amazonaws.com/data/google/${asset}_google_16k.tgz" -o "$file.partial"
        tar -tzf "$file.partial" >/dev/null
        mv "$file.partial" "$file"
    fi
    sha256sum "$file"
done
