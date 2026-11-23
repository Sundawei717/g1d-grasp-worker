#!/usr/bin/env bash
set -euo pipefail
project=/data/sundawei/g1d_grasp_worker
python=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
cd "$project"
output=$(realpath -m -- "$1")
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^sam2_setup_[0-9]+$ ]]
mkdir "$output"
on_exit() {
    local code=$?
    python3 "$project/background_status.py" --record-exit "$code" "$job"
    printf '%s\n' "$code" > "$output.exit"
}
trap on_exit EXIT
install="$project/cache/sam2_2b90b9f"
commit=2b90b9f5ceec907a1c18123530e92e794ad901a4
checkpoint_source='https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt'
mkdir -p "$install"
if [[ ! -d "$install/source/.git" ]]; then
    timeout 75 git clone --depth 1 https://github.com/facebookresearch/sam2.git "$install/source"
fi
[[ $(git -C "$install/source" rev-parse HEAD) == "$commit" ]]
if [[ ! -f "$install/sam2.1_hiera_tiny.pt" ]]; then
    curl --fail --location --retry 2 --connect-timeout 12 --max-time 180 "$checkpoint_source" -o "$install/checkpoint.partial"
    [[ $(stat -c %s "$install/checkpoint.partial") == 156008466 ]]
    mv -T -- "$install/checkpoint.partial" "$install/sam2.1_hiera_tiny.pt"
fi
[[ $(stat -c %s "$install/sam2.1_hiera_tiny.pt") == 156008466 ]]
export PYTHONPATH="$install/source:$install/dependencies${PYTHONPATH:+:$PYTHONPATH}"
if ! "$python" -B -c 'import iopath, portalocker' >/dev/null 2>&1; then
    timeout 90 "$python" -m pip --disable-pip-version-check install --target "$install/dependencies" --no-deps iopath==0.1.10 portalocker==3.2.0
fi
"$python" -B - "$install" "$commit" "$checkpoint_source" "$output" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch
from sam2.build_sam import build_sam2

install, commit, url, output = sys.argv[1:]
checkpoint = Path(install) / "sam2.1_hiera_tiny.pt"
config = "configs/sam2.1/sam2.1_hiera_t.yaml"
model = build_sam2(config, str(checkpoint), device="cpu", apply_postprocessing=False)
result = {"phase": "sam2_tracking_setup", "source_commit": commit, "checkpoint": str(checkpoint),
          "checkpoint_source": url, "checkpoint_sha256": hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest(),
          "parameters": sum(param.numel() for param in model.parameters()), "config": config, "torch": torch.__version__,
          "passed": True}
(Path(output) / "result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
PY
