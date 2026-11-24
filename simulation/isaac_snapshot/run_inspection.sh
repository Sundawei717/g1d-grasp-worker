#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
project=/data/sundawei/g1d_grasp_worker
data=/data/sundawei/vla_rsi_g1d
unset CUDA_VISIBLE_DEVICES VK_INSTANCE_LAYERS VK_LAYER_PATH VK_ADD_LAYER_PATH VK_LOADER_LAYERS_ENABLE
export PYTHONDONTWRITEBYTECODE=1 OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/sundawe1/vla_rsi_g1d/repos/pyrua-ewam-agent
export LD_LIBRARY_PATH="$data/tools/native_lib60_os_deps"
export XDG_CACHE_HOME="$project/cache/xdg" CUDA_CACHE_PATH="$project/cache/cuda" TMPDIR="$project/cache/tmp"
mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TMPDIR"
status=0
output=$1
shift
timeout -k 10 240 "$data/envs/sim60/bin/python" -u "$project/inspect_environment.py" "$project/env_resolved.json" "$output" "$@" || status=$?
if ((status == 0)) && [[ ! -s "$output/environment.json" || ! -s "$output/head_left_raw.png" || -e "$output/error.txt" ]]; then
    status=1
fi
exit "$status"
