#!/usr/bin/env bash
set -euo pipefail
cd /data/sundawei/g1d_grasp_worker
ulimit -c 0
case_id=$1
shift
data=/data/sundawei/vla_rsi_g1d
unset CUDA_VISIBLE_DEVICES VK_INSTANCE_LAYERS VK_LAYER_PATH VK_ADD_LAYER_PATH VK_LOADER_LAYERS_ENABLE
source "$data/sim60/environment.sh"
export OMNI_KIT_ACCEPT_EULA=YES ENABLE_CAMERAS=1 LD_LIBRARY_PATH="$data/tools/native_lib60_os_deps"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$G1D_GPU_UUID"
export XDG_CACHE_HOME="$G1D_RUNTIME_DIR/xdg" CUDA_CACHE_PATH="$G1D_RUNTIME_DIR/cuda" TMPDIR="$G1D_RUNTIME_DIR/tmp"
mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TMPDIR"
export PATH="$data/envs/sim60/bin:$PATH"
timeout -k 15 700 "$data/envs/sim60/bin/python" -Bu v3_a_audit_live.py tap "$case_id" "$@"
