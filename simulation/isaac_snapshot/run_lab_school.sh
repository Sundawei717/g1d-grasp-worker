#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
project=/data/sundawei/g1d_grasp_worker
data=/data/sundawei/vla_rsi_g1d
output=$(realpath -m -- "$1")
shift
job=${output##*/}
[[ $output == "$project/runs/$job" && $job =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]
[[ ! -e $output && ! -e $output.exit && ! -e $output.complete ]]
complete_job() {
    local code=$?
    if [[ ${G1D_RUNTIME_DIR:-} == "/dev/shm/g1d-$(id -u)-$job" && ! -L ${G1D_RUNTIME_DIR:-} ]]; then
        # Only delete this shard's disposable runtime caches. Its result and logs live under runs/.
        rm -rf -- "$G1D_RUNTIME_DIR/xdg" "$G1D_RUNTIME_DIR/cuda" "$G1D_RUNTIME_DIR/tmp"
        for kit_cache in "$G1D_RUNTIME_DIR"/g1d_three_view_kit_*/cache; do
            if [[ -d $kit_cache && ! -L $kit_cache && "$(realpath -e -- "$kit_cache")" == "$kit_cache" ]]; then
                rm -rf -- "$kit_cache"
            fi
        done
        if [[ $job == step3b_01_* && "$(realpath -e -- "$G1D_RUNTIME_DIR")" == "$G1D_RUNTIME_DIR" ]]; then
            rm -rf -- "$G1D_RUNTIME_DIR"
        fi
    fi
    if [[ $job == step3b_01_* ]]; then
        local cache_bytes=0 removed=false
        if [[ -f $output/kit_cache_before_cleanup.bytes ]]; then
            cache_bytes=$(< "$output/kit_cache_before_cleanup.bytes")
        fi
        [[ ! -e "$G1D_RUNTIME_DIR" ]] && removed=true
        printf '{"job":"%s","kit_cache_bytes":%s,"runtime_removed":%s,"exit":%s}\n' "$job" "$cache_bytes" "$removed" "$code" > "$project/runs/step3b_01/cache_events/$job.tmp"
        mv -- "$project/runs/step3b_01/cache_events/$job.tmp" "$project/runs/step3b_01/cache_events/$job.json"
    fi
    python3 "$project/background_status.py" --record-exit "$code" "$job" || printf 'Cannot write completion marker for %s\n' "$job" >&2
}
trap complete_job EXIT
config=$(realpath -e -- "${G1D_WORKER_CONFIG:-$project/env_resolved.json}")
[[ $config == "$project/"* && -f $config ]]
wall_seconds=360
manifest=""
for ((i=1; i<=$#; i++)); do
    option=${!i}
    if [[ $option == --admission-manifest ]]; then
        wall_seconds=1800
    fi
    if [[ $option == --grouped-manifest ]]; then
        wall_seconds=${G1D_GROUPED_WALL_SECONDS:-2400}
        next=$((i+1))
        manifest=${!next}
    fi
done
unset CUDA_VISIBLE_DEVICES VK_INSTANCE_LAYERS VK_LAYER_PATH VK_ADD_LAYER_PATH VK_LOADER_LAYERS_ENABLE
source "$data/sim60/environment.sh"
export OMNI_KIT_ACCEPT_EULA=YES ENABLE_CAMERAS=1
export LD_LIBRARY_PATH="$data/tools/native_lib60_os_deps"
if [[ -n $manifest ]] && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(d.get("detector") is None)' "$manifest"; then
    export PYTHONPATH="$project/cache/three_view_detector/deps${PYTHONPATH:+:$PYTHONPATH}"
fi
export XDG_CACHE_HOME="$project/cache/xdg" CUDA_CACHE_PATH="$project/cache/cuda" TMPDIR="$project/cache/tmp"
if [[ -n ${G1D_GPU_UUID:-} ]]; then
    [[ $G1D_GPU_UUID =~ ^GPU-[0-9a-f-]+$ && $G1D_RENDER_GPU =~ ^[0-2]$ && -d $G1D_RUNTIME_DIR ]]
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$G1D_GPU_UUID"
    export XDG_CACHE_HOME="$G1D_RUNTIME_DIR/xdg" CUDA_CACHE_PATH="$G1D_RUNTIME_DIR/cuda" TMPDIR="$G1D_RUNTIME_DIR/tmp"
fi
export PATH="$data/envs/sim60/bin:$PATH"
mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TMPDIR"
cd "$project"
status=0
timeout -k 15 "$wall_seconds" "$data/envs/sim60/bin/python" -u "$project/lab_school_probe.py" "$config" "$output" "$@" || status=$?
if ((status == 0)); then
    "$data/envs/sim60/bin/python" -c 'import json, sys; assert json.load(open(sys.argv[1])).get("passed") is True' "$output/result.json" || status=1
fi
exit "$status"
