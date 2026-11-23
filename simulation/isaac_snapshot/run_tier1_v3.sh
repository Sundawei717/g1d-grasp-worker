#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath -- "$0")")"
job=${2:-tier1_v3_01}
[[ $job =~ ^tier1_v3_[0-9]+$ ]]
work="$PWD/runs/$job"

case ${1:-status} in
status)
    python3 background_status.py "$job" "${job}_plan" "${job}_collection" "${job}_audit" "${job}_cache" "${job}_model"
    for path in "$work/progress.json" "$work/cache_progress.json" "$work/final_inventory.json"; do
        if [[ -f $path ]]; then cat "$path"; fi
    done
    ;;
resume)
    [[ -f "$work/run.sh" && -f "$work/plan.sh" ]]
    exec 8>"$work/resume.lock"
    flock -n 8
    archive="$work/restarts/$(date +%Y%m%dT%H%M%S)"
    succeeded() {
        [[ -f $1 ]] && python3 -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["outcome"]!="PASS")' "$1"
    }
    revive() {
        local name=$1 script=$2 pattern=$3 pid=""
        if [[ -f "runs/$name.pid" ]]; then
            read -r pid < "runs/$name.pid"
        fi
        if [[ $pid =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq "$pattern"; then
            printf '%s running at PID %s\n' "$name" "$pid"
            return
        fi
        if succeeded "runs/$name.complete"; then return; fi
        mkdir -p "$archive"
        local marks=("runs/$name.complete")
        if [[ $name == "$job" ]]; then
            marks+=("runs/${job}_collection.complete" "runs/${job}_cache.complete")
        fi
        for mark in "${marks[@]}"; do
            if [[ -f $mark ]] && ! succeeded "$mark"; then mv -- "$mark" "$archive/"; fi
        done
        nohup bash "$script" >> "runs/$name.log" 2>&1 </dev/null 8>&- &
        printf '%s\n' "$!" > "runs/$name.pid"
        printf '%s resumed at PID %s\n' "$name" "$!"
    }
    if ! succeeded "runs/${job}_collection.complete"; then
        revive "${job}_plan" "$work/plan.sh" "$job/plan.sh"
    fi
    revive "$job" "$work/run.sh" "$job/run.sh"
    ;;
*)
    printf 'Usage: %s status|resume [tier1_v3_01]\n' "$0" >&2
    exit 2
    ;;
esac
