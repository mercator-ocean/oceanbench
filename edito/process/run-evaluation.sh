#!/bin/bash

# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# Entry script of the EDITO "generic process" job that runs an OceanBench evaluation.
# The job runs it with "sh", so it restarts itself with bash. Inputs are environment variables,
# described in edito/process/README.md. Never add "set -x": the environment holds credentials.

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

readonly PROCESS_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_DIRECTORY="$(cd "$PROCESS_DIRECTORY/../.." && pwd)"
readonly JOB_HELPER="$PROCESS_DIRECTORY/edito_job.py"
readonly SHARDS_HELPER="$PROCESS_DIRECTORY/shards.py"

readonly CHALLENGER="${CHALLENGER:-}"
readonly MODE="${MODE:-evaluate}"
readonly SHARD_PREFIXES="${SHARD_PREFIXES:-}"
readonly START_DATES="${START_DATES:-}"
readonly START_COUNT="${START_COUNT:-}"
readonly REGION="${REGION:-}"
readonly VIEWER="${VIEWER:-true}"
readonly VIEWER_ONLY="${VIEWER_ONLY:-false}"
readonly SCORES_FILE="${SCORES_FILE:-}"
OUTPUT_BUCKET="${OUTPUT_BUCKET:-}"
readonly OUTPUT_PREFIX="${OUTPUT_PREFIX:-}"
readonly OVERWRITE="${OVERWRITE:-false}"
readonly UPLOAD="${UPLOAD:-true}"
readonly PRINT_SCORES="${PRINT_SCORES:-auto}"
readonly WORK_DIRECTORY="${WORK_DIR:-/tmp/oceanbench-job}"
readonly PY_EDDY_TRACKER_SPEC="${OCEANBENCH_PY_EDDY_TRACKER_SPEC:-}"

readonly MONITOR_DIRECTORY="$WORK_DIRECTORY/monitor"
readonly RUN_DIRECTORY="$WORK_DIRECTORY/run"
readonly VIEWER_DIRECTORY="$WORK_DIRECTORY/viewer"
readonly CHALLENGER_DIRECTORY="$WORK_DIRECTORY/challenger"
readonly MERGE_INPUTS_DIRECTORY="$RUN_DIRECTORY/merge-inputs"
readonly SHARDS_DIRECTORY="$WORK_DIRECTORY/shards"
readonly JOB_START_SECONDS=$(date +%s)

CURRENT_STAGE="setup"
STAGE_START_SECONDS=$JOB_START_SECONDS
SAMPLER_PID=""

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

fail() {
    log "ERROR: $*" >&2
    exit 1
}

is_true() {
    case "${1,,}" in
        true | 1 | yes | on) return 0 ;;
        *) return 1 ;;
    esac
}

cgroup_file() {
    for path in "$@"; do
        if [ -r "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

memory_usage_bytes() {
    cat "$(cgroup_file /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory/memory.usage_in_bytes)" 2>/dev/null || echo 0
}

anonymous_memory_bytes() {
    local statistics
    statistics="$(cgroup_file /sys/fs/cgroup/memory.stat /sys/fs/cgroup/memory/memory.stat)" || {
        echo 0
        return
    }
    awk '$1 == "anon" || $1 == "total_rss" {print $2; exit}' "$statistics"
}

to_gigabytes() {
    awk -v bytes="$1" 'BEGIN {printf "%.2f GB", bytes / 1e9}'
}

memory_events() {
    local events
    events="$(cgroup_file /sys/fs/cgroup/memory.events)" || {
        echo "-1 -1 -1 -1"
        return
    }
    awk '{value[$1] = $2} END {printf "%d %d %d %d", value["high"], value["max"], value["oom"], value["oom_kill"]}' "$events"
}

memory_sampler() {
    local sample=0 minute_usage=0 minute_anonymous=0
    local alert_bytes
    alert_bytes=$(awk -v gb="${MEMORY_ALERT_GB:-20}" 'BEGIN {printf "%d", gb * 1e9}')
    echo "epoch,current_bytes,anon_bytes,events_high,events_max,events_oom,events_oom_kill" >"$MONITOR_DIRECTORY/memory.csv"
    while true; do
        local usage anonymous stage_usage stage_anonymous job_usage job_anonymous high max oom oom_kill
        usage=$(memory_usage_bytes)
        anonymous=$(anonymous_memory_bytes)
        read -r high max oom oom_kill <<<"$(memory_events)"
        echo "$(date +%s),$usage,$anonymous,$high,$max,$oom,$oom_kill" >>"$MONITOR_DIRECTORY/memory.csv"
        minute_usage=$((usage > minute_usage ? usage : minute_usage))
        minute_anonymous=$((anonymous > minute_anonymous ? anonymous : minute_anonymous))
        if [ "$anonymous" -gt "$alert_bytes" ]; then
            log "[memory high] anonymous $(to_gigabytes "$anonymous"), with page cache $(to_gigabytes "$usage")"
        fi
        if [ $((sample % 12)) -eq 0 ]; then
            log "[memory] anonymous $(to_gigabytes "$anonymous"), with page cache $(to_gigabytes "$usage");" \
                "events high $high max $max oom $oom oom_kill $oom_kill;" \
                "max of the 5 s samples since the last line: anonymous $(to_gigabytes "$minute_anonymous")," \
                "with page cache $(to_gigabytes "$minute_usage")"
            minute_usage=0 minute_anonymous=0
        fi
        sample=$((sample + 1))
        read -r stage_usage stage_anonymous <"$MONITOR_DIRECTORY/stage_peak" || true
        read -r job_usage job_anonymous <"$MONITOR_DIRECTORY/job_peak" || true
        echo "$((usage > ${stage_usage:-0} ? usage : ${stage_usage:-0})) $((anonymous > ${stage_anonymous:-0} ? anonymous : ${stage_anonymous:-0}))" >"$MONITOR_DIRECTORY/stage_peak.new"
        mv "$MONITOR_DIRECTORY/stage_peak.new" "$MONITOR_DIRECTORY/stage_peak"
        echo "$((usage > ${job_usage:-0} ? usage : ${job_usage:-0})) $((anonymous > ${job_anonymous:-0} ? anonymous : ${job_anonymous:-0}))" >"$MONITOR_DIRECTORY/job_peak.new"
        mv "$MONITOR_DIRECTORY/job_peak.new" "$MONITOR_DIRECTORY/job_peak"
        sleep 5
    done
}

print_resources() {
    local cpu_limit memory_limit
    cpu_limit="$(cat "$(cgroup_file /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpu/cpu.cfs_quota_us)" 2>/dev/null || echo unknown)"
    memory_limit="$(cat "$(cgroup_file /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes)" 2>/dev/null || echo unknown)"
    log "cgroup CPU limit (quota period): $cpu_limit; usable CPUs: $(python3 "$JOB_HELPER" cpu-limit 2>/dev/null || nproc)"
    log "cgroup memory limit: $memory_limit bytes; memory in use: $(to_gigabytes "$(memory_usage_bytes)")"
    log "free disk in $WORK_DIRECTORY: $(df -h --output=avail "$WORK_DIRECTORY" | tail -1 | tr -d ' ')"
}

begin_stage() {
    CURRENT_STAGE="$1"
    STAGE_START_SECONDS=$(date +%s)
    echo "0 0" >"$MONITOR_DIRECTORY/stage_peak"
    log "=== begin stage: $CURRENT_STAGE"
}

end_stage() {
    local stage_usage stage_anonymous job_usage job_anonymous kernel_peak=""
    sleep 1
    read -r stage_usage stage_anonymous <"$MONITOR_DIRECTORY/stage_peak" || true
    read -r job_usage job_anonymous <"$MONITOR_DIRECTORY/job_peak" || true
    if [ -r /sys/fs/cgroup/memory.peak ]; then
        kernel_peak="; cgroup memory.peak $(to_gigabytes "$(cat /sys/fs/cgroup/memory.peak)")"
    elif [ -r /sys/fs/cgroup/memory/memory.max_usage_in_bytes ]; then
        kernel_peak="; cgroup max usage $(to_gigabytes "$(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes)")"
    fi
    echo "$(($(date +%s) - STAGE_START_SECONDS))	$CURRENT_STAGE" >>"$MONITOR_DIRECTORY/stage_times"
    log "=== end stage: $CURRENT_STAGE after $(($(date +%s) - STAGE_START_SECONDS)) s;" \
        "peak in stage: anonymous $(to_gigabytes "${stage_anonymous:-0}"), with page cache $(to_gigabytes "${stage_usage:-0}");" \
        "peak in job: anonymous $(to_gigabytes "${job_anonymous:-0}"), with page cache $(to_gigabytes "${job_usage:-0}")$kernel_peak;" \
        "free disk $(df -h --output=avail "$WORK_DIRECTORY" | tail -1 | tr -d ' ')"
}

print_monitor_files() {
    local file
    for file in memory.csv step_markers.txt; do
        [ -f "$MONITOR_DIRECTORY/$file" ] || continue
        echo "BEGIN MONITOR FILE $file"
        cat "$MONITOR_DIRECTORY/$file"
        echo "END MONITOR FILE $file"
    done
}

on_exit() {
    local status=$?
    if [ -n "$SAMPLER_PID" ]; then
        kill "$SAMPLER_PID" 2>/dev/null || true
    fi
    print_monitor_files
    if [ "$status" -ne 0 ]; then
        log "FAILED in stage '$CURRENT_STAGE' with exit status $status after $(($(date +%s) - JOB_START_SECONDS)) s" >&2
    else
        log "Job finished after $(($(date +%s) - JOB_START_SECONDS)) s"
    fi
    exit "$status"
}

with_timestamps() {
    "$@" 2>&1 | while IFS= read -r line; do echo "[$(date -u +%H:%M:%S)] $line"; done
    return "${PIPESTATUS[0]}"
}

challenger_source() {
    case "$CHALLENGER" in
        http://* | https://* | /*) echo "$CHALLENGER" ;;
        *.py) echo "$REPOSITORY_DIRECTORY/$CHALLENGER" ;;
        *) echo "$REPOSITORY_DIRECTORY/challenger_datasets/$CHALLENGER.py" ;;
    esac
}

challenger_name() {
    local file_name="${CHALLENGER##*/}"
    echo "${RUN_NAME:-${file_name%.py}}"
}

default_dask_workers() {
    local cpus
    cpus=$(python3 "$JOB_HELPER" cpu-limit)
    echo $((2 * cpus > 32 ? 32 : 2 * cpus))
}

check_inputs() {
    if ! is_true "$UPLOAD"; then
        if [ -n "$OUTPUT_BUCKET" ]; then
            log "UPLOAD=$UPLOAD: OUTPUT_BUCKET is ignored, nothing is read from or written to S3"
        fi
        OUTPUT_BUCKET=""
    fi
    readonly OUTPUT_BUCKET
    [ -n "$CHALLENGER" ] || fail "CHALLENGER is not set (for example glonet, challenger_datasets/glonet.py or an https URL)"
    if [ -n "$START_DATES" ] && [ -n "$START_COUNT" ]; then
        fail "set START_DATES or START_COUNT, not both"
    fi
    case "$MODE" in
        evaluate) ;;
        shard)
            is_true "$VIEWER" || fail "MODE=shard needs VIEWER=true"
            ! is_true "$VIEWER_ONLY" || fail "MODE=shard does not take VIEWER_ONLY"
            ;;
        merge)
            [ -n "$SHARD_PREFIXES" ] || fail "MODE=merge needs SHARD_PREFIXES"
            [ -n "$OUTPUT_BUCKET" ] || fail "MODE=merge reads the shards from OUTPUT_BUCKET: it needs the upload"
            ! is_true "$VIEWER_ONLY" || fail "MODE=merge does not take VIEWER_ONLY"
            ;;
        mld-store) [ -n "${OCEANBENCH_MLD_STORE:-}" ] || fail "MODE=mld-store needs OCEANBENCH_MLD_STORE" ;;
        mld-ab) ;;
        *) fail "MODE is evaluate, shard, merge, mld-store or mld-ab, not $MODE" ;;
    esac
    if is_true "$VIEWER_ONLY" && ! is_true "$VIEWER"; then
        fail "VIEWER_ONLY=true needs VIEWER=true"
    fi
    if [ -n "$OUTPUT_BUCKET" ] && [ -z "$OUTPUT_PREFIX" ]; then
        fail "OUTPUT_PREFIX is not set: the job never writes at the root of a bucket"
    fi
    if [ -z "$OUTPUT_BUCKET" ]; then
        log "no upload: the results stay in $WORK_DIRECTORY and are lost when the job ends;" \
            "the scores are printed to the log at the end"
    elif [ -z "${AWS_S3_ENDPOINT:-}${BOTO3_ENDPOINT_URL:-}" ] || [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
        fail "OUTPUT_BUCKET is set but the S3 variables are missing: enable S3 in the job form"
    fi
}

repository_commit() {
    local git_directory="$REPOSITORY_DIRECTORY/.git" head
    head="$(cat "$git_directory/HEAD" 2>/dev/null)" || {
        echo unknown
        return
    }
    case "$head" in
        "ref: "*)
            cat "$git_directory/${head#ref: }" 2>/dev/null ||
                awk -v ref="${head#ref: }" '$2 == ref {print $1}' "$git_directory/packed-refs" 2>/dev/null ||
                echo unknown
            ;;
        *) echo "$head" ;;
    esac
}

print_inputs() {
    log "mode: $MODE${SHARD_PREFIXES:+; shards: $SHARD_PREFIXES}"
    log "challenger: $CHALLENGER ($(challenger_source)), name: $(challenger_name)"
    log "start dates: ${START_DATES:-all}; start count: ${START_COUNT:-all}; region: ${REGION:-global}"
    log "viewer: $VIEWER; viewer only: $VIEWER_ONLY; scores file: ${SCORES_FILE:-default}"
    log "output: ${OUTPUT_BUCKET:+s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/} (S3 endpoint: ${BOTO3_ENDPOINT_URL:-${AWS_S3_ENDPOINT:-unset}}); overwrite: $OVERWRITE"
    log "Copernicus Marine credentials: $([ -n "${COPERNICUSMARINE_SERVICE_USERNAME:-}" ] && echo set || echo 'not set')"
    log "DASK_NUM_WORKERS=$DASK_NUM_WORKERS DASK_SCHEDULER=$DASK_SCHEDULER OMP_NUM_THREADS=$OMP_NUM_THREADS" \
        "OCEANBENCH_EDDY_CENSUS_MAX_WORKERS=${OCEANBENCH_EDDY_CENSUS_MAX_WORKERS:-memory-aware default}" \
        "OCEANBENCH_STAGE_MAX_WORKERS=${OCEANBENCH_STAGE_MAX_WORKERS:-default}"
    log "repository: $REPOSITORY_DIRECTORY at commit $(repository_commit)"
}

install_oceanbench() {
    begin_stage "install oceanbench"
    with_timestamps python3 -m pip install --no-cache-dir --progress-bar off "$REPOSITORY_DIRECTORY[viewer]"
    if [ -n "$PY_EDDY_TRACKER_SPEC" ]; then
        with_timestamps python3 -m pip install --no-cache-dir --progress-bar off "$PY_EDDY_TRACKER_SPEC"
        # opencv-python needs libGL, which a slim image may not have: the headless build has the same cv2 API.
        if ! python3 -c "import cv2" 2>/dev/null; then
            log "cv2 does not import: replacing opencv-python with opencv-python-headless"
            with_timestamps python3 -m pip uninstall --yes opencv-python
            with_timestamps python3 -m pip install --no-cache-dir --progress-bar off opencv-python-headless
        fi
        python3 -c "import py_eddy_tracker.dataset.grid" ||
            fail "py-eddy-tracker from OCEANBENCH_PY_EDDY_TRACKER_SPEC does not import"
    fi
    if python3 -c "import py_eddy_tracker" 2>/dev/null; then
        log "py-eddy-tracker is installed: the viewer data include the eddy census"
    else
        log "py-eddy-tracker is not installed: the eddy census is skipped" \
            "(set OCEANBENCH_PY_EDDY_TRACKER_SPEC to a pip-installable URL to include it)"
    fi
    log "oceanbench $(oceanbench --version 2>&1 | awk '{print $2}') from $(cd / && python3 -c 'import oceanbench; print(oceanbench.__file__)')"
    end_stage
}

check_destination() {
    [ -n "$OUTPUT_BUCKET" ] || return 0
    begin_stage "check S3 destination"
    local overwrite_argument=()
    is_true "$OVERWRITE" && overwrite_argument=(--overwrite)
    if is_true "$VIEWER_ONLY"; then
        python3 "$JOB_HELPER" check-destination "$OUTPUT_BUCKET" "$OUTPUT_PREFIX/viewer" "${overwrite_argument[@]}"
    else
        python3 "$JOB_HELPER" check-destination "$OUTPUT_BUCKET" "$OUTPUT_PREFIX" "${overwrite_argument[@]}"
    fi
    end_stage
}

prepare_challenger() {
    begin_stage "prepare challenger"
    CHALLENGER_FILE="$CHALLENGER_DIRECTORY/$(challenger_name).py"
    python3 "$JOB_HELPER" write-challenger "$(challenger_source)" "$CHALLENGER_FILE" \
        --start-dates "$START_DATES" --start-count "$START_COUNT"
    log "challenger file: $CHALLENGER_FILE"
    sed 's/^/    /' "$CHALLENGER_FILE"
    end_stage
}

scores_file_name() {
    echo "$(challenger_name).${REGION:-global}.scores.json"
}

fetch_scores_file() {
    local source="$SCORES_FILE"
    if [ -z "$source" ]; then
        [ -n "$OUTPUT_BUCKET" ] || fail "VIEWER_ONLY=true needs SCORES_FILE or OUTPUT_BUCKET"
        source="s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$(scores_file_name)"
    fi
    log "scores file: $source"
    python3 "$JOB_HELPER" download "$source" "$RUN_DIRECTORY/$(scores_file_name)"
}

run_mld_store() {
    begin_stage "build reference mixed layer depth store"
    (cd "$RUN_DIRECTORY" && with_timestamps env PYTHONUNBUFFERED=1 python3 -c \
        'import sys; from oceanbench.core import mld_store; mld_store.build(open(sys.argv[1]).read())' "$CHALLENGER_FILE")
    end_stage
}

run_mld_ab() {
    local output_directory="$RUN_DIRECTORY/mld-ab"
    begin_stage "install py-spy"
    python3 -m pip install --quiet py-spy || log "py-spy install failed"
    end_stage
    begin_stage "mixed layer depth A/B ${MLD_AB_RUNS:-A,B,A,B}"
    (cd "$RUN_DIRECTORY" && with_timestamps env PYTHONUNBUFFERED=1 python3 "$PROCESS_DIRECTORY/mld_ab.py" \
        "$CHALLENGER_FILE" "$output_directory" "${MLD_AB_RUNS:-A,B,A,B}" "${MLD_AB_PROFILED_RUN:-3}")
    end_stage
    [ -z "$OUTPUT_BUCKET" ] || python3 "$JOB_HELPER" upload "$OUTPUT_BUCKET" "$OUTPUT_PREFIX" "$output_directory"
}

run_shard() {
    local arguments=("$CHALLENGER_FILE" --viewer-artifacts-dir "$VIEWER_DIRECTORY" --merge-inputs-dir "$MERGE_INPUTS_DIRECTORY"
        --step-markers "$MONITOR_DIRECTORY/step_markers.txt")
    if [ -n "$REGION" ]; then
        arguments+=(--region "$REGION")
    fi
    begin_stage "oceanbench evaluate shard"
    log "shards.py evaluate-shard ${arguments[*]}"
    (cd "$RUN_DIRECTORY" && with_timestamps env PYTHONUNBUFFERED=1 python3 "$SHARDS_HELPER" evaluate-shard "${arguments[@]}")
    end_stage
}

run_merge() {
    begin_stage "fetch shards"
    local shard_directories=() prefix index=0
    IFS=, read -r -a prefixes <<<"$SHARD_PREFIXES"
    for prefix in "${prefixes[@]}"; do
        prefix="${prefix#"${prefix%%[![:space:]]*}"}"
        prefix="${prefix%"${prefix##*[![:space:]]}"}"
        [ -n "$prefix" ] || continue
        index=$((index + 1))
        python3 "$JOB_HELPER" download-shard "$OUTPUT_BUCKET" "$prefix" "$SHARDS_DIRECTORY/$index"
        shard_directories+=("$SHARDS_DIRECTORY/$index")
    done
    end_stage
    local arguments=("$CHALLENGER_FILE" "${shard_directories[@]}" --output-dir "$RUN_DIRECTORY")
    if is_true "$VIEWER"; then
        arguments+=(--viewer-artifacts-dir "$VIEWER_DIRECTORY")
    fi
    if [ -n "$REGION" ]; then
        arguments+=(--region "$REGION")
    fi
    begin_stage "merge shards"
    log "shards.py merge ${arguments[*]}"
    (cd "$RUN_DIRECTORY" && with_timestamps env PYTHONUNBUFFERED=1 python3 "$SHARDS_HELPER" merge "${arguments[@]}")
    end_stage
}

run_evaluation() {
    local arguments=("$CHALLENGER_FILE" --max-workers 1)
    if [ -n "$REGION" ]; then
        arguments+=(--region "$REGION")
    fi
    if is_true "$VIEWER"; then
        arguments+=(--viewer-artifacts-dir "$VIEWER_DIRECTORY")
    fi
    if is_true "$VIEWER_ONLY"; then
        begin_stage "fetch scores file"
        fetch_scores_file
        end_stage
        arguments+=(--viewer-only --scores-file "$RUN_DIRECTORY/$(scores_file_name)")
    fi
    begin_stage "oceanbench evaluate"
    log "oceanbench evaluate ${arguments[*]}"
    (cd "$RUN_DIRECTORY" && with_timestamps env PYTHONUNBUFFERED=1 oceanbench evaluate "${arguments[@]}")
    end_stage
}

upload_results() {
    [ -n "$OUTPUT_BUCKET" ] || return 0
    begin_stage "upload results"
    local sources=()
    if ! is_true "$VIEWER_ONLY"; then
        sources+=("$RUN_DIRECTORY/$(scores_file_name)")
    fi
    if [ "$MODE" = "shard" ]; then
        sources+=("$MERGE_INPUTS_DIRECTORY" "$MONITOR_DIRECTORY/memory.csv" "$MONITOR_DIRECTORY/step_markers.txt")
    fi
    if is_true "$VIEWER"; then
        sources+=("$VIEWER_DIRECTORY")
    fi
    python3 "$JOB_HELPER" upload "$OUTPUT_BUCKET" "$OUTPUT_PREFIX" "${sources[@]}"
    log "results in s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/"
    end_stage
}

print_summary() {
    local job_usage job_anonymous kernel_peak="not exposed"
    read -r job_usage job_anonymous <"$MONITOR_DIRECTORY/job_peak" || true
    if [ -r /sys/fs/cgroup/memory.peak ]; then
        kernel_peak="$(to_gigabytes "$(cat /sys/fs/cgroup/memory.peak)")"
    elif [ -r /sys/fs/cgroup/memory/memory.max_usage_in_bytes ]; then
        kernel_peak="$(to_gigabytes "$(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes)")"
    fi
    echo "BEGIN JOB SUMMARY"
    echo "challenger: $(challenger_name); mode: $MODE; start dates: ${START_DATES:-all}; viewer: $VIEWER; commit: $(repository_commit)"
    echo "wall time per stage (s):"
    sed 's/^/    /' "$MONITOR_DIRECTORY/stage_times"
    echo "total wall time: $(($(date +%s) - JOB_START_SECONDS)) s"
    echo "peak memory: anonymous (RSS) $(to_gigabytes "${job_anonymous:-0}"), with page cache $(to_gigabytes "${job_usage:-0}"), cgroup peak $kernel_peak"
    if is_true "$VIEWER"; then
        echo "viewer data: $(du -sh "$VIEWER_DIRECTORY" | cut -f1), $(find "$VIEWER_DIRECTORY" -type f | wc -l) files"
    fi
    echo "END JOB SUMMARY"
    local scores="$RUN_DIRECTORY/$(scores_file_name)"
    if [ "$PRINT_SCORES" = "auto" ] && [ -n "$OUTPUT_BUCKET" ] || [ "$PRINT_SCORES" = "false" ]; then
        return 0
    fi
    if [ -f "$scores" ] && ! is_true "$VIEWER_ONLY"; then
        python3 "$JOB_HELPER" print-scores "$scores"
    fi
}

main() {
    mkdir -p "$MONITOR_DIRECTORY" "$RUN_DIRECTORY" "$VIEWER_DIRECTORY" "$CHALLENGER_DIRECTORY"
    echo "0 0" >"$MONITOR_DIRECTORY/job_peak"
    echo "0 0" >"$MONITOR_DIRECTORY/stage_peak"
    trap on_exit EXIT
    memory_sampler &
    SAMPLER_PID=$!

    if [ -n "${AWS_SESSION_TOKEN+x}" ] && [ -z "$AWS_SESSION_TOKEN" ]; then
        unset AWS_SESSION_TOKEN
    fi
    export DASK_NUM_WORKERS="${DASK_NUM_WORKERS:-$(default_dask_workers)}"
    export DASK_SCHEDULER="${DASK_SCHEDULER:-threads}"
    local cpus
    cpus=$(python3 "$JOB_HELPER" cpu-limit)
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$cpus}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-$cpus}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$cpus}" NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-$cpus}"
    export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$WORK_DIRECTORY/numba-cache}"
    unset OCEANBENCH_OUTPUT_BUCKET OCEANBENCH_OUTPUT_PREFIX OCEANBENCH_VIEWER_ARTIFACTS_DIR

    log "OceanBench EDITO process job, python $(python3 -c 'import platform; print(platform.python_version())')"
    print_resources
    check_inputs
    print_inputs
    install_oceanbench
    check_destination
    prepare_challenger
    case "$MODE" in
        shard) run_shard ;;
        merge) run_merge ;;
        mld-store) run_mld_store ;;
        mld-ab) run_mld_ab ;;
        *) run_evaluation ;;
    esac
    upload_results
    print_resources
    print_summary
}

main "$@"
