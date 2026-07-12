#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
START_BENCH_SH="$ROOT_DIR/scripts/issue003/start_bench.sh"

RAY_ADDR="${RAY_ADDR:-10.102.97.33:7799}"
MASTER_ADDR="${MASTER_ADDR:-10.102.97.33:29500}"

MODEL_PATH="${MODEL_PATH:-/mnt/nvme1n1/ml_research/models/models--moonshotai--Kimi-K2-Instruct-0905/snapshots/7152993552508c9f22042b3bb93b5e6acd06ce73}"
DATASET_PATH="${DATASET_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/kimi/processed_conversation.csv}"

SEG="${SEG:-65536}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SCHEDULER="${SCHEDULER:-centralized}"
GPU_MEM="${GPU_MEM:-141}"
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-1000000}"
LOOP_COUNT="${LOOP_COUNT:-16}"
FIXED_SP_SEGMENTS="${FIXED_SP_SEGMENTS:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER="${USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER:-0}"

DP2SP8_ENABLE_DYNAMIC_SP_SIZE="${DP2SP8_ENABLE_DYNAMIC_SP_SIZE:-1}"
DP2SP8_DYNAMIC_SP_SIZE_STRATEGY="${DP2SP8_DYNAMIC_SP_SIZE_STRATEGY:-long_short_sp8}"
DP2SP8_LONG_REQUEST_SP_THRESHOLD="${DP2SP8_LONG_REQUEST_SP_THRESHOLD:-100000}"
DP2SP8_LONG_REQUEST_SP_THRESHOLDS="${DP2SP8_LONG_REQUEST_SP_THRESHOLDS:-100000 65536}"

DP16_ENABLE_DYNAMIC_SP_SIZE="${DP16_ENABLE_DYNAMIC_SP_SIZE:-0}"
DP16_DYNAMIC_SP_SIZE_STRATEGY="${DP16_DYNAMIC_SP_SIZE_STRATEGY:-legacy}"
DP16_LONG_REQUEST_SP_THRESHOLD="${DP16_LONG_REQUEST_SP_THRESHOLD:-100000}"

STEP10_START_RATE="${STEP10_START_RATE:-10}"
STEP10_STEP_RATE="${STEP10_STEP_RATE:-10}"
STEP10_MAX_RATE="${STEP10_MAX_RATE:-100}"
STEP5_START_RATE="${STEP5_START_RATE:-10}"
STEP5_STEP_RATE="${STEP5_STEP_RATE:-5}"
STEP5_MAX_RATE="${STEP5_MAX_RATE:-100}"

STOP_THRESHOLD_MS="${STOP_THRESHOLD_MS:-100}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-20}"
MAX_RETRIES="${MAX_RETRIES:-5}"

RUN_TAG="${RUN_TAG:-kimi_conversation_2node_16gpu_$(date -u +%Y%m%d_%H%M%S)}"
CHAIN_LOG_DIR="${CHAIN_LOG_DIR:-$ROOT_DIR/bench_logs/$RUN_TAG}"
CHAIN_PROGRESS="$CHAIN_LOG_DIR/chain.progress"
CHAIN_SUMMARY="$CHAIN_LOG_DIR/chain_summary.tsv"
LOG_SEARCH_ROOT="${LOG_SEARCH_ROOT:-$(dirname "$CHAIN_LOG_DIR")}"
RESUME_SKIP_SUCCESS="${RESUME_SKIP_SUCCESS:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

mkdir -p "$CHAIN_LOG_DIR"

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$CHAIN_PROGRESS"
}

ensure_headers() {
    if [[ ! -f "$CHAIN_SUMMARY" ]]; then
        printf "stage\trate\tn_reqs\tstatus\titl_avg_ms\titl_p99_ms\tqueue_avg_ms\tqueue_p99_ms\tdecode_queue_avg_ms\tdecode_queue_p99_ms\tlog_file\tjson_file\n" > "$CHAIN_SUMMARY"
    fi
}

record_summary() {
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" >> "$CHAIN_SUMMARY"
}

rate_to_nreqs() {
    awk "BEGIN {printf \"%d\", int(600 * $1 + 0.5)}"
}

rate_add() {
    awk "BEGIN {printf \"%.10g\", $1 + $2}"
}

rate_gt() {
    awk "BEGIN {exit !($1 > $2)}"
}

extract_metric_from_section() {
    local file="$1"
    local section_regex="$2"
    local metric_label="$3"
    awk -v section_regex="$section_regex" -v metric_label="$metric_label" '
        $0 ~ section_regex {in_section=1; next}
        in_section && /^--- / {exit}
        in_section && $1 == metric_label {
            value=$2
            gsub(/[^0-9eE+.-]/, "", value)
            print value
            exit
        }
    ' "$file"
}

parse_metric() {
    local file="$1"
    local key="$2"
    case "$key" in
        itl_avg) extract_metric_from_section "$file" "^--- ITL With Decode Queue" "Avg:" ;;
        itl_p99) extract_metric_from_section "$file" "^--- ITL With Decode Queue" "P99:" ;;
        queue_avg) extract_metric_from_section "$file" "^--- Queueing Time \\(ms\\) ---" "Avg:" ;;
        queue_p99) extract_metric_from_section "$file" "^--- Queueing Time \\(ms\\) ---" "P99:" ;;
        decode_queue_avg) extract_metric_from_section "$file" "^--- Decode Queue Time \\(ms\\) ---" "Avg:" ;;
        decode_queue_p99) extract_metric_from_section "$file" "^--- Decode Queue Time \\(ms\\) ---" "P99:" ;;
        *) return 1 ;;
    esac
}

is_number() {
    [[ "$1" =~ ^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]
}

clear_resume_result() {
    RESUME_ITL_AVG=""
    RESUME_ITL_P99=""
    RESUME_QUEUE_AVG=""
    RESUME_QUEUE_P99=""
    RESUME_DECODE_QUEUE_AVG=""
    RESUME_DECODE_QUEUE_P99=""
    RESUME_LOG_FILE=""
    RESUME_JSON_FILE=""
    RESUME_HIT_SOURCE=""
}

find_latest_artifact_dir() {
    local base_log_dir="$1"
    local dp="$2"
    local sp="$3"
    local n_reqs="$4"
    local rate="$5"
    local dynamic_sp_size_strategy="$6"
    local long_request_sp_threshold="$7"
    local seg_short=$((SEG / 1024))k
    local dataset_name
    local model_name
    local maxin_pattern=""
    local pattern=""
    local search_dir=""
    dataset_name="$(basename "$DATASET_PATH" .csv)"
    model_name="$(basename "$MODEL_PATH")"
    search_dir="$base_log_dir/$model_name/$dataset_name"
    [[ -d "$search_dir" ]] || return 1
    if [[ -n "${MAX_INPUT_LEN:-}" ]]; then
        maxin_pattern="_maxin$((MAX_INPUT_LEN / 1000))k"
    fi
    pattern="dp${dp}sp${sp}_seg${seg_short}_n${n_reqs}_r${rate}_bs${BATCH_SIZE}_LB_cen${maxin_pattern}*"
    if [[ "$dynamic_sp_size_strategy" != "legacy" ]]; then
        pattern="${pattern}_${dynamic_sp_size_strategy}_thr${long_request_sp_threshold}"
    fi
    find "$search_dir" -maxdepth 1 -type d \
        -name "$pattern" \
        | sort | tail -n 1
}

find_latest_log() {
    local dir="$1"
    [[ -n "${dir:-}" && -d "$dir" ]] || return 1
    find "$dir" -maxdepth 1 -type f -name '*.log' | sort | tail -n 1
}

find_latest_json() {
    local dir="$1"
    [[ -n "${dir:-}" && -d "$dir" ]] || return 1
    find "$dir" -maxdepth 1 -type f -name '*.json' | sort | tail -n 1
}

find_candidate_artifact_dirs() {
    local stage_log_dir="$1"
    local dp="$2"
    local sp="$3"
    local n_reqs="$4"
    local rate="$5"
    local dynamic_sp_size_strategy="$6"
    local long_request_sp_threshold="$7"
    local stage_log_name

    stage_log_name="$(basename "$stage_log_dir")"

    find "$LOG_SEARCH_ROOT" -type d -path "*/$stage_log_name" | while read -r candidate_stage_log_dir; do
        find_latest_artifact_dir \
            "$candidate_stage_log_dir" \
            "$dp" \
            "$sp" \
            "$n_reqs" \
            "$rate" \
            "$dynamic_sp_size_strategy" \
            "$long_request_sp_threshold" || true
    done | awk 'NF' | sort -r
}

load_existing_ok_result() {
    local stage_log_dir="$1"
    local dp="$2"
    local sp="$3"
    local n_reqs="$4"
    local rate="$5"
    local dynamic_sp_size_strategy="$6"
    local long_request_sp_threshold="$7"
    local artifact_dir=""
    local log_file=""
    local json_file=""
    local itl_avg=""
    local itl_p99=""
    local queue_avg=""
    local queue_p99=""
    local decode_queue_avg=""
    local decode_queue_p99=""

    clear_resume_result
    if (( FORCE_RERUN != 0 || RESUME_SKIP_SUCCESS == 0 )); then
        return 1
    fi

    while read -r artifact_dir; do
        [[ -n "${artifact_dir:-}" && -d "$artifact_dir" ]] || continue

        log_file="$(find_latest_log "$artifact_dir" || true)"
        [[ -n "${log_file:-}" && -f "$log_file" ]] || continue

        json_file="$(find_latest_json "$artifact_dir" || true)"
        itl_avg="$(parse_metric "$log_file" itl_avg || true)"
        [[ -n "${itl_avg:-}" ]] && is_number "$itl_avg" || continue

        itl_p99="$(parse_metric "$log_file" itl_p99 || true)"
        queue_avg="$(parse_metric "$log_file" queue_avg || true)"
        queue_p99="$(parse_metric "$log_file" queue_p99 || true)"
        decode_queue_avg="$(parse_metric "$log_file" decode_queue_avg || true)"
        decode_queue_p99="$(parse_metric "$log_file" decode_queue_p99 || true)"

        RESUME_ITL_AVG="$itl_avg"
        RESUME_ITL_P99="$itl_p99"
        RESUME_QUEUE_AVG="$queue_avg"
        RESUME_QUEUE_P99="$queue_p99"
        RESUME_DECODE_QUEUE_AVG="$decode_queue_avg"
        RESUME_DECODE_QUEUE_P99="$decode_queue_p99"
        RESUME_LOG_FILE="$log_file"
        RESUME_JSON_FILE="$json_file"
        RESUME_HIT_SOURCE="artifact"
        return 0
    done < <(
        find_candidate_artifact_dirs \
            "$stage_log_dir" \
            "$dp" \
            "$sp" \
            "$n_reqs" \
            "$rate" \
            "$dynamic_sp_size_strategy" \
            "$long_request_sp_threshold"
    )

    clear_resume_result
    return 1
}

run_rate() {
    local stage_name="$1"
    local stage_log_dir="$2"
    local dp="$3"
    local sp="$4"
    local enable_dynamic_sp_size="$5"
    local dynamic_sp_size_strategy="$6"
    local long_request_sp_threshold="$7"
    local rate="$8"

    local n_reqs
    local attempt=1
    local status=""
    local artifact_dir=""
    local log_file=""
    local json_file=""
    local itl_avg=""
    local itl_p99=""
    local queue_avg=""
    local queue_p99=""
    local decode_queue_avg=""
    local decode_queue_p99=""

    n_reqs="$(rate_to_nreqs "$rate")"

    if load_existing_ok_result "$stage_log_dir" "$dp" "$sp" "$n_reqs" "$rate" "$dynamic_sp_size_strategy" "$long_request_sp_threshold"; then
        status="ok"
        itl_avg="$RESUME_ITL_AVG"
        itl_p99="$RESUME_ITL_P99"
        queue_avg="$RESUME_QUEUE_AVG"
        queue_p99="$RESUME_QUEUE_P99"
        decode_queue_avg="$RESUME_DECODE_QUEUE_AVG"
        decode_queue_p99="$RESUME_DECODE_QUEUE_P99"
        log_file="$RESUME_LOG_FILE"
        json_file="$RESUME_JSON_FILE"
        log "SKIP stage=$stage_name rate=$rate reason=existing_ok_${RESUME_HIT_SOURCE} itl_avg=${itl_avg}ms log_file=$log_file"
    else
        while (( attempt <= MAX_RETRIES )); do
            local cmd=(
                bash "$START_BENCH_SH"
                --master-addr "$MASTER_ADDR"
                --ray-addr "$RAY_ADDR"
                --dataset-path "$DATASET_PATH"
                --model-path "$MODEL_PATH"
                --segment-size "$SEG"
                --batch-size "$BATCH_SIZE"
                --num-requests "$n_reqs"
                --gpu-mem "$GPU_MEM"
                --gpu-util "$GPU_UTIL"
                --max-model-len "$MAX_MODEL_LEN"
                --routing-strategy LeastBatch
                --scheduler-mode "$SCHEDULER"
                --loop-count "$LOOP_COUNT"
                --fixed-sp-segments "$FIXED_SP_SEGMENTS"
                --max-input-len "$MAX_INPUT_LEN"
                --dynamic-sp-size-strategy "$dynamic_sp_size_strategy"
                --long-request-sp-threshold "$long_request_sp_threshold"
                --dp-size "$dp"
                --sp-size "$sp"
            )

            if (( ENFORCE_EAGER != 0 )); then
                cmd+=(--enforce-eager)
            fi
            if (( USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER != 0 )); then
                cmd+=(--use-new-decode-dynamic-sp-scheduler)
            fi
            if (( enable_dynamic_sp_size != 0 )); then
                cmd+=(--enable-dynamic-sp-size)
            fi
            cmd+=("$rate")

            log "RUN stage=$stage_name rate=$rate n_reqs=$n_reqs attempt=$attempt/$MAX_RETRIES"

            set +e
            env BASE_LOG_DIR="$stage_log_dir" "${cmd[@]}" >> "$stage_log_dir/sweep.out" 2>&1
            local rc=$?
            set -e

            artifact_dir="$(find_latest_artifact_dir "$stage_log_dir" "$dp" "$sp" "$n_reqs" "$rate" "$dynamic_sp_size_strategy" "$long_request_sp_threshold" || true)"
            log_file="$(find_latest_log "$artifact_dir" || true)"
            json_file="$(find_latest_json "$artifact_dir" || true)"

            if [[ $rc -eq 0 && -n "${log_file:-}" && -f "$log_file" ]]; then
                itl_avg="$(parse_metric "$log_file" itl_avg || true)"
                itl_p99="$(parse_metric "$log_file" itl_p99 || true)"
                queue_avg="$(parse_metric "$log_file" queue_avg || true)"
                queue_p99="$(parse_metric "$log_file" queue_p99 || true)"
                decode_queue_avg="$(parse_metric "$log_file" decode_queue_avg || true)"
                decode_queue_p99="$(parse_metric "$log_file" decode_queue_p99 || true)"
                if [[ -n "${itl_avg:-}" ]] && is_number "$itl_avg"; then
                    status="ok"
                    break
                fi
                status="parse_failed"
            else
                status="run_failed"
            fi

            log "WARN stage=$stage_name rate=$rate attempt=$attempt status=$status"
            attempt=$((attempt + 1))
            sleep 10
        done
    fi

    if [[ "$status" != "ok" ]]; then
        record_summary "$stage_name" "$rate" "$n_reqs" "$status" "" "" "" "" "" "" "${log_file:-}" "${json_file:-}"
        log "ERROR stage=$stage_name rate=$rate final_status=$status"
        return 1
    fi

    record_summary "$stage_name" "$rate" "$n_reqs" "ok" "$itl_avg" "${itl_p99:-}" "${queue_avg:-}" "${queue_p99:-}" "${decode_queue_avg:-}" "${decode_queue_p99:-}" "$log_file" "${json_file:-}"
    log "RESULT stage=$stage_name rate=$rate itl_avg=${itl_avg}ms itl_p99=${itl_p99}ms queue_avg=${queue_avg}ms queue_p99=${queue_p99}ms decode_queue_avg=${decode_queue_avg}ms decode_queue_p99=${decode_queue_p99}ms"

    if awk -v itl_avg="$itl_avg" -v stop_ms="$STOP_THRESHOLD_MS" "BEGIN {exit !(itl_avg > stop_ms)}"; then
        log "STOP stage=$stage_name because itl_avg=${itl_avg}ms > ${STOP_THRESHOLD_MS}ms"
        return 2
    fi

    return 0
}

run_stage_at_rate() {
    local stage_name="$1"
    local stage_log_dir="$2"
    local dp="$3"
    local sp="$4"
    local enable_dynamic_sp_size="$5"
    local dynamic_sp_size_strategy="$6"
    local long_request_sp_threshold="$7"
    local rate="$8"

    mkdir -p "$stage_log_dir"

    local rc
    set +e
    run_rate "$stage_name" "$stage_log_dir" "$dp" "$sp" "$enable_dynamic_sp_size" "$dynamic_sp_size_strategy" "$long_request_sp_threshold" "$rate"
    rc=$?
    set -e

    return "$rc"
}

run_interleaved_rate_sweep() {
    local start_rate="$1"
    local step_rate="$2"
    local max_rate="$3"

    local rate="$start_rate"
    local dp16_active=1
    declare -A dp2sp8_active=()

    local threshold
    for threshold in $DP2SP8_LONG_REQUEST_SP_THRESHOLDS; do
        dp2sp8_active["$threshold"]=1
    done

    log "START interleaved_rate_sweep start_rate=$start_rate step_rate=$step_rate max_rate=$max_rate"

    while ! rate_gt "$rate" "$max_rate"; do
        log "START rate=$rate"

        for threshold in $DP2SP8_LONG_REQUEST_SP_THRESHOLDS; do
            local safe_threshold="${threshold//./p}"
            local stage_name="dp2sp8_thr${safe_threshold}"
            local stage_log_dir="$CHAIN_LOG_DIR/dp2sp8_step10"
            local rc

            if [[ "${dp2sp8_active["$threshold"]:-0}" -eq 0 ]]; then
                log "SKIP stage=$stage_name rate=$rate reason=stopped"
                continue
            fi

            if run_stage_at_rate \
                "$stage_name" \
                "$stage_log_dir" \
                2 \
                8 \
                "$DP2SP8_ENABLE_DYNAMIC_SP_SIZE" \
                "$DP2SP8_DYNAMIC_SP_SIZE_STRATEGY" \
                "$threshold" \
                "$rate"; then
                rc=0
            else
                rc=$?
            fi

            if (( rc == 2 )); then
                dp2sp8_active["$threshold"]=0
            elif (( rc != 0 )); then
                return "$rc"
            fi
        done

        if (( dp16_active != 0 )); then
            local rc
            if run_stage_at_rate \
                "dp16" \
                "$CHAIN_LOG_DIR/dp16sp1_step10" \
                16 \
                1 \
                "$DP16_ENABLE_DYNAMIC_SP_SIZE" \
                "$DP16_DYNAMIC_SP_SIZE_STRATEGY" \
                "$DP16_LONG_REQUEST_SP_THRESHOLD" \
                "$rate"; then
                rc=0
            else
                rc=$?
            fi

            if (( rc == 2 )); then
                dp16_active=0
            elif (( rc != 0 )); then
                return "$rc"
            fi
        else
            log "SKIP stage=dp16 rate=$rate reason=stopped"
        fi

        local any_stage_active="$dp16_active"
        for threshold in $DP2SP8_LONG_REQUEST_SP_THRESHOLDS; do
            if [[ "${dp2sp8_active["$threshold"]:-0}" -ne 0 ]]; then
                any_stage_active=1
                break
            fi
        done
        if (( any_stage_active == 0 )); then
            log "STOP all stages exhausted at rate=$rate"
            break
        fi

        rate="$(rate_add "$rate" "$step_rate")"
        if ! rate_gt "$rate" "$max_rate"; then
            sleep "$SLEEP_BETWEEN_RUNS"
        fi
    done

    log "DONE interleaved_rate_sweep"
}

main() {
    ensure_headers
    log "RUN_TAG=$RUN_TAG"
    log "CHAIN_LOG_DIR=$CHAIN_LOG_DIR"
    log "LOG_SEARCH_ROOT=$LOG_SEARCH_ROOT RESUME_SKIP_SUCCESS=$RESUME_SKIP_SUCCESS FORCE_RERUN=$FORCE_RERUN"
    log "RAY_ADDR=$RAY_ADDR MASTER_ADDR=$MASTER_ADDR"
    log "MODEL=$MODEL_PATH"
    log "DATASET=$DATASET_PATH"
    log "BATCH_SIZE=$BATCH_SIZE GPU_UTIL=$GPU_UTIL GPU_MEM=$GPU_MEM LOOP_COUNT=$LOOP_COUNT ENFORCE_EAGER=$ENFORCE_EAGER"
    log "DP2SP8=dynamic_sp:$DP2SP8_ENABLE_DYNAMIC_SP_SIZE strategy:$DP2SP8_DYNAMIC_SP_SIZE_STRATEGY thresholds:${DP2SP8_LONG_REQUEST_SP_THRESHOLDS}"
    log "DP16=dynamic_sp:$DP16_ENABLE_DYNAMIC_SP_SIZE strategy:$DP16_DYNAMIC_SP_SIZE_STRATEGY threshold:$DP16_LONG_REQUEST_SP_THRESHOLD"
    log "RATES=start:$STEP10_START_RATE step:$STEP10_STEP_RATE max:$STEP10_MAX_RATE | STOP_THRESHOLD_MS=$STOP_THRESHOLD_MS"
    log "ORDER=for rate -> dp2sp8_thr100000 -> dp2sp8_thr65536 -> dp16"

    run_interleaved_rate_sweep "$STEP10_START_RATE" "$STEP10_STEP_RATE" "$STEP10_MAX_RATE"

    log "ALL_DONE"
}

main "$@"
