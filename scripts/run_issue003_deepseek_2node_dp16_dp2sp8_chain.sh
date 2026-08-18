#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
START_BENCH_SH="$ROOT_DIR/scripts/issue003/start_bench.sh"

RAY_ADDR="${RAY_ADDR:-10.102.97.179:7799}"
MASTER_ADDR="${MASTER_ADDR:-10.102.97.179:29500}"

DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-/mnt/nvme1n1/ml_research/models/deepseek-v3}"
ISSUE003_DATASET="${ISSUE003_DATASET:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.03_n60000.csv}"

SEG="${SEG:-65536}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SCHEDULER_ARCH="${SCHEDULER_ARCH:-legacy_global}"
GPU_MEM="${GPU_MEM:-141}"
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-1000000}"
LOOP_COUNT="${LOOP_COUNT:-16}"
FIXED_SP_SEGMENTS="${FIXED_SP_SEGMENTS:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER="${USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER:-0}"

START_RATE="${START_RATE:-10}"
STEP_RATE="${STEP_RATE:-10}"
MAX_RATE="${MAX_RATE:-500}"
STOP_THRESHOLD_MS="${STOP_THRESHOLD_MS:-100}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-20}"
MAX_RETRIES="${MAX_RETRIES:-5}"

CP_SIZE_THRESHOLD="${CP_SIZE_THRESHOLD:-100000}"

RUN_TAG="${RUN_TAG:-issue003_deepseek_v3_2node_dp16_dp2sp8_$(date -u +%Y%m%d_%H%M%S)}"
CHAIN_LOG_DIR="${CHAIN_LOG_DIR:-$ROOT_DIR/bench_logs/$RUN_TAG}"
CHAIN_PROGRESS="$CHAIN_LOG_DIR/chain.progress"
CHAIN_SUMMARY="$CHAIN_LOG_DIR/chain_summary.tsv"

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

find_latest_artifact_dir() {
    local base_log_dir="$1"
    local strat_prefix="$2"
    local n_reqs="$3"
    local rate="$4"
    local seg_short=$((SEG / 1024))k
    local dataset_name
    local model_name
    dataset_name="$(basename "$ISSUE003_DATASET" .csv)"
    model_name="$(basename "$DEEPSEEK_MODEL")"
    find "$base_log_dir/$model_name/$dataset_name" -maxdepth 1 -type d \
        -name "${strat_prefix}_seg${seg_short}_n${n_reqs}_r${rate}_bs${BATCH_SIZE}_LB_cen*" \
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
    find "$dir" -maxdepth 1 -type f \
        \( -name '*.jsonl' -o -name '*.json' \) \
        ! -name '*.summary.json' ! -name '*.hier_trace.jsonl' \
        | sort | tail -n 1
}

run_stage_sweep() {
    local stage_name="$1"
    local stage_log_dir="$2"
    local dp="$3"
    local sp="$4"
    local enable_dynamic_sp_size="$5"
    local dynamic_sp_size_strategy="$6"
    local long_request_sp_threshold="$7"

    local rate="$START_RATE"
    local strat_prefix="dp${dp}sp${sp}"

    mkdir -p "$stage_log_dir"
    log "START stage=$stage_name dp=$dp sp=$sp dynamic_sp=$enable_dynamic_sp_size strategy=$dynamic_sp_size_strategy threshold=$long_request_sp_threshold start_rate=$START_RATE step_rate=$STEP_RATE"

    while ! rate_gt "$rate" "$MAX_RATE"; do
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

        while (( attempt <= MAX_RETRIES )); do
            log "RUN stage=$stage_name rate=$rate n_reqs=$n_reqs attempt=$attempt/$MAX_RETRIES"

            set +e
            BASE_LOG_DIR="$stage_log_dir" bash "$START_BENCH_SH" \
                --master-addr "$MASTER_ADDR" \
                --ray-addr "$RAY_ADDR" \
                --dataset-path "$ISSUE003_DATASET" \
                --model-path "$DEEPSEEK_MODEL" \
                --segment-size "$SEG" \
                --batch-size "$BATCH_SIZE" \
                --num-requests "$n_reqs" \
                --gpu-mem "$GPU_MEM" \
                --gpu-util "$GPU_UTIL" \
                --max-model-len "$MAX_MODEL_LEN" \
                --routing-strategy LeastBatch \
                --scheduler-arch "$SCHEDULER_ARCH" \
                --loop-count "$LOOP_COUNT" \
                --fixed-sp-segments "$FIXED_SP_SEGMENTS" \
                --max-input-len "$MAX_INPUT_LEN" \
                --dynamic-sp-size-strategy "$dynamic_sp_size_strategy" \
                --long-request-sp-threshold "$long_request_sp_threshold" \
                --dp-size "$dp" \
                --sp-size "$sp" \
                $( ((ENFORCE_EAGER != 0)) && echo --enforce-eager ) \
                $( ((USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER != 0)) && echo --use-new-decode-dynamic-sp-scheduler ) \
                $( ((enable_dynamic_sp_size != 0)) && echo --enable-dynamic-sp-size ) \
                "$rate" >> "$stage_log_dir/sweep.out" 2>&1
            local rc=$?
            set -e

            artifact_dir="$(find_latest_artifact_dir "$stage_log_dir" "$strat_prefix" "$n_reqs" "$rate" || true)"
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

        if [[ "$status" != "ok" ]]; then
            record_summary "$stage_name" "$rate" "$n_reqs" "$status" "" "" "" "" "" "" "${log_file:-}" "${json_file:-}"
            log "ERROR stage=$stage_name rate=$rate final_status=$status"
            return 1
        fi

        record_summary "$stage_name" "$rate" "$n_reqs" "ok" "$itl_avg" "${itl_p99:-}" "${queue_avg:-}" "${queue_p99:-}" "${decode_queue_avg:-}" "${decode_queue_p99:-}" "$log_file" "${json_file:-}"
        log "RESULT stage=$stage_name rate=$rate itl_avg=${itl_avg}ms itl_p99=${itl_p99}ms queue_avg=${queue_avg}ms queue_p99=${queue_p99}ms decode_queue_avg=${decode_queue_avg}ms decode_queue_p99=${decode_queue_p99}ms"

        if awk -v itl_avg="$itl_avg" -v stop_ms="$STOP_THRESHOLD_MS" "BEGIN {exit !(itl_avg > stop_ms)}"; then
            log "STOP stage=$stage_name because itl_avg=${itl_avg}ms > ${STOP_THRESHOLD_MS}ms"
            break
        fi

        rate="$(rate_add "$rate" "$STEP_RATE")"
        sleep "$SLEEP_BETWEEN_RUNS"
    done

    log "DONE stage=$stage_name"
}

main() {
    ensure_headers
    log "RUN_TAG=$RUN_TAG"
    log "CHAIN_LOG_DIR=$CHAIN_LOG_DIR"
    log "RAY_ADDR=$RAY_ADDR MASTER_ADDR=$MASTER_ADDR"
    log "MODEL=$DEEPSEEK_MODEL"
    log "DATASET=$ISSUE003_DATASET"
    log "START_RATE=$START_RATE STEP_RATE=$STEP_RATE MAX_RATE=$MAX_RATE STOP_THRESHOLD_MS=$STOP_THRESHOLD_MS"
    log "ORDER=dp16sp1 -> dp2sp8_legacy_segment_cp -> dp2sp8_threshold_cp"

    run_stage_sweep "dp16sp1" "$CHAIN_LOG_DIR/dp16_issue003_deepseek_v3" 16 1 0 "legacy" "$CP_SIZE_THRESHOLD"
    run_stage_sweep "dp2sp8_legacy_segment_cp" "$CHAIN_LOG_DIR/dp2sp8_issue003_deepseek_v3_legacy_segment_cp" 2 8 1 "legacy" "$CP_SIZE_THRESHOLD"
    run_stage_sweep "dp2sp8_threshold_cp" "$CHAIN_LOG_DIR/dp2sp8_issue003_deepseek_v3_threshold_cp" 2 8 1 "long_short_sp8" "$CP_SIZE_THRESHOLD"

    log "ALL_DONE"
}

main "$@"
