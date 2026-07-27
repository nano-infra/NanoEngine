#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
START_BENCH_SH="$SCRIPT_DIR/start_bench.sh"

RAY_ADDR="${RAY_ADDR:-10.102.97.179:7799}"
MASTER_ADDR="${MASTER_ADDR:-10.102.97.179:29500}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme1n1/ml_research/models/deepseek-v3}"
DATASET_PATH="${DATASET_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.03_n60000.csv}"

SEG="${SEG:-65536}"
SCHEDULER="${SCHEDULER:-centralized}"
BATCH_SIZE="${BATCH_SIZE:-192}"
GPU_MEM="${GPU_MEM:-141}"
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-1000000}"
LOOP_COUNT="${LOOP_COUNT:-16}"
FIXED_SP_SEGMENTS="${FIXED_SP_SEGMENTS:-0}"
ENABLE_DYNAMIC_SP_SIZE="${ENABLE_DYNAMIC_SP_SIZE:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER="${USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER:-0}"
DYNAMIC_SP_SIZE_STRATEGY="${DYNAMIC_SP_SIZE_STRATEGY:-legacy}"
LONG_REQUEST_SP_THRESHOLD="${LONG_REQUEST_SP_THRESHOLD:-100000}"
SWEEP_STRATEGY="${SWEEP_STRATEGY:-dp4sp8}"

START_RATE="${START_RATE:-10}"
STEP_RATE="${STEP_RATE:-10}"
MAX_RATE="${MAX_RATE:-500}"
STOP_THRESHOLD_MS="${STOP_THRESHOLD_MS:-100}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-20}"
MAX_RETRIES="${MAX_RETRIES:-5}"

RUN_TAG="${RUN_TAG:-issue003_dp4sp8_lb_$(date -u +%Y%m%d_%H%M%S)}"
BASE_LOG_DIR="${BASE_LOG_DIR:-$ROOT_DIR/bench_logs/$RUN_TAG}"
PROGRESS_FILE="$BASE_LOG_DIR/sweep.progress"
SUMMARY_FILE="$BASE_LOG_DIR/sweep_summary.tsv"
STATE_FILE="$BASE_LOG_DIR/sweep_state.tsv"
OUT_LOG="$BASE_LOG_DIR/sweep.out"

mkdir -p "$BASE_LOG_DIR"

log() {
    local now
    now=$(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')
    echo "[$now] $*" | tee -a "$PROGRESS_FILE"
}

ensure_headers() {
    if [[ ! -f "$SUMMARY_FILE" ]]; then
        printf "strategy\trouting\trate\tn_reqs\tstatus\titl_avg_ms\titl_p99_ms\tqueue_avg_ms\tqueue_p99_ms\tdecode_queue_avg_ms\tdecode_queue_p99_ms\tlog_file\tjson_file\n" > "$SUMMARY_FILE"
    fi
    if [[ ! -f "$STATE_FILE" ]]; then
        printf "strategy\trouting\tmax_ok_rate\tfirst_over_rate\tstop_reason\n" > "$STATE_FILE"
    fi
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

routing_short() {
    case "$1" in
        LeastBatch) echo "LB" ;;
        LeastCache) echo "LC" ;;
        RoundRobin) echo "RR" ;;
        VLLMLoadBalance) echo "VLLM" ;;
        *) echo "$1" ;;
    esac
}

strategy_prefix() {
    case "$1" in
        dp4sp8) echo "dp4sp8" ;;
        dp32sp1) echo "dp32sp1" ;;
        *) return 1 ;;
    esac
}

strategy_args() {
    case "$1" in
        dp4sp8) echo "--dp-size 4 --sp-size 8" ;;
        dp32sp1) echo "--dp-size 32 --sp-size 1" ;;
        *) return 1 ;;
    esac
}

find_latest_artifact_dir() {
    local strat_prefix="$1"
    local n_reqs="$2"
    local rate="$3"
    local routing_tag="$4"
    local seg_short=$((SEG / 1024))k
    local dataset_name
    local model_name
    dataset_name="$(basename "$DATASET_PATH" .csv)"
    model_name="$(basename "$MODEL_PATH")"
    find "$BASE_LOG_DIR/$model_name/$dataset_name" -maxdepth 1 -type d \
        -name "${strat_prefix}_seg${seg_short}_n${n_reqs}_r${rate}_bs${BATCH_SIZE}_${routing_tag}_cen*" \
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

record_summary() {
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "$10" "${11}" "${12}" "${13}" >> "$SUMMARY_FILE"
}

record_state() {
    printf "%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" >> "$STATE_FILE"
}

run_one_attempt() {
    local strategy="$1"
    local routing="$2"
    local rate="$3"
    local n_reqs="$4"
    local strat_args
    local eager_args=()
    local dynamic_sp_args=()
    local scheduler_args=()
    local sp_strategy_args=()

    strat_args="$(strategy_args "$strategy")"
    if [[ "$ENFORCE_EAGER" -ne 0 ]]; then
        eager_args+=(--enforce-eager)
    fi
    if [[ "$ENABLE_DYNAMIC_SP_SIZE" -ne 0 ]]; then
        dynamic_sp_args+=(--enable-dynamic-sp-size)
    fi
    if [[ "$USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER" -ne 0 ]]; then
        scheduler_args+=(--use-new-decode-dynamic-sp-scheduler)
    fi
    if [[ -n "${DYNAMIC_SP_SIZE_STRATEGY:-}" ]]; then
        sp_strategy_args+=(--dynamic-sp-size-strategy "$DYNAMIC_SP_SIZE_STRATEGY")
        sp_strategy_args+=(--long-request-sp-threshold "$LONG_REQUEST_SP_THRESHOLD")
    fi
    # shellcheck disable=SC2086
    BASE_LOG_DIR="$BASE_LOG_DIR" bash "$START_BENCH_SH" \
        --master-addr "$MASTER_ADDR" \
        --ray-addr "$RAY_ADDR" \
        --dataset-path "$DATASET_PATH" \
        --model-path "$MODEL_PATH" \
        --segment-size "$SEG" \
        --batch-size "$BATCH_SIZE" \
        --num-requests "$n_reqs" \
        --gpu-mem "$GPU_MEM" \
        --gpu-util "$GPU_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --routing-strategy "$routing" \
        --scheduler-mode "$SCHEDULER" \
        --loop-count "$LOOP_COUNT" \
        --fixed-sp-segments "$FIXED_SP_SEGMENTS" \
        "${dynamic_sp_args[@]}" \
        "${scheduler_args[@]}" \
        "${sp_strategy_args[@]}" \
        "${eager_args[@]}" \
        --max-input-len "$MAX_INPUT_LEN" \
        $strat_args \
        "$rate"
}

sweep_one_setting() {
    local strategy="$1"
    local routing="$2"
    local rate="$3"
    local max_ok_rate="NA"
    local first_over_rate=""
    local stop_reason="max_rate_reached"
    local strat_prefix
    local routing_tag

    strat_prefix="$(strategy_prefix "$strategy")"
    routing_tag="$(routing_short "$routing")"

    log "===== START strategy=$strategy routing=$routing start=$rate step=$STEP_RATE ====="

    while ! rate_gt "$rate" "$MAX_RATE"; do
        local n_reqs
        local attempt=1
        local status=""
        local log_file=""
        local json_file=""
        local artifact_dir=""
        local itl_avg=""
        local itl_p99=""
        local queue_avg=""
        local queue_p99=""
        local decode_queue_avg=""
        local decode_queue_p99=""

        n_reqs="$(rate_to_nreqs "$rate")"

        while (( attempt <= MAX_RETRIES )); do
            log "RUN strategy=$strategy routing=$routing rate=$rate n_reqs=$n_reqs attempt=$attempt/$MAX_RETRIES"

            set +e
            run_one_attempt "$strategy" "$routing" "$rate" "$n_reqs" >> "$OUT_LOG" 2>&1
            local rc=$?
            set -e

            artifact_dir="$(find_latest_artifact_dir "$strat_prefix" "$n_reqs" "$rate" "$routing_tag" || true)"
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

            log "WARN strategy=$strategy routing=$routing rate=$rate attempt=$attempt status=$status"
            attempt=$((attempt + 1))
            sleep 10
        done

        if [[ "$status" != "ok" ]]; then
            record_summary "$strategy" "$routing" "$rate" "$n_reqs" "$status" "" "" "" "" "" "" "${log_file:-}" "${json_file:-}"
            record_state "$strategy" "$routing" "$max_ok_rate" "$first_over_rate" "${status}_at_${rate}"
            log "ERROR strategy=$strategy routing=$routing rate=$rate final_status=$status"
            return 1
        fi

        record_summary "$strategy" "$routing" "$rate" "$n_reqs" "ok" "$itl_avg" "${itl_p99:-}" "${queue_avg:-}" "${queue_p99:-}" "${decode_queue_avg:-}" "${decode_queue_p99:-}" "$log_file" "${json_file:-}"
        log "RESULT strategy=$strategy routing=$routing rate=$rate itl_avg=${itl_avg}ms itl_p99=${itl_p99}ms queue_avg=${queue_avg}ms queue_p99=${queue_p99}ms decode_queue_avg=${decode_queue_avg}ms decode_queue_p99=${decode_queue_p99}ms"

        if awk -v itl_avg="$itl_avg" -v stop_ms="$STOP_THRESHOLD_MS" "BEGIN {exit !(itl_avg > stop_ms)}"; then
            first_over_rate="$rate"
            stop_reason="itl_avg_gt_${STOP_THRESHOLD_MS}_at_${rate}"
            log "STOP strategy=$strategy routing=$routing because itl_avg=${itl_avg}ms > ${STOP_THRESHOLD_MS}ms"
            break
        fi

        max_ok_rate="$rate"
        rate="$(rate_add "$rate" "$STEP_RATE")"
        sleep "$SLEEP_BETWEEN_RUNS"
    done

    record_state "$strategy" "$routing" "$max_ok_rate" "$first_over_rate" "$stop_reason"
    log "===== DONE strategy=$strategy routing=$routing max_ok=$max_ok_rate first_over=${first_over_rate:-none} reason=$stop_reason ====="
}

main() {
    ensure_headers
    log "RUN_TAG=$RUN_TAG"
    log "BASE_LOG_DIR=$BASE_LOG_DIR"
    log "MODEL_PATH=$MODEL_PATH"
    log "DATASET_PATH=$DATASET_PATH"
    log "RAY_ADDR=$RAY_ADDR MASTER_ADDR=$MASTER_ADDR"
    log "SEG=$SEG BATCH_SIZE=$BATCH_SIZE MAX_INPUT_LEN=$MAX_INPUT_LEN"
    log "ENABLE_DYNAMIC_SP_SIZE=$ENABLE_DYNAMIC_SP_SIZE"
    log "ENFORCE_EAGER=$ENFORCE_EAGER"
    log "USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER=$USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER"
    log "ORDER=${SWEEP_STRATEGY}/LB"

    sweep_one_setting "$SWEEP_STRATEGY" LeastBatch "$START_RATE"

    log "ALL_DONE"
}

main "$@"
