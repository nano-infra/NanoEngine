#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
START_BENCH_SH="$SCRIPT_DIR/start_bench.sh"

RAY_ADDR="${RAY_ADDR:-10.102.252.174:6379}"
MASTER_ADDR="${MASTER_ADDR:-10.102.252.174:29500}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme1n1/ml_research/chenjiefei/models/deepseek-v3}"
DATASET_PATH="${DATASET_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.03_n60000.csv}"

SEG="${SEG:-65536}"
SCHEDULER_ARCH="${SCHEDULER_ARCH:-legacy_global}"
BATCH_SIZE="${BATCH_SIZE:-192}"
GPU_MEM="${GPU_MEM:-141}"
GPU_UTIL="${GPU_UTIL:-0.9}"
PIECEWISE_GPU_UTIL="${PIECEWISE_GPU_UTIL:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-1000000}"
LOOP_COUNT="${LOOP_COUNT:-16}"
RATE_DURATION_SEC="${RATE_DURATION_SEC:-600}"
FIXED_SP_SIZE="${FIXED_SP_SIZE:-0}"
SP_BACKEND="${SP_BACKEND:-hao_basic}"
CUDA_GRAPH_MODE="${CUDA_GRAPH_MODE:-full}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DYNAMIC_SP_SIZE_STRATEGY="${DYNAMIC_SP_SIZE_STRATEGY:-legacy}"
DYNAMIC_SP_BUCKET_PRESET="${DYNAMIC_SP_BUCKET_PRESET:-none}"
SWEEP_STRATEGY="${SWEEP_STRATEGY:-dp4sp8}"
SWEEP_ROUTING="${SWEEP_ROUTING:-LeastBatch}"
BENCH_VARIANTS="${BENCH_VARIANTS:-current}"
CURRENT_VARIANT="${CURRENT_VARIANT:-current}"
ENABLE_PIECEWISE_VARIANT="${ENABLE_PIECEWISE_VARIANT:-0}"

START_RATE="${START_RATE:-10}"
STEP_RATE="${STEP_RATE:-10}"
MAX_RATE="${MAX_RATE:-500}"
STOP_THRESHOLD_MS="${STOP_THRESHOLD_MS:-100}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-20}"
MAX_RETRIES="${MAX_RETRIES:-5}"
RESUME_SKIP_SUCCESS="${RESUME_SKIP_SUCCESS:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
NUM_REQUESTS_OVERRIDE="${NUM_REQUESTS_OVERRIDE:-}"

RUN_TAG="${RUN_TAG:-issue003_dp4sp8_4node_lb_$(date -u +%Y%m%d_%H%M%S)}"
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
        printf "strategy\trouting\tpolicy\trate\tn_reqs\tstatus\titl_avg_ms\titl_p99_ms\tqueue_avg_ms\tqueue_p99_ms\tdecode_queue_avg_ms\tdecode_queue_p99_ms\tlog_file\tjson_file\n" > "$SUMMARY_FILE"
    fi
    if [[ ! -f "$STATE_FILE" ]]; then
        printf "strategy\trouting\tpolicy\tmax_ok_rate\tfirst_over_rate\tstop_reason\n" > "$STATE_FILE"
    fi
}

rate_to_nreqs() {
    if [[ -n "$NUM_REQUESTS_OVERRIDE" ]]; then
        echo "$NUM_REQUESTS_OVERRIDE"
        return
    fi
    awk -v duration="$RATE_DURATION_SEC" -v rate="$1" 'BEGIN {printf "%d", int(duration * rate + 0.5)}'
}

rate_add() {
    awk "BEGIN {printf \"%.10g\", $1 + $2}"
}

rate_gt() {
    awk "BEGIN {exit !($1 > $2)}"
}

rate_lt() {
    awk "BEGIN {exit !($1 < $2)}"
}

should_continue_rate() {
    local rate="$1"
    local step="$2"
    local max_rate="$3"

    if awk -v step="$step" 'BEGIN {exit !(step > 0)}'; then
        ! rate_gt "$rate" "$max_rate"
        return
    fi
    if awk -v step="$step" 'BEGIN {exit !(step < 0)}'; then
        ! rate_lt "$rate" "$max_rate"
        return
    fi
    return 1
}

routing_short() {
    case "$1" in
        LeastBatch) echo "LB" ;;
        LeastCache) echo "LC" ;;
        RoundRobin) echo "RR" ;;
        *) echo "$1" ;;
    esac
}

strategy_prefix() {
    case "$1" in
        dp2sp8) echo "dp2sp8" ;;
        dp4sp8) echo "dp4sp8" ;;
        dp16sp1) echo "dp16sp1" ;;
        dp32sp1) echo "dp32sp1" ;;
        *) return 1 ;;
    esac
}

strategy_args() {
    case "$1" in
        dp2sp8) echo "--dp-size 2 --sp-size 8" ;;
        dp4sp8) echo "--dp-size 4 --sp-size 8" ;;
        dp16sp1) echo "--dp-size 16 --sp-size 1" ;;
        dp32sp1) echo "--dp-size 32 --sp-size 1" ;;
        *) return 1 ;;
    esac
}

base_policy_key_for_strategy() {
    if [[ "$DYNAMIC_SP_SIZE_STRATEGY" == "legacy" ]]; then
        echo "legacy"
        return
    fi
    echo "${DYNAMIC_SP_SIZE_STRATEGY}_${DYNAMIC_SP_BUCKET_PRESET}"
}

policy_key_for_strategy() {
    local strategy="$1"
    local base_policy
    base_policy="$(base_policy_key_for_strategy "$strategy")"

    case "$CURRENT_VARIANT" in
        current) echo "current_${base_policy}" ;;
        fixed_sp8) echo "fixed_sp8" ;;
        piecewise) echo "piecewise_${base_policy}" ;;
        nccl) echo "nccl_${base_policy}" ;;
        *) echo "${CURRENT_VARIANT}_${base_policy}" ;;
    esac
}

validate_variant() {
    case "$1" in
        current|fixed_sp8|piecewise|nccl) ;;
        *) echo "Error: Invalid BENCH_VARIANTS entry '$1'."; exit 1 ;;
    esac
}

find_latest_artifact_dir() {
    local strat_prefix="$1"
    local n_reqs="$2"
    local rate="$3"
    local routing_tag="$4"
    local policy_key="$5"
    local seg_short=$((SEG / 1024))k
    local dataset_name
    local model_name
    dataset_name="$(basename "$DATASET_PATH" .csv)"
    model_name="$(basename "$MODEL_PATH")"
    find "$BASE_LOG_DIR/$model_name/$dataset_name" -maxdepth 1 -type d \
        -name "${strat_prefix}_seg${seg_short}_n${n_reqs}_r${rate}_bs${BATCH_SIZE}_${routing_tag}_cen*_${policy_key}" \
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

clear_resume_result() {
    RESUME_HIT_SOURCE=""
    RESUME_ITL_AVG=""
    RESUME_ITL_P99=""
    RESUME_QUEUE_AVG=""
    RESUME_QUEUE_P99=""
    RESUME_DECODE_QUEUE_AVG=""
    RESUME_DECODE_QUEUE_P99=""
    RESUME_LOG_FILE=""
    RESUME_JSON_FILE=""
}

load_summary_ok_result() {
    local strategy="$1"
    local routing="$2"
    local policy="$3"
    local rate="$4"
    local row=""
    local _strategy=""
    local _routing=""
    local _policy=""
    local _rate=""
    local _n_reqs=""
    local _status=""

    [[ -f "$SUMMARY_FILE" ]] || return 1

    row="$(awk -F '\t' -v strategy="$strategy" -v routing="$routing" -v policy="$policy" -v rate="$rate" '
        NR == 1 { next }
        $1 == strategy && $2 == routing && $3 == policy && ($4 + 0) == (rate + 0) && $6 == "ok" {
            row = $0
        }
        END {
            if (row != "") {
                print row
            }
        }
    ' "$SUMMARY_FILE")"
    [[ -n "${row:-}" ]] || return 1

    IFS=$'\t' read -r _strategy _routing _policy _rate _n_reqs _status \
        RESUME_ITL_AVG RESUME_ITL_P99 RESUME_QUEUE_AVG RESUME_QUEUE_P99 \
        RESUME_DECODE_QUEUE_AVG RESUME_DECODE_QUEUE_P99 RESUME_LOG_FILE RESUME_JSON_FILE <<< "$row"

    if [[ -z "${RESUME_ITL_AVG:-}" ]] || ! is_number "$RESUME_ITL_AVG"; then
        clear_resume_result
        return 1
    fi

    RESUME_HIT_SOURCE="summary"
    return 0
}

load_artifact_ok_result() {
    local strategy="$1"
    local routing="$2"
    local rate="$3"
    local n_reqs="$4"
    local policy="$5"
    local strat_prefix=""
    local routing_tag=""
    local artifact_dir=""

    strat_prefix="$(strategy_prefix "$strategy")"
    routing_tag="$(routing_short "$routing")"
    artifact_dir="$(find_latest_artifact_dir "$strat_prefix" "$n_reqs" "$rate" "$routing_tag" "$policy" || true)"
    [[ -n "${artifact_dir:-}" && -d "$artifact_dir" ]] || return 1

    RESUME_LOG_FILE="$(find_latest_log "$artifact_dir" || true)"
    if [[ -z "${RESUME_LOG_FILE:-}" || ! -f "$RESUME_LOG_FILE" ]]; then
        clear_resume_result
        return 1
    fi

    RESUME_JSON_FILE="$(find_latest_json "$artifact_dir" || true)"
    RESUME_ITL_AVG="$(parse_metric "$RESUME_LOG_FILE" itl_avg || true)"
    if [[ -z "${RESUME_ITL_AVG:-}" ]] || ! is_number "$RESUME_ITL_AVG"; then
        clear_resume_result
        return 1
    fi
    RESUME_ITL_P99="$(parse_metric "$RESUME_LOG_FILE" itl_p99 || true)"
    RESUME_QUEUE_AVG="$(parse_metric "$RESUME_LOG_FILE" queue_avg || true)"
    RESUME_QUEUE_P99="$(parse_metric "$RESUME_LOG_FILE" queue_p99 || true)"
    RESUME_DECODE_QUEUE_AVG="$(parse_metric "$RESUME_LOG_FILE" decode_queue_avg || true)"
    RESUME_DECODE_QUEUE_P99="$(parse_metric "$RESUME_LOG_FILE" decode_queue_p99 || true)"
    RESUME_HIT_SOURCE="artifact"
    return 0
}

hydrate_existing_ok_result() {
    local strategy="$1"
    local routing="$2"
    local policy="$3"
    local rate="$4"
    local n_reqs="$5"

    clear_resume_result
    if [[ "$FORCE_RERUN" -ne 0 || "$RESUME_SKIP_SUCCESS" -eq 0 ]]; then
        return 1
    fi

    if load_summary_ok_result "$strategy" "$routing" "$policy" "$rate"; then
        return 0
    fi

    if load_artifact_ok_result "$strategy" "$routing" "$rate" "$n_reqs" "$policy"; then
        return 0
    fi

    clear_resume_result
    return 1
}

record_summary() {
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" >> "$SUMMARY_FILE"
}

record_state() {
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" "$6" >> "$STATE_FILE"
}

run_one_attempt() {
    local strategy="$1"
    local routing="$2"
    local rate="$3"
    local n_reqs="$4"
    local strat_args
    local eager_args=()
    local sp_strategy_args=()
    local fixed_sp_size="$FIXED_SP_SIZE"
    local sp_backend="$SP_BACKEND"
    local cuda_graph_mode="$CUDA_GRAPH_MODE"
    local gpu_util="$GPU_UTIL"
    local dynamic_sp_size_strategy="$DYNAMIC_SP_SIZE_STRATEGY"
    local dynamic_sp_bucket_preset="$DYNAMIC_SP_BUCKET_PRESET"
    local policy_key

    policy_key="$(policy_key_for_strategy "$strategy")"

    case "$CURRENT_VARIANT" in
        current)
            ;;
        fixed_sp8)
            fixed_sp_size=8
            sp_backend="hao_basic"
            cuda_graph_mode="full"
            dynamic_sp_size_strategy="legacy"
            dynamic_sp_bucket_preset="none"
            ;;
        piecewise)
            fixed_sp_size=0
            sp_backend="hao_basic"
            cuda_graph_mode="piecewise"
            gpu_util="$PIECEWISE_GPU_UTIL"
            ;;
        nccl)
            fixed_sp_size=0
            sp_backend="nccl"
            cuda_graph_mode="full"
            ;;
    esac

    strat_args="$(strategy_args "$strategy")"
    if [[ "$ENFORCE_EAGER" -ne 0 ]]; then
        eager_args+=(--enforce-eager)
    fi
    if [[ -n "${dynamic_sp_size_strategy:-}" ]]; then
        sp_strategy_args+=(--dynamic-sp-size-strategy "$dynamic_sp_size_strategy")
        sp_strategy_args+=(--dynamic-sp-bucket-preset "$dynamic_sp_bucket_preset")
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
        --gpu-util "$gpu_util" \
        --max-model-len "$MAX_MODEL_LEN" \
        --routing-strategy "$routing" \
        --scheduler-arch "$SCHEDULER_ARCH" \
        --loop-count "$LOOP_COUNT" \
        --fixed-sp-size "$fixed_sp_size" \
        --sp-backend "$sp_backend" \
        --cuda-graph-mode "$cuda_graph_mode" \
        --run-label "$policy_key" \
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
    local policy_key=""
    local strat_prefix
    local routing_tag

    strat_prefix="$(strategy_prefix "$strategy")"
    routing_tag="$(routing_short "$routing")"
    policy_key="$(policy_key_for_strategy "$strategy")"

    log "===== START strategy=$strategy routing=$routing policy=$policy_key start=$rate step=$STEP_RATE max=$MAX_RATE ====="

    while should_continue_rate "$rate" "$STEP_RATE" "$MAX_RATE"; do
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

        if hydrate_existing_ok_result "$strategy" "$routing" "$policy_key" "$rate" "$n_reqs"; then
            status="ok"
            itl_avg="$RESUME_ITL_AVG"
            itl_p99="$RESUME_ITL_P99"
            queue_avg="$RESUME_QUEUE_AVG"
            queue_p99="$RESUME_QUEUE_P99"
            decode_queue_avg="$RESUME_DECODE_QUEUE_AVG"
            decode_queue_p99="$RESUME_DECODE_QUEUE_P99"
            log_file="$RESUME_LOG_FILE"
            json_file="$RESUME_JSON_FILE"
            log "SKIP strategy=$strategy routing=$routing policy=$policy_key rate=$rate reason=existing_ok_${RESUME_HIT_SOURCE} itl_avg=${itl_avg}ms"
        else
            while (( attempt <= MAX_RETRIES )); do
                log "RUN strategy=$strategy routing=$routing policy=$policy_key rate=$rate n_reqs=$n_reqs attempt=$attempt/$MAX_RETRIES"

                set +e
                run_one_attempt "$strategy" "$routing" "$rate" "$n_reqs" >> "$OUT_LOG" 2>&1
                local rc=$?
                set -e

                artifact_dir="$(find_latest_artifact_dir "$strat_prefix" "$n_reqs" "$rate" "$routing_tag" "$policy_key" || true)"
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

                log "WARN strategy=$strategy routing=$routing policy=$policy_key rate=$rate attempt=$attempt status=$status"
                attempt=$((attempt + 1))
                sleep 10
            done
        fi

        if [[ "$status" != "ok" ]]; then
            record_summary "$strategy" "$routing" "$policy_key" "$rate" "$n_reqs" "$status" "" "" "" "" "" "" "${log_file:-}" "${json_file:-}"
            record_state "$strategy" "$routing" "$policy_key" "$max_ok_rate" "$first_over_rate" "${status}_at_${rate}"
            log "ERROR strategy=$strategy routing=$routing policy=$policy_key rate=$rate final_status=$status"
            return 1
        fi

        if [[ -z "${RESUME_HIT_SOURCE:-}" || "$RESUME_HIT_SOURCE" == "artifact" ]]; then
            record_summary "$strategy" "$routing" "$policy_key" "$rate" "$n_reqs" "ok" "$itl_avg" "${itl_p99:-}" "${queue_avg:-}" "${queue_p99:-}" "${decode_queue_avg:-}" "${decode_queue_p99:-}" "$log_file" "${json_file:-}"
        fi
        if [[ -z "${RESUME_HIT_SOURCE:-}" ]]; then
            log "RESULT strategy=$strategy routing=$routing policy=$policy_key rate=$rate itl_avg=${itl_avg}ms itl_p99=${itl_p99}ms queue_avg=${queue_avg}ms queue_p99=${queue_p99}ms decode_queue_avg=${decode_queue_avg}ms decode_queue_p99=${decode_queue_p99}ms"
        fi

        if awk -v itl_avg="$itl_avg" -v stop_ms="$STOP_THRESHOLD_MS" "BEGIN {exit !(stop_ms > 0 && itl_avg > stop_ms)}"; then
            first_over_rate="$rate"
            stop_reason="itl_avg_gt_${STOP_THRESHOLD_MS}_at_${rate}"
            log "STOP strategy=$strategy routing=$routing policy=$policy_key because itl_avg=${itl_avg}ms > ${STOP_THRESHOLD_MS}ms"
            break
        fi

        max_ok_rate="$rate"
        rate="$(rate_add "$rate" "$STEP_RATE")"
        sleep "$SLEEP_BETWEEN_RUNS"
    done

    record_state "$strategy" "$routing" "$policy_key" "$max_ok_rate" "$first_over_rate" "$stop_reason"
    log "===== DONE strategy=$strategy routing=$routing policy=$policy_key max_ok=$max_ok_rate first_over=${first_over_rate:-none} reason=$stop_reason ====="
}

main() {
    ensure_headers
    log "RUN_TAG=$RUN_TAG"
    log "BASE_LOG_DIR=$BASE_LOG_DIR"
    log "MODEL_PATH=$MODEL_PATH"
    log "DATASET_PATH=$DATASET_PATH"
    log "RAY_ADDR=$RAY_ADDR MASTER_ADDR=$MASTER_ADDR"
    log "SEG=$SEG BATCH_SIZE=$BATCH_SIZE MAX_INPUT_LEN=$MAX_INPUT_LEN RATE_DURATION_SEC=$RATE_DURATION_SEC"
    log "ENFORCE_EAGER=$ENFORCE_EAGER"
    log "DYNAMIC_SP_SIZE_STRATEGY=$DYNAMIC_SP_SIZE_STRATEGY DYNAMIC_SP_BUCKET_PRESET=$DYNAMIC_SP_BUCKET_PRESET"
    log "SP_BACKEND=$SP_BACKEND CUDA_GRAPH_MODE=$CUDA_GRAPH_MODE FIXED_SP_SIZE=$FIXED_SP_SIZE"
    log "GPU_UTIL=$GPU_UTIL PIECEWISE_GPU_UTIL=$PIECEWISE_GPU_UTIL"
    log "RESUME_SKIP_SUCCESS=$RESUME_SKIP_SUCCESS FORCE_RERUN=$FORCE_RERUN"
    log "ENABLE_PIECEWISE_VARIANT=$ENABLE_PIECEWISE_VARIANT"
    log "ORDER=${SWEEP_STRATEGY}/$(routing_short "$SWEEP_ROUTING") variants=$BENCH_VARIANTS"

    local ran_variant=0
    IFS=',' read -r -a variant_list <<< "$BENCH_VARIANTS"
    for variant in "${variant_list[@]}"; do
        variant="${variant//[[:space:]]/}"
        [[ -n "$variant" ]] || continue
        validate_variant "$variant"
        if [[ "$variant" == "piecewise" && "$ENABLE_PIECEWISE_VARIANT" -eq 0 ]]; then
            log "VARIANT_SKIP=piecewise reason=disabled_set_ENABLE_PIECEWISE_VARIANT_1_to_run"
            continue
        fi
        CURRENT_VARIANT="$variant"
        ran_variant=1
        log "VARIANT_START=$CURRENT_VARIANT"
        sweep_one_setting "$SWEEP_STRATEGY" "$SWEEP_ROUTING" "$START_RATE"
        log "VARIANT_DONE=$CURRENT_VARIANT"
    done
    if [[ "$ran_variant" -eq 0 ]]; then
        log "ERROR no enabled benchmark variants; set BENCH_VARIANTS or ENABLE_PIECEWISE_VARIANT=1"
        return 1
    fi

    log "ALL_DONE"
}

main "$@"
