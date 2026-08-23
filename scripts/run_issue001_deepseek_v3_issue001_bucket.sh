#!/bin/bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_SCRIPT="$ROOT_DIR/scripts/issue003/bench_serving_overhead.py"
BUILD_LIB_DIR="$ROOT_DIR/build/lib"
GIT_SHA="$(git -C "$ROOT_DIR" rev-parse HEAD)"

if [[ -d "$BUILD_LIB_DIR" ]]; then
    export PYTHONPATH="$ROOT_DIR:$BUILD_LIB_DIR${PYTHONPATH:+:$PYTHONPATH}"
else
    export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

RAY_ADDR="${RAY_ADDR:-10.102.97.179:7799}"
MASTER_ADDR="${MASTER_ADDR:-10.102.97.179:29500}"

MODEL_PATH="${MODEL_PATH:-/mnt/nvme1n1/ml_research/models/deepseek-v3}"
DATASET_PATH="${DATASET_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv}"

DP="${DP:-4}"
SP="${SP:-8}"
TP="${TP:-1}"
EP="${EP:-$((DP * SP * TP))}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEG="${SEG:-65536}"
GPU_MEM="${GPU_MEM:-141}"
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-1000000}"
LOOP_COUNT="${LOOP_COUNT:-16}"
FIXED_SP_SIZE="${FIXED_SP_SIZE:-0}"
SCHEDULER_ARCH="${SCHEDULER_ARCH:-legacy_global}"
ROUTING="${ROUTING:-LeastBatch}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
DYNAMIC_SP_SIZE_STRATEGY="${DYNAMIC_SP_SIZE_STRATEGY:-bucket}"
DYNAMIC_SP_BUCKET_PRESET="${DYNAMIC_SP_BUCKET_PRESET:-deepseek_v3}"
DISABLE_NON_UNIFORM_SPLIT="${DISABLE_NON_UNIFORM_SPLIT:-0}"
STOP_THRESHOLD_MS="${STOP_THRESHOLD_MS:-100}"
DRY_RUN="${DRY_RUN:-0}"
DURATION_SECONDS="${DURATION_SECONDS:-600}"
ENABLE_PROFILER="${ENABLE_PROFILER:-0}"
PROFILER_MODE="${PROFILER_MODE:-default}"
PROFILER_START_STEP="${PROFILER_START_STEP:-40}"
PROFILING_STEP="${PROFILING_STEP:-16}"
PROFILER_RANKS="${PROFILER_RANKS:-}"
RUNTIME_OVERHEAD_TIMING="${RUNTIME_OVERHEAD_TIMING:-0}"

RUN_TAG="${RUN_TAG:-issue001_deepseek_v3_issue001_bucket_cp_${DYNAMIC_SP_BUCKET_PRESET}_$(date -u +%Y%m%d_%H%M%S)}"
BASE_LOG_DIR="${BASE_LOG_DIR:-$ROOT_DIR/bench_logs/$RUN_TAG}"
PROGRESS_FILE="$BASE_LOG_DIR/run.progress"

mkdir -p "$BASE_LOG_DIR"

log() {
    local now
    now=$(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')
    echo "[$now] $*" | tee -a "$PROGRESS_FILE"
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

rate_to_nreqs() {
    awk -v duration="$DURATION_SECONDS" \
        "BEGIN {printf \"%d\", int(duration * $1 + 0.5)}"
}

routing_short() {
    case "$1" in
        LeastBatch) echo "LB" ;;
        LeastCache) echo "LC" ;;
        RoundRobin) echo "RR" ;;
        *) echo "$1" ;;
    esac
}

scheduler_short() {
    case "$1" in
        legacy_global) echo "legacy" ;;
        hierarchical) echo "hier" ;;
        *) echo "$1" ;;
    esac
}

RATES=("$@")
if [[ ${#RATES[@]} -eq 0 ]]; then
    RATES=("35")
fi

DATASET_NAME="$(basename "$DATASET_PATH" .csv)"
MODEL_NAME="$(basename "$MODEL_PATH")"
RT_SHORT="$(routing_short "$ROUTING")"
SC_SHORT="$(scheduler_short "$SCHEDULER_ARCH")"
SEG_SHORT="$((SEG / 1024))k"

log "RUN_TAG=$RUN_TAG"
log "GIT_SHA=$GIT_SHA"
log "MODEL=$MODEL_PATH"
log "DATASET=$DATASET_PATH"
log "RAY_ADDR=$RAY_ADDR MASTER_ADDR=$MASTER_ADDR"
log "TOPOLOGY=dp${DP}sp${SP}tp${TP}ep${EP}"
log "DYNAMIC_SP_SIZE_STRATEGY=$DYNAMIC_SP_SIZE_STRATEGY DYNAMIC_SP_BUCKET_PRESET=$DYNAMIC_SP_BUCKET_PRESET"
log "STOP_THRESHOLD_MS=$STOP_THRESHOLD_MS"
log "DURATION_SECONDS=$DURATION_SECONDS"
log "PROFILER enabled=$ENABLE_PROFILER mode=$PROFILER_MODE ranks=${PROFILER_RANKS:-all}"
log "RUNTIME_OVERHEAD_TIMING=$RUNTIME_OVERHEAD_TIMING"
log "RATES=${RATES[*]}"

for rate in "${RATES[@]}"; do
    timestamp=$(TZ='Asia/Shanghai' date '+%Y%m%d_%H%M%S')
    num_reqs="$(rate_to_nreqs "$rate")"
    rate_tag="${rate//./p}"

    extra_tags="_bucket_cp_${DYNAMIC_SP_BUCKET_PRESET}"
    if [[ "$FIXED_SP_SIZE" -ne 0 ]]; then
        extra_tags="${extra_tags}_fsp${FIXED_SP_SIZE}"
    fi
    if [[ "$ENFORCE_EAGER" -ne 0 ]]; then
        extra_tags="${extra_tags}_eager"
    fi
    if [[ "$DISABLE_NON_UNIFORM_SPLIT" -ne 0 ]]; then
        extra_tags="${extra_tags}_uni"
    fi

    strategy_str="dp${DP}sp${SP}_seg${SEG_SHORT}_n${num_reqs}_r${rate}_bs${BATCH_SIZE}_${RT_SHORT}_${SC_SHORT}_maxin$((MAX_INPUT_LEN / 1000))k${extra_tags}"
    current_log_dir="$BASE_LOG_DIR/$MODEL_NAME/$DATASET_NAME/$strategy_str"
    mkdir -p "$current_log_dir"
    profiler_dir="${PROFILER_DIR:-$current_log_dir/profiler}"

    log_file="$current_log_dir/${timestamp}.log"
    json_file="$current_log_dir/${timestamp}.json"

    bench_args=(
        --dataset csv
        --csv-path "$DATASET_PATH"
        --num-requests "$num_reqs"
        --request-rate "$rate"
        --sp "$SP"
        --dp "$DP"
        --ep "$EP"
        --tp "$TP"
        --max-num-seqs "$BATCH_SIZE"
        --gpu-memory-limit-gb "$GPU_MEM"
        --gpu-memory-utilization "$GPU_UTIL"
        --max-model-len "$MAX_MODEL_LEN"
        --dummy-prefill
        --ray-address "$RAY_ADDR"
        --master-address "$MASTER_ADDR"
        --loop-count "$LOOP_COUNT"
        --model-path "$MODEL_PATH"
        --routing-strategy "$ROUTING"
        --itl-log-path "$json_file"
        --segment-size "$SEG"
        --scheduler-arch "$SCHEDULER_ARCH"
        --fixed-sp-size "$FIXED_SP_SIZE"
        --dynamic-sp-size-strategy "$DYNAMIC_SP_SIZE_STRATEGY"
        --dynamic-sp-bucket-preset "$DYNAMIC_SP_BUCKET_PRESET"
        --max-input-len "$MAX_INPUT_LEN"
        --profiler-dir "$profiler_dir"
    )

    if [[ "$ENFORCE_EAGER" -ne 0 ]]; then
        bench_args+=(--enforce-eager)
    fi
    if [[ "$DISABLE_NON_UNIFORM_SPLIT" -ne 0 ]]; then
        bench_args+=(--disable-non-uniform-split)
    fi
    if [[ "$ENABLE_PROFILER" -ne 0 ]]; then
        bench_args+=(
            --enable-profiler
            --profiler-mode "$PROFILER_MODE"
            --profiler-start-step "$PROFILER_START_STEP"
            --profiling-step "$PROFILING_STEP"
        )
        if [[ -n "$PROFILER_RANKS" ]]; then
            bench_args+=(--profiler-ranks "$PROFILER_RANKS")
        fi
    fi
    if [[ "$RUNTIME_OVERHEAD_TIMING" -ne 0 ]]; then
        bench_args+=(--runtime-overhead-timing)
    fi

    {
        echo "================= Benchmark Metadata ================="
        echo "Time (Beijing): $timestamp"
        echo "Model Path: $MODEL_PATH"
        echo "Dataset Path: $DATASET_PATH"
        echo "Rate: $rate"
        echo "Num Requests: $num_reqs"
        echo "Topology: dp=$DP sp=$SP tp=$TP ep=$EP"
        echo "Scheduler Architecture: $SCHEDULER_ARCH"
        echo "Routing: $ROUTING"
        echo "Dynamic SP Strategy: $DYNAMIC_SP_SIZE_STRATEGY"
        echo "Dynamic SP Bucket Preset: $DYNAMIC_SP_BUCKET_PRESET"
        echo "Output Dir: $current_log_dir"
        echo ""
        echo "================= Wrapped Command ================="
        echo "python3 -u \"$BENCH_SCRIPT\" ${bench_args[*]}"
        echo "====================================================="
        echo ""
    } > "$log_file"

    log "START rate=$rate n_reqs=$num_reqs output=$current_log_dir"

    if [[ "$DRY_RUN" -ne 0 ]]; then
        log "DRY_RUN rate=$rate command=${bench_args[*]}"
        continue
    fi

    cd "$ROOT_DIR"
    set -o pipefail
    RAY_DEDUP_LOGS=0 \
    python3 -u "$BENCH_SCRIPT" "${bench_args[@]}" 2>&1 | tee -a "$log_file"
    exit_code=$?
    set +o pipefail

    if [[ $exit_code -eq 0 ]]; then
        log "DONE rate=$rate status=SUCCESS output=$current_log_dir"
    else
        log "DONE rate=$rate status=FAILED exit_code=$exit_code output=$current_log_dir"
        exit $exit_code
    fi

    itl_avg="$(parse_metric "$log_file" itl_avg || true)"
    if [[ -n "${itl_avg:-}" ]] && is_number "$itl_avg"; then
        log "RESULT rate=$rate itl_avg=${itl_avg}ms"
        if awk -v itl_avg="$itl_avg" -v stop_ms="$STOP_THRESHOLD_MS" 'BEGIN {exit !(itl_avg > stop_ms)}'; then
            log "STOP because itl_avg=${itl_avg}ms > ${STOP_THRESHOLD_MS}ms at rate=$rate"
            break
        fi
    else
        log "WARN rate=$rate unable_to_parse_itl_avg log_file=$log_file"
    fi

    sleep 15
done

log "ALL_DONE"
