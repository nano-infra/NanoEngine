#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
START_BENCH_SH="$ROOT_DIR/scripts/issue003/start_bench.sh"

RAY_ADDR="${RAY_ADDR:-10.102.252.174:7799}"
MASTER_ADDR="${MASTER_ADDR:-10.102.252.174:29500}"

MODEL_PATH="${MODEL_PATH:-/mnt/nvme1n1/ml_research/models/deepseek-v3}"
DATASET_PATH="${DATASET_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.03_n60000.csv}"

SEG="${SEG:-65536}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SCHEDULER="${SCHEDULER:-centralized}"
ROUTING_STRATEGY="${ROUTING_STRATEGY:-LeastBatch}"
GPU_MEM="${GPU_MEM:-141}"
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-1000000}"
LOOP_COUNT="${LOOP_COUNT:-16}"
FIXED_SP_SEGMENTS="${FIXED_SP_SEGMENTS:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER="${USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER:-0}"

DP_SIZE="${DP_SIZE:-2}"
SP_SIZE="${SP_SIZE:-8}"
ENABLE_DYNAMIC_SP_SIZE="${ENABLE_DYNAMIC_SP_SIZE:-1}"
DYNAMIC_SP_SIZE_STRATEGY="${DYNAMIC_SP_SIZE_STRATEGY:-long_short_sp8}"
LONG_REQUEST_SP_THRESHOLD="${LONG_REQUEST_SP_THRESHOLD:-100000}"
LONG_REQUEST_SP_SIZE="${LONG_REQUEST_SP_SIZE:-8}"

RATES="${RATES:-10}"
RUN_TAG="${RUN_TAG:-issue003_deepseek_v3_2node_dp2sp8_long_sp8_$(date -u +%Y%m%d_%H%M%S)}"
BASE_LOG_DIR="${BASE_LOG_DIR:-$ROOT_DIR/bench_logs/$RUN_TAG}"
DRIVER_LOG="$BASE_LOG_DIR/driver.log"

mkdir -p "$BASE_LOG_DIR"

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$DRIVER_LOG"
}

rate_to_nreqs() {
    awk "BEGIN {printf \"%d\", int(600 * $1 + 0.5)}"
}

main() {
    log "RUN_TAG=$RUN_TAG"
    log "RAY_ADDR=$RAY_ADDR MASTER_ADDR=$MASTER_ADDR"
    log "MODEL_PATH=$MODEL_PATH"
    log "DATASET_PATH=$DATASET_PATH"
    log "CONFIG=dp${DP_SIZE}sp${SP_SIZE} strategy=${DYNAMIC_SP_SIZE_STRATEGY} long_request_sp_size=${LONG_REQUEST_SP_SIZE} threshold=${LONG_REQUEST_SP_THRESHOLD}"
    log "RATES=$RATES BATCH_SIZE=$BATCH_SIZE SEG=$SEG"

    local rate
    for rate in $RATES; do
        local n_reqs
        n_reqs="$(rate_to_nreqs "$rate")"
        log "START rate=$rate n_reqs=$n_reqs"

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
            --routing-strategy "$ROUTING_STRATEGY" \
            --scheduler-mode "$SCHEDULER" \
            --loop-count "$LOOP_COUNT" \
            --fixed-sp-segments "$FIXED_SP_SEGMENTS" \
            --max-input-len "$MAX_INPUT_LEN" \
            --dynamic-sp-size-strategy "$DYNAMIC_SP_SIZE_STRATEGY" \
            --long-request-sp-threshold "$LONG_REQUEST_SP_THRESHOLD" \
            --long-request-sp-size "$LONG_REQUEST_SP_SIZE" \
            --dp-size "$DP_SIZE" \
            --sp-size "$SP_SIZE" \
            $( (( ENABLE_DYNAMIC_SP_SIZE != 0 )) && echo --enable-dynamic-sp-size ) \
            $( (( ENFORCE_EAGER != 0 )) && echo --enforce-eager ) \
            $( (( USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER != 0 )) && echo --use-new-decode-dynamic-sp-scheduler ) \
            "$rate" 2>&1 | tee -a "$DRIVER_LOG"

        log "DONE rate=$rate"
    done

    log "ALL_DONE"
}

main "$@"
