#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
START_BENCH_SH="$SCRIPT_DIR/start_bench.sh"

RAY_ADDR="${RAY_ADDR:-10.102.97.179:7799}"
MASTER_ADDR="${MASTER_ADDR:-10.102.97.179:29500}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme1n1/ml_research/models/deepseek-v3}"
DATASET_PATH="${DATASET_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.03_n60000.csv}"

BATCH_SIZE="${BATCH_SIZE:-128}"
GPU_MEM="${GPU_MEM:-141}"
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
LOOP_COUNT="${LOOP_COUNT:-16}"
THRESHOLD="${THRESHOLD:-100000}"

BASE_LOG_DIR="${BASE_LOG_DIR:-$ROOT_DIR/bench_logs/issue003_longshort_r20_40_60_ndnew}"
DRIVER_LOG="$BASE_LOG_DIR/driver.log"
mkdir -p "$BASE_LOG_DIR"

for rate in 20 40 60; do
    nreq=$(awk "BEGIN {printf \"%d\", int(600 * $rate + 0.5)}")
    echo "[$(date -u +%F' '%T)] START rate=$rate nreq=$nreq" | tee -a "$DRIVER_LOG"

    bash "$START_BENCH_SH" \
        --master-addr "$MASTER_ADDR" \
        --ray-addr "$RAY_ADDR" \
        --dataset-path "$DATASET_PATH" \
        --model-path "$MODEL_PATH" \
        --batch-size "$BATCH_SIZE" \
        --num-requests "$nreq" \
        --gpu-util "$GPU_UTIL" \
        --gpu-mem "$GPU_MEM" \
        --max-model-len "$MAX_MODEL_LEN" \
        --routing-strategy LeastBatch \
        --scheduler-mode centralized \
        --loop-count "$LOOP_COUNT" \
        --fixed-sp-segments 0 \
        --enable-dynamic-sp-size \
        --dynamic-sp-size-strategy long_short_sp8 \
        --long-request-sp-threshold "$THRESHOLD" \
        --dp-size 4 \
        --sp-size 8 \
        "$rate" 2>&1 | tee -a "$DRIVER_LOG"

    echo "[$(date -u +%F' '%T)] END rate=$rate" | tee -a "$DRIVER_LOG"
    sleep 20
done
