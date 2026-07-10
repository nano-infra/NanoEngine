#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_SCRIPT="$SCRIPT_DIR/bench_serving_overhead.py"
BUILD_LIB_DIR="$ROOT_DIR/build/lib"

if [[ -d "$BUILD_LIB_DIR" ]]; then
    export PYTHONPATH="$ROOT_DIR:$BUILD_LIB_DIR${PYTHONPATH:+:$PYTHONPATH}"
else
    export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

MASTER_IP="${MASTER_IP:-10.102.243.60}"
RAY_ADDR="${RAY_ADDR:-${MASTER_IP}:8776}"
MASTER_ADDR="${MASTER_ADDR:-${MASTER_IP}:27799}"

MODEL_PATH="${MODEL_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3}"
DATASET_PATH="${DATASET_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv}"

REQUEST_RATE="${REQUEST_RATE:-2}"
if [[ $# -gt 0 ]]; then
    REQUEST_RATE="$1"
fi

DP="${DP:-2}"
SP="${SP:-8}"
TP="${TP:-1}"
EP="${EP:-$((DP * SP * TP))}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_REQUESTS="${NUM_REQUESTS:-$(awk "BEGIN {printf \"%d\", int(600 * $REQUEST_RATE + 0.5)}")}"

SEG="${SEG:-65536}"
GPU_MEM="${GPU_MEM:-141}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-1000000}"
ROUTING="${ROUTING:-LeastBatch}"
SCHEDULER="${SCHEDULER:-centralized}"
SP_BACKEND="${SP_BACKEND:-nccl}"
LOOP_COUNT="${LOOP_COUNT:-1}"
MAX_NUM_RECV_SEQS="${MAX_NUM_RECV_SEQS:-192}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
DISABLE_NON_UNIFORM_SPLIT="${DISABLE_NON_UNIFORM_SPLIT:-0}"
DRY_RUN="${DRY_RUN:-0}"

LOONGSERVE_MIN_COMP_BOUND_BATCH_SIZE="${LOONGSERVE_MIN_COMP_BOUND_BATCH_SIZE:-64}"
LOONGSERVE_MAX_LOCAL_DECODE_SP="${LOONGSERVE_MAX_LOCAL_DECODE_SP:-8}"
LOONGSERVE_DECODE_PROFILE_PATH="${LOONGSERVE_DECODE_PROFILE_PATH:-}"
LOONGSERVE_DECODE_CROSS_NODE_SP="${LOONGSERVE_DECODE_CROSS_NODE_SP:-0}"

RUN_TAG="${RUN_TAG:-loongserve_issue001_2node_dp${DP}sp${SP}_r${REQUEST_RATE}_$(date -u +%Y%m%d_%H%M%S)}"
BASE_LOG_DIR="${BASE_LOG_DIR:-$ROOT_DIR/bench_logs/$RUN_TAG}"
DATASET_NAME="$(basename "$DATASET_PATH" .csv)"
MODEL_NAME="$(basename "$MODEL_PATH")"
RATE_TAG="${REQUEST_RATE//./p}"
SEG_SHORT="$((SEG / 1024))k"
STRATEGY_STR="dp${DP}sp${SP}_seg${SEG_SHORT}_n${NUM_REQUESTS}_r${RATE_TAG}_bs${BATCH_SIZE}_${ROUTING}_${SCHEDULER}_loongserve_thr${LOONGSERVE_MIN_COMP_BOUND_BATCH_SIZE}"
CURRENT_LOG_DIR="$BASE_LOG_DIR/$MODEL_NAME/$DATASET_NAME/$STRATEGY_STR"
TIMESTAMP="$(TZ='Asia/Shanghai' date '+%Y%m%d_%H%M%S')"
LOG_FILE="$CURRENT_LOG_DIR/${TIMESTAMP}.log"
JSON_FILE="$CURRENT_LOG_DIR/${TIMESTAMP}.json"
PROGRESS_FILE="$BASE_LOG_DIR/run.progress"

mkdir -p "$CURRENT_LOG_DIR"

log() {
    local now
    now="$(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')"
    echo "[$now] $*" | tee -a "$PROGRESS_FILE"
}

CMD=(
    python "$BENCH_SCRIPT"
    --dataset csv
    --csv-path "$DATASET_PATH"
    --num-requests "$NUM_REQUESTS"
    --request-rate "$REQUEST_RATE"
    --sp "$SP"
    --dp "$DP"
    --ep "$EP"
    --tp "$TP"
    --max-num-seqs "$BATCH_SIZE"
    --max-num-recv-seqs "$MAX_NUM_RECV_SEQS"
    --gpu-memory-limit-gb "$GPU_MEM"
    --gpu-memory-utilization "$GPU_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --dummy-prefill
    --ray-address "$RAY_ADDR"
    --master-address "$MASTER_ADDR"
    --loop-count "$LOOP_COUNT"
    --model-path "$MODEL_PATH"
    --routing-strategy "$ROUTING"
    --itl-log-path "$JSON_FILE"
    --segment-size "$SEG"
    --sp-backend "$SP_BACKEND"
    --scheduler-mode "$SCHEDULER"
    --cuda-graph-mode full
    --loongserve-decode-scheduler
    --loongserve-min-comp-bound-batch-size "$LOONGSERVE_MIN_COMP_BOUND_BATCH_SIZE"
    --loongserve-max-local-decode-sp "$LOONGSERVE_MAX_LOCAL_DECODE_SP"
)

if [[ -n "$MAX_INPUT_LEN" ]]; then
    CMD+=(--max-input-len "$MAX_INPUT_LEN")
fi
if [[ -n "$LOONGSERVE_DECODE_PROFILE_PATH" ]]; then
    CMD+=(--loongserve-decode-profile-path "$LOONGSERVE_DECODE_PROFILE_PATH")
fi
if [[ "$LOONGSERVE_DECODE_CROSS_NODE_SP" -ne 0 ]]; then
    CMD+=(--loongserve-decode-cross-node-sp)
fi
if [[ "$ENFORCE_EAGER" -ne 0 ]]; then
    CMD+=(--enforce-eager)
fi
if [[ "$DISABLE_NON_UNIFORM_SPLIT" -ne 0 ]]; then
    CMD+=(--disable-non-uniform-split)
fi

{
    echo "================= Benchmark Metadata ================="
    echo "Time (Beijing): $TIMESTAMP"
    echo "Run Tag: $RUN_TAG"
    echo "Model Path: $MODEL_PATH"
    echo "Dataset Path: $DATASET_PATH"
    echo "Ray Address: $RAY_ADDR"
    echo "Master Address: $MASTER_ADDR"
    echo "Topology: dp=$DP sp=$SP tp=$TP ep=$EP"
    echo "Request Rate: $REQUEST_RATE"
    echo "Num Requests: $NUM_REQUESTS"
    echo "Batch Size: $BATCH_SIZE"
    echo "SP Backend: $SP_BACKEND"
    echo "Loop Count: $LOOP_COUNT"
    echo "LoongServe Threshold: $LOONGSERVE_MIN_COMP_BOUND_BATCH_SIZE"
    echo "LoongServe Profile Path: ${LOONGSERVE_DECODE_PROFILE_PATH:-<fallback>}"
    echo "Output Dir: $CURRENT_LOG_DIR"
    echo ""
    echo "================= Command ================="
    printf 'RAY_DEDUP_LOGS=0'
    printf ' %q' "${CMD[@]}"
    echo
    echo "====================================================="
    echo ""
} > "$LOG_FILE"

log "RUN_TAG=$RUN_TAG"
log "START loongserve issue001 rate=$REQUEST_RATE n_reqs=$NUM_REQUESTS output=$CURRENT_LOG_DIR"
log "RAY_ADDR=$RAY_ADDR MASTER_ADDR=$MASTER_ADDR topology=dp${DP}sp${SP}ep${EP}"
log "loongserve_min_comp_bound_batch_size=$LOONGSERVE_MIN_COMP_BOUND_BATCH_SIZE profile=${LOONGSERVE_DECODE_PROFILE_PATH:-fallback}"

if [[ "$DRY_RUN" -ne 0 ]]; then
    log "DRY_RUN command written to $LOG_FILE"
    exit 0
fi

cd "$ROOT_DIR"
set -o pipefail
RAY_DEDUP_LOGS=0 "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=$?
set +o pipefail

if [[ "$EXIT_CODE" -eq 0 ]]; then
    log "DONE status=SUCCESS log=$LOG_FILE json=$JSON_FILE"
else
    log "DONE status=FAILED exit_code=$EXIT_CODE log=$LOG_FILE"
fi

exit "$EXIT_CODE"
