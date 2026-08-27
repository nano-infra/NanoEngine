#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
START_BENCH_SH="$ROOT_DIR/scripts/sp_ablation/start_bench.sh"

RAY_ADDR="${RAY_ADDR:-10.102.235.13:8745}"
MASTER_ADDR="${MASTER_ADDR:-10.102.235.13:29500}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3}"
DATASET_PATH="${DATASET_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv}"

RATE="${RATE:-50}"
DURATION_SECONDS="${DURATION_SECONDS:-300}"
NUM_REQUESTS="${NUM_REQUESTS:-$(awk -v rate="$RATE" -v seconds="$DURATION_SECONDS" 'BEGIN {printf "%d", rate * seconds}')}"
MOE_ROUTING_SEED="${MOE_ROUTING_SEED:-0}"
ROUNDS="${ROUNDS:-2}"
DRY_RUN="${DRY_RUN:-0}"

RUN_TAG="${RUN_TAG:-two_node_r50_uniform_random_2round_4way_$(date -u +%Y%m%d_%H%M%S)}"
COMPARE_LOG_DIR="${COMPARE_LOG_DIR:-$ROOT_DIR/bench_logs/$RUN_TAG}"
PROGRESS_LOG="$COMPARE_LOG_DIR/compare.progress"

export SLIME_VISIBLE_DEVICES="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
export SLIME_GID_INDEX=3
export SLIME_QP_NUM=4
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

log() {
    local now
    now="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "[$now] $*" | tee -a "$PROGRESS_LOG"
}

print_command() {
    local stage_log_dir="$1"
    local worker_transport="$2"
    shift 2
    {
        printf "env BASE_LOG_DIR=%q " "$stage_log_dir"
        printf "NANODEPLOY_HIER_WORKER_TRANSPORT=%q " "$worker_transport"
        printf "%q " "$@"
        printf "\n"
    } | tee -a "$PROGRESS_LOG"
}

for value_name in DURATION_SECONDS NUM_REQUESTS ROUNDS; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: $value_name must be a positive integer." >&2
        exit 2
    fi
done
if ! [[ "$RATE" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$RATE" == "0" ]]; then
    echo "Error: RATE must be a positive number." >&2
    exit 2
fi
if ! [[ "$MOE_ROUTING_SEED" =~ ^-?[0-9]+$ ]]; then
    echo "Error: MOE_ROUTING_SEED must be an integer." >&2
    exit 2
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "Error: DRY_RUN must be 0 or 1." >&2
    exit 2
fi
if [[ ! -f "$START_BENCH_SH" ]]; then
    echo "Error: start_bench.sh not found: $START_BENCH_SH" >&2
    exit 2
fi
if [[ "$DRY_RUN" == "0" ]]; then
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo "Error: model directory not found: $MODEL_PATH" >&2
        exit 2
    fi
    if [[ ! -f "$DATASET_PATH" ]]; then
        echo "Error: dataset not found: $DATASET_PATH" >&2
        exit 2
    fi
fi

mkdir -p "$COMPARE_LOG_DIR"

COMMON_ARGS=(
    --master-addr "$MASTER_ADDR"
    --ray-addr "$RAY_ADDR"
    --dataset-path "$DATASET_PATH"
    --model-path "$MODEL_PATH"
    --segment-size 65536
    --dp-size 2
    --sp-size 8
    --tp-size 1
    --batch-size 192
    --num-requests "$NUM_REQUESTS"
    --gpu-mem 141
    --gpu-util 0.9
    --max-model-len 1000000
    --max-request-tokens 910000
    --sp-master-selector LeastBatch
    --loop-count 16
    --fixed-sp-size 0
    --sp-backend hao_basic
    --cuda-graph-mode full
    --dynamic-sp-size-strategy legacy
    --dynamic-sp-bucket-preset none
    --moe-routing-simulation-strategy uniform_random
    --moe-routing-seed "$MOE_ROUTING_SEED"
    --diagnostic-log-interval 0
    --slow-add-threshold-ms 20
    --quantum-diagnostics
)

run_stage() {
    local stage="$1"
    local scheduler_arch="$2"
    local router_policy="$3"
    local routing_strategy="$4"
    local worker_transport="$5"
    local run_label="$6"
    local stage_log_dir="$COMPARE_LOG_DIR/$stage"
    local cmd=(
        bash "$START_BENCH_SH"
        "${COMMON_ARGS[@]}"
        --scheduler-arch "$scheduler_arch"
        --router-policy "$router_policy"
        --routing-strategy "$routing_strategy"
        --run-label "$run_label"
        "$RATE"
    )

    log "STAGE=$stage SCHEDULER_ARCH=$scheduler_arch ROUTING_STRATEGY=$routing_strategy ROUTER_POLICY=$router_policy WORKER_TRANSPORT=$worker_transport"
    if [[ "$DRY_RUN" == "1" ]]; then
        print_command "$stage_log_dir" "$worker_transport" "${cmd[@]}"
        return
    fi

    env \
        BASE_LOG_DIR="$stage_log_dir" \
        NANODEPLOY_HIER_WORKER_TRANSPORT="$worker_transport" \
        "${cmd[@]}"
    log "DONE stage=$stage"
}

log "RUN_TAG=$RUN_TAG"
log "RAY_ADDR=$RAY_ADDR MASTER_ADDR=$MASTER_ADDR"
log "MODEL_PATH=$MODEL_PATH"
log "DATASET_PATH=$DATASET_PATH"
log "RATE=$RATE DURATION_SECONDS=$DURATION_SECONDS NUM_REQUESTS=$NUM_REQUESTS"
log "ROUNDS=$ROUNDS EXPERIMENTS_PER_ROUND=4 TOTAL_EXPERIMENTS=$((ROUNDS * 4))"
log "MOE_ROUTING_SIMULATION_STRATEGY=uniform_random MOE_ROUTING_SEED=$MOE_ROUTING_SEED"
log "SLIME_VISIBLE_DEVICES=$SLIME_VISIBLE_DEVICES SLIME_GID_INDEX=$SLIME_GID_INDEX SLIME_QP_NUM=$SLIME_QP_NUM"

for ((round = 1; round <= ROUNDS; ++round)); do
    log "ROUND_START round=$round/$ROUNDS"

    run_stage \
        "round${round}_central_least_batch" \
        legacy_global \
        least_batch \
        LeastBatch \
        ray \
        "round${round}_central_least_batch_uniform_random"

    run_stage \
        "round${round}_hierarchical_least_batch" \
        hierarchical \
        least_batch \
        LeastBatch \
        zmq \
        "round${round}_hierarchical_zmq_least_batch_uniform_random"

    run_stage \
        "round${round}_central_least_projected_load" \
        legacy_global \
        least_projected_load \
        LeastProjectedLoad \
        ray \
        "round${round}_central_projected_load_uniform_random"

    run_stage \
        "round${round}_hierarchical_least_projected_load" \
        hierarchical \
        least_projected_load \
        LeastBatch \
        zmq \
        "round${round}_hierarchical_zmq_rppl_uniform_random"

    log "ROUND_DONE round=$round/$ROUNDS"
done

log "COMPARE_DONE log_dir=$COMPARE_LOG_DIR"
