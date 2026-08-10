#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
START_BENCH_SH="$SCRIPT_DIR/start_bench.sh"

# Four-node Ray cluster: 4 DP groups * 8 SP ranks = 32 GPUs.
RAY_ADDR="${RAY_ADDR:-10.102.252.174:6380}"
MASTER_ADDR="${MASTER_ADDR:-10.102.252.174:29500}"

MODEL_PATH="${MODEL_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/Kimi-K2-Instruct-0905}"
DATASET_PATH="${DATASET_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv}"

REQUEST_RATES="${REQUEST_RATES:-10 15 20 30 40}"
SEND_DURATION_SEC="${SEND_DURATION_SEC:-600}"

DP_SIZE="${DP_SIZE:-4}"
SP_SIZE="${SP_SIZE:-8}"
TP_SIZE="${TP_SIZE:-1}"
SEGMENT_SIZE="${SEGMENT_SIZE:-65536}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GPU_MEM_GB="${GPU_MEM_GB:-141}"
GPU_UTIL="${GPU_UTIL:-0.9}"

# Keep the previous end-to-end length limits; these are independent of the
# scheduler/backend settings taken from the new launch command.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-1000000}"
MAX_REQUEST_TOKENS="${MAX_REQUEST_TOKENS:-910000}"

ROUTING_STRATEGY="${ROUTING_STRATEGY:-LeastBatch}"
SCHEDULER_ARCH="${SCHEDULER_ARCH:-legacy_global}"
LOOP_COUNT="${LOOP_COUNT:-16}"
FIXED_SP_SIZE="${FIXED_SP_SIZE:-0}"
SP_BACKEND="${SP_BACKEND:-hao_basic}"
CUDA_GRAPH_MODE="${CUDA_GRAPH_MODE:-full}"
DYNAMIC_SP_SIZE_STRATEGY="${DYNAMIC_SP_SIZE_STRATEGY:-long_short_sp8}"
LONG_REQUEST_SP_THRESHOLD="${LONG_REQUEST_SP_THRESHOLD:-100000}"
LONG_REQUEST_SP_SIZE="${LONG_REQUEST_SP_SIZE:-8}"

RUN_TAG="${RUN_TAG:-kimi_issue001_4node_dp4sp8_longshort_600s_$(date -u +%Y%m%d_%H%M%S)}"
E2E_LOG_ROOT="${E2E_LOG_ROOT:-$REPO_ROOT/bench_logs/e2e}"
OUTPUT_DIR="${OUTPUT_DIR:-$E2E_LOG_ROOT/$RUN_TAG}"
DRY_RUN="${DRY_RUN:-0}"

log() {
    local message="$*"
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$message" \
        | tee -a "$PROGRESS_LOG"
}

rate_to_num_requests() {
    local rate="$1"
    awk -v rate="$rate" -v duration="$SEND_DURATION_SEC" \
        'BEGIN {printf "%d", int(rate * duration + 0.5)}'
}

if [[ "$DP_SIZE" -ne 4 || "$SP_SIZE" -ne 8 || "$TP_SIZE" -ne 1 ]]; then
    printf 'This launch requires DP_SIZE=4, SP_SIZE=8, TP_SIZE=1; got %s/%s/%s\n' \
        "$DP_SIZE" "$SP_SIZE" "$TP_SIZE" >&2
    exit 1
fi
if ! awk -v duration="$SEND_DURATION_SEC" 'BEGIN {exit !(duration > 0)}'; then
    printf 'SEND_DURATION_SEC must be positive; got %s\n' "$SEND_DURATION_SEC" >&2
    exit 1
fi
if [[ ! -f "$START_BENCH_SH" ]]; then
    printf 'Missing benchmark launcher: %s\n' "$START_BENCH_SH" >&2
    exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    printf 'Missing model directory: %s\n' "$MODEL_PATH" >&2
    exit 1
fi
if [[ ! -f "$DATASET_PATH" ]]; then
    printf 'Missing dataset: %s\n' "$DATASET_PATH" >&2
    exit 1
fi

read -r -a rate_list <<< "$REQUEST_RATES"
if [[ ${#rate_list[@]} -eq 0 ]]; then
    printf 'REQUEST_RATES must contain at least one request rate.\n' >&2
    exit 1
fi
for rate in "${rate_list[@]}"; do
    if ! awk -v rate="$rate" 'BEGIN {exit !(rate > 0)}'; then
        printf 'Every request rate must be positive; got %s\n' "$rate" >&2
        exit 1
    fi
done

if [[ -e "$OUTPUT_DIR" ]]; then
    printf 'Refusing to overwrite existing output directory: %s\n' "$OUTPUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

PROGRESS_LOG="$OUTPUT_DIR/run.progress"
SUMMARY_FILE="$OUTPUT_DIR/run_summary.tsv"
CONFIG_FILE="$OUTPUT_DIR/config.txt"

printf 'rate\tnum_requests\tstatus\tlog_dir\n' > "$SUMMARY_FILE"
{
    printf 'RUN_TAG=%q\n' "$RUN_TAG"
    printf 'RAY_ADDR=%q\n' "$RAY_ADDR"
    printf 'MASTER_ADDR=%q\n' "$MASTER_ADDR"
    printf 'MODEL_PATH=%q\n' "$MODEL_PATH"
    printf 'DATASET_PATH=%q\n' "$DATASET_PATH"
    printf 'REQUEST_RATES=%q\n' "$REQUEST_RATES"
    printf 'SEND_DURATION_SEC=%q\n' "$SEND_DURATION_SEC"
    printf 'TOPOLOGY=dp%q_sp%q_tp%q\n' "$DP_SIZE" "$SP_SIZE" "$TP_SIZE"
    printf 'SP_BACKEND=%q\n' "$SP_BACKEND"
    printf 'DYNAMIC_SP_SIZE_STRATEGY=%q\n' "$DYNAMIC_SP_SIZE_STRATEGY"
    printf 'OUTPUT_DIR=%q\n' "$OUTPUT_DIR"
} > "$CONFIG_FILE"

# Ray Client/GCS traffic and the benchmark driver must bypass HTTP proxies.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# TCP/bootstrap network.
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"

# NCCL RDMA data plane.
export NCCL_IB_HCA="${NCCL_IB_HCA:-=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_TC="${NCCL_IB_TC:-186}"

# DeepEP/NVSHMEM RDMA.
export SLIME_VISIBLE_DEVICES="${SLIME_VISIBLE_DEVICES:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export SLIME_GID_INDEX="${SLIME_GID_INDEX:-3}"
export SLIME_QP_NUM="${SLIME_QP_NUM:-4}"

# Native DeepEP backend settings.
export DEEPEP_SMS="${DEEPEP_SMS:-16}"
export DEEPEP_MAX_TOKENS_PER_RANK="${DEEPEP_MAX_TOKENS_PER_RANK:-256}"
export DEEPEP_ENABLE_MNNVL="${DEEPEP_ENABLE_MNNVL:-0}"
export DEEPEP_MODE="${DEEPEP_MODE:-auto}"
export NVSHMEM_QP_DEPTH="${NVSHMEM_QP_DEPTH:-1024}"

export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NANODEPLOY_LOG_DECODE_STEP_DETAIL="${NANODEPLOY_LOG_DECODE_STEP_DETAIL:-1}"

# Keep DeepGEMM/JIT input debugging disabled for this long sweep.
unset DG_PRINT_CONFIGS DG_JIT_DEBUG
unset NANODEPLOY_MOE_GEMM_DEBUG
unset NANODEPLOY_MOE_GEMM_DEBUG_RANKS
unset NANODEPLOY_MOE_GEMM_DEBUG_LAYERS
unset NANODEPLOY_MOE_GEMM_DEBUG_GEMMS
unset NANODEPLOY_MOE_GEMM_DEBUG_MAX_CALLS
unset NANODEPLOY_MOE_GEMM_DEBUG_SAMPLE_ELEMENTS

log "START run_tag=$RUN_TAG rates=$REQUEST_RATES duration=${SEND_DURATION_SEC}s topology=dp${DP_SIZE}sp${SP_SIZE}tp${TP_SIZE}"
log "RAY_ADDR=$RAY_ADDR MASTER_ADDR=$MASTER_ADDR output_dir=$OUTPUT_DIR"

for rate in "${rate_list[@]}"; do
    num_requests="$(rate_to_num_requests "$rate")"
    rate_tag="${rate//./p}"
    rate_log_dir="$OUTPUT_DIR/rate_${rate_tag}"
    bench_log_dir="$rate_log_dir/bench"
    driver_log="$rate_log_dir/driver.log"
    command_file="$rate_log_dir/command.txt"
    mkdir -p "$bench_log_dir"

    cmd=(
        bash "$START_BENCH_SH"
        --ray-addr "$RAY_ADDR"
        --master-addr "$MASTER_ADDR"
        --model-path "$MODEL_PATH"
        --dataset-path "$DATASET_PATH"
        --dp-size "$DP_SIZE"
        --sp-size "$SP_SIZE"
        --tp-size "$TP_SIZE"
        --segment-size "$SEGMENT_SIZE"
        --block-size "$BLOCK_SIZE"
        --batch-size "$BATCH_SIZE"
        --num-requests "$num_requests"
        --gpu-mem "$GPU_MEM_GB"
        --gpu-util "$GPU_UTIL"
        --max-model-len "$MAX_MODEL_LEN"
        --max-input-len "$MAX_INPUT_LEN"
        --max-request-tokens "$MAX_REQUEST_TOKENS"
        --routing-strategy "$ROUTING_STRATEGY"
        --scheduler-arch "$SCHEDULER_ARCH"
        --loop-count "$LOOP_COUNT"
        --fixed-sp-size "$FIXED_SP_SIZE"
        --sp-backend "$SP_BACKEND"
        --cuda-graph-mode "$CUDA_GRAPH_MODE"
        --enable-dynamic-sp-size
        --dynamic-sp-size-strategy "$DYNAMIC_SP_SIZE_STRATEGY"
        --long-request-sp-threshold "$LONG_REQUEST_SP_THRESHOLD"
        --long-request-sp-size "$LONG_REQUEST_SP_SIZE"
        --run-label "600s_r${rate_tag}"
        "$rate"
    )

    {
        printf 'BASE_LOG_DIR=%q' "$bench_log_dir"
        printf ' %q' "${cmd[@]}"
        printf '\n'
    } > "$command_file"

    log "RUN rate=$rate num_requests=$num_requests log_dir=$rate_log_dir"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY_RUN: '
        cat "$command_file"
        printf '%s\t%s\tplanned\t%s\n' \
            "$rate" "$num_requests" "$rate_log_dir" >> "$SUMMARY_FILE"
        continue
    fi

    set +e
    BASE_LOG_DIR="$bench_log_dir" "${cmd[@]}" 2>&1 | tee "$driver_log"
    run_status=${PIPESTATUS[0]}
    set -e

    # start_bench.sh records the child status but currently returns zero after
    # its loop, so promote a recorded child failure to this wrapper's status.
    if [[ -f "$bench_log_dir/run_progress.log" ]] \
        && grep -Fq "Finished Rate: $rate. Status: FAILED" "$bench_log_dir/run_progress.log"; then
        run_status=1
    fi

    if [[ "$run_status" -ne 0 ]]; then
        printf '%s\t%s\tfailed\t%s\n' \
            "$rate" "$num_requests" "$rate_log_dir" >> "$SUMMARY_FILE"
        log "FAILED rate=$rate status=$run_status; stopping sweep"
        exit "$run_status"
    fi

    printf '%s\t%s\tok\t%s\n' \
        "$rate" "$num_requests" "$rate_log_dir" >> "$SUMMARY_FILE"
    log "DONE rate=$rate num_requests=$num_requests"
done

if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN complete; no benchmark was launched"
else
    log "COMPLETE all rates finished and all submitted requests drained"
fi
