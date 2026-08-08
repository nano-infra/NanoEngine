#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
case "$SCRIPT_PATH" in
    /*) ;;
    *) SCRIPT_PATH="$PWD/$SCRIPT_PATH" ;;
esac
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
START_BENCH_SH="$SCRIPT_DIR/start_bench.sh"

RAY_ADDR="${RAY_ADDR:-10.102.252.174:7799}"
MASTER_ADDR="${MASTER_ADDR:-10.102.252.174:29500}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/Kimi-K2-Instruct-0905}"
DATASET_PATH="${DATASET_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv}"

REQUEST_RATE="${REQUEST_RATE:-5}"
SEND_DURATION_SEC="${SEND_DURATION_SEC:-60}"
DP_SIZE="${DP_SIZE:-16}"
SP_SIZE="${SP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
SEGMENT_SIZE="${SEGMENT_SIZE:-65536}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GPU_MEM_GB="${GPU_MEM_GB:-141}"
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-1000000}"
LOOP_COUNT="${LOOP_COUNT:-16}"
ROUTING_STRATEGY="${ROUTING_STRATEGY:-LeastBatch}"
SCHEDULER_ARCH="${SCHEDULER_ARCH:-legacy_global}"
DRY_RUN="${DRY_RUN:-0}"

if ! awk -v value="$REQUEST_RATE" 'BEGIN {exit !(value > 0)}'; then
    printf 'REQUEST_RATE must be positive; got %s\n' "$REQUEST_RATE" >&2
    exit 1
fi
if ! awk -v value="$SEND_DURATION_SEC" 'BEGIN {exit !(value > 0)}'; then
    printf 'SEND_DURATION_SEC must be positive; got %s\n' "$SEND_DURATION_SEC" >&2
    exit 1
fi

# The issue003 benchmark submits a fixed request count according to Poisson
# arrival times, then keeps stepping the engine until every request finishes.
NUM_REQUESTS="${NUM_REQUESTS:-$(
    awk -v rate="$REQUEST_RATE" -v duration="$SEND_DURATION_SEC" \
        'BEGIN {printf "%d", int(rate * duration + 0.5)}'
)}"
if ! [[ "$NUM_REQUESTS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'NUM_REQUESTS must be a positive integer; got %s\n' "$NUM_REQUESTS" >&2
    exit 1
fi
if [[ "$DP_SIZE" -ne 16 || "$SP_SIZE" -ne 1 || "$TP_SIZE" -ne 1 ]]; then
    printf 'This debug run requires DP_SIZE=16, SP_SIZE=1, TP_SIZE=1; got %s/%s/%s\n' \
        "$DP_SIZE" "$SP_SIZE" "$TP_SIZE" >&2
    exit 1
fi

RATE_TAG="${REQUEST_RATE//./p}"
RUN_TAG="${RUN_TAG:-kimi_issue001_2node_dp16_ep16_deepgemm_eager_${SEND_DURATION_SEC}s_r${RATE_TAG}_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/bench_logs/$RUN_TAG}"
BENCH_LOG_DIR="$OUTPUT_DIR/bench"
RUN_LOG="$OUTPUT_DIR/run.log"
COMMAND_PATH="$OUTPUT_DIR/command.txt"
DEEPGEMM_CONFIG_LOG="$OUTPUT_DIR/deepgemm_configs.log"
MOE_INPUT_LOG="$OUTPUT_DIR/deepgemm_inputs.jsonl"

if [[ -e "$OUTPUT_DIR" ]]; then
    printf 'Refusing to overwrite existing output directory: %s\n' "$OUTPUT_DIR" >&2
    exit 1
fi
mkdir -p "$BENCH_LOG_DIR"

export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH:-}"
export LD_LIBRARY_PATH="$REPO_ROOT/build/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/build/lib${PYTHONPATH:+:$PYTHONPATH}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export SLIME_VISIBLE_DEVICES="${SLIME_VISIBLE_DEVICES:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export SLIME_GID_INDEX="${SLIME_GID_INDEX:-3}"
export SLIME_QP_NUM="${SLIME_QP_NUM:-4}"

# Match max_num_seqs so the decode-only DeepEP buffers can hold a full batch.
export DEEPEP_SMS="${DEEPEP_SMS:-16}"
export DEEPEP_MAX_TOKENS_PER_RANK="${DEEPEP_MAX_TOKENS_PER_RANK:-$BATCH_SIZE}"
export DEEPEP_ENABLE_MNNVL="${DEEPEP_ENABLE_MNNVL:-0}"
export DEEPEP_MODE="${DEEPEP_MODE:-auto}"

# DeepGEMM prints each newly observed shape and the selected kernel config.
export DG_PRINT_CONFIGS="${DG_PRINT_CONFIGS:-1}"
export DG_JIT_DEBUG="${DG_JIT_DEBUG:-0}"

# Trace the per-call masked grouped GEMM inputs in first-seen order. Sampling is
# disabled to keep the log focused on shape metadata. Set MAX_CALLS above zero
# to cap the number of records per rank/layer/GEMM when a smaller log is needed.
export NANODEPLOY_MOE_GEMM_DEBUG="${NANODEPLOY_MOE_GEMM_DEBUG:-1}"
export NANODEPLOY_MOE_GEMM_DEBUG_RANKS="${NANODEPLOY_MOE_GEMM_DEBUG_RANKS:-all}"
export NANODEPLOY_MOE_GEMM_DEBUG_LAYERS="${NANODEPLOY_MOE_GEMM_DEBUG_LAYERS:-1}"
export NANODEPLOY_MOE_GEMM_DEBUG_GEMMS="${NANODEPLOY_MOE_GEMM_DEBUG_GEMMS:-gate_up}"
export NANODEPLOY_MOE_GEMM_DEBUG_MAX_CALLS="${NANODEPLOY_MOE_GEMM_DEBUG_MAX_CALLS:-0}"
export NANODEPLOY_MOE_GEMM_DEBUG_SAMPLE_ELEMENTS="${NANODEPLOY_MOE_GEMM_DEBUG_SAMPLE_ELEMENTS:-0}"

# Ray Client and GCS traffic must bypass HTTP proxies.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

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
    --num-requests "$NUM_REQUESTS"
    --gpu-mem "$GPU_MEM_GB"
    --gpu-util "$GPU_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --max-input-len "$MAX_INPUT_LEN"
    --routing-strategy "$ROUTING_STRATEGY"
    --scheduler-arch "$SCHEDULER_ARCH"
    --loop-count "$LOOP_COUNT"
    --fixed-sp-size 0
    --sp-backend hao_basic
    --cuda-graph-mode full
    --dynamic-sp-size-strategy legacy
    --enforce-eager
    --run-label deepgemm_shapes_1min_eager
    "$REQUEST_RATE"
)

{
    printf 'RUN_TAG=%q\n' "$RUN_TAG"
    printf 'OUTPUT_DIR=%q\n' "$OUTPUT_DIR"
    printf 'SEND_DURATION_SEC=%q\n' "$SEND_DURATION_SEC"
    printf 'REQUEST_RATE=%q\n' "$REQUEST_RATE"
    printf 'NUM_REQUESTS=%q\n' "$NUM_REQUESTS"
    printf 'TOPOLOGY=dp%q_sp%q_tp%q_ep16\n' "$DP_SIZE" "$SP_SIZE" "$TP_SIZE"
    printf 'DG_PRINT_CONFIGS=%q\n' "$DG_PRINT_CONFIGS"
    printf 'DG_JIT_DEBUG=%q\n' "$DG_JIT_DEBUG"
    printf 'NANODEPLOY_MOE_GEMM_DEBUG=%q\n' "$NANODEPLOY_MOE_GEMM_DEBUG"
    printf 'NANODEPLOY_MOE_GEMM_DEBUG_RANKS=%q\n' "$NANODEPLOY_MOE_GEMM_DEBUG_RANKS"
    printf 'NANODEPLOY_MOE_GEMM_DEBUG_LAYERS=%q\n' "$NANODEPLOY_MOE_GEMM_DEBUG_LAYERS"
    printf 'NANODEPLOY_MOE_GEMM_DEBUG_GEMMS=%q\n' "$NANODEPLOY_MOE_GEMM_DEBUG_GEMMS"
    printf 'NANODEPLOY_MOE_GEMM_DEBUG_MAX_CALLS=%q\n' "$NANODEPLOY_MOE_GEMM_DEBUG_MAX_CALLS"
    printf 'NANODEPLOY_MOE_GEMM_DEBUG_SAMPLE_ELEMENTS=%q\n' "$NANODEPLOY_MOE_GEMM_DEBUG_SAMPLE_ELEMENTS"
    printf 'BASE_LOG_DIR=%q' "$BENCH_LOG_DIR"
    printf ' %q' "${cmd[@]}"
    printf '\n'
} | tee "$COMMAND_PATH"

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'Dry run completed: %s\n' "$OUTPUT_DIR"
    exit 0
fi

set +e
BASE_LOG_DIR="$BENCH_LOG_DIR" "${cmd[@]}" 2>&1 | tee "$RUN_LOG"
run_status=${PIPESTATUS[0]}
set -e

# start_bench.sh records a child failure but currently completes its own loop;
# promote that recorded status so this wrapper still exits nonzero.
if [[ -f "$BENCH_LOG_DIR/run_progress.log" ]] \
    && grep -Fq 'Status: FAILED' "$BENCH_LOG_DIR/run_progress.log"; then
    run_status=1
fi

grep -F 'GEMM type:' "$RUN_LOG" > "$DEEPGEMM_CONFIG_LOG" || true
sed -n 's/^.*\[NANODEPLOY_MOE_GEMM_INPUT\] //p' "$RUN_LOG" > "$MOE_INPUT_LOG"

config_count="$(wc -l < "$DEEPGEMM_CONFIG_LOG")"
input_count="$(wc -l < "$MOE_INPUT_LOG")"
printf 'DeepGEMM config records: %s (%s)\n' "$config_count" "$DEEPGEMM_CONFIG_LOG"
printf 'Per-call GEMM input records: %s (%s)\n' "$input_count" "$MOE_INPUT_LOG"

if [[ "$run_status" -ne 0 ]]; then
    printf 'Benchmark failed with status %s; partial logs remain in %s\n' \
        "$run_status" "$OUTPUT_DIR" >&2
    exit "$run_status"
fi

printf 'Benchmark and request drain completed: %s\n' "$OUTPUT_DIR"
