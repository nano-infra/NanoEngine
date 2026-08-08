#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
case "$SCRIPT_PATH" in
  /*) ;;
  *) SCRIPT_PATH="$PWD/$SCRIPT_PATH" ;;
esac
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/asplos-camera-august/nano-gemm-debug-${RUN_TIMESTAMP}}"
RUN_DIR="$OUTPUT_ROOT/dp16_ep16_eager"
LOG_PATH="$RUN_DIR/run.log"
COMMAND_PATH="$RUN_DIR/command.txt"

MODEL_PATH="${MODEL_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/Kimi-K2-Instruct-0905}"
DLBLAS_ROOT="${DLBLAS_ROOT:-/mnt/nvme1n1/ml_research/linbinbin1/dlBLAS-aug}"
MASTER_ADDRESS="${MASTER_ADDRESS:-10.102.252.174:29500}"
RAY_ADDRESS="${RAY_ADDRESS:-10.102.252.174:7789}"

NUM_SEQS="${NUM_SEQS:-512}"
SEQ_LEN="${SEQ_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_SEND_SEQS="${MAX_NUM_SEND_SEQS:-32}"
MAX_NUM_RECV_SEQS="${MAX_NUM_RECV_SEQS:-34}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
DRY_RUN="${DRY_RUN:-0}"

# The default produces one JSON record per rank for the first routed MoE layer.
export DLBLAS_MOE_GEMM_DEBUG="${DLBLAS_MOE_GEMM_DEBUG:-1}"
export DLBLAS_MOE_GEMM_DEBUG_RANKS="${DLBLAS_MOE_GEMM_DEBUG_RANKS:-all}"
export DLBLAS_MOE_GEMM_DEBUG_LAYERS="${DLBLAS_MOE_GEMM_DEBUG_LAYERS:-1}"
export DLBLAS_MOE_GEMM_DEBUG_GEMMS="${DLBLAS_MOE_GEMM_DEBUG_GEMMS:-gate_up}"
export DLBLAS_MOE_GEMM_DEBUG_MAX_CALLS="${DLBLAS_MOE_GEMM_DEBUG_MAX_CALLS:-1}"
export DLBLAS_MOE_GEMM_DEBUG_SAMPLE_ELEMENTS="${DLBLAS_MOE_GEMM_DEBUG_SAMPLE_ELEMENTS:-8}"

export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH:-}"
export LD_LIBRARY_PATH="$REPO_ROOT/build/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/build/lib:$DLBLAS_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export SLIME_VISIBLE_DEVICES="${SLIME_VISIBLE_DEVICES:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export SLIME_GID_INDEX="${SLIME_GID_INDEX:-3}"
export SLIME_QP_NUM="${SLIME_QP_NUM:-4}"

# Ray Client and GCS traffic must bypass HTTP proxies.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

if [[ ! -d "$MODEL_PATH" ]]; then
  printf 'MODEL_PATH not found: %s\n' "$MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$DLBLAS_ROOT/dlblas/layers/moe/ep_moe.py" ]]; then
  printf 'Instrumented dlBLAS checkout not found: %s\n' "$DLBLAS_ROOT" >&2
  exit 1
fi
if [[ "$NUM_SEQS" -ne 512 || "$MAX_NUM_SEQS" -ne 32 ]]; then
  printf 'This check requires NUM_SEQS=512 and MAX_NUM_SEQS=32; got %s and %s\n' \
    "$NUM_SEQS" "$MAX_NUM_SEQS" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  printf 'Refusing to overwrite existing debug run: %s\n' "$RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

cmd=(
  python3 -u "$REPO_ROOT/examples/dummy_prefill.py"
  --num-seqs "$NUM_SEQS"
  --seq-len "$SEQ_LEN"
  --max-tokens 1
  --max-num-seqs "$MAX_NUM_SEQS"
  --dp 16
  --sp 1
  --ep 16
  --model-path "$MODEL_PATH"
  --master-address "$MASTER_ADDRESS"
  --ray-address "$RAY_ADDRESS"
  --max-num-send-seqs "$MAX_NUM_SEND_SEQS"
  --max-num-recv-seqs "$MAX_NUM_RECV_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --loop-count 1
  --num-steps 1
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --routing-strategy LeastBatch
  --sp-backend hao_basic
  --sp-size-policy legacy
  --segment-size 65536
  --long-request-sp-threshold 100000
  --enable-non-uniform-split
  --enforce-eager
)

{
  printf 'DLBLAS_MOE_GEMM_DEBUG=%q ' "$DLBLAS_MOE_GEMM_DEBUG"
  printf 'DLBLAS_MOE_GEMM_DEBUG_RANKS=%q ' "$DLBLAS_MOE_GEMM_DEBUG_RANKS"
  printf 'DLBLAS_MOE_GEMM_DEBUG_LAYERS=%q ' "$DLBLAS_MOE_GEMM_DEBUG_LAYERS"
  printf 'DLBLAS_MOE_GEMM_DEBUG_GEMMS=%q ' "$DLBLAS_MOE_GEMM_DEBUG_GEMMS"
  printf 'DLBLAS_MOE_GEMM_DEBUG_MAX_CALLS=%q ' "$DLBLAS_MOE_GEMM_DEBUG_MAX_CALLS"
  printf 'DLBLAS_MOE_GEMM_DEBUG_SAMPLE_ELEMENTS=%q ' "$DLBLAS_MOE_GEMM_DEBUG_SAMPLE_ELEMENTS"
  printf 'SLIME_QP_NUM=%q' "$SLIME_QP_NUM"
  printf ' %q' "${cmd[@]}"
  printf '\n'
} | tee "$COMMAND_PATH"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'Dry run completed: %s\n' "$RUN_DIR"
  exit 0
fi

"${cmd[@]}" 2>&1 | tee "$LOG_PATH"

printf 'GEMM input records:\n'
grep -F '[DLBLAS_MOE_GEMM_INPUT]' "$LOG_PATH" || true
printf 'Debug run completed: %s\n' "$RUN_DIR"
