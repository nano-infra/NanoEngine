#!/usr/bin/env bash

# Run and validate the 30 FlashMLA cases used by Figure 3 panel (a).
# This script measures Attention only; Figure 3 plotting is a separate step.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FIG3_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
AE_ROOT="$(cd -- "${FIG3_DIR}/.." && pwd)"
BENCHMARK_PY="${AE_ROOT}/microbench/attention/benchmark_flashmla.py"
VALIDATOR_PY="${SCRIPT_DIR}/validate_mla_benchmark_csv.py"

GPU_ID="${CUDA_DEVICE:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${RESULT_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TOTAL_TOKENS="65536,131072,196608,262144,393216,524288,655360,786432,917504,1048576"
BATCH_SIZES="1,128,1024"
NUM_HEADS=128
HEAD_DIM=576
V_HEAD_DIM=512
REP_MS=100
BENCH_REPEATS=5
WARMUP=10

usage() {
  cat <<'EOF'
Usage: ./run_attention.sh [options]

Run Figure 3 Attention cases with the external flash_mla package. The default
arguments run all 30 paper cases. No benchmark cache is read or accepted.

Options:
  --gpu ID          Physical GPU ID exposed through CUDA_VISIBLE_DEVICES (default: 0)
  --run-id ID       Output suffix (default: current YYYYmmdd_HHMMSS)
  --output-dir DIR  Result directory (default: fig3/results/attention/${RUN_ID}/)
  --python PATH     Python command (default: python3)
  --total-tokens N  Comma-separated total-token counts (default: full paper sweep)
  --batch-sizes N   Comma-separated batch sizes (default: 1,128,1024)
  -h, --help        Show this help

The same settings can be supplied through CUDA_DEVICE, RUN_ID, RESULT_DIR,
and PYTHON_BIN.
EOF
}

require_option_value() {
  if [[ $# -lt 2 || -z "${2}" ]]; then
    echo "Error: $1 requires a value." >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      require_option_value "$@"
      GPU_ID="$2"
      shift 2
      ;;
    --run-id)
      require_option_value "$@"
      RUN_ID="$2"
      shift 2
      ;;
    --output-dir)
      require_option_value "$@"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --python)
      require_option_value "$@"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --total-tokens)
      require_option_value "$@"
      TOTAL_TOKENS="$2"
      shift 2
      ;;
    --batch-sizes)
      require_option_value "$@"
      BATCH_SIZES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

IFS=',' read -r -a TOTAL_TOKEN_VALUES <<< "$TOTAL_TOKENS"
IFS=',' read -r -a BATCH_SIZE_VALUES <<< "$BATCH_SIZES"
if (( ${#TOTAL_TOKEN_VALUES[@]} == 0 || ${#BATCH_SIZE_VALUES[@]} == 0 )); then
  echo "Error: --total-tokens and --batch-sizes must not be empty." >&2
  exit 2
fi
EXPECTED_CASE_COUNT=$((${#TOTAL_TOKEN_VALUES[@]} * ${#BATCH_SIZE_VALUES[@]}))

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${FIG3_DIR}/results/attention/${RUN_ID}"
fi

if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "Error: --gpu must be one non-negative integer GPU ID." >&2
  exit 2
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Error: --run-id may only contain letters, digits, dot, underscore, and hyphen." >&2
  exit 2
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "Error: Python not found: $PYTHON_BIN" >&2
  exit 2
}

for required_path in "$BENCHMARK_PY" "$VALIDATOR_PY"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Error: required benchmark artifact is missing: $required_path" >&2
    exit 2
  fi
done

# Required by the NanoDeploy experiment environment and harmless for this
# single-process kernel microbenchmark.
export SLIME_QP_NUM=4

mkdir -p "$OUTPUT_DIR"

CSV_PATH="${OUTPUT_DIR}/external_flashmla_cudagraph_total_tokens_${RUN_ID}.csv"
LOG_PATH="${OUTPUT_DIR}/external_flashmla_cudagraph_total_tokens_${RUN_ID}.log"
ENV_PATH="${OUTPUT_DIR}/external_flashmla_cudagraph_total_tokens_${RUN_ID}_environment.txt"

for path in "$CSV_PATH" "$LOG_PATH" "$ENV_PATH"; do
  if [[ -e "$path" ]]; then
    echo "Error: refusing to overwrite existing output: $path" >&2
    echo "Choose a different --run-id or --output-dir." >&2
    exit 2
  fi
done

echo "Preflight: checking the Attention benchmark environment."
if ! CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" - <<'PY'
import importlib.metadata
from pathlib import Path

import flash_mla
import torch
import triton

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
for name in ("get_mla_metadata", "flash_mla_with_kvcache"):
    if not callable(getattr(flash_mla, name, None)):
        raise SystemExit(f"external flash_mla is missing callable: {name}")
torch.cuda.set_device(0)
try:
    flashmla_version = importlib.metadata.version("flash_mla")
except importlib.metadata.PackageNotFoundError:
    flashmla_version = "unknown"
print(f"PyTorch: {torch.__version__}")
print(f"Triton: {triton.__version__}")
print(f"external flash_mla: {flashmla_version}")
print(f"external flash_mla module: {Path(flash_mla.__file__).resolve()}")
print(f"Visible GPU: {torch.cuda.get_device_name(0)}")
PY
then
  echo "Error: the benchmark environment needs CUDA-enabled PyTorch, Triton, and external flash_mla." >&2
  exit 2
fi

{
  echo "run_id=${RUN_ID}"
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "working_directory=${SCRIPT_DIR}"
  echo "cuda_visible_devices=${GPU_ID}"
  echo "total_tokens=${TOTAL_TOKENS}"
  echo "batch_sizes=${BATCH_SIZES}"
  echo "num_heads=${NUM_HEADS}"
  echo "head_dim=${HEAD_DIM}"
  echo "v_head_dim=${V_HEAD_DIM}"
  echo "rep_ms=${REP_MS}"
  echo "bench_repeats=${BENCH_REPEATS}"
  echo "warmup=${WARMUP}"
  echo "python_bin=${PYTHON_BIN}"
  echo "slime_qp_num=${SLIME_QP_NUM}"
  echo
  "$PYTHON_BIN" - <<'PY'
import importlib.metadata
import pathlib
import platform
import sys

import flash_mla
import torch
import triton

try:
    flashmla_version = importlib.metadata.version("flash_mla")
except importlib.metadata.PackageNotFoundError:
    flashmla_version = "unknown"
print("platform=" + platform.platform())
print("python=" + sys.version.replace(chr(10), " "))
print("torch=" + torch.__version__)
print("torch_cuda=" + str(torch.version.cuda))
print("triton=" + triton.__version__)
print("flash_mla=" + flashmla_version)
print("flash_mla_module=" + str(pathlib.Path(flash_mla.__file__).resolve()))
print("flash_mla_operator_module=" + flash_mla.flash_mla_with_kvcache.__module__)
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --id="$GPU_ID" --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
  fi
} >"$ENV_PATH"

echo "Running ${EXPECTED_CASE_COUNT} external FlashMLA cases from scratch (cache disabled)."
echo "Benchmark CSV: $CSV_PATH"
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
  "$BENCHMARK_PY" \
  --total_tokens "$TOTAL_TOKENS" \
  --batch_sizes "$BATCH_SIZES" \
  --num_heads "$NUM_HEADS" \
  --head_dim "$HEAD_DIM" \
  --v_head_dim "$V_HEAD_DIM" \
  --rep-ms "$REP_MS" \
  --bench-repeats "$BENCH_REPEATS" \
  --warmup "$WARMUP" \
  --output "$CSV_PATH" \
  --skip-plot 2>&1 | tee "$LOG_PATH"

"$PYTHON_BIN" "$VALIDATOR_PY" \
  --total-tokens "$TOTAL_TOKENS" \
  --batch-sizes "$BATCH_SIZES" \
  "$CSV_PATH"

echo
echo "Attention measurement completed successfully."
echo "  CSV:         $CSV_PATH"
echo "  log:         $LOG_PATH"
echo "  environment: $ENV_PATH"
