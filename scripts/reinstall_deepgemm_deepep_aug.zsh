#!/usr/bin/env zsh

set -euo pipefail

NANODEPLOY_ML_ROOT="${NANODEPLOY_ML_ROOT:-/mnt/nvme1n1/ml_research/linbinbin1}"
DEEPGEMM_AUG_DIR="${DEEPGEMM_AUG_DIR:-${NANODEPLOY_ML_ROOT}/DeepGEMM-aug}"
DEEPEP_AUG_DIR="${DEEPEP_AUG_DIR:-${NANODEPLOY_ML_ROOT}/DeepEP-aug}"
NANODEPLOY_NVSHMEM_DIR="${NANODEPLOY_NVSHMEM_DIR:-/sgl-workspace/nvshmem/install}"
NANODEPLOY_PYTHON="${NANODEPLOY_PYTHON:-python}"
DEEP_BACKEND_MAX_JOBS="${DEEP_BACKEND_MAX_JOBS:-16}"

readonly DEEPGEMM_COMMIT=477618cd51baffca09c4b0b87e97c03fe827ef03
readonly DEEPEP_COMMIT=73b6ea4a439ba03a695563f9fd242c8e4b02b37c
readonly CUTLASS_COMMIT=f3fde58372d33e9a5650ba7b80fc48b3b49d40c8
readonly FMT_COMMIT=553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28

test "$(git -C "${DEEPGEMM_AUG_DIR}" rev-parse HEAD)" = "${DEEPGEMM_COMMIT}"
test "$(git -C "${DEEPEP_AUG_DIR}" rev-parse HEAD)" = "${DEEPEP_COMMIT}"
test -z "$(git -C "${DEEPGEMM_AUG_DIR}" status --porcelain --untracked-files=no)"
test -z "$(git -C "${DEEPEP_AUG_DIR}" status --porcelain --untracked-files=no)"

test "$(git -C "${DEEPGEMM_AUG_DIR}" rev-parse HEAD:third-party/cutlass)" = "${CUTLASS_COMMIT}"
test "$(git -C "${DEEPGEMM_AUG_DIR}/third-party/cutlass" rev-parse HEAD)" = "${CUTLASS_COMMIT}"
test "$(git -C "${DEEPGEMM_AUG_DIR}" rev-parse HEAD:third-party/fmt)" = "${FMT_COMMIT}"
test "$(git -C "${DEEPGEMM_AUG_DIR}/third-party/fmt" rev-parse HEAD)" = "${FMT_COMMIT}"

test -f "${DEEPGEMM_AUG_DIR}/third-party/cutlass/include/cutlass/arch/barrier.h"
test -f "${NANODEPLOY_NVSHMEM_DIR}/include/nvshmem.h"

DEEPGEMM_WHEEL_DIR="$(mktemp -d /tmp/nanodeploy-deep-gemm-2.3.0.XXXXXX)"
DEEPEP_WHEEL_DIR="$(mktemp -d /tmp/nanodeploy-deep-ep-1.2.1.XXXXXX)"

(
  cd "${DEEPGEMM_AUG_DIR}"
  "${NANODEPLOY_PYTHON}" setup.py clean --all
  DG_FORCE_BUILD=1 \
  DG_USE_LOCAL_VERSION=1 \
  MAX_JOBS="${DEEP_BACKEND_MAX_JOBS}" \
    "${NANODEPLOY_PYTHON}" setup.py bdist_wheel \
      --dist-dir "${DEEPGEMM_WHEEL_DIR}"
)

(
  cd "${DEEPEP_AUG_DIR}"
  env -u TOPK_IDX_BITS \
    NVSHMEM_DIR="${NANODEPLOY_NVSHMEM_DIR}" \
    TORCH_CUDA_ARCH_LIST=9.0 \
    "${NANODEPLOY_PYTHON}" setup.py clean --all
  env -u TOPK_IDX_BITS \
    NVSHMEM_DIR="${NANODEPLOY_NVSHMEM_DIR}" \
    TORCH_CUDA_ARCH_LIST=9.0 \
    MAX_JOBS="${DEEP_BACKEND_MAX_JOBS}" \
    "${NANODEPLOY_PYTHON}" setup.py bdist_wheel \
      --dist-dir "${DEEPEP_WHEEL_DIR}"
)

typeset -a DEEPGEMM_WHEELS
typeset -a DEEPEP_WHEELS
DEEPGEMM_WHEELS=("${DEEPGEMM_WHEEL_DIR}"/deep_gemm-2.3.0+477618c-*.whl(N))
DEEPEP_WHEELS=("${DEEPEP_WHEEL_DIR}"/deep_ep-1.2.1+73b6ea4-*.whl(N))

if (( ${#DEEPGEMM_WHEELS} != 1 )); then
  print -u2 -r -- "Expected exactly one DeepGEMM wheel under ${DEEPGEMM_WHEEL_DIR}"
  exit 1
fi
if (( ${#DEEPEP_WHEELS} != 1 )); then
  print -u2 -r -- "Expected exactly one DeepEP wheel under ${DEEPEP_WHEEL_DIR}"
  exit 1
fi

"${NANODEPLOY_PYTHON}" -m pip uninstall -y deep-gemm deep-ep

PIP_NO_INDEX=1 "${NANODEPLOY_PYTHON}" -m pip install \
  --no-deps "${DEEPGEMM_WHEELS[1]}"

PIP_NO_INDEX=1 "${NANODEPLOY_PYTHON}" -m pip install \
  --no-deps "${DEEPEP_WHEELS[1]}"

"${NANODEPLOY_PYTHON}" - <<'PY'
from importlib import metadata
from pathlib import Path

import torch
import deep_ep
import deep_gemm


assert metadata.version("deep_gemm") == "2.3.0+477618c"
assert metadata.version("deep_ep") == "1.2.1+73b6ea4"
assert deep_ep.topk_idx_t is torch.int64

deep_gemm_symbols = (
    "fp8_gemm_nt",
    "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nt_masked",
    "fp8_m_grouped_gemm_nt_masked",
)
deep_ep_symbols = (
    "Buffer",
    "Config",
    "EventOverlap",
)
deep_ep_buffer_symbols = (
    "destroy",
    "low_latency_dispatch",
    "low_latency_combine",
)

for symbol in deep_gemm_symbols:
    assert hasattr(deep_gemm, symbol), symbol
for symbol in deep_ep_symbols:
    assert hasattr(deep_ep, symbol), symbol
for symbol in deep_ep_buffer_symbols:
    assert hasattr(deep_ep.Buffer, symbol), symbol

barrier_header = (
    Path(deep_gemm.__file__).parent
    / "include/cutlass/arch/barrier.h"
)
assert barrier_header.is_file(), barrier_header

print("deep_gemm:", metadata.version("deep_gemm"), deep_gemm.__file__)
print("deep_ep:", metadata.version("deep_ep"), deep_ep.__file__)
print("JIT header:", barrier_header)
print("installation: OK")
PY

print -r -- "DeepGEMM wheel: ${DEEPGEMM_WHEELS[1]}"
print -r -- "DeepEP wheel: ${DEEPEP_WHEELS[1]}"
