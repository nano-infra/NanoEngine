"""Vendored slice of ``sglang.jit_kernel`` (DSV4 fused kernels only).

Source: https://github.com/sgl-project/sglang
        python/sglang/jit_kernel/

Why vendored: the upstream ``sglang`` Python package pulls in ~2 GB of
deps (flash-attn, deep_gemm, full sglang.srt, ...). For NanoDeploy we
only need the JIT loaders for a handful of DSV4 kernels (fused_rope,
rmsnorm_self, fused_norm_rope_inplace). This subpackage strips the
upstream surface to the minimum needed runtime, and stubs out the few
``sglang.srt.*`` references with no-ops.

Runtime deps (must be installed by the user):
- torch
- triton (transitively, via tvm-ffi build path)
- tvm-ffi (provides ``tvm_ffi.cpp.load_inline`` and the C++ headers
  for ``tvm::ffi::TensorView`` etc.)

License: Apache-2.0 (matches upstream sgl-kernel; see LICENSE).
"""

import os


def fused_kernels_enabled() -> bool:
    """Whether the vendored sglang JIT fused kernels should be used.

    These kernels (fused RMSNorm+RoPE, SwiGLU, hyper-connection,
    compressor, ...) are written for Hopper (sm_90: H100/H200) and rely
    on TMA / wgmma / PDL code paths that fail to build or crash at
    runtime on other GPU architectures. On non-Hopper GPUs callers must
    fall back to their eager implementations: slower, but correct,
    portable, and still CUDAGraph-capturable — so the same image runs on
    H200 and non-H200 alike instead of getting stuck on a Hopper-only
    kernel.

    Selection is automatic by GPU arch. Override with
    ``NANODEPLOY_SGLANG_FUSED_KERNELS={auto,1,0}`` (default ``auto``):
      * ``auto`` — enable only on Hopper (compute capability major == 9)
      * ``1`` / ``on`` / ``force`` — force-enable regardless of arch
      * ``0`` / ``off`` — force-disable (always use eager fallbacks)
    """
    override = os.getenv("NANODEPLOY_SGLANG_FUSED_KERNELS", "auto")
    override = override.strip().lower()
    if override in ("0", "off", "false", "no"):
        return False
    if override in ("1", "on", "true", "yes", "force"):
        return True
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        # Hopper only. Bump to ``>= 9`` (or add the Blackwell major) once
        # these kernels are validated on newer architectures.
        return torch.cuda.get_device_capability()[0] == 9
    except Exception:
        return False
