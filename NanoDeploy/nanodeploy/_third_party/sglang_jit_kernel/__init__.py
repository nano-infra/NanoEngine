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
