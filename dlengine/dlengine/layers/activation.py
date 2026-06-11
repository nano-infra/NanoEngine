from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from dlengine.compile_utils import maybe_compile

# Optional vendored sglang fused kernel: chunk + silu + mul + (clamp) in
# one CUDA kernel. Falls back to the torch.compile path when the vendor
# isn't available or shapes/dtype don't fit. The kernel is Hopper-only,
# so it is gated behind the shared GPU-arch check — on non-Hopper GPUs we
# keep it ``None`` and use the eager/compiled path (CUDAGraph still works).
# Source: https://github.com/sgl-project/sglang
#   python/sglang/jit_kernel/deepseek_v4.py::silu_and_mul_clamp
from dlengine.kernel.jit.sgl import fused_kernels_enabled

if fused_kernels_enabled():
    try:
        from dlengine.kernel.jit.sgl.deepseek_v4 import (
            silu_and_mul_clamp as _SGL_SILU_AND_MUL_CLAMP,
        )
    except Exception:
        _SGL_SILU_AND_MUL_CLAMP = None
else:
    _SGL_SILU_AND_MUL_CLAMP = None


class SiluAndMul(nn.Module):
    """SwiGLU activation, optionally clamping ``silu(gate) * up`` to
    ``[-swiglu_limit, +swiglu_limit]`` before it leaves the activation.

    DSV4 ships ``swiglu_limit=10.0``; older Deepseek/Qwen variants do
    not clamp (``swiglu_limit=None``). The clamp must run *before* the
    downstream FP8 quant or it changes the absmax → scale → output
    rounding and flips top-1 token picks.
    """

    def __init__(self, swiglu_limit: Optional[float] = None):
        super().__init__()
        # Stored as +inf when unset so the fused kernel's clamp branch
        # is a no-op (the kernel always runs; +inf disables the bound).
        self._swiglu_limit = swiglu_limit
        self._effective_limit: float = (
            float("inf") if swiglu_limit is None else float(swiglu_limit)
        )
        # Lazy compile to avoid attaching ConfigModuleInstance refs at
        # class level (breaks cloudpickle in Ray actors on torch >= 2.10).
        self._compiled_forward = maybe_compile(self._compiled_forward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast path: single-kernel SwiGLU when bf16 contig + sglang
        # fused kernel is available.
        if (
            _SGL_SILU_AND_MUL_CLAMP is not None
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.shape[-1] % 2 == 0
        ):
            try:
                D = x.shape[-1] // 2
                out = torch.empty(*x.shape[:-1], D, dtype=x.dtype, device=x.device)
                _SGL_SILU_AND_MUL_CLAMP(x, out, self._effective_limit)
                return out
            except Exception as _exc:
                global _SILU_AND_MUL_CLAMP_WARNED
                if "_SILU_AND_MUL_CLAMP_WARNED" not in globals():
                    _SILU_AND_MUL_CLAMP_WARNED = set()
                _key = type(_exc).__name__
                if _key not in _SILU_AND_MUL_CLAMP_WARNED:
                    _SILU_AND_MUL_CLAMP_WARNED.add(_key)
                    from dlengine.logging import get_logger

                    get_logger().warning(
                        "silu_and_mul_clamp fast path bailed: %s. x.shape=%s "
                        "dtype=%s contig=%s limit=%s. Eager fallback.",
                        _exc,
                        tuple(x.shape),
                        x.dtype,
                        x.is_contiguous(),
                        self._effective_limit,
                    )
        return self._eager_forward(x)

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._swiglu_limit is None:
            return self._compiled_forward(x)
        # DSV4 reference (model.py:600-603): clamp ``up`` two-sided,
        # ``gate`` upper-bound only, BEFORE silu * up.
        gate, up = x.chunk(2, -1)
        up = up.clamp(-self._effective_limit, self._effective_limit)
        gate = gate.clamp(max=self._effective_limit)
        return F.silu(gate) * up

    def _compiled_forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, -1)
        return F.silu(a) * b
