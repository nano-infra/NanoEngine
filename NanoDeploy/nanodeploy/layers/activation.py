import torch
import torch.nn.functional as F
from torch import nn

# Optional vendored sglang fused kernel: chunk + silu + mul + (clamp) in
# one CUDA kernel. Falls back to the torch.compile path when the vendor
# isn't available or shapes/dtype don't fit.
# Source: https://github.com/sgl-project/sglang
#   python/sglang/jit_kernel/deepseek_v4.py::silu_and_mul_clamp
try:
    from nanodeploy._third_party.sglang_jit_kernel.deepseek_v4 import (
        silu_and_mul_clamp as _SGL_SILU_AND_MUL_CLAMP,
    )
except Exception:
    _SGL_SILU_AND_MUL_CLAMP = None


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast path: single-kernel SwiGLU when bf16 contig + sglang
        # fused kernel is available. We pass swiglu_limit=+inf to skip
        # the clamp branch (matching plain SwiGLU semantics).
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
                _SGL_SILU_AND_MUL_CLAMP(x, out, float("inf"))
                return out
            except Exception:
                pass
        return self._compiled_forward(x)

    @torch.compile
    def _compiled_forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, -1)
        return F.silu(a) * b
