"""GDN output transform component: gated RMSNorm + output projection."""

import torch
import torch.nn.functional as F
from torch import nn

try:
    from dlengine.runtime.kernel.triton.generic.rmsnorm_gated import (
        can_use_rms_norm_gated_kernel,
        rms_norm_gated_triton,
    )
except ImportError:
    can_use_rms_norm_gated_kernel = None
    rms_norm_gated_triton = None


class RMSNormGated(nn.Module):
    """RMSNorm followed by SiLU-gated multiplication.

    Applied per-head: weight has shape [head_v_dim].
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., hidden_size] — the value to normalize
            gate: [..., hidden_size] — gating signal (SiLU applied)
        """
        if can_use_rms_norm_gated_kernel is not None and can_use_rms_norm_gated_kernel(
            x, gate, self.weight
        ):
            return rms_norm_gated_triton(x, gate, self.weight, self.eps)

        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight * x.to(input_dtype)
        x = x * F.silu(gate.to(torch.float32)).to(input_dtype)
        return x


class OutputTransformMixin:
    """Gated RMSNorm + output projection, shared by forward & lazy verify."""

    def _apply_output_transform(
        self, core_attn_out: torch.Tensor, z: torch.Tensor, total_tokens: int
    ) -> torch.Tensor:
        z = z.view(total_tokens, self.num_v_heads, self.head_v_dim)
        out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        out = self.norm(out, z_flat)
        out = out.view(total_tokens, self.num_v_heads, self.head_v_dim)
        out = out.reshape(total_tokens, self.value_dim)
        return self.out_proj(out)


__all__ = ["RMSNormGated", "OutputTransformMixin"]
