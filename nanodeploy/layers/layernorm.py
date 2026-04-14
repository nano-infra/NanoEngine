import torch
from torch import nn

try:
    from nanodeploy.kernels.layernorm import (fused_add_rms_norm_triton,
                                              rms_norm_triton)
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    from vllm import _custom_ops as ops
    HAS_VLLM_OPS = True
except ImportError:
    HAS_VLLM_OPS = False


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if HAS_VLLM_OPS and x.is_cuda:
            out = torch.empty_like(x)
            ops.rms_norm(out, x, self.weight.data, self.eps)
            return out
        if HAS_TRITON and x.is_cuda:
            return rms_norm_triton(x, self.weight, self.eps)

        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if HAS_VLLM_OPS and x.is_cuda and residual.is_cuda:
            ops.fused_add_rms_norm(x, residual, self.weight.data, self.eps)
            return x, residual
        if HAS_TRITON and x.is_cuda and residual.is_cuda:
            return fused_add_rms_norm_triton(x, residual, self.weight, self.eps)

        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
