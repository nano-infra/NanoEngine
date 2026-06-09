import torch
from torch import nn

try:
    from nanodeploy.backends.gpu_generic.kernels.rmsnorm import (
        add_rms_norm_triton,
        can_use_rms_norm_kernel,
        rms_norm_triton,
    )
except ImportError:
    add_rms_norm_triton = None
    can_use_rms_norm_kernel = None
    rms_norm_triton = None


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        add_unit_offset: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        if self.add_unit_offset:
            self.weight = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if can_use_rms_norm_kernel is not None and can_use_rms_norm_kernel(
            x, self.weight
        ):
            return rms_norm_triton(
                x,
                self.weight,
                self.eps,
                add_unit_offset=self.add_unit_offset,
            )

        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))

        weight = (
            (1.0 + self.weight.float()).to(orig_dtype)
            if self.add_unit_offset
            else self.weight
        )
        x = x.to(orig_dtype).mul_(weight)
        return x

    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if (
            add_rms_norm_triton is not None
            and can_use_rms_norm_kernel is not None
            and can_use_rms_norm_kernel(x, self.weight)
            and residual.is_cuda
            and residual.device == x.device
            and residual.is_contiguous()
            and residual.dtype == x.dtype
            and residual.shape == x.shape
        ):
            # print("add_rms_norm_triton")
            return add_rms_norm_triton(
                x,
                residual,
                self.weight,
                self.eps,
                add_unit_offset=self.add_unit_offset,
            )
        # print("no add_rms_norm_triton")
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())

        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)

        x.mul_(torch.rsqrt(var + self.eps))

        weight = (
            (1.0 + self.weight.float()).to(orig_dtype)
            if self.add_unit_offset
            else self.weight
        )
        x = x.to(orig_dtype).mul_(weight)
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
