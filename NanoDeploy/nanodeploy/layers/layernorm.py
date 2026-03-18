import logging

import torch
from torch import nn


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
        # Cache for (1 + weight) when add_unit_offset=True.
        # Populated lazily on first forward; avoids per-call Add/Cast.
        self._offset_weight: torch.Tensor | None = None

    def _get_weight(self) -> torch.Tensor:
        """Return effective weight, caching the add_unit_offset computation."""
        if not self.add_unit_offset:
            return self.weight
        if self._offset_weight is None:
            self._offset_weight = (1.0 + self.weight).detach()
        return self._offset_weight

    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Fast path on Ascend NPU via npu_rms_norm
        try:
            import torch_npu

            out, _ = torch_npu.npu_rms_norm(x, self._get_weight(), epsilon=self.eps)
            return out
        except (ImportError, AttributeError):
            pass

        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))

        weight = self._get_weight()
        if weight.dtype != orig_dtype:
            weight = weight.to(orig_dtype)
        x = x.to(orig_dtype).mul_(weight)
        return x

    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fast path on Ascend NPU via npu_add_rms_norm
        try:
            import torch_npu

            x, _, residual = torch_npu.npu_add_rms_norm(
                x, residual, self._get_weight(), self.eps
            )
            return x, residual
        except (ImportError, AttributeError):
            pass

        orig_dtype = x.dtype
        x = x.float().add_(residual.float())

        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)

        x.mul_(torch.rsqrt(var + self.eps))

        weight = self._get_weight()
        if weight.dtype != orig_dtype:
            weight = weight.to(orig_dtype)
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
