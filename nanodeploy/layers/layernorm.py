import torch
from torch import nn


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
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())

        # Debug logging for fused value
        if x.numel() > 10 * 2048:  # Only log for meaningful sequences
            from nanodeploy.logging import get_logger

            logger = get_logger()
            logger.info(
                f"[Pre-Norm Fused] Mean: {x.mean().item():.6f}, Max: {x.max().item():.4f}, Sum: {x.sum().item():.4f}"
            )
            logger.info(f"[Pre-Norm Fused] First 10: {x.flatten()[:10].tolist()}")
            logger.info(
                f"[Pre-Add] Residual First 10: {residual.flatten()[:10].tolist()}"
            )

        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)

        # Debug logging for RMSNorm computation
        if x.numel() > 10 * 2048:
            rsqrt_val = torch.rsqrt(var + self.eps)
            logger.info(
                f"[RMSNorm Debug] Input Mean: {x.mean().item():.6f}, Input Max: {x.max().item():.4f}"
            )
            logger.info(
                f"[RMSNorm Debug] Variance Mean: {var.mean().item():.6f}, Variance Max: {var.max().item():.4f}"
            )
            logger.info(f"[RMSNorm Debug] Eps: {self.eps}")
            logger.info(
                f"[RMSNorm Debug] Rsqrt Mean: {rsqrt_val.mean().item():.6f}, Rsqrt Max: {rsqrt_val.max().item():.4f}"
            )

        x.mul_(torch.rsqrt(var + self.eps))

        # Debug logging after rsqrt
        if x.numel() > 10 * 2048:
            logger.info(
                f"[RMSNorm Debug] After Rsqrt Mean: {x.mean().item():.6f}, After Rsqrt Max: {x.max().item():.4f}"
            )
            logger.info(
                f"[RMSNorm Debug] Weight Mean: {self.weight.mean().item():.4f}, Weight Max: {self.weight.max().item():.4f}"
            )

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
