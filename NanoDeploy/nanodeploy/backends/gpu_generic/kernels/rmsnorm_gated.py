import os

import torch
import triton
import triton.language as tl


SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
MAX_FUSED_HIDDEN_SIZE = 4096
VALIDATED_HIDDEN_SIZES = frozenset({128})
ALLOW_UNVALIDATED_HIDDEN_SIZES = (
    os.getenv("NANODEPLOY_RMSNORM_GATED_ALLOW_UNVALIDATED", "0") == "1"
)


@triton.jit
def _rms_norm_gated_kernel(
    x_ptr,
    gate_ptr,
    weight_ptr,
    out_ptr,
    stride_x_row,
    stride_gate_row,
    stride_out_row,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    x = tl.load(x_ptr + row_idx * stride_x_row + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    variance = tl.sum(x * x, axis=0) / hidden_size
    inv_rms = tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(
        gate_ptr + row_idx * stride_gate_row + cols, mask=mask, other=0.0
    ).to(tl.float32)
    gate = gate * tl.sigmoid(gate)

    out = x * inv_rms * weight * gate
    tl.store(out_ptr + row_idx * stride_out_row + cols, out, mask=mask)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _num_warps(hidden_size: int) -> int:
    if hidden_size <= 256:
        return 4
    if hidden_size <= 1024:
        return 8
    return 16


def can_use_rms_norm_gated_kernel(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor
) -> bool:
    if not x.is_cuda:
        return False
    if not gate.is_cuda or gate.device != x.device:
        return False
    if not weight.is_cuda or weight.device != x.device:
        return False
    if not x.is_contiguous() or not gate.is_contiguous() or not weight.is_contiguous():
        return False
    if x.dtype not in SUPPORTED_DTYPES:
        return False
    if gate.dtype != x.dtype or weight.dtype != x.dtype:
        return False
    if x.shape != gate.shape:
        return False
    if x.shape[-1] != weight.numel():
        return False
    if x.shape[-1] > MAX_FUSED_HIDDEN_SIZE:
        return False
    if not ALLOW_UNVALIDATED_HIDDEN_SIZES and x.shape[-1] not in VALIDATED_HIDDEN_SIZES:
        return False
    if x.numel() == 0:
        return False
    if torch.is_grad_enabled() and (
        x.requires_grad or gate.requires_grad or weight.requires_grad
    ):
        return False
    return True


def rms_norm_gated_triton(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    hidden_size = x.shape[-1]
    x_2d = x.reshape(-1, hidden_size)
    gate_2d = gate.reshape(-1, hidden_size)
    out = torch.empty_like(x_2d)

    block_size = _next_power_of_2(hidden_size)
    num_warps = _num_warps(hidden_size)

    _rms_norm_gated_kernel[(x_2d.shape[0],)](
        x_2d,
        gate_2d,
        weight,
        out,
        x_2d.stride(0),
        gate_2d.stride(0),
        out.stride(0),
        hidden_size,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=4,
    )
    return out.reshape_as(x)
