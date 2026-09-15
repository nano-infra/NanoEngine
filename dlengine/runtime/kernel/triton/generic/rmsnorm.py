import torch
import triton
import triton.language as tl

SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
MAX_FUSED_HIDDEN_SIZE = 8192


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_x_row,
    stride_out_row,
    hidden_size,
    eps,
    add_unit_offset: tl.constexpr,
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
    if add_unit_offset:
        weight = weight + 1.0

    out = x * inv_rms * weight
    tl.store(out_ptr + row_idx * stride_out_row + cols, out, mask=mask)


@triton.jit
def _add_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    residual_out_ptr,
    stride_x_row,
    stride_residual_row,
    stride_out_row,
    stride_residual_out_row,
    hidden_size,
    eps,
    add_unit_offset: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    x = tl.load(x_ptr + row_idx * stride_x_row + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    residual = tl.load(
        residual_ptr + row_idx * stride_residual_row + cols, mask=mask, other=0.0
    ).to(tl.float32)
    summed = x + residual
    tl.store(
        residual_out_ptr + row_idx * stride_residual_out_row + cols, summed, mask=mask
    )

    variance = tl.sum(summed * summed, axis=0) / hidden_size
    inv_rms = tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    if add_unit_offset:
        weight = weight + 1.0

    out = summed * inv_rms * weight
    tl.store(out_ptr + row_idx * stride_out_row + cols, out, mask=mask)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _num_warps(hidden_size: int) -> int:
    if hidden_size <= 256:
        return 4
    if hidden_size <= 1024:
        return 8
    if hidden_size <= 4096:
        return 8
    return 16


def can_use_rms_norm_kernel(x: torch.Tensor, weight: torch.Tensor) -> bool:
    if not x.is_cuda:
        return False
    if not weight.is_cuda or weight.device != x.device:
        return False
    if not x.is_contiguous() or not weight.is_contiguous():
        return False
    if x.dtype not in SUPPORTED_DTYPES or weight.dtype not in SUPPORTED_DTYPES:
        return False
    if x.shape[-1] > MAX_FUSED_HIDDEN_SIZE:
        return False
    if x.shape[-1] != weight.numel():
        return False
    if x.numel() == 0:
        return False
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return False
    return True


def rms_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    add_unit_offset: bool = False,
) -> torch.Tensor:
    hidden_size = x.shape[-1]
    x_2d = x.reshape(-1, hidden_size)
    out = torch.empty_like(x_2d)

    block_size = _next_power_of_2(hidden_size)
    num_warps = _num_warps(hidden_size)

    _rms_norm_kernel[(x_2d.shape[0],)](
        x_2d,
        weight,
        out,
        x_2d.stride(0),
        out.stride(0),
        hidden_size,
        eps,
        add_unit_offset=add_unit_offset,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=4,
    )
    return out.reshape_as(x)


def can_use_strided_inplace_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Whether x can be RMS-normalized in place with a row stride.

    MLA stores the normalized 512-wide compressed latent in the first part of
    a 576-wide projection buffer. That slice is contiguous within each row,
    but Tensor.is_contiguous() is false because the row stride remains 576.
    Accepting that layout avoids materializing the slice and copying it back.
    """
    if not x.is_cuda or x.dim() != 2 or x.stride(1) != 1:
        return False
    if not weight.is_cuda or weight.device != x.device or not weight.is_contiguous():
        return False
    if x.dtype not in SUPPORTED_DTYPES or weight.dtype not in SUPPORTED_DTYPES:
        return False
    if x.shape[-1] > MAX_FUSED_HIDDEN_SIZE or x.shape[-1] != weight.numel():
        return False
    if x.numel() == 0:
        return False
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return False
    return True


def rms_norm_strided_inplace(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    add_unit_offset: bool = False,
) -> torch.Tensor:
    """Apply RMSNorm to a 2-D row-strided tensor without a temporary buffer."""
    if not can_use_strided_inplace_rms_norm(x, weight):
        raise ValueError(
            "rms_norm_strided_inplace requires a CUDA 2-D tensor with "
            "stride(-1) == 1 and a matching contiguous weight"
        )

    hidden_size = x.shape[-1]
    block_size = _next_power_of_2(hidden_size)
    _rms_norm_kernel[(x.shape[0],)](
        x,
        weight,
        x,
        x.stride(0),
        x.stride(0),
        hidden_size,
        eps,
        add_unit_offset=add_unit_offset,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(hidden_size),
        num_stages=4,
    )
    return x


def add_rms_norm_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    add_unit_offset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = x.shape[-1]
    x_2d = x.reshape(-1, hidden_size)
    residual_2d = residual.reshape(-1, hidden_size)

    out = torch.empty_like(x_2d)
    residual_out = torch.empty_like(residual_2d)

    block_size = _next_power_of_2(hidden_size)
    num_warps = _num_warps(hidden_size)

    _add_rms_norm_kernel[(x_2d.shape[0],)](
        x_2d,
        residual_2d,
        weight,
        out,
        residual_out,
        x_2d.stride(0),
        residual_2d.stride(0),
        out.stride(0),
        residual_out.stride(0),
        hidden_size,
        eps,
        add_unit_offset=add_unit_offset,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=4,
    )
    return out.reshape_as(x), residual_out.reshape_as(residual)
