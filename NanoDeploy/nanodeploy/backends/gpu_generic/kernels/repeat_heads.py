import os

import torch
import triton
import triton.language as tl


SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
FORCE_REPEAT_HEADS_TRITON = (
    os.getenv("NANODEPLOY_FORCE_REPEAT_HEADS_TRITON", "0") == "1"
)


@triton.jit
def _repeat_heads_kernel(
    x_ptr,
    out_ptr,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    kv_ratio,
    out_heads,
    head_dim,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    cols = tl.arange(0, BLOCK_D)
    mask = cols < head_dim
    src_h = pid_h // kv_ratio

    x_ptrs = x_ptr + pid_t * stride_x_t + src_h * stride_x_h + cols * stride_x_d
    out_ptrs = (
        out_ptr + pid_t * stride_out_t + pid_h * stride_out_h + cols * stride_out_d
    )

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, x, mask=mask)


def can_use_repeat_heads_triton(x: torch.Tensor, kv_ratio: int) -> bool:
    if not FORCE_REPEAT_HEADS_TRITON:
        return False
    if not x.is_cuda:
        return False
    if not x.is_contiguous():
        return False
    if x.ndim != 3:
        return False
    if x.dtype not in SUPPORTED_DTYPES:
        return False
    if kv_ratio <= 1:
        return False
    if x.numel() == 0:
        return False
    if torch.is_grad_enabled() and x.requires_grad:
        return False
    return True


def repeat_heads_triton(x: torch.Tensor, kv_ratio: int) -> torch.Tensor:
    tokens, num_heads, head_dim = x.shape
    out_heads = num_heads * kv_ratio
    out = torch.empty((tokens, out_heads, head_dim), device=x.device, dtype=x.dtype)

    block_d = 1 << (head_dim - 1).bit_length()
    num_warps = 4 if head_dim <= 128 else 8

    _repeat_heads_kernel[(tokens, out_heads)](
        x,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        kv_ratio,
        out_heads,
        head_dim,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=2,
    )
    return out
