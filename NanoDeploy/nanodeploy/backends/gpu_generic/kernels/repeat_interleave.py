import torch
import triton
import triton.language as tl


SUPPORTED_VALUE_DTYPES = {torch.int32, torch.int64}


@triton.jit
def _repeat_interleave_from_prefix_kernel(
    values_ptr,
    prefix_ptr,
    out_ptr,
    num_values,
    output_size,
    BLOCK_SIZE: tl.constexpr,
    MAX_BINARY_ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < output_size
    target = tl.where(mask, offs.to(tl.int64), 0)

    low = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    high = tl.full([BLOCK_SIZE], num_values + 1, dtype=tl.int32)

    for _ in range(MAX_BINARY_ITERS):
        mid = (low + high) // 2
        prefix_mid = tl.load(prefix_ptr + mid).to(tl.int64)
        go_right = prefix_mid <= target
        low = tl.where(go_right, mid + 1, low)
        high = tl.where(go_right, high, mid)

    src_idx = low - 1
    out = tl.load(values_ptr + src_idx, mask=mask, other=0)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _repeat_interleave_segment_fill_kernel(
    values_ptr,
    prefix_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_value = tl.program_id(0)
    pid_block = tl.program_id(1)

    start = tl.load(prefix_ptr + pid_value).to(tl.int64)
    end = tl.load(prefix_ptr + pid_value + 1).to(tl.int64)
    seg_len = end - start
    block_start = pid_block * BLOCK_SIZE

    cols = block_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < seg_len

    value = tl.load(values_ptr + pid_value)
    tl.store(out_ptr + start + cols, value, mask=mask)


def can_use_repeat_interleave_from_prefix_triton(
    values: torch.Tensor,
    prefix: torch.Tensor,
) -> bool:
    if not values.is_cuda or not prefix.is_cuda:
        return False
    if values.device != prefix.device:
        return False
    if not values.is_contiguous() or not prefix.is_contiguous():
        return False
    if values.ndim != 1 or prefix.ndim != 1:
        return False
    if values.dtype not in SUPPORTED_VALUE_DTYPES:
        return False
    if prefix.dtype != torch.int64:
        return False
    if prefix.numel() != values.numel() + 1:
        return False
    if values.numel() == 0:
        return False
    return True


def _repeat_interleave_from_prefix_binary_search_triton(
    values: torch.Tensor,
    prefix: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    out = torch.empty(output_size, device=values.device, dtype=values.dtype)

    _repeat_interleave_from_prefix_kernel[(triton.cdiv(output_size, 256),)](
        values,
        prefix,
        out,
        values.numel(),
        output_size,
        BLOCK_SIZE=256,
        MAX_BINARY_ITERS=16,
        num_warps=4,
        num_stages=2,
    )
    return out


def _repeat_interleave_from_prefix_segment_fill_triton(
    values: torch.Tensor,
    prefix: torch.Tensor,
    output_size: int,
    max_repeat_hint: int,
) -> torch.Tensor:
    out = torch.empty(output_size, device=values.device, dtype=values.dtype)
    num_value_blocks = triton.cdiv(max_repeat_hint, 256)

    _repeat_interleave_segment_fill_kernel[(values.numel(), num_value_blocks)](
        values,
        prefix,
        out,
        BLOCK_SIZE=256,
        num_warps=4,
        num_stages=2,
    )
    return out


def repeat_interleave_from_prefix_triton(
    values: torch.Tensor,
    prefix: torch.Tensor,
    output_size: int,
    max_repeat_hint: int | None = None,
) -> torch.Tensor:
    if output_size <= 0:
        return torch.empty((0,), device=values.device, dtype=values.dtype)

    if max_repeat_hint is not None and max_repeat_hint > 0:
        return _repeat_interleave_from_prefix_segment_fill_triton(
            values,
            prefix,
            output_size,
            max_repeat_hint,
        )

    return _repeat_interleave_from_prefix_binary_search_triton(
        values,
        prefix,
        output_size,
    )
