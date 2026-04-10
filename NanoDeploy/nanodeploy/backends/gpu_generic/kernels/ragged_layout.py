import torch
import triton
import triton.language as tl


SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _ragged_to_padded_kernel(
    x_ptr,
    prefix_ptr,
    out_ptr,
    stride_x_t,
    stride_x_d,
    stride_out_b,
    stride_out_d,
    stride_out_t,
    num_seqs,
    total_tokens,
    dim,
    BLOCK_D: tl.constexpr,
    MAX_BINARY_ITERS: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    if pid_t >= total_tokens:
        return

    low = 0
    high = num_seqs
    target = pid_t

    for _ in range(MAX_BINARY_ITERS):
        mid = (low + high) // 2
        prefix_next = tl.load(prefix_ptr + mid + 1).to(tl.int64)
        if target < prefix_next:
            high = mid
        else:
            low = mid + 1

    seq_id = low
    seq_start = tl.load(prefix_ptr + seq_id).to(tl.int64)
    pos_in_seq = target - seq_start

    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < dim

    x_ptrs = x_ptr + pid_t * stride_x_t + cols * stride_x_d
    out_ptrs = (
        out_ptr
        + seq_id * stride_out_b
        + cols * stride_out_d
        + pos_in_seq * stride_out_t
    )
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, x, mask=mask)


@triton.jit
def _padded_to_ragged_kernel(
    x_ptr,
    prefix_ptr,
    out_ptr,
    stride_x_b,
    stride_x_d,
    stride_x_t,
    stride_out_t,
    stride_out_d,
    num_seqs,
    total_tokens,
    dim,
    BLOCK_D: tl.constexpr,
    MAX_BINARY_ITERS: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    if pid_t >= total_tokens:
        return

    low = 0
    high = num_seqs
    target = pid_t

    for _ in range(MAX_BINARY_ITERS):
        mid = (low + high) // 2
        prefix_next = tl.load(prefix_ptr + mid + 1).to(tl.int64)
        if target < prefix_next:
            high = mid
        else:
            low = mid + 1

    seq_id = low
    seq_start = tl.load(prefix_ptr + seq_id).to(tl.int64)
    pos_in_seq = target - seq_start

    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < dim

    x_ptrs = x_ptr + seq_id * stride_x_b + cols * stride_x_d + pos_in_seq * stride_x_t
    out_ptrs = out_ptr + pid_t * stride_out_t + cols * stride_out_d
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, x, mask=mask)


def _can_use_prefix(prefix: torch.Tensor) -> bool:
    return (
        prefix.is_cuda
        and prefix.is_contiguous()
        and prefix.ndim == 1
        and prefix.dtype == torch.int64
        and prefix.numel() >= 2
    )


def can_use_ragged_to_padded_triton(
    x: torch.Tensor,
    prefix: torch.Tensor,
    out: torch.Tensor | None = None,
) -> bool:
    if not x.is_cuda or not x.is_contiguous() or x.ndim != 2:
        return False
    if x.dtype not in SUPPORTED_DTYPES:
        return False
    if not _can_use_prefix(prefix) or prefix.device != x.device:
        return False
    if int(prefix[-1].item()) != x.shape[0]:
        return False
    if out is not None:
        if not out.is_cuda or out.device != x.device:
            return False
        if out.dtype != x.dtype or not out.is_contiguous() or out.ndim != 3:
            return False
        if out.shape[0] < prefix.numel() - 1 or out.shape[1] < x.shape[1]:
            return False
    return True


def ragged_to_padded_triton(
    x: torch.Tensor,
    prefix: torch.Tensor,
    max_seqlen: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    total_tokens, dim = x.shape
    num_seqs = prefix.numel() - 1

    if out is None:
        out = torch.zeros((num_seqs, dim, max_seqlen), device=x.device, dtype=x.dtype)
    else:
        out = out[:num_seqs, :dim, :max_seqlen]
        out.zero_()

    grid = (total_tokens, triton.cdiv(dim, 128))
    _ragged_to_padded_kernel[grid](
        x,
        prefix,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_seqs,
        total_tokens,
        dim,
        BLOCK_D=128,
        MAX_BINARY_ITERS=16,
        num_warps=4,
        num_stages=2,
    )
    return out


def can_use_padded_to_ragged_triton(
    x: torch.Tensor,
    prefix: torch.Tensor,
    out: torch.Tensor | None = None,
) -> bool:
    if not x.is_cuda or not x.is_contiguous() or x.ndim != 3:
        return False
    if x.dtype not in SUPPORTED_DTYPES:
        return False
    if not _can_use_prefix(prefix) or prefix.device != x.device:
        return False
    num_seqs = prefix.numel() - 1
    if x.shape[0] < num_seqs:
        return False
    total_tokens = int(prefix[-1].item())
    if out is not None:
        if not out.is_cuda or out.device != x.device:
            return False
        if out.dtype != x.dtype or not out.is_contiguous() or out.ndim != 2:
            return False
        if out.shape[0] < total_tokens or out.shape[1] < x.shape[1]:
            return False
    return True


def padded_to_ragged_triton(
    x: torch.Tensor,
    prefix: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    total_tokens = int(prefix[-1].item())
    dim = x.shape[1]
    num_seqs = prefix.numel() - 1

    if out is None:
        out = torch.empty((total_tokens, dim), device=x.device, dtype=x.dtype)
    else:
        out = out[:total_tokens, :dim]

    grid = (total_tokens, triton.cdiv(dim, 128))
    _padded_to_ragged_kernel[grid](
        x,
        prefix,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        num_seqs,
        total_tokens,
        dim,
        BLOCK_D=128,
        MAX_BINARY_ITERS=16,
        num_warps=4,
        num_stages=2,
    )
    return out
