"""Fused paged-to-ragged gather for the FP8 Indexer cache."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_indexer_cache_kernel(
    cache_ptr,
    block_table_ptr,
    cu_seqlens_k_ptr,
    key_out_ptr,
    scale_out_ptr,
    block_table_stride,
    cache_page_stride,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
):
    seq_id = tl.program_id(0)
    token_start = tl.program_id(1) * TOKENS_PER_PROGRAM
    token_offsets = token_start + tl.arange(0, TOKENS_PER_PROGRAM)

    ragged_start = tl.load(cu_seqlens_k_ptr + seq_id).to(tl.int64)
    ragged_end = tl.load(cu_seqlens_k_ptr + seq_id + 1).to(tl.int64)
    seq_len = ragged_end - ragged_start
    token_mask = token_offsets < seq_len

    logical_page = token_offsets // page_size
    in_page = token_offsets % page_size
    physical_page = tl.load(
        block_table_ptr + seq_id * block_table_stride + logical_page,
        mask=token_mask,
        other=0,
    ).to(tl.int64)
    page_base = physical_page * cache_page_stride

    key_offsets = tl.arange(0, head_dim)
    key_src = page_base[:, None] + in_page[:, None] * head_dim + key_offsets[None, :]
    ragged_token = ragged_start + token_offsets
    key_dst = ragged_token[:, None] * head_dim + key_offsets[None, :]
    key_mask = token_mask[:, None]
    key = tl.load(cache_ptr + key_src, mask=key_mask, other=0)
    tl.store(key_out_ptr + key_dst, key, mask=key_mask)

    scale_offsets = tl.arange(0, 4)
    scale_src = (
        page_base[:, None]
        + page_size * head_dim
        + in_page[:, None] * 4
        + scale_offsets[None, :]
    )
    scale_dst = ragged_token[:, None] * 4 + scale_offsets[None, :]
    scale_mask = token_mask[:, None]
    scale = tl.load(cache_ptr + scale_src, mask=scale_mask, other=0)
    tl.store(scale_out_ptr + scale_dst, scale, mask=scale_mask)


def gather_indexer_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    page_size: int,
    head_dim: int,
    total_k: int | None = None,
    max_seqlen_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather paged Indexer FP8 keys and scales into contiguous ragged rows.

    Each cache page stores all ``page_size * head_dim`` FP8 key bytes first,
    followed by one four-byte FP32 scale per token.
    """
    if cache.ndim != 2 or cache.dtype != torch.uint8 or not cache.is_contiguous():
        raise ValueError("cache must be a contiguous 2D uint8 tensor")
    if block_table.ndim != 2 or block_table.stride(1) != 1:
        raise ValueError("block_table must be 2D and contiguous in its last dimension")
    if cu_seqlens_k.ndim != 1 or not cu_seqlens_k.is_contiguous():
        raise ValueError("cu_seqlens_k must be a contiguous 1D tensor")
    if cu_seqlens_k.numel() != block_table.shape[0] + 1:
        raise ValueError("cu_seqlens_k and block_table batch dimensions differ")
    expected_page_bytes = page_size * (head_dim + 4)
    if cache.shape[1] != expected_page_bytes:
        raise ValueError(
            f"Indexer cache page has {cache.shape[1]} bytes, expected "
            f"{expected_page_bytes}"
        )

    if total_k is None:
        total_k = int(cu_seqlens_k[-1].item())
    key_out = torch.empty((total_k, head_dim), dtype=torch.uint8, device=cache.device)
    scale_out = torch.empty((total_k, 4), dtype=torch.uint8, device=cache.device)
    if total_k == 0:
        return key_out, scale_out

    if max_seqlen_k is None:
        max_seqlen_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item())
    tokens_per_program = 256
    _gather_indexer_cache_kernel[
        (block_table.shape[0], triton.cdiv(max_seqlen_k, tokens_per_program))
    ](
        cache,
        block_table,
        cu_seqlens_k,
        key_out,
        scale_out,
        block_table.stride(0),
        cache.stride(0),
        page_size=page_size,
        head_dim=head_dim,
        TOKENS_PER_PROGRAM=tokens_per_program,
    )
    return key_out, scale_out
