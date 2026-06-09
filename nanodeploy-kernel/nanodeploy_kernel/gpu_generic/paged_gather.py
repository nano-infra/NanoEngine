"""Triton kernel for building paged gather indices from block tables."""

import torch
import triton
import triton.language as tl


@triton.jit
def _build_paged_gather_indices_kernel(
    block_table_ptr,  # [num_seqs, max_num_blocks]
    cu_seqlens_k_ptr,  # [num_seqs + 1]
    linear_indices_ptr,  # [total_k]
    max_num_blocks,  # block_table stride-0 (= block_table.shape[1])
    BLOCK_SIZE: tl.constexpr,  # tokens per KV cache block
    BLOCK_N: tl.constexpr,  # Triton tile width (number of tokens per iteration)
):
    seq_id = tl.program_id(0)

    start = tl.load(cu_seqlens_k_ptr + seq_id).to(tl.int64)
    end = tl.load(cu_seqlens_k_ptr + seq_id + 1).to(tl.int64)
    seqlen = end - start

    # Loop over tokens of this sequence in tiles of BLOCK_N
    for offset in range(0, tl.cdiv(seqlen, BLOCK_N) * BLOCK_N, BLOCK_N):
        tok_offsets = offset + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask = tok_offsets < seqlen

        block_idx = tok_offsets // BLOCK_SIZE
        in_block_offset = tok_offsets % BLOCK_SIZE

        # Gather physical block IDs from block_table[seq_id, block_idx]
        bt_ptrs = block_table_ptr + seq_id * max_num_blocks + block_idx
        block_ids = tl.load(bt_ptrs, mask=mask, other=0).to(tl.int64)

        linear_idx = block_ids * BLOCK_SIZE + in_block_offset

        out_ptrs = linear_indices_ptr + start + tok_offsets
        tl.store(out_ptrs, linear_idx, mask=mask)


def build_paged_gather_indices(
    block_table: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Build flat linear indices for gathering tokens from a paged cache.

    Args:
        block_table: [num_seqs, max_num_blocks] — int32 physical block IDs
        cu_seqlens_k:[num_seqs + 1] — cumulative K lengths (int32)
        block_size:  tokens per cache block

    Returns:
        linear_indices: [total_k_tokens] — index into cache.reshape(-1, ...)
    """
    num_seqs = block_table.shape[0]
    total_k = int(cu_seqlens_k[-1].item())
    device = block_table.device

    linear_indices = torch.empty(total_k, dtype=torch.int64, device=device)

    if total_k == 0:
        return linear_indices

    BLOCK_N = 256  # tile size — each program processes 256 tokens per iteration
    _build_paged_gather_indices_kernel[(num_seqs,)](
        block_table,
        cu_seqlens_k,
        linear_indices,
        block_table.shape[1],  # max_num_blocks (stride)
        BLOCK_SIZE=block_size,
        BLOCK_N=BLOCK_N,
    )

    return linear_indices
