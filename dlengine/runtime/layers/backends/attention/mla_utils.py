"""Shared MLA paged-cache helpers.

Gather/scatter utilities and logical->physical index translation shared by the
dense MLA attention implementations (``fa3``/``fa4``/``generic``) and the DSA
sparse-attention family. Previously duplicated across ``hopper/attention.py`` and
``generic/attention.py``; this is now the single definition.
"""

import torch


def merge_attention_states(
    left: torch.Tensor,
    left_lse: torch.Tensor,
    right: torch.Tensor,
    right_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two attention outputs using their per-head log-sum-exp states."""
    merged_lse = torch.logaddexp(left_lse, right_lse)
    left_scale = torch.exp(left_lse - merged_lse).transpose(0, 1).unsqueeze(-1)
    right_scale = torch.exp(right_lse - merged_lse).transpose(0, 1).unsqueeze(-1)
    merged = left.float() * left_scale + right.float() * right_scale
    return merged.to(left.dtype), merged_lse


def chunked_prefix_mla_attention(
    q: torch.Tensor,
    k_fresh: torch.Tensor,
    v_fresh: torch.Tensor,
    cached_latent: torch.Tensor,
    cached_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kc_weight: torch.Tensor,
    vc_weight: torch.Tensor,
    *,
    chunk_size: int,
    softmax_scale: float,
    attention_func,
) -> torch.Tensor:
    """Attend to fresh tokens and a bounded expansion of cached MLA rows."""
    fresh, fresh_lse = attention_func(
        q,
        k_fresh,
        v_fresh,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_q,
        max_seqlen_q=int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()),
        max_seqlen_k=int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()),
        softmax_scale=softmax_scale,
        causal=True,
        return_lse=True,
    )
    output, lse = fresh, fresh_lse
    lengths = [int(value) for value in cached_lens.tolist()]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    max_cached = max(lengths, default=0)
    heads = q.shape[1]
    nope_dim = kc_weight.shape[1] // heads
    value_dim = vc_weight.shape[1] // heads
    for offset in range(0, max_cached, chunk_size):
        pieces = [
            cached_latent[
                starts[i] + offset : starts[i] + min(length, offset + chunk_size)
            ]
            for i, length in enumerate(lengths)
        ]
        lens = [piece.shape[0] for piece in pieces]
        if not sum(lens):
            continue
        rows = torch.cat(pieces, dim=0)
        compressed, rope = rows[:, : kc_weight.shape[0]], rows[:, kc_weight.shape[0] :]
        k_nope = (compressed @ kc_weight).view(-1, heads, nope_dim)
        k = torch.cat([k_nope, rope[:, None, :].expand(-1, heads, -1)], dim=-1)
        v = (compressed @ vc_weight).view(-1, heads, value_dim)
        cu_k = torch.zeros(len(lens) + 1, device=q.device, dtype=torch.int32)
        cu_k[1:] = torch.tensor(lens, device=q.device, dtype=torch.int32).cumsum(0)
        part, part_lse = attention_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()),
            max_seqlen_k=max(lens),
            softmax_scale=softmax_scale,
            causal=False,
            return_lse=True,
        )
        output, lse = merge_attention_states(output, lse, part, part_lse)
    return output


from dlengine.runtime.kernel.triton.generic.paged_gather import (
    build_paged_gather_indices as _build_paged_gather_indices,
)


def _compute_cached_split(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-sequence cached/fresh split for chunked prefill.

    Returns:
        cached_lens:  [num_seqs] — number of previously-cached tokens per sequence
        cu_cached:    [num_seqs + 1] — cumulative cached lengths
    """
    seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).long()
    seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).long()
    cached_lens = seqlens_k - seqlens_q
    cu_cached = torch.zeros_like(cu_seqlens_k)
    cu_cached[1:] = cached_lens.cumsum(0)
    return cached_lens, cu_cached


def _interleave_cached_fresh(
    cached: torch.Tensor,
    fresh: torch.Tensor,
    cached_lens: torch.Tensor,
    cu_cached: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> torch.Tensor:
    """Interleave cached and fresh tensors into ragged K layout.

    Per sequence i the output is [cached_tokens_i, fresh_tokens_i] contiguously.

    Fully vectorized: per-row destination indices are computed on-device and the
    rows are scattered in two ``index_put`` ops. This replaces the former Python
    per-sequence loop, which issued ~5 ``.item()`` host syncs per sequence (×2
    for K and V, ×num_layers) and serialized the GPU during chunked prefill.
    """
    # total_k == total_cached + total_fresh, and both are host-known tensor
    # shapes, so the output is allocated without a device->host sync.
    total_cached = cached.shape[0]
    total_fresh = fresh.shape[0]
    ref = cached if cached.numel() > 0 else fresh
    out = ref.new_empty(total_cached + total_fresh, *ref.shape[1:])

    device = cu_seqlens_k.device
    cu_k = cu_seqlens_k.to(torch.int64)
    cu_q = cu_seqlens_q.to(torch.int64)
    cu_c = cu_cached.to(torch.int64)
    clens = cached_lens.to(torch.int64)

    # Cached rows: row j of `cached` belongs to seq s where cu_c[s] <= j <
    # cu_c[s+1]; it lands at cu_k[s] + (j - cu_c[s]) (prefix occupies the head).
    if total_cached > 0:
        idx_c = torch.arange(total_cached, device=device, dtype=torch.int64)
        seq_c = torch.searchsorted(cu_c, idx_c, right=True) - 1
        dest_c = cu_k[seq_c] + (idx_c - cu_c[seq_c])
        out[dest_c] = cached

    # Fresh rows: row j of `fresh` belongs to seq s where cu_q[s] <= j <
    # cu_q[s+1]; it lands at cu_k[s] + cached_lens[s] + (j - cu_q[s]) (after the
    # cached prefix for that sequence).
    if total_fresh > 0:
        idx_f = torch.arange(total_fresh, device=device, dtype=torch.int64)
        seq_f = torch.searchsorted(cu_q, idx_f, right=True) - 1
        dest_f = cu_k[seq_f] + clens[seq_f] + (idx_f - cu_q[seq_f])
        out[dest_f] = fresh

    return out


def _gather_kv_cached_concat(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_fresh: torch.Tensor,
    v_fresh: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather only previously-cached K/V from paged cache, concat with fresh K/V.

    Avoids redundantly re-reading fresh tokens that were just written to cache.
    Falls back to full gather when there are no cached tokens.
    """
    cached_lens, cu_cached = _compute_cached_split(cu_seqlens_q, cu_seqlens_k)
    total_cached = int(cu_cached[-1].item())

    if total_cached == 0:
        return k_fresh, v_fresh

    cached_indices = _build_paged_gather_indices(
        block_table, cu_cached, block_size, total_k=total_cached
    )
    _, _, num_kv_heads, head_dim = k_cache.shape
    k_flat = k_cache.reshape(-1, num_kv_heads, head_dim)
    v_flat = v_cache.reshape(-1, num_kv_heads, head_dim)
    k_cached = k_flat[cached_indices]
    v_cached = v_flat[cached_indices]

    k_out = _interleave_cached_fresh(
        k_cached, k_fresh, cached_lens, cu_cached, cu_seqlens_q, cu_seqlens_k
    )
    v_out = _interleave_cached_fresh(
        v_cached, v_fresh, cached_lens, cu_cached, cu_seqlens_q, cu_seqlens_k
    )
    return k_out, v_out


def _gather_cache_cached_only(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather only previously-cached tokens from a single paged cache.

    Returns:
        gathered:    [total_cached, ...] — cached tokens from paged cache
        cached_lens: [num_seqs] — per-sequence cached counts
        cu_cached:   [num_seqs + 1] — cumulative cached lengths
    """
    cached_lens, cu_cached = _compute_cached_split(cu_seqlens_q, cu_seqlens_k)
    total_cached = int(cu_cached[-1].item())

    if total_cached == 0:
        trailing = cache.shape[2:]
        gathered = cache.new_empty(0, *trailing)
        return gathered, cached_lens, cu_cached

    cached_indices = _build_paged_gather_indices(
        block_table, cu_cached, block_size, total_k=total_cached
    )
    trailing = cache.shape[2:]
    flat = cache.reshape(-1, *trailing)
    return flat[cached_indices], cached_lens, cu_cached


def topk_indices_to_physical(
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Convert logical token indices to physical paged KV cache indices.

    Args:
        topk_indices: (batch, topk) int32 — logical token positions (0..ctx_len-1)
                      May contain -1 for padding.
        block_table:  (batch, max_num_blocks) int32 — page table
        block_size:   int — tokens per page (64 for MLA)

    Returns:
        physical_indices: (batch, topk) int32 — physical slot indices
                          (physical_block * block_size + offset)
                          Padding entries (-1 in input) remain -1.
    """
    # Clamp negative indices to 0 so gather doesn't fail; result will be masked later
    valid_mask = topk_indices >= 0
    safe_indices = topk_indices.clamp(min=0)

    logical_block = safe_indices // block_size  # (batch, topk)
    offset_in_block = safe_indices % block_size  # (batch, topk)

    # Gather physical block IDs from block_table: (batch, topk)
    physical_block = torch.gather(block_table, dim=1, index=logical_block.long()).to(
        torch.int32
    )

    physical_indices = physical_block * block_size + offset_in_block
    # Invalid entries (-1 in input) must remain -1 so that sparse_decode_fwd
    # correctly skips them.  Using 0 would cause the kernel to attend to
    # physical slot 0 for every invalid index, corrupting the output.
    physical_indices = torch.where(valid_mask, physical_indices, -1)
    return physical_indices


__all__ = [
    "chunked_prefix_mla_attention",
    "merge_attention_states",
    "_compute_cached_split",
    "_interleave_cached_fresh",
    "_gather_kv_cached_concat",
    "_gather_cache_cached_only",
    "topk_indices_to_physical",
]
