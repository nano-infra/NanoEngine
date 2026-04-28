"""Fused triton kernels for the index-construction launch storm in
``_decode_attention_flash_mla``.

The eager-mode trace localized ~700 launches/step coming from the
arange/where/clamp/floor_divide/remainder/gather chain that builds
``swa_indices`` and ``extra_indices`` (the physical-slot tensors fed to
``flash_mla.flash_mla_with_kvcache``). Per call site each chain is
~15 elementwise + 1-2 gather kernels; replacing each with one triton
kernel collapses the host-side launch count and the GPU-side per-op
overhead.

Bit-equivalent to the eager torch chain — verified per kernel against
the corresponding torch implementation on synthetic shapes.
"""

import torch
import triton
import triton.language as tl


# ─── SWA (sliding-window) physical-slot indices ─────────────────────────────


@triton.jit
def _build_swa_indices_kernel(
    context_lens_ptr,  # int32 [bs]
    block_tables_ptr,  # int32 [bs, MAX_BLOCKS]
    out_ptr,  # int32 [bs, SWA_TOPK]
    BS: tl.constexpr,
    SWA_TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_TABLES_STRIDE: tl.constexpr,
    SWA_TOPK_MAX: tl.constexpr,  # min(window_size, SWA_TOPK)
):
    """One program per sequence (b in [0, BS)). Each program emits a
    full row of size SWA_TOPK into ``out``.

    Math (one row, batch index ``b``):
        ctx_len    = context_lens[b]
        win_len    = min(ctx_len, SWA_TOPK_MAX)
        start_pos  = ctx_len - win_len
        logical    = start_pos + arange(SWA_TOPK)
        valid      = arange(SWA_TOPK) < win_len
        page_idx   = min(logical // PAGE_SIZE, MAX_BLOCKS - 1)
        tok_in_pg  = logical % PAGE_SIZE
        block      = block_tables[b, page_idx]
        phys       = block * PAGE_SIZE + tok_in_pg
        out[b, t]  = phys if valid else -1
    """
    bs_id = tl.program_id(0)

    tok = tl.arange(0, SWA_TOPK)
    ctx_len = tl.load(context_lens_ptr + bs_id).to(tl.int32)
    # win_len = min(ctx_len, SWA_TOPK_MAX)
    win_len = tl.minimum(ctx_len, SWA_TOPK_MAX)
    start = ctx_len - win_len
    logical = start + tok  # [SWA_TOPK]
    valid = tok < win_len

    page_idx = logical // PAGE_SIZE
    tok_in_page = logical % PAGE_SIZE
    # Clamp page_idx into [0, MAX_BLOCKS) so the gather is safe; invalid
    # entries are masked out by ``valid`` below.
    page_idx_safe = tl.minimum(page_idx, MAX_BLOCKS - 1)
    block = tl.load(block_tables_ptr + bs_id * BLOCK_TABLES_STRIDE + page_idx_safe).to(
        tl.int32
    )
    phys = block * PAGE_SIZE + tok_in_page
    out = tl.where(valid, phys, -1)
    tl.store(out_ptr + bs_id * SWA_TOPK + tok, out)


def build_swa_indices(
    context_lens: torch.Tensor,  # [bs] int32
    block_tables: torch.Tensor,  # [bs, max_blocks] int32
    swa_topk: int,
    page_size: int,
    swa_topk_max: int,
) -> torch.Tensor:
    """Drop-in replacement for the SWA-indices block in
    ``_decode_attention_flash_mla`` (deepseek_v4.py:1709-1738).

    Returns ``swa_indices`` shape ``[bs, swa_topk]`` int32 (caller
    unsqueeze(1) for [bs, 1, swa_topk] if needed).
    """
    bs = context_lens.shape[0]
    out = torch.empty(bs, swa_topk, dtype=torch.int32, device=context_lens.device)
    grid = (bs,)
    _build_swa_indices_kernel[grid](
        context_lens.contiguous(),
        block_tables.contiguous(),
        out,
        BS=bs,
        SWA_TOPK=swa_topk,
        PAGE_SIZE=page_size,
        MAX_BLOCKS=block_tables.shape[1],
        BLOCK_TABLES_STRIDE=block_tables.stride(0),
        SWA_TOPK_MAX=swa_topk_max,
    )
    return out


# ─── Extra (compressed paged) physical-slot indices ─────────────────────────


@triton.jit
def _build_extra_indices_paged_kernel(
    seq_slots_ptr,  # int64 [bs]
    compressed_counts_ptr,  # int32 [num_state_slots]
    comp_bt_ptr,  # int32 [num_state_slots, max_blocks]
    out_ptr,  # int32 [bs, EXTRA_TOPK]
    out_lengths_ptr,  # int32 [bs]
    BS: tl.constexpr,
    EXTRA_TOPK: tl.constexpr,
    PAGE_SIZE_C: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    COMP_BT_STRIDE: tl.constexpr,
):
    """One program per sequence. Computes:

    cur_count   = compressed_counts[seq_slots[b]]
    length      = min(cur_count, EXTRA_TOPK)
    out_len[b]  = length
    for t in [0, EXTRA_TOPK):
        block_idx     = t // PAGE_SIZE_C
        tok_in_block  = t % PAGE_SIZE_C
        block_safe    = min(block_idx, MAX_BLOCKS - 1)
        page_id       = comp_bt[seq_slots[b], block_safe]
        phys          = page_id * PAGE_SIZE_C + tok_in_block
        out[b, t]     = phys if t < length else -1
    """
    bs_id = tl.program_id(0)
    tok = tl.arange(0, EXTRA_TOPK)

    seq_slot = tl.load(seq_slots_ptr + bs_id).to(tl.int64)
    cur_count = tl.load(compressed_counts_ptr + seq_slot).to(tl.int32)
    length = tl.minimum(cur_count, EXTRA_TOPK)
    if bs_id < BS:
        tl.store(out_lengths_ptr + bs_id, length)

    block_idx = tok // PAGE_SIZE_C
    tok_in_block = tok % PAGE_SIZE_C
    block_safe = tl.minimum(block_idx, MAX_BLOCKS - 1)
    # Gather page_ids: comp_bt is [num_state_slots, MAX_BLOCKS], index by
    # (seq_slot, block_safe). Compute flat offset using stride.
    page_ids = tl.load(comp_bt_ptr + seq_slot * COMP_BT_STRIDE + block_safe).to(
        tl.int32
    )
    phys = page_ids * PAGE_SIZE_C + tok_in_block
    valid = tok < length
    out = tl.where(valid, phys, -1)
    tl.store(out_ptr + bs_id * EXTRA_TOPK + tok, out)


def build_extra_indices_paged(
    seq_slots: torch.Tensor,  # [bs] int64
    compressed_counts: torch.Tensor,  # [num_state_slots] int32
    comp_bt: torch.Tensor,  # [num_state_slots, max_blocks] int32
    extra_topk: int,
    page_size_c: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for the paged extra-indices block in
    ``_decode_attention_flash_mla`` (deepseek_v4.py:1757-1785).

    Returns ``(extra_indices [bs, extra_topk], extra_topk_lengths [bs])``,
    both int32. Caller unsqueeze(1) if needed.
    """
    bs = seq_slots.shape[0]
    device = seq_slots.device
    out = torch.empty(bs, extra_topk, dtype=torch.int32, device=device)
    out_lengths = torch.empty(bs, dtype=torch.int32, device=device)
    grid = (bs,)
    _build_extra_indices_paged_kernel[grid](
        seq_slots.contiguous(),
        compressed_counts.contiguous(),
        comp_bt.contiguous(),
        out,
        out_lengths,
        BS=bs,
        EXTRA_TOPK=extra_topk,
        PAGE_SIZE_C=page_size_c,
        MAX_BLOCKS=comp_bt.shape[1],
        COMP_BT_STRIDE=comp_bt.stride(0),
    )
    return out, out_lengths
