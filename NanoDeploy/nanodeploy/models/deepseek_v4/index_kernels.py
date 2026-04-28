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


# ─── Compressor post-shift (overlap case) ──────────────────────────────────


@triton.jit
def _compress_post_shift_overlap_kernel(
    kv_states_ptr,  # fp32 [n_slots, 2*ratio, 2*head_dim]
    score_states_ptr,  # fp32 [n_slots, 2*ratio, 2*head_dim]
    seq_slots_ptr,  # int64 [bs]
    should_compress_ptr,  # bool [bs]
    BS: tl.constexpr,
    RATIO: tl.constexpr,
    TWO_HEAD_DIM: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    KV_STRIDE_0: tl.constexpr,
    KV_STRIDE_1: tl.constexpr,
):
    """In-place post-shift: when should_compress[b] is True, copy
    ``_kv_states[seq_slots[b], ratio:2*ratio, :]`` into
    ``_kv_states[seq_slots[b], 0:ratio, :]`` (and same for _score_states).
    When False, leave the row unchanged.

    Replaces the eager 8-kernel chain (4 fancy gathers + 2 ``torch.where``
    + 2 fancy index_puts) with one Triton launch. Pure load/where/store —
    no reductions, no shared-memory ops, so safe to coexist with tilelang
    HC kernels.
    """
    bs_id = tl.program_id(0)
    r_id = tl.program_id(1)
    chunk_id = tl.program_id(2)

    h_start = chunk_id * BLOCK_HD
    h = h_start + tl.arange(0, BLOCK_HD)
    h_mask = h < TWO_HEAD_DIM

    seq_slot = tl.load(seq_slots_ptr + bs_id).to(tl.int64)
    sc = tl.load(should_compress_ptr + bs_id).to(tl.int1)

    base = seq_slot * KV_STRIDE_0
    src_off = base + (RATIO + r_id) * KV_STRIDE_1 + h
    dst_off = base + r_id * KV_STRIDE_1 + h

    shifted_kv = tl.load(kv_states_ptr + src_off, mask=h_mask)
    current_kv = tl.load(kv_states_ptr + dst_off, mask=h_mask)
    tl.store(
        kv_states_ptr + dst_off,
        tl.where(sc, shifted_kv, current_kv),
        mask=h_mask,
    )

    shifted_sc = tl.load(score_states_ptr + src_off, mask=h_mask)
    current_sc = tl.load(score_states_ptr + dst_off, mask=h_mask)
    tl.store(
        score_states_ptr + dst_off,
        tl.where(sc, shifted_sc, current_sc),
        mask=h_mask,
    )


def compress_post_shift_overlap(
    kv_states: torch.Tensor,
    score_states: torch.Tensor,
    seq_slots: torch.Tensor,
    should_compress: torch.Tensor,
    ratio: int,
) -> None:
    """In-place post-shift on _kv_states / _score_states (overlap case)."""
    bs = seq_slots.shape[0]
    n_rows = kv_states.shape[1]
    two_head_dim = kv_states.shape[2]
    assert n_rows == 2 * ratio
    BLOCK_HD = 256
    n_chunks = (two_head_dim + BLOCK_HD - 1) // BLOCK_HD
    grid = (bs, ratio, n_chunks)
    _compress_post_shift_overlap_kernel[grid](
        kv_states,
        score_states,
        seq_slots.contiguous(),
        should_compress.contiguous(),
        BS=bs,
        RATIO=ratio,
        TWO_HEAD_DIM=two_head_dim,
        BLOCK_HD=BLOCK_HD,
        KV_STRIDE_0=kv_states.stride(0),
        KV_STRIDE_1=kv_states.stride(1),
    )


# ─── Physical slots compute (paged) ────────────────────────────────────────


@triton.jit
def _compress_physical_slots_paged_kernel(
    compressed_counts_ptr,  # int32 [n_state_slots]
    seq_slots_ptr,  # int64 [bs]
    comp_bt_ptr,  # int32 [n_state_slots, max_blocks]
    should_compress_ptr,  # bool [bs]
    out_ptr,  # int64 [bs]
    BS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    DUMMY_SLOT: tl.constexpr,
    COMP_BT_STRIDE: tl.constexpr,
):
    """One program per batch entry. Computes:

    cur_count    = compressed_counts[seq_slots[b]]
    block_idx    = cur_count // page_size
    tok_in_block = cur_count % page_size
    block_safe   = min(block_idx, max_blocks - 1)
    page_id      = comp_bt[seq_slots[b], block_safe]
    phys         = page_id * page_size + tok_in_block
    out[b]       = phys if should_compress[b] else dummy_slot
    """
    bs_id = tl.program_id(0)
    if bs_id >= BS:
        return

    seq_slot = tl.load(seq_slots_ptr + bs_id).to(tl.int64)
    cur_count = tl.load(compressed_counts_ptr + seq_slot).to(tl.int64)
    block_idx = cur_count // PAGE_SIZE
    tok_in_block = cur_count % PAGE_SIZE
    block_idx_safe = tl.minimum(block_idx, MAX_BLOCKS - 1)
    page_id = tl.load(comp_bt_ptr + seq_slot * COMP_BT_STRIDE + block_idx_safe).to(
        tl.int64
    )
    phys = page_id * PAGE_SIZE + tok_in_block
    sc = tl.load(should_compress_ptr + bs_id).to(tl.int1)
    out_val = tl.where(sc, phys, DUMMY_SLOT)
    tl.store(out_ptr + bs_id, out_val)


@triton.jit
def _compute_compress_metadata_kernel(
    positions_ptr,  # int64 [bs]
    compressed_pos_ptr,  # int64 [bs] output
    should_compress_ptr,  # bool  [bs] output
    BS: tl.constexpr,
    RATIO: tl.constexpr,
):
    """Fuse the per-step compressor metadata math:

        next_pos = positions + 1
        compressed_pos = max(next_pos - ratio, 0)
        should_compress = (next_pos % ratio) == 0

    Replaces ~5 small elementwise launches (add, sub, clamp, mod, eq)
    with one Triton kernel.
    """
    bs_id = tl.program_id(0)
    if bs_id >= BS:
        return
    pos = tl.load(positions_ptr + bs_id).to(tl.int64)
    next_pos = pos + 1
    cp = tl.maximum(next_pos - RATIO, 0)
    sc = (next_pos % RATIO) == 0
    tl.store(compressed_pos_ptr + bs_id, cp)
    tl.store(should_compress_ptr + bs_id, sc)


def compute_compress_metadata(
    positions: torch.Tensor, ratio: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (compressed_pos[bs] int64, should_compress[bs] bool)."""
    bs = positions.shape[0]
    cp = torch.empty(bs, dtype=torch.int64, device=positions.device)
    sc = torch.empty(bs, dtype=torch.bool, device=positions.device)
    _compute_compress_metadata_kernel[(bs,)](
        positions.contiguous(), cp, sc, BS=bs, RATIO=ratio
    )
    return cp, sc


@triton.jit
def _compress_counts_update_kernel(
    counts_ptr,  # int32 [n_slots] (in-place, atomic increment)
    seq_slots_ptr,  # int64 [bs]
    should_compress_ptr,  # bool  [bs]
    BS: tl.constexpr,
):
    """Fuse ``inc = should_compress.to(int32); counts.scatter_add_(0, seq_slots, inc)``
    (1 cast + 1 scatter_add) into one atomic-add kernel.
    """
    bs_id = tl.program_id(0)
    if bs_id >= BS:
        return
    sc = tl.load(should_compress_ptr + bs_id).to(tl.int1)
    if sc:
        seq_slot = tl.load(seq_slots_ptr + bs_id).to(tl.int64)
        tl.atomic_add(counts_ptr + seq_slot, 1)


def compress_counts_update(
    counts: torch.Tensor,
    seq_slots: torch.Tensor,
    should_compress: torch.Tensor,
) -> None:
    """In-place atomic increment of counts[seq_slots] where should_compress is True."""
    bs = seq_slots.shape[0]
    _compress_counts_update_kernel[(bs,)](
        counts,
        seq_slots.contiguous(),
        should_compress.contiguous(),
        BS=bs,
    )


@triton.jit
def _compress_scatter_update_kernel(
    kv_states_ptr,  # fp32 [n_slots, 2*ratio, coeff*head_dim]
    score_states_ptr,  # fp32 [n_slots, 2*ratio, coeff*head_dim]
    kv_all_ptr,  # fp32 [bs, coeff*head_dim]
    score_all_ptr,  # fp32 [bs, coeff*head_dim]
    ape_ptr,  # fp32 [ratio, coeff*head_dim]
    seq_slots_ptr,  # int64 [bs]
    positions_ptr,  # int64 [bs]
    BS: tl.constexpr,
    RATIO: tl.constexpr,
    COEFF_HEAD_DIM: tl.constexpr,
    OVERLAP: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    KV_STRIDE_0: tl.constexpr,
    KV_STRIDE_1: tl.constexpr,
    APE_STRIDE_0: tl.constexpr,
):
    """Fused scatter-update for the compressor's per-step kv/score write.

    Replaces 6 eager kernels:
      pos_mod = positions % ratio
      ape_vals = ape[pos_mod]
      update_idx = ratio + pos_mod   (overlap) or pos_mod
      _kv_states[seq_slots, update_idx]    = kv_all
      _score_states[seq_slots, update_idx] = score_all + ape_vals

    Pure load/elementwise/store — no reductions, safe alongside tilelang.
    """
    bs_id = tl.program_id(0)
    chunk_id = tl.program_id(1)
    h_start = chunk_id * BLOCK_HD
    h = h_start + tl.arange(0, BLOCK_HD)
    h_mask = h < COEFF_HEAD_DIM

    seq_slot = tl.load(seq_slots_ptr + bs_id).to(tl.int64)
    pos = tl.load(positions_ptr + bs_id).to(tl.int64)
    pos_mod = pos % RATIO
    if OVERLAP:
        update_idx = RATIO + pos_mod
    else:
        update_idx = pos_mod

    kv = tl.load(kv_all_ptr + bs_id * COEFF_HEAD_DIM + h, mask=h_mask)
    score = tl.load(score_all_ptr + bs_id * COEFF_HEAD_DIM + h, mask=h_mask)
    ape = tl.load(ape_ptr + pos_mod * APE_STRIDE_0 + h, mask=h_mask)

    kv_dst = kv_states_ptr + seq_slot * KV_STRIDE_0 + update_idx * KV_STRIDE_1 + h
    sc_dst = score_states_ptr + seq_slot * KV_STRIDE_0 + update_idx * KV_STRIDE_1 + h
    tl.store(kv_dst, kv, mask=h_mask)
    tl.store(sc_dst, score + ape, mask=h_mask)


def compress_scatter_update(
    kv_states: torch.Tensor,
    score_states: torch.Tensor,
    kv_all: torch.Tensor,
    score_all: torch.Tensor,
    ape: torch.Tensor,
    seq_slots: torch.Tensor,
    positions: torch.Tensor,
    ratio: int,
    overlap: bool,
) -> None:
    """In-place fused scatter update on _kv_states/_score_states."""
    bs = seq_slots.shape[0]
    coeff_head_dim = kv_all.shape[1]
    BLOCK_HD = 256
    n_chunks = (coeff_head_dim + BLOCK_HD - 1) // BLOCK_HD
    grid = (bs, n_chunks)
    _compress_scatter_update_kernel[grid](
        kv_states,
        score_states,
        kv_all.contiguous(),
        score_all.contiguous(),
        ape.contiguous(),
        seq_slots.contiguous(),
        positions.contiguous(),
        BS=bs,
        RATIO=ratio,
        COEFF_HEAD_DIM=coeff_head_dim,
        OVERLAP=overlap,
        BLOCK_HD=BLOCK_HD,
        KV_STRIDE_0=kv_states.stride(0),
        KV_STRIDE_1=kv_states.stride(1),
        APE_STRIDE_0=ape.stride(0),
    )


def compress_physical_slots_paged(
    compressed_counts: torch.Tensor,  # int32 [n_state_slots]
    seq_slots: torch.Tensor,  # int64 [bs]
    compressed_block_table: torch.Tensor,  # int32 [n_state_slots, max_blocks]
    should_compress: torch.Tensor,  # bool [bs]
    page_size: int,
    dummy_slot: int,
) -> torch.Tensor:
    """Returns physical_slots [bs] int64."""
    bs = seq_slots.shape[0]
    out = torch.empty(bs, dtype=torch.int64, device=seq_slots.device)
    grid = (bs,)
    _compress_physical_slots_paged_kernel[grid](
        compressed_counts.contiguous(),
        seq_slots.contiguous(),
        compressed_block_table.contiguous(),
        should_compress.contiguous(),
        out,
        BS=bs,
        PAGE_SIZE=page_size,
        MAX_BLOCKS=compressed_block_table.shape[1],
        DUMMY_SLOT=dummy_slot,
        COMP_BT_STRIDE=compressed_block_table.stride(0),
    )
    return out
