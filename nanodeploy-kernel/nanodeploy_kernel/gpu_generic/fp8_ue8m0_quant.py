"""Fused UE8M0 FP8 quantization kernels (DSV4 KV-cache layout).

Replaces the eager-PyTorch op chains in ``deepseek_v4.py``:

    - ``_pack_kv_fp8``                   → ``pack_kv_fp8``
    - ``_fp8_quant_dequant_inplace``     → ``fp8_quant_dequant_inplace``
    - ``_pack_kv_fp8 + _store_dsv4_fp8_batched``
                                         → ``store_dsv4_kv_fp8_fused``

Each eager path was: ``view.abs().amax(-1)`` (1 reduce) + ``exp2 / ceil
/ log2 / div / clamp / cast`` (7+ elementwise) per token per block.
With 60 layers × per-token quant in the compressor write path, that
chain dominates the ``at::native::reduce_kernel`` and ``at::native::
elementwise_kernel`` events seen in the profiler.

The UE8M0 scale format encodes the scale as a power of two:
``scale = 2^E`` where ``E = ceil(log2(amax / fp8_max))``. The exponent
``E`` is stored as ``uint8 = E + 127`` (offset binary, like an FP32
exponent field). The dequant cost is just one multiply by ``scale``.
"""

import torch
import triton
import triton.language as tl

_FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max
_UE8M0_BIAS = 127  # IEEE-754-fp32-style exponent bias


@triton.jit
def _ue8m0_quant_dequant_inplace_kernel(
    x_ptr,
    LAST_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    """One program per (row, block). Read BLOCK_SIZE bf16 → fp32 →
    UE8M0 quant → fp8e4m3 → fp32 → bf16, written back in place.

    Note: this DELIBERATELY round-trips through fp8 (not bf16), even
    though reference's ``act_quant(inplace=True)`` is a no-op for ue8m0
    inputs. The reason: reference's KV cache is BF16 end-to-end, so its
    attention reads BF16 values; NanoDeploy's KV cache is FP8 (paged),
    so attention reads FP8 values via flash_mla. Pre-quantizing the
    local kv tensor here means the bf16 view (used by the eager-
    fallback ``_attend_sparse_one`` path during graph capture and
    elsewhere) sees the same FP8-quant noise the cache writer would
    introduce, keeping the two attention paths numerically consistent.
    Removing this round-trip causes the eager-fallback path to disagree
    with the flash_mla cached path and regresses end-to-end greedy
    parity (verified empirically: removing the round-trip moved the
    first-divergence token from 128 to 52)."""
    pid_row = tl.program_id(0)
    pid_blk = tl.program_id(1)

    cols = tl.arange(0, BLOCK_SIZE)
    offset = pid_row * LAST_DIM + pid_blk * BLOCK_SIZE
    x = tl.load(x_ptr + offset + cols).to(tl.float32)

    amax = tl.max(tl.abs(x), axis=0)
    amax = tl.maximum(amax, EPS)
    exponent = tl.ceil(tl.log2(amax / FP8_MAX))
    scale = tl.exp2(exponent)

    quant = x / scale
    quant = tl.minimum(tl.maximum(quant, -FP8_MAX), FP8_MAX)
    fp8 = quant.to(tl.float8e4nv)  # cast to fp8 e4m3
    dequant = fp8.to(tl.float32) * scale  # straight back

    tl.store(x_ptr + offset + cols, dequant.to(tl.bfloat16))


def fp8_quant_dequant_inplace(
    x: torch.Tensor, block_size: int = 64, eps: float = 1e-4
) -> torch.Tensor:
    """Drop-in replacement for ``_fp8_quant_dequant_inplace`` in
    ``deepseek_v4.py``. ``x`` is bf16 on CUDA, last-dim divisible by
    ``block_size``. Mutated in place; returns ``x``.
    """
    if x.numel() == 0:
        return x
    last_dim = x.shape[-1]
    if last_dim % block_size != 0:
        raise ValueError(
            f"fp8_quant_dequant_inplace: last_dim {last_dim} not divisible "
            f"by block_size {block_size}"
        )
    n_rows = x.numel() // last_dim
    num_blocks = last_dim // block_size

    # The kernel writes back via ``x``'s storage; the input must be
    # contiguous so reshape doesn't materialise a copy.
    if not x.is_contiguous():
        raise RuntimeError("fp8_quant_dequant_inplace requires contiguous x")

    x_flat = x.view(n_rows, last_dim)
    grid = (n_rows, num_blocks)
    _ue8m0_quant_dequant_inplace_kernel[grid](
        x_flat,
        LAST_DIM=last_dim,
        BLOCK_SIZE=block_size,
        FP8_MAX=_FP8_MAX,
        EPS=eps,
    )
    return x


@triton.jit
def _ue8m0_pack_fp8_kernel(
    kv_ptr,
    nope_fp8_ptr,
    scales_u8_ptr,
    KV_STRIDE_ROW,
    NOPE_DIM: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
    BIAS: tl.constexpr,
):
    """One program per (row, tile). Reads one nope tile (TILE_SIZE bf16
    elements), computes UE8M0 scale, casts to fp8, writes both fp8 data
    and the uint8 scale exponent."""
    pid_row = tl.program_id(0)
    pid_tile = tl.program_id(1)

    cols = tl.arange(0, TILE_SIZE)
    in_offset = pid_row * KV_STRIDE_ROW + pid_tile * TILE_SIZE
    out_offset = pid_row * NOPE_DIM + pid_tile * TILE_SIZE

    x = tl.load(kv_ptr + in_offset + cols).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    amax = tl.maximum(amax, EPS)
    exponent = tl.ceil(tl.log2(amax / FP8_MAX))
    scale = tl.exp2(exponent)

    quant = x / scale
    quant = tl.minimum(tl.maximum(quant, -FP8_MAX), FP8_MAX)
    tl.store(nope_fp8_ptr + out_offset + cols, quant.to(tl.float8e4nv))

    # One scale per tile. ``exponent`` is a per-program scalar after the
    # reduction. Broadcast the store to all lanes but mask so only lane
    # 0 actually issues — equivalent to a single scalar store, but the
    # tile-shape store (``cols`` array) is what triton's IR can express.
    scale_u8 = (exponent.to(tl.int32) + BIAS).to(tl.uint8)
    scale_offsets = pid_row * NUM_TILES + pid_tile + cols * 0
    tl.store(scales_u8_ptr + scale_offsets, scale_u8, mask=cols == 0)


def pack_kv_fp8(
    kv_bf16: torch.Tensor,
    nope_dim: int = 448,
    rope_dim: int = 64,
    tile_size: int = 64,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in replacement for ``_pack_kv_fp8`` in ``deepseek_v4.py``.

    Returns ``(nope_fp8, rope_bf16, scales_u8)`` with shapes
    ``([T, nope_dim], [T, rope_dim], [T, num_tiles])``.
    """
    assert kv_bf16.dtype == torch.bfloat16
    assert kv_bf16.shape[-1] == nope_dim + rope_dim
    if nope_dim % tile_size != 0:
        raise ValueError(
            f"pack_kv_fp8: nope_dim {nope_dim} not divisible by tile_size {tile_size}"
        )
    n_rows = kv_bf16.numel() // (nope_dim + rope_dim)
    num_tiles = nope_dim // tile_size
    kv_flat = kv_bf16.reshape(n_rows, nope_dim + rope_dim).contiguous()

    nope_fp8 = torch.empty(
        n_rows, nope_dim, dtype=torch.float8_e4m3fn, device=kv_bf16.device
    )
    scales_u8 = torch.empty(n_rows, num_tiles, dtype=torch.uint8, device=kv_bf16.device)
    # Rope is just the trailing slice of the input — one memcpy, no
    # need for a fused kernel.
    rope_bf16 = kv_flat[:, nope_dim:].contiguous()

    grid = (n_rows, num_tiles)
    _ue8m0_pack_fp8_kernel[grid](
        kv_flat,
        nope_fp8,
        scales_u8,
        kv_flat.stride(0),
        NOPE_DIM=nope_dim,
        TILE_SIZE=tile_size,
        NUM_TILES=num_tiles,
        FP8_MAX=_FP8_MAX,
        EPS=eps,
        BIAS=_UE8M0_BIAS,
    )
    return nope_fp8, rope_bf16, scales_u8


@triton.jit
def _store_dsv4_fp8_kernel(
    kv_ptr,  # bf16 [T, NOPE_DIM + ROPE_DIM] contiguous
    cache_fp8_ptr,  # cache_buf viewed as torch.float8_e4m3fn (1 byte/elt)
    cache_bf16_ptr,  # cache_buf viewed as torch.bfloat16     (2 bytes/elt)
    cache_u8_ptr,  # cache_buf viewed as torch.uint8        (1 byte/elt)
    slot_mapping_ptr,  # int64 [T]; -1 → DUMMY_SLOT
    DUMMY_SLOT,
    PAGE_SIZE: tl.constexpr,
    NOPE_DIM: tl.constexpr,  # 448
    ROPE_DIM: tl.constexpr,  # 64 (bf16 elements)
    TILE_SIZE: tl.constexpr,  # 64 (nope fp8 elements per tile)
    NUM_NOPE_TILES: tl.constexpr,  # 7
    BYTES_PER_TOKEN: tl.constexpr,  # 584 (576 nope+rope + 8 padded scales)
    NOPE_ROPE_BYTES: tl.constexpr,  # 576
    SCALE_PADDED: tl.constexpr,  # 8 (scales per token, padded)
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
    BIAS: tl.constexpr,
):
    """Pack-and-scatter DSV4 KV cache in one launch.

    Grid: ``(T, NUM_NOPE_TILES + 1)``. Programs ``[0, NUM_NOPE_TILES)``
    each handle one nope tile (UE8M0 fp8 quant + scale-byte write).
    Program ``NUM_NOPE_TILES`` copies the bf16 rope segment.

    Page layout (matches ``_store_dsv4_fp8_batched`` byte addressing):
        page_base + tok_in_page * NOPE_ROPE_BYTES   → nope (448 fp8) + rope (64 bf16 = 128 B)
        page_base + PAGE_SIZE * NOPE_ROPE_BYTES
            + tok_in_page * SCALE_PADDED            → 7 UE8M0 scale bytes (+1 pad)
    """
    pid_tok = tl.program_id(0)
    pid_tile = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + pid_tok)
    slot = tl.where(slot >= 0, slot, DUMMY_SLOT)

    page_idx = slot // PAGE_SIZE
    tok_in_page = slot % PAGE_SIZE
    page_byte_base = page_idx * (PAGE_SIZE * BYTES_PER_TOKEN)
    token_byte_base = page_byte_base + tok_in_page * NOPE_ROPE_BYTES

    KV_STRIDE: tl.constexpr = NOPE_DIM + ROPE_DIM

    if pid_tile < NUM_NOPE_TILES:
        cols = tl.arange(0, TILE_SIZE)
        in_off = pid_tok * KV_STRIDE + pid_tile * TILE_SIZE
        x = tl.load(kv_ptr + in_off + cols).to(tl.float32)

        amax = tl.max(tl.abs(x), axis=0)
        amax = tl.maximum(amax, EPS)
        exponent = tl.ceil(tl.log2(amax / FP8_MAX))
        scale = tl.exp2(exponent)
        quant = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)

        # FP8 nope tile: byte offset = token_byte_base + pid_tile * 64.
        # cache_fp8_ptr has 1 byte/elt so byte offset == element offset.
        nope_off = token_byte_base + pid_tile * TILE_SIZE
        tl.store(cache_fp8_ptr + nope_off + cols, quant.to(tl.float8e4nv))

        # One UE8M0 scale byte per tile.
        scale_u8 = (exponent.to(tl.int32) + BIAS).to(tl.uint8)
        scale_off = (
            page_byte_base
            + PAGE_SIZE * NOPE_ROPE_BYTES
            + tok_in_page * SCALE_PADDED
            + pid_tile
        )
        tl.store(cache_u8_ptr + scale_off + cols * 0, scale_u8, mask=cols == 0)
    else:
        # Rope copy. cache_bf16_ptr is the cache_buf reinterpreted as
        # bf16, so element offset = byte_offset / 2 (all rope writes
        # land at 2-byte-aligned offsets — verified in the wrapper).
        cols = tl.arange(0, ROPE_DIM)
        in_off = pid_tok * KV_STRIDE + NOPE_DIM
        rope = tl.load(kv_ptr + in_off + cols)

        rope_bf16_off = (token_byte_base + NOPE_DIM) // 2
        tl.store(cache_bf16_ptr + rope_bf16_off + cols, rope)


def store_dsv4_kv_fp8_fused(
    kv_bf16: torch.Tensor,
    cache_buf: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_size: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    tile_size: int = 64,
    eps: float = 1e-8,
) -> None:
    """One-launch replacement for ``_pack_kv_fp8 + _store_dsv4_fp8_batched``.

    Mutates ``cache_buf`` in place. The torch fallback in
    ``deepseek_v4.py`` performs ~12 elementwise/reduce launches per
    call; this kernel folds them into one ``(T, NUM_TILES + 1)``-grid
    triton launch.

    Layout invariants (asserted in the caller):
      - ``kv_bf16``    : ``[T, nope_dim + rope_dim]`` bf16 contiguous
      - ``cache_buf``  : ``[num_pages, page_size, 1, BYTES_PER_TOKEN]`` uint8
                         where ``BYTES_PER_TOKEN = nope_dim + rope_dim*2 +
                         (num_tiles + 1)``. Caller MUST reserve a dummy
                         last slot for graph-safe ``slot_mapping == -1``
                         redirects.
      - ``slot_mapping``: ``[T]`` int (any signed dtype). ``-1`` → dummy
                          slot (last slot of last page).
    """
    if kv_bf16.numel() == 0:
        return
    assert kv_bf16.dtype == torch.bfloat16, kv_bf16.dtype
    assert cache_buf.dtype == torch.uint8, cache_buf.dtype
    assert kv_bf16.shape[-1] == nope_dim + rope_dim
    assert kv_bf16.is_contiguous()
    assert cache_buf.is_contiguous()
    if nope_dim % tile_size != 0:
        raise ValueError(
            f"store_dsv4_kv_fp8_fused: nope_dim {nope_dim} not divisible "
            f"by tile_size {tile_size}"
        )

    num_tiles = nope_dim // tile_size
    nope_rope_bytes = nope_dim + rope_dim * 2
    scale_padded = num_tiles + 1
    bytes_per_token = nope_rope_bytes + scale_padded

    # Sanity-check the layout: the kernel assumes ``cache_buf`` has the
    # exact byte stride the eager path documents.
    assert cache_buf.shape[-1] == bytes_per_token, (
        f"cache_buf last-dim {cache_buf.shape[-1]} != expected "
        f"BYTES_PER_TOKEN {bytes_per_token}"
    )

    # bf16 view requires the byte offset to be even. ``nope_dim`` is
    # 448 (even) and ``nope_rope_bytes`` is 576 (even), so all rope
    # writes are 2-byte aligned. Page byte offsets are
    # ``page_idx * page_size * 584`` — also even.
    assert nope_dim % 2 == 0 and nope_rope_bytes % 2 == 0

    T = kv_bf16.shape[0]
    num_pages = cache_buf.shape[0]
    total_slots = num_pages * page_size
    dummy_slot = total_slots - 1

    cache_flat = cache_buf.view(-1)
    cache_fp8 = cache_flat.view(torch.float8_e4m3fn)
    cache_bf16 = cache_flat.view(torch.bfloat16)

    slots_i64 = slot_mapping.to(torch.int64)

    grid = (T, num_tiles + 1)
    _store_dsv4_fp8_kernel[grid](
        kv_bf16,
        cache_fp8,
        cache_bf16,
        cache_flat,
        slots_i64,
        dummy_slot,
        PAGE_SIZE=page_size,
        NOPE_DIM=nope_dim,
        ROPE_DIM=rope_dim,
        TILE_SIZE=tile_size,
        NUM_NOPE_TILES=num_tiles,
        BYTES_PER_TOKEN=bytes_per_token,
        NOPE_ROPE_BYTES=nope_rope_bytes,
        SCALE_PADDED=scale_padded,
        FP8_MAX=_FP8_MAX,
        EPS=eps,
        BIAS=_UE8M0_BIAS,
    )
