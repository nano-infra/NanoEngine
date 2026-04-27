"""Fused UE8M0 FP8 quantization kernels (DSV4 KV-cache layout).

Replaces the eager-PyTorch op chains in ``deepseek_v4.py``:

    - ``_pack_kv_fp8``                   → ``pack_kv_fp8``
    - ``_fp8_quant_dequant_inplace``     → ``fp8_quant_dequant_inplace``

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
    UE8M0 quant → fp8e4m3 → fp32 → bf16, written back in place."""
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
