"""FP8 MLA KV cache quantization utilities for DeepSeek V3.2.

Per-token layout (656 bytes, V32_FP8Sparse):
  [0..511]    512 bytes  float8_e4m3fn  — quantized NoPE (4 tiles × 128)
  [512..527]   16 bytes  4 × float32    — per-tile scale factors (UE8M0)
  [528..655]  128 bytes  64 × bfloat16  — RoPE (kept in original precision)
"""

import torch
import triton
import triton.language as tl

# V32 FP8 MLA constants
D_NOPE = 512
D_ROPE = 64
D_TOTAL = D_NOPE + D_ROPE  # 576
TILE_SIZE = 128
NUM_TILES = D_NOPE // TILE_SIZE  # 4
SCALE_BYTES = NUM_TILES * 4  # 16 bytes (4 × float32)
ROPE_BYTES = D_ROPE * 2  # 128 bytes (64 × bfloat16)
FP8_BYTES_PER_TOKEN = D_NOPE + SCALE_BYTES + ROPE_BYTES  # 656

# FP8 E4M3 max representable value
_FP8_E4M3_MAX = 448.0


def _cast_to_ue8m0(scale_inv: torch.Tensor) -> torch.Tensor:
    """Round scale to nearest power-of-2 (UE8M0 format)."""
    return torch.pow(2, torch.clamp_min(scale_inv, 1e-12).log2().ceil())


def quantize_nope_fp8(
    nope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize NoPE part of MLA compressed KV to FP8 with per-tile UE8M0 scales.

    Args:
        nope: [N, 512] bfloat16 — the NoPE portion of compressed KV.

    Returns:
        fp8_nope: [N, 512] float8_e4m3fn
        scales:   [N, 4] float32 — per-tile scale factors
    """
    N = nope.shape[0]
    assert nope.shape[-1] == D_NOPE

    nope_f32 = nope.float()
    nope_reshaped = nope_f32.view(N, NUM_TILES, TILE_SIZE)

    # Per-tile absmax / 448.0 → UE8M0
    absmax = nope_reshaped.abs().amax(dim=-1)  # [N, 4]
    scales = _cast_to_ue8m0(absmax / _FP8_E4M3_MAX)  # [N, 4]

    # Quantize: val / scale → fp8
    fp8_nope = (nope_reshaped / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
    fp8_nope = fp8_nope.view(N, D_NOPE)

    return fp8_nope, scales


def dequantize_nope_fp8(
    fp8_nope: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Dequantize FP8 NoPE back to bfloat16.

    Args:
        fp8_nope: [N, 512] float8_e4m3fn
        scales:   [N, 4] float32

    Returns:
        nope: [N, 512] bfloat16
    """
    N = fp8_nope.shape[0]
    nope_f32 = fp8_nope.view(N, NUM_TILES, TILE_SIZE).float()
    nope_f32 = nope_f32 * scales.unsqueeze(-1)
    return nope_f32.view(N, D_NOPE).to(torch.bfloat16)


def pack_mla_fp8(
    fp8_nope: torch.Tensor,
    scales: torch.Tensor,
    rope: torch.Tensor,
) -> torch.Tensor:
    """Pack FP8 NoPE + scales + BF16 RoPE into a flat uint8 buffer.

    Args:
        fp8_nope: [N, 512] float8_e4m3fn
        scales:   [N, 4] float32
        rope:     [N, 64] bfloat16

    Returns:
        packed: [N, 656] uint8
    """
    N = fp8_nope.shape[0]
    device = fp8_nope.device
    packed = torch.empty(N, FP8_BYTES_PER_TOKEN, dtype=torch.uint8, device=device)

    # NoPE: 512 bytes (fp8 → uint8 view)
    packed[:, :D_NOPE] = fp8_nope.view(torch.uint8)

    # Scales: 16 bytes (float32 → uint8 view)
    packed[:, D_NOPE : D_NOPE + SCALE_BYTES] = scales.contiguous().view(torch.uint8)

    # RoPE: 128 bytes (bfloat16 → uint8 view)
    packed[:, D_NOPE + SCALE_BYTES :] = rope.contiguous().view(torch.uint8)

    return packed


def unpack_mla_fp8(
    packed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack flat uint8 buffer into FP8 NoPE + scales + BF16 RoPE.

    Args:
        packed: [N, 656] uint8

    Returns:
        fp8_nope: [N, 512] float8_e4m3fn
        scales:   [N, 4] float32
        rope:     [N, 64] bfloat16
    """
    fp8_nope = packed[:, :D_NOPE].view(torch.float8_e4m3fn)
    scales = packed[:, D_NOPE : D_NOPE + SCALE_BYTES].view(torch.float32)
    rope = packed[:, D_NOPE + SCALE_BYTES :].view(torch.bfloat16)
    return fp8_nope, scales, rope


def quantize_and_pack_mla(
    compressed_kv: torch.Tensor,
) -> torch.Tensor:
    """Quantize compressed KV [N, 1, 576] bf16 → packed [N, 1, 656] uint8.

    Splits into NoPE [N, 512] and RoPE [N, 64], quantizes NoPE, packs all.
    """
    if compressed_kv.ndim == 3:
        N = compressed_kv.shape[0]
        compressed_kv_2d = compressed_kv.view(N, D_TOTAL)
    else:
        N = compressed_kv.shape[0]
        compressed_kv_2d = compressed_kv

    nope = compressed_kv_2d[:, :D_NOPE]
    rope = compressed_kv_2d[:, D_NOPE:]

    fp8_nope, scales = quantize_nope_fp8(nope)
    packed = pack_mla_fp8(fp8_nope, scales, rope)

    return packed.view(N, 1, FP8_BYTES_PER_TOKEN)


def dequantize_and_unpack_mla(
    packed: torch.Tensor,
) -> torch.Tensor:
    """Unpack + dequantize packed [N, 1, 656] uint8 → BF16 [N, 1, 576].

    Returns full compressed KV (NoPE + RoPE) in bfloat16.
    """
    if packed.ndim == 3:
        N = packed.shape[0]
        packed_2d = packed.view(N, FP8_BYTES_PER_TOKEN)
    else:
        packed_2d = packed

    fp8_nope, scales, rope = unpack_mla_fp8(packed_2d)
    nope = dequantize_nope_fp8(fp8_nope, scales)

    result = torch.empty(
        packed_2d.shape[0], D_TOTAL, dtype=torch.bfloat16, device=packed.device
    )
    result[:, :D_NOPE] = nope
    result[:, D_NOPE:] = rope

    if packed.ndim == 3:
        return result.view(packed_2d.shape[0], 1, D_TOTAL)
    return result


# ---------- Triton kernel: FP8 quantize + scatter to paged cache ----------


@triton.jit
def _store_kcache_fp8_kernel(
    # Pointers
    kv_ptr,  # [N, D_TOTAL] bfloat16 source (NoPE+RoPE)
    cache_ptr,  # flat paged cache, uint8
    slot_mapping_ptr,
    # Strides
    kv_row_stride,
    # Constants
    BYTES_PER_TOKEN: tl.constexpr,
    D_NOPE_C: tl.constexpr,
    D_ROPE_C: tl.constexpr,
    TILE_SIZE_C: tl.constexpr,
    NUM_TILES_C: tl.constexpr,
    SCALE_BYTES_C: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + pid)
    if slot == -1:
        return

    # -- Read NoPE [D_NOPE] as float32 --
    nope_offs = tl.arange(0, D_NOPE_C)
    nope_bf16 = tl.load(kv_ptr + pid * kv_row_stride + nope_offs)
    nope_f32 = nope_bf16.to(tl.float32)

    cache_base = slot * BYTES_PER_TOKEN

    # -- Per-tile quantize + store --
    for tile_idx in tl.static_range(NUM_TILES_C):
        tile_start = tile_idx * TILE_SIZE_C
        tile_offs = tile_start + tl.arange(0, TILE_SIZE_C)
        tile_vals = tl.load(kv_ptr + pid * kv_row_stride + tile_offs).to(tl.float32)

        absmax = tl.max(tl.abs(tile_vals))
        scale_inv = absmax / 448.0
        scale_inv = tl.where(scale_inv < 1e-12, 1e-12, scale_inv)
        # UE8M0: round up to power of 2
        log2_val = tl.math.log2(scale_inv)
        log2_ceil = tl.math.ceil(log2_val)
        scale = tl.math.pow2(log2_ceil)

        quantized = (tile_vals / scale).to(tl.float8e4nv)

        # Store FP8 nope
        tl.store(cache_ptr + cache_base + tile_offs, quantized)

        # Store scale as float32 (4 bytes)
        scale_offset = cache_base + D_NOPE_C + tile_idx * 4
        scale_ptr = (cache_ptr + scale_offset).to(tl.pointer_type(tl.float32))
        tl.store(scale_ptr, scale)

    # -- copy RoPE as-is (bfloat16 → 2 bytes each) --
    rope_offs = tl.arange(0, D_ROPE_C)
    rope_bf16 = tl.load(kv_ptr + pid * kv_row_stride + D_NOPE_C + rope_offs)
    rope_out_offset = cache_base + D_NOPE_C + SCALE_BYTES_C
    # Cast to bfloat16* so pointer arithmetic advances by 2 bytes per element
    rope_ptr = (cache_ptr + rope_out_offset).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_ptr + rope_offs, rope_bf16)


def store_kcache_fp8(
    key: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Store compressed KV into FP8 paged cache with per-tile quantization.

    Args:
        key:          [N, 1, 576] bfloat16 — compressed KV (NoPE+RoPE)
        k_cache:      paged cache tensor [num_blocks, block_size, 1, 656] float8_e4m3fn
                      (may be a stride-padded slice from block_size+1 allocation)
        slot_mapping: [N] int32 — target slot indices (-1 = skip)
                      slot = block_idx * block_size + offset_in_block
    """
    N = key.shape[0]
    block_size = k_cache.shape[1]
    key_2d = key.view(N, D_TOTAL)

    fp8_nope, scales = quantize_nope_fp8(key_2d[:, :D_NOPE])
    rope = key_2d[:, D_NOPE:]
    packed = pack_mla_fp8(fp8_nope, scales, rope)  # [N, 656] uint8

    # Convert packed uint8 → fp8 view for assignment to fp8 cache
    packed_fp8 = packed.view(torch.float8_e4m3fn)  # [N, 656]

    # Scatter into paged cache — use clamped indices for CUDA graph compatibility
    # (boolean masking produces dynamic shapes, which breaks graph capture)
    safe_slots = torch.where(
        slot_mapping >= 0,
        slot_mapping.long(),
        torch.zeros_like(slot_mapping, dtype=torch.long),
    )
    block_idx = safe_slots // block_size
    offset_in_block = safe_slots % block_size

    # Write all rows; invalid slots (originally -1) write to slot 0 harmlessly
    # (they'll be overwritten by real data later)
    k_cache[block_idx, offset_in_block, 0, :] = packed_fp8
