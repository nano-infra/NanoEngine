"""Fused interleaved-to-half RoPE for Hopper MLA decode.

The GLM/DeepSeek projections store rotary pairs interleaved, while FlashMLA's
rope sub-vector uses the half layout.  This kernel performs the permutation
and rotates Q and K together in one launch, replacing two layout copies and
two separately compiled elementwise RoPE kernels.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_mla_qk_rope_half_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    cos_sin_ptr,
    positions_ptr,
    q_stride_token: tl.constexpr,
    q_stride_head: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_head: tl.constexpr,
    q_out_stride_token: tl.constexpr,
    q_out_stride_head: tl.constexpr,
    k_out_stride_token: tl.constexpr,
    k_out_stride_head: tl.constexpr,
    cache_stride_position: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    linear_head = tl.program_id(0)
    heads_per_token: tl.constexpr = NUM_Q_HEADS + NUM_K_HEADS
    token = linear_head // heads_per_token
    head = linear_head - token * heads_per_token
    is_q = head < NUM_Q_HEADS
    local_head = tl.where(is_q, head, head - NUM_Q_HEADS)

    input_base = tl.where(
        is_q,
        q_ptr + token * q_stride_token + local_head * q_stride_head,
        k_ptr + token * k_stride_token + local_head * k_stride_head,
    )
    output_base = tl.where(
        is_q,
        q_out_ptr + token * q_out_stride_token + local_head * q_out_stride_head,
        k_out_ptr + token * k_out_stride_token + local_head * k_out_stride_head,
    )

    offsets = tl.arange(0, HALF_DIM)
    x_real = tl.load(input_base + offsets * 2).to(tl.float32)
    x_imag = tl.load(input_base + offsets * 2 + 1).to(tl.float32)
    position = tl.load(positions_ptr + token)
    cache_base = cos_sin_ptr + position * cache_stride_position
    cos = tl.load(cache_base + offsets)
    sin = tl.load(cache_base + HALF_DIM + offsets)

    y_real = x_real * cos - x_imag * sin
    y_imag = x_imag * cos + x_real * sin
    tl.store(output_base + offsets, y_real)
    tl.store(output_base + HALF_DIM + offsets, y_imag)


def fused_mla_qk_rope_half(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate interleaved BF16 Q/K and return fresh half-layout tensors."""
    if q.dim() != 3 or k.dim() != 3 or q.shape[0] != k.shape[0]:
        raise ValueError(
            f"incompatible Q/K shapes: q={tuple(q.shape)} k={tuple(k.shape)}"
        )
    if q.shape[-1] != 64 or k.shape[-1] != 64:
        raise ValueError("fused MLA Q/K RoPE requires rope_dim=64")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype:
        raise TypeError("fused MLA Q/K RoPE requires BF16 Q/K")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("fused MLA Q/K RoPE requires contiguous last dimensions")
    if cos_sin_cache.dtype != torch.float32 or cos_sin_cache.shape[-1] != 64:
        raise TypeError("cos/sin cache must be float32 with last dimension 64")
    if positions.dim() != 1 or positions.shape[0] != q.shape[0]:
        raise ValueError("positions must contain one entry per token")

    q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
    k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
    total_heads = q.shape[0] * (q.shape[1] + k.shape[1])
    _fused_mla_qk_rope_half_kernel[(total_heads,)](
        q,
        k,
        q_out,
        k_out,
        cos_sin_cache,
        positions,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        q_out.stride(0),
        q_out.stride(1),
        k_out.stride(0),
        k_out.stride(1),
        cos_sin_cache.stride(0),
        NUM_Q_HEADS=q.shape[1],
        NUM_K_HEADS=k.shape[1],
        HALF_DIM=32,
        num_warps=1,
    )
    return q_out, k_out


__all__ = ["fused_mla_qk_rope_half"]
