"""Fused decode transforms for the NSA Indexer."""

import torch
import triton
import triton.language as tl


@triton.jit
def _indexer_layer_norm_bf16_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    stride_row,
    eps,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < HEAD_DIM
    x = tl.load(input_ptr + row * stride_row + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    mean = tl.sum(x, axis=0) / HEAD_DIM
    centered = x - mean
    variance = tl.sum(centered * centered, axis=0) / HEAD_DIM
    inv_std = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    output = centered * inv_std * weight + bias
    tl.store(output_ptr + row * stride_row + cols, output, mask=mask)


def indexer_layer_norm_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """LayerNorm in FP32 with a BF16 output, in one launch."""
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("Indexer LayerNorm expects a contiguous [T, D] tensor")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"Indexer LayerNorm input must be bfloat16, got {x.dtype}")
    head_dim = x.shape[1]
    output = torch.empty_like(x)
    _indexer_layer_norm_bf16_kernel[(x.shape[0],)](
        x,
        weight,
        bias,
        output,
        x.stride(0),
        eps,
        HEAD_DIM=head_dim,
        BLOCK=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return output


@triton.jit
def _indexer_qk_rope_inplace_kernel(
    query_ptr,
    key_ptr,
    positions_ptr,
    cos_sin_ptr,
    query_stride_token,
    query_stride_head,
    key_stride_token,
    NUM_Q_HEADS: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    pair = tl.arange(0, ROPE_DIM // 2)
    is_query = head < NUM_Q_HEADS

    query_base = query_ptr + token * query_stride_token + head * query_stride_head
    key_base = key_ptr + token * key_stride_token
    base = tl.where(is_query, query_base, key_base)

    # Input is interleaved: [real0, imag0, real1, imag1, ...]. Load every
    # pair before writing the half-layout output in place.
    real = tl.load(base + pair * 2).to(tl.float32)
    imag = tl.load(base + pair * 2 + 1).to(tl.float32)
    position = tl.load(positions_ptr + token)
    freq_base = position * ROPE_DIM
    cos = tl.load(cos_sin_ptr + freq_base + pair)
    sin = tl.load(cos_sin_ptr + freq_base + ROPE_DIM // 2 + pair)

    out_real = real * cos - imag * sin
    out_imag = imag * cos + real * sin
    tl.store(base + pair, out_real)
    tl.store(base + ROPE_DIM // 2 + pair, out_imag)


def indexer_qk_rope_inplace(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> None:
    """Convert interleaved Q/K RoPE dims to half layout and rotate in place."""
    if query.ndim != 3 or key.ndim != 2:
        raise ValueError("query/key must have shapes [T, H, D] and [T, D]")
    if query.shape[0] != key.shape[0] or positions.numel() != query.shape[0]:
        raise ValueError("query, key, and positions must use the same token count")
    if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
        raise TypeError("Indexer fused RoPE requires bfloat16 Q and K")
    if rope_dim != 64:
        raise ValueError(
            f"Indexer fused RoPE currently requires rope_dim=64, got {rope_dim}"
        )

    _indexer_qk_rope_inplace_kernel[(query.shape[0], query.shape[1] + 1)](
        query,
        key,
        positions,
        cos_sin_cache,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        NUM_Q_HEADS=query.shape[1],
        ROPE_DIM=rope_dim,
        num_warps=1,
    )
