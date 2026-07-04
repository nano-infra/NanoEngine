"""Fused (add +) RMSNorm + per-128-group FP8 quantisation.

Replaces the eager pair

    out, residual = add_rms_norm_triton(x, residual, weight, eps)   # bf16 out
    q, s = quant_fp8_tma(out, 128)                                  # fp8 + scales

with one launch, skipping the bf16 round-trip of ``out`` through HBM.
Output layout is bit-compatible with ``quant_fp8_tma``: TMA-aligned M,
column-major scales, padded rows quantised from zeros. The normalised
value is rounded through bf16 in-register before quantisation so results
match the eager two-kernel chain bit-for-bit.

CUDA-Graph safe: static shapes, no host sync.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

QUANT_GROUP_SIZE = 128
MAX_FUSED_HIDDEN_SIZE = 8192


@triton.jit
def _add_rms_norm_quant_fp8_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    q_ptr,
    scale_ptr,
    residual_out_ptr,
    stride_x_row,
    stride_residual_row,
    stride_q_row,
    stride_residual_out_row,
    stride_s_m,
    stride_s_g,
    M,
    hidden_size,
    eps,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    ROUND_UE8M0: tl.constexpr,
    ADD_UNIT_OFFSET: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)  # runs over the TMA-aligned (padded) M
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size
    is_real_row = row < M
    load_mask = mask & is_real_row

    x = tl.load(x_ptr + row * stride_x_row + cols, mask=load_mask, other=0.0).to(
        tl.float32
    )
    residual = tl.load(
        residual_ptr + row * stride_residual_row + cols, mask=load_mask, other=0.0
    ).to(tl.float32)
    summed = x + residual
    tl.store(
        residual_out_ptr + row * stride_residual_out_row + cols, summed, mask=load_mask
    )

    variance = tl.sum(summed * summed, axis=0) / hidden_size
    inv_rms = tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    if ADD_UNIT_OFFSET:
        weight = weight + 1.0

    out = summed * inv_rms * weight
    # Round through bf16 so the quantised values are identical to the
    # eager path (norm kernel stores bf16, quant kernel reloads it).
    out = out.to(tl.bfloat16).to(tl.float32)
    # Padded rows compute 0 * weight, which yields -0.0 for negative
    # weights; force +0.0 to stay bit-identical with quant_fp8_tma.
    out = tl.where(is_real_row, out, 0.0)

    # Per-128-group quantisation (same math as _quant_fp8_kernel).
    grouped = tl.reshape(out, (NUM_GROUPS, GROUP_SIZE))
    absmax = tl.max(tl.abs(grouped), axis=1)
    scale = tl.maximum(absmax, 1e-6) * (1.0 / fp8_max)
    if ROUND_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    q = grouped / scale[:, None]
    q = tl.clamp(q, fp8_min, fp8_max).to(q_ptr.dtype.element_ty)

    # Padded rows (row >= M) still get zeros + eps-scale written, same
    # as quant_fp8_tma, so DeepGEMM can safely read the aligned tail.
    tl.store(q_ptr + row * stride_q_row + cols, tl.reshape(q, (BLOCK_SIZE,)), mask=mask)
    g_offs = tl.arange(0, NUM_GROUPS)
    tl.store(scale_ptr + row * stride_s_m + g_offs * stride_s_g, scale)


def can_use_add_rms_norm_quant_fp8(hidden_size: int) -> bool:
    return (
        hidden_size % QUANT_GROUP_SIZE == 0
        and hidden_size <= MAX_FUSED_HIDDEN_SIZE
        # tl.reshape needs the block to factor exactly into groups
        and triton.next_power_of_2(hidden_size) == hidden_size
    )


def add_rms_norm_quant_fp8(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float,
    dtype: torch.dtype = torch.float8_e4m3fn,
    add_unit_offset: bool = False,
    round_ue8m0: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fused residual-add + RMSNorm + FP8 group quant.

    Returns ``(q, scales, residual_out)`` where ``q``/``scales`` have the
    exact TMA-aligned layout of ``quant_fp8_tma`` (ready for
    ``deep_gemm_fp8``) and ``residual_out`` matches ``add_rms_norm_triton``.
    """
    from deep_gemm import ceil_div, get_m_alignment_for_contiguous_layout

    hidden_size = x.shape[-1]
    assert can_use_add_rms_norm_quant_fp8(hidden_size)
    x_2d = x.reshape(-1, hidden_size)
    residual_2d = residual.reshape(-1, hidden_size)
    M = x_2d.shape[0]
    num_groups = hidden_size // QUANT_GROUP_SIZE

    alignment = get_m_alignment_for_contiguous_layout()
    aligned_M = ceil_div(M, alignment) * alignment

    q = x_2d.new_empty(aligned_M, hidden_size, dtype=dtype)
    scales = x_2d.new_empty(num_groups, aligned_M, dtype=torch.float32).T
    residual_out = torch.empty_like(residual_2d)

    finfo = torch.finfo(dtype)

    _add_rms_norm_quant_fp8_kernel[(aligned_M,)](
        x_2d,
        residual_2d,
        weight,
        q,
        scales,
        residual_out,
        x_2d.stride(0),
        residual_2d.stride(0),
        q.stride(0),
        residual_out.stride(0),
        scales.stride(0),
        scales.stride(1),
        M,
        hidden_size,
        eps,
        fp8_min=finfo.min,
        fp8_max=finfo.max,
        ROUND_UE8M0=round_ue8m0,
        ADD_UNIT_OFFSET=add_unit_offset,
        GROUP_SIZE=QUANT_GROUP_SIZE,
        NUM_GROUPS=num_groups,
        BLOCK_SIZE=hidden_size,
        num_warps=16 if hidden_size > 4096 else 8,
        num_stages=1,
    )
    return q, scales, residual_out.reshape_as(residual)
