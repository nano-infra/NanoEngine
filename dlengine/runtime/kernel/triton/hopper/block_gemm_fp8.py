# Copyright (c) OpenMMLab. All rights reserved.
import os

import torch
import triton
import triton.language as tl
from torch import Tensor

from dlengine.runtime.kernel.triton.generic.utils import get_device_props

_USE_PACKED_SMALL_M_QUANT = os.getenv(
    "DLENGINE_USE_PACKED_SMALL_M_QUANT", "1"
).lower() not in {"0", "false", "off"}
_PACKED_QUANT_MAX_M = 64
_PACKED_QUANT_TASKS = 16
_PACKED_QUANT_TAIL_VALUES = 2048
_PACKED_QUANT_TAIL_SCALES = 16


def _should_use_packed_small_m_quant(
    M: int,
    group_size: int,
    input_column_stride: int,
    output_column_stride: int,
) -> bool:
    return (
        _USE_PACKED_SMALL_M_QUANT
        and 0 < M <= _PACKED_QUANT_MAX_M
        and group_size == 128
        and input_column_stride == 1
        and output_column_stride == 1
    )


@triton.jit
def _quant_fp8_packed_small_m_kernel(
    a_ptr,
    out_ptr,
    scale_ptr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    stride_am,
    stride_om,
    stride_sm,
    stride_sg,
    M: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ACTUAL_TASKS: tl.constexpr,
    TAIL_OUT_VALUES: tl.constexpr,
    TAIL_SCALE_VALUES: tl.constexpr,
    TASKS_PER_PROGRAM: tl.constexpr,
    TAIL_VALUES_PER_PROGRAM: tl.constexpr,
    TAIL_SCALES_PER_PROGRAM: tl.constexpr,
    ROUND_UE8M0: tl.constexpr = False,
    MIN_ABSMAX: tl.constexpr = 1e-6,
):
    """Quantize small logical M while preserving DeepGEMM's padded layout.

    SGLang's CUDA kernel packs several (token, group) reductions into one CTA.
    Keep that scheduling property here instead of launching one Triton program
    for every padded row and group. Padded rows still receive exactly the same
    zero values and scales as ``_quant_fp8_kernel``, but use vector stores
    rather than running redundant absmax reductions.
    """
    pid = tl.program_id(0)
    rfp8_max = 1.0 / fp8_max

    task_ids = pid * TASKS_PER_PROGRAM + tl.arange(0, TASKS_PER_PROGRAM)
    task_mask = task_ids < ACTUAL_TASKS
    m_ids = task_ids // NUM_GROUPS
    group_ids = task_ids % NUM_GROUPS
    group_offsets = tl.arange(0, GROUP_SIZE)
    a_offsets = (
        m_ids[:, None] * stride_am
        + group_ids[:, None] * GROUP_SIZE
        + group_offsets[None, :]
    )
    values = tl.load(a_ptr + a_offsets, mask=task_mask[:, None], other=0.0).to(
        tl.float32
    )
    scales = tl.maximum(tl.max(tl.abs(values), axis=1), MIN_ABSMAX) * rfp8_max
    if ROUND_UE8M0:
        scales = tl.exp2(tl.ceil(tl.log2(scales)))
    quantized = tl.clamp(values / scales[:, None], fp8_min, fp8_max).to(
        out_ptr.dtype.element_ty
    )
    out_offsets = (
        m_ids[:, None] * stride_om
        + group_ids[:, None] * GROUP_SIZE
        + group_offsets[None, :]
    )
    tl.store(out_ptr + out_offsets, quantized, mask=task_mask[:, None])
    tl.store(
        scale_ptr + m_ids * stride_sm + group_ids * stride_sg,
        scales,
        mask=task_mask,
    )

    tail_offsets = pid * TAIL_VALUES_PER_PROGRAM + tl.arange(0, TAIL_VALUES_PER_PROGRAM)
    tail_rows = tail_offsets // K
    tail_columns = tail_offsets % K
    tl.store(
        out_ptr + (M + tail_rows) * stride_om + tail_columns,
        0.0,
        mask=tail_offsets < TAIL_OUT_VALUES,
    )

    tail_scale_offsets = pid * TAIL_SCALES_PER_PROGRAM + tl.arange(
        0, TAIL_SCALES_PER_PROGRAM
    )
    tail_scale_rows = tail_scale_offsets // NUM_GROUPS
    tail_scale_groups = tail_scale_offsets % NUM_GROUPS
    tail_scale = MIN_ABSMAX * rfp8_max
    if ROUND_UE8M0:
        tail_scale = tl.exp2(tl.ceil(tl.log2(tail_scale)))
    tl.store(
        scale_ptr + (M + tail_scale_rows) * stride_sm + tail_scale_groups * stride_sg,
        tail_scale,
        mask=tail_scale_offsets < TAIL_SCALE_VALUES,
    )


@triton.jit
def _quant_fp8_kernel(
    a_ptr,
    out_ptr,
    scale_ptr,
    M,
    M_out,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    stride_am,
    stride_ak: tl.constexpr,
    stride_om,
    stride_ok: tl.constexpr,
    stride_sm,
    stride_sg,
    GROUP_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    ROUND_UE8M0: tl.constexpr = False,
    MIN_ABSMAX: tl.constexpr = 1e-6,
):
    """Quant fp8 kernel.

    When ``ROUND_UE8M0`` is True, round per-block scales up to the next
    power of two (matches DSV4's QAT scale_fmt='ue8m0' regime). Without
    this, plain FP32 scales are used — which is more accurate but does
    NOT match the trained-on activation distribution and accumulates
    drift across layers.
    """
    group_id = tl.program_id(0)
    m_id_start = tl.program_id(1)
    m_id_stride = tl.num_programs(1)

    g_offs = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    g_offs = tl.max_contiguous(tl.multiple_of(g_offs, GROUP_SIZE), GROUP_SIZE)
    rfp8_max = 1 / fp8_max

    m_id = m_id_start
    a_ptrs = a_ptr + m_id * stride_am + g_offs * stride_ak
    o_ptrs = out_ptr + m_id * stride_om + g_offs * stride_ok
    s_ptr = scale_ptr + m_id * stride_sm + group_id * stride_sg

    for m_id in tl.range(m_id_start, M_out, m_id_stride, num_stages=NUM_STAGES):

        a = tl.load(a_ptrs, mask=m_id < M, other=0).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(a)), MIN_ABSMAX) * rfp8_max
        if ROUND_UE8M0:
            # Round up to nearest power of two so the scale is a single
            # FP32-mantissa-free value (UE8M0). Equivalent to
            # scale = exp2(ceil(log2(scale))).
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        out = a / scale

        out = tl.clamp(out, fp8_min, fp8_max)
        out = out.to(out_ptr.dtype.element_ty)

        tl.store(o_ptrs, out)
        tl.store(s_ptr, scale)

        a_ptrs += m_id_stride * stride_am
        o_ptrs += m_id_stride * stride_om
        s_ptr += m_id_stride * stride_sm


def _quant_fp8_launcher(
    A: Tensor,
    group_size: int,
    out: Tensor,
    scales: Tensor,
    round_ue8m0: bool = False,
    min_absmax: float = 1e-6,
):
    """Quant online."""
    M, K = A.shape
    num_groups = K // group_size
    M_out = out.size(0)

    dtype = out.dtype
    finfo = torch.finfo(dtype)
    fmin = finfo.min
    fmax = finfo.max

    if _should_use_packed_small_m_quant(
        M,
        group_size,
        A.stride(1),
        out.stride(1),
    ):
        actual_tasks = M * num_groups
        tail_rows = M_out - M
        tail_out_values = tail_rows * K
        tail_scale_values = tail_rows * num_groups
        num_programs = max(
            triton.cdiv(actual_tasks, _PACKED_QUANT_TASKS),
            triton.cdiv(tail_out_values, _PACKED_QUANT_TAIL_VALUES),
            triton.cdiv(tail_scale_values, _PACKED_QUANT_TAIL_SCALES),
        )
        _quant_fp8_packed_small_m_kernel[(num_programs,)](
            A,
            out,
            scales,
            fp8_min=fmin,
            fp8_max=fmax,
            stride_am=A.stride(0),
            stride_om=out.stride(0),
            stride_sm=scales.stride(0),
            stride_sg=scales.stride(1),
            M=M,
            K=K,
            NUM_GROUPS=num_groups,
            GROUP_SIZE=group_size,
            ACTUAL_TASKS=actual_tasks,
            TAIL_OUT_VALUES=tail_out_values,
            TAIL_SCALE_VALUES=tail_scale_values,
            TASKS_PER_PROGRAM=_PACKED_QUANT_TASKS,
            TAIL_VALUES_PER_PROGRAM=_PACKED_QUANT_TAIL_VALUES,
            TAIL_SCALES_PER_PROGRAM=_PACKED_QUANT_TAIL_SCALES,
            ROUND_UE8M0=round_ue8m0,
            MIN_ABSMAX=min_absmax,
            num_warps=8,
            num_stages=1,
        )
        return out, scales

    num_warps = 1

    props = get_device_props(A.device.index)
    num_sm = props["multi_processor_count"]
    warps_per_sm = props["warps_per_sm"]
    max_ctas = num_sm * warps_per_sm // num_warps
    grid_size1 = min(M_out, max_ctas // num_groups)
    if grid_size1 == 0:
        raise ValueError(
            f"quant_fp8 grid_size1=0: M={M}, K={K}, M_out={M_out}, "
            f"group_size={group_size}, num_groups={num_groups}, "
            f"num_sm={num_sm}, warps_per_sm={warps_per_sm}, max_ctas={max_ctas}, "
            f"A.shape={A.shape}, A.device={A.device}"
        )
    assert grid_size1 < 65536
    num_stages = min(5, max(1, triton.cdiv(M_out, grid_size1)))
    grid = (num_groups, grid_size1)
    _quant_fp8_kernel[grid](
        A,
        out,
        scales,
        M,
        M_out,
        fp8_min=fmin,
        fp8_max=fmax,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_om=out.stride(0),
        stride_ok=out.stride(1),
        stride_sm=scales.stride(0),
        stride_sg=scales.stride(1),
        GROUP_SIZE=group_size,
        NUM_STAGES=num_stages,
        ROUND_UE8M0=round_ue8m0,
        MIN_ABSMAX=min_absmax,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out, scales


def quant_fp8(
    A: Tensor,
    group_size: int,
    dtype: torch.dtype = torch.float8_e4m3fn,
    trans_scale: bool = False,
    round_ue8m0: bool = False,
    min_absmax: float = 1e-6,
):
    """Quant fp8."""
    assert A.dim() == 2
    M, K = A.shape
    assert K % group_size == 0
    num_groups = K // group_size
    out = torch.empty_like(A, dtype=dtype)
    if trans_scale:
        scales = A.new_empty(num_groups, M, dtype=torch.float32).T
    else:
        scales = A.new_empty(M, num_groups, dtype=torch.float32)
    return _quant_fp8_launcher(
        A,
        group_size,
        out,
        scales,
        round_ue8m0=round_ue8m0,
        min_absmax=min_absmax,
    )


def quant_fp8_tma(
    A: Tensor,
    group_size: int,
    dtype: torch.dtype = torch.float8_e4m3fn,
    round_ue8m0: bool = False,
    min_absmax: float = 1e-6,
):
    """Quant fp8 tma."""
    from deep_gemm import ceil_div, get_m_alignment_for_contiguous_layout

    assert A.dim() == 2
    M, K = A.shape
    assert K % group_size == 0
    num_groups = K // group_size
    alignment = get_m_alignment_for_contiguous_layout()
    aligned_M = ceil_div(M, alignment) * alignment
    out = A.new_empty(aligned_M, K, dtype=dtype)
    scales = A.new_empty(num_groups, aligned_M, dtype=torch.float32).T
    return _quant_fp8_launcher(
        A,
        group_size,
        out,
        scales,
        round_ue8m0=round_ue8m0,
        min_absmax=min_absmax,
    )


def deep_gemm_fp8(
    A: Tensor,
    A_scale: Tensor,
    B: Tensor,
    B_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
):
    """Deepgemm fp8."""
    import deep_gemm

    M, _ = A.shape
    N, _ = B.shape
    assert out_dtype == torch.bfloat16, "DeepGemm requires bf16 output."
    C = A.new_empty(M, N, dtype=out_dtype)
    if hasattr(deep_gemm, "fp8_gemm_nt"):
        deep_gemm.fp8_gemm_nt((A, A_scale), (B, B_scale), C)
    elif hasattr(deep_gemm, "gemm_fp8_fp8_bf16_nt"):
        deep_gemm.gemm_fp8_fp8_bf16_nt((A, A_scale), (B, B_scale), C)
    else:
        raise RuntimeError("deep_gemm version mismatch")
    return C
