# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor

from nanodeploy.backends.gpu_generic.kernels.utils import get_device_props


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
):
    """Quant fp8 kernel."""
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
        scale = tl.maximum(tl.max(tl.abs(a)), 1e-6) * rfp8_max
        out = a / scale

        out = tl.clamp(out, fp8_min, fp8_max)
        out = out.to(out_ptr.dtype.element_ty)

        tl.store(o_ptrs, out)
        tl.store(s_ptr, scale)

        a_ptrs += m_id_stride * stride_am
        o_ptrs += m_id_stride * stride_om
        s_ptr += m_id_stride * stride_sm


def _quant_fp8_launcher(A: Tensor, group_size: int, out: Tensor, scales: Tensor):
    """Quant online."""
    M, K = A.shape
    num_groups = K // group_size
    M_out = out.size(0)

    dtype = out.dtype
    finfo = torch.finfo(dtype)
    fmin = finfo.min
    fmax = finfo.max

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
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out, scales


def quant_fp8(
    A: Tensor,
    group_size: int,
    dtype: torch.dtype = torch.float8_e4m3fn,
    trans_scale: bool = False,
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
    return _quant_fp8_launcher(A, group_size, out, scales)


def quant_fp8_tma(A: Tensor, group_size: int, dtype: torch.dtype = torch.float8_e4m3fn):
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
    return _quant_fp8_launcher(A, group_size, out, scales)


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
