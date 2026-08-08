"""FP8 quantization and token-layout kernels for DeepEP/DeepGEMM MoE.

These kernels are adapted from NanoDeploy-Pure_dp's first in-tree MoE
implementation. Communication remains in DeepEP and GEMM remains in
DeepGEMM.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from nanodeploy.kernels.deep_gemm_backend import (
    m_grouped_fp8_gemm_nt_contiguous,
)


@triton.jit
def _per_token_group_quant_fp8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    group_size,
    num_columns,
    input_row_stride,
    eps,
    fp8_min,
    fp8_max,
    BLOCK: tl.constexpr,
):
    groups_per_row = num_columns // group_size
    group_id = tl.program_id(0)
    row = group_id // groups_per_row
    group_in_row = group_id % groups_per_row

    input_ptr += row * input_row_stride + group_in_row * group_size
    output_ptr += group_id * group_size
    scale_ptr += group_id

    offsets = tl.arange(0, BLOCK)
    mask = offsets < group_size
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(values)), eps)
    scale = absmax / fp8_max
    quantized = tl.clamp(values / scale, fp8_min, fp8_max).to(
        output_ptr.dtype.element_ty
    )
    tl.store(output_ptr + offsets, quantized, mask=mask)
    tl.store(scale_ptr, scale)


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension in fixed-size groups with FP32 scales."""

    if x.dim() != 2:
        raise ValueError(f"expected a 2D activation tensor, got shape {tuple(x.shape)}")
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"activation width {x.shape[-1]} must be divisible by {group_size}"
        )
    if x.stride(-1) != 1:
        raise ValueError("activation groups must be contiguous")

    finfo = torch.finfo(dtype)
    output = torch.empty_like(x, dtype=dtype)
    scales = torch.empty(
        (x.shape[0], x.shape[1] // group_size),
        dtype=torch.float32,
        device=x.device,
    )
    num_groups = x.numel() // group_size
    block = triton.next_power_of_2(group_size)
    _per_token_group_quant_fp8_kernel[(num_groups,)](
        x,
        output,
        scales,
        group_size,
        x.shape[1],
        x.stride(0),
        eps,
        finfo.min,
        finfo.max,
        BLOCK=block,
        num_warps=min(max(block // 256, 1), 8),
        num_stages=1,
    )
    return output, scales


@triton.jit
def _silu_and_mul_masked_post_quant_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    masked_m_ptr,
    size_n,
    fp8_max,
    fp8_min,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_block = tl.program_id(0)
    token_stride = tl.num_programs(1)
    valid_tokens = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, tl.int64)
    stride_input_1 = tl.cast(stride_input_1, tl.int64)
    stride_output_0 = tl.cast(stride_output_0, tl.int64)
    stride_output_1 = tl.cast(stride_output_1, tl.int64)

    offsets = hidden_block * BLOCK_N + tl.arange(0, BLOCK_N)
    input_base = input_ptr + expert_id * stride_input_0 + offsets
    output_base = output_ptr + expert_id * stride_output_0 + offsets
    scale_base = (
        output_scale_ptr
        + expert_id * stride_output_scale_0
        + hidden_block * stride_output_scale_2
    )

    for token_index in tl.range(
        token_id, valid_tokens, token_stride, num_stages=NUM_STAGES
    ):
        gate = tl.load(
            input_base + token_index * stride_input_1,
            mask=offsets < size_n,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_base + token_index * stride_input_1 + size_n,
            mask=offsets < size_n,
            other=0.0,
        ).to(tl.float32)
        activated = (gate / (1.0 + tl.exp(-gate))) * up
        absmax = tl.maximum(tl.max(tl.abs(activated)), 1e-10)
        scale = absmax / fp8_max
        quantized = tl.clamp(activated / scale, fp8_min, fp8_max).to(
            output_ptr.dtype.element_ty
        )
        tl.store(
            output_base + token_index * stride_output_1,
            quantized,
            mask=offsets < size_n,
        )
        tl.store(scale_base + token_index * stride_output_scale_1, scale)


def silu_and_mul_masked_post_quant_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
) -> None:
    """Apply SwiGLU and quantize valid rows of a masked grouped tensor."""

    if input.dim() != 3 or not input.is_contiguous():
        raise ValueError("masked SwiGLU input must be a contiguous 3D tensor")
    if not output.is_contiguous():
        raise ValueError("masked SwiGLU output must be contiguous")
    if input.shape[0] != masked_m.numel():
        raise ValueError("masked_m must contain one count per local expert")
    if input.shape[-1] % 2 != 0:
        raise ValueError("gate/up width must be even")
    size_n = input.shape[-1] // 2
    if size_n % quant_group_size != 0:
        raise ValueError("SwiGLU output width must be divisible by quant group size")
    expected_scale_shape = (
        input.shape[0],
        input.shape[1],
        size_n // quant_group_size,
    )
    if tuple(output.shape) != (input.shape[0], input.shape[1], size_n):
        raise ValueError("masked SwiGLU output has an incompatible shape")
    if tuple(output_scale.shape) != expected_scale_shape:
        raise ValueError("masked SwiGLU scale output has an incompatible shape")

    expert_count = masked_m.numel()
    blocks_per_expert = 64 if expert_count < 4 else 32
    finfo = torch.finfo(output.dtype)
    grid = (size_n // quant_group_size, blocks_per_expert, expert_count)
    _silu_and_mul_masked_post_quant_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        masked_m,
        size_n,
        finfo.max,
        finfo.min,
        BLOCK_N=quant_group_size,
        NUM_STAGES=6,
        num_warps=1,
    )


def _get_tma_aligned_size(x: int, element_size: int) -> int:
    alignment_bytes = 16
    if alignment_bytes % element_size != 0:
        raise ValueError("scale element size does not divide TMA alignment")
    alignment = alignment_bytes // element_size
    return (x + alignment - 1) // alignment * alignment


@triton.jit
def _tma_align_input_scale_kernel(
    input_ptr,
    output_ptr,
    m,
    k_groups,
    input_stride_m,
    input_stride_k,
    output_stride_m,
    output_stride_k,
    BLOCK_K: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_K)
    for row in range(row_start, m, row_step):
        values = tl.load(
            input_ptr + row * input_stride_m + offsets * input_stride_k,
            mask=offsets < k_groups,
        )
        tl.store(
            output_ptr + row * output_stride_m + offsets * output_stride_k,
            values,
            mask=offsets < k_groups,
        )


def tma_align_input_scale(input_scale: torch.Tensor) -> torch.Tensor:
    """Return an MN-major, TMA-aligned view of a 2D FP32 scale tensor."""

    if input_scale.dim() != 2:
        raise ValueError("TMA scale alignment expects a 2D tensor")
    m, k_groups = input_scale.shape
    padded_m = _get_tma_aligned_size(m, input_scale.element_size())
    backing = torch.empty(
        (k_groups, padded_m),
        dtype=input_scale.dtype,
        device=input_scale.device,
    )
    output = backing.t()[:m]
    _tma_align_input_scale_kernel[(min(m, 8192),)](
        input_scale,
        output,
        m,
        k_groups,
        input_scale.stride(0),
        input_scale.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_K=triton.next_power_of_2(k_groups),
    )
    return output


try:
    from packaging import version as _version

    _triton_version = _version.parse(triton.__version__)
except Exception:
    _triton_version = None

if _triton_version is not None and _triton_version >= _version.parse("3.0.0"):
    _fast_exp = tl.math.exp
else:
    _fast_exp = tl.math.fast_expf


@functools.lru_cache
def _device_props(device: int):
    props = torch.cuda.get_device_properties(device)
    warps_per_sm = 64 if (props.major, props.minor) == (9, 0) else 32
    return props.multi_processor_count, warps_per_sm


@triton.jit
def _silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    n: tl.constexpr,
    m,
    stride_input_m: tl.constexpr,
    stride_input_n: tl.constexpr,
    stride_output_m: tl.constexpr,
    stride_output_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    n_block = tl.program_id(0)
    row_start = tl.program_id(1)
    row_step = tl.num_programs(1)
    offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n
    gate_ptr = input_ptr + row_start * stride_input_m + offsets * stride_input_n
    up_ptr = gate_ptr + n * stride_input_n
    out_ptr = output_ptr + row_start * stride_output_m + offsets * stride_output_n
    for _ in tl.range(row_start, m, row_step):
        gate = tl.load(gate_ptr, mask=mask).to(tl.float32)
        up = tl.load(up_ptr, mask=mask).to(tl.float32)
        result = (gate / (1.0 + _fast_exp(-gate))) * up
        tl.store(out_ptr, result, mask=mask)
        gate_ptr += row_step * stride_input_m
        up_ptr += row_step * stride_input_m
        out_ptr += row_step * stride_output_m


def silu_and_mul(input: torch.Tensor, output: torch.Tensor) -> None:
    """Apply SwiGLU to a 2D gate/up tensor."""

    if input.dim() != 2 or input.shape[1] % 2 != 0:
        raise ValueError("SwiGLU input must be 2D with an even width")
    m = input.shape[0]
    n = input.shape[1] // 2
    if tuple(output.shape) != (m, n):
        raise ValueError("SwiGLU output has an incompatible shape")
    block_n = min(triton.next_power_of_2(n), 512)
    num_warps = 4
    num_sms, warps_per_sm = _device_props(input.device.index)
    grid = (
        triton.cdiv(n, block_n),
        min(m, num_sms * warps_per_sm // num_warps),
    )
    _silu_and_mul_kernel[grid](
        input,
        output,
        n,
        m,
        stride_input_m=input.stride(0),
        stride_input_n=input.stride(1),
        stride_output_m=output.stride(0),
        stride_output_n=output.stride(1),
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=1,
    )


@triton.jit
def _build_expert_offsets_kernel(
    counts,
    offsets,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
):
    expert = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_EXPERTS)
    values = tl.load(counts + expert_offsets, mask=expert_offsets < num_experts, other=0)
    starts = tl.cumsum(values) - values
    tl.store(offsets + expert_offsets, starts, mask=expert_offsets < num_experts)
    expert_start = tl.load(offsets + expert)
    expert_count = tl.load(counts + expert)
    block_offsets = tl.arange(0, BLOCK_E)
    for start in tl.range(0, expert_count, BLOCK_E, num_stages=4):
        tl.store(m_indices + expert_start + start + block_offsets, expert)


@triton.jit
def _scatter_fp8_kernel(
    total_tokens,
    expert_offsets,
    recv_x,
    recv_x_stride_m,
    recv_scale,
    recv_scale_stride_m,
    recv_topk,
    recv_topk_stride_m,
    output,
    output_stride_m,
    output_scale,
    output_scale_stride_m,
    output_index,
    output_index_stride_m,
    top_k: tl.constexpr,
    HIDDEN: tl.constexpr,
    HIDDEN_PAD: tl.constexpr,
    SCALE_HIDDEN: tl.constexpr,
    SCALE_HIDDEN_PAD: tl.constexpr,
):
    token_start = tl.program_id(0)
    token_step = tl.num_programs(0)
    hidden_offsets = tl.arange(0, HIDDEN_PAD)
    hidden_mask = hidden_offsets < HIDDEN
    scale_offsets = tl.arange(0, SCALE_HIDDEN_PAD)
    scale_mask = scale_offsets < SCALE_HIDDEN
    for token in range(token_start, total_tokens, token_step):
        values = tl.load(
            recv_x + token * recv_x_stride_m + hidden_offsets,
            mask=hidden_mask,
        )
        scales = tl.load(
            recv_scale + token * recv_scale_stride_m + scale_offsets,
            mask=scale_mask,
        )
        for topk_index in tl.range(0, top_k, num_stages=4):
            expert = tl.load(recv_topk + token * recv_topk_stride_m + topk_index)
            if expert >= 0:
                destination = tl.atomic_add(expert_offsets + expert, 1).to(tl.int64)
                tl.store(
                    output_index + token * output_index_stride_m + topk_index,
                    destination,
                )
                tl.store(
                    output + destination * output_stride_m + hidden_offsets,
                    values,
                    mask=hidden_mask,
                )
                tl.store(
                    output_scale + destination * output_scale_stride_m + scale_offsets,
                    scales,
                    mask=scale_mask,
                )


@torch.no_grad()
def ep_scatter_fp8(
    recv_x: torch.Tensor,
    recv_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
) -> None:
    """Duplicate received tokens into DeepGEMM's contiguous expert layout."""

    block_e = 128
    num_experts = counts.numel()
    if m_indices.numel() % block_e != 0:
        raise ValueError("DeepEP expert counts must be aligned to 128")
    _build_expert_offsets_kernel[(num_experts,)](
        counts,
        expert_offsets,
        m_indices,
        num_experts=num_experts,
        BLOCK_E=block_e,
        BLOCK_EXPERTS=triton.next_power_of_2(num_experts),
        num_warps=8,
    )
    hidden = recv_x.shape[1]
    scale_hidden = recv_scale.shape[1]
    _scatter_fp8_kernel[(min(recv_topk.shape[0], 8192),)](
        recv_topk.shape[0],
        expert_offsets,
        recv_x,
        recv_x.stride(0),
        recv_scale,
        recv_scale.stride(0),
        recv_topk,
        recv_topk.stride(0),
        output,
        output.stride(0),
        output_scale,
        output_scale.stride(0),
        output_index,
        output_index.stride(0),
        top_k=recv_topk.shape[1],
        HIDDEN=hidden,
        HIDDEN_PAD=triton.next_power_of_2(hidden),
        SCALE_HIDDEN=scale_hidden,
        SCALE_HIDDEN_PAD=triton.next_power_of_2(scale_hidden),
        num_warps=8,
    )


@triton.jit
def _gather_kernel(
    total_tokens,
    input_ptr,
    input_stride_m,
    topk_ids,
    topk_ids_stride_m,
    topk_weights,
    topk_weights_stride_m,
    input_index,
    input_index_stride_m,
    output,
    output_stride_m,
    top_k: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    hidden_block = tl.program_id(0)
    token_start = tl.program_id(1)
    token_step = tl.num_programs(1)
    offsets = tl.arange(0, BLOCK_D)
    for token in range(token_start, total_tokens, token_step):
        accumulator = tl.zeros((BLOCK_D,), tl.float32)
        for topk_index in range(0, top_k):
            expert = tl.load(topk_ids + token * topk_ids_stride_m + topk_index)
            if expert >= 0:
                source = tl.load(
                    input_index + token * input_index_stride_m + topk_index
                )
                weight = tl.load(
                    topk_weights + token * topk_weights_stride_m + topk_index
                )
                values = tl.load(
                    input_ptr
                    + source * input_stride_m
                    + hidden_block * BLOCK_D
                    + offsets
                )
                accumulator += values.to(tl.float32) * weight
        tl.store(
            output + token * output_stride_m + hidden_block * BLOCK_D + offsets,
            accumulator.to(output.dtype.element_ty),
        )


@torch.no_grad()
def ep_gather(
    input: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    input_index: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Gather contiguous expert outputs and apply routing weights."""

    block_d = 1024
    if input.shape[1] % block_d != 0:
        raise ValueError("MoE hidden size must be divisible by 1024")
    grid = (input.shape[1] // block_d, min(output.shape[0], 1024))
    _gather_kernel[grid](
        output.shape[0],
        input,
        input.stride(0),
        topk_ids,
        topk_ids.stride(0),
        topk_weights,
        topk_weights.stride(0),
        input_index,
        input_index.stride(0),
        output,
        output.stride(0),
        top_k=topk_ids.shape[1],
        BLOCK_D=block_d,
        num_warps=2,
    )


def fused_moe_fp8_contiguous(
    hidden_states: Tuple[torch.Tensor, torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_weight: Tuple[torch.Tensor, torch.Tensor],
    down_weight: Tuple[torch.Tensor, torch.Tensor],
    tokens_per_expert: list[int],
    block_size: int = 128,
) -> torch.Tensor:
    """Compute the normal/prefill MoE after a DeepEP normal dispatch."""

    recv_x, recv_scale = hidden_states
    all_tokens = sum(tokens_per_expert)
    if all_tokens <= 0:
        return torch.zeros(
            (recv_x.shape[0], gate_up_weight[0].shape[-1]),
            dtype=torch.bfloat16,
            device=recv_x.device,
        )

    hidden = recv_x.shape[1]
    intermediate_twice = gate_up_weight[0].shape[1]
    contiguous_x = torch.empty(
        (all_tokens, hidden), dtype=recv_x.dtype, device=recv_x.device
    )
    contiguous_scale = torch.empty(
        (all_tokens, hidden // block_size),
        dtype=torch.float32,
        device=recv_x.device,
    )
    m_indices = torch.empty(all_tokens, dtype=torch.int32, device=recv_x.device)
    output_index = torch.empty_like(topk_ids)
    counts = torch.tensor(
        tokens_per_expert,
        dtype=torch.int32,
        pin_memory=True,
    ).to(recv_x.device, non_blocking=True)
    expert_offsets = torch.empty_like(counts)
    ep_scatter_fp8(
        recv_x,
        recv_scale,
        topk_ids,
        counts,
        expert_offsets,
        contiguous_x,
        contiguous_scale,
        m_indices,
        output_index,
    )

    gate_up_output = torch.empty(
        (all_tokens, intermediate_twice),
        dtype=torch.bfloat16,
        device=recv_x.device,
    )
    m_grouped_fp8_gemm_nt_contiguous(
        (contiguous_x, tma_align_input_scale(contiguous_scale)),
        gate_up_weight,
        gate_up_output,
        m_indices,
    )

    down_input_bf16 = torch.empty(
        (all_tokens, intermediate_twice // 2),
        dtype=torch.bfloat16,
        device=recv_x.device,
    )
    silu_and_mul(gate_up_output, down_input_bf16)
    down_input, down_scale = per_token_group_quant_fp8(
        down_input_bf16, block_size
    )
    down_output = torch.empty(
        (all_tokens, hidden),
        dtype=torch.bfloat16,
        device=recv_x.device,
    )
    m_grouped_fp8_gemm_nt_contiguous(
        (down_input, tma_align_input_scale(down_scale)),
        down_weight,
        down_output,
        m_indices,
    )

    output = torch.empty(
        (recv_x.shape[0], hidden),
        dtype=torch.bfloat16,
        device=recv_x.device,
    )
    ep_gather(down_output, topk_ids, topk_weights, output_index, output)
    return output


__all__ = [
    "fused_moe_fp8_contiguous",
    "per_token_group_quant_fp8",
    "silu_and_mul_masked_post_quant_fwd",
]
