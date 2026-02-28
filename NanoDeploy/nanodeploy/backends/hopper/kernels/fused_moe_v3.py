import functools
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from nanodeploy.backends.hopper.kernels.fp8 import per_token_group_quant_fp8

try:
    import deep_gemm

    use_deep_gemm = True
except ImportError:
    use_deep_gemm = False

# ---------------------------------------------------------------------------
# Ported utilities (originally from dlblas)
# ---------------------------------------------------------------------------

# ---- tma_align_input_scale (from dlblas.kernels.quant_dequant) ----


def _get_tma_aligned_size(x: int, element_size: int) -> int:
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return ((x + alignment - 1) // alignment) * alignment


@triton.jit
def _tma_align_input_scale_kernel(
    input_scale_ptr,
    output_ptr,
    m,
    k_div_block_size,
    input_scale_stride_m,
    input_scale_stride_k,
    output_stride_m,
    output_stride_k,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    grid_m = tl.num_programs(0)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    for m_base in range(pid_m, m, grid_m):
        input_offset = (
            input_scale_ptr
            + m_base * input_scale_stride_m
            + k_offsets * input_scale_stride_k
        )
        input_data = tl.load(input_offset, mask=k_offsets < k_div_block_size)
        output_offset = (
            output_ptr + k_offsets * output_stride_k + m_base * output_stride_m
        )
        tl.store(output_offset, input_data, mask=k_offsets < k_div_block_size)


def tma_align_input_scale(input_scale: torch.Tensor):
    assert input_scale.dim() == 2
    m, k_div_block_size = input_scale.shape
    padd_m = _get_tma_aligned_size(m, input_scale.element_size())
    output = torch.empty(
        (k_div_block_size, padd_m), dtype=input_scale.dtype, device=input_scale.device
    )
    grid_m = min(m, 8192)
    BLOCK_SIZE_K = triton.next_power_of_2(k_div_block_size)
    _tma_align_input_scale_kernel[(grid_m,)](
        input_scale_ptr=input_scale,
        output_ptr=output,
        m=m,
        k_div_block_size=k_div_block_size,
        input_scale_stride_m=input_scale.stride(0),
        input_scale_stride_k=input_scale.stride(1),
        output_stride_m=output.stride(1),
        output_stride_k=output.stride(0),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return output.t()[:m]


# ---- silu_and_mul (from dlblas.layers.moe.kernels.activation) ----


@functools.lru_cache
def _get_device_props(device=None):
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    WARPS_PER_SM = {(9, 0): 64}
    warps_per_sm = WARPS_PER_SM.get((props.major, props.minor), 32)
    return dict(
        multi_processor_count=props.multi_processor_count, warps_per_sm=warps_per_sm
    )


try:
    from packaging import version as _version

    _TRITON_VERSION = _version.parse(triton.__version__)
except Exception:
    _TRITON_VERSION = None

if _TRITON_VERSION is not None and _TRITON_VERSION >= _version.parse("3.0.0"):
    _fast_expf = tl.math.exp
else:
    _fast_expf = tl.math.fast_expf


@triton.jit
def _silu_and_mul_kernel(
    gateup_ptr,
    out_ptr,
    N: tl.constexpr,
    M,
    stride_gum: tl.constexpr,
    stride_gun: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    n_block_id = tl.program_id(0)
    m_id_start = tl.program_id(1)
    m_id_stride = tl.num_programs(1)

    up_ptr = gateup_ptr + N * stride_gun
    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if N % BLOCK_SIZE_N == 0:
        mask = None
    else:
        mask = offs_n < N

    gate_ptrs = gateup_ptr + m_id_start * stride_gum + offs_n * stride_gun
    up_ptrs = up_ptr + m_id_start * stride_gum + offs_n * stride_gun
    out_ptrs = out_ptr + m_id_start * stride_om + offs_n * stride_on

    for _ in tl.range(m_id_start, M, m_id_stride):
        gate = tl.load(gate_ptrs, mask=mask)
        up = tl.load(up_ptrs, mask=mask)
        gate = gate.to(tl.float32)
        up = up.to(tl.float32)

        gate = gate / (1 + _fast_expf(-gate))
        out = gate * up

        tl.store(out_ptrs, out, mask=mask)

        gate_ptrs += m_id_stride * stride_gum
        up_ptrs += m_id_stride * stride_gum
        out_ptrs += m_id_stride * stride_om


def silu_and_mul(gate_up: torch.Tensor, out: torch.Tensor = None):
    """silu and mul."""
    assert gate_up.dim() == 2

    M = gate_up.size(0)
    N = gate_up.size(-1) // 2
    if out is None:
        out_shape = (M, N)
        out = gate_up.new_empty(out_shape)

    BLOCK_SIZE_N = triton.next_power_of_2(N)
    BLOCK_SIZE_N = min(BLOCK_SIZE_N, 512)
    num_warps = 4
    num_stages = 1

    props = _get_device_props(gate_up.device.index)
    num_sm = props["multi_processor_count"]
    warps_per_sm = props["warps_per_sm"]
    grid_size0 = triton.cdiv(N, BLOCK_SIZE_N)
    grid_size1 = min(M, num_sm * warps_per_sm // num_warps)
    assert grid_size0 < 65536 and grid_size1 < 65536
    grid = (grid_size0, grid_size1)
    _silu_and_mul_kernel[grid](
        gate_up,
        out,
        N,
        M,
        stride_gum=gate_up.stride(0),
        stride_gun=gate_up.stride(1),
        stride_om=out.stride(0),
        stride_on=out.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out


# ---------------------------------------------------------------------------
# ep_scatter / ep_gather kernels (FP8 and BF16)
# ---------------------------------------------------------------------------


@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)
    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)
    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)
    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)
    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        tl.store(
            m_indices_start_ptr + start_m + off_expert,
            cur_expert,
        )


@triton.jit
def _fwd_kernel_ep_scatter_2(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)
    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE
    offset_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
    mask_s = offset_in_s < SCALE_HIDDEN_SIZE
    for token_id in range(start_token_id, total_token_num, grid_num):
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)
        to_copy_s = tl.load(
            recv_x_scale + token_id * recv_x_scale_stride0 + offset_in_s, mask=mask_s
        )
        for topk_index in tl.range(0, topk_num, 1, num_stages=4):
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)
            if expert_id >= 0:
                dest_token_index = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest_token_index = dest_token_index.to(tl.int64)
                tl.store(
                    output_index + token_id * output_index_stride0 + topk_index,
                    dest_token_index,
                )
                output_tensor_ptr = (
                    output_tensor + dest_token_index * output_tensor_stride0
                )
                output_tensor_scale_ptr = (
                    output_tensor_scale + dest_token_index * output_tensor_scale_stride0
                )
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)
                tl.store(output_tensor_scale_ptr + offset_in_s, to_copy_s, mask=mask_s)


# copy from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/deepep_scatter_gather.py
@torch.no_grad()
def ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
):
    BLOCK_E = 128  # token num of per expert is aligned to 128
    BLOCK_D = 128  # block size of quantization
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    # grid = (triton.cdiv(hidden_size, BLOCK_D), num_experts)
    grid = num_experts
    assert m_indices.shape[0] % BLOCK_E == 0
    _fwd_kernel_ep_scatter_1[(grid,)](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )
    grid = min(recv_topk.shape[0], 1024 * 8)
    _fwd_kernel_ep_scatter_2[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        output_tensor_scale.stride(0),
        output_tensor_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        SCALE_HIDDEN_SIZE=hidden_size // BLOCK_D,
        SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size // BLOCK_D),
    )
    return


# ---- BF16 scatter kernel (no scale handling) ----


@triton.jit
def _fwd_kernel_ep_scatter_bf16(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_topk,
    recv_topk_stride0,
    output_tensor,
    output_tensor_stride0,
    output_index,
    output_index_stride0,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)
    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE
    for token_id in range(start_token_id, total_token_num, grid_num):
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)
        for topk_index in tl.range(0, topk_num, 1, num_stages=4):
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)
            if expert_id >= 0:
                dest_token_index = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest_token_index = dest_token_index.to(tl.int64)
                tl.store(
                    output_index + token_id * output_index_stride0 + topk_index,
                    dest_token_index,
                )
                output_tensor_ptr = (
                    output_tensor + dest_token_index * output_tensor_stride0
                )
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)


@torch.no_grad()
def ep_scatter_bf16(
    recv_x: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
):
    """BF16 version of ep_scatter (no FP8 scale handling)."""
    BLOCK_E = 128
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]

    grid = num_experts
    assert m_indices.shape[0] % BLOCK_E == 0
    _fwd_kernel_ep_scatter_1[(grid,)](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )
    grid = min(recv_topk.shape[0], 1024 * 8)
    _fwd_kernel_ep_scatter_bf16[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_topk,
        recv_topk.stride(0),
        output_tensor,
        output_tensor.stride(0),
        output_index,
        output_index.stride(0),
        topk_num=recv_topk.shape[1],
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
    )
    return


# ---- ep_gather (dtype-agnostic, works for both FP8 output and BF16 output) ----


@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block = tl.program_id(0)
    start_cur_token = tl.program_id(1)
    grid_num = tl.num_programs(1)
    for cur_token in range(start_cur_token, total_token_num, grid_num):
        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index in range(0, topk_num):
            expert_id = tl.load(
                recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index
            )
            if expert_id >= 0:
                source_token_index = tl.load(
                    input_index + cur_token * input_index_stride0 + topk_index
                )
                acc_weight = tl.load(
                    recv_topk_weight + cur_token * recv_topk_weight_stride0 + topk_index
                )
                tmp = tl.load(
                    input_tensor
                    + source_token_index * input_tensor_stride0
                    + cur_block * BLOCK_D
                    + off_d
                )
                accumulator += tmp.to(tl.float32) * acc_weight
        tl.store(
            output_tensor
            + cur_token * output_tensor_stride0
            + cur_block * BLOCK_D
            + off_d,
            accumulator.to(output_tensor.dtype.element_ty),
        )


@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
):
    BLOCK_D = 1024  # block size of quantization
    num_warps = 2
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    assert hidden_size % BLOCK_D == 0
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))
    _fwd_kernel_ep_gather[grid](
        num_tokens,
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=recv_topk_ids.shape[1],
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
    )
    return


# ---------------------------------------------------------------------------
# DeepGEMM wrappers
# ---------------------------------------------------------------------------


def _deepgemm_grouped_fp8_nt_contiguous(
    input_tuple: Tuple[torch.Tensor, torch.Tensor],
    w_tuple: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
):
    if hasattr(deep_gemm, "m_grouped_gemm_fp8_fp8_bf16_nt_contiguous"):
        return deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            input_tuple, w_tuple, out, m_indices
        )
    if hasattr(deep_gemm, "m_grouped_fp8_gemm_nt_contiguous"):
        return deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            input_tuple, w_tuple, out, m_indices
        )
    raise RuntimeError("deep_gemm version mismatch")


# ---------------------------------------------------------------------------
# fused_moe_v3 (FP8 path)
# ---------------------------------------------------------------------------


def fused_moe_v3(
    hidden_states_fp8: Tuple[torch.Tensor, torch.Tensor],
    topk_idx,
    topk_weights,
    w13_weight_fp8: Tuple[torch.Tensor, torch.Tensor],
    w2_weight_fp8: Tuple[torch.Tensor, torch.Tensor],
    num_recv_tokens_per_expert: Optional[List[int]],
):
    hidden_states_fp8, hidden_states_scale = hidden_states_fp8
    if num_recv_tokens_per_expert is None:
        return hidden_states_fp8.to(torch.bfloat16)
    all_tokens = sum(num_recv_tokens_per_expert)
    if all_tokens <= 0:
        return hidden_states_fp8.to(torch.bfloat16)
    M, K = hidden_states_fp8.size()
    N = w13_weight_fp8[0].size(1)
    scale_block_size = 128

    gather_out = torch.empty_like(
        hidden_states_fp8,
        device=hidden_states_fp8.device,
        dtype=torch.bfloat16,
    )
    input_tensor = torch.empty(
        (all_tokens, K),
        device=hidden_states_fp8.device,
        dtype=hidden_states_fp8.dtype,
    )
    input_tensor_scale = torch.empty(
        (all_tokens, K // 128),
        device=hidden_states_fp8.device,
        dtype=torch.float32,
    )
    m_indices = torch.empty(
        all_tokens, device=hidden_states_fp8.device, dtype=torch.int32
    )
    output_index = torch.empty_like(topk_idx)
    num_recv_tokens_per_expert_gpu = torch.tensor(
        num_recv_tokens_per_expert,
        dtype=torch.int32,
        pin_memory=True,
        device="cpu",
    ).cuda(non_blocking=True)
    expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)
    ep_scatter(
        hidden_states_fp8,
        hidden_states_scale,
        topk_idx,
        num_recv_tokens_per_expert_gpu,
        expert_start_loc,
        input_tensor,
        input_tensor_scale,
        m_indices,
        output_index,
    )

    del hidden_states_fp8
    gateup_output = torch.empty(
        (all_tokens, N),
        device=gather_out.device,
        dtype=torch.bfloat16,
    )
    input_tensor_scale = tma_align_input_scale(input_tensor_scale)
    assert use_deep_gemm, "Please install deep_gemm"
    _deepgemm_grouped_fp8_nt_contiguous(
        [input_tensor, input_tensor_scale], w13_weight_fp8, gateup_output, m_indices
    )

    down_input = torch.empty(
        (
            all_tokens,
            N // 2,
        ),
        device=gateup_output.device,
        dtype=torch.bfloat16,
    )
    down_input_scale = torch.empty(
        (
            all_tokens,
            N // 2,
        ),
        device=gateup_output.device,
        dtype=torch.float32,
    )
    silu_and_mul(gateup_output.view(-1, N), down_input)

    down_output = torch.empty(
        (all_tokens, K),
        device=gather_out.device,
        dtype=torch.bfloat16,
    )
    down_input_fp8, down_input_scale = per_token_group_quant_fp8(
        down_input,
        scale_block_size,
    )
    down_input_scale = tma_align_input_scale(down_input_scale)
    _deepgemm_grouped_fp8_nt_contiguous(
        (down_input_fp8, down_input_scale),
        w2_weight_fp8,
        down_output,
        m_indices,
    )

    ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)

    return gather_out


# ---------------------------------------------------------------------------
# fused_moe_v3_bf16 (BF16 path — no FP8 quantization)
# ---------------------------------------------------------------------------


def fused_moe_v3_bf16(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    num_recv_tokens_per_expert: Optional[List[int]],
):
    """
    BF16 fused MoE computation for prefill (normal) mode.
    Uses BF16 scatter + BF16 grouped GEMM + silu_and_mul + gather.
    """
    if num_recv_tokens_per_expert is None:
        return hidden_states.to(torch.bfloat16)
    all_tokens = sum(num_recv_tokens_per_expert)
    if all_tokens <= 0:
        return hidden_states.to(torch.bfloat16)

    M, K = hidden_states.size()
    N = w13_weight.size(1)  # intermediate_size * 2

    # Prepare output buffer (same shape as hidden_states but BF16)
    gather_out = torch.empty(
        (M, K),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )

    # Scatter: reorder tokens by expert (BF16 version, no scales)
    input_tensor = torch.empty(
        (all_tokens, K),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    m_indices = torch.empty(all_tokens, device=hidden_states.device, dtype=torch.int32)
    output_index = torch.empty_like(topk_idx)
    num_recv_tokens_per_expert_gpu = torch.tensor(
        num_recv_tokens_per_expert,
        dtype=torch.int32,
        pin_memory=True,
        device="cpu",
    ).cuda(non_blocking=True)
    expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)

    ep_scatter_bf16(
        hidden_states,
        topk_idx,
        num_recv_tokens_per_expert_gpu,
        expert_start_loc,
        input_tensor,
        m_indices,
        output_index,
    )

    # Gate-Up GEMM (BF16)
    gateup_output = torch.empty(
        (all_tokens, N),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    assert use_deep_gemm, "Please install deep_gemm"
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
        input_tensor, w13_weight, gateup_output, m_indices
    )

    # SiLU + mul (BF16, no re-quantization needed)
    down_input = torch.empty(
        (all_tokens, N // 2),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    silu_and_mul(gateup_output.view(-1, N), down_input)
    del gateup_output

    # Down GEMM (BF16)
    down_output = torch.empty(
        (all_tokens, K),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
        down_input, w2_weight, down_output, m_indices
    )
    del down_input

    # Gather: reorder back and apply topk_weights
    ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)

    return gather_out
