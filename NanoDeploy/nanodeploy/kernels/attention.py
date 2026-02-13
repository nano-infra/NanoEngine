from typing import Optional, Union

import torch
import triton
import triton.language as tl
from flash_attn_interface import _flash_attn_forward
from triton.language.extra import libdevice


@triton.jit
def kernel_inter_rank_gqa_fwd_batch_decode_combine_kv(
    Mid_O,  # output tensor
    Mid_LSE,  # lse tensor
    o,
    B_Seqlens,
    batch,
    q_heads,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_lse_ob,
    stride_mid_lse_oh,
    stride_mid_lse_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len_ptr = B_Seqlens + cur_batch

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_lse = cur_batch * stride_mid_lse_ob + cur_head * stride_mid_lse_oh

    for split_kv_id in range(0, NUM_KV_SPLITS):
        effective_kv_len = tl.load(cur_batch_seq_len_ptr + split_kv_id * batch)

        if effective_kv_len > 0:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_LSE + offs_lse + split_kv_id * stride_mid_lse_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = libdevice.fast_expf(e_max - n_e_max)
            acc *= old_scale
            exp_logic = libdevice.fast_expf(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def inter_rank_gqa_fwd_batch_decode_combine_kv(
    all_ranks_output_combine: torch.Tensor,
    all_ranks_lse_output_combine: torch.Tensor,
    context_lens: torch.Tensor,
    num_attention_heads: int,
    v_head_dim: int,
    max_num_seqs: int,
    sp_size: int,
):
    final_output = torch.empty(
        (max_num_seqs, num_attention_heads, v_head_dim),
        device=all_ranks_output_combine.device,
        dtype=all_ranks_output_combine.dtype,
    )
    kernel_inter_rank_gqa_fwd_batch_decode_combine_kv[
        (max_num_seqs, num_attention_heads, 1)
    ](
        all_ranks_output_combine,
        all_ranks_lse_output_combine,
        final_output,
        context_lens,
        max_num_seqs,
        num_attention_heads,
        all_ranks_output_combine.stride(1),  # batch stride for output
        all_ranks_output_combine.stride(2),  # head stride for output
        all_ranks_output_combine.stride(0),  # num_ranks stride for output
        all_ranks_lse_output_combine.stride(1),  # batch stride for lse
        all_ranks_lse_output_combine.stride(2),  # head stride for lse
        all_ranks_lse_output_combine.stride(0),  # num_ranks stride for lse
        final_output.stride(0),  # batch stride for final output
        final_output.stride(1),  # head stride for final output
        sp_size,  # split_kv
        512,  # BLOCK_DV
        v_head_dim,  # Lv
    )
    return final_output
