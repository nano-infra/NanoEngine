import torch

from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
from nanodeploy.kernels.attention import inter_rank_gqa_fwd_batch_decode_combine_kv

from nanodeploy.kernels.copy import copy_batch_indexed_triton
from nanodeploy.kernels.kvcache import store_kvcache
from nanodeploy.logging import get_logger
from nanodeploy.worker.context import get_context
from nanodeploy.worker.distributed import get_dist_context
from nanodeploy.worker.sp_context import get_sp_context

from torch import nn


logger = get_logger()

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel() and not get_context().is_dummy:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size
        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
            )
        else:  # decode
            bs, num_head, head_dim = q.shape
            if sp_size > 1:
                max_num_seqs = get_sp_context().max_num_seqs
                q_buffer = get_sp_context().q_buffer

                # Q 拷贝
                local_q_buffer_3d = q_buffer.local_buffer.view(get_sp_context().dtype)[
                    : sp_size * max_num_seqs * num_head * head_dim
                ].view(sp_size * max_num_seqs, num_head, head_dim)
                copy_batch_indexed_triton(
                    q,
                    local_q_buffer_3d,
                    context.q_slice_get,
                    context.q_slice_fill,
                    context.q_copy_mask,
                )

                q = q_buffer.all_to_all_ll(
                    q.view([bs, -1]),
                    mask=context.q_mask,
                    offsets=context.q_offsets,
                ).view([sp_size * max_num_seqs, num_head, head_dim])

                q = q[: context.attention_compute_bs]
                context_lens_for_attn = context.context_lens_for_attn[
                    : context.attention_compute_bs
                ]
                block_tables = context.block_tables[: context.attention_compute_bs]
            else:
                q = q[: context.attention_compute_bs]
                context_lens_for_attn = context.context_lens_for_attn[
                    : context.attention_compute_bs
                ]
                block_tables = context.block_tables[: context.attention_compute_bs]

            o, lse = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context_lens_for_attn,
                page_table=block_tables,
                softmax_scale=self.scale,
                causal=False,
                return_softmax_lse=True,
            )[:2]

            if sp_size > 1:
                res_buffer = get_sp_context().res_buffer
                lse_buffer = get_sp_context().lse_buffer
                lse = lse.to(torch.bfloat16)
                gathered_o = o.view([context.attention_compute_bs, num_head, head_dim])
                gathered_lse = lse.view([context.attention_compute_bs, num_head, 1])

                # 1. 拷贝 gathered_o 到 res_local_buffer
                res_local_buffer_3d = res_buffer.local_buffer.view(
                    get_sp_context().dtype
                )[: sp_size * max_num_seqs * num_head * head_dim].view(
                    sp_size * max_num_seqs, num_head, head_dim
                )
                copy_batch_indexed_triton(
                    gathered_o.view(-1, num_head, head_dim),
                    res_local_buffer_3d,
                    context.res_slice_get_to_buffer_output,
                    context.res_slice_fill_to_buffer_output,
                    context.res_to_buffer_output_mask,
                )

                # 2. 拷贝 gathered_lse 到 lse_local_buffer
                lse_local_buffer_3d = lse_buffer.local_buffer.view(
                    get_sp_context().dtype
                )[: sp_size * max_num_seqs * num_head * 1].view(
                    sp_size * max_num_seqs, num_head, 1
                )
                copy_batch_indexed_triton(
                    gathered_lse.view(-1, num_head, 1),
                    lse_local_buffer_3d,
                    context.res_slice_get_to_buffer_output,
                    context.res_slice_fill_to_buffer_output,
                    context.res_to_buffer_output_mask,
                )

                # 3. 分配 All-to-All Input Buffer
                res_all_to_all_input_buffer = torch.empty(
                    (sp_size * max_num_seqs, num_head, head_dim),
                    dtype=gathered_o.dtype,
                    device=gathered_o.device,
                )
                lse_all_to_all_input_buffer = torch.empty(
                    (sp_size * max_num_seqs, num_head, 1),
                    dtype=gathered_lse.dtype,
                    device=gathered_lse.device,
                )

                # 4. 拷贝 gathered_o 到 res_all_to_all_input_buffer
                copy_batch_indexed_triton(
                    gathered_o.view(-1, num_head, head_dim),
                    res_all_to_all_input_buffer,
                    context.res_slice_get_to_buffer_input,
                    context.res_slice_fill_to_buffer_input,
                    context.res_to_buffer_input_mask,
                )

                # 5. 拷贝 gathered_lse 到 lse_all_to_all_input_buffer
                copy_batch_indexed_triton(
                    gathered_lse.view(-1, num_head, 1),
                    lse_all_to_all_input_buffer,
                    context.res_slice_get_to_buffer_input,
                    context.res_slice_fill_to_buffer_input,
                    context.res_to_buffer_input_mask,
                )

                all_ranks_res_output_combine = res_buffer.all_to_all_ll(
                    res_all_to_all_input_buffer.view(sp_size * max_num_seqs, -1),
                    mask=context.res_lse_mask,
                    is_transpose=True,
                ).view(sp_size, max_num_seqs, num_head, head_dim)
                all_ranks_lse_output_combine = lse_buffer.all_to_all_ll(
                    lse_all_to_all_input_buffer.view(sp_size * max_num_seqs, -1),
                    mask=context.res_lse_mask,
                    is_transpose=True,
                ).view(sp_size, max_num_seqs, num_head, 1)

                o = inter_rank_gqa_fwd_batch_decode_combine_kv(
                    all_ranks_res_output_combine,
                    all_ranks_lse_output_combine,
                    context.global_context_lens,
                    num_head,
                    head_dim,
                    get_sp_context().max_num_seqs,
                    sp_size,
                ).view([max_num_seqs, num_head, head_dim])[:bs]

        return o
