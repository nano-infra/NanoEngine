import torch

from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
from nanodeploy.kernels.attention import inter_rank_gqa_fwd_batch_decode_combine_kv

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
                q = q_buffer.all_to_all_ll(
                    q.view([bs, -1]),
                    mask=context.global_context_lens,
                ).view([sp_size * max_num_seqs, num_head, head_dim])
                context_lens = context.context_lens.view(-1)
                block_tables = context.block_tables.view(sp_size * max_num_seqs, -1)
            else:
                context_lens = context.context_lens[sp_rank][:bs]
                block_tables = context.block_tables[sp_rank][:bs]

            o, lse = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context_lens,
                page_table=block_tables,
                softmax_scale=self.scale,
                causal=False,
                return_softmax_lse=True,
            )[:2]

            if sp_size > 1:
                res_lse_buffer = get_sp_context().res_lse_buffer
                lse = lse.to(torch.bfloat16)
                gathered_o = o.view([sp_size, max_num_seqs, num_head, head_dim])
                gathered_lse = lse.view([sp_size, max_num_seqs, num_head, 1])

                all_ranks_output_combine_0 = torch.cat(
                    [gathered_o, gathered_lse], dim=3
                )
                all_ranks_output_combine = res_lse_buffer.all_to_all_ll(
                    all_ranks_output_combine_0.view(sp_size * max_num_seqs, -1),
                    mask=context.context_lens,
                    is_transpose=True,
                ).view(sp_size, max_num_seqs, num_head, head_dim + 1)

                o = inter_rank_gqa_fwd_batch_decode_combine_kv(
                    all_ranks_output_combine,
                    context.global_context_lens,
                    num_head,
                    head_dim,
                    get_sp_context().max_num_seqs,
                    sp_size,
                )

                o = o.view([max_num_seqs, num_head, head_dim])[:bs]

        return o
