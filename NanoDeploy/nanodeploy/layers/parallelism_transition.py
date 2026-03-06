"""Parallelism transition layers for asymmetric Attention-TP / FFN-EP configurations.

When the attention phase uses TP (e.g., attn_tp=2) and the FFN phase uses EP
(e.g., ffn_ep=8), hidden states must be redistributed between the two phases:

  Attention (TP=2) → chunk → FFN (EP=8) → AllGather → Attention (TP=2)

These layers are nn.Modules so they compose naturally in DecoderLayer.forward()
and degrade to nn.Identity()-equivalent no-ops when no transition is needed.
"""

import torch
import torch.distributed as dist
from nanodeploy.context.distributed import get_dist_context
from torch import nn


class AttnToFfnTransition(nn.Module):
    """Scatter transition: Attention → FFN.

    After attention's TP all-reduce, all GPUs in a TP pair hold identical
    hidden states for the same batch subset.  This layer chunks the batch
    so each GPU gets a unique slice before entering EP dispatch.

    When attn_tp <= 1 the layer is a no-op.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_dist_context()
        src_tp = ctx.attn_tp_world_size
        if src_tp <= 1:
            return hidden_states
        tp_rank = ctx.attn_tp_rank
        return hidden_states.chunk(src_tp, dim=0)[tp_rank].contiguous()


class FfnToAttnTransition(nn.Module):
    """Gather transition: FFN → Attention.

    After FFN (EP dispatch + combine), each GPU holds results for its chunk
    of the batch.  This layer AllGathers within the attn_tp group to restore
    the full batch on all TP-pair GPUs for the next attention layer.

    When attn_tp <= 1 the layer is a no-op.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_dist_context()
        dst_tp = ctx.attn_tp_world_size
        if dst_tp <= 1:
            return hidden_states
        gathered = [torch.empty_like(hidden_states) for _ in range(dst_tp)]
        dist.all_gather(gathered, hidden_states, group=ctx.attn_tp_group)
        return torch.cat(gathered, dim=0)
