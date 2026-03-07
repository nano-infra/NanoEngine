"""Parallelism transition layers for asymmetric Attention-TP / FFN-EP configurations.

When the attention phase uses TP (e.g., attn_tp=2) and the FFN phase uses EP
(e.g., ffn_ep=8), hidden states must be redistributed between the two phases:

  Attention (TP=2) → chunk → FFN (EP=8) → AllGather → Attention (TP=2)

These layers are nn.Modules so they compose naturally in DecoderLayer.forward()
and degrade to nn.Identity()-equivalent no-ops when no transition is needed.

Batch padding: when batch_size is not divisible by attn_tp (e.g., bs=1, tp=2),
AttnToFfnTransition pads with zeros before chunking.  FfnToAttnTransition
AllGathers and slices back to the original batch size.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from nanodeploy.context.distributed import get_dist_context
from torch import nn


class AttnToFfnTransition(nn.Module):
    """Scatter transition: Attention → FFN.

    After attention's TP all-reduce, all GPUs in a TP pair hold identical
    hidden states for the same batch subset.  This layer pads the batch to
    a multiple of attn_tp (if needed), then chunks so each GPU gets a unique
    slice before entering EP dispatch.

    Stores ``_original_bs`` so that ``FfnToAttnTransition`` can strip the
    padding after AllGather.

    When attn_tp <= 1 the layer is a no-op.
    """

    def __init__(self) -> None:
        super().__init__()
        self._original_bs: int = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_dist_context()
        src_tp = ctx.attn_tp_world_size
        if src_tp <= 1:
            return hidden_states

        bs = hidden_states.shape[0]
        self._original_bs = bs

        # Pad to multiple of src_tp
        remainder = bs % src_tp
        if remainder != 0:
            pad_size = src_tp - remainder
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))

        tp_rank = ctx.attn_tp_rank
        return hidden_states.chunk(src_tp, dim=0)[tp_rank].contiguous()


class FfnToAttnTransition(nn.Module):
    """Gather transition: FFN → Attention.

    After FFN (EP dispatch + combine), each GPU holds results for its chunk
    of the batch.  This layer AllGathers within the attn_tp group to restore
    the full batch on all TP-pair GPUs for the next attention layer.

    If the batch was padded by ``AttnToFfnTransition``, the result is sliced
    back to the original batch size.

    When attn_tp <= 1 the layer is a no-op.
    """

    def __init__(self, scatter_layer: "AttnToFfnTransition | None" = None) -> None:
        super().__init__()
        self._scatter_layer = scatter_layer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_dist_context()
        dst_tp = ctx.attn_tp_world_size
        if dst_tp <= 1:
            return hidden_states
        gathered = [torch.empty_like(hidden_states) for _ in range(dst_tp)]
        dist.all_gather(gathered, hidden_states, group=ctx.attn_tp_group)
        out = torch.cat(gathered, dim=0)

        # Strip padding if AttnToFfnTransition padded the batch
        if self._scatter_layer is not None and self._scatter_layer._original_bs > 0:
            original_bs = self._scatter_layer._original_bs
            if out.shape[0] > original_bs:
                out = out[:original_bs]

        return out
