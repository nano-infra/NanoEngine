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
        out = torch.empty(
            dst_tp * hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        dist.all_gather_into_tensor(out, hidden_states, group=ctx.attn_tp_group)

        # Strip padding if AttnToFfnTransition padded the batch
        if self._scatter_layer is not None and self._scatter_layer._original_bs > 0:
            original_bs = self._scatter_layer._original_bs
            if out.shape[0] > original_bs:
                out = out[:original_bs]

        return out


class AttnDpToFfnTransition(nn.Module):
    """AllGather across attn_dp group to give all ranks all tokens before FFN.

    When attention uses DP replication (attention_dp > 1), each DP group holds
    a different subset of tokens.  Before entering FFN with full TP (no EP),
    all ranks need all tokens.  This layer AllGathers along the DP dimension.

    During prefill (eager), DP groups may have different batch sizes (real
    prompt vs dummy).  We AllReduce(max) to find the largest batch, pad the
    smaller ones, then AllGather.  During decode (graph capture/replay),
    batch sizes are guaranteed equal so padding is skipped.

    When attention_dp <= 1, this is a no-op.
    """

    def __init__(self) -> None:
        super().__init__()
        self._original_bs: int = 0
        self._padded_bs: int = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from nanodeploy.context.context import get_context

        ctx = get_dist_context()
        dp = ctx.attn_dp_world_size
        if dp <= 1:
            return hidden_states

        self._original_bs = hidden_states.shape[0]

        # Prefill: DP groups may have different token counts (real vs dummy).
        # Decode / graph capture: sizes are uniform, skip the AllReduce+pad.
        if get_context().is_prefill:
            local_bs = torch.tensor(
                [hidden_states.shape[0]],
                device=hidden_states.device,
                dtype=torch.int64,
            )
            dist.all_reduce(local_bs, op=dist.ReduceOp.MAX, group=ctx.attn_dp_group)
            max_bs = local_bs.item()
            if hidden_states.shape[0] < max_bs:
                hidden_states = F.pad(
                    hidden_states, (0, 0, 0, max_bs - hidden_states.shape[0])
                )
        else:
            max_bs = hidden_states.shape[0]

        self._padded_bs = max_bs

        # all_gather_into_tensor avoids ConcatD kernel (~2ms/step).
        # Buffer allocated inside forward so graph capture records it.
        gather_buf = torch.empty(
            dp * hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        dist.all_gather_into_tensor(gather_buf, hidden_states, group=ctx.attn_dp_group)
        return gather_buf


class FfnToAttnDpTransition(nn.Module):
    """Slice by attn_dp_rank to restore DP-local tokens after FFN.

    The inverse of ``AttnDpToFfnTransition``: after the FFN phase produces
    output for all tokens, each DP group slices out only its own tokens.
    Handles padding introduced by the gather layer during prefill.

    When attention_dp <= 1, this is a no-op.
    """

    def __init__(self, gather_layer: "AttnDpToFfnTransition") -> None:
        super().__init__()
        self._gather_layer = gather_layer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_dist_context()
        dp = ctx.attn_dp_world_size
        if dp <= 1:
            return hidden_states

        padded_bs = self._gather_layer._padded_bs
        orig_bs = self._gather_layer._original_bs
        dp_rank = ctx.attn_dp_rank
        # hidden_states: [padded_bs * dp, H] — slice out this rank's portion,
        # trimming any padding added during prefill.
        start = dp_rank * padded_bs
        return hidden_states[start : start + orig_bs].contiguous()
