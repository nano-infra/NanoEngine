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
from torch import nn

from dlengine.context.distributed import get_dist_context


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

    def __init__(self, use_k3_sp: bool = False) -> None:
        super().__init__()
        self._original_bs: int = 0
        self._k3_sp = None
        if use_k3_sp and torch.cuda.is_available():
            from dlengine.kernel.jit.sgl.communicator import get_k3_sp_communicator

            self._k3_sp = get_k3_sp_communicator()

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

    def reduce_scatter(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Reduce TP-partial attention output directly into FFN row shards.

        K3's attention output projection is row parallel.  Deferring its
        all-reduce lets this operation replace ``all_reduce + chunk`` with one
        NCCL reduce-scatter while preserving the independent FFN-EP topology.
        """
        ctx = get_dist_context()
        src_tp = ctx.attn_tp_world_size
        if src_tp <= 1:
            return hidden_states
        bs = hidden_states.shape[0]
        self._original_bs = bs
        remainder = bs % src_tp
        if remainder:
            hidden_states = F.pad(hidden_states, (0, 0, 0, src_tp - remainder))
        output = torch.empty(
            (hidden_states.shape[0] // src_tp, *hidden_states.shape[1:]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if self._k3_sp is not None:
            from dlengine.kernel.jit.sgl import sp_collective

            dispatch = sp_collective.get_dispatch(
                "reduce_scatter",
                src_tp,
                hidden_states.shape[-1],
                hidden_states.shape[0],
                hidden_states.device,
            )
            if dispatch is not None and dispatch.strategy == "push":
                sp_collective.register_comm(
                    self._k3_sp.obj, pull_sem_mc_ptr=self._k3_sp.pull_sem_mc_ptr
                )
                return sp_collective.reduce_scatter_res(
                    src_tp,
                    hidden_states.contiguous(),
                    output,
                    tuning=dispatch.tuning,
                )
        dist.reduce_scatter_tensor(
            output, hidden_states.contiguous(), group=ctx.attn_tp_group
        )
        return output

    def local_rows(self, hidden_states: torch.Tensor) -> slice:
        """Rows owned by this attention-TP rank after scatter/padding."""
        ctx = get_dist_context()
        tp = ctx.attn_tp_world_size
        if tp <= 1:
            return slice(0, hidden_states.shape[0])
        padded = ((hidden_states.shape[0] + tp - 1) // tp) * tp
        shard = padded // tp
        start = ctx.attn_tp_rank * shard
        return slice(start, min(start + shard, hidden_states.shape[0]))


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
        self._k3_sp = scatter_layer._k3_sp if scatter_layer is not None else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_dist_context()
        dst_tp = ctx.attn_tp_world_size
        if dst_tp <= 1:
            return hidden_states
        out = torch.empty(
            (hidden_states.shape[0] * dst_tp, *hidden_states.shape[1:]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        gathered = False
        if self._k3_sp is not None:
            from dlengine.kernel.jit.sgl import sp_collective

            dispatch = sp_collective.get_dispatch(
                "all_gather",
                dst_tp,
                hidden_states.shape[-1],
                out.shape[0],
                hidden_states.device,
            )
            if dispatch is not None:
                sp_collective.register_comm(
                    self._k3_sp.obj, pull_sem_mc_ptr=self._k3_sp.pull_sem_mc_ptr
                )
                if dispatch.strategy == "push":
                    out = sp_collective.all_gather(
                        dst_tp,
                        hidden_states.contiguous(),
                        out,
                        ws_mc_base=self._k3_sp.mc_base_ptr,
                        tuning=dispatch.tuning,
                    )
                    gathered = True
                elif dispatch.strategy == "direct":
                    out, mc_ptr = self._k3_sp.symmetric_buffer(
                        "sp_all_gather", out.shape[0], out.shape[1], out.dtype
                    )
                    out = sp_collective.all_gather_direct(
                        dst_tp,
                        hidden_states.contiguous(),
                        out,
                        output_mc_ptr=mc_ptr,
                        tuning=dispatch.tuning,
                    )
                    gathered = True
        if not gathered:
            dist.all_gather_into_tensor(
                out, hidden_states.contiguous(), group=ctx.attn_tp_group
            )

        # Strip padding if AttnToFfnTransition padded the batch
        if self._scatter_layer is not None and self._scatter_layer._original_bs > 0:
            original_bs = self._scatter_layer._original_bs
            if out.shape[0] > original_bs:
                out = out[:original_bs]

        return out
