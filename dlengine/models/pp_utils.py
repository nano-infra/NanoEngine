"""Helpers for pipeline parallelism (PP).

The engine splits a model's decoder layers into ``pp`` contiguous stages. Each stage builds only the layers it owns (plus the input
embedding on the first stage and the final norm / lm_head on the last stage)
and exchanges the residual stream with its neighbours via point-to-point
send/recv along the pipeline dimension.

To keep global layer indices (checkpoint names, KV cache slots, state dicts)
aligned across stages, ``make_pp_layers`` returns a full-length ``ModuleList``
in which layers outside this stage's range are :class:`PPMissingLayer`
placeholders that never execute.
"""

from typing import Callable, Tuple

import torch
import torch.distributed as dist
from torch import nn

from dlengine.context_v2.distributed import get_dist_context


def get_pp_layer_range(
    num_layers: int, final_stage_start: int | None = None
) -> Tuple[int, int]:
    """Return the ``[start, end)`` decoder-layer range owned by this stage.

    Layers are split as evenly as possible; when ``num_layers`` is not a
    multiple of the pipeline size, the first ``num_layers % pp`` stages get one
    extra layer each.
    """
    ctx = get_dist_context()
    pp_size = ctx.pp_world_size
    pp_rank = ctx.pp_rank
    if pp_size <= 1:
        return 0, num_layers
    # Some architectures have cross-layer dependencies which must remain on
    # one stage.  Gemma4 shared-KV layers, for example, consume K/V produced by
    # the last non-sharing layer of each attention type.  Reserve that suffix
    # for the final stage and balance the independent prefix over the others.
    if final_stage_start is not None and pp_size > 1:
        if not 0 < final_stage_start < num_layers:
            raise ValueError(
                f"invalid final_stage_start={final_stage_start} for {num_layers} layers"
            )
        if final_stage_start < pp_size - 1:
            raise ValueError(
                "not enough independent prefix layers for pipeline stages: "
                f"prefix={final_stage_start}, pp={pp_size}"
            )
        if pp_rank == pp_size - 1:
            return final_stage_start, num_layers
        split_size = pp_size - 1
        base = final_stage_start // split_size
        remainder = final_stage_start % split_size
    else:
        split_size = pp_size
        base = num_layers // split_size
        remainder = num_layers % split_size

    start = pp_rank * base + min(pp_rank, remainder)
    count = base + (1 if pp_rank < remainder else 0)
    return start, start + count


def get_gemma4_shared_kv_source_start(config) -> int | None:
    """Return the earliest KV source that must accompany Gemma4's shared tail."""
    num_layers = config.num_hidden_layers
    num_shared = int(getattr(config, "num_kv_shared_layers", 0) or 0)
    if num_shared <= 0:
        return None
    first_shared = num_layers - num_shared
    layer_types = list(config.layer_types)
    shared_types = set(layer_types[first_shared:])
    source_indices = []
    for layer_type in shared_types:
        candidates = [
            idx
            for idx, candidate_type in enumerate(layer_types[:first_shared])
            if candidate_type == layer_type
        ]
        if not candidates:
            raise ValueError(
                "Gemma4 shared-KV layer has no preceding source for "
                f"layer_type={layer_type!r}"
            )
        source_indices.append(candidates[-1])
    return min(source_indices)


def get_gemma4_pp_layer_range(config) -> Tuple[int, int]:
    return get_pp_layer_range(
        config.num_hidden_layers,
        final_stage_start=get_gemma4_shared_kv_source_start(config),
    )


class PPMissingLayer(nn.Module):
    """Placeholder for a decoder layer that lives on another pipeline stage.

    Occupies a ``ModuleList`` slot so global layer indexing stays aligned, but
    holds no parameters and must never be invoked.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):  # pragma: no cover - defensive
        raise RuntimeError(
            "PPMissingLayer.forward called: a layer outside this pipeline "
            "stage's range was executed"
        )


def make_pp_layers(
    num_layers: int,
    builder: Callable[[int], nn.Module],
    layer_range: Tuple[int, int] | None = None,
) -> Tuple[int, int, nn.ModuleList]:
    """Build a full-length ``ModuleList`` with only this stage's layers real.

    Args:
        num_layers: total decoder layers in the model.
        builder: ``builder(global_layer_idx) -> nn.Module`` constructing one
            real decoder layer.

    Returns:
        ``(start_layer, end_layer, layers)`` where ``layers[i]`` is a real
        module for ``start_layer <= i < end_layer`` and a
        :class:`PPMissingLayer` otherwise.
    """
    start, end = layer_range or get_pp_layer_range(num_layers)
    modules = []
    for idx in range(num_layers):
        if start <= idx < end:
            modules.append(builder(idx))
        else:
            modules.append(PPMissingLayer())
    return start, end, nn.ModuleList(modules)


def pp_send_hidden(hidden_states: torch.Tensor) -> None:
    """Send the residual stream to the next pipeline stage."""
    ctx = get_dist_context()
    dist.send(hidden_states.contiguous(), dst=ctx.pp_next_global_rank)


def pp_recv_hidden(
    num_tokens: int, hidden_size: int, dtype: torch.dtype
) -> torch.Tensor:
    """Receive the residual stream from the previous pipeline stage."""
    ctx = get_dist_context()
    buffer = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")
    dist.recv(buffer, src=ctx.pp_prev_global_rank)
    return buffer


def pp_weight_belongs_to_stage(
    weight_name: str, start_layer: int, end_layer: int
) -> bool:
    """Decide whether a checkpoint weight should be loaded on this stage.

    - decoder-layer weights (``...layers.<idx>...``) load only when ``idx`` is
      in ``[start_layer, end_layer)``;
    - the input embedding loads only on the first stage;
    - the final norm and ``lm_head`` load only on the last stage;
    - all other weights load on every stage.

    Tied-embedding models are handled by the caller (the last stage also copies
    ``embed_tokens`` into ``lm_head``).
    """
    import re

    ctx = get_dist_context()
    m = re.search(r"layers\.(\d+)\.", weight_name)
    if m is not None:
        idx = int(m.group(1))
        return start_layer <= idx < end_layer
    if "embed_tokens" in weight_name:
        return ctx.is_first_pp_stage
    if weight_name.startswith("model.norm."):
        return ctx.is_last_pp_stage
    if weight_name.startswith("model.hc_head."):
        return ctx.is_last_pp_stage
    if "lm_head" in weight_name:
        return ctx.is_last_pp_stage
    return True
