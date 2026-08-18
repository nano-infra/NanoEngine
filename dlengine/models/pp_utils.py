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

from dlengine.context.distributed import get_dist_context


def pp_layer_partition(
    num_layers: int, pp_size: int, final_stage_start: int | None = None
) -> list[Tuple[int, int]]:
    """Return the ``[start, end)`` layer range for every pipeline stage.

    This is the single source of truth for the split policy. It is used both
    when a worker builds its local model and when a decode engine needs a
    remote prefill engine's per-stage layer ownership for KV migration.

    Layers are split as evenly as possible; when ``num_layers`` is not a
    multiple of ``pp_size``, the earlier stages get one extra layer each.

    ``final_stage_start`` handles architectures with cross-layer dependencies
    that must stay on one stage (e.g. Gemma4 shared-KV layers consume K/V from
    the last non-sharing layer of each attention type): the suffix
    ``[final_stage_start, num_layers)`` is reserved for the last stage and the
    independent prefix is balanced over the remaining stages.
    """
    pp_size = max(1, pp_size)
    if pp_size == 1:
        return [(0, num_layers)]

    if final_stage_start is not None:
        if not 0 < final_stage_start < num_layers:
            raise ValueError(
                f"invalid final_stage_start={final_stage_start} for "
                f"{num_layers} layers"
            )
        if final_stage_start < pp_size - 1:
            raise ValueError(
                "not enough independent prefix layers for pipeline stages: "
                f"prefix={final_stage_start}, pp={pp_size}"
            )
        split_size = pp_size - 1
        base = final_stage_start // split_size
        remainder = final_stage_start % split_size
    else:
        split_size = pp_size
        base = num_layers // split_size
        remainder = num_layers % split_size

    ranges: list[Tuple[int, int]] = []
    start = 0
    for stage in range(split_size):
        count = base + (1 if stage < remainder else 0)
        ranges.append((start, start + count))
        start += count
    if final_stage_start is not None:
        ranges.append((final_stage_start, num_layers))
    return ranges


def pp_stage_of_layer(layer_idx: int, stage_ranges: list[Tuple[int, int]]) -> int:
    """Return the pipeline stage index that owns ``layer_idx``."""
    for stage, (start, end) in enumerate(stage_ranges):
        if start <= layer_idx < end:
            return stage
    raise ValueError(
        f"layer {layer_idx} outside any pipeline stage range {stage_ranges}"
    )


def cache_layer_indices(
    config, *, gemma_hisparse_only_full_attention: bool = False
) -> list[int]:
    """Global decoder-layer indices represented in the primary KV tensor.

    Most architectures cache every decoder layer. Hybrid linear-attention
    models cache only full-attention layers, while Gemma4 shared-KV layers
    reuse the source layer's cache and therefore have no independent slot.
    """
    num_layers = int(config.num_hidden_layers)
    arch = (getattr(config, "architectures", None) or [""])[0]
    layer_types = list(getattr(config, "layer_types", None) or [])
    if arch in ("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"):
        first_shared = num_layers - int(getattr(config, "num_kv_shared_layers", 0) or 0)
        if gemma_hisparse_only_full_attention:
            return [
                idx
                for idx, layer_type in enumerate(layer_types[:first_shared])
                if layer_type == "full_attention"
            ]
        return list(range(max(0, first_shared)))
    if layer_types and "linear_attention" in layer_types:
        return [
            idx
            for idx, layer_type in enumerate(layer_types)
            if layer_type == "full_attention"
        ]
    return list(range(num_layers))


def partition_layer_indices(
    layer_indices: list[int], stage_ranges: list[Tuple[int, int]]
) -> list[list[int]]:
    """Partition global layer indices using decoder-layer stage ownership."""
    result = [[] for _ in stage_ranges]
    for layer_idx in layer_indices:
        result[pp_stage_of_layer(layer_idx, stage_ranges)].append(layer_idx)
    return result


def pp_global_rank(
    pp_idx: int,
    dp_idx: int,
    sp_idx: int,
    tp_idx: int,
    *,
    dp_size: int,
    sp_size: int,
    tp_size: int,
) -> int:
    """Map a ``(pp, dp, sp, tp)`` mesh coordinate to flat global rank."""
    coordinates = (pp_idx, dp_idx, sp_idx, tp_idx)
    sizes = (None, dp_size, sp_size, tp_size)
    if pp_idx < 0 or any(
        coordinate < 0 or coordinate >= size
        for coordinate, size in zip(coordinates[1:], sizes[1:], strict=True)
    ):
        raise ValueError(
            f"invalid PP mesh coordinate {coordinates} for "
            f"inner shape ({dp_size}, {sp_size}, {tp_size})"
        )
    inner_world_size = dp_size * sp_size * tp_size
    inner_rank = (dp_idx * sp_size + sp_idx) * tp_size + tp_idx
    return pp_idx * inner_world_size + inner_rank


def get_pp_layer_range(
    num_layers: int, final_stage_start: int | None = None
) -> Tuple[int, int]:
    """Return the ``[start, end)`` decoder-layer range owned by this stage."""
    ctx = get_dist_context()
    partition = pp_layer_partition(num_layers, ctx.pp_world_size, final_stage_start)
    return partition[ctx.pp_rank]


def get_gemma4_shared_kv_source_start(config) -> int | None:
    """Earliest KV source that must accompany Gemma4's shared tail."""
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
