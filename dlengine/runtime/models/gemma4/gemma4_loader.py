import re
from typing import Generator, Tuple

import torch
from torch import nn

from dlengine.logging import get_logger
from dlengine.runtime.context.distributed import get_dist_context
from dlengine.runtime.models.pp_utils import pp_weight_belongs_to_stage
from dlengine.runtime.runner.loader import default_weight_loader

logger = get_logger()

_SKIP_PATTERNS = (
    "rotary_emb.",
    "vision_tower.",
    "audio_tower.",
    "model.embed_audio.",
    "model.embed_vision.",
    "multi_modal_projector.",
)

_SHARED_KV_WEIGHT = re.compile(
    r"^model\.layers\.(\d+)\.self_attn\.(?:k_proj|v_proj|k_norm)\.weight$"
)


def _is_unused_shared_kv_weight(model: nn.Module, weight_name: str) -> bool:
    """Return whether a checkpoint tensor belongs to a KV-sharing layer.

    Gemma4 checkpoints retain K/V projection tensors for every layer, while the
    reference architecture intentionally does not instantiate them for the last
    ``num_kv_shared_layers`` layers. Those layers consume K/V produced by the
    last non-sharing layer of the same attention type.
    """
    match = _SHARED_KV_WEIGHT.match(weight_name)
    if match is None:
        return False
    attention = model.get_submodule(f"model.layers.{int(match.group(1))}.self_attn")
    return bool(getattr(attention, "is_kv_shared_layer", False))


def load_weights(
    model: nn.Module,
    weights: Generator[Tuple[str, str, torch.Tensor], None, None],
) -> None:
    loaded_count = 0
    skipped_count = 0
    not_found_names: list[str] = []
    ctx = get_dist_context()
    pp_size = ctx.pp_world_size
    start_layer = getattr(model.model, "start_layer", 0)
    end_layer = getattr(model.model, "end_layer", None)
    tied = bool(getattr(model.config, "tie_word_embeddings", False))

    for weight_name, _raw_weight_name, tensor in weights:
        if weight_name.startswith("language_model."):
            weight_name = weight_name[len("language_model.") :]
        if any(pattern in weight_name for pattern in _SKIP_PATTERNS):
            skipped_count += 1
            continue

        if pp_size > 1 and end_layer is not None:
            # PLE parameters (embed_tokens_per_layer and the per-layer model
            # projection) are intentionally replicated: every stage computes
            # the auxiliary input for its own decoder-layer slice.
            is_replicated_ple = any(
                name in weight_name
                for name in (
                    "embed_tokens_per_layer",
                    "per_layer_model_projection",
                    "per_layer_projection_norm",
                )
            )
            if not is_replicated_ple and not pp_weight_belongs_to_stage(
                weight_name, start_layer, end_layer
            ):
                if (
                    tied
                    and ctx.is_last_pp_stage
                    and weight_name == "model.embed_tokens.weight"
                    and getattr(model, "lm_head", None) is not None
                ):
                    loader = getattr(
                        model.lm_head.weight, "weight_loader", default_weight_loader
                    )
                    loader(model.lm_head.weight, tensor)
                    loaded_count += 1
                continue
        if _is_unused_shared_kv_weight(model, weight_name):
            skipped_count += 1
            continue

        try:
            param = model.get_parameter(weight_name)
        except AttributeError:
            not_found_names.append(weight_name)
            skipped_count += 1
            continue

        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, tensor)
        loaded_count += 1

    logger.warning(
        f"Gemma4 weight loading complete: {loaded_count} loaded, "
        f"{skipped_count} skipped"
    )
    if not_found_names:
        unique_patterns = set()
        for name in not_found_names:
            unique_patterns.add(re.sub(r"layers\.\d+\.", "layers.N.", name))
        logger.warning(
            f"  {len(not_found_names)} Gemma4 weights NOT FOUND "
            f"(unique patterns: {sorted(unique_patterns)})"
        )
