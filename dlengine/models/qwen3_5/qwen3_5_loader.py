"""Per-model weight loader for Qwen3.5 dense."""

import re
from typing import Generator, Tuple

import torch
from torch import nn

from dlengine.context.distributed import get_dist_context
from dlengine.logging import get_logger
from dlengine.models.pp_utils import pp_weight_belongs_to_stage
from dlengine.worker.loader import default_weight_loader

logger = get_logger()

_PACKED_MODULES_MAPPING = {
    "self_attn.q_proj": ("self_attn.qkv_proj", "q"),
    "self_attn.k_proj": ("self_attn.qkv_proj", "k"),
    "self_attn.v_proj": ("self_attn.qkv_proj", "v"),
    "mlp.gate_proj": ("mlp.gate_up_proj", 0),
    "mlp.up_proj": ("mlp.gate_up_proj", 1),
}


def _normalize_weight_name(weight_name: str) -> str | None:
    if weight_name.startswith("model.language_model."):
        return "model." + weight_name[len("model.language_model.") :]
    if weight_name.startswith("language_model."):
        return "model." + weight_name[len("language_model.") :]
    if weight_name.startswith(
        ("model.embed_tokens.", "model.layers.", "model.norm.", "lm_head.")
    ):
        return weight_name
    return None


def load_weights(
    model: nn.Module,
    weights: Generator[Tuple[str, str, torch.Tensor], None, None],
) -> None:
    """Load Qwen3.5 dense language-model weights.

    Qwen3.5 checkpoints can include vision and MTP tensors. This loader only
    consumes the main language model tensors under ``model.language_model``.
    """
    loaded_count = 0
    skipped_count = 0
    not_found_names: list[str] = []

    ctx = get_dist_context()
    pp_size = ctx.pp_world_size
    start_layer = getattr(model.model, "start_layer", 0)
    end_layer = getattr(model.model, "end_layer", None)
    tie_word_embeddings = bool(getattr(model.config, "tie_word_embeddings", False))

    for weight_name, raw_weight_name, tensor in weights:
        param_weight_name = _normalize_weight_name(weight_name)
        if param_weight_name is None:
            skipped_count += 1
            continue

        if pp_size > 1 and end_layer is not None:
            if not pp_weight_belongs_to_stage(
                param_weight_name, start_layer, end_layer
            ):
                if (
                    tie_word_embeddings
                    and ctx.is_last_pp_stage
                    and "embed_tokens" in param_weight_name
                    and getattr(model, "lm_head", None) is not None
                ):
                    lm_head_param = model.lm_head.weight
                    loader = getattr(
                        lm_head_param, "weight_loader", default_weight_loader
                    )
                    loader(lm_head_param, tensor)
                    loaded_count += 1
                continue

        matched = False
        for k, (v, shard_id) in _PACKED_MODULES_MAPPING.items():
            if k in param_weight_name:
                param_name = param_weight_name.replace(k, v)
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    continue
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, tensor, shard_id, weight_name)
                matched = True
                loaded_count += 1
                break

        if matched:
            continue

        try:
            param = model.get_parameter(param_weight_name)
        except AttributeError:
            not_found_names.append(param_weight_name)
            skipped_count += 1
            continue

        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, tensor)
        loaded_count += 1

    logger.warning(
        f"Weight loading complete: {loaded_count} loaded, {skipped_count} skipped"
    )
    if not_found_names:
        unique_patterns = set()
        for name in not_found_names:
            unique_patterns.add(re.sub(r"layers\.\d+\.", "layers.N.", name))
        logger.warning(
            f"  {len(not_found_names)} weights NOT FOUND in model "
            f"(unique patterns: {sorted(unique_patterns)})"
        )

    model_params = set(name for name, _ in model.named_parameters())
    logger.warning(f"  Model has {len(model_params)} parameters total")
