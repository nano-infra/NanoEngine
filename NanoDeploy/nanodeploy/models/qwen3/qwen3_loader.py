"""Per-model weight loader for Qwen3.

Handles:
- packed_modules_mapping: q/k/v_proj -> qkv_proj, gate/up_proj -> gate_up_proj
- Default param.weight_loader for all linear layers
- No experts, no special transforms (simplest loader)
"""

import re
from typing import Generator, Tuple

import torch
from nanodeploy.logging import get_logger
from nanodeploy.worker.loader import default_weight_loader
from torch import nn

logger = get_logger()

# Qwen3 packed modules: QKV packing + gate_up packing
_PACKED_MODULES_MAPPING = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}


def load_weights(
    model: nn.Module,
    weights: Generator[Tuple[str, str, torch.Tensor], None, None],
) -> None:
    """Load weights for Qwen3 model.

    Args:
        model: The Qwen3ForCausalLM model instance
        weights: Generator yielding (weight_name, raw_weight_name, tensor) tuples
    """
    loaded_count = 0
    skipped_count = 0
    not_found_names: list[str] = []

    for weight_name, raw_weight_name, tensor in weights:
        # packed_modules_mapping: q/k/v -> qkv_proj, gate/up -> gate_up_proj
        matched = False
        for k, (v, shard_id) in _PACKED_MODULES_MAPPING.items():
            if k in weight_name:
                param_name = weight_name.replace(k, v)
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

        # Default: direct parameter loading
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
        f"Weight loading complete: {loaded_count} loaded, {skipped_count} skipped"
    )
    if not_found_names:
        unique_patterns = set()
        for n in not_found_names:
            pat = re.sub(r"layers\.\d+\.", "layers.N.", n)
            unique_patterns.add(pat)
        logger.warning(
            f"  {len(not_found_names)} weights NOT FOUND in model "
            f"(unique patterns: {sorted(unique_patterns)})"
        )

    model_params = set(name for name, _ in model.named_parameters())
    logger.warning(f"  Model has {len(model_params)} parameters total")
