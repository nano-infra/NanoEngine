"""Per-model weight loader for Qwen3 MoE.

Handles:
- packed_modules_mapping: q/k/v_proj -> qkv_proj, gate/up_proj -> gate_up_proj,
  gate/up_scale_inv -> gate_up_scale_inv
- Per-expert weights: experts.{i}.gate_proj/up_proj/down_proj -> routed_experts 3D tensors
- Default param.weight_loader for all linear layers
"""

import re
from typing import Generator, Tuple

import torch
from nanodeploy.logging import get_logger
from nanodeploy.worker.loader import (
    EXPERT_RE,
    default_weight_loader,
    load_per_expert_weight,
)
from torch import nn

logger = get_logger()

# Qwen3 MoE packed modules
_PACKED_MODULES_MAPPING = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    "gate_scale_inv": ("gate_up_scale_inv", 0),
    "up_scale_inv": ("gate_up_scale_inv", 1),
}


def load_weights(
    model: nn.Module,
    weights: Generator[Tuple[str, str, torch.Tensor], None, None],
) -> None:
    """Load weights for Qwen3 MoE model.

    Args:
        model: The Qwen3MoeForCausalLM model instance
        weights: Generator yielding (weight_name, raw_weight_name, tensor) tuples
    """
    config = model.config

    loaded_count = 0
    skipped_count = 0
    not_found_names: list[str] = []

    for weight_name, raw_weight_name, tensor in weights:
        # Per-expert weights -> combined 3D tensors
        if "experts." in weight_name and EXPERT_RE.match(weight_name):
            if load_per_expert_weight(model, weight_name, tensor, config):
                loaded_count += 1
                continue

        # packed_modules_mapping dispatch
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
            pat = re.sub(r"experts\.\d+\.", "experts.E.", pat)
            unique_patterns.add(pat)
        logger.warning(
            f"  {len(not_found_names)} weights NOT FOUND in model "
            f"(unique patterns: {sorted(unique_patterns)})"
        )

    model_params = set(name for name, _ in model.named_parameters())
    logger.warning(f"  Model has {len(model_params)} parameters total")
