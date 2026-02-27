"""Per-model weight loader for Qwen3.5 MoE.

Handles:
- packed_modules_mapping: self_attn.q/k/v_proj -> qkv_proj,
  shared_expert.gate/up_proj -> gate_up_proj
- Per-expert weights: experts.{i}.gate_proj/up_proj/down_proj -> routed_experts 3D
  (Qwen3.5-397B-A17B-FP8 checkpoint format, with EP + TP slicing)
- Packed 3D expert weights: experts.gate_up_proj, experts.down_proj
  (EP + TP slicing applied, alternate checkpoint format)
- Packed 3D expert scales: experts.gate_up_proj_scale_inv, experts.down_proj_scale_inv
  (FP8 only, EP + TP slicing applied, alternate checkpoint format)
- Default param.weight_loader for all linear layers
"""

import re
from typing import Generator, Tuple

import torch
from nanodeploy.logging import get_logger
from nanodeploy.worker.loader import (
    EXPERT_RE,
    PACKED_EXPERT_RE,
    PACKED_EXPERT_SCALE_RE,
    default_weight_loader,
    load_packed_expert_scale,
    load_packed_expert_weight,
    load_per_expert_weight,
)
from torch import nn

logger = get_logger()

# Qwen3.5 MoE packed modules
_PACKED_MODULES_MAPPING = {
    # Full attention: q/k/v -> qkv_proj (packed)
    "self_attn.q_proj": ("self_attn.qkv_proj", "q"),
    "self_attn.k_proj": ("self_attn.qkv_proj", "k"),
    "self_attn.v_proj": ("self_attn.qkv_proj", "v"),
    # Shared expert gate/up -> gate_up_proj (packed)
    "shared_expert.gate_proj": ("shared_expert.gate_up_proj", 0),
    "shared_expert.up_proj": ("shared_expert.gate_up_proj", 1),
}


def load_weights(
    model: nn.Module,
    weights: Generator[Tuple[str, str, torch.Tensor], None, None],
) -> None:
    """Load weights for Qwen3.5 MoE model.

    Args:
        model: The Qwen3_5MoeForConditionalGeneration model instance
        weights: Generator yielding (weight_name, raw_weight_name, tensor) tuples
    """
    config = model.config

    loaded_count = 0
    skipped_count = 0
    not_found_names: list[str] = []

    for weight_name, raw_weight_name, tensor in weights:
        # --- Expert weight handling (two checkpoint formats) ---
        if "experts." in weight_name:
            # Format 1: Per-expert weights (e.g. Qwen3.5-397B-A17B-FP8)
            # experts.{i}.gate_proj.weight / .weight_scale_inv
            if EXPERT_RE.match(weight_name):
                if load_per_expert_weight(model, weight_name, tensor, config):
                    loaded_count += 1
                    continue

            # Format 2: Packed 3D expert weights (alternate checkpoint format)
            # experts.gate_up_proj [E, 2I, H], experts.down_proj [E, H, I]
            if PACKED_EXPERT_RE.match(weight_name):
                if load_packed_expert_weight(model, weight_name, tensor):
                    loaded_count += 1
                    continue

            # Format 2: Packed 3D expert scale tensors (FP8 only)
            # experts.gate_up_proj_scale_inv, experts.down_proj_scale_inv
            if PACKED_EXPERT_SCALE_RE.match(weight_name):
                if load_packed_expert_scale(model, weight_name, tensor):
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
