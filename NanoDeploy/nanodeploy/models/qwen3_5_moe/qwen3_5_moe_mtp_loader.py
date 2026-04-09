"""Weight loader for Qwen3.5-MoE MTP layers.

Handles:
- Weight name remapping: mtp.* → model.* prefix
- embed_tokens / lm_head from model.language_model.* prefix
- Packed QKV: q_proj/k_proj/v_proj → qkv_proj
- Packed gate_up: shared_expert.gate_proj/up_proj → gate_up_proj
- Per-expert weights and packed 3D expert weights
- Packed 3D expert scales (FP8)

Reference: vLLM qwen3_5_mtp.py load_weights / remap_weight_names
"""

import re
from typing import Generator, Tuple

import torch
from torch import nn

from nanodeploy.logging import get_logger
from nanodeploy.worker.loader import (
    default_weight_loader,
    EXPERT_RE,
    load_packed_expert_scale,
    load_packed_expert_weight,
    load_per_expert_weight,
    PACKED_EXPERT_RE,
    PACKED_EXPERT_SCALE_RE,
)

logger = get_logger()

# Qwen3.5 MoE MTP packed modules
_PACKED_MODULES_MAPPING = {
    # Full attention: q/k/v → qkv_proj (packed)
    "self_attn.q_proj": ("self_attn.qkv_proj", "q"),
    "self_attn.k_proj": ("self_attn.qkv_proj", "k"),
    "self_attn.v_proj": ("self_attn.qkv_proj", "v"),
    # Shared expert gate/up → gate_up_proj (packed)
    "shared_expert.gate_proj": ("shared_expert.gate_up_proj", 0),
    "shared_expert.up_proj": ("shared_expert.gate_up_proj", 1),
}


def _remap_weight_name(name: str) -> str | None:
    """Remap checkpoint weight name for MTP model.

    Returns remapped name, or None if weight should be skipped.

    Qwen3_5MTP is a flat module (no .model sub-module), so parameters are
    like fc.weight, layers.0.*, norm.weight — NOT model.fc.weight.

    Checkpoint formats:
      - mtp.fc.weight → fc.weight
      - mtp.layers.0.* → layers.0.*
      - mtp.pre_fc_norm_embedding.weight → pre_fc_norm_embedding.weight
      - mtp.pre_fc_norm_hidden.weight → pre_fc_norm_hidden.weight
      - mtp.norm.weight → norm.weight
      - model.language_model.embed_tokens.* → embed_tokens.*
      - model.embed_tokens.* → embed_tokens.*
      - lm_head.* → lm_head.*
    """
    if name.startswith("mtp."):
        # mtp.fc.weight → fc.weight
        return name[len("mtp.") :]

    if "embed_tokens" in name:
        # model.language_model.embed_tokens.weight → embed_tokens.weight
        # model.embed_tokens.weight → embed_tokens.weight
        name = re.sub(r"^model\.language_model\.", "", name)
        name = re.sub(r"^model\.", "", name)
        return name

    if "lm_head" in name:
        # model.language_model.lm_head.weight → lm_head.weight
        name = re.sub(r"^model\.language_model\.", "", name)
        name = re.sub(r"^model\.", "", name)
        return name

    # Not an MTP weight
    return None


def load_weights(
    model: nn.Module,
    weights: Generator[Tuple[str, str, torch.Tensor], None, None],
) -> None:
    """Load weights for Qwen3.5-MoE MTP model.

    Args:
        model: The Qwen3_5MTP model instance
        weights: Generator yielding (weight_name, raw_weight_name, tensor) tuples
    """
    config = model.config

    loaded_count = 0
    skipped_count = 0
    not_found_names: list[str] = []

    for weight_name, raw_weight_name, tensor in weights:
        # Remap weight name for MTP
        name = _remap_weight_name(weight_name)
        if name is None:
            skipped_count += 1
            continue

        # --- Expert weight handling ---
        if "experts." in name:
            # Format 1: Per-expert weights
            if EXPERT_RE.match(name):
                if load_per_expert_weight(model, name, tensor, config):
                    loaded_count += 1
                    continue

            # Format 2: Packed 3D expert weights
            if PACKED_EXPERT_RE.match(name):
                if load_packed_expert_weight(model, name, tensor):
                    loaded_count += 1
                    continue

            # Format 2: Packed 3D expert scales (FP8)
            if PACKED_EXPERT_SCALE_RE.match(name):
                if load_packed_expert_scale(model, name, tensor):
                    loaded_count += 1
                    continue

        # Packed modules mapping
        matched = False
        for k, (v, shard_id) in _PACKED_MODULES_MAPPING.items():
            if k in name:
                param_name = name.replace(k, v)
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    continue
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, tensor, shard_id, name)
                matched = True
                loaded_count += 1
                break

        if matched:
            continue

        # Default: direct parameter loading
        try:
            param = model.get_parameter(name)
        except AttributeError:
            not_found_names.append(name)
            skipped_count += 1
            continue

        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, tensor)
        loaded_count += 1

    logger.warning(
        f"MTP weight loading: {loaded_count} loaded, {skipped_count} skipped"
    )
    if not_found_names:
        unique_patterns = set()
        for n in not_found_names:
            pat = re.sub(r"layers\.\d+\.", "layers.N.", n)
            pat = re.sub(r"experts\.\d+\.", "experts.E.", pat)
            unique_patterns.add(pat)
        logger.warning(
            f"  MTP: {len(not_found_names)} weights NOT FOUND "
            f"(unique patterns: {sorted(unique_patterns)})"
        )
