"""Weight loader for DeepSeek V2/V3 MTP layers.

Handles:
- Weight name remapping: spec-layer-specific vs transformer sub-weights
- Stacked params: gate_up_proj packing
- kv_b_proj decomposition into kc + vc BMMs
- Per-expert weights for MoE in MTP blocks
- Shared embedding: only loaded for the first MTP layer

Reference: vLLM deepseek_mtp.py _rewrite_spec_layer_name()
"""

import re
from typing import Generator, Tuple

import torch
from torch import nn

from nanodeploy.logging import get_logger
from nanodeploy.worker.loader import (
    _dequant_fp8_block,
    default_weight_loader,
    EXPERT_RE,
    load_per_expert_weight,
)

logger = get_logger()

# Stacked parameter mapping for DeepSeek MTP
_PACKED_MODULES_MAPPING = {
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}

# Weight names that belong to MTP-specific modules (NOT the transformer block)
_SPEC_LAYER_WEIGHT_NAMES = ("embed_tokens", "enorm", "hnorm", "eh_proj", "shared_head")

# Shared across all MTP layers (only load for the first layer)
_SHARED_WEIGHT_NAMES = ("embed_tokens",)


def _get_spec_layer_idx(config, name: str) -> int | None:
    """Extract MTP layer index from weight name, or None if not an MTP weight."""
    num_hidden = config.num_hidden_layers
    num_mtp = config.num_nextn_predict_layers
    m = re.search(r"model\.layers\.(\d+)\.", name)
    if m:
        idx = int(m.group(1))
        if num_hidden <= idx < num_hidden + num_mtp:
            return idx
    return None


def _rewrite_spec_layer_name(spec_layer: int, name: str) -> str:
    """Rewrite checkpoint weight name to match DeepSeekMTP parameter layout.

    MTP-specific submodules (enorm, hnorm, eh_proj, shared_head) stay on the
    MTP layer directly.  All other submodules (attention, mlp, norms used by
    the transformer block) get a `.mtp_block` prefix.
    Shared weights (embed_tokens) are moved to top-level model.
    """
    is_spec_weight = False
    is_shared = False
    for wn in _SPEC_LAYER_WEIGHT_NAMES:
        if wn in name:
            is_spec_weight = True
            if wn in _SHARED_WEIGHT_NAMES:
                is_shared = True
            break

    # DeepSeekMTP is flat (no .model sub-module): parameters are
    # embed_tokens.weight, layers.{N}.enorm.weight, layers.{N}.mtp_block.*, etc.
    if not is_spec_weight:
        # Transformer sub-weight: add .mtp_block prefix, strip model. prefix
        name = name.replace(
            f"model.layers.{spec_layer}.",
            f"layers.{spec_layer}.mtp_block.",
        )
    elif is_shared:
        # Shared weight (embed_tokens): promote to top-level, strip model. prefix
        name = name.replace(f"model.layers.{spec_layer}.", "")
    else:
        # MTP-specific weight (enorm, hnorm, eh_proj, shared_head): strip model. prefix
        name = name.replace(f"model.layers.{spec_layer}.", f"layers.{spec_layer}.")

    return name


def _handle_kv_b_proj(
    model: nn.Module,
    weight_name: str,
    tensor: torch.Tensor,
    kv_b_proj_scales: dict[str, torch.Tensor],
    config,
    block_size: list[int],
) -> bool:
    """Decompose kv_b_proj into kc and vc BMM weights for MTP layers."""
    if "kv_b_proj" not in weight_name:
        return False
    if "weight_scale_inv" in weight_name:
        return True  # scale consumed when we load the weight
    if not weight_name.endswith("kv_b_proj.weight"):
        return False

    num_heads = config.num_attention_heads
    qk_nope_head_dim = config.qk_nope_head_dim
    v_head_dim = config.v_head_dim
    kv_lora_rank = config.kv_lora_rank

    weight_fp8 = tensor
    if weight_fp8.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        scale_key = weight_name + "_scale_inv"
        base_name = weight_name
        scale_inv = kv_b_proj_scales.get(base_name)
        if scale_inv is None:
            raise RuntimeError(f"Missing scale tensor: {scale_key}")
        weight_bf16 = _dequant_fp8_block(weight_fp8, scale_inv, block_size)
    else:
        weight_bf16 = weight_fp8.to(torch.bfloat16)

    weight_bf16 = weight_bf16.view(
        num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
    )
    kc_weight = weight_bf16[:, :qk_nope_head_dim, :]
    vc_weight = weight_bf16[:, qk_nope_head_dim:, :].transpose(1, 2)

    prefix = weight_name.replace("kv_b_proj.weight", "")
    kc_param_name = f"{prefix}kc.weight"
    vc_param_name = f"{prefix}vc.weight"

    try:
        kc_param = model.get_parameter(kc_param_name)
        kc_param.data.copy_(kc_weight)
    except (AttributeError, RuntimeError) as e:
        logger.error(f"Failed to load kc weight from {weight_name}: {e}")
        raise

    try:
        vc_param = model.get_parameter(vc_param_name)
        vc_param.data.copy_(vc_weight)
    except (AttributeError, RuntimeError) as e:
        logger.error(f"Failed to load vc weight from {weight_name}: {e}")
        raise

    logger.debug(
        f"MTP: Decomposed {weight_name} -> kc {kc_weight.shape} + vc {vc_weight.shape}"
    )
    return True


def load_weights(
    model: nn.Module,
    weights: Generator[Tuple[str, str, torch.Tensor], None, None],
) -> None:
    """Load weights for DeepSeek MTP model.

    Args:
        model: The DeepSeekMTP model instance
        weights: Generator yielding (weight_name, raw_weight_name, tensor) tuples
    """
    config = model.config
    quant_config = getattr(model, "quantization_config", None)
    block_size = getattr(quant_config, "block_size", [128, 128])

    mtp_start = model.mtp_start_layer_idx

    loaded_count = 0
    skipped_count = 0
    not_found_names: list[str] = []

    # Buffer kv_b_proj weights/scales for deferred processing
    kv_b_proj_weights: dict[str, torch.Tensor] = {}
    kv_b_proj_scales: dict[str, torch.Tensor] = {}

    for weight_name, raw_weight_name, tensor in weights:
        # Only process MTP layer weights
        spec_layer = _get_spec_layer_idx(config, weight_name)
        if spec_layer is None:
            skipped_count += 1
            continue

        # Shared weights: only load for the first MTP layer
        is_shared = any(sn in weight_name for sn in _SHARED_WEIGHT_NAMES)
        if is_shared and spec_layer != mtp_start:
            skipped_count += 1
            continue

        # Rewrite name to match model parameter layout
        name = _rewrite_spec_layer_name(spec_layer, weight_name)

        # Buffer kv_b_proj for deferred processing
        if "kv_b_proj" in name:
            if "weight_scale_inv" in name:
                base = name.replace("_scale_inv", "")
                kv_b_proj_scales[base] = tensor
            elif name.endswith("kv_b_proj.weight"):
                kv_b_proj_weights[name] = tensor
            skipped_count += 1
            continue

        # Per-expert weights
        if "experts." in name and EXPERT_RE.match(name):
            if load_per_expert_weight(model, name, tensor, config):
                loaded_count += 1
                continue

        # Packed modules (gate_up_proj)
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

    # Process buffered kv_b_proj weights
    for wname, tensor in kv_b_proj_weights.items():
        if _handle_kv_b_proj(
            model, wname, tensor, kv_b_proj_scales, config, block_size
        ):
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
