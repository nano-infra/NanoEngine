"""Per-model weight loader for DeepSeek V2/V3.

Handles:
- packed_modules_mapping: gate_proj -> (gate_up_proj, 0), up_proj -> (gate_up_proj, 1)
- Per-expert weights: experts.{i}.gate_proj/up_proj/down_proj -> routed_experts 3D tensors
- kv_b_proj decomposition -> kc.weight + vc.weight (with FP8 dequant if needed)
- e_score_correction_bias direct load
- Default param.weight_loader for all standard linear layers
"""

import re
from typing import Callable, Generator, Tuple

import torch
from torch import nn

from dlengine.context_v2.distributed import get_dist_context
from dlengine.logging import get_logger
from dlengine.models.pp_utils import pp_weight_belongs_to_stage
from dlengine.worker.loader import (
    _dequant_fp8_block,
    default_weight_loader,
    EXPERT_RE,
    load_per_expert_weight,
)

logger = get_logger()

# DeepSeek packed modules: shared expert gate/up packing
_PACKED_MODULES_MAPPING = {
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}


def _handle_kv_b_proj(
    model: nn.Module,
    weight_name: str,
    tensor: torch.Tensor,
    get_tensor_by_name: dict[str, torch.Tensor],
    config,
    block_size: list[int],
) -> bool:
    """Handle kv_b_proj decomposition -> kc and vc BMM weights.

    kv_b_proj.weight shape: (num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
    Decompose into:
      kc.weight: (num_heads, qk_nope_head_dim, kv_lora_rank)
      vc.weight: (num_heads, kv_lora_rank, v_head_dim)

    Returns True if handled.
    """
    if "kv_b_proj" not in weight_name:
        return False

    # Only handle the weight, skip the scale_inv (used only for dequant)
    if "weight_scale_inv" in weight_name:
        return True  # Skip — scale is consumed when we load the weight

    if not weight_name.endswith("kv_b_proj.weight"):
        return False

    num_heads = config.num_attention_heads
    qk_nope_head_dim = config.qk_nope_head_dim
    v_head_dim = config.v_head_dim
    kv_lora_rank = config.kv_lora_rank

    weight_fp8 = tensor

    # Check if we need to dequantize
    if weight_fp8.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        scale_name = weight_name + "_scale_inv"
        scale_inv = get_tensor_by_name.get(scale_name)
        if scale_inv is None:
            logger.error(f"Cannot find {scale_name} for FP8 dequant of {weight_name}")
            raise RuntimeError(f"Missing scale tensor: {scale_name}")
        weight_bf16 = _dequant_fp8_block(weight_fp8, scale_inv, block_size)
    else:
        weight_bf16 = weight_fp8.to(torch.bfloat16)

    # Reshape: (num_heads, qk_nope + v_head, kv_lora_rank)
    weight_bf16 = weight_bf16.view(
        num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
    )

    # Split into kc (W_UK) and vc (W_UV)
    ctx = get_dist_context()
    if num_heads % ctx.attn_tp_world_size:
        raise ValueError(
            f"num_attention_heads={num_heads} is not divisible by "
            f"attention_tp={ctx.attn_tp_world_size}"
        )
    local_heads = num_heads // ctx.attn_tp_world_size
    head_start = ctx.attn_tp_rank * local_heads
    weight_bf16 = weight_bf16.narrow(0, head_start, local_heads)
    kc_weight = weight_bf16[:, :qk_nope_head_dim, :]
    vc_weight = weight_bf16[:, qk_nope_head_dim:, :].transpose(1, 2)

    # Derive parameter path
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
        f"Decomposed {weight_name} -> kc {kc_weight.shape} + vc {vc_weight.shape}"
    )
    return True


def load_weights(
    model: nn.Module,
    weights: Generator[Tuple[str, str, torch.Tensor], None, None],
) -> None:
    """Load weights for DeepSeek V2/V3 model.

    Args:
        model: The DeepseekV2ForCausalLM model instance
        weights: Generator yielding (weight_name, raw_weight_name, tensor) tuples
    """
    config = model.config

    # Determine FP8 block size from quantization config
    quant_config = getattr(model, "quantization_config", None)
    block_size = getattr(quant_config, "block_size", [128, 128])

    loaded_count = 0
    skipped_count = 0
    not_found_names: list[str] = []

    # We need to collect scale tensors for kv_b_proj FP8 dequant.
    # Strategy: buffer all kv_b_proj scale tensors as we encounter them,
    # and also buffer kv_b_proj weight tensors for deferred processing.
    kv_b_proj_weights: dict[str, torch.Tensor] = {}  # weight_name -> tensor
    kv_b_proj_scales: dict[str, torch.Tensor] = {}  # weight_name -> scale tensor

    ctx = get_dist_context()
    pp_size = ctx.pp_world_size
    start_layer = getattr(model.model, "start_layer", 0)
    end_layer = getattr(model.model, "end_layer", None)
    tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))

    for weight_name, raw_weight_name, tensor in weights:
        if pp_size > 1 and end_layer is not None:
            if not pp_weight_belongs_to_stage(weight_name, start_layer, end_layer):
                if (
                    tie_word_embeddings
                    and ctx.is_last_pp_stage
                    and "embed_tokens" in weight_name
                    and getattr(model, "lm_head", None) is not None
                ):
                    lm_head_param = model.lm_head.weight
                    loader = getattr(
                        lm_head_param, "weight_loader", default_weight_loader
                    )
                    loader(lm_head_param, tensor)
                    loaded_count += 1
                continue

        # Buffer kv_b_proj weights and scales for deferred processing
        if "kv_b_proj" in weight_name:
            if "weight_scale_inv" in weight_name:
                # Store scale, keyed by the base weight name
                base_name = weight_name.replace("_scale_inv", "")
                kv_b_proj_scales[base_name] = tensor
            elif weight_name.endswith("kv_b_proj.weight"):
                kv_b_proj_weights[weight_name] = tensor
            skipped_count += 1  # Will be counted when processed
            continue

        # Per-expert weights -> combined 3D tensors
        if "experts." in weight_name and EXPERT_RE.match(weight_name):
            if load_per_expert_weight(model, weight_name, tensor, config):
                loaded_count += 1
                continue

        # packed_modules_mapping: gate_proj -> gate_up_proj, up_proj -> gate_up_proj
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

    # Process buffered kv_b_proj weights
    for weight_name, tensor in kv_b_proj_weights.items():
        # Build a dict for scale lookup: _handle_kv_b_proj looks up
        # "{weight_name}_scale_inv", and kv_b_proj_scales is keyed by
        # base_name (== weight_name), so we re-key for the lookup.
        get_tensor_by_name = {}
        scale_key = weight_name + "_scale_inv"
        if weight_name in kv_b_proj_scales:
            get_tensor_by_name[scale_key] = kv_b_proj_scales[weight_name]

        if _handle_kv_b_proj(
            model, weight_name, tensor, get_tensor_by_name, config, block_size
        ):
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
