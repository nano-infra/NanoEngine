import os
import re
from glob import glob

import torch
import torch.distributed as dist
from nanodeploy.context.distributed import get_dist_context
from nanodeploy.logging import get_logger
from safetensors import safe_open
from torch import nn
from tqdm import tqdm

logger = get_logger()

# Weight name patterns for MTP / next-token prediction layers (skip these)
_MTP_PATTERNS = ("eh_proj", "enorm", "hnorm", "shared_head")

# Regex to parse layer index from weight name
_LAYER_RE = re.compile(r"layers\.(\d+)\.")

# Regex to parse expert index from weight name
# e.g. "model.layers.3.mlp.experts.5.gate_proj.weight" -> expert_idx=5
_EXPERT_RE = re.compile(r"(.+\.mlp)\.experts\.(\d+)\.(\w+)\.(weight(?:_scale_inv)?)")


def default_weight_loader(param, tensor):
    """Default weight loader (copy directly)."""
    param.data.copy_(tensor)


def _dequant_fp8_block(
    weight_fp8: torch.Tensor, scale_inv: torch.Tensor, block_size: list[int]
) -> torch.Tensor:
    """Dequantize FP8 block-quantized weight to bfloat16.

    Args:
        weight_fp8: (N, K) in float8_e4m3fn
        scale_inv: (ceil(N/bn), ceil(K/bk)) in float32
        block_size: [bn, bk]

    Returns:
        weight in bfloat16 of shape (N, K)
    """
    N, K = weight_fp8.shape
    bn, bk = block_size
    N_blocks = (N + bn - 1) // bn
    K_blocks = (K + bk - 1) // bk

    # Pad weight to full block grid
    weight_padded = torch.zeros(
        N_blocks * bn, K_blocks * bk, dtype=weight_fp8.dtype, device=weight_fp8.device
    )
    weight_padded[:N, :K] = weight_fp8
    # Reshape: (N_blocks, bn, K_blocks, bk)
    weight_padded = weight_padded.view(N_blocks, bn, K_blocks, bk)
    # scale_inv: (N_blocks, K_blocks) -> (N_blocks, 1, K_blocks, 1)
    scale = scale_inv.unsqueeze(1).unsqueeze(3)
    weight_dequant = weight_padded.to(torch.bfloat16) * scale.to(torch.bfloat16)
    weight_dequant = weight_dequant.reshape(N_blocks * bn, K_blocks * bk)[:N, :K]
    return weight_dequant


def _is_mtp_weight(weight_name: str, num_hidden_layers: int | None = None) -> bool:
    """Check if weight belongs to MTP (multi-token prediction) layers."""
    for pat in _MTP_PATTERNS:
        if pat in weight_name:
            return True
    # Per-layer embed_tokens (MTP layer)
    if re.search(r"layers\.\d+\.embed_tokens\.", weight_name):
        return True
    # Layers beyond num_hidden_layers are MTP prediction layers
    if num_hidden_layers is not None:
        m = _LAYER_RE.search(weight_name)
        if m and int(m.group(1)) >= num_hidden_layers:
            return True
    return False


def _handle_expert_weight(
    model: nn.Module, weight_name: str, tensor: torch.Tensor, config
) -> bool:
    """Handle per-expert weight -> combined 3D tensor.

    Maps:
      experts.{i}.gate_proj.weight      -> gate_up_proj[local, :intermediate, :]
      experts.{i}.up_proj.weight        -> gate_up_proj[local, intermediate:, :]
      experts.{i}.down_proj.weight      -> down_proj[local, :, :]
      experts.{i}.gate_proj.weight_scale_inv -> gate_up_scale_inv[local, :scale_rows, :]
      experts.{i}.up_proj.weight_scale_inv   -> gate_up_scale_inv[local, scale_rows:, :]
      experts.{i}.down_proj.weight_scale_inv -> down_scale_inv[local, :, :]

    Returns True if handled, False otherwise.
    """
    m = _EXPERT_RE.match(weight_name)
    if m is None:
        return False

    mlp_prefix = m.group(1)  # e.g. "model.layers.3.mlp"
    expert_idx = int(m.group(2))  # e.g. 5
    proj_name = m.group(3)  # e.g. "gate_proj", "up_proj", "down_proj"
    suffix = m.group(4)  # "weight" or "weight_scale_inv"

    # Determine EP rank and which experts belong to this rank
    ep_world_size = get_dist_context().ffn_ep_world_size
    ep_group = get_dist_context().ffn_ep_group
    ep_rank = dist.get_rank(group=ep_group) if ep_group is not None else 0
    num_experts = config.n_routed_experts
    experts_per_rank = num_experts // ep_world_size
    expert_start = ep_rank * experts_per_rank
    expert_end = expert_start + experts_per_rank

    # Skip experts not on this rank
    if expert_idx < expert_start or expert_idx >= expert_end:
        return True  # Handled (skipped)

    local_idx = expert_idx - expert_start
    is_scale = "scale_inv" in suffix

    if proj_name in ("gate_proj", "up_proj"):
        # Target: gate_up_proj or gate_up_scale_inv
        param_suffix = "gate_up_scale_inv" if is_scale else "gate_up_proj"
        param_name = f"{mlp_prefix}.{param_suffix}"
        try:
            param = model.get_parameter(param_name)
        except AttributeError:
            logger.warning(f"Parameter {param_name} not found, skipping {weight_name}")
            return True

        # Determine shard offset
        # gate_up_proj shape: (experts_per_rank, intermediate*2[/bs], hidden[/bs])
        # gate_proj is first half, up_proj is second half along dim 1
        total_dim1 = param.data.shape[1]
        half = total_dim1 // 2
        if proj_name == "gate_proj":
            start, end = 0, half
        else:  # up_proj
            start, end = half, total_dim1

        if is_scale:
            tensor = tensor.to(torch.float32)
        param.data[local_idx, start:end, :].copy_(tensor)

    elif proj_name == "down_proj":
        param_suffix = "down_scale_inv" if is_scale else "down_proj"
        param_name = f"{mlp_prefix}.{param_suffix}"
        try:
            param = model.get_parameter(param_name)
        except AttributeError:
            logger.warning(f"Parameter {param_name} not found, skipping {weight_name}")
            return True
        if is_scale:
            tensor = tensor.to(torch.float32)
        param.data[local_idx].copy_(tensor)

    else:
        logger.warning(f"Unknown expert proj {proj_name} in {weight_name}")
        return False

    return True


def _handle_kv_b_proj(
    model: nn.Module,
    weight_name: str,
    tensor: torch.Tensor,
    safe_file,
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

    # Load the FP8 weight
    weight_fp8 = tensor  # (num_heads * (qk_nope + v_head), kv_lora_rank)

    # Check if we need to dequantize
    if weight_fp8.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        # Load the corresponding scale_inv
        scale_name = weight_name + "_scale_inv"
        try:
            scale_inv = safe_file.get_tensor(scale_name)
        except Exception:
            logger.error(f"Cannot find {scale_name} for FP8 dequant of {weight_name}")
            raise
        weight_bf16 = _dequant_fp8_block(weight_fp8, scale_inv, block_size)
    else:
        weight_bf16 = weight_fp8.to(torch.bfloat16)

    # Reshape: (num_heads, qk_nope + v_head, kv_lora_rank)
    weight_bf16 = weight_bf16.view(
        num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
    )

    # Split into kc (W_UK) and vc (W_UV)
    kc_weight = weight_bf16[
        :, :qk_nope_head_dim, :
    ]  # (num_heads, qk_nope, kv_lora_rank)
    vc_weight = weight_bf16[:, qk_nope_head_dim:, :].transpose(
        1, 2
    )  # (num_heads, kv_lora_rank, v_head_dim)

    # Derive parameter path: replace "kv_b_proj.weight" with "kc.weight" / "vc.weight"
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


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    config = getattr(model, "config", None)

    # Determine FP8 block size from quantization config
    quant_config = getattr(model, "quantization_config", None)
    block_size = getattr(quant_config, "block_size", [128, 128])

    num_hidden_layers = getattr(config, "num_hidden_layers", None) if config else None

    # Collect all weight files
    weight_files = sorted(glob(os.path.join(path, "*.safetensors")))
    pbar = tqdm(weight_files, desc="Loading weights", unit="files")

    skipped_count = 0
    loaded_count = 0

    try:
        for file in pbar:
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    # 1. Skip MTP / nextn prediction weights
                    if _is_mtp_weight(weight_name, num_hidden_layers):
                        skipped_count += 1
                        continue

                    # 2. Handle per-expert weights -> combined 3D tensors
                    if "experts." in weight_name and config is not None:
                        tensor = f.get_tensor(weight_name)
                        if _handle_expert_weight(model, weight_name, tensor, config):
                            loaded_count += 1
                            continue

                    # 3. Handle kv_b_proj decomposition -> kc/vc
                    if "kv_b_proj" in weight_name and config is not None:
                        tensor = f.get_tensor(weight_name)
                        if _handle_kv_b_proj(
                            model, weight_name, tensor, f, config, block_size
                        ):
                            loaded_count += 1
                            continue

                    # 4. Handle packed_modules_mapping (gate_proj->gate_up_proj, etc.)
                    matched = False
                    for k in packed_modules_mapping:
                        if k in weight_name:
                            v, shard_id = packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            try:
                                param = model.get_parameter(param_name)
                            except AttributeError:
                                logger.warning(
                                    f"Packed param {param_name} not found for {weight_name}"
                                )
                                matched = True
                                break
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(
                                param, f.get_tensor(weight_name), shard_id, weight_name
                            )
                            matched = True
                            loaded_count += 1
                            break

                    if matched:
                        continue

                    # 5. Default: direct parameter loading
                    try:
                        param = model.get_parameter(weight_name)
                    except AttributeError:
                        logger.warning(
                            f"Parameter {weight_name} not found in model, skipping"
                        )
                        skipped_count += 1
                        continue
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, f.get_tensor(weight_name))
                    loaded_count += 1
    finally:
        pbar.close()

    logger.info(
        f"Weight loading complete: {loaded_count} loaded, {skipped_count} skipped"
    )
