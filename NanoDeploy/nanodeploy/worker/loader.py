import os
import re
from glob import glob
from typing import Callable, Generator, Tuple

import torch
import torch.distributed as dist
from nanodeploy.context.distributed import get_dist_context
from nanodeploy.logging import get_logger
from safetensors import safe_open
from torch import nn
from tqdm import tqdm

logger = get_logger()

# Weight name patterns for MTP / next-token prediction layers (skip these)
_MTP_PATTERNS = ("eh_proj", "enorm", "hnorm", "shared_head", "mtp.")

# Weight name patterns to always skip (e.g., vision module, rotary cache)
_SKIP_PATTERNS = (
    "visual.",
    "rotary_emb.inv_freq",
    "rotary_emb.cos_cached",
    "rotary_emb.sin_cached",
)

# Regex to parse layer index from weight name
_LAYER_RE = re.compile(r"layers\.(\d+)\.")

# Regex to parse expert index from weight name
# e.g. "model.layers.3.mlp.experts.5.gate_proj.weight" -> expert_idx=5
EXPERT_RE = re.compile(r"(.+\.mlp)\.experts\.(\d+)\.(\w+)\.(weight(?:_scale_inv)?)")

# Regex for already-packed 3D expert weights (no per-expert index)
# e.g. "model.layers.0.mlp.experts.gate_up_proj" -> mlp_prefix, proj_name
PACKED_EXPERT_RE = re.compile(r"(.+\.mlp)\.experts\.(gate_up_proj|down_proj)$")

# Regex for already-packed 3D expert scale tensors
# e.g. "model.layers.0.mlp.experts.gate_up_proj_scale_inv" -> mlp_prefix, proj_name
PACKED_EXPERT_SCALE_RE = re.compile(
    r"(.+\.mlp)\.experts\.(gate_up_proj|down_proj)_scale_inv$"
)


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


def _should_skip_weight(weight_name: str, num_hidden_layers: int | None = None) -> bool:
    """Check if weight should be skipped (MTP, vision, rotary cache, etc.)."""
    for pat in _MTP_PATTERNS:
        if pat in weight_name:
            return True
    for pat in _SKIP_PATTERNS:
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


def _is_mtp_weight(weight_name: str, num_hidden_layers: int | None = None) -> bool:
    """Check if weight belongs to MTP layers (inverse of base-model skip logic)."""
    for pat in _MTP_PATTERNS:
        if pat in weight_name:
            return True
    if re.search(r"layers\.\d+\.embed_tokens\.", weight_name):
        return True
    if num_hidden_layers is not None:
        m = _LAYER_RE.search(weight_name)
        if m and int(m.group(1)) >= num_hidden_layers:
            return True
    return False


def _strip_vlm_prefix(weight_name: str) -> str:
    """Strip VLM prefix (e.g. 'model.language_model.' -> 'model.') for text models."""
    if weight_name.startswith("model.language_model."):
        return "model." + weight_name[len("model.language_model.") :]
    return weight_name


def iterate_mtp_weights(
    path: str, num_hidden_layers: int | None = None
) -> Generator[Tuple[str, str, torch.Tensor], None, None]:
    """Iterate over safetensors, yielding ONLY MTP-layer weights.

    This is the inverse of iterate_weights() which skips MTP weights.
    Also includes embed_tokens and lm_head weights needed for MTP.
    """
    weight_files = sorted(glob(os.path.join(path, "*.safetensors")))
    pbar = tqdm(weight_files, desc="Loading MTP weights", unit="files")

    try:
        for file in pbar:
            with safe_open(file, "pt", "cpu") as f:
                for raw_weight_name in f.keys():
                    # Skip non-MTP weights (vision, rotary cache)
                    for pat in _SKIP_PATTERNS:
                        if pat in raw_weight_name:
                            break
                    else:
                        # Include MTP weights + embed_tokens/lm_head for sharing
                        is_mtp = _is_mtp_weight(raw_weight_name, num_hidden_layers)
                        is_embed_or_head = (
                            "embed_tokens" in raw_weight_name
                            or "lm_head" in raw_weight_name
                        )
                        if is_mtp or is_embed_or_head:
                            weight_name = _strip_vlm_prefix(raw_weight_name)
                            yield weight_name, raw_weight_name, f.get_tensor(
                                raw_weight_name
                            )
    finally:
        pbar.close()


def load_mtp_model(model: nn.Module, path: str) -> None:
    """Load MTP model weights from safetensors files.

    The MTP model must implement load_weights().
    """
    config = getattr(model, "config", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None) if config else None

    if not hasattr(model, "load_weights"):
        raise RuntimeError("MTP model must implement load_weights()")

    weights = iterate_mtp_weights(path, num_hidden_layers)
    model.load_weights(weights)


def iterate_weights(
    path: str, num_hidden_layers: int | None = None
) -> Generator[Tuple[str, str, Callable[[], torch.Tensor]], None, None]:
    """Iterate over safetensors weight files, yielding (weight_name, raw_weight_name, get_tensor_fn).

    Applies universal skip/strip logic:
    - Skips MTP, vision, rotary cache weights
    - Strips VLM prefix (model.language_model. -> model.)
    - Uses lazy get_tensor_fn to avoid loading tensors that the model will skip

    Yields:
        (weight_name, raw_weight_name, get_tensor_fn) tuples where:
        - weight_name: cleaned name (VLM prefix stripped)
        - raw_weight_name: original name in safetensors file
        - get_tensor_fn: callable that returns the tensor when called
    """
    weight_files = sorted(glob(os.path.join(path, "*.safetensors")))
    pbar = tqdm(weight_files, desc="Loading weights", unit="files")

    try:
        for file in pbar:
            with safe_open(file, "pt", "cpu") as f:
                for raw_weight_name in f.keys():
                    # Skip MTP / vision / rotary cache weights
                    if _should_skip_weight(raw_weight_name, num_hidden_layers):
                        continue

                    # Strip VLM prefix
                    weight_name = _strip_vlm_prefix(raw_weight_name)

                    # Yield with a lazy tensor loader bound to this file handle
                    # We need to load the tensor eagerly since the file handle
                    # closes when we leave the `with` block
                    yield weight_name, raw_weight_name, f.get_tensor(raw_weight_name)
    finally:
        pbar.close()


def load_model(model: nn.Module, path: str):
    """Load model weights from safetensors files.

    If the model implements `load_weights()`, delegates to it.
    Otherwise falls back to generic loading logic for backwards compatibility.
    """
    config = getattr(model, "config", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None) if config else None

    if hasattr(model, "load_weights"):
        # Per-model loader: model handles its own weight mapping
        weights = iterate_weights(path, num_hidden_layers)
        model.load_weights(weights)
        return

    # --- Fallback: generic loading for models without load_weights() ---
    _load_model_generic(model, path, num_hidden_layers)


def _load_model_generic(model: nn.Module, path: str, num_hidden_layers: int | None):
    """Generic weight loader (fallback for models without load_weights())."""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    config = getattr(model, "config", None)

    # Determine FP8 block size from quantization config
    quant_config = getattr(model, "quantization_config", None)
    block_size = getattr(quant_config, "block_size", [128, 128])

    # Collect all weight files
    weight_files = sorted(glob(os.path.join(path, "*.safetensors")))
    pbar = tqdm(weight_files, desc="Loading weights", unit="files")

    skipped_count = 0
    loaded_count = 0
    not_found_names: list[str] = []

    try:
        for file in pbar:
            with safe_open(file, "pt", "cpu") as f:
                for raw_weight_name in f.keys():
                    weight_name = _strip_vlm_prefix(raw_weight_name)

                    if _should_skip_weight(raw_weight_name, num_hidden_layers):
                        skipped_count += 1
                        continue

                    # Handle packed_modules_mapping
                    matched = False
                    for k in packed_modules_mapping:
                        if k in weight_name:
                            v, shard_id = packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            try:
                                param = model.get_parameter(param_name)
                            except AttributeError:
                                continue
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(
                                param,
                                f.get_tensor(raw_weight_name),
                                shard_id,
                                weight_name,
                            )
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
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, f.get_tensor(raw_weight_name))
                    loaded_count += 1
    finally:
        pbar.close()

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


# --- Shared utility functions for per-model loaders ---


def load_per_expert_weight(
    model: nn.Module,
    weight_name: str,
    tensor: torch.Tensor,
    config,
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
    m = EXPERT_RE.match(weight_name)
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
    num_experts = getattr(config, "n_routed_experts", None) or getattr(
        config, "num_experts", None
    )
    if num_experts is None:
        logger.warning(
            f"Cannot determine num_experts from config, skipping {weight_name}"
        )
        return False
    experts_per_rank = num_experts // ep_world_size
    expert_start = ep_rank * experts_per_rank
    expert_end = expert_start + experts_per_rank

    # Skip experts not on this rank
    if expert_idx < expert_start or expert_idx >= expert_end:
        return True  # Handled (skipped)

    local_idx = expert_idx - expert_start
    is_scale = "scale_inv" in suffix

    tp_world_size = get_dist_context().ffn_tp_world_size
    tp_rank = get_dist_context().ffn_tp_rank

    if tp_world_size > 1:
        if proj_name in ("gate_proj", "up_proj"):
            chunk = tensor.shape[0] // tp_world_size
            tensor = tensor[tp_rank * chunk : (tp_rank + 1) * chunk]
        elif proj_name == "down_proj":
            if tensor.dim() >= 2:
                chunk = tensor.shape[1] // tp_world_size
                tensor = tensor[:, tp_rank * chunk : (tp_rank + 1) * chunk]

    if proj_name in ("gate_proj", "up_proj"):
        # Target: gate_up_proj or gate_up_scale_inv
        param_suffix = "gate_up_scale_inv" if is_scale else "gate_up_proj"
        param_name = f"{mlp_prefix}.routed_experts.{param_suffix}"
        try:
            param = model.get_parameter(param_name)
        except AttributeError:
            logger.warning(f"Parameter {param_name} not found, skipping {weight_name}")
            return True

        if is_scale:
            tensor = tensor.to(torch.float32)
            if param.data.numel() == 0:
                combined_shape = (
                    experts_per_rank,
                    tensor.shape[0] * 2,
                    *tensor.shape[1:],
                )
                param.data = torch.zeros(
                    combined_shape, dtype=torch.float32, device=param.data.device
                )

        # gate_up_proj shape: (experts_per_rank, intermediate*2[/bs], hidden[/bs])
        # gate_proj is first half, up_proj is second half along dim 1
        total_dim1 = param.data.shape[1]
        half = total_dim1 // 2
        if proj_name == "gate_proj":
            start, end = 0, half
        else:  # up_proj
            start, end = half, total_dim1

        if is_scale:
            if param.data.dim() == 2:
                param.data[local_idx, start:end].copy_(tensor.view(-1))
            elif param.data.dim() == 3:
                param.data[local_idx, start:end, :].copy_(tensor)
            else:
                param.data[local_idx, start:end, :].copy_(tensor)
            return True

        param.data[local_idx, start:end, :].copy_(tensor)

    elif proj_name == "down_proj":
        param_suffix = "down_scale_inv" if is_scale else "down_proj"
        param_name = f"{mlp_prefix}.routed_experts.{param_suffix}"
        try:
            param = model.get_parameter(param_name)
        except AttributeError:
            logger.warning(f"Parameter {param_name} not found, skipping {weight_name}")
            return True
        if is_scale:
            tensor = tensor.to(torch.float32)
            if param.data.numel() == 0:
                param.data = torch.zeros(
                    (experts_per_rank, *tensor.shape),
                    dtype=torch.float32,
                    device=param.data.device,
                )

            if param.data.dim() == 2:
                param.data[local_idx].copy_(tensor.view(-1))
            else:
                param.data[local_idx].copy_(tensor)
        else:
            param.data[local_idx].copy_(tensor)

    else:
        logger.warning(f"Unknown expert proj {proj_name} in {weight_name}")
        return False

    return True


def load_packed_expert_weight(
    model: nn.Module, weight_name: str, tensor: torch.Tensor
) -> bool:
    """Handle already-packed 3D expert weight (Qwen3.5 format).

    Checkpoint stores experts as combined tensors:
      experts.gate_up_proj  shape (num_experts, 2*intermediate, hidden)
      experts.down_proj     shape (num_experts, hidden, intermediate)

    These map to mlp.routed_experts.gate_up_proj / mlp.routed_experts.down_proj,
    with EP slicing (dim 0) and TP slicing (intermediate dim) applied.
    """
    m = PACKED_EXPERT_RE.match(weight_name)
    if m is None:
        return False

    mlp_prefix = m.group(1)  # e.g. "model.layers.0.mlp"
    proj_name = m.group(2)  # "gate_up_proj" or "down_proj"
    param_name = f"{mlp_prefix}.routed_experts.{proj_name}"

    try:
        param = model.get_parameter(param_name)
    except AttributeError:
        logger.warning(
            f"Parameter {param_name} not found for packed expert {weight_name}"
        )
        return False

    # --- EP slicing: select only this rank's local experts along dim 0 ---
    dist_ctx = get_dist_context()
    ep_world_size = dist_ctx.ffn_ep_world_size
    ep_rank = dist_ctx.ffn_ep_rank
    num_total_experts = tensor.shape[0]
    experts_per_rank = num_total_experts // ep_world_size
    expert_start = ep_rank * experts_per_rank
    expert_end = expert_start + experts_per_rank
    tensor = tensor[expert_start:expert_end]

    # --- TP slicing: slice the intermediate dimension ---
    tp_world_size = dist_ctx.ffn_tp_world_size
    tp_rank = dist_ctx.ffn_tp_rank
    if tp_world_size > 1:
        if proj_name == "gate_up_proj":
            # Shape: (local_experts, intermediate*2, hidden)
            # gate is first half of dim1, up is second half
            full_inter2 = tensor.shape[1]
            full_inter = full_inter2 // 2
            chunk = full_inter // tp_world_size
            gate_slice = tensor[:, tp_rank * chunk : (tp_rank + 1) * chunk, :]
            up_slice = tensor[
                :, full_inter + tp_rank * chunk : full_inter + (tp_rank + 1) * chunk, :
            ]
            tensor = torch.cat([gate_slice, up_slice], dim=1)
        elif proj_name == "down_proj":
            # Shape: (local_experts, hidden, intermediate)
            full_inter = tensor.shape[2]
            chunk = full_inter // tp_world_size
            tensor = tensor[:, :, tp_rank * chunk : (tp_rank + 1) * chunk]

    if param.data.shape != tensor.shape:
        logger.warning(
            f"Shape mismatch for {param_name}: param={list(param.data.shape)} "
            f"checkpoint={list(tensor.shape)} (after EP/TP slicing)"
        )

    param.data.copy_(tensor)
    logger.debug(
        f"Loaded packed expert weight: {weight_name} -> {param_name} "
        f"shape={list(tensor.shape)} (ep_rank={ep_rank}/{ep_world_size}, "
        f"tp_rank={tp_rank}/{tp_world_size})"
    )
    return True


def load_packed_expert_scale(
    model: nn.Module, weight_name: str, tensor: torch.Tensor
) -> bool:
    """Handle already-packed 3D expert scale tensors (Qwen3.5 FP8 format).

    Checkpoint stores expert scales as combined tensors:
      experts.gate_up_proj_scale_inv  shape (num_experts, 2*intermediate//bs, hidden//bs)
      experts.down_proj_scale_inv     shape (num_experts, hidden//bs, intermediate//bs)

    These map to mlp.routed_experts.gate_up_scale_inv / mlp.routed_experts.down_scale_inv,
    with EP slicing (dim 0) and TP slicing (scale block dim) applied.
    """
    m = PACKED_EXPERT_SCALE_RE.match(weight_name)
    if m is None:
        return False

    mlp_prefix = m.group(1)  # e.g. "model.layers.0.mlp"
    proj_name = m.group(2)  # "gate_up_proj" or "down_proj"

    # Map checkpoint name to model param name:
    #   gate_up_proj_scale_inv -> gate_up_scale_inv
    #   down_proj_scale_inv    -> down_scale_inv
    if proj_name == "gate_up_proj":
        param_name = f"{mlp_prefix}.routed_experts.gate_up_scale_inv"
    else:
        param_name = f"{mlp_prefix}.routed_experts.down_scale_inv"

    try:
        param = model.get_parameter(param_name)
    except AttributeError:
        logger.warning(
            f"Parameter {param_name} not found for packed expert scale {weight_name}"
        )
        return False

    tensor = tensor.to(torch.float32)

    # --- EP slicing: select only this rank's local experts along dim 0 ---
    dist_ctx = get_dist_context()
    ep_world_size = dist_ctx.ffn_ep_world_size
    ep_rank = dist_ctx.ffn_ep_rank
    num_total_experts = tensor.shape[0]
    experts_per_rank = num_total_experts // ep_world_size
    expert_start = ep_rank * experts_per_rank
    expert_end = expert_start + experts_per_rank
    tensor = tensor[expert_start:expert_end]

    # --- TP slicing: slice the scale block dimension ---
    tp_world_size = dist_ctx.ffn_tp_world_size
    tp_rank = dist_ctx.ffn_tp_rank
    if tp_world_size > 1 and tensor.dim() >= 2:
        if proj_name == "gate_up_proj":
            # Scale shape: (local_experts, 2*intermediate//bs, hidden//bs)
            # gate is first half of dim1, up is second half
            full_scale_dim1 = tensor.shape[1]
            half = full_scale_dim1 // 2
            chunk = half // tp_world_size
            gate_slice = tensor[:, tp_rank * chunk : (tp_rank + 1) * chunk]
            up_slice = tensor[:, half + tp_rank * chunk : half + (tp_rank + 1) * chunk]
            tensor = torch.cat([gate_slice, up_slice], dim=1)
        elif proj_name == "down_proj":
            # Scale shape: (local_experts, hidden//bs, intermediate//bs)
            # TP slices intermediate (last dim)
            if tensor.dim() == 3:
                full_scale_dim2 = tensor.shape[2]
                chunk = full_scale_dim2 // tp_world_size
                tensor = tensor[:, :, tp_rank * chunk : (tp_rank + 1) * chunk]

    # Overwrite the empty parameter with the correctly shaped scale tensor.
    # Ensure the tensor is on the same device as the parameter (model may be on GPU).
    param.data = tensor.to(device=param.data.device)
    logger.debug(
        f"Loaded packed expert scale: {weight_name} -> {param_name} "
        f"shape={list(tensor.shape)} (ep_rank={ep_rank}/{ep_world_size}, "
        f"tp_rank={tp_rank}/{tp_world_size})"
    )
    return True
