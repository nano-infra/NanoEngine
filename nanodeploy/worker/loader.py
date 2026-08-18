import os
import re
from collections.abc import Iterable, Iterator
from glob import glob

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm

from nanodeploy.models.weight_loadable import (
    WeightLoadableModel,
    default_weight_loader,
)


_DEEPSEEK_LAYER_RE = re.compile(r"^model\.layers\.(?P<layer_id>\d+)\.")
_DEEPSEEK_EXPERT_RE = re.compile(
    r"^(?P<moe_name>model\.layers\.\d+\.mlp)\.experts\."
    r"(?P<expert_id>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight|weight_scale_inv)$"
)
_DEEPSEEK_KV_B_RE = re.compile(
    r"^(?P<attention_name>model\.layers\.\d+\.self_attn)\."
    r"kv_b_proj\.(?P<kind>weight|weight_scale_inv)$"
)
_DEEPSEEK_PACKED_MAPPINGS = (
    (".self_attn.q_a_proj.", ".self_attn.fused_qkv_a_proj.", 0),
    (
        ".self_attn.kv_a_proj_with_mqa.",
        ".self_attn.fused_qkv_a_proj.",
        1,
    ),
    (".gate_proj.", ".gate_up_proj.", 0),
    (".up_proj.", ".gate_up_proj.", 1),
)
_FP8_DTYPES = (torch.float8_e4m3fn,)


def _iter_safetensor_weights(
    model: WeightLoadableModel,
    weight_files: Iterable[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    for file in weight_files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for weight_name in handle.keys():
                if not model.should_load_weight(weight_name):
                    continue
                yield weight_name, handle.get_tensor(weight_name)


def load_model(model: WeightLoadableModel, path: str) -> None:
    """Load safetensor checkpoints, delegating model-specific layouts when needed."""
    weight_files = sorted(glob(os.path.join(path, "*.safetensors")))
    progress = tqdm(weight_files, desc="Loading weights", unit="files")
    weights = _iter_safetensor_weights(model, progress)

    try:
        model.load_weights(weights)
    finally:
        progress.close()


def _deepseek_local_expert_indices(
    model: WeightLoadableModel,
    moe_name: str,
) -> dict[int, int]:
    cache = model._weight_loader_local_expert_indices
    if moe_name not in cache:
        moe = model.get_submodule(moe_name)
        cache[moe_name] = {
            expert_id: local_id
            for local_id, expert_id in enumerate(moe.expert_list_this_rank)
        }
    return cache[moe_name]


def should_load_deepseek_weight(
    model: WeightLoadableModel,
    weight_name: str,
) -> bool:
    """Return whether a DeepSeek checkpoint tensor is useful on this rank."""
    if "rotary_emb.inv_freq" in weight_name:
        return False

    layer_match = _DEEPSEEK_LAYER_RE.match(weight_name)
    if layer_match is not None:
        layer_id = int(layer_match.group("layer_id"))
        if layer_id >= len(model.model.layers):
            # DeepSeek-V3 checkpoints append MTP/speculative layers after the
            # main decoder layers. They are not part of this model instance.
            return False

    expert_match = _DEEPSEEK_EXPERT_RE.match(weight_name)
    if expert_match is None:
        return True
    local_experts = _deepseek_local_expert_indices(
        model, expert_match.group("moe_name")
    )
    return int(expert_match.group("expert_id")) in local_experts


def _copy_tensor(
    destination: torch.Tensor,
    source: torch.Tensor,
    weight_name: str,
) -> None:
    if tuple(destination.shape) != tuple(source.shape):
        raise ValueError(
            f"Cannot load {weight_name}: checkpoint shape {tuple(source.shape)} "
            f"does not match destination shape {tuple(destination.shape)}"
        )
    destination.copy_(source)


def _load_deepseek_packed_weight(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: int,
    weight_name: str,
) -> None:
    if loaded_weight.ndim == 0 or param.ndim == 0:
        raise ValueError(f"Packed weight {weight_name} must have at least one dimension")
    shard_size = loaded_weight.shape[0]
    shard_offset = 0 if shard_id == 0 else param.shape[0] - shard_size
    if shard_offset < 0:
        raise ValueError(
            f"Cannot load packed weight {weight_name}: checkpoint shard has "
            f"{shard_size} rows but destination has {param.shape[0]}"
        )
    destination = param.data.narrow(0, shard_offset, shard_size)
    _copy_tensor(destination, loaded_weight, weight_name)


def _load_deepseek_expert_weight(
    model: WeightLoadableModel,
    params: dict[str, nn.Parameter],
    match: re.Match[str],
    loaded_weight: torch.Tensor,
    weight_name: str,
) -> str:
    moe_name = match.group("moe_name")
    expert_id = int(match.group("expert_id"))
    local_experts = _deepseek_local_expert_indices(model, moe_name)
    if expert_id not in local_experts:
        return ""
    local_expert_id = local_experts[expert_id]
    projection = match.group("projection")
    kind = match.group("kind")

    if projection in {"gate_proj", "up_proj"}:
        param_suffix = "gate_up_scale_inv" if kind == "weight_scale_inv" else "gate_up_proj"
        param_name = f"{moe_name}.{param_suffix}"
        destination = params[param_name].data[local_expert_id]
        shard_id = 0 if projection == "gate_proj" else 1
        shard_offset = 0 if shard_id == 0 else destination.shape[0] - loaded_weight.shape[0]
        destination = destination.narrow(0, shard_offset, loaded_weight.shape[0])
    else:
        param_suffix = "down_scale_inv" if kind == "weight_scale_inv" else "down_proj"
        param_name = f"{moe_name}.{param_suffix}"
        destination = params[param_name].data[local_expert_id]

    _copy_tensor(destination, loaded_weight, weight_name)
    return param_name


def _dequantize_fp8_block_weight(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int],
    destination: torch.Tensor,
    weight_name: str,
) -> torch.Tensor:
    expected_scale_shape = tuple(
        (size + block - 1) // block
        for size, block in zip(weight.shape, block_size, strict=True)
    )
    if tuple(scale_inv.shape) != expected_scale_shape:
        raise ValueError(
            f"Cannot dequantize {weight_name}: scale shape {tuple(scale_inv.shape)} "
            f"does not match expected shape {expected_scale_shape}"
        )

    dequantized = weight.to(device=destination.device, dtype=destination.dtype)
    expanded_scale = scale_inv.to(
        device=destination.device,
        dtype=destination.dtype,
    )
    expanded_scale = expanded_scale.repeat_interleave(block_size[0], dim=0)
    expanded_scale = expanded_scale.repeat_interleave(block_size[1], dim=1)
    dequantized.mul_(expanded_scale[: weight.shape[0], : weight.shape[1]])
    return dequantized


def _load_deepseek_kv_b_weight(
    model: WeightLoadableModel,
    params: dict[str, nn.Parameter],
    attention_name: str,
    weight: torch.Tensor,
    scale_inv: torch.Tensor | None,
) -> tuple[str, str]:
    kc_name = f"{attention_name}.kc.weight"
    vc_name = f"{attention_name}.vc.weight"
    kc_param = params[kc_name]
    vc_param = params[vc_name]

    if weight.dtype in _FP8_DTYPES:
        if scale_inv is None:
            raise ValueError(f"Missing FP8 scale for {attention_name}.kv_b_proj.weight")
        configured_block_size = tuple(model.quantization_config.block_size)
        if len(configured_block_size) != 2:
            raise ValueError(
                "DeepSeek FP8 kv_b_proj requires a two-dimensional weight block size"
            )
        weight = _dequantize_fp8_block_weight(
            weight,
            scale_inv,
            configured_block_size,
            kc_param,
            f"{attention_name}.kv_b_proj.weight",
        )
    else:
        weight = weight.to(device=kc_param.device, dtype=kc_param.dtype)

    num_heads = model.config.num_attention_heads
    qk_nope_head_dim = model.config.qk_nope_head_dim
    v_head_dim = model.config.v_head_dim
    kv_lora_rank = model.config.kv_lora_rank
    expected_shape = (
        num_heads * (qk_nope_head_dim + v_head_dim),
        kv_lora_rank,
    )
    if tuple(weight.shape) != expected_shape:
        raise ValueError(
            f"Cannot split {attention_name}.kv_b_proj.weight: shape "
            f"{tuple(weight.shape)} does not match expected shape {expected_shape}"
        )

    per_head = weight.view(
        num_heads,
        qk_nope_head_dim + v_head_dim,
        kv_lora_rank,
    )
    _copy_tensor(
        kc_param.data,
        per_head[:, :qk_nope_head_dim, :],
        f"{attention_name}.kv_b_proj.weight (key projection)",
    )
    _copy_tensor(
        vc_param.data,
        per_head[:, qk_nope_head_dim:, :].transpose(1, 2),
        f"{attention_name}.kv_b_proj.weight (value projection)",
    )
    return kc_name, vc_name


def load_deepseek_weights(
    model: WeightLoadableModel,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load a Hugging Face DeepSeek-V3/Kimi-K2 checkpoint into NanoDeploy."""
    params = dict(model.named_parameters())
    loaded_params: set[str] = set()
    pending_kv_b: dict[str, dict[str, torch.Tensor]] = {}

    for weight_name, loaded_weight in weights:
        if not should_load_deepseek_weight(model, weight_name):
            continue

        expert_match = _DEEPSEEK_EXPERT_RE.match(weight_name)
        if expert_match is not None:
            param_name = _load_deepseek_expert_weight(
                model,
                params,
                expert_match,
                loaded_weight,
                weight_name,
            )
            if param_name:
                loaded_params.add(param_name)
            continue

        kv_b_match = _DEEPSEEK_KV_B_RE.match(weight_name)
        if kv_b_match is not None:
            attention_name = kv_b_match.group("attention_name")
            kind = kv_b_match.group("kind")
            entry = pending_kv_b.setdefault(attention_name, {})
            entry[kind] = loaded_weight
            weight = entry.get("weight")
            scale_inv = entry.get("weight_scale_inv")
            if weight is None:
                continue
            if weight.dtype in _FP8_DTYPES and scale_inv is None:
                continue
            loaded_params.update(
                _load_deepseek_kv_b_weight(
                    model,
                    params,
                    attention_name,
                    weight,
                    scale_inv,
                )
            )
            del pending_kv_b[attention_name]
            continue

        for checkpoint_name, param_component, shard_id in _DEEPSEEK_PACKED_MAPPINGS:
            if checkpoint_name not in weight_name:
                continue
            param_name = weight_name.replace(checkpoint_name, param_component)
            param = params[param_name]
            _load_deepseek_packed_weight(
                param,
                loaded_weight,
                shard_id,
                weight_name,
            )
            loaded_params.add(param_name)
            break
        else:
            if weight_name not in params:
                raise KeyError(f"Unexpected DeepSeek checkpoint weight: {weight_name}")
            param = params[weight_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(weight_name)

    if pending_kv_b:
        incomplete = ", ".join(sorted(pending_kv_b))
        raise ValueError(f"Incomplete kv_b_proj FP8 weights for: {incomplete}")
    return loaded_params
