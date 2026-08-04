"""Kimi K3 checkpoint loader."""

from __future__ import annotations

import re

import torch

from dlengine.context_v2.distributed import get_dist_context
from dlengine.logging import get_logger
from dlengine.models.deepseek_v2.deepseek_v2_loader import _handle_kv_b_proj
from dlengine.worker.loader import default_weight_loader

logger = get_logger()
_EXPERT = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(w[123])\."
    r"(weight_packed|weight_scale)$"
)
_CONV = re.compile(r"(.+\.self_attn)\.([qkv])_conv1d\.weight$")


def _normalize_name(name: str) -> str | None:
    if name.startswith(("vision_tower.", "mm_projector.")):
        return None
    if name.startswith("language_model."):
        return name[len("language_model.") :]
    return name


def load_weights(model, weights) -> None:
    loaded = skipped = 0
    missing: set[str] = set()
    kv_b: list[tuple[str, torch.Tensor]] = []
    ctx = get_dist_context()

    for name, raw_name, tensor in weights:
        name = _normalize_name(raw_name)
        if name is None:
            skipped += 1
            continue
        name = name.replace(".block_sparse_moe.", ".mlp.")
        name = name.replace(
            ".mlp.gate.e_score_correction_bias",
            ".mlp.e_score_correction_bias",
        )

        expert_match = _EXPERT.match(name)
        if expert_match:
            layer, expert, projection, kind = expert_match.groups()
            module = model.get_submodule(f"model.layers.{layer}.mlp.experts")
            module.load_expert_weight(
                int(expert), projection, kind, tensor,
                ep_rank=ctx.ffn_ep_rank,
            )
            loaded += 1
            continue

        conv_match = _CONV.match(name)
        if conv_match:
            prefix, which = conv_match.groups()
            param = model.get_parameter(prefix + ".conv1d.weight")
            local = param.shape[0] // 3
            full = tensor.shape[0]
            shard = full // ctx.attn_tp_world_size
            start = ctx.attn_tp_rank * shard
            part = tensor[start : start + shard]
            offset = {"q": 0, "k": local, "v": 2 * local}[which]
            param.data[offset : offset + local].copy_(part)
            loaded += 1
            continue

        if name.endswith("self_attn.A_log"):
            param = model.get_parameter(name)
            start = ctx.attn_tp_rank * param.numel()
            param.data.copy_(tensor.flatten()[start : start + param.numel()])
            loaded += 1
            continue
        if name.endswith("self_attn.dt_bias"):
            param = model.get_parameter(name)
            start = ctx.attn_tp_rank * param.numel()
            param.data.copy_(tensor.flatten()[start : start + param.numel()])
            loaded += 1
            continue

        if name.endswith("self_attn.kv_b_proj.weight"):
            kv_b.append((name, tensor))
            continue

        try:
            param = model.get_parameter(name)
        except AttributeError:
            missing.add(re.sub(r"layers\.\d+", "layers.N", name))
            skipped += 1
            continue
        loader = getattr(param, "weight_loader", default_weight_loader)
        loader(param, tensor)
        loaded += 1

    for name, tensor in kv_b:
        _handle_kv_b_proj(
            model, name, tensor, {}, model.config,
            getattr(model.quantization_config, "block_size", []),
        )
        loaded += 1

    transformed = 0
    for module in model.modules():
        prepare = getattr(module, "prepare_mega_weights", None)
        if callable(prepare):
            prepare()
            transformed += 1
        prepare_front = getattr(module, "prepare_fused_front", None)
        if callable(prepare_front):
            prepare_front()
    # Merged front parameters now view the consolidated allocations. Release
    # the superseded gate/down storage before KV-cache capacity is profiled.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.warning(
        "K3 weights: loaded=%d skipped=%d MegaMoE layers=%d missing=%s",
        loaded, skipped, transformed, sorted(missing),
    )
