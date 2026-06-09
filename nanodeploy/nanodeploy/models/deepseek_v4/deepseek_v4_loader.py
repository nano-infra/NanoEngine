"""Weight loader for the initial DeepSeek-V4 NanoDeploy path."""

import re
from typing import Generator, Tuple

import torch
from torch import nn

from nanodeploy.context.distributed import get_dist_context
from nanodeploy.logging import get_logger
from nanodeploy.worker.loader import (
    default_weight_loader,
    EXPERT_RE,
    load_per_expert_weight,
)

logger = get_logger()


FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


_PACKED_MODULES_MAPPING = {
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}


def _remap_name(name: str) -> str:
    if name == "embed.weight":
        return "model.embed_tokens.weight"
    if name == "head.weight":
        return "lm_head.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if name.startswith("hc_head_"):
        return "model." + name.replace("hc_head_", "hc_head.")

    if name.startswith("layers."):
        name = "model." + name
    name = name.replace(".attn.", ".self_attn.")
    name = name.replace(".ffn.", ".mlp.")
    name = name.replace(".attn_norm.", ".input_layernorm.")
    name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

    name = name.replace(".hc_attn_fn", ".hc_attn.fn")
    name = name.replace(".hc_ffn_fn", ".hc_ffn.fn")
    name = name.replace(".hc_attn_base", ".hc_attn.base")
    name = name.replace(".hc_ffn_base", ".hc_ffn.base")
    name = name.replace(".hc_attn_scale", ".hc_attn.scale")
    name = name.replace(".hc_ffn_scale", ".hc_ffn.scale")

    name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
    name = name.replace(".w1.", ".gate_proj.")
    name = name.replace(".w2.", ".down_proj.")
    name = name.replace(".w3.", ".up_proj.")
    if name.endswith(".scale") and (".self_attn." in name or ".mlp." in name):
        name = name[: -len(".scale")] + ".weight_scale_inv"
    return name


def _maybe_cast_for_param(
    param: nn.Parameter, tensor: torch.Tensor, name: str
) -> torch.Tensor:
    if tensor.dtype == param.dtype:
        return tensor
    if tensor.dtype == torch.float4_e2m1fn_x2:
        raise RuntimeError(
            f"{name} is FP4 but NanoDeploy H200 initial path expects FP8/BF16. "
            "Run scripts/convert_dsv4_weight.py with --expert-dtype fp8."
        )
    if param.dtype in (torch.bfloat16, torch.float32) and tensor.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ):
        return tensor.to(param.dtype)
    return tensor


def _convert_fp4_weight_to_fp8(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert HF packed E2M1 FP4 expert weight into Hopper FP8 block format."""
    weight = weight.cpu()
    scale = scale.cpu()
    assert weight.dtype == torch.int8
    assert weight.ndim == 2
    out_dim, packed_in_dim = weight.shape
    in_dim = packed_in_dim * 2
    fp8_block_size = 128
    fp4_block_size = 32
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
    assert scale.shape == (out_dim, in_dim // fp4_block_size)

    x = weight.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    fp4_table = FP4_TABLE.cpu()
    x = torch.stack([fp4_table[low.long()], fp4_table[high.long()]], dim=-1).flatten(1)

    b_out = out_dim // fp8_block_size
    b_in = in_dim // fp8_block_size
    x = x.view(b_out, fp8_block_size, b_in, fp8_block_size).transpose(1, 2)
    scale = scale.float().view(b_out, fp8_block_size, b_in, -1).transpose(1, 2)
    scale = scale.flatten(2)
    scale_max_offset = scale.amax(dim=-1, keepdim=True) / 64
    offset = scale / scale_max_offset
    offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(
        fp4_block_size, dim=-1
    )
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn), scale_max_offset.squeeze(-1).to(
        torch.float8_e8m0fnu
    )


def _dequant_fp8_block_to_bf16(
    weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Dequantize FP8 block-quantized dense weight for BF16-only kernels."""
    weight = weight.cpu()
    scale = scale.cpu()
    assert weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    out_dim, in_dim = weight.shape
    block_size = 128
    out_blocks, in_blocks = scale.shape
    padded = torch.zeros(
        out_blocks * block_size,
        in_blocks * block_size,
        dtype=weight.dtype,
        device=weight.device,
    )
    padded[:out_dim, :in_dim] = weight
    padded = padded.view(out_blocks, block_size, in_blocks, block_size)
    dequant = padded.to(torch.bfloat16) * scale.to(torch.bfloat16)[:, None, :, None]
    return dequant.reshape(out_blocks * block_size, in_blocks * block_size)[
        :out_dim, :in_dim
    ].contiguous()


def _is_local_expert_weight(weight_name: str, config) -> bool:
    match = EXPERT_RE.match(weight_name)
    if match is None:
        return True
    expert_idx = int(match.group(2))
    num_experts = getattr(config, "n_routed_experts", None) or getattr(
        config, "num_experts", None
    )
    if num_experts is None:
        return True
    dist_ctx = get_dist_context()
    ep_size = dist_ctx.ffn_ep_world_size
    ep_rank = dist_ctx.ffn_ep_rank
    experts_per_rank = num_experts // ep_size
    return ep_rank * experts_per_rank <= expert_idx < (ep_rank + 1) * experts_per_rank


def load_weights(
    model: nn.Module,
    weights: Generator[Tuple[str, str, torch.Tensor], None, None],
) -> None:
    config = model.config
    loaded_count = 0
    skipped_count = 0
    not_found_names: list[str] = []
    pending_fp4_experts: dict[str, torch.Tensor] = {}
    pending_wo_a: dict[str, dict[str, torch.Tensor]] = {}

    def load_one(weight_name: str, tensor: torch.Tensor) -> bool:
        nonlocal loaded_count, skipped_count

        if "experts." in weight_name and EXPERT_RE.match(weight_name):
            if load_per_expert_weight(model, weight_name, tensor, config):
                loaded_count += 1
                return True

        matched = False
        for source, (target, shard_id) in _PACKED_MODULES_MAPPING.items():
            if source in weight_name:
                param_name = weight_name.replace(source, target)
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    continue
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(
                    param,
                    _maybe_cast_for_param(param, tensor, weight_name),
                    shard_id,
                    weight_name,
                )
                loaded_count += 1
                matched = True
                break
        if matched:
            return True

        try:
            param = model.get_parameter(weight_name)
        except AttributeError:
            not_found_names.append(weight_name)
            skipped_count += 1
            return False

        loader = getattr(param, "weight_loader", default_weight_loader)
        loader(param, _maybe_cast_for_param(param, tensor, weight_name))
        loaded_count += 1
        return True

    for weight_name, raw_weight_name, tensor in weights:
        weight_name = _remap_name(weight_name)

        if ".indexer." in weight_name:
            skipped_count += 1
            continue

        if weight_name.endswith(".self_attn.wo_a.weight") or weight_name.endswith(
            ".self_attn.wo_a.weight_scale_inv"
        ):
            if (
                weight_name.endswith(".self_attn.wo_a.weight")
                and tensor.dtype == torch.bfloat16
            ):
                load_one(weight_name, tensor)
                continue
            base_name = weight_name.replace(".weight_scale_inv", ".weight")
            pair = pending_wo_a.setdefault(base_name, {})
            pair[weight_name.rsplit(".", 1)[-1]] = tensor
            if "weight" in pair and "weight_scale_inv" in pair:
                load_one(
                    base_name,
                    _dequant_fp8_block_to_bf16(
                        pair["weight"],
                        pair["weight_scale_inv"],
                    ),
                )
                pending_wo_a.pop(base_name, None)
            continue

        if "experts." in weight_name and (
            weight_name.endswith(".weight") or weight_name.endswith(".weight_scale_inv")
        ):
            if not _is_local_expert_weight(weight_name, config):
                loaded_count += 1
                continue

            base_name = weight_name.replace(".weight_scale_inv", ".weight")
            is_hf_fp4_scale = (
                weight_name.endswith(".weight_scale_inv")
                and tensor.ndim == 2
                and tensor.shape[0] > 128
            )
            if (
                tensor.dtype == torch.int8
                or is_hf_fp4_scale
                or base_name in pending_fp4_experts
            ):
                pending_fp4_experts[weight_name] = tensor
                weight = pending_fp4_experts.get(base_name)
                scale = pending_fp4_experts.get(
                    base_name.replace(".weight", ".weight_scale_inv")
                )
                if weight is not None and scale is not None:
                    fp8_weight, fp8_scale = _convert_fp4_weight_to_fp8(weight, scale)
                    load_one(base_name, fp8_weight)
                    load_one(
                        base_name.replace(".weight", ".weight_scale_inv"), fp8_scale
                    )
                    pending_fp4_experts.pop(base_name, None)
                    pending_fp4_experts.pop(
                        base_name.replace(".weight", ".weight_scale_inv"), None
                    )
                continue

        load_one(weight_name, tensor)

    logger.warning(
        f"DeepSeek-V4 weight loading complete: {loaded_count} loaded, {skipped_count} skipped"
    )
    if pending_fp4_experts or pending_wo_a:
        logger.warning(
            f"DeepSeek-V4 pending paired weights after loading: "
            f"fp4_experts={len(pending_fp4_experts)}, wo_a={list(pending_wo_a)}"
        )

    # Mega-MoE post-load hook: walk the model tree and transform every
    # routed-experts layer's weights into the layout
    # ``deep_gemm.fp8_fp4_mega_moe`` expects. No-op when use_mega_moe
    # is off or when a layer isn't FP8 (the method bails early).
    try:
        from nanodeploy.worker.runner_config import get_runner_config

        if getattr(get_runner_config(), "use_mega_moe", False):
            transformed = 0
            failed = 0
            first_failure_tb: str | None = None
            for module in model.modules():
                if not (
                    hasattr(module, "prepare_mega_weights")
                    and callable(module.prepare_mega_weights)
                ):
                    continue
                try:
                    module.prepare_mega_weights()
                except Exception:
                    failed += 1
                    if first_failure_tb is None:
                        import traceback as _tb

                        first_failure_tb = _tb.format_exc()
                    continue
                if getattr(module, "mega_l1_weights", None) is not None:
                    transformed += 1
            if transformed:
                logger.warning(
                    f"DeepSeek-V4 mega-MoE: transformed weights for {transformed} layers"
                )
            if failed:
                logger.warning(
                    f"DeepSeek-V4 mega-MoE: {failed} layer(s) failed weight transform; "
                    f"first traceback:\n{first_failure_tb}"
                )
    except Exception:
        import traceback as _tb

        logger.warning(
            f"DeepSeek-V4 mega-MoE post-load hook failed:\n{_tb.format_exc()}"
        )
    if not_found_names:
        unique_patterns = set()
        for name in not_found_names:
            pat = re.sub(r"layers\.\d+\.", "layers.N.", name)
            pat = re.sub(r"experts\.\d+\.", "experts.E.", pat)
            unique_patterns.add(pat)
        logger.warning(
            f"  {len(not_found_names)} DSV4 weights NOT FOUND "
            f"(unique patterns: {sorted(unique_patterns)[:30]})"
        )
