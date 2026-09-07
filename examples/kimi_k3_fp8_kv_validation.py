"""Validate K3 checkpoint MLA attention with BF16 and FP8 KV caches.

Run from an installed checkout:
    python -m examples.kimi_k3_fp8_kv_validation --model /path/to/Kimi-K3

Uses real attention weights and normalized synthetic activations on one GPU.
It does not load the MoE/KDA stack or claim end-to-end generation validation.
"""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from dlengine.runtime.context.batch import reset_batch_context, set_batch_context
from dlengine.runtime.context.cache.mla import (
    allocate_mla_kvcache,
    get_mla_block_bytes,
    resolve_mla_cache_format,
)
from dlengine.runtime.context.cache.plan import kimi_k3_cache_plan
from dlengine.runtime.context.distributed import clear_dist_context, set_dist_context
from dlengine.runtime.layers import init_backend, reset_backend
from dlengine.runtime.models.kimi_k3.kimi_k3 import KimiMLAAttention
from dlengine.runtime.models.kimi_k3.kimi_k3_loader import load_weights
from dlengine.runtime.models.quant_config import QuantizationConfig
from dlengine.runtime.models.trait import apply_hf_config_compatibility_fixes
from safetensors import safe_open
from torch import nn


def load_attention(model_dir, hf, index, layer):
    module = nn.Module()
    module.config = hf
    module.quantization_config = QuantizationConfig()
    module.model = nn.Module()
    module.model.layers = nn.ModuleDict({str(layer): nn.Module()})
    with torch.device("cuda"):
        attention = KimiMLAAttention(hf, layer_idx=layer, cache_layer_idx=0)
    module.model.layers[str(layer)].self_attn = attention
    prefix = f"language_model.model.layers.{layer}.self_attn."
    weights = {}
    for name, shard in index.items():
        if name.startswith(prefix):
            with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
                weights[name] = handle.get_tensor(name)
    expected = {
        prefix + name
        for name, _ in attention.named_parameters()
        if name not in ("kc.weight", "vc.weight")
        and not name.endswith("weight_scale_inv")
    } | {prefix + "kv_b_proj.weight"}
    if expected != weights.keys():
        raise ValueError(
            f"Incomplete MLA checkpoint weights: {expected ^ weights.keys()}"
        )
    load_weights(module, ((name, name, tensor) for name, tensor in weights.items()))
    norm_name = f"language_model.model.layers.{layer}.input_layernorm.weight"
    with safe_open(
        model_dir / index[norm_name], framework="pt", device="cpu"
    ) as handle:
        norm = handle.get_tensor(norm_name).cuda()
    return attention.eval(), norm


def allocate_cache(hf, dtype, pages):
    config = SimpleNamespace(hf_config=hf, kv_cache_dtype=dtype)
    fp8, raw = resolve_mla_cache_format(config, kimi_k3_cache_plan(), "blackwell")
    context = SimpleNamespace(
        is_fp8_kvcache=fp8,
        raw_fp8_mla_layout=raw,
        device="cuda",
        num_hidden_layers=1,
        num_local_kvcache_blocks=pages,
        num_local_kv_heads=1,
        block_size=64,
        head_dim=hf.kv_lora_rank + hf.qk_rope_head_dim,
        kv_lora_rank=hf.kv_lora_rank,
        qk_rope_head_dim=hf.qk_rope_head_dim,
        dtype=torch.bfloat16,
    )
    bytes_per_block = get_mla_block_bytes(context)
    allocate_mla_kvcache(context)
    assert context.kv_cache.untyped_storage().nbytes() == bytes_per_block * pages
    context.kv_cache.zero_()
    return context.kv_cache[0, 0]


def error_metrics(actual, expected):
    actual, expected = actual.float(), expected.float()
    assert torch.isfinite(actual).all(), "non-finite attention output"
    relative_l2 = ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    return {"relative_l2": relative_l2, "cosine": cosine}


def check_error(actual, expected, *, fp8):
    metrics = error_metrics(actual, expected)
    assert metrics["relative_l2"] < (0.10 if fp8 else 0.02), metrics
    assert metrics["cosine"] > (0.99 if fp8 else 0.999), metrics
    return metrics


@torch.inference_mode()
def validate_attention(
    attention, norm, hf, *, prefill_tokens, chunk_tokens, decode_steps
):
    total = prefill_tokens + decode_steps
    pages = max(2, (total + 63) // 64)
    # Reverse physical page order so logical positions cannot accidentally be
    # used as physical slots. Include a page boundary in both prefill modes.
    block_tables = torch.arange(
        pages - 1, -1, -1, dtype=torch.int32, device="cuda"
    ).view(1, 1, pages)
    positions = torch.arange(total, dtype=torch.int64, device="cuda")
    slots = block_tables.flatten()[positions // 64].long() * 64 + positions % 64
    hidden = F.rms_norm(
        torch.randn(total, hf.hidden_size, device="cuda", dtype=torch.bfloat16),
        (hf.hidden_size,),
        norm,
        hf.rms_norm_eps,
    )
    caches = {
        dtype: allocate_cache(hf, dtype, pages) for dtype in ("bfloat16", "fp8_e4m3")
    }
    assert (
        caches["fp8_e4m3"].untyped_storage().nbytes() * 2
        == caches["bfloat16"].untyped_storage().nbytes()
    )

    # A full BF16 prefill supplies a causal reference for every later token.
    attention.attn_fwd.k_cache = caches["bfloat16"]
    cu = torch.tensor([0, total], dtype=torch.int32, device="cuda")
    set_batch_context(
        is_prefill=True,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=total,
        max_seqlen_k=total,
    )
    reference = attention(positions, hidden)
    report = {}
    for dtype, cache in caches.items():
        attention.attn_fwd.k_cache = cache
        cache.zero_()
        fp8 = dtype == "fp8_e4m3"
        chunk_outputs = []
        for start in range(0, prefill_tokens, chunk_tokens):
            end = min(prefill_tokens, start + chunk_tokens)
            set_batch_context(
                is_prefill=True,
                cu_seqlens_q=torch.tensor(
                    [0, end - start], dtype=torch.int32, device="cuda"
                ),
                cu_seqlens_k=torch.tensor([0, end], dtype=torch.int32, device="cuda"),
                max_seqlen_q=end - start,
                max_seqlen_k=end,
                slot_mapping=slots[start:end],
                block_tables=block_tables,
            )
            chunk_outputs.append(attention(positions[start:end], hidden[start:end]))
        chunked = torch.cat(chunk_outputs)
        prefill_metrics = check_error(chunked, reference[:prefill_tokens], fp8=fp8)
        # Match each prefill GEMM's row count when checking exact stored bits;
        # a different GEMM tiling can round a few BF16 values differently.
        keys = torch.cat(
            [
                attention._kv_proj(
                    hidden[start : min(prefill_tokens, start + chunk_tokens)]
                )[0]
                for start in range(0, prefill_tokens, chunk_tokens)
            ]
        )
        stored = cache.reshape(-1, cache.shape[-1]).bfloat16()[slots[:prefill_tokens]]
        expected_keys = (
            keys.clamp(-448, 448).to(torch.float8_e4m3fn).bfloat16() if fp8 else keys
        )
        torch.testing.assert_close(stored, expected_keys, rtol=0, atol=0)

        decode_outputs = []
        graph_outputs = []
        # Capture a fixed one-token decode and replay with updated input and
        # metadata, as the runtime does. Cache writes are part of the graph.
        static_hidden = hidden[prefill_tokens : prefill_tokens + 1].clone()
        static_position = positions[prefill_tokens : prefill_tokens + 1].clone()
        static_slot = slots[prefill_tokens : prefill_tokens + 1].clone()
        static_lens = torch.tensor(
            [[prefill_tokens + 1]], dtype=torch.int32, device="cuda"
        )
        set_batch_context(
            is_prefill=False,
            context_lens=static_lens,
            block_tables=block_tables,
            slot_mapping=static_slot,
        )
        attention(static_position, static_hidden)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = attention(static_position, static_hidden)
        for token in range(prefill_tokens, total):
            static_hidden.copy_(hidden[token : token + 1])
            static_position.copy_(positions[token : token + 1])
            static_slot.copy_(slots[token : token + 1])
            static_lens.fill_(token + 1)
            eager = attention(static_position, static_hidden)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, eager, rtol=0, atol=0)
            decode_outputs.append(eager.clone())
            graph_outputs.append(captured.clone())
        decoded = torch.cat(decode_outputs)
        decode_metrics = check_error(decoded, reference[prefill_tokens:], fp8=fp8)
        check_error(torch.cat(graph_outputs), reference[prefill_tokens:], fp8=fp8)
        snapshot = cache.view(torch.uint8).clone()
        set_batch_context(
            is_prefill=False,
            is_dummy=True,
            context_lens=torch.ones(1, 1, dtype=torch.int32, device="cuda"),
            block_tables=torch.empty(1, 0, 0, dtype=torch.int32, device="cuda"),
        )
        dummy = attention(positions[:1], hidden[:1])
        assert torch.isfinite(dummy).all()
        assert torch.equal(
            cache.view(torch.uint8), snapshot
        ), "dummy decode wrote cache"
        report[dtype] = {
            "cache_bytes": cache.untyped_storage().nbytes(),
            "prefill": prefill_metrics,
            "decode": decode_metrics,
            "graph_matches_eager": True,
            "dummy_preserves_cache": True,
        }
        del graph
    reset_batch_context()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--layers", default="3,92", help="Zero-based MLA indices, or all"
    )
    parser.add_argument("--prefill-tokens", type=int, default=129)
    parser.add_argument("--chunk-tokens", type=int, default=63)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.prefill_tokens, args.chunk_tokens, args.decode_steps) <= 0:
        parser.error("token counts must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        parser.error("this validation requires a Blackwell GPU")
    raw = json.loads((args.model / "config.json").read_text())
    if raw.get("model_type") != "kimi_k3":
        parser.error("expected a Kimi-K3 checkpoint")
    hf = SimpleNamespace(**raw["text_config"])
    apply_hf_config_compatibility_fixes(hf, raw)
    full_layers = [
        i for i, kind in enumerate(hf.layer_types) if kind == "full_attention"
    ]
    layers = (
        full_layers
        if args.layers == "all"
        else [int(i) for i in args.layers.split(",")]
    )
    if any(i not in full_layers for i in layers):
        parser.error(f"choose MLA layer indices from {full_layers}")
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(2026)
    results = {
        "model_type": "kimi_k3",
        "num_attention_heads": hf.num_attention_heads,
        "prefill_tokens": args.prefill_tokens,
        "chunk_tokens": args.chunk_tokens,
        "decode_steps": args.decode_steps,
        "layers": {},
    }
    with tempfile.TemporaryDirectory(prefix="k3-fp8-dist-") as rendezvous:
        dist.init_process_group(
            "nccl", init_method=f"file://{rendezvous}/init", rank=0, world_size=1
        )
        try:
            set_dist_context(rank=0, world_size=1)
            init_backend(QuantizationConfig(), backend_type="blackwell")
            for layer in layers:
                attention, norm = load_attention(args.model, hf, index, layer)
                result = validate_attention(
                    attention,
                    norm,
                    hf,
                    prefill_tokens=args.prefill_tokens,
                    chunk_tokens=args.chunk_tokens,
                    decode_steps=args.decode_steps,
                )
                results["layers"][str(layer)] = result
                print(json.dumps({"layer": layer, **result}), flush=True)
                del attention, norm
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            reset_batch_context()
            reset_backend()
            clear_dist_context()
            dist.destroy_process_group()
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
