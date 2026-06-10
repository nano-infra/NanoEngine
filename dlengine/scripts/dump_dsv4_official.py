#!/usr/bin/env python3
"""Dump DeepSeek-V4 official inference tensors for DLEngine comparison.

Run with torchrun, for example:

  torchrun --nproc-per-node 8 dlengine/scripts/dump_dsv4_official.py \
    --inference-dir /models_cfs/models--deepseek-ai--DeepSeek-V4-Flash/inference \
    --ckpt-path /models_cfs/models--deepseek-ai--DeepSeek-V4-Flash-converted \
    --config /models_cfs/models--deepseek-ai--DeepSeek-V4-Flash/inference/config.json \
    --tokenizer-path /models_cfs/models--deepseek-ai--DeepSeek-V4-Flash \
    --prompt "1+1=?" \
    --out-dir /tmp/dsv4_debug/official
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_model
from transformers import PreTrainedTokenizerFast

BASE_HOOKS = (
    "embed",
    "norm",
    "head",
)


def _parse_layers(value: str) -> set[int]:
    return {int(item) for item in value.replace(",", " ").split() if item.strip()}


def _layer_hooks(layers: set[int]) -> set[str]:
    hooks = set(BASE_HOOKS)
    suffixes = (
        "attn_norm",
        "attn.wq_a",
        "attn.q_norm",
        "attn.wq_b",
        "attn.wkv",
        "attn.kv_norm",
        "attn.wo_a",
        "attn.wo_b",
        "attn",
        "ffn_norm",
        "ffn.gate",
        "ffn",
    )
    for layer in layers:
        hooks.update(f"layers.{layer}.{suffix}" for suffix in suffixes)
        hooks.add(f"layers.{layer}")
    return hooks


def _dump_tensor(out_dir: Path, rank: int, name: str, value, max_tokens: int) -> None:
    if isinstance(value, tuple):
        for idx, item in enumerate(value):
            _dump_tensor(out_dir, rank, f"{name}.{idx}", item, max_tokens)
        return
    if not torch.is_tensor(value):
        return
    tensor = value.detach()
    if tensor.ndim > 1 and tensor.shape[0] == 1:
        tensor = tensor.reshape(-1, *tensor.shape[2:])
    if tensor.ndim > 0:
        tensor = tensor[:max_tokens]
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "name": name,
            "rank": rank,
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "tensor": tensor.cpu().contiguous(),
        },
        out_dir / f"official_rank{rank}_{name.replace('.', '_')}.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--prompt", default="1+1=?")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--layers", default="0")
    args = parser.parse_args()
    layers = _parse_layers(args.layers)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    sys.path.insert(0, args.inference_dir)
    import model as official_model  # type: ignore
    from model import ModelArgs, Transformer  # type: ignore

    with open(args.config) as f:
        model_args = ModelArgs(**json.load(f))
    model_args.max_batch_size = 1
    model_args.max_seq_len = max(4096, args.max_tokens)
    model = Transformer(model_args)
    load_model(
        model,
        os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors"),
        strict=False,
    )
    model.eval()

    out_dir = Path(args.out_dir)
    sparse_attn_calls = {"count": 0}
    original_sparse_attn = official_model.sparse_attn

    def sparse_attn_wrapper(*sparse_args, **sparse_kwargs):
        layer_idx = sparse_attn_calls["count"]
        if layer_idx in layers:
            q, kv, attn_sink, topk_idxs, _softmax_scale = sparse_args
            _dump_tensor(
                out_dir,
                rank,
                f"layers.{layer_idx}.attn_q_after_rope",
                q,
                args.max_tokens,
            )
            _dump_tensor(
                out_dir,
                rank,
                f"layers.{layer_idx}.attn_kv_after_rope",
                kv,
                args.max_tokens,
            )
            _dump_tensor(
                out_dir,
                rank,
                f"layers.{layer_idx}.attn_sink",
                attn_sink,
                args.max_tokens,
            )
            _dump_tensor(
                out_dir,
                rank,
                f"layers.{layer_idx}.attn_topk_idxs",
                topk_idxs,
                args.max_tokens,
            )
        out = original_sparse_attn(*sparse_args, **sparse_kwargs)
        if layer_idx in layers:
            _dump_tensor(
                out_dir, rank, f"layers.{layer_idx}.attn_context", out, args.max_tokens
            )
        sparse_attn_calls["count"] += 1
        return out

    official_model.sparse_attn = sparse_attn_wrapper

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            _dump_tensor(out_dir, rank, name, output, args.max_tokens)

        return hook

    def make_pre_hook(name: str):
        def hook(_module, inputs):
            if inputs:
                _dump_tensor(out_dir, rank, name, inputs[0], args.max_tokens)

        return hook

    hooks = _layer_hooks(layers)
    for name, module in model.named_modules():
        if name in hooks:
            module.register_forward_hook(make_hook(name))
        if any(name == f"layers.{layer}.attn.wo_b" for layer in layers):
            # Official demo applies wo_a via an inline einsum over wo_a.weight,
            # so the input to wo_b is the best matching wo_a output tensor.
            layer = int(name.split(".")[1])
            module.register_forward_pre_hook(make_pre_hook(f"layers.{layer}.attn.wo_a"))

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    input_ids = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device="cuda"
    )
    with torch.inference_mode():
        logits = model(input_ids, start_pos=0)
    _dump_tensor(out_dir, rank, "logits", logits, args.max_tokens)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
