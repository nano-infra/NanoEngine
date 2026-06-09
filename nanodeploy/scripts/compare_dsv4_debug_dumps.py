#!/usr/bin/env python3
"""Compare official DeepSeek-V4 and NanoDeploy debug tensor dumps."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _mapping(layer: int) -> dict[str, tuple[str, str, str]]:
    return {
        "embed": ("embed", "global_embed", "rank0"),
        f"layers.{layer}.attn_norm": (
            f"layers_{layer}_attn_norm",
            f"layer{layer}_attn_norm",
            "rank0",
        ),
        f"layers.{layer}.attn.wq_a": (
            f"layers_{layer}_attn_wq_a",
            f"layer{layer}_attn_q_lora_pre_norm",
            "rank0",
        ),
        f"layers.{layer}.attn.q_norm": (
            f"layers_{layer}_attn_q_norm",
            f"layer{layer}_attn_q_lora",
            "rank0",
        ),
        f"layers.{layer}.attn.wq_b": (
            f"layers_{layer}_attn_wq_b",
            f"layer{layer}_attn_wq_b",
            "concat_last",
        ),
        f"layers.{layer}.attn.wkv": (
            f"layers_{layer}_attn_wkv",
            f"layer{layer}_attn_kv_pre_norm",
            "rank0",
        ),
        f"layers.{layer}.attn.kv_norm": (
            f"layers_{layer}_attn_kv_norm",
            f"layer{layer}_attn_kv",
            "rank0",
        ),
        f"layers.{layer}.attn.q_rope": (
            f"layers_{layer}_attn_q_after_rope",
            f"layer{layer}_attn_q_after_rope",
            "concat_heads",
        ),
        f"layers.{layer}.attn.kv_rope": (
            f"layers_{layer}_attn_kv_after_rope",
            f"layer{layer}_attn_kv_after_rope",
            "rank0_squeeze_head",
        ),
        f"layers.{layer}.attn.context": (
            f"layers_{layer}_attn_context",
            f"layer{layer}_attn_context",
            "concat_heads",
        ),
        f"layers.{layer}.attn.wo_a": (
            f"layers_{layer}_attn_wo_a",
            f"layer{layer}_attn_wo_a",
            "concat_last",
        ),
        f"layers.{layer}.attn.wo_b": (
            f"layers_{layer}_attn_wo_b",
            f"layer{layer}_attn_out",
            "rank0",
        ),
        f"layers.{layer}.attn": (
            f"layers_{layer}_attn",
            f"layer{layer}_attn_block_out",
            "rank0",
        ),
        f"layers.{layer}.ffn_norm": (
            f"layers_{layer}_ffn_norm",
            f"layer{layer}_ffn_norm",
            "rank0",
        ),
        f"layers.{layer}.ffn": (
            f"layers_{layer}_ffn",
            f"layer{layer}_ffn_block_out",
            "rank0",
        ),
        f"layers.{layer}": (
            f"layers_{layer}",
            f"layer{layer}_layer_out",
            "rank0",
        ),
    }


def _load_payload(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    tensor = obj["tensor"] if isinstance(obj, dict) and "tensor" in obj else obj
    if tensor.ndim > 1 and tensor.shape[0] == 1:
        tensor = tensor.reshape(-1, *tensor.shape[2:])
    return tensor.float()


def _official(official_dir: Path, stem: str, mode: str) -> torch.Tensor | None:
    files = sorted(official_dir.glob(f"official_rank*_{stem}.pt"))
    if not files:
        return None
    if mode in ("rank0", "rank0_squeeze_head"):
        f = official_dir / f"official_rank0_{stem}.pt"
        return _load_payload(f) if f.exists() else _load_payload(files[0])
    tensors = [_load_payload(f) for f in files]
    if mode == "concat_last":
        return torch.cat(tensors, dim=-1)
    if mode == "concat_heads":
        return torch.cat(tensors, dim=-2)
    raise ValueError(f"Unknown mode={mode}")


def _nano(nano_dir: Path, stem: str, mode: str) -> torch.Tensor | None:
    files = sorted(nano_dir.glob(f"nanodeploy_rank*_{stem}.pt"))
    if not files:
        return None
    # In DP runs with a single request, rank0 is often a dummy worker.  Pick
    # the largest tensor among dumped ranks, which corresponds to the real
    # prompt chunk instead of a one-token dummy sequence.
    tensors = [_load_payload(file) for file in files]
    tensor = max(tensors, key=lambda tensor: tensor.numel())
    if mode == "concat_last" and tensor.ndim == 3:
        tensor = tensor.flatten(1)
    if mode == "rank0_squeeze_head" and tensor.ndim == 3 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    return tensor


def _align(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.shape == b.shape:
        return a, b
    # Compare flattened common prefix as a fallback so shape mismatches are still informative.
    n = min(a.numel(), b.numel())
    return a.reshape(-1)[:n], b.reshape(-1)[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dir", required=True)
    parser.add_argument("--nanodeploy-dir", required=True)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()

    official_dir = Path(args.official_dir)
    nano_dir = Path(args.nanodeploy_dir)
    print(
        f"{'name':32s} {'official':24s} {'nano':24s} {'max_abs':>12s} {'mean_abs':>12s} {'cos':>10s}"
    )
    print("-" * 120)
    for name, (official_stem, nano_stem, mode) in _mapping(args.layer).items():
        a = _official(official_dir, official_stem, mode)
        b = _nano(nano_dir, nano_stem, mode)
        if a is None or b is None:
            print(f"{name:32s} missing official={a is None} nano={b is None}")
            continue
        aa, bb = _align(a, b)
        diff = (aa - bb).abs()
        denom = aa.norm() * bb.norm()
        cos = (
            float((aa.flatten() @ bb.flatten()) / denom) if denom > 0 else float("nan")
        )
        print(
            f"{name:32s} {str(tuple(a.shape)):24s} {str(tuple(b.shape)):24s} "
            f"{float(diff.max()):12.5g} {float(diff.mean()):12.5g} {cos:10.6f}"
        )


if __name__ == "__main__":
    main()
