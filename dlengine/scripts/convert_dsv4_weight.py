#!/usr/bin/env python3
"""Convert DeepSeek-V4 HuggingFace checkpoints to sharded inference weights.

This is adapted from the DeepSeek-V4 release inference converter, with a
DLEngine-friendly CLI:

    python dlengine/scripts/convert_dsv4_weight.py -i /path/to/hf -o /path/to/out
"""

import json
import os
import shutil
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm, trange

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


MAPPING = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "lm_head": ("head", 0),
    "embed": ("embed", 0),
    "wq_b": ("wq_b", 0),
    "wo_a": ("wo_a", 0),
    "wo_b": ("wo_b", 1),
    "head": ("head", 0),
    "attn_sink": ("attn_sink", 0),
    "weights_proj": ("weights_proj", 0),
}


def cast_e2m1fn_to_e4m3fn(
    x: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast packed FP4 expert weights to FP8 using DeepSeek's lossless recipe."""
    assert x.dtype == torch.int8
    assert x.ndim == 2
    out_dim, in_dim = x.size()
    in_dim *= 2
    fp8_block_size = 128
    fp4_block_size = 32
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
    assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size

    x = x.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)

    max_offset_bits = 6
    b_out = out_dim // fp8_block_size
    b_in = in_dim // fp8_block_size
    x = x.view(b_out, fp8_block_size, b_in, fp8_block_size).transpose(1, 2)
    scale = scale.float().view(b_out, fp8_block_size, b_in, -1).transpose(1, 2)
    scale = scale.flatten(2)
    scale_max_offset_bits = scale.amax(dim=-1, keepdim=True) / (2**max_offset_bits)
    offset = scale / scale_max_offset_bits
    offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(
        fp4_block_size, dim=-1
    )
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn), scale_max_offset_bits.squeeze(-1).to(
        torch.float8_e8m0fnu
    )


def infer_n_experts(input_dir: Path) -> int:
    config_path = input_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Cannot infer n_experts because {config_path} does not exist. "
            "Pass --n-experts explicitly."
        )
    with config_path.open() as f:
        config = json.load(f)
    n_experts = config.get("n_routed_experts")
    if not isinstance(n_experts, int):
        raise ValueError(
            f"Cannot find integer n_routed_experts in {config_path}. "
            "Pass --n-experts explicitly."
        )
    return n_experts


def is_lfs_pointer(file_path: Path) -> bool:
    with file_path.open("rb") as f:
        return f.read(48).startswith(b"version https://git-lfs.github.com/spec")


def validate_input_files(input_dir: Path) -> list[str]:
    weight_files = sorted(glob(str(input_dir / "*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors files found under {input_dir}")
    pointer_files = [p for p in weight_files if is_lfs_pointer(Path(p))]
    if pointer_files:
        sample = pointer_files[0]
        raise RuntimeError(
            "Found Git LFS pointer files instead of real safetensors weights. "
            f"Example: {sample}. Run git lfs pull or download the full model files."
        )
    return weight_files


def convert_name(name: str) -> tuple[str, str | None, int | None]:
    if name.startswith("model."):
        name = name[len("model.") :]

    name = name.replace("self_attn", "attn")
    name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    if any(x in name for x in ["hc", "attn_sink", "tie2eid", "ape"]):
        key = name.split(".")[-1]
    else:
        key = name.split(".")[-2]
    new_key, dim = MAPPING.get(key, (key, None))
    return name.replace(key, new_key), key, dim


def convert_weights(
    input_dir: Path,
    output_dir: Path,
    n_experts: int,
    model_parallel: int,
    expert_dtype: str,
    parallel_mode: str = "dp-ep",
) -> None:
    torch.set_num_threads(8)
    weight_files = validate_input_files(input_dir)
    if n_experts % model_parallel != 0:
        raise ValueError(
            f"n_experts={n_experts} must be divisible by model_parallel={model_parallel}"
        )

    n_local_experts = n_experts // model_parallel
    state_dicts = [{} for _ in range(model_parallel)]

    for file_path in tqdm(weight_files, desc="Reading HF weights", unit="file"):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for raw_name in f.keys():
                name, _, dim = convert_name(raw_name)
                if name.startswith("mtp.") and (
                    "emb" in name or name.endswith("head.weight")
                ):
                    continue
                param: torch.Tensor = f.get_tensor(raw_name)
                for rank in range(model_parallel):
                    new_param = param
                    if "experts" in name and "shared_experts" not in name:
                        idx = int(name.split(".")[-3])
                        if (
                            idx < rank * n_local_experts
                            or idx >= (rank + 1) * n_local_experts
                        ):
                            continue
                    elif parallel_mode == "dp-ep":
                        if rank != 0:
                            continue
                    elif dim is not None:
                        if param.size(dim) % model_parallel != 0:
                            raise ValueError(
                                f"{raw_name}: dimension {dim} with size "
                                f"{param.size(dim)} is not divisible by "
                                f"model_parallel={model_parallel}"
                            )
                        shard_size = param.size(dim) // model_parallel
                        new_param = param.narrow(
                            dim, rank * shard_size, shard_size
                        ).contiguous()
                    state_dicts[rank][name] = new_param

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "ep" if parallel_mode == "dp-ep" else "mp"
    for rank in trange(model_parallel, desc="Writing shards", unit="rank"):
        names = list(state_dicts[rank].keys())
        for name in names:
            if name.endswith("wo_a.weight"):
                weight = state_dicts[rank][name]
                scale = state_dicts[rank].pop(name.replace("weight", "scale"))
                weight = (
                    weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float()
                    * scale[:, None, :, None].float()
                )
                state_dicts[rank][name] = weight.flatten(2, 3).flatten(0, 1).bfloat16()
            elif "experts" in name and state_dicts[rank][name].dtype == torch.int8:
                if expert_dtype == "fp8":
                    scale_name = name.replace("weight", "scale")
                    weight = state_dicts[rank].pop(name)
                    scale = state_dicts[rank].pop(scale_name)
                    state_dicts[rank][name], state_dicts[rank][scale_name] = (
                        cast_e2m1fn_to_e4m3fn(weight, scale)
                    )
                else:
                    state_dicts[rank][name] = state_dicts[rank][name].view(
                        torch.float4_e2m1fn_x2
                    )
        save_file(
            state_dicts[rank],
            str(output_dir / f"model{rank}-{suffix}{model_parallel}.safetensors"),
        )

    for file_name in ["tokenizer.json", "tokenizer_config.json"]:
        src = input_dir / file_name
        if src.exists():
            shutil.copyfile(src, output_dir / file_name)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="HF model folder")
    parser.add_argument("-o", "--output", required=True, help="Converted output folder")
    parser.add_argument(
        "-m",
        "--model-parallel",
        type=int,
        default=8,
        help="Number of shards. In dp-ep mode this is EP size; in mp mode this is demo-style MP size. Default: 8",
    )
    parser.add_argument(
        "--parallel-mode",
        choices=["dp-ep", "mp"],
        default="dp-ep",
        help=(
            "Output layout. dp-ep keeps dense weights replicated and shards only "
            "routed experts; mp matches the DeepSeek demo-style tensor/expert split. "
            "Default: dp-ep"
        ),
    )
    parser.add_argument(
        "--n-experts",
        type=int,
        default=None,
        help="Number of routed experts. Default: read n_routed_experts from config.json",
    )
    parser.add_argument(
        "--expert-dtype",
        choices=["fp4", "fp8"],
        default="fp8",
        help="Output dtype for packed expert weights. Default: fp8 for H200/DLEngine",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    n_experts = (
        args.n_experts if args.n_experts is not None else infer_n_experts(input_dir)
    )
    convert_weights(
        input_dir=input_dir,
        output_dir=output_dir,
        n_experts=n_experts,
        model_parallel=args.model_parallel,
        expert_dtype=args.expert_dtype,
        parallel_mode=args.parallel_mode,
    )


if __name__ == "__main__":
    main()
