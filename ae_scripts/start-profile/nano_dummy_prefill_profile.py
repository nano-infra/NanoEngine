#!/usr/bin/env python3
"""Collect NanoDeploy dummy-prefill profiler traces from an installed package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SP_SEQ_LENS: list[Any] = []
DEFAULT_RDMA_DEVICES = ",".join(f"mlx5_{index}" for index in range(8))
NETWORK_ENV_DEFAULTS = {
    "GLOO_SOCKET_IFNAME": "bond0",
    "NCCL_SOCKET_IFNAME": "bond0",
    "NCCL_IB_HCA": f"={DEFAULT_RDMA_DEVICES}",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_TC": "186",
    "SLIME_VISIBLE_DEVICES": DEFAULT_RDMA_DEVICES,
    "SLIME_GID_INDEX": "3",
    "SLIME_QP_NUM": "4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="dp32leastbatch",
        choices=(
            "dp32",
            "dp32leastbatch",
            "dp32leastcache",
            "dp4sp8",
            "dp4fixedsp8",
        ),
    )
    parser.add_argument(
        "--sp-backend",
        default="legacy_ll",
        choices=("legacy_ll", "hao_basic"),
    )
    parser.add_argument(
        "--sp-size-policy",
        default="legacy",
        choices=("legacy", "long_short"),
    )
    parser.add_argument("--enable-dynamic-sp-size", action="store_true")
    parser.add_argument("--segment-size", type=int, default=65_536)
    parser.add_argument("--long-request-sp-threshold", type=int, default=100_000)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--sp-seq-lens-file")
    parser.add_argument("--profiler-dir", required=True)
    parser.add_argument("--profiler-start-step", type=int, default=3)
    parser.add_argument("--profiling-step", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--loop-count", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-recv-seqs", type=int, default=70)
    parser.add_argument("--max-num-send-seqs", type=int, default=70)
    parser.add_argument("--max-model-len", type=int, default=1_000_000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()

    positive_fields = (
        "segment_size",
        "long_request_sp_threshold",
        "profiling_step",
        "max_tokens",
        "loop_count",
        "max_num_seqs",
        "max_num_recv_seqs",
        "max_num_send_seqs",
        "max_model_len",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.profiler_start_step < 0:
        parser.error("--profiler-start-step must be non-negative")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    return args


def get_config_params(config_name: str) -> dict[str, int]:
    if config_name in {"dp32", "dp32leastbatch", "dp32leastcache"}:
        return {
            "attention_dp": 32,
            "attention_sp": 1,
            "attention_tp": 1,
            "ffn_dp": 1,
            "ffn_ep": 32,
            "ffn_tp": 1,
        }
    if config_name in {"dp4sp8", "dp4fixedsp8"}:
        return {
            "attention_dp": 4,
            "attention_sp": 8,
            "attention_tp": 1,
            "ffn_dp": 1,
            "ffn_ep": 32,
            "ffn_tp": 1,
        }
    raise ValueError(f"unknown NanoDeploy profiler config: {config_name}")


def resolve_sp_policy(
    args: argparse.Namespace,
    attention_sp: int,
) -> tuple[str, str, dict[str, Any]]:
    common = {"segment_size": args.segment_size}
    if args.config == "dp4fixedsp8":
        return (
            "fixed_sp8",
            "fixed_sp8",
            {**common, "dynamic_sp_size_strategy": "legacy", "fixed_sp_size": 8},
        )
    if args.sp_size_policy == "long_short":
        if attention_sp <= 1:
            raise ValueError("long_short requires attention_sp > 1")
        return (
            "bucket",
            "bucket_deepseek_v3",
            {
                **common,
                "dynamic_sp_size_strategy": "bucket",
                "dynamic_sp_bucket_preset": "deepseek_v3",
                "fixed_sp_size": 0,
            },
        )
    if args.enable_dynamic_sp_size:
        return (
            "bucket",
            "bucket_deepseek_v3",
            {
                **common,
                "dynamic_sp_size_strategy": "bucket",
                "dynamic_sp_bucket_preset": "deepseek_v3",
                "fixed_sp_size": 0,
            },
        )
    return (
        "legacy",
        f"legacy_seg{args.segment_size}",
        {**common, "dynamic_sp_size_strategy": "legacy", "fixed_sp_size": 0},
    )


def infer_dataset_tag(input_path: str | None) -> str:
    if not input_path:
        return "embedded"
    path = Path(input_path)
    return path.parent.name or path.stem or "unknown"


def load_sp_seq_lens(input_path: str | None) -> list[Any]:
    if input_path is None:
        print("Using default embedded sp_seq_lens data", flush=True)
        return DEFAULT_SP_SEQ_LENS
    path = Path(input_path)
    print(f"Loading sp_seq_lens from: {path}", flush=True)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "sp_seq_lens" in payload:
        payload = payload["sp_seq_lens"]
    if not isinstance(payload, list):
        raise TypeError(f"sp_seq_lens must be a list, got {type(payload).__name__}")
    return payload


def require_installed_nanodeploy() -> str:
    spec = importlib.util.find_spec("nanodeploy")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "the nanodeploy package is not installed in the active Python environment"
        )
    return str(Path(spec.origin).resolve())


def configure_network_environment() -> None:
    """Apply the validated H200 RoCE defaults to driver and Ray workers."""
    for name, value in NETWORK_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)

    from nanodeploy.engine import ray_executor

    original_model_runner = ray_executor.ModelRunner

    class ModelRunnerWithNetworkEnv:
        @staticmethod
        def options(**kwargs):
            runtime_env = dict(kwargs.get("runtime_env") or {})
            env_vars = dict(runtime_env.get("env_vars") or {})
            env_vars.update(
                {name: os.environ[name] for name in NETWORK_ENV_DEFAULTS}
            )
            runtime_env["env_vars"] = env_vars
            kwargs["runtime_env"] = runtime_env
            return original_model_runner.options(**kwargs)

    ray_executor.ModelRunner = ModelRunnerWithNetworkEnv


def main() -> None:
    args = parse_args()
    package_origin = require_installed_nanodeploy()
    configure_network_environment()

    # Import only after parsing and validating the installed-package dependency.
    from nanodeploy import LLM, SamplingParams
    from nanodeploy.engine.sequence import Sequence

    config = get_config_params(args.config)
    policy_name, profiler_tag, policy_kwargs = resolve_sp_policy(
        args, config["attention_sp"]
    )
    dataset_tag = infer_dataset_tag(args.sp_seq_lens_file)
    args.profiler_dir = str(
        Path(args.profiler_dir) / dataset_tag / args.config / profiler_tag
    )

    selector = "LeastCache" if args.config == "dp32leastcache" else "LeastBatch"
    print("\n" + "=" * 60, flush=True)
    print(f"NanoDeploy package: {package_origin}", flush=True)
    print(f"Configuration: {args.config}", flush=True)
    print(f"  attention_dp={config['attention_dp']}", flush=True)
    print(f"  attention_sp={config['attention_sp']}", flush=True)
    print(f"  ffn_ep={config['ffn_ep']}", flush=True)
    print(f"  dataset={dataset_tag}", flush=True)
    print(f"  sp_backend={args.sp_backend}", flush=True)
    print(f"  sp_size_policy={policy_name}", flush=True)
    print(f"  segment_size={policy_kwargs['segment_size']}", flush=True)
    print(
        f"  enable_dynamic_sp_size={policy_kwargs['dynamic_sp_size_strategy'] == 'bucket'}",
        flush=True,
    )
    print(
        f"  dynamic_sp_size_strategy={policy_kwargs['dynamic_sp_size_strategy']}",
        flush=True,
    )
    print(f"  fixed_sp_size={policy_kwargs['fixed_sp_size']}", flush=True)
    print("  scheduler_arch=legacy_global", flush=True)
    print(f"  sp_master_selector={selector}", flush=True)
    print("=" * 60 + "\n", flush=True)

    print("Initializing LLM engine...", flush=True)
    print(f"  Profiler enabled: True", flush=True)
    print(f"  Profiler start step: {args.profiler_start_step}", flush=True)
    print(f"  Profiling steps: {args.profiling_step}", flush=True)
    print(f"  Profiler dir: {args.profiler_dir}", flush=True)

    decode = LLM(
        args.model_path,
        enforce_eager=False,
        **config,
        **policy_kwargs,
        mode="decode",
        master_address=args.master_address,
        ray_address=args.ray_address,
        dummy_prefill=True,
        dummy_weight=True,
        perfect_eplb=False,
        moe_routing_simulation_strategy="uniform_random",
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        loop_count=args.loop_count,
        max_num_recv_seqs=args.max_num_recv_seqs,
        kvcache_block_size=64,
        enable_profiler=True,
        profiler_start_step=args.profiler_start_step,
        profiling_step=args.profiling_step,
        profiler_dir=args.profiler_dir,
        enable_non_uniform_split=True,
        sp_backend=args.sp_backend,
        sp_master_selector=selector,
        scheduler_arch="legacy_global",
        routing_strategy=selector,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    sp_seq_lens = load_sp_seq_lens(args.sp_seq_lens_file)
    lengths = [
        seq_len
        for dp_group in sp_seq_lens
        for sp_rank_seqs in dp_group
        for seq_len in sp_rank_seqs
    ]
    print(f"Creating {len(lengths)} sequences for profiling", flush=True)
    sequences = [
        Sequence(
            np.random.randint(0, 10_001, size=seq_len).tolist(),
            sampling_params=sampling_params,
        )
        for seq_len in lengths
    ]
    decode.add_request(sequences)
    decode.generate()
    print("\n" + "=" * 60, flush=True)
    print("Profiling completed!", flush=True)
    print(f"Results saved to: {args.profiler_dir}", flush=True)
    print("=" * 60 + "\n", flush=True)


if __name__ == "__main__":
    main()
