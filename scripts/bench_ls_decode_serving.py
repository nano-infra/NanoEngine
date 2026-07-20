#!/usr/bin/env python3
"""CSV serving benchmark for NanoDeploy's LoongServe-style Decode scheduler."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ISSUE003_DIR = SCRIPT_DIR / "issue003"
if str(ISSUE003_DIR) not in sys.path:
    sys.path.insert(0, str(ISSUE003_DIR))

try:
    from ls_decode_issue001_profile import (
        add_formal_profile_arguments,
        clear_ray_proxy_env,
        formal_engine_kwargs,
        resolved_manifest,
    )
except ModuleNotFoundError:  # pragma: no cover - module-style invocation
    from scripts.ls_decode_issue001_profile import (
        add_formal_profile_arguments,
        clear_ray_proxy_env,
        formal_engine_kwargs,
        resolved_manifest,
    )


DEFAULT_DATASET = Path(
    "/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/"
    "sharegpt-4o-mixlong-0326/"
    "sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bench-serve a CSV workload with the LoongServe-style Decode-only "
            "multi-master scheduler."
        )
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3",
    )
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ray-address", default="10.102.243.60:8776")
    parser.add_argument("--master-address", default="10.102.243.60:29906")
    parser.add_argument("--duration-sec", type=float, default=360.0)
    parser.add_argument("--request-rate", type=float, default=20.0)
    parser.add_argument("--num-requests", type=int, default=0)
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Defaults to <output-jsonl stem>.manifest.json.",
    )
    parser.add_argument("--verbose-nanodeploy-logs", action="store_true")

    parser.add_argument("--attention-dp", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--attention-sp", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-recv-seqs", type=int, default=128)
    parser.add_argument("--loop-count", type=int, choices=range(1, 17), default=1)
    parser.add_argument("--max-model-len", type=int, default=1_000_000)
    parser.add_argument("--max-input-len", type=int, default=1_000_000)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1_024_000)
    parser.add_argument("--gpu-memory-limit-gb", type=float, default=141.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--segment-size", type=int, default=65_536)
    parser.add_argument("--cuda-graph-mode", choices=["full", "piecewise"], default="full")
    parser.add_argument("--enforce-eager", action="store_true")

    add_formal_profile_arguments(parser)
    parser.add_argument(
        "--ls-batch-per-master",
        type=int,
        default=64,
        help="Legacy constructor ABI value; recorded in the resolved manifest.",
    )
    args = parser.parse_args()

    if args.duration_sec <= 0 or args.request_rate <= 0 or args.burstiness <= 0:
        parser.error("duration-sec, request-rate, and burstiness must be positive")
    if args.num_requests < 0:
        parser.error("num-requests must be non-negative")
    if args.ls_batch_per_master <= 0:
        parser.error("--ls-batch-per-master must be > 0")
    if not args.csv_path.is_file():
        parser.error(f"CSV file not found: {args.csv_path}")
    if args.attention_sp != 8:
        parser.error("LoongServe-style formal benchmark requires attention-sp=8")
    if args.num_requests == 0:
        args.num_requests = int(round(args.duration_sec * args.request_rate))
    return args


def build_engine(args: argparse.Namespace, serving):
    ls_kwargs = formal_engine_kwargs(
        args.ls_max_num_ooe,
        ls_decode_batch_per_master=args.ls_batch_per_master,
    )
    return serving.LLM(
        args.model_path,
        enforce_eager=args.enforce_eager,
        cuda_graph_mode=args.cuda_graph_mode,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_memory_limit_gb=args.gpu_memory_limit_gb,
        master_address=args.master_address,
        ray_address=args.ray_address,
        mode="decode",
        dummy_prefill=True,
        dummy_weight=True,
        perfect_eplb=True,
        attention_dp=args.attention_dp,
        attention_sp=args.attention_sp,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=args.attention_dp * args.attention_sp,
        ffn_tp=1,
        max_num_seqs=args.max_num_seqs,
        max_num_recv_seqs=args.max_num_recv_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        loop_count=args.loop_count,
        scheduler_mode="centralized",
        segment_size=args.segment_size,
        kvcache_block_size=64,
        fixed_sp_size=0,
        sp_backend="hao_basic",
        use_dlslime_rpc=True,
        optimize_decode_block_table=True,
        enable_non_uniform_split=False,
        enable_dynamic_sp_size=False,
        use_new_decode_dynamic_sp_scheduler=False,
        dynamic_sp_size_strategy="legacy",
        **ls_kwargs,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest_json is not None:
        return args.manifest_json
    return args.output_jsonl.with_suffix(".manifest.json")


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _build_manifest(
    engine,
    args: argparse.Namespace,
    manifest_path: Path,
    cleared_proxy_keys: list[str],
) -> dict[str, Any]:
    resolved = resolved_manifest(
        engine.config,
        args.ls_max_num_ooe,
        expected_loop_count=args.loop_count,
    )
    num_blocks = int(engine.config.num_kvcache_blocks)
    pool_kv_tokens = (
        engine.config.attention_sp
        * num_blocks
        * engine.config.kvcache_block_size
    )
    return {
        "schema_version": 1,
        "status": "initialized",
        "created_utc": _utc_now(),
        "baseline": resolved,
        "initialization": {
            "engine_id": engine.engine_id,
            "scheduler_state_initialized_once": True,
            "warmup_policy": "disabled_for_formal_issue001",
            "warmup_requests": 0,
            "engine_reused_after_request_warmup": False,
            "cleared_http_proxy_env_keys": cleared_proxy_keys,
        },
        "topology_capacity": {
            "num_kvcache_blocks_per_rank": num_blocks,
            "pool_total_kv_tokens": pool_kv_tokens,
            "resolved_admission_max_tokens_per_pool": resolved[
                "ls_admission_max_tokens_per_pool"
            ],
        },
        "workload": {
            "kind": "issue001_csv",
            "csv_path": str(args.csv_path),
            "duration_sec": args.duration_sec,
            "request_rate": args.request_rate,
            "num_requests": args.num_requests,
            "burstiness": args.burstiness,
            "seed": args.seed,
            "max_input_len": args.max_input_len,
            "max_model_len": args.max_model_len,
            "max_tokens_contract": ">=1",
            "ignore_eos_required": True,
            "dp_assignment": "arrival_round_robin",
        },
        "engine_runtime": {
            "model_path": args.model_path,
            "ray_address": args.ray_address,
            "master_address": args.master_address,
            "max_num_seqs": args.max_num_seqs,
            "max_num_recv_seqs": args.max_num_recv_seqs,
            "loop_count": args.loop_count,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_limit_gb": args.gpu_memory_limit_gb,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "segment_size": args.segment_size,
            "cuda_graph_mode": args.cuda_graph_mode,
            "enforce_eager": args.enforce_eager,
        },
        "artifacts": {
            "completion_jsonl": str(args.output_jsonl),
            "manifest_json": str(manifest_path),
        },
        "result": None,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(args)
    cleared_proxy_keys = clear_ray_proxy_env()

    config = vars(args).copy()
    config["csv_path"] = str(args.csv_path)
    config["output_jsonl"] = str(args.output_jsonl)
    config["manifest_json"] = str(manifest_path)
    config["ffn_ep"] = args.attention_dp * args.attention_sp
    config["loop_count"] = args.loop_count
    print("LS_BENCH_CONFIG " + json.dumps(config, sort_keys=True), flush=True)

    if not args.verbose_nanodeploy_logs:
        from nanodeploy.logging import get_logger

        nanodeploy_logger = get_logger()
        nanodeploy_logger.setLevel(logging.WARNING)
        for handler in nanodeploy_logger.handlers:
            handler.setLevel(logging.WARNING)

    # Importing the legacy dataset/metrics helpers loads GPU model modules, so
    # keep it after CLI validation and proxy cleanup. In particular, --help and
    # CPU-only profile tests must not initialize CUDA.
    import bench_serving_overhead as serving

    engine = build_engine(args, serving)
    serving.print_model_config(engine)
    manifest = _build_manifest(engine, args, manifest_path, cleared_proxy_keys)
    _write_manifest(manifest_path, manifest)
    print("LS_BENCH_MANIFEST " + json.dumps(manifest, sort_keys=True), flush=True)

    dataset_args = argparse.Namespace(
        dataset="csv",
        csv_path=str(args.csv_path),
        num_requests=args.num_requests,
        max_input_len=args.max_input_len,
        max_model_len=args.max_model_len,
    )
    request_generator = serving.get_dataset_generator(dataset_args)
    arrival_times = serving.generate_arrival_times(
        args.num_requests, args.request_rate, args.burstiness
    )
    print(
        "LS_BENCH_ARRIVALS "
        + json.dumps(
            {
                "num_requests": args.num_requests,
                "target_duration_sec": args.duration_sec,
                "sampled_last_arrival_sec": float(arrival_times[-1]),
            }
        ),
        flush=True,
    )
    try:
        total_time, seq_map = serving.run_benchmark(
            engine, request_generator, arrival_times, args.num_requests
        )
        serving.calculate_and_print_metrics(
            total_time,
            seq_map,
            args.num_requests,
            itl_log_path=str(args.output_jsonl),
        )
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["completed_utc"] = _utc_now()
        manifest["result"] = {
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_manifest(manifest_path, manifest)
        raise

    completed = sum(
        1
        for sequence in seq_map.values()
        if sequence.metric is not None and sequence.metric.completion_time
    )
    manifest["status"] = "success"
    manifest["completed_utc"] = _utc_now()
    manifest["result"] = {
        "total_time_sec": total_time,
        "requests_sent": args.num_requests,
        "requests_completed": completed,
        "sampled_last_arrival_sec": float(arrival_times[-1]),
    }
    _write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
