# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline benchmark harness with Poisson arrivals using AsyncLLM directly."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, cast

# The migrated harness lives outside the vLLM source tree. The multi-node
# runner starts it after changing into --vllm-workdir, so make that checkout
# importable even when vLLM has not been installed as a site package.
_VLLM_WORKDIR = str(Path.cwd())
if _VLLM_WORKDIR not in sys.path:
    sys.path.insert(0, _VLLM_WORKDIR)

import numpy as np
import uvloop

from vllm.benchmarks.datasets import RandomDataset, SampleRequest
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import token_inputs
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.config.scheduler import SchedulerConfig

logger = init_logger(__name__)

DEFAULT_KV_TRANSFER_CONFIG = {
    "kv_connector": "DecodeBenchConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "fill_mean": 0.015,
        "fill_std": 0.0,
        "dummy_prefill": True,
        "dummy_output_token_id": 2,
    },
}

DEFAULT_COMPILATION_CONFIG = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
}

DEFAULT_ENV_VARS = {
    "VLLM_DEEP_GEMM_WARMUP": "skip",
    "VLLM_RANDOMIZE_DP_DUMMY_INPUTS": "1",
    "VLLM_MOE_ROUTING_SIMULATION_STRATEGY": "uniform_random",
}

DECODE_BENCH_EXTRA_CONFIG_KEYS = frozenset(
    {
        "fill_mean",
        "fill_std",
        "dummy_prefill",
        "dummy_output_token_id",
    }
)

TTFT_SEMANTICS_FIRST_REAL_TOKEN = "first_real_token_latency"
TTFT_SEMANTICS_DECODE_BENCH_DUMMY_PREFILL = (
    "decode_bench_dummy_prefill_pre_forward_schedule_boundary"
)
FRONTEND_TEARDOWN_HEARTBEAT_SEC = 15.0
FRONTEND_TEARDOWN_TIMEOUT_SEC = 60.0
GPU_KV_CACHE_CAPACITY_LOG_MARKER = "poisson_gpu_kv_cache_capacity"
DEFAULT_RESERVED_NULL_BLOCKS = 1
MAX_REQUESTS_CSV_ROWS = "csv_rows"
MaxRequestsArg = int | Literal["csv_rows"] | None


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer.")
    return parsed


def _max_requests_arg(value: str) -> int | Literal["csv_rows"]:
    lowered = value.strip().lower()
    if lowered in {"csv", "csv_rows", "all_csv_rows"}:
        return MAX_REQUESTS_CSV_ROWS
    return _non_negative_int(value)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def _positive_float_or_inf(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("Expected a positive float or inf.")
    return parsed


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonify(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _format_frontend_log_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if np.isfinite(value):
            return f"{value:.3f}"
        return str(value)
    return str(value)


def _format_frontend_log_fields(fields: Mapping[str, Any]) -> str:
    return " ".join(
        f"{key}={_format_frontend_log_value(value)}"
        for key, value in fields.items()
        if value is not None
    )


def _log_frontend_phase(
    phase: str,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    frontend_started_at_s: float,
    measured_requests: int | None = None,
    summary_written: bool | None = None,
    parquet_written: bool | None = None,
    exception: BaseException | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> None:
    fields: dict[str, Any] = {
        "phase": phase,
        "pid": os.getpid(),
        "output_dir": output_dir,
        "request_rate": getattr(args, "request_rate", None),
        "dispatch_policy": getattr(args, "data_parallel_dispatch_policy", None),
        "measured_requests": (
            measured_requests if measured_requests is not None else "pending"
        ),
        "dp_size": getattr(args, "data_parallel_size", None),
        "dp_size_local": getattr(args, "data_parallel_size_local", None),
        "save_parquet": getattr(args, "save_merged_parquet", False),
        "elapsed_s": time.monotonic() - frontend_started_at_s,
    }
    if summary_written is not None:
        fields["summary_written"] = summary_written
    if parquet_written is not None:
        fields["parquet_written"] = parquet_written
    if exception is not None:
        fields["exception_type"] = type(exception).__name__
    if extra_fields:
        fields.update(extra_fields)
    logger.info("poisson_frontend %s", _format_frontend_log_fields(fields))


def _log_gpu_kv_cache_capacity(
    async_llm: Any,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    frontend_started_at_s: float,
) -> None:
    cache_config = getattr(getattr(async_llm, "vllm_config", None),
                           "cache_config", None)
    if cache_config is None:
        logger.warning("%s unavailable: missing cache_config",
                       GPU_KV_CACHE_CAPACITY_LOG_MARKER)
        return

    num_gpu_blocks = getattr(cache_config, "num_gpu_blocks", None)
    block_size = getattr(cache_config, "block_size", None)
    if (not isinstance(num_gpu_blocks, int) or num_gpu_blocks <= 0
            or not isinstance(block_size, int) or block_size <= 0):
        logger.warning(
            "%s unavailable: num_gpu_blocks=%s block_size=%s",
            GPU_KV_CACHE_CAPACITY_LOG_MARKER,
            num_gpu_blocks,
            block_size,
        )
        return

    engine_ranks_managed = getattr(async_llm.engine_core, "engine_ranks_managed",
                                   ())
    managed_engine_count = max(len(engine_ranks_managed), 1)
    reserved_null_blocks = DEFAULT_RESERVED_NULL_BLOCKS
    usable_gpu_blocks = max(num_gpu_blocks - reserved_null_blocks, 0)
    total_tokens = usable_gpu_blocks * block_size

    fields = {
        "scope": "aggregate_managed_engines",
        "pid": os.getpid(),
        "output_dir": output_dir,
        "request_rate": getattr(args, "request_rate", None),
        "dp_size": getattr(args, "data_parallel_size", None),
        "dp_size_local": getattr(args, "data_parallel_size_local", None),
        "managed_engines": managed_engine_count,
        "num_gpu_blocks": num_gpu_blocks,
        "reserved_null_blocks": reserved_null_blocks,
        "usable_gpu_blocks": usable_gpu_blocks,
        "block_size": block_size,
        "total_tokens": total_tokens,
        "elapsed_s": time.monotonic() - frontend_started_at_s,
    }
    logger.info("%s %s", GPU_KV_CACHE_CAPACITY_LOG_MARKER,
                _format_frontend_log_fields(fields))


def _resolve_profile_prefix(
    args: argparse.Namespace,
    output_dir: Path,
) -> str:
    explicit_prefix = getattr(args, "profile_prefix", None)
    if explicit_prefix:
        return explicit_prefix
    return output_dir.name


async def _start_frontend_profile_if_requested(
    async_llm: Any,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    frontend_started_at_s: float,
    measured_requests: int | None,
) -> str | None:
    if not getattr(args, "profile_after_warmup", False):
        return None

    profile_prefix = _resolve_profile_prefix(args, output_dir)
    _log_frontend_phase(
        "profile_start_start",
        args=args,
        output_dir=output_dir,
        frontend_started_at_s=frontend_started_at_s,
        measured_requests=measured_requests,
        extra_fields={"profile_prefix": profile_prefix},
    )
    await async_llm.start_profile(profile_prefix=profile_prefix)
    _log_frontend_phase(
        "profile_start_done",
        args=args,
        output_dir=output_dir,
        frontend_started_at_s=frontend_started_at_s,
        measured_requests=measured_requests,
        extra_fields={"profile_prefix": profile_prefix},
    )
    return profile_prefix


async def _stop_frontend_profile(
    async_llm: Any,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    frontend_started_at_s: float,
    measured_requests: int | None,
    profile_prefix: str,
) -> None:
    _log_frontend_phase(
        "profile_stop_start",
        args=args,
        output_dir=output_dir,
        frontend_started_at_s=frontend_started_at_s,
        measured_requests=measured_requests,
        extra_fields={"profile_prefix": profile_prefix},
    )
    await async_llm.stop_profile()
    _log_frontend_phase(
        "profile_stop_done",
        args=args,
        output_dir=output_dir,
        frontend_started_at_s=frontend_started_at_s,
        measured_requests=measured_requests,
        extra_fields={"profile_prefix": profile_prefix},
    )


def _jsonify(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonify(asdict(value))

    if isinstance(value, Enum):
        return value.name

    if isinstance(value, Path):
        try:
            return str(value.expanduser().resolve())
        except Exception:
            return str(value)

    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]

    if isinstance(value, set):
        return sorted((_jsonify(item) for item in value), key=repr)

    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonify(value.model_dump())

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonify(value.to_dict())

    if hasattr(value, "to_json_string") and callable(value.to_json_string):
        json_text = value.to_json_string()
        try:
            return _jsonify(json.loads(json_text))
        except Exception:
            return json_text

    return value


def _interval_ms(start_s: float, end_s: float) -> float | None:
    if start_s == 0.0 or end_s == 0.0:
        return None
    return max(end_s - start_s, 0.0) * 1000.0


def _build_request_id(prefix: str, index: int) -> str:
    return f"{prefix}{index:06d}"


def default_cudagraph_capture_sizes(max_num_seqs: int) -> list[int]:
    if max_num_seqs <= 0:
        return []

    sizes = [1, 2, 4, 8, 16]
    if max_num_seqs <= 16:
        return [size for size in sizes if size <= max_num_seqs]

    sizes.extend(range(32, max_num_seqs + 1, 16))
    return sorted(set(size for size in sizes if size <= max_num_seqs))


def _apply_benchmark_env_defaults() -> None:
    for key, value in DEFAULT_ENV_VARS.items():
        os.environ.setdefault(key, value)


def _normalize_benchmark_kv_transfer_config(args: argparse.Namespace) -> None:
    kv_transfer_config = getattr(args, "kv_transfer_config", None)
    if not isinstance(kv_transfer_config, dict):
        return

    if kv_transfer_config.get("kv_connector") != "DecodeBenchConnector":
        return

    legacy_extra_keys = sorted(
        key for key in DECODE_BENCH_EXTRA_CONFIG_KEYS if key in kv_transfer_config
    )
    if not legacy_extra_keys:
        return

    normalized_config = dict(kv_transfer_config)
    extra_config = normalized_config.get("kv_connector_extra_config")
    extra_config = dict(extra_config) if isinstance(extra_config, dict) else {}
    for key in legacy_extra_keys:
        extra_config.setdefault(key, normalized_config.pop(key))

    normalized_config["kv_connector_extra_config"] = extra_config
    args.kv_transfer_config = normalized_config

    logger.warning(
        "DecodeBenchConnector benchmark config uses legacy top-level keys %s; "
        "moving them into kv_connector_extra_config.",
        legacy_extra_keys,
    )


def _apply_benchmark_arg_defaults(args: argparse.Namespace) -> None:
    _normalize_benchmark_kv_transfer_config(args)

    if args.cudagraph_capture_sizes is None:
        compilation_config = getattr(args, "compilation_config", None)
        existing_capture_sizes = None
        if isinstance(compilation_config, dict):
            existing_capture_sizes = compilation_config.get("cudagraph_capture_sizes")
        elif compilation_config is not None:
            existing_capture_sizes = getattr(
                compilation_config,
                "cudagraph_capture_sizes",
                None,
            )

        if existing_capture_sizes is not None:
            return

        max_num_seqs = args.max_num_seqs or SchedulerConfig.DEFAULT_MAX_NUM_SEQS
        args.cudagraph_capture_sizes = default_cudagraph_capture_sizes(max_num_seqs)


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(
            "--output-dir must be a directory path, but got an existing file: "
            f"{output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def connector_mode_from_config(kv_transfer_config: Any) -> str:
    if kv_transfer_config is None:
        return "none"

    if isinstance(kv_transfer_config, dict):
        connector_name = kv_transfer_config.get("kv_connector")
    else:
        connector_name = getattr(kv_transfer_config, "kv_connector", None)

    return "decode_bench" if connector_name == "DecodeBenchConnector" else "none"


def _get_config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _is_decode_bench_dummy_prefill(kv_transfer_config: Any) -> bool:
    if connector_mode_from_config(kv_transfer_config) != "decode_bench":
        return False

    extra_config = _get_config_value(kv_transfer_config, "kv_connector_extra_config")
    dummy_prefill = _get_config_value(extra_config, "dummy_prefill")
    if dummy_prefill is None:
        dummy_prefill = _get_config_value(kv_transfer_config, "dummy_prefill")
    return bool(dummy_prefill)


def _ttft_semantics_from_config(kv_transfer_config: Any) -> str:
    if _is_decode_bench_dummy_prefill(kv_transfer_config):
        return TTFT_SEMANTICS_DECODE_BENCH_DUMMY_PREFILL
    return TTFT_SEMANTICS_FIRST_REAL_TOKEN


def _ttft_definition_from_config(kv_transfer_config: Any) -> dict[str, str]:
    if _is_decode_bench_dummy_prefill(kv_transfer_config):
        return {
            "mode": "pre_forward_schedule_boundary",
            "semantics": TTFT_SEMANTICS_DECODE_BENCH_DUMMY_PREFILL,
            "applies_when": "DecodeBenchConnector && dummy_prefill",
            "formula": (
                "api_preprocess_ms + ipc_in_decode_ms + engine_preprocess_ms + "
                "(scheduled_ts - queued_ts) * 1000"
            ),
        }

    return {
        "mode": "first_real_token_latency",
        "semantics": TTFT_SEMANTICS_FIRST_REAL_TOKEN,
        "applies_when": "otherwise",
        "formula": "RequestOutput.metrics.first_token_latency * 1000",
    }


def build_poisson_arrival_deadlines_ns(
    num_requests: int,
    request_rate: float,
    seed: int,
) -> list[int]:
    if num_requests <= 0:
        return []

    if request_rate == float("inf"):
        return [0] * num_requests

    rng = np.random.default_rng(seed + 1)
    deltas_s = rng.exponential(1.0 / request_rate, size=num_requests)
    deadlines_s = np.cumsum(deltas_s)
    return [int(deadline * 1e9) for deadline in deadlines_s]


def load_length_requests(
    csv_path: str,
    tokenizer: Any,
    seed: int,
    request_id_prefix: str,
    csv_repeat: int = 1,
) -> list[SampleRequest]:
    if csv_repeat <= 0:
        raise ValueError("csv_repeat must be >= 1.")

    dataset = RandomDataset(
        random_seed=seed,
        random_csv_path=csv_path,
        disable_shuffle=True,
    )
    total_rows = len(dataset.csv_lengths or [])
    total_requests = total_rows * csv_repeat
    requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=total_requests,
        request_id_prefix="",
        no_oversample=False,
    )
    for index, request in enumerate(requests):
        request.request_id = _build_request_id(request_id_prefix, index)
    return requests


def resolve_csv_repeat(
    total_rows: int,
    warmup_requests: int,
    max_requests: int | None,
    csv_repeat: int | None,
) -> int:
    if total_rows <= 0:
        raise ValueError("CSV must contain at least one data row.")

    if csv_repeat is not None:
        if csv_repeat <= 0:
            raise ValueError("csv_repeat must be >= 1.")
        return csv_repeat

    if max_requests is None:
        return 1

    total_needed = warmup_requests + max_requests
    if total_needed <= 0:
        return 1

    return max(1, int(np.ceil(total_needed / total_rows)))


def split_warmup_and_measured_requests(
    requests: list[SampleRequest],
    warmup_requests: int,
    max_requests: int | None,
) -> tuple[list[SampleRequest], list[SampleRequest]]:
    warmup = requests[:warmup_requests]
    measured = requests[warmup_requests:]
    if max_requests is not None:
        measured = measured[:max_requests]
    return warmup, measured


def resolve_max_requests_against_csv(
    total_rows: int,
    max_requests: MaxRequestsArg,
) -> int | None:
    if max_requests is None:
        return None
    if max_requests == MAX_REQUESTS_CSV_ROWS:
        return total_rows
    return max_requests


def validate_request_lengths(
    requests: list[SampleRequest],
    max_model_len: int,
) -> None:
    too_long = [
        request for request in requests
        if request.prompt_len + request.expected_output_len > max_model_len
    ]
    if not too_long:
        return

    examples = ", ".join(
        f"{req.request_id}:{req.prompt_len}+{req.expected_output_len}"
        for req in too_long[:3]
    )
    raise ValueError(
        "Some requests exceed max_model_len="
        f"{max_model_len}. Example(s): {examples}"
    )


def validate_request_routing(
    requests: list[SampleRequest],
    routing_mode: str,
) -> None:
    if routing_mode != "explicit_rank_replay":
        return

    missing_rank = [
        request.request_id
        for request in requests
        if request.data_parallel_rank is None
    ]
    if not missing_rank:
        return

    examples = ", ".join(str(request_id) for request_id in missing_rank[:3])
    raise ValueError(
        "explicit_rank_replay requires every CSV row to specify "
        f"data_parallel_rank. Example request(s): {examples}"
    )


def _compute_pre_forward_ttft_ms(metrics: Any) -> float | None:
    if metrics is None:
        return None

    ttft_trace = getattr(metrics, "ttft_trace", None)
    if ttft_trace is None:
        return None

    queued_ts = getattr(metrics, "queued_ts", 0.0)
    scheduled_ts = getattr(metrics, "scheduled_ts", 0.0)
    if queued_ts == 0.0 or scheduled_ts == 0.0:
        return None

    preprocess_ns = (
        getattr(ttft_trace, "api_preprocess_ns", 0)
        + getattr(ttft_trace, "ipc_in_decode_ns", 0)
        + getattr(ttft_trace, "engine_preprocess_ns", 0)
    )
    queued_to_first_schedule_ms = max(scheduled_ts - queued_ts, 0.0) * 1000.0
    return float(preprocess_ns / 1e6 + queued_to_first_schedule_ms)


def build_success_record(
    request: SampleRequest,
    output: RequestOutput,
    submit_ts_ns: int,
    finish_ts_ns: int,
    kv_transfer_config: Any = None,
) -> dict[str, Any]:
    if not output.finished or not output.outputs:
        raise ValueError("Received a finished request without final outputs.")

    completion = output.outputs[0]
    metrics = output.metrics
    ttft_trace = None if metrics is None else metrics.ttft_trace
    kv_fill_ms = 0.0
    if ttft_trace is not None:
        kv_fill_ms = float(ttft_trace.first_batch_load_kv_ns / 1e6)

    queued_time_ms = None
    prefill_time_ms = None
    decode_time_ms = None
    inference_time_ms = None
    ttft_ms = None
    ttft_including_forward_ms = None
    ttft_post_schedule_to_first_token_ms = None
    if metrics is not None:
        queued_time_ms = _interval_ms(metrics.queued_ts, metrics.scheduled_ts)
        prefill_time_ms = _interval_ms(metrics.scheduled_ts, metrics.first_token_ts)
        decode_time_ms = _interval_ms(metrics.first_token_ts, metrics.last_token_ts)
        inference_time_ms = _interval_ms(metrics.scheduled_ts, metrics.last_token_ts)
        ttft_including_forward_ms = float(metrics.first_token_latency * 1000.0)
        ttft_ms = ttft_including_forward_ms
        if _is_decode_bench_dummy_prefill(kv_transfer_config):
            ttft_ms = _compute_pre_forward_ttft_ms(metrics)
            if ttft_ms is None:
                ttft_ms = ttft_including_forward_ms
            else:
                ttft_post_schedule_to_first_token_ms = max(
                    ttft_including_forward_ms - ttft_ms, 0.0
                )

    return {
        "request_id": request.request_id,
        "data_parallel_rank": request.data_parallel_rank,
        "submit_ts_ns": submit_ts_ns,
        "finish_ts_ns": finish_ts_ns,
        "e2e_ms": float((finish_ts_ns - submit_ts_ns) / 1e6),
        "ttft_ms": ttft_ms,
        "ttft_including_forward_ms": ttft_including_forward_ms,
        "ttft_post_schedule_to_first_token_ms": (
            ttft_post_schedule_to_first_token_ms
        ),
        "queued_time_ms": queued_time_ms,
        "kv_fill_ms": kv_fill_ms,
        "prefill_time_ms": prefill_time_ms,
        "decode_time_ms": decode_time_ms,
        "inference_time_ms": inference_time_ms,
        "prompt_len": request.prompt_len,
        "expected_output_len": request.expected_output_len,
        "actual_output_tokens": len(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "num_cached_tokens": int(output.num_cached_tokens or 0),
        "is_error": False,
        "error_message": None,
    }


def build_error_record(
    request: SampleRequest,
    submit_ts_ns: int,
    finish_ts_ns: int,
    error: Exception | str,
) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "data_parallel_rank": request.data_parallel_rank,
        "submit_ts_ns": submit_ts_ns,
        "finish_ts_ns": finish_ts_ns,
        "e2e_ms": float((finish_ts_ns - submit_ts_ns) / 1e6),
        "ttft_ms": None,
        "ttft_including_forward_ms": None,
        "ttft_post_schedule_to_first_token_ms": None,
        "queued_time_ms": None,
        "kv_fill_ms": 0.0,
        "prefill_time_ms": None,
        "decode_time_ms": None,
        "inference_time_ms": None,
        "prompt_len": request.prompt_len,
        "expected_output_len": request.expected_output_len,
        "actual_output_tokens": 0,
        "finish_reason": None,
        "num_cached_tokens": 0,
        "is_error": True,
        "error_message": str(error),
    }


def _metric_summary(records: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = [
        float(record[key])
        for record in records
        if not record["is_error"] and record.get(key) is not None
    ]
    if not values:
        return None

    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


def build_summary(
    records: list[dict[str, Any]],
    connector_mode: str,
    ttft_semantics: str,
    first_submit_ts_ns: int | None,
    last_finish_ts_ns: int | None,
) -> dict[str, Any]:
    total_requests = len(records)
    failed_requests = sum(1 for record in records if record["is_error"])
    successful_requests = total_requests - failed_requests

    runtime_s = 0.0
    if first_submit_ts_ns is not None and last_finish_ts_ns is not None:
        runtime_s = max(last_finish_ts_ns - first_submit_ts_ns, 0) / 1e9

    summary = {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "failure_ratio": (
            float(failed_requests / total_requests) if total_requests else 0.0
        ),
        "benchmark_runtime_s": runtime_s,
        "achieved_request_throughput_rps": (
            float(total_requests / runtime_s) if runtime_s > 0 else 0.0
        ),
        "connector_mode": connector_mode,
        "ttft_semantics": ttft_semantics,
        "e2e_ms": _metric_summary(records, "e2e_ms"),
        "ttft_ms": _metric_summary(records, "ttft_ms"),
        "ttft_including_forward_ms": _metric_summary(
            records, "ttft_including_forward_ms"
        ),
        "ttft_post_schedule_to_first_token_ms": _metric_summary(
            records, "ttft_post_schedule_to_first_token_ms"
        ),
        "queued_time_ms": _metric_summary(records, "queued_time_ms"),
    }
    if connector_mode == "decode_bench":
        summary["kv_fill_ms"] = _metric_summary(records, "kv_fill_ms")
    return summary


@dataclass
class RunRecorder:
    requests_path: Path
    progress_log_interval: int
    records: list[dict[str, Any]]
    first_submit_ts_ns: int | None = None
    last_finish_ts_ns: int | None = None
    completed_requests: int = 0

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()
        self._fh = self.requests_path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._fh.close()

    async def note_submit(self, submit_ts_ns: int) -> None:
        async with self._lock:
            if self.first_submit_ts_ns is None or submit_ts_ns < self.first_submit_ts_ns:
                self.first_submit_ts_ns = submit_ts_ns

    async def append(self, record: dict[str, Any]) -> None:
        async with self._lock:
            self._fh.write(json.dumps(record, sort_keys=True) + "\n")
            self._fh.flush()
            self.records.append(record)
            self.completed_requests += 1
            finish_ts_ns = int(record["finish_ts_ns"])
            if self.last_finish_ts_ns is None or finish_ts_ns > self.last_finish_ts_ns:
                self.last_finish_ts_ns = finish_ts_ns
            if (
                self.progress_log_interval > 0
                and self.completed_requests % self.progress_log_interval == 0
            ):
                logger.info(
                    "Completed %d measured requests.",
                    self.completed_requests,
                )


def _enforce_harness_observability(args: argparse.Namespace) -> None:
    if getattr(args, "disable_log_stats", False):
        raise ValueError(
            "offline_poisson_harness requires log stats. "
            "Remove --disable-log-stats."
        )
    if getattr(args, "skip_tokenizer_init", False):
        raise ValueError(
            "offline_poisson_harness length_csv mode requires tokenizer init. "
            "Remove --skip-tokenizer-init."
        )

    args.disable_log_stats = False
    args.enable_logging_step_timing_details = True
    args.enable_graph_replay_timing = True
    args.logging_step_timing_interval = 10
    args.enable_logging_ttft_timing_details = True
    args.logging_ttft_timing_interval = 1


def _validate_frontend_args(args: argparse.Namespace) -> None:
    if args.csv_format != "length_csv":
        raise NotImplementedError("Only --csv-format=length_csv is implemented.")
    if args.routing_mode not in {"internal_dplb", "explicit_rank_replay"}:
        raise NotImplementedError(
            "Unsupported --routing-mode="
            f"{args.routing_mode!r}."
        )
    if args.pause_before_profile and not args.profile_after_warmup:
        raise ValueError(
            "--pause-before-profile requires --profile-after-warmup so the "
            "frontend can resume generation immediately after profiling starts."
        )


def _build_sampling_params(output_len: int) -> SamplingParams:
    return SamplingParams(
        max_tokens=output_len,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )


async def _submit_one_request(
    *,
    engine: Any,
    request: SampleRequest,
    recorder: RunRecorder | None,
    inflight: set[asyncio.Task[None]],
    kv_transfer_config: Any = None,
) -> None:
    assert request.request_id is not None
    assert isinstance(request.prompt, list)

    submit_ts_ns = time.time_ns()
    if recorder is not None:
        await recorder.note_submit(submit_ts_ns)

    try:
        collector = await engine.add_request(
            request_id=request.request_id,
            prompt=token_inputs(prompt_token_ids=cast(list[int], request.prompt)),
            params=_build_sampling_params(request.expected_output_len),
            arrival_time=submit_ts_ns / 1e9,
            data_parallel_rank=request.data_parallel_rank,
        )
    except Exception as exc:
        if recorder is None:
            raise
        finish_ts_ns = time.time_ns()
        await recorder.append(
            build_error_record(request, submit_ts_ns, finish_ts_ns, exc)
        )
        return

    async def wait_for_completion() -> None:
        try:
            final_output = await collector.get()
            finish_ts_ns = time.time_ns()
            if not isinstance(final_output, RequestOutput):
                raise TypeError(
                    f"Expected RequestOutput, got {type(final_output).__name__}."
                )
            record = build_success_record(
                request=request,
                output=final_output,
                submit_ts_ns=submit_ts_ns,
                finish_ts_ns=finish_ts_ns,
                kv_transfer_config=kv_transfer_config,
            )
        except Exception as exc:
            if recorder is None:
                raise
            finish_ts_ns = time.time_ns()
            record = build_error_record(request, submit_ts_ns, finish_ts_ns, exc)

        if recorder is not None:
            await recorder.append(record)

    task = asyncio.create_task(wait_for_completion())
    inflight.add(task)
    task.add_done_callback(inflight.discard)


async def _run_warmup(engine: Any, requests: list[SampleRequest]) -> None:
    if not requests:
        return

    logger.info("Running %d warmup request(s).", len(requests))
    inflight: set[asyncio.Task[None]] = set()
    for request in requests:
        await _submit_one_request(
            engine=engine,
            request=request,
            recorder=None,
            inflight=inflight,
        )
    while inflight:
        await asyncio.gather(*tuple(inflight))


def _write_parquet_if_requested(
    output_dir: Path,
    records: list[dict[str, Any]],
    enabled: bool,
) -> None:
    if not enabled:
        return

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "--save-merged-parquet requires pandas to be installed."
        ) from exc

    parquet_path = output_dir / "merged_requests.parquet"
    pd.DataFrame(records).to_parquet(parquet_path, index=False)


async def _request_clean_cluster_shutdown(
    async_llm: Any,
    *,
    frontend_args: argparse.Namespace | None = None,
    output_dir: Path | None = None,
    frontend_started_at_s: float | None = None,
    measured_requests: int | None = None,
) -> bool:
    def log_phase(
        phase: str,
        *,
        exception: BaseException | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        if (frontend_args is not None and output_dir is not None
                and frontend_started_at_s is not None):
            _log_frontend_phase(
                phase,
                args=frontend_args,
                output_dir=output_dir,
                frontend_started_at_s=frontend_started_at_s,
                measured_requests=measured_requests,
                exception=exception,
                extra_fields=extra_fields,
            )

    parallel_config = getattr(async_llm.vllm_config, "parallel_config", None)
    if parallel_config is None:
        log_phase(
            "clean_cluster_shutdown_skipped",
            extra_fields={"reason": "missing_parallel_config"},
        )
        return True

    # In pure internal DPLB multi-node runs, the frontend manages both local
    # and remote EngineCore sockets. Ask every EngineCore to enter its normal
    # shutdown path before tearing down the local client resources so remote
    # nodes do not see rank-0 disappear out from under torch.distributed.
    has_remote_dp_engines = (
        parallel_config.data_parallel_size > parallel_config.data_parallel_size_local
    )
    if not has_remote_dp_engines:
        log_phase(
            "clean_cluster_shutdown_skipped",
            extra_fields={"reason": "local_only_run"},
        )
        return True

    engine_core = getattr(async_llm, "engine_core", None)
    call_utility_async = getattr(engine_core, "call_utility_async", None)
    if not callable(call_utility_async):
        log_phase(
            "clean_cluster_shutdown_skipped",
            extra_fields={"reason": "missing_call_utility_async"},
        )
        logger.warning(
            "Skipping clean multi-node shutdown request because EngineCore "
            "client does not expose call_utility_async()."
        )
        return True

    log_phase(
        "clean_cluster_shutdown_request_start",
        extra_fields={
            "remote_dp": True,
            "heartbeat_s": FRONTEND_TEARDOWN_HEARTBEAT_SEC,
            "max_wait_s": FRONTEND_TEARDOWN_TIMEOUT_SEC,
        },
    )
    logger.info("Requesting shutdown across all managed EngineCore processes.")
    wait_started_at_s = time.monotonic()
    deadline = wait_started_at_s + FRONTEND_TEARDOWN_TIMEOUT_SEC
    wait_task = asyncio.create_task(
        call_utility_async("request_shutdown_and_wait"),
    )
    try:
        # Request the EngineCore busy loops to exit gracefully and wait for each
        # managed EngineCore to acknowledge that its shutdown path completed
        # before tearing down the local client resources.
        while True:
            now = time.monotonic()
            elapsed_s = now - wait_started_at_s
            remaining_s = deadline - now
            if remaining_s <= 0:
                log_phase(
                    "clean_cluster_shutdown_timeout",
                    extra_fields={
                        "wait_elapsed_s": elapsed_s,
                        "max_wait_s": FRONTEND_TEARDOWN_TIMEOUT_SEC,
                    },
                )
                logger.warning(
                    "Timed out waiting %.1fs for clean multi-node shutdown "
                    "acknowledgement; falling back to forced local cleanup.",
                    FRONTEND_TEARDOWN_TIMEOUT_SEC,
                )
                wait_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await wait_task
                return False
            try:
                await asyncio.wait_for(
                    asyncio.shield(wait_task),
                    timeout=min(FRONTEND_TEARDOWN_HEARTBEAT_SEC, remaining_s),
                )
            except asyncio.TimeoutError:
                elapsed_s = time.monotonic() - wait_started_at_s
                log_phase(
                    "clean_cluster_shutdown_waiting",
                    extra_fields={
                        "wait_elapsed_s": elapsed_s,
                        "max_wait_s": FRONTEND_TEARDOWN_TIMEOUT_SEC,
                    },
                )
                continue
            break

        log_phase(
            "clean_cluster_shutdown_request_done",
            extra_fields={
                "wait_elapsed_s": time.monotonic() - wait_started_at_s,
            },
        )
        return True
    except Exception as exc:
        log_phase(
            "clean_cluster_shutdown_failed",
            exception=exc,
            extra_fields={
                "wait_elapsed_s": time.monotonic() - wait_started_at_s,
                "max_wait_s": FRONTEND_TEARDOWN_TIMEOUT_SEC,
            },
        )
        logger.warning(
            "Failed to request clean multi-node shutdown before local cleanup.",
            exc_info=True,
        )
        wait_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await wait_task
        return False


async def _shutdown_frontend_async_llm(
    async_llm: Any,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    frontend_started_at_s: float,
    measured_requests: int | None,
    summary_written: bool,
    parquet_written: bool,
) -> None:
    # Mark the frontend as intentionally shutting down before asking managed
    # EngineCore processes to exit so local liveness monitors do not misclassify
    # the expected teardown as an engine failure.
    begin_shutdown = getattr(async_llm, "begin_shutdown", None)
    if callable(begin_shutdown):
        _log_frontend_phase(
            "begin_shutdown_start",
            args=args,
            output_dir=output_dir,
            frontend_started_at_s=frontend_started_at_s,
            measured_requests=measured_requests,
            summary_written=summary_written,
            parquet_written=parquet_written,
        )
        begin_shutdown()
        _log_frontend_phase(
            "begin_shutdown_done",
            args=args,
            output_dir=output_dir,
            frontend_started_at_s=frontend_started_at_s,
            measured_requests=measured_requests,
            summary_written=summary_written,
            parquet_written=parquet_written,
        )

    _log_frontend_phase(
        "clean_cluster_shutdown_start",
        args=args,
        output_dir=output_dir,
        frontend_started_at_s=frontend_started_at_s,
        measured_requests=measured_requests,
        summary_written=summary_written,
        parquet_written=parquet_written,
    )
    graceful_shutdown = await _request_clean_cluster_shutdown(
        async_llm,
        frontend_args=args,
        output_dir=output_dir,
        frontend_started_at_s=frontend_started_at_s,
        measured_requests=measured_requests,
    )
    _log_frontend_phase(
        "clean_cluster_shutdown_done",
        args=args,
        output_dir=output_dir,
        frontend_started_at_s=frontend_started_at_s,
        measured_requests=measured_requests,
        summary_written=summary_written,
        parquet_written=parquet_written,
        extra_fields={"graceful": graceful_shutdown},
    )

    shutdown_timeout: float | None = None
    if not graceful_shutdown:
        shutdown_timeout = 0.0
        _log_frontend_phase(
            "clean_cluster_shutdown_fallback",
            args=args,
            output_dir=output_dir,
            frontend_started_at_s=frontend_started_at_s,
            measured_requests=measured_requests,
            summary_written=summary_written,
            parquet_written=parquet_written,
            extra_fields={"engine_shutdown_timeout_s": shutdown_timeout},
        )

    _log_frontend_phase(
        "async_llm_shutdown_start",
        args=args,
        output_dir=output_dir,
        frontend_started_at_s=frontend_started_at_s,
        measured_requests=measured_requests,
        summary_written=summary_written,
        parquet_written=parquet_written,
        extra_fields={
            "graceful": graceful_shutdown,
            "timeout_s": shutdown_timeout,
        },
    )
    async_llm.shutdown(timeout=shutdown_timeout)
    _log_frontend_phase(
        "async_llm_shutdown_done",
        args=args,
        output_dir=output_dir,
        frontend_started_at_s=frontend_started_at_s,
        measured_requests=measured_requests,
        summary_written=summary_written,
        parquet_written=parquet_written,
        extra_fields={
            "graceful": graceful_shutdown,
            "timeout_s": shutdown_timeout,
        },
    )


async def run_frontend(args: argparse.Namespace) -> None:
    _validate_frontend_args(args)
    _enforce_harness_observability(args)
    _apply_benchmark_env_defaults()
    _apply_benchmark_arg_defaults(args)

    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs.from_cli_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    _prepare_output_dir(output_dir)
    frontend_started_at_s = time.monotonic()

    async_llm: Any | None = None
    measured_requests_count: int | None = None
    active_profile_prefix: str | None = None
    generation_paused = False
    summary_written = False
    parquet_written = False
    recorder = RunRecorder(
        requests_path=output_dir / "requests.jsonl",
        progress_log_interval=args.progress_log_interval,
        records=[],
    )
    _log_frontend_phase(
        "frontend_start",
        args=args,
        output_dir=output_dir,
        frontend_started_at_s=frontend_started_at_s,
    )

    try:
        async_llm = AsyncLLM.from_engine_args(
            engine_args,
            usage_context=UsageContext.OPENAI_API_SERVER,
        )
        _log_frontend_phase(
            "async_llm_created",
            args=args,
            output_dir=output_dir,
            frontend_started_at_s=frontend_started_at_s,
        )
        _log_gpu_kv_cache_capacity(
            async_llm,
            args=args,
            output_dir=output_dir,
            frontend_started_at_s=frontend_started_at_s,
        )
        tokenizer = async_llm.renderer.tokenizer
        if tokenizer is None:
            raise RuntimeError("AsyncLLM did not initialize a tokenizer.")

        request_id_prefix = args.request_id_prefix or f"r{args.seed}-"
        dataset = RandomDataset(
            random_seed=args.seed,
            random_csv_path=args.input_csv,
            disable_shuffle=True,
        )
        total_rows = len(dataset.csv_lengths or [])
        effective_max_requests = resolve_max_requests_against_csv(
            total_rows=total_rows,
            max_requests=args.max_requests,
        )
        csv_repeat = resolve_csv_repeat(
            total_rows=total_rows,
            warmup_requests=args.warmup_requests,
            max_requests=effective_max_requests,
            csv_repeat=args.csv_repeat,
        )
        all_requests = load_length_requests(
            csv_path=args.input_csv,
            tokenizer=tokenizer,
            seed=args.seed,
            request_id_prefix=request_id_prefix,
            csv_repeat=csv_repeat,
        )
        warmup_requests, measured_requests = split_warmup_and_measured_requests(
            all_requests,
            warmup_requests=args.warmup_requests,
            max_requests=effective_max_requests,
        )
        if not measured_requests:
            raise ValueError(
                "No measured requests remain after warmup/max-requests. "
                "Increase --csv-repeat, reduce --warmup-requests, or reduce "
                "--max-requests."
            )
        measured_requests_count = len(measured_requests)

        validate_request_lengths(measured_requests, async_llm.model_config.max_model_len)
        validate_request_lengths(warmup_requests, async_llm.model_config.max_model_len)
        validate_request_routing(all_requests, args.routing_mode)

        connector_mode = connector_mode_from_config(args.kv_transfer_config)
        ttft_semantics = _ttft_semantics_from_config(args.kv_transfer_config)
        run_meta = {
            "model": args.model,
            "topology": {
                "dp_size": args.data_parallel_size,
                "dp_size_local": args.data_parallel_size_local,
                "tp_size": args.tensor_parallel_size,
                "dcp_size": args.decode_context_parallel_size,
                "ep_enabled": bool(args.enable_expert_parallel),
            },
            "routing_mode": args.routing_mode,
            "dispatch_policy": args.data_parallel_dispatch_policy,
            "connector_mode": connector_mode,
            "csv_path": str(Path(args.input_csv).expanduser().resolve()),
            "csv_rows": total_rows,
            "csv_repeat": csv_repeat,
            "loaded_requests": len(all_requests),
            "request_rate": args.request_rate,
            "arrival_process": args.arrival_process,
            "pause_before_profile": bool(args.pause_before_profile),
            "pause_before_measured": bool(args.pause_before_measured),
            "seed": args.seed,
            "ttft_definition": _ttft_definition_from_config(
                args.kv_transfer_config
            ),
            "warmup_requests": len(warmup_requests),
            "measured_requests": len(measured_requests),
            "effective_max_requests": effective_max_requests,
            "benchmark_start_time": datetime.now().astimezone().isoformat(),
            "engine_args": {
                key: value
                for key, value in vars(args).items()
                if key not in {"output_dir"}
            },
        }
        _json_dump(output_dir / "run_meta.json", run_meta)

        await _run_warmup(async_llm, warmup_requests)
        _log_frontend_phase(
            "warmup_done",
            args=args,
            output_dir=output_dir,
            frontend_started_at_s=frontend_started_at_s,
            measured_requests=measured_requests_count,
            extra_fields={"warmup_requests": len(warmup_requests)},
        )
        inflight: set[asyncio.Task[None]] = set()
        pause_before_measured = (
            args.pause_before_profile or args.pause_before_measured
        )
        if pause_before_measured:
            pause_phase = (
                "pause_before_profile"
                if args.pause_before_profile
                else "pause_before_measured"
            )
            _log_frontend_phase(
                f"{pause_phase}_start",
                args=args,
                output_dir=output_dir,
                frontend_started_at_s=frontend_started_at_s,
                measured_requests=measured_requests_count,
            )
            logger.info(
                "Pausing generation with mode=keep before measured submission."
            )
            await async_llm.pause_generation(mode="keep", clear_cache=False)
            generation_paused = True
            _log_frontend_phase(
                f"{pause_phase}_done",
                args=args,
                output_dir=output_dir,
                frontend_started_at_s=frontend_started_at_s,
                measured_requests=measured_requests_count,
                extra_fields={"submission_mode": "paused_burst"},
            )
            logger.info(
                "Submitting %d measured request(s) while generation is paused.",
                len(measured_requests),
            )
            for request in measured_requests:
                await _submit_one_request(
                    engine=async_llm,
                    request=request,
                    recorder=recorder,
                    inflight=inflight,
                    kv_transfer_config=args.kv_transfer_config,
                )
            _log_frontend_phase(
                "submission_done",
                args=args,
                output_dir=output_dir,
                frontend_started_at_s=frontend_started_at_s,
                measured_requests=measured_requests_count,
                extra_fields={"submission_mode": "paused_burst"},
            )
            active_profile_prefix = await _start_frontend_profile_if_requested(
                async_llm,
                args=args,
                output_dir=output_dir,
                frontend_started_at_s=frontend_started_at_s,
                measured_requests=measured_requests_count,
            )
            resume_phase = (
                "resume_after_profile_start"
                if args.pause_before_profile
                else "resume_after_measured_submission"
            )
            _log_frontend_phase(
                resume_phase,
                args=args,
                output_dir=output_dir,
                frontend_started_at_s=frontend_started_at_s,
                measured_requests=measured_requests_count,
            )
            resume_reason = (
                "profiler start"
                if args.pause_before_profile
                else "measured submission"
            )
            logger.info("Resuming generation after %s.", resume_reason)
            await async_llm.resume_generation()
            generation_paused = False
        else:
            active_profile_prefix = await _start_frontend_profile_if_requested(
                async_llm,
                args=args,
                output_dir=output_dir,
                frontend_started_at_s=frontend_started_at_s,
                measured_requests=measured_requests_count,
            )

            arrival_deadlines_ns = build_poisson_arrival_deadlines_ns(
                num_requests=len(measured_requests),
                request_rate=args.request_rate,
                seed=args.seed,
            )

            logger.info("Submitting %d measured request(s).", len(measured_requests))
            scheduling_start_ns = time.perf_counter_ns()
            for index, request in enumerate(measured_requests):
                deadline_ns = scheduling_start_ns + arrival_deadlines_ns[index]
                sleep_ns = deadline_ns - time.perf_counter_ns()
                if sleep_ns > 0:
                    await asyncio.sleep(sleep_ns / 1e9)
                await _submit_one_request(
                    engine=async_llm,
                    request=request,
                    recorder=recorder,
                    inflight=inflight,
                    kv_transfer_config=args.kv_transfer_config,
                )
            _log_frontend_phase(
                "submission_done",
                args=args,
                output_dir=output_dir,
                frontend_started_at_s=frontend_started_at_s,
                measured_requests=measured_requests_count,
            )

        while inflight:
            await asyncio.gather(*tuple(inflight))
        _log_frontend_phase(
            "inflight_drained",
            args=args,
            output_dir=output_dir,
            frontend_started_at_s=frontend_started_at_s,
            measured_requests=measured_requests_count,
        )

        summary = build_summary(
            records=recorder.records,
            connector_mode=connector_mode,
            ttft_semantics=ttft_semantics,
            first_submit_ts_ns=recorder.first_submit_ts_ns,
            last_finish_ts_ns=recorder.last_finish_ts_ns,
        )
        _log_frontend_phase(
            "summary_write_start",
            args=args,
            output_dir=output_dir,
            frontend_started_at_s=frontend_started_at_s,
            measured_requests=measured_requests_count,
        )
        _json_dump(output_dir / "summary.json", summary)
        summary_written = True
        _log_frontend_phase(
            "summary_write_done",
            args=args,
            output_dir=output_dir,
            frontend_started_at_s=frontend_started_at_s,
            measured_requests=measured_requests_count,
            summary_written=summary_written,
        )
        if args.save_merged_parquet:
            _log_frontend_phase(
                "parquet_write_start",
                args=args,
                output_dir=output_dir,
                frontend_started_at_s=frontend_started_at_s,
                measured_requests=measured_requests_count,
                summary_written=summary_written,
            )
        _write_parquet_if_requested(
            output_dir=output_dir,
            records=recorder.records,
            enabled=args.save_merged_parquet,
        )
        if args.save_merged_parquet:
            parquet_written = True
            _log_frontend_phase(
                "parquet_write_done",
                args=args,
                output_dir=output_dir,
                frontend_started_at_s=frontend_started_at_s,
                measured_requests=measured_requests_count,
                summary_written=summary_written,
                parquet_written=parquet_written,
            )
    finally:
        active_exception = sys.exc_info()[1]
        profile_stop_exception: BaseException | None = None
        _log_frontend_phase(
            "teardown_enter",
            args=args,
            output_dir=output_dir,
            frontend_started_at_s=frontend_started_at_s,
            measured_requests=measured_requests_count,
            summary_written=summary_written,
            parquet_written=parquet_written,
            exception=active_exception,
        )
        if async_llm is not None and generation_paused:
            try:
                logger.info("Resuming generation during teardown cleanup.")
                await async_llm.resume_generation()
            except Exception:
                logger.warning(
                    "Failed to resume generation during frontend teardown cleanup.",
                    exc_info=True,
                )
        if async_llm is not None and active_profile_prefix is not None:
            try:
                await _stop_frontend_profile(
                    async_llm,
                    args=args,
                    output_dir=output_dir,
                    frontend_started_at_s=frontend_started_at_s,
                    measured_requests=measured_requests_count,
                    profile_prefix=active_profile_prefix,
                )
            except Exception as exc:
                profile_stop_exception = exc
                _log_frontend_phase(
                    "profile_stop_failed",
                    args=args,
                    output_dir=output_dir,
                    frontend_started_at_s=frontend_started_at_s,
                    measured_requests=measured_requests_count,
                    summary_written=summary_written,
                    parquet_written=parquet_written,
                    exception=exc,
                    extra_fields={"profile_prefix": active_profile_prefix},
                )
                if active_exception is not None:
                    logger.warning(
                        "Failed to stop the measured-phase profiler during "
                        "frontend teardown.",
                        exc_info=True,
                    )
        effective_exception = active_exception or profile_stop_exception
        _log_frontend_phase(
            "recorder_close_start",
            args=args,
            output_dir=output_dir,
            frontend_started_at_s=frontend_started_at_s,
            measured_requests=measured_requests_count,
            summary_written=summary_written,
            parquet_written=parquet_written,
        )
        recorder.close()
        _log_frontend_phase(
            "recorder_close_done",
            args=args,
            output_dir=output_dir,
            frontend_started_at_s=frontend_started_at_s,
            measured_requests=measured_requests_count,
            summary_written=summary_written,
            parquet_written=parquet_written,
        )
        if async_llm is not None:
            await _shutdown_frontend_async_llm(
                async_llm,
                args=args,
                output_dir=output_dir,
                frontend_started_at_s=frontend_started_at_s,
                measured_requests=measured_requests_count,
                summary_written=summary_written,
                parquet_written=parquet_written,
            )
        _log_frontend_phase(
            "frontend_exit",
            args=args,
            output_dir=output_dir,
            frontend_started_at_s=frontend_started_at_s,
            measured_requests=measured_requests_count,
            summary_written=summary_written,
            parquet_written=parquet_written,
            exception=effective_exception,
        )
        if active_exception is None and profile_stop_exception is not None:
            raise profile_stop_exception


def run_headless_engine(args: argparse.Namespace) -> None:
    _enforce_harness_observability(args)
    _apply_benchmark_env_defaults()
    _apply_benchmark_arg_defaults(args)

    from vllm.utils.network_utils import get_tcp_uri
    from vllm.v1.engine.utils import CoreEngineProcManager
    from vllm.v1.executor import Executor
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    engine_args = AsyncEngineArgs.from_cli_args(args)
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(
        usage_context=usage_context,
        headless=True,
    )

    if engine_args.data_parallel_hybrid_lb:
        raise ValueError("data_parallel_hybrid_lb is not applicable in headless mode")

    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local
    if local_engine_count <= 0:
        raise ValueError("data_parallel_size_local must be > 0 in headless mode")

    shutdown_requested = False

    def signal_handler(signum, frame) -> None:  # type: ignore[override]
        nonlocal shutdown_requested
        logger.debug("Received %d signal.", signum)
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    if parallel_config.node_rank_within_dp > 0:
        executor = MultiprocExecutor(vllm_config, monitor_workers=False)
        executor.start_worker_monitor(inline=True)
        return

    host = parallel_config.data_parallel_master_ip
    port = parallel_config.data_parallel_rpc_port
    handshake_address = get_tcp_uri(host, port)

    logger.info(
        "Launching %d data parallel engine(s) in headless mode, with head node "
        "address %s.",
        local_engine_count,
        handshake_address,
    )

    engine_manager = CoreEngineProcManager(
        local_engine_count=local_engine_count,
        start_index=vllm_config.parallel_config.data_parallel_rank,
        local_start_index=0,
        vllm_config=vllm_config,
        local_client=False,
        handshake_address=handshake_address,
        executor_class=Executor.get_class(vllm_config),
        log_stats=True,
    )

    try:
        engine_manager.join_first()
    finally:
        timeout = None
        if shutdown_requested:
            timeout = vllm_config.shutdown_timeout
            logger.info("Waiting up to %d seconds for processes to exit", timeout)
        engine_manager.shutdown(timeout=timeout)


def build_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(
        description="Offline Poisson benchmark harness using AsyncLLM directly.",
    )
    subparsers = parser.add_subparsers(dest="role", required=True)

    frontend = subparsers.add_parser(
        "frontend",
        help="Run the benchmark frontend and local engine(s).",
    )
    frontend.add_argument("--input-csv", required=True, help="Input CSV path.")
    frontend.add_argument(
        "--csv-repeat",
        type=_positive_int,
        default=None,
        help=(
            "Repeat the input CSV this many times before applying "
            "--warmup-requests and --max-requests. Defaults to the minimum "
            "repeat count needed to satisfy warmup + max-requests when "
            "--max-requests is set; otherwise defaults to 1."
        ),
    )
    frontend.add_argument(
        "--csv-format",
        default="length_csv",
        choices=["length_csv", "prompt_csv"],
        help="Input CSV format.",
    )
    frontend.add_argument(
        "--arrival-process",
        default="poisson",
        choices=["poisson"],
        help="Arrival process for request submission.",
    )
    frontend.add_argument(
        "--request-rate",
        type=_positive_float_or_inf,
        required=True,
        help="Total request rate in requests/sec. Use inf for no delay.",
    )
    frontend.add_argument(
        "--max-requests",
        type=_max_requests_arg,
        default=None,
        help=(
            "Cap the number of measured requests. Use "
            f"'{MAX_REQUESTS_CSV_ROWS}' to run as many measured requests as "
            "there are original CSV rows."
        ),
    )
    frontend.add_argument(
        "--warmup-requests",
        type=_non_negative_int,
        default=0,
        help="Number of warmup requests to run before measurement.",
    )
    frontend.add_argument(
        "--output-dir",
        required=True,
        help="Directory for run_meta.json, requests.jsonl and summary.json.",
    )
    frontend.add_argument(
        "--save-merged-parquet",
        action="store_true",
        help="Also export merged_requests.parquet.",
    )
    frontend.add_argument(
        "--request-id-prefix",
        default=None,
        help="Request id prefix. Default: r<seed>-",
    )
    frontend.add_argument(
        "--profile-after-warmup",
        action="store_true",
        help=(
            "Call AsyncLLM.start_profile() after warmup completes and stop the "
            "profiler during frontend teardown. Requires profiler-config "
            "flags to be set."
        ),
    )
    frontend.add_argument(
        "--profile-prefix",
        default=None,
        help=(
            "Optional prefix passed to AsyncLLM.start_profile(). Defaults to "
            "the output directory name."
        ),
    )
    frontend.add_argument(
        "--pause-before-profile",
        action="store_true",
        help=(
            "After warmup, pause generation with mode=keep, submit all measured "
            "requests into the engine queue, start profiling, then resume "
            "generation. This changes request latency/queue-time statistics and "
            "is intended for trace capture rather than throughput comparison."
        ),
    )
    frontend.add_argument(
        "--pause-before-measured",
        action="store_true",
        help=(
            "After warmup, pause generation with mode=keep, submit all measured "
            "requests into the engine queue, then resume generation without "
            "starting a profiler. This provides the same burst batch setup as "
            "--pause-before-profile for eager debugging."
        ),
    )
    frontend.add_argument(
        "--routing-mode",
        default="internal_dplb",
        choices=["internal_dplb", "explicit_rank_replay"],
        help="Dispatch mode. Only internal_dplb is implemented.",
    )
    frontend.add_argument(
        "--progress-log-interval",
        type=_non_negative_int,
        default=100,
        help="Log progress every N measured completions. Use 0 to disable.",
    )
    AsyncEngineArgs.add_cli_args(frontend)
    frontend.set_defaults(
        enable_expert_parallel=True,
        all2all_backend="deepep_low_latency",
        attention_backend="FLASHMLA",
        kv_transfer_config=DEFAULT_KV_TRANSFER_CONFIG.copy(),
        load_format="dummy",
        compilation_config=DEFAULT_COMPILATION_CONFIG.copy(),
        enable_prefix_caching=False,
    )

    headless = subparsers.add_parser(
        "headless-engine",
        help="Run engine-core processes only for remote multi-node DP.",
    )
    AsyncEngineArgs.add_cli_args(headless)
    headless.set_defaults(
        enable_expert_parallel=True,
        all2all_backend="deepep_low_latency",
        attention_backend="FLASHMLA",
        kv_transfer_config=DEFAULT_KV_TRANSFER_CONFIG.copy(),
        load_format="dummy",
        compilation_config=DEFAULT_COMPILATION_CONFIG.copy(),
        enable_prefix_caching=False,
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.role == "frontend":
        uvloop.run(run_frontend(args))
    elif args.role == "headless-engine":
        run_headless_engine(args)
    else:
        raise ValueError(f"Unknown role: {args.role}")


if __name__ == "__main__":
    main()
