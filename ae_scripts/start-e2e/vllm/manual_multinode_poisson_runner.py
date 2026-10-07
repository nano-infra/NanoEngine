#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual multi-node launcher for offline_poisson_harness experiments.

Edit the config block in this file directly to add or tweak experiment cases.
The CLI intentionally stays thin: list cases, run one case, or run them all.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping, TextIO

BENCHMARKS_DIR = Path(__file__).resolve().parent
AE_ROOT = BENCHMARKS_DIR.parents[1]
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))
if str(AE_ROOT) not in sys.path:
    sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require  # noqa: E402
from offline_profile_strategy_defaults import (  # noqa: E402
    STRATEGY_PROFILE_DEFAULTS,
)


def build_rate_range(start: float, stop: float, step: float) -> tuple[float, ...]:
    if step <= 0.0:
        raise ValueError("step must be > 0")
    if stop < start:
        raise ValueError("stop must be >= start")

    count = int(round((stop - start) / step))
    rates = tuple(round(start + step * index, 10) for index in range(count + 1))
    if not rates:
        raise ValueError("rate range cannot be empty")
    if not math.isclose(rates[-1], stop):
        raise ValueError(
            f"rate range {start:g}..{stop:g} step {step:g} is not evenly divisible"
        )
    return rates


HARNESS_ENTRYPOINT = os.environ.get(
    "VLLM_OFFLINE_POISSON_HARNESS",
    str(BENCHMARKS_DIR / "offline_poisson_harness.py"),
)
DEFAULT_WORKDIR = os.environ.get("VLLM_WORKDIR", "/vllm")
DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get(
        "VLLM_MULTINODE_ARTIFACT_ROOT",
        "/vllm/offline_bench/manual_multinode",
    )
)
DEFAULT_CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
DEFAULT_SSH_OPTS = (
    "-F",
    "/root/.ssh/config",
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "UpdateHostKeys=no",
)
FORWARDED_ENV_KEYS = (
    "PATH",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_TC",
    "NCCL_SOCKET_IFNAME",
    "NVSHMEM_HCA_LIST",
    "NVSHMEM_IB_GID_INDEX",
    "NVSHMEM_IBGDA_NUM_RC_PER_PE",
    "NVSHMEM_IB_TRAFFIC_CLASS",
    "NVSHMEM_DISABLE_NVLS",
    "VLLM_DEEP_GEMM_WARMUP",
    "VLLM_MOE_ROUTING_SIMULATION_STRATEGY",
    "VLLM_RANDOMIZE_DP_DUMMY_INPUTS",
    "CUDA_LAUNCH_BLOCKING",
)
DEFAULT_ENV_OVERRIDES = {
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_DEEP_GEMM_WARMUP": "skip",
    "VLLM_LOG_STATS_INTERVAL": "1",
    "VLLM_MOE_ROUTING_SIMULATION_STRATEGY": "uniform_random",
    "VLLM_RANDOMIZE_DP_DUMMY_INPUTS": "1",
}
DEFAULT_SHARED_CLI_ARGS = (
    "--no-enable-prefix-caching",
    "--trust-remote-code",
)
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_BENCH_TIMEOUT_SEC = 60 * 60
DEFAULT_SWEEP_BENCH_DURATION_SEC = 600.0
PRESTART_CLEANUP_MAX_ATTEMPTS = 3
PRESTART_CLEANUP_WAIT_SEC = 10.0
PRESTART_CLEANUP_POLL_INTERVAL_SEC = 1.0
LOCAL_SHUTDOWN_POLL_INTERVAL_SEC = 1.0
WAIT_STATUS_LOG_INTERVAL_SEC = 15.0
FATAL_LOG_SCAN_LINES = 200
FATAL_LOG_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("AsyncLLM output_handler failed.", ),
    ("EngineCore encountered a fatal error.", ),
    ("Worker proc", "died unexpectedly, shutting down executor."),
    ("EngineDeadError: EngineCore encountered an issue.", ),
)
TPOT_BY_E2E_EARLY_STOP_MS = 100.0
TREND_RERUN_METRIC_KEY = "tpot_by_e2e"
TREND_RERUN_SUBMETRIC = "mean"
DEFAULT_TREND_RERUN_MAX_ATTEMPTS = 1
DEFAULT_TREND_RERUN_RELATIVE_THRESHOLD_PCT = 12.0
DEFAULT_TREND_RERUN_ABSOLUTE_THRESHOLD_MS = 8.0
GPU_KV_CACHE_CAPACITY_LOG_MARKER = "poisson_gpu_kv_cache_capacity "
RATE_SWEEP_START = 10
RATE_SWEEP_STOP = 90
RATE_SWEEP_STEP = 10
MID_RATE_SWEEP_OFFSET = RATE_SWEEP_STEP // 2
SWEEP_REQUEST_RATES: tuple[float, ...] = tuple(
    float(rate) for rate in range(RATE_SWEEP_START, RATE_SWEEP_STOP + 1,
                                  RATE_SWEEP_STEP))
MID_SWEEP_REQUEST_RATES: tuple[float, ...] = tuple(
    float(rate)
    for rate in range(RATE_SWEEP_START + MID_RATE_SWEEP_OFFSET,
                      RATE_SWEEP_STOP, RATE_SWEEP_STEP))
ISSUE01_TARGET_REQUEST_RATES: tuple[float, ...] = (40.0, 45.0)
FULL2P5_SWEEP_REQUEST_RATES: tuple[float, ...] = build_rate_range(
    2.5, 90.0, 2.5)
COARSE1_SWEEP_REQUEST_RATES: tuple[float, ...] = build_rate_range(
    1.0, 90.0, 1.0)
MID0P5_SWEEP_REQUEST_RATES: tuple[float, ...] = build_rate_range(
    1.5, 89.5, 1.0)
COARSE1_TO10_SWEEP_REQUEST_RATES: tuple[float, ...] = build_rate_range(
    1.0, 10.0, 1.0)
MID0P5_TO10_SWEEP_REQUEST_RATES: tuple[float, ...] = build_rate_range(
    1.5, 9.5, 1.0)
DEFAULT_RATE_PLAN = "coarse10"
RATE_PLAN_PHASES: dict[str, tuple[tuple[str, tuple[float, ...]], ...]] = {
    "coarse10": (("coarse10", SWEEP_REQUEST_RATES), ),
    "coarse10_then_mid5": (
        ("coarse10", SWEEP_REQUEST_RATES),
        ("mid5", MID_SWEEP_REQUEST_RATES),
    ),
    "issue01_40_45": (("issue01_40_45", ISSUE01_TARGET_REQUEST_RATES), ),
    "full2p5": (("full2p5", FULL2P5_SWEEP_REQUEST_RATES), ),
    "coarse1_then_mid0p5": (
        ("coarse1", COARSE1_SWEEP_REQUEST_RATES),
        ("mid0p5", MID0P5_SWEEP_REQUEST_RATES),
    ),
    "coarse1_to10_then_mid0p5": (
        ("coarse1", COARSE1_TO10_SWEEP_REQUEST_RATES),
        ("mid0p5", MID0P5_TO10_SWEEP_REQUEST_RATES),
    ),
}
DEFAULT_DISPATCH_POLICY = "waiting_x4_plus_running"
DISPATCH_POLICY_CHOICES: tuple[str, ...] = (
    DEFAULT_DISPATCH_POLICY,
    "least_cache",
    "least_batch",
)
CASE_CSV_FIELDNAMES: tuple[str, ...] = (
    "enabled",
    "name",
    "cluster",
    "model",
    "dataset",
    "strategy",
    "dispatch_policy",
    "request_rate",
    "rate_phase",
    "max_num_seqs",
    "gpu_memory_utilization",
    "max_requests",
    "warmup_requests",
    "max_model_len",
    "data_parallel_rpc_port",
    "reason",
    "historical_reference",
)
MODEL_SHORT_NAMES: dict[str, str] = {
    "deepseek_v3_1024k": "DPSK",
    "kimi_k2_instruct_0905": "KIMI",
    "qwen3_235b_fp8_1024k": "QWEN3_235B",
    "qwen3_235b_fp8": "QWEN3_235B",
}
DATASET_SHORT_NAMES: dict[str, str] = {
    "issue01_random": "issue01_random",
    "long_full": "long_full",
}
MAX_REQUESTS_CSV_ROWS = "csv_rows"
MaxRequestsValue = int | Literal["csv_rows"] | None


@dataclass(frozen=True)
class ClusterSpec:
    master_addr: str
    master_port: int
    remote_hosts: tuple[str, ...]
    workdir: str = DEFAULT_WORKDIR
    ssh_opts: tuple[str, ...] = DEFAULT_SSH_OPTS
    local_shell: str = "zsh"
    local_shell_flags: tuple[str, ...] = ("-lc",)
    remote_shell: str = "zsh"
    remote_shell_flags: tuple[str, ...] = ("-lc",)
    local_env_script: str | None = None
    remote_env_script: str | None = None

    @property
    def nnodes(self) -> int:
        return 1 + len(self.remote_hosts)


@dataclass(frozen=True)
class StrategySpec:
    data_parallel_size: int
    data_parallel_size_local: int | None
    tensor_parallel_size: int
    decode_context_parallel_size: int = 1
    data_parallel_backend: str = "mp"
    enable_expert_parallel: bool = True
    attention_backend: str | None = "FLASHMLA"
    all2all_backend: str | None = "deepep_low_latency"
    dcp_comm_backend: str | None = None
    max_num_batched_tokens: int | None = None


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    cluster: str
    strategy: str
    dataset: str
    model: str = "deepseek_v3_1024k"
    dispatch_policy: str = DEFAULT_DISPATCH_POLICY
    request_rate: float = 100.0
    rate_phase: str = DEFAULT_RATE_PLAN
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    max_requests: MaxRequestsValue = None
    warmup_requests: int = 0
    max_num_seqs: int | None = None
    max_model_len: int | None = None
    max_num_batched_tokens: int | None = None
    data_parallel_rpc_port: int = 29550
    cuda_visible_devices: str = DEFAULT_CUDA_VISIBLE_DEVICES
    env: Mapping[str, str] = field(default_factory=dict)
    frontend_env: Mapping[str, str] = field(default_factory=dict)
    headless_env: Mapping[str, str] = field(default_factory=dict)
    shared_cli_args: tuple[str, ...] = DEFAULT_SHARED_CLI_ARGS
    frontend_extra_args: tuple[str, ...] = ()
    headless_extra_args: tuple[str, ...] = ()
    request_id_prefix: str | None = None
    seed: int = 0
    progress_log_interval: int = 100
    save_merged_parquet: bool = False
    remote_hosts_override: tuple[str, ...] | None = None
    start_grace_sec: float = 8.0
    remote_shutdown_grace_sec: float = 30.0
    local_shutdown_grace_sec: float = 30.0
    cleanup_worker_processes: bool = True
    max_bench_duration_sec: float | None = DEFAULT_BENCH_TIMEOUT_SEC


@dataclass(frozen=True)
class ResolvedCase:
    case: ExperimentCase
    cluster_name: str
    cluster: ClusterSpec
    strategy_name: str
    strategy: StrategySpec
    dataset_path: Path
    model_path: Path


@dataclass(frozen=True)
class ArtifactPaths:
    run_dir: Path
    case_dir: Path
    benchmark_dir: Path
    case_manifest_path: Path
    frontend_command_path: Path
    frontend_cleanup_command_path: Path
    frontend_pid_path: Path
    frontend_pgid_path: Path
    frontend_log_path: Path
    frontend_launch_log_path: Path
    rank_command_paths: dict[int, Path]
    rank_cleanup_command_paths: dict[int, Path]
    rank_pid_paths: dict[int, Path]
    rank_pgid_paths: dict[int, Path]
    rank_log_paths: dict[int, Path]
    rank_launch_log_paths: dict[int, Path]


@dataclass
class NodeRuntime:
    node_rank: int
    host: str
    process: subprocess.Popen[str]
    launch_log_path: Path
    launch_log_handle: TextIO
    runtime_log_path: Path


@dataclass
class ActiveCaseRuntime:
    resolved: ResolvedCase
    artifacts: ArtifactPaths
    frontend: NodeRuntime | None = None
    headless_nodes: list[NodeRuntime] = field(default_factory=list)


@dataclass(frozen=True)
class CaseResult:
    case_name: str
    status: str
    exit_code: int | None
    started_at: str
    finished_at: str
    case_dir: Path
    benchmark_dir: Path
    summary_json: Path
    detail: str


@dataclass(frozen=True)
class TrendOutlierCandidate:
    case: ExperimentCase
    result: CaseResult
    lower_case: ExperimentCase
    lower_result: CaseResult
    upper_case: ExperimentCase
    upper_result: CaseResult
    actual_value: float
    expected_value: float
    allowed_delta_ms: float
    delta_ms: float


class BenchTimeoutError(TimeoutError):
    pass


# -----------------------------------------------------------------------------
# Manual config block: edit these directly for your cluster and experiments.
# Machine-specific roots live in paths.env; only file names live here.
# -----------------------------------------------------------------------------

MIXLONG_0326 = require("AE_DATASET_MIXLONG_0326")
SHAREGPT4O = require("AE_DATASET_SHAREGPT4O")

MODELS: dict[str, str] = {
    "deepseek_v3_1024k": require("AE_DPSK_MODEL"),
    "kimi_k2_instruct_0905":
    f"{require('AE_KIMI_MODEL_HF')}/snapshots/7152993552508c9f22042b3bb93b5e6acd06ce73",
    "qwen3_235b_fp8_1024k": require("AE_QWEN3_MODEL_1024K"),
    "qwen3_235b_fp8": require("AE_QWEN3_MODEL"),
}

DATASETS: dict[str, str] = {
    "1k1k": require("AE_DATASET_0110"),
    "long_full": require("AE_DATASET_MADHA"),
    # "issue01_halfhalf":
    # f"{MIXLONG_0326}/sharegpt4o-halfhalf_geminiissue_r0.01_n60000_60k.csv",
    "issue01_random":
    f"{MIXLONG_0326}/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv",
    "issue03_random":
    f"{MIXLONG_0326}/sharegpt4o-random_geminiissue_r0.03_n60000.csv",
    # "issue05_halfhalf":
    # f"{MIXLONG_0326}/sharegpt4o-halfhalf_geminiissue_r0.05_n60000_60k.csv",
    "issue05_random":
    f"{MIXLONG_0326}/sharegpt4o-random_geminiissue_r0.05_n60000_60k.csv",
    # "short_halfhalf":
    # f"{SHAREGPT4O}/sharegpt4o-mixed-half-half-60k.csv",
    "short_random":
    f"{SHAREGPT4O}/sharegpt4o-mixed-random-60k.csv",
}

CLUSTERS: dict[str, ClusterSpec] = {
    "1node_h200":
    ClusterSpec(
        master_addr=os.environ.get("VLLM_1NODE_H200_MASTER_ADDR", "127.0.0.1"),
        master_port=29579,
        remote_hosts=(),
    ),
    "2node_h200":
    ClusterSpec(
        master_addr=os.environ.get("VLLM_2NODE_H200_MASTER_ADDR",
                                   "10.102.97.179"),
        master_port=29579,
        remote_hosts=(os.environ.get("VLLM_2NODE_H200_REMOTE_HOST",
                                     "h200-rjob1"), ),
        local_env_script="/root/.zshrc",
        remote_env_script="/root/.zshrc",
    ),
    "2node-1and3":
    ClusterSpec(
        master_addr=os.environ.get("VLLM_2NODE_1AND3_MASTER_ADDR",
                                   "10.102.215.76"),
        master_port=29579,
        remote_hosts=("h200-rjob2", ),
    ),
    # "4node_h200":
    # ClusterSpec(
    #     master_addr="10.102.97.179",
    #     master_port=29579,
    #     remote_hosts=("h200-rjob1", "h200-rjob2", "h200-rjob3"),
    # ),
    "4node_h200":
    ClusterSpec(
        master_addr=os.environ.get("VLLM_4NODE_H200_MASTER_ADDR",
                                   "10.102.252.174"),
        master_port=29579,
        remote_hosts=("h200-rjob1", "h200-rjob2", "h200-rjob3"),
        local_env_script="/root/.zshrc",
        remote_env_script="/root/.zshrc",
    ),
}

STRATEGIES: dict[str, StrategySpec] = {
    "dp2dcp8":
    StrategySpec(
        data_parallel_size=2,
        data_parallel_size_local=1,
        tensor_parallel_size=8,
        decode_context_parallel_size=8,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASHMLA",
        all2all_backend="deepep_low_latency",
        dcp_comm_backend="a2a",
    ),
    "dp4dcp4":
    StrategySpec(
        data_parallel_size=4,
        data_parallel_size_local=2,
        tensor_parallel_size=4,
        decode_context_parallel_size=4,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASHMLA",
        all2all_backend="deepep_low_latency",
        dcp_comm_backend="a2a",
    ),
    "dp8cp2":
    StrategySpec(
        data_parallel_size=8,
        data_parallel_size_local=4,
        tensor_parallel_size=2,
        decode_context_parallel_size=2,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASHMLA",
        all2all_backend="deepep_low_latency",
        dcp_comm_backend="a2a",
    ),
    "dp4dcp8":
    StrategySpec(
        data_parallel_size=4,
        data_parallel_size_local=1,
        tensor_parallel_size=8,
        decode_context_parallel_size=8,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASHMLA",
        all2all_backend="deepep_low_latency",
        dcp_comm_backend="a2a",
    ),
    "dp8dcp4":
    StrategySpec(
        data_parallel_size=8,
        data_parallel_size_local=2,
        tensor_parallel_size=4,
        decode_context_parallel_size=4,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASHMLA",
        all2all_backend="deepep_low_latency",
        dcp_comm_backend="a2a",
    ),
    "dp16cp2":
    StrategySpec(
        data_parallel_size=16,
        data_parallel_size_local=4,
        tensor_parallel_size=2,
        decode_context_parallel_size=2,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASHMLA",
        all2all_backend="deepep_low_latency",
        dcp_comm_backend="a2a",
    ),
    "dp32":
    StrategySpec(
        data_parallel_size=32,
        data_parallel_size_local=8,
        tensor_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASHMLA",
        all2all_backend="deepep_low_latency",
    ),
    "dp16":
    StrategySpec(
        data_parallel_size=16,
        data_parallel_size_local=8,
        tensor_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASHMLA",
        all2all_backend="deepep_low_latency",
    ),
    "dp4tp8dcp2":
    StrategySpec(
        data_parallel_size=4,
        data_parallel_size_local=1,
        tensor_parallel_size=8,
        decode_context_parallel_size=2,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASH_ATTN",
        all2all_backend="deepep_low_latency",
        dcp_comm_backend="a2a",
    ),
    "dp1tp8dcp2":
    StrategySpec(
        data_parallel_size=1,
        data_parallel_size_local=1,
        tensor_parallel_size=8,
        decode_context_parallel_size=2,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASH_ATTN",
        all2all_backend="deepep_low_latency",
        dcp_comm_backend="a2a",
    ),
    "dp2tp8dcp2":
    StrategySpec(
        data_parallel_size=2,
        data_parallel_size_local=1,
        tensor_parallel_size=8,
        decode_context_parallel_size=2,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASH_ATTN",
        all2all_backend="deepep_low_latency",
        dcp_comm_backend="a2a",
    ),
    "dp4tp8dcp2_ar":
    StrategySpec(
        data_parallel_size=4,
        data_parallel_size_local=1,
        tensor_parallel_size=8,
        decode_context_parallel_size=2,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASH_ATTN",
        all2all_backend="deepep_low_latency",
        dcp_comm_backend="ag_rs",
    ),
    "dp4tp8":
    StrategySpec(
        data_parallel_size=4,
        data_parallel_size_local=1,
        tensor_parallel_size=8,
        decode_context_parallel_size=1,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASH_ATTN",
        all2all_backend="deepep_low_latency",
    ),
    "dp4tp4":
    StrategySpec(
        data_parallel_size=4,
        data_parallel_size_local=None,
        tensor_parallel_size=4,
        decode_context_parallel_size=1,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASH_ATTN",
        all2all_backend="deepep_low_latency",
    ),
    "dp8tp4":
    StrategySpec(
        data_parallel_size=8,
        data_parallel_size_local=2,
        tensor_parallel_size=4,
        decode_context_parallel_size=1,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASH_ATTN",
        all2all_backend="deepep_low_latency",
    ),
    "dp16tp2":
    StrategySpec(
        data_parallel_size=16,
        data_parallel_size_local=4,
        tensor_parallel_size=2,
        decode_context_parallel_size=1,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASH_ATTN",
        all2all_backend="deepep_low_latency",
    ),
}

QWEN_SUPPORTED_STRATEGIES: tuple[str, ...] = (
    "dp4tp8dcp2",
    "dp4tp4",
    "dp8tp4",
    "dp4tp8",
    "dp16tp2",
)
QWEN_CASE_ONLY_STRATEGIES: tuple[str, ...] = (
    "dp1tp8dcp2",
    "dp2tp8dcp2",
)
QWEN_ALLOWED_STRATEGIES: tuple[str, ...] = (QWEN_SUPPORTED_STRATEGIES +
                                             QWEN_CASE_ONLY_STRATEGIES)
QWEN_STRATEGY_PROFILE_DEFAULTS: dict[str, tuple[int, float]] = {
    "dp4tp8dcp2": (1024, DEFAULT_GPU_MEMORY_UTILIZATION),
    "dp1tp8dcp2": (768, DEFAULT_GPU_MEMORY_UTILIZATION),
    "dp4tp4": (384, DEFAULT_GPU_MEMORY_UTILIZATION),
    "dp8tp4": (768, DEFAULT_GPU_MEMORY_UTILIZATION),
    "dp4tp8": (1024, DEFAULT_GPU_MEMORY_UTILIZATION),
    "dp16tp2": (512, DEFAULT_GPU_MEMORY_UTILIZATION),
}

SWEEP_CLUSTER = "4node_h200"
SWEEP_DATASET = "issue01_random"
SWEEP_MODELS: tuple[str, ...] = (
    "kimi_k2_instruct_0905",
    "deepseek_v3_1024k",
)
SWEEP_STRATEGIES: tuple[str, ...] = (
    "dp4dcp8",
    "dp8dcp4",
    "dp16cp2",
    "dp32",
)
DEEPSEEK_TWO_NODE_STRATEGIES: tuple[str, ...] = (
    "dp2dcp8",
    "dp4dcp4",
    "dp8cp2",
    "dp16",
)
DEEPSEEK_ALLOWED_STRATEGIES: tuple[str, ...] = (
    SWEEP_STRATEGIES + DEEPSEEK_TWO_NODE_STRATEGIES
)
DEEPSEEK_TWO_NODE_PROFILE_DEFAULTS: dict[str, tuple[int, float]] = {
    "dp2dcp8": (64, 0.85),
    "dp4dcp4": (64, 0.85),
    "dp8cp2": (64, 0.85),
    "dp16": (64, 0.85),
}
STRATEGY_MAX_NUM_SEQS: dict[str, int] = {
    strategy_name: defaults.max_num_seqs
    for strategy_name, defaults in STRATEGY_PROFILE_DEFAULTS.items()
}
STRATEGY_MAX_NUM_SEQS.update({
    strategy_name: defaults[0]
    for strategy_name, defaults in QWEN_STRATEGY_PROFILE_DEFAULTS.items()
})
STRATEGY_MAX_NUM_SEQS.update({
    strategy_name: defaults[0]
    for strategy_name, defaults in DEEPSEEK_TWO_NODE_PROFILE_DEFAULTS.items()
})
STRATEGY_GPU_MEMORY_UTILIZATION: dict[str, float] = {
    strategy_name: defaults.gpu_memory_utilization
    for strategy_name, defaults in STRATEGY_PROFILE_DEFAULTS.items()
}
STRATEGY_GPU_MEMORY_UTILIZATION.update({
    strategy_name: defaults[1]
    for strategy_name, defaults in QWEN_STRATEGY_PROFILE_DEFAULTS.items()
})
STRATEGY_GPU_MEMORY_UTILIZATION.update({
    strategy_name: defaults[1]
    for strategy_name, defaults in DEEPSEEK_TWO_NODE_PROFILE_DEFAULTS.items()
})


def is_qwen_model(value: str | Path) -> bool:
    text = value.name if isinstance(value, Path) else value
    return "qwen" in text.lower()


def supported_strategies_for_model(model_name: str) -> tuple[str, ...]:
    if is_qwen_model(model_name):
        return QWEN_SUPPORTED_STRATEGIES
    return SWEEP_STRATEGIES


def validate_model_strategy_pair(
    *,
    model_name: str,
    model_path: Path | None,
    strategy_name: str,
) -> None:
    is_qwen = is_qwen_model(model_name)
    if model_path is not None:
        is_qwen = is_qwen or is_qwen_model(model_path)

    if is_qwen:
        if strategy_name not in QWEN_ALLOWED_STRATEGIES:
            supported = ", ".join(QWEN_ALLOWED_STRATEGIES)
            raise ValueError(
                f"Qwen models only support strategies: {supported}. "
                f"Got '{strategy_name}'."
            )
        return

    if strategy_name in QWEN_ALLOWED_STRATEGIES:
        raise ValueError(
            f"Strategy '{strategy_name}' is only supported for Qwen models.")
    if strategy_name not in DEEPSEEK_ALLOWED_STRATEGIES:
        supported = ", ".join(DEEPSEEK_ALLOWED_STRATEGIES)
        raise ValueError(
            f"DeepSeek/Kimi models only support strategies: {supported}. "
            f"Got '{strategy_name}'."
        )


def bench_duration_to_max_requests(request_rate: float,
                                   bench_duration_sec: float) -> int:
    if math.isinf(request_rate):
        raise ValueError(
            "bench_duration_to_max_requests does not support inf request_rate."
        )
    if bench_duration_sec <= 0.0:
        raise ValueError("bench_duration_sec must be > 0.")
    return max(1, int(round(request_rate * bench_duration_sec)))


def unique_candidate_rates_for_rate_plan(rate_plan: str) -> tuple[float, ...]:
    try:
        rate_plan_phases = RATE_PLAN_PHASES[rate_plan]
    except KeyError as exc:
        supported = ", ".join(sorted(RATE_PLAN_PHASES))
        raise ValueError(
            f"Unknown rate plan '{rate_plan}'. Supported values: {supported}"
        ) from exc

    return tuple(
        sorted({
            float(rate)
            for _phase_name, request_rates in rate_plan_phases
            for rate in request_rates
        }))


def build_experiment_matrix(
    rate_plan: str = DEFAULT_RATE_PLAN,
    *,
    models: list[str] | tuple[str, ...] | None = None,
    datasets: list[str] | tuple[str, ...] | None = None,
    strategies: list[str] | tuple[str, ...] | None = None,
) -> list[ExperimentCase]:
    try:
        rate_plan_phases = RATE_PLAN_PHASES[rate_plan]
    except KeyError as exc:
        supported = ", ".join(sorted(RATE_PLAN_PHASES))
        raise ValueError(
            f"Unknown rate plan '{rate_plan}'. Supported values: {supported}"
        ) from exc

    selected_models = ordered_models(models)
    selected_datasets = ordered_datasets(datasets)
    selected_strategies = ordered_strategies(strategies)
    use_model_default_strategies = not strategies

    experiments: list[ExperimentCase] = []
    for rate_phase, request_rates in rate_plan_phases:
        for dataset_name in selected_datasets:
            dataset_tag = DATASET_SHORT_NAMES.get(dataset_name, dataset_name)
            for model_name in selected_models:
                model_tag = MODEL_SHORT_NAMES.get(model_name, model_name)
                if use_model_default_strategies:
                    model_strategies = supported_strategies_for_model(model_name)
                else:
                    model_strategies = selected_strategies
                for strategy_name in model_strategies:
                    validate_model_strategy_pair(
                        model_name=model_name,
                        model_path=None,
                        strategy_name=strategy_name,
                    )
                    max_num_seqs = STRATEGY_MAX_NUM_SEQS[strategy_name]
                    gpu_memory_utilization = STRATEGY_GPU_MEMORY_UTILIZATION[
                        strategy_name]
                    for request_rate in request_rates:
                        rate_tag = f"{request_rate:g}"
                        max_requests = bench_duration_to_max_requests(
                            request_rate,
                            DEFAULT_SWEEP_BENCH_DURATION_SEC,
                        )
                        experiments.append(
                            ExperimentCase(
                                name=(f"{model_tag}__{dataset_tag}__"
                                      f"{strategy_name}__rate{rate_tag}__"
                                      f"bs{max_num_seqs}"),
                                cluster=(
                                    "2node_h200"
                                    if strategy_name in DEEPSEEK_TWO_NODE_STRATEGIES
                                    else SWEEP_CLUSTER
                                ),
                                strategy=strategy_name,
                                dataset=dataset_name,
                                model=model_name,
                                request_rate=request_rate,
                                rate_phase=rate_phase,
                                gpu_memory_utilization=
                                gpu_memory_utilization,
                                max_requests=max_requests,
                                warmup_requests=32,
                                max_num_seqs=max_num_seqs,
                                max_model_len=1000000,
                                data_parallel_rpc_port=29550,
                            ))
    return experiments

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def current_iso_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def current_run_tag() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def shell_join(args: list[str] | tuple[str, ...]) -> str:
    return shlex.join(list(args))


def shell_command_argv(shell: str, flags: tuple[str, ...], command: str) -> list[str]:
    return [shell, *flags, command]


def sanitize_tag(raw_value: str) -> str:
    lowered = raw_value.strip().lower()
    output: list[str] = []
    last_was_sep = False
    for ch in lowered:
        if ch.isalnum():
            output.append(ch)
            last_was_sep = False
            continue
        if not last_was_sep:
            output.append("_")
            last_was_sep = True
    return "".join(output).strip("_") or "case"


def stringify_request_rate(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:g}"


def infer_bench_duration_sec(case: ExperimentCase) -> float:
    if isinstance(case.max_requests, int) and not math.isinf(case.request_rate):
        return case.max_requests / case.request_rate
    return DEFAULT_SWEEP_BENCH_DURATION_SEC


def stringify_bench_duration_sec(value: float) -> str:
    if math.isclose(value, round(value)):
        return f"{int(round(value))}"
    return f"{value:g}"


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("Expected a positive float.")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def _ordered_selection(
    values: list[str] | tuple[str, ...] | None,
    *,
    supported: Mapping[str, Any],
    default: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    if not values:
        return default

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        if value not in supported:
            supported_values = ", ".join(sorted(supported))
            raise ValueError(
                f"Unknown {label} '{value}'. Supported values: {supported_values}"
            )
        deduped.append(value)
        seen.add(value)
    return tuple(deduped)


def ordered_models(
    values: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    return _ordered_selection(
        values,
        supported=MODELS,
        default=SWEEP_MODELS,
        label="model",
    )


def ordered_datasets(
    values: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    return _ordered_selection(
        values,
        supported=DATASETS,
        default=(SWEEP_DATASET, ),
        label="dataset",
    )


def ordered_strategies(
    values: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    return _ordered_selection(
        values,
        supported=STRATEGIES,
        default=SWEEP_STRATEGIES,
        label="strategy",
    )


EXPERIMENTS: list[ExperimentCase] = build_experiment_matrix()


def model_short_name(model: str) -> str:
    return MODEL_SHORT_NAMES.get(model, sanitize_tag(model).upper())


def dataset_short_name(dataset: str) -> str:
    return DATASET_SHORT_NAMES.get(dataset, sanitize_tag(dataset))


def stringify_gpu_memory_utilization(value: float) -> str:
    percentage = value * 100.0
    if math.isclose(percentage, round(percentage)):
        return f"mem{int(round(percentage))}"
    return f"mem{percentage:g}".replace(".", "_")


def normalize_dispatch_policy(policy: str) -> str:
    normalized = policy.strip()
    if normalized not in DISPATCH_POLICY_CHOICES:
        supported = ", ".join(DISPATCH_POLICY_CHOICES)
        raise ValueError(
            f"unknown dispatch_policy '{policy}'. Supported values: {supported}"
        )
    return normalized


def _dispatch_policy_group_key_component(case: ExperimentCase) -> str | None:
    dispatch_policy = normalize_dispatch_policy(case.dispatch_policy)
    if dispatch_policy == DEFAULT_DISPATCH_POLICY:
        return None
    return f"dispatch_{sanitize_tag(dispatch_policy)}"


def _dispatch_policy_artifact_suffix(case: ExperimentCase) -> str:
    component = _dispatch_policy_group_key_component(case)
    return f"-{component}" if component is not None else ""


def case_artifact_batch_size_tag(case: ExperimentCase) -> str:
    return f"bs{case.max_num_seqs}" if case.max_num_seqs is not None else "bsauto"


def case_artifact_scenario_match_prefix(case: ExperimentCase) -> str:
    return f"{sanitize_tag(case.strategy)}{_dispatch_policy_artifact_suffix(case)}"


def case_group_key(case: ExperimentCase) -> str:
    parts = [
        model_short_name(case.model),
        dataset_short_name(case.dataset),
        sanitize_tag(case.strategy),
    ]
    if (dispatch_component := _dispatch_policy_group_key_component(case)) is not None:
        parts.append(dispatch_component)
    return "/".join(parts)


def case_artifact_dataset_dir(artifact_root: Path, case: ExperimentCase) -> Path:
    return (artifact_root / model_short_name(case.model) /
            dataset_short_name(case.dataset))


def case_artifact_rate_duration_tag(case: ExperimentCase) -> str:
    duration_tag = f"dur{stringify_bench_duration_sec(infer_bench_duration_sec(case))}"
    return f"rate{stringify_request_rate(case.request_rate)}-{duration_tag}"


def case_artifact_scenario_prefix(case: ExperimentCase) -> str:
    return (
        f"{case_artifact_scenario_match_prefix(case)}-"
        f"{stringify_gpu_memory_utilization(case.gpu_memory_utilization)}"
    )


def case_artifact_group_dir(artifact_root: Path, case: ExperimentCase) -> Path:
    batch_size_tag = case_artifact_batch_size_tag(case)
    scenario_tag = (f"{case_artifact_scenario_prefix(case)}-"
                    f"{batch_size_tag}-"
                    f"{case_artifact_rate_duration_tag(case)}")
    return case_artifact_dataset_dir(artifact_root, case) / scenario_tag


def bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def tail_file(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def iter_runtime_log_paths(runtime: ActiveCaseRuntime,
                           include_frontend: bool) -> list[Path]:
    paths: list[Path] = []
    if include_frontend and runtime.frontend is not None:
        paths.append(runtime.frontend.runtime_log_path)
    paths.extend(node.runtime_log_path for node in runtime.headless_nodes)
    return paths


def _fatal_log_excerpt(log_lines: list[str], required_parts: tuple[str,
                                                                  ...]) -> str:
    matching_indices = [
        index for index, line in enumerate(log_lines)
        if any(part in line for part in required_parts)
    ]
    if not matching_indices:
        return "\n".join(log_lines)

    start = max(0, min(matching_indices) - 3)
    end = min(len(log_lines), max(matching_indices) + 4)
    return "\n".join(log_lines[start:end])


def find_fatal_signal_in_log(path: Path, *, lines: int) -> str | None:
    excerpt = tail_file(path, lines=lines)
    if not excerpt:
        return None

    log_lines = excerpt.splitlines()
    for required_parts in FATAL_LOG_PATTERNS:
        if all(part in excerpt for part in required_parts):
            return _fatal_log_excerpt(log_lines, required_parts)
    return None


def check_runtime_logs_for_fatal(runtime: ActiveCaseRuntime,
                                 *,
                                 phase: str,
                                 include_frontend: bool) -> None:
    for path in iter_runtime_log_paths(runtime, include_frontend):
        excerpt = find_fatal_signal_in_log(path, lines=FATAL_LOG_SCAN_LINES)
        if excerpt is None:
            continue
        raise RuntimeError(f"detected fatal engine failure in {path} during "
                           f"{phase}.\n{excerpt}")


def extract_gpu_kv_cache_capacity_fields(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in reversed(
            path.read_text(encoding="utf-8",
                           errors="replace").splitlines()):
        marker_index = line.find(GPU_KV_CACHE_CAPACITY_LOG_MARKER)
        if marker_index >= 0:
            return line[marker_index +
                        len(GPU_KV_CACHE_CAPACITY_LOG_MARKER):].strip()
    return None


def format_gpu_kv_cache_capacity_message(fields_text: str) -> str:
    parsed: dict[str, str] = {}
    for field in fields_text.split():
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        parsed[key] = value

    total_tokens = parsed.get("total_tokens")
    managed_engines = parsed.get("managed_engines")
    usable_gpu_blocks = parsed.get("usable_gpu_blocks")
    block_size = parsed.get("block_size")
    if (total_tokens is not None and managed_engines is not None
            and usable_gpu_blocks is not None and block_size is not None):
        return ("GPU KV cache total token capacity="
                f"{total_tokens} "
                f"(managed_engines={managed_engines}, "
                f"usable_gpu_blocks={usable_gpu_blocks}, "
                f"block_size={block_size})")
    return f"GPU KV cache capacity: {fields_text}"


def jsonify(value: Any) -> Any:
    if is_dataclass(value):
        return jsonify(asdict(value))
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonify(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(
                    f"Expected JSON object line in {path}, got {type(payload).__name__}"
                )
            records.append(payload)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(jsonify(record), sort_keys=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def write_shell_script(path: Path, command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + command + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def resolve_alias_path(
    raw_value: str,
    aliases: Mapping[str, str],
    *,
    label: str,
    expect_file: bool,
) -> Path:
    resolved = aliases.get(raw_value, raw_value)
    path = Path(resolved).expanduser()
    if expect_file:
        if not path.is_file():
            raise SystemExit(f"{label} file not found: {path}")
    elif not path.exists():
        raise SystemExit(f"{label} path not found: {path}")
    return path.resolve()


def experiments_by_name(
    experiments: list[ExperimentCase],
) -> dict[str, ExperimentCase]:
    return {case.name: case for case in experiments}


def resolve_case(
    case: ExperimentCase,
    *,
    clusters: Mapping[str, ClusterSpec] = CLUSTERS,
    strategies: Mapping[str, StrategySpec] = STRATEGIES,
    datasets: Mapping[str, str] = DATASETS,
    models: Mapping[str, str] = MODELS,
) -> ResolvedCase:
    try:
        cluster = clusters[case.cluster]
    except KeyError as exc:
        supported = ", ".join(sorted(clusters))
        raise SystemExit(
            f"Unknown cluster '{case.cluster}'. Supported values: {supported}"
        ) from exc

    try:
        strategy = strategies[case.strategy]
    except KeyError as exc:
        supported = ", ".join(sorted(strategies))
        raise SystemExit(
            f"Unknown strategy '{case.strategy}'. Supported values: {supported}"
        ) from exc

    if case.request_rate <= 0.0 and not math.isinf(case.request_rate):
        raise SystemExit(
            f"request_rate must be > 0 or inf for case '{case.name}', got {case.request_rate}"
        )

    resolved_remote_hosts = tuple(case.remote_hosts_override or cluster.remote_hosts)
    resolved_cluster = replace(cluster, remote_hosts=resolved_remote_hosts)

    if resolved_cluster.nnodes <= 0:
        raise SystemExit(f"Resolved cluster for case '{case.name}' has no nodes.")

    if strategy.data_parallel_size_local is None:
        if strategy.data_parallel_size % resolved_cluster.nnodes != 0:
            raise SystemExit(
                "Strategy / cluster mismatch for "
                f"'{case.name}': dp={strategy.data_parallel_size}, "
                f"nnodes={resolved_cluster.nnodes}, "
                "cannot evenly derive dp_local from the node count."
            )
        effective_dp_local = (
            strategy.data_parallel_size // resolved_cluster.nnodes
        )
    else:
        effective_dp_local = strategy.data_parallel_size_local

    strategy = replace(strategy, data_parallel_size_local=effective_dp_local)
    expected_dp = resolved_cluster.nnodes * strategy.data_parallel_size_local
    if strategy.data_parallel_size != expected_dp:
        raise SystemExit(
            "Strategy / cluster mismatch for "
            f"'{case.name}': dp={strategy.data_parallel_size}, "
            f"nnodes={resolved_cluster.nnodes}, "
            f"dp_local={strategy.data_parallel_size_local}, "
            f"expected dp={expected_dp}"
        )

    dataset_path = resolve_alias_path(
        case.dataset,
        datasets,
        label=f"Dataset for case '{case.name}'",
        expect_file=True,
    )
    model_path = resolve_alias_path(
        case.model,
        models,
        label=f"Model for case '{case.name}'",
        expect_file=False,
    )
    try:
        validate_model_strategy_pair(
            model_name=case.model,
            model_path=model_path,
            strategy_name=case.strategy,
        )
    except ValueError as exc:
        raise SystemExit(
            f"Invalid model/strategy for case '{case.name}': {exc}"
        ) from exc

    return ResolvedCase(
        case=case,
        cluster_name=case.cluster,
        cluster=resolved_cluster,
        strategy_name=case.strategy,
        strategy=strategy,
        dataset_path=dataset_path,
        model_path=model_path,
    )


def prepare_artifact_paths(case_group_dir: Path,
                           remote_hosts: tuple[str, ...],
                           *,
                           run_tag: str) -> ArtifactPaths:
    case_dir = case_group_dir / run_tag
    benchmark_dir = case_dir / "benchmark"
    rank_command_paths: dict[int, Path] = {}
    rank_cleanup_command_paths: dict[int, Path] = {}
    rank_pid_paths: dict[int, Path] = {}
    rank_pgid_paths: dict[int, Path] = {}
    rank_log_paths: dict[int, Path] = {}
    rank_launch_log_paths: dict[int, Path] = {}

    for offset, _host in enumerate(remote_hosts, start=1):
        rank_command_paths[offset] = case_dir / f"rank{offset}.command.sh"
        rank_cleanup_command_paths[offset] = (
            case_dir / f"rank{offset}.cleanup.command.sh")
        rank_pid_paths[offset] = case_dir / f"rank{offset}.pid"
        rank_pgid_paths[offset] = case_dir / f"rank{offset}.pgid"
        rank_log_paths[offset] = case_dir / f"rank{offset}.log"
        rank_launch_log_paths[offset] = case_dir / f"rank{offset}.launch.log"

    return ArtifactPaths(
        run_dir=case_group_dir,
        case_dir=case_dir,
        benchmark_dir=benchmark_dir,
        case_manifest_path=case_dir / "case_manifest.json",
        frontend_command_path=case_dir / "frontend.command.sh",
        frontend_cleanup_command_path=case_dir / "frontend.cleanup.command.sh",
        frontend_pid_path=case_dir / "frontend.pid",
        frontend_pgid_path=case_dir / "frontend.pgid",
        frontend_log_path=case_dir / "frontend.log",
        frontend_launch_log_path=case_dir / "frontend.launch.log",
        rank_command_paths=rank_command_paths,
        rank_cleanup_command_paths=rank_cleanup_command_paths,
        rank_pid_paths=rank_pid_paths,
        rank_pgid_paths=rank_pgid_paths,
        rank_log_paths=rank_log_paths,
        rank_launch_log_paths=rank_launch_log_paths,
    )


def effective_max_num_batched_tokens(resolved: ResolvedCase) -> int | None:
    if resolved.case.max_num_batched_tokens is not None:
        return resolved.case.max_num_batched_tokens
    return resolved.strategy.max_num_batched_tokens


def expand_command_arg_templates(
    extra_args: tuple[str, ...],
    *,
    benchmark_dir: Path | None = None,
) -> tuple[str, ...]:
    if benchmark_dir is None:
        return extra_args

    benchmark_dir_text = str(benchmark_dir)
    return tuple(
        arg.replace("{benchmark_dir}", benchmark_dir_text)
        for arg in extra_args
    )


def apply_extra_argv_overrides(
    cases: list[ExperimentCase],
    *,
    frontend_extra_args: tuple[str, ...] = (),
    headless_extra_args: tuple[str, ...] = (),
) -> list[ExperimentCase]:
    if not frontend_extra_args and not headless_extra_args:
        return cases

    return [
        replace(
            case,
            frontend_extra_args=case.frontend_extra_args +
            frontend_extra_args,
            headless_extra_args=case.headless_extra_args +
            headless_extra_args,
        ) for case in cases
    ]


def build_common_harness_argv(
    resolved: ResolvedCase,
    *,
    role: str,
    node_rank: int,
) -> list[str]:
    dispatch_policy = normalize_dispatch_policy(resolved.case.dispatch_policy)
    argv = [
        "python3",
        HARNESS_ENTRYPOINT,
        role,
        "--model",
        str(resolved.model_path),
        "--master-addr",
        resolved.cluster.master_addr,
        "--master-port",
        str(resolved.cluster.master_port),
        "--nnodes",
        str(resolved.cluster.nnodes),
        "--node-rank",
        str(node_rank),
        "--data-parallel-size",
        str(resolved.strategy.data_parallel_size),
        "--data-parallel-size-local",
        str(resolved.strategy.data_parallel_size_local),
        "--data-parallel-rpc-port",
        str(resolved.case.data_parallel_rpc_port),
        "--data-parallel-backend",
        resolved.strategy.data_parallel_backend,
        "--data-parallel-dispatch-policy",
        dispatch_policy,
        "--tensor-parallel-size",
        str(resolved.strategy.tensor_parallel_size),
        "--decode-context-parallel-size",
        str(resolved.strategy.decode_context_parallel_size),
        bool_flag("enable-expert-parallel",
                  resolved.strategy.enable_expert_parallel),
    ]

    if resolved.strategy.attention_backend is not None:
        argv.extend(["--attention-backend", resolved.strategy.attention_backend])
    if resolved.strategy.all2all_backend is not None:
        argv.extend(["--all2all-backend", resolved.strategy.all2all_backend])
    if resolved.strategy.dcp_comm_backend is not None:
        argv.extend(["--dcp-comm-backend", resolved.strategy.dcp_comm_backend])
    argv.extend([
        "--gpu-memory-utilization",
        f"{resolved.case.gpu_memory_utilization:g}",
    ])
    if resolved.case.max_num_seqs is not None:
        argv.extend(["--max-num-seqs", str(resolved.case.max_num_seqs)])
    if resolved.case.max_model_len is not None:
        argv.extend(["--max-model-len", str(resolved.case.max_model_len)])
    max_num_batched_tokens = effective_max_num_batched_tokens(resolved)
    if max_num_batched_tokens is not None:
        argv.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])

    argv.extend(resolved.case.shared_cli_args)
    return argv


def build_frontend_argv(resolved: ResolvedCase, output_dir: Path) -> list[str]:
    argv = build_common_harness_argv(
        resolved,
        role="frontend",
        node_rank=0,
    )
    argv.extend([
        "--input-csv",
        str(resolved.dataset_path),
        "--request-rate",
        stringify_request_rate(resolved.case.request_rate),
        "--warmup-requests",
        str(resolved.case.warmup_requests),
        "--output-dir",
        str(output_dir),
        "--progress-log-interval",
        str(resolved.case.progress_log_interval),
        "--seed",
        str(resolved.case.seed),
    ])
    if resolved.case.max_requests is not None:
        argv.extend(["--max-requests", str(resolved.case.max_requests)])
    if resolved.case.request_id_prefix is not None:
        argv.extend(["--request-id-prefix", resolved.case.request_id_prefix])
    if resolved.case.save_merged_parquet:
        argv.append("--save-merged-parquet")

    argv.extend(
        expand_command_arg_templates(
            resolved.case.frontend_extra_args,
            benchmark_dir=output_dir,
        ))
    return argv


def build_headless_argv(
    resolved: ResolvedCase,
    node_rank: int,
    benchmark_dir: Path | None = None,
) -> list[str]:
    argv = build_common_harness_argv(
        resolved,
        role="headless-engine",
        node_rank=node_rank,
    )
    argv.extend(
        expand_command_arg_templates(
            resolved.case.headless_extra_args,
            benchmark_dir=benchmark_dir,
        ))
    return argv


def collect_role_env(
    resolved: ResolvedCase,
    *,
    role: str,
    forward_process_env: bool = True,
) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    if forward_process_env:
        for key in FORWARDED_ENV_KEYS:
            value = os.environ.get(key)
            if value:
                forwarded[key] = value
    forwarded.update(DEFAULT_ENV_OVERRIDES)
    forwarded["CUDA_VISIBLE_DEVICES"] = resolved.case.cuda_visible_devices
    forwarded.update({key: str(value) for key, value in resolved.case.env.items()})
    role_overrides = (resolved.case.frontend_env if role == "frontend" else
                      resolved.case.headless_env)
    forwarded.update({key: str(value) for key, value in role_overrides.items()})
    return forwarded


def build_runtime_shell_command(
    *,
    cwd: str,
    argv: list[str],
    env: Mapping[str, str],
    env_script: str | None,
    log_path: Path,
    pid_path: Path,
    pgid_path: Path,
) -> str:
    env_prefix = " ".join(
        f"{key}={shlex.quote(str(value))}" for key, value in env.items())
    command = shell_join(argv)
    if env_prefix:
        command = f"env {env_prefix} {command}"

    steps = [
        f"cd {shlex.quote(cwd)}",
        f"rm -f {shlex.quote(str(pid_path))} >/dev/null 2>&1 || true",
        f"rm -f {shlex.quote(str(pgid_path))} >/dev/null 2>&1 || true",
    ]
    if env_script:
        steps.append(f"source {shlex.quote(env_script)}")
    steps.extend([
        "set +e",
        f"{command} > {shlex.quote(str(log_path))} 2>&1 &",
        "child_pid=$!",
        f"printf '%s\\n' \"$child_pid\" > {shlex.quote(str(pid_path))}",
        f"printf '%s\\n' \"$$\" > {shlex.quote(str(pgid_path))}",
        "wait \"$child_pid\"",
        "child_exit_code=$?",
        "exit \"$child_exit_code\"",
    ])
    return "\n".join(steps)


def build_node_cleanup_command(
    *,
    resolved: ResolvedCase,
    pid_path: Path,
    pgid_path: Path,
    include_frontend_pattern: bool,
) -> str:
    def append_pattern_cleanup(
        pattern: str,
        *,
        wait_pattern: str | None = None,
    ) -> None:
        wait_pattern = wait_pattern or pattern
        lines.append(
            f"pkill -TERM -f -- {shlex.quote(pattern)} >/dev/null 2>&1 || true")
        lines.extend([
            "for _ in 1 2 3 4 5; do",
            f"  if ! pgrep -f -- {shlex.quote(wait_pattern)} >/dev/null 2>&1; then",
            "    break",
            "  fi",
            "  sleep 1",
            "done",
            f"pkill -KILL -f -- {shlex.quote(wait_pattern)} >/dev/null 2>&1 || true",
        ])

    lines = [
        "set +e",
        f"if [ -f {shlex.quote(str(pgid_path))} ]; then",
        f"  pgid=$(cat {shlex.quote(str(pgid_path))} 2>/dev/null || true)",
        "  if [ -n \"$pgid\" ] && kill -0 -- \"-$pgid\" >/dev/null 2>&1; then",
        "    kill -TERM -- \"-$pgid\" >/dev/null 2>&1 || true",
        "    sleep 5",
        "    if kill -0 -- \"-$pgid\" >/dev/null 2>&1; then",
        "      kill -KILL -- \"-$pgid\" >/dev/null 2>&1 || true",
        "    fi",
        "  fi",
        "fi",
        f"if [ -f {shlex.quote(str(pid_path))} ]; then",
        f"  pid=$(cat {shlex.quote(str(pid_path))} 2>/dev/null || true)",
        "  if [ -n \"$pid\" ] && kill -0 \"$pid\" >/dev/null 2>&1; then",
        "    kill -TERM \"$pid\" >/dev/null 2>&1 || true",
        "    sleep 5",
        "    if kill -0 \"$pid\" >/dev/null 2>&1; then",
        "      kill -KILL \"$pid\" >/dev/null 2>&1 || true",
        "    fi",
        "  fi",
        "fi",
    ]

    append_pattern_cleanup(
        f"python3 {HARNESS_ENTRYPOINT} headless-engine",
        wait_pattern=f"^python3 {HARNESS_ENTRYPOINT} headless-engine( |$)",
    )
    if include_frontend_pattern:
        append_pattern_cleanup(
            f"python3 {HARNESS_ENTRYPOINT} frontend",
            wait_pattern=f"^python3 {HARNESS_ENTRYPOINT} frontend( |$)",
        )

    append_pattern_cleanup("^VLLM::EngineCore")
    append_pattern_cleanup("^VLLM::DPCoordinator$")
    append_pattern_cleanup("^VLLM::APIServer$")
    append_pattern_cleanup("^VLLM::Worker_")
    append_pattern_cleanup(
        ("vllm serve .*--data-parallel-rpc-port "
         f"{resolved.case.data_parallel_rpc_port}([[:space:]]|$)"))
    for pattern in (
            f"--master-port {resolved.cluster.master_port}",
            f"--data-parallel-rpc-port {resolved.case.data_parallel_rpc_port}"):
        lines.append(
            f"pkill -f -- {shlex.quote(pattern)} >/dev/null 2>&1 || true")
    if resolved.case.cleanup_worker_processes:
        append_pattern_cleanup("^VLLM::Worker_")
    return "\n".join(lines)


def build_pidfile_probe_command(pid_path: Path) -> str:
    return "\n".join([
        "set +e",
        f"if [ -f {shlex.quote(str(pid_path))} ]; then",
        f"  pid=$(cat {shlex.quote(str(pid_path))} 2>/dev/null || true)",
        "  if [ -n \"$pid\" ] && kill -0 \"$pid\" >/dev/null 2>&1; then",
        "    exit 0",
        "  fi",
        "fi",
        "exit 1",
    ])


def build_local_launch_command(resolved: ResolvedCase, artifacts: ArtifactPaths,
                               frontend_argv: list[str]) -> str:
    return build_runtime_shell_command(
        cwd=resolved.cluster.workdir,
        argv=frontend_argv,
        env=collect_role_env(
            resolved,
            role="frontend",
            forward_process_env=resolved.cluster.local_env_script is None,
        ),
        env_script=resolved.cluster.local_env_script,
        log_path=artifacts.frontend_log_path,
        pid_path=artifacts.frontend_pid_path,
        pgid_path=artifacts.frontend_pgid_path,
    )


def build_remote_launch_command(
    resolved: ResolvedCase,
    artifacts: ArtifactPaths,
    *,
    node_rank: int,
) -> str:
    return build_runtime_shell_command(
        cwd=resolved.cluster.workdir,
        argv=build_headless_argv(
            resolved,
            node_rank,
            benchmark_dir=artifacts.benchmark_dir,
        ),
        # The remote shell sources its own environment script. Do not copy the
        # head node's PATH, LD_LIBRARY_PATH, NCCL, or NVSHMEM settings over it.
        env=collect_role_env(
            resolved,
            role="headless-engine",
            forward_process_env=resolved.cluster.remote_env_script is None,
        ),
        env_script=resolved.cluster.remote_env_script,
        log_path=artifacts.rank_log_paths[node_rank],
        pid_path=artifacts.rank_pid_paths[node_rank],
        pgid_path=artifacts.rank_pgid_paths[node_rank],
    )


def build_remote_ssh_command(cluster: ClusterSpec, host: str,
                             remote_command: str) -> list[str]:
    remote_shell_argv = shell_command_argv(
        cluster.remote_shell,
        cluster.remote_shell_flags,
        remote_command,
    )
    return [
        "ssh",
        *cluster.ssh_opts,
        host,
        shell_join(remote_shell_argv),
    ]


def write_case_manifest(
    resolved: ResolvedCase,
    artifacts: ArtifactPaths,
    *,
    frontend_launch_command: str,
    remote_launch_commands: dict[int, str],
    local_cleanup_command: str,
    remote_cleanup_commands: dict[int, str],
    status: str,
    started_at: str | None,
    finished_at: str | None,
    exit_code: int | None,
    detail: str | None,
) -> None:
    payload = {
        "case_name": resolved.case.name,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": exit_code,
        "detail": detail,
        "shared_artifact_mount": True,
        "cluster_name": resolved.cluster_name,
        "strategy_name": resolved.strategy_name,
        "cluster": resolved.cluster,
        "strategy": resolved.strategy,
        "case": resolved.case,
        "model_path": resolved.model_path,
        "dataset_path": resolved.dataset_path,
        "paths": artifacts,
        "commands": {
            "frontend_launch_command": frontend_launch_command,
            "remote_launch_commands": remote_launch_commands,
            "frontend_cleanup_command": local_cleanup_command,
            "remote_cleanup_commands": remote_cleanup_commands,
        },
    }
    write_json(artifacts.case_manifest_path, payload)


def write_case_command_files(
    resolved: ResolvedCase,
    artifacts: ArtifactPaths,
    *,
    frontend_launch_command: str,
    remote_launch_commands: dict[int, str],
    local_cleanup_command: str,
    remote_cleanup_commands: dict[int, str],
) -> None:
    write_shell_script(artifacts.frontend_command_path, frontend_launch_command)
    write_shell_script(artifacts.frontend_cleanup_command_path,
                       local_cleanup_command)
    for node_rank, command in remote_launch_commands.items():
        write_shell_script(artifacts.rank_command_paths[node_rank], command)
    for node_rank, command in remote_cleanup_commands.items():
        write_shell_script(artifacts.rank_cleanup_command_paths[node_rank], command)


def select_cases(args: argparse.Namespace,
                 experiments: list[ExperimentCase]) -> list[ExperimentCase]:
    if args.case_csv and (args.all or args.case):
        raise SystemExit("Use either --case-csv or --all/--case, not both.")
    if args.case_csv and (args.model or args.dataset or args.strategy):
        raise SystemExit(
            "Use either --case-csv or --model/--dataset/--strategy filters, "
            "not both.")
    if args.case_csv:
        return load_cases_from_csv(Path(args.case_csv))

    available = experiments_by_name(experiments)
    if args.all and args.case:
        raise SystemExit("Use either --all or --case, not both.")
    if not args.all and not args.case and not args.list:
        raise SystemExit("Select at least one case via --case or use --all.")
    if args.all:
        return list(experiments)
    selected: list[ExperimentCase] = []
    seen: set[str] = set()
    for case_name in args.case or []:
        if case_name in seen:
            continue
        try:
            selected.append(available[case_name])
        except KeyError as exc:
            supported = ", ".join(sorted(available))
            raise SystemExit(
                f"Unknown case '{case_name}'. Supported values: {supported}"
            ) from exc
        seen.add(case_name)
    return selected


def apply_bench_duration_override(
    cases: list[ExperimentCase],
    bench_duration_sec: float | None,
) -> list[ExperimentCase]:
    if bench_duration_sec is None:
        return cases

    overridden: list[ExperimentCase] = []
    for case in cases:
        if math.isinf(case.request_rate):
            raise SystemExit(
                "--bench-duration-sec does not support cases with request_rate=inf."
            )
        overridden.append(
            replace(
                case,
                max_requests=bench_duration_to_max_requests(
                    case.request_rate,
                    bench_duration_sec,
                ),
            ))
    return overridden


def describe_case(case: ExperimentCase) -> str:
    return (
        f"{case.name}: model={model_short_name(case.model)}, "
        f"cluster={case.cluster}, strategy={case.strategy}, "
        f"dispatch={normalize_dispatch_policy(case.dispatch_policy)}, "
        f"dataset={case.dataset}, phase={case.rate_phase}, "
        f"rate={stringify_request_rate(case.request_rate)}, "
        f"mem={stringify_gpu_memory_utilization(case.gpu_memory_utilization)}, "
        f"bs={case.max_num_seqs}"
    )


def generate_case_name(
    *,
    model: str,
    dataset: str,
    strategy: str,
    request_rate: float,
    max_num_seqs: int | None,
) -> str:
    bs_tag = f"bs{max_num_seqs}" if max_num_seqs is not None else "bsauto"
    return (
        f"{model_short_name(model)}__{dataset_short_name(dataset)}__"
        f"{strategy}__rate{stringify_request_rate(request_rate)}__{bs_tag}"
    )


def _csv_cell(row: Mapping[str, str], key: str) -> str:
    return (row.get(key) or "").strip()


def _csv_bool(row: Mapping[str, str], key: str, *,
              default: bool) -> bool:
    raw = _csv_cell(row, key)
    if not raw:
        return default
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value '{raw}' for column '{key}'")


def _csv_int(row: Mapping[str, str], key: str, *,
             default: int | None = None) -> int | None:
    raw = _csv_cell(row, key)
    if not raw:
        return default
    return int(raw)


def _csv_max_requests(
    row: Mapping[str, str],
    key: str,
    *,
    default: MaxRequestsValue = None,
) -> MaxRequestsValue:
    raw = _csv_cell(row, key)
    if not raw:
        return default
    lowered = raw.lower()
    if lowered in {"csv", "csv_rows", "all_csv_rows"}:
        return MAX_REQUESTS_CSV_ROWS
    return int(raw)


def _csv_float(row: Mapping[str, str], key: str, *,
               default: float | None = None) -> float | None:
    raw = _csv_cell(row, key)
    if not raw:
        return default
    return float(raw)


def load_cases_from_csv(path: Path) -> list[ExperimentCase]:
    if not path.is_file():
        raise SystemExit(f"Case CSV not found: {path}")

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise SystemExit(f"Case CSV has no header row: {path}")

            rows = list(reader)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"Failed to read case CSV {path}: {exc}") from exc

    cases: list[ExperimentCase] = []
    for row_index, row in enumerate(rows, start=2):
        try:
            if not _csv_bool(row, "enabled", default=True):
                continue

            name = _csv_cell(row, "name")
            cluster = _csv_cell(row, "cluster")
            model = _csv_cell(row, "model")
            dataset = _csv_cell(row, "dataset")
            strategy = _csv_cell(row, "strategy")
            request_rate_raw = _csv_cell(row, "request_rate")
            if not all((cluster, model, dataset, strategy, request_rate_raw)):
                raise ValueError(
                    "missing one of required columns: "
                    "cluster, model, dataset, strategy, request_rate")

            request_rate = float(request_rate_raw)
            max_num_seqs = _csv_int(row, "max_num_seqs")
            if not name:
                name = generate_case_name(
                    model=model,
                    dataset=dataset,
                    strategy=strategy,
                    request_rate=request_rate,
                    max_num_seqs=max_num_seqs,
                )

            cases.append(
                ExperimentCase(
                    name=name,
                    cluster=cluster,
                    strategy=strategy,
                    dataset=dataset,
                    model=model,
                    dispatch_policy=(
                        _csv_cell(row, "dispatch_policy")
                        or DEFAULT_DISPATCH_POLICY
                    ),
                    request_rate=request_rate,
                    rate_phase=_csv_cell(row, "rate_phase") or DEFAULT_RATE_PLAN,
                    gpu_memory_utilization=_csv_float(
                        row,
                        "gpu_memory_utilization",
                        default=DEFAULT_GPU_MEMORY_UTILIZATION,
                    ) or DEFAULT_GPU_MEMORY_UTILIZATION,
                    max_requests=_csv_max_requests(row, "max_requests"),
                    warmup_requests=_csv_int(row, "warmup_requests",
                                             default=0) or 0,
                    max_num_seqs=max_num_seqs,
                    max_model_len=_csv_int(row, "max_model_len"),
                    data_parallel_rpc_port=_csv_int(
                        row,
                        "data_parallel_rpc_port",
                        default=29550,
                    ) or 29550,
                ))
            normalize_dispatch_policy(cases[-1].dispatch_policy)
        except Exception as exc:
            raise SystemExit(
                f"Invalid case CSV row {row_index} in {path}: {exc}") from exc
    return cases


def _metric_summary(records: list[dict[str, Any]],
                    key: str) -> dict[str, float] | None:
    import numpy as np

    values = [
        float(record[key]) for record in records
        if not record.get("is_error") and record.get(key) is not None
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


def _compute_tpot_metrics(
    record: Mapping[str, Any]
) -> tuple[float | None, float | None]:
    if record.get("is_error"):
        return None, None

    actual_output_tokens = int(record.get("actual_output_tokens") or 0)
    if actual_output_tokens <= 0:
        return None, None

    decode_time_ms = record.get("decode_time_ms")
    e2e_ms = record.get("e2e_ms")

    tpot_without_queue_ms = None
    decode_output_tokens = actual_output_tokens - 1
    if decode_time_ms is not None and decode_output_tokens > 0:
        tpot_without_queue_ms = float(decode_time_ms) / decode_output_tokens

    tpot_by_e2e = None
    if e2e_ms is not None:
        tpot_by_e2e = float(e2e_ms) / actual_output_tokens

    return tpot_without_queue_ms, tpot_by_e2e


def augment_benchmark_outputs(benchmark_dir: Path) -> None:
    requests_path = benchmark_dir / "requests.jsonl"
    summary_path = benchmark_dir / "summary.json"

    if not requests_path.is_file():
        raise FileNotFoundError(f"Benchmark requests file not found: {requests_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Benchmark summary file not found: {summary_path}")

    records = load_jsonl(requests_path)
    for record in records:
        (record["tpot_without_queue_ms"],
         record["tpot_by_e2e"]) = _compute_tpot_metrics(record)
        record.pop("tpot_with_initial_queue_ms", None)

    summary = load_json(summary_path)
    summary["tpot_without_queue_ms"] = _metric_summary(records,
                                                       "tpot_without_queue_ms")
    summary["tpot_by_e2e"] = _metric_summary(records, "tpot_by_e2e")
    summary.pop("tpot_with_initial_queue_ms", None)

    write_jsonl(requests_path, records)
    write_json(summary_path, summary)


def extract_summary_tpot_by_e2e_mean(summary: Mapping[str, Any]) -> float | None:
    return extract_summary_metric_value(summary,
                                        metric_key=TREND_RERUN_METRIC_KEY,
                                        submetric=TREND_RERUN_SUBMETRIC)


def extract_summary_metric_value(summary: Mapping[str, Any], *,
                                 metric_key: str,
                                 submetric: str) -> float | None:
    metric = summary.get(metric_key)
    if not isinstance(metric, Mapping):
        return None

    raw_value = metric.get(submetric)
    if raw_value is None:
        return None

    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(parsed):
        return None
    return parsed


def load_case_result_summary_metric(
    result: CaseResult,
    *,
    metric_key: str = TREND_RERUN_METRIC_KEY,
    submetric: str = TREND_RERUN_SUBMETRIC,
) -> float | None:
    try:
        summary = load_json(result.summary_json)
    except Exception:
        return None
    return extract_summary_metric_value(summary,
                                        metric_key=metric_key,
                                        submetric=submetric)


def interpolate_case_metric(lower_case: ExperimentCase,
                            lower_value: float,
                            upper_case: ExperimentCase,
                            upper_value: float,
                            current_case: ExperimentCase) -> float | None:
    span = upper_case.request_rate - lower_case.request_rate
    if math.isclose(span, 0.0):
        return None
    weight = ((current_case.request_rate - lower_case.request_rate) / span)
    return lower_value + (upper_value - lower_value) * weight


def detect_group_trend_outlier_candidates(
    group_cases: list[ExperimentCase],
    latest_results_by_case_name: Mapping[str, CaseResult],
    *,
    metric_key: str = TREND_RERUN_METRIC_KEY,
    submetric: str = TREND_RERUN_SUBMETRIC,
    relative_threshold_pct: float = DEFAULT_TREND_RERUN_RELATIVE_THRESHOLD_PCT,
    absolute_threshold_ms: float = DEFAULT_TREND_RERUN_ABSOLUTE_THRESHOLD_MS,
) -> list[TrendOutlierCandidate]:
    if len(group_cases) < 3:
        return []

    ordered_cases = sorted(group_cases, key=lambda case: case.request_rate)
    successful_points: list[tuple[ExperimentCase, CaseResult, float]] = []
    for case in ordered_cases:
        result = latest_results_by_case_name.get(case.name)
        if result is None or result.status != "ok":
            continue
        metric_value = load_case_result_summary_metric(
            result,
            metric_key=metric_key,
            submetric=submetric,
        )
        if metric_value is None:
            continue
        successful_points.append((case, result, metric_value))

    if len(successful_points) < 3:
        return []

    candidates: list[TrendOutlierCandidate] = []
    for index in range(1, len(successful_points) - 1):
        lower_case, lower_result, lower_value = successful_points[index - 1]
        current_case, current_result, current_value = successful_points[index]
        upper_case, upper_result, upper_value = successful_points[index + 1]

        expected_value = interpolate_case_metric(lower_case, lower_value,
                                                 upper_case, upper_value,
                                                 current_case)
        if expected_value is None:
            continue

        delta_ms = abs(current_value - expected_value)
        allowed_delta_ms = max(
            absolute_threshold_ms,
            abs(expected_value) * (relative_threshold_pct / 100.0),
        )
        lower_bound = min(lower_value, upper_value) - allowed_delta_ms
        upper_bound = max(lower_value, upper_value) + allowed_delta_ms
        if lower_bound <= current_value <= upper_bound:
            continue

        candidates.append(
            TrendOutlierCandidate(
                case=current_case,
                result=current_result,
                lower_case=lower_case,
                lower_result=lower_result,
                upper_case=upper_case,
                upper_result=upper_result,
                actual_value=current_value,
                expected_value=expected_value,
                allowed_delta_ms=allowed_delta_ms,
                delta_ms=delta_ms,
            ))
    return sorted(candidates, key=lambda item: item.delta_ms, reverse=True)


def should_stop_followup_rates(result: CaseResult) -> tuple[bool, str | None]:
    if result.status == "dry_run":
        return False, None

    if result.status != "ok":
        return True, f"case status={result.status}"

    try:
        summary = load_json(result.summary_json)
    except Exception as exc:
        return True, f"unable to read summary.json: {exc}"

    tpot_by_e2e_mean = extract_summary_tpot_by_e2e_mean(summary)
    if tpot_by_e2e_mean is None:
        return True, "missing tpot_by_e2e.mean"

    if tpot_by_e2e_mean > TPOT_BY_E2E_EARLY_STOP_MS:
        return (
            True,
            ("tpot_by_e2e.mean="
             f"{tpot_by_e2e_mean:.3f}ms > {TPOT_BY_E2E_EARLY_STOP_MS:g}ms"),
        )

    return False, None


def benchmark_summary_indicates_success(summary: Mapping[str, Any]) -> bool:
    try:
        total_requests = int(summary.get("total_requests"))
        successful_requests = int(summary.get("successful_requests"))
        failed_requests = int(summary.get("failed_requests", 0))
        failure_ratio = float(summary.get("failure_ratio", 0.0))
    except (TypeError, ValueError):
        return False

    if total_requests <= 0:
        return False
    if successful_requests != total_requests:
        return False
    if failed_requests != 0:
        return False
    if not math.isclose(failure_ratio, 0.0, abs_tol=1e-12):
        return False
    return True


def validate_benchmark_summary(benchmark_dir: Path) -> None:
    summary_path = benchmark_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Benchmark summary file not found: {summary_path}")

    summary = load_json(summary_path)
    if benchmark_summary_indicates_success(summary):
        return

    raise RuntimeError(
        "benchmark summary indicates failure:\n"
        f"total_requests={summary.get('total_requests')} "
        f"successful_requests={summary.get('successful_requests')} "
        f"failed_requests={summary.get('failed_requests')} "
        f"failure_ratio={summary.get('failure_ratio')}")


def iso_timestamp_for_path(path: Path) -> str | None:
    try:
        stat_result = path.stat()
    except OSError:
        return None
    return datetime.fromtimestamp(stat_result.st_mtime).astimezone().isoformat(
        timespec="seconds")


def benchmark_started_at_from_run_meta(benchmark_dir: Path) -> str | None:
    run_meta_path = benchmark_dir / "run_meta.json"
    if not run_meta_path.is_file():
        return None

    try:
        run_meta = load_json(run_meta_path)
    except Exception:
        return None

    started_at = run_meta.get("benchmark_start_time")
    return started_at if isinstance(started_at, str) and started_at else None


def maybe_repair_case_manifest_as_success(case_dir: Path,
                                          manifest: dict[str, Any] | None) -> None:
    if manifest is None:
        return

    updated_manifest = dict(manifest)
    benchmark_dir = case_dir / "benchmark"
    manifest_was_ok = updated_manifest.get("status") == "ok"
    if not manifest_was_ok:
        updated_manifest["status"] = "ok"
        updated_manifest["exit_code"] = 0
        updated_manifest["detail"] = (
            "completed (recovered from successful benchmark artifacts)")
    if not updated_manifest.get("started_at"):
        updated_manifest["started_at"] = (benchmark_started_at_from_run_meta(
            benchmark_dir) or iso_timestamp_for_path(case_dir))
    if not updated_manifest.get("finished_at"):
        updated_manifest["finished_at"] = (iso_timestamp_for_path(
            benchmark_dir / "summary.json") or iso_timestamp_for_path(
                benchmark_dir / "requests.jsonl") or iso_timestamp_for_path(
                    case_dir))

    if updated_manifest != manifest:
        write_json(case_dir / "case_manifest.json", updated_manifest)


def maybe_finalize_interrupted_case_manifest(
    case_dir: Path,
    *,
    manifest: dict[str, Any] | None,
    signal_name: str,
    exit_code: int,
    finished_at: str,
) -> None:
    if complete_successful_benchmark_tpot_by_e2e_mean(case_dir,
                                                      manifest=manifest) is not None:
        return
    if manifest is None:
        return
    if manifest.get("status") not in {"prepared", "running"}:
        return

    updated_manifest = dict(manifest)
    updated_manifest["status"] = "interrupted"
    updated_manifest["exit_code"] = exit_code

    interrupt_detail = f"interrupted by {signal_name}"
    detail = updated_manifest.get("detail")
    if not detail:
        updated_manifest["detail"] = interrupt_detail
    elif interrupt_detail not in str(detail):
        updated_manifest["detail"] = f"{detail}\n{interrupt_detail}"

    if not updated_manifest.get("started_at"):
        updated_manifest["started_at"] = (benchmark_started_at_from_run_meta(
            case_dir / "benchmark") or iso_timestamp_for_path(case_dir))
    updated_manifest["finished_at"] = finished_at
    write_json(case_dir / "case_manifest.json", updated_manifest)


def complete_successful_benchmark_tpot_by_e2e_mean(
        case_dir: Path,
        *,
        manifest: dict[str, Any] | None = None) -> float | None:
    benchmark_dir = case_dir / "benchmark"
    summary_path = benchmark_dir / "summary.json"
    requests_path = benchmark_dir / "requests.jsonl"
    if not summary_path.is_file() or not requests_path.is_file():
        return None

    try:
        summary = load_json(summary_path)
    except Exception:
        return None

    if not benchmark_summary_indicates_success(summary):
        return None

    tpot_by_e2e_mean = extract_summary_tpot_by_e2e_mean(summary)
    if tpot_by_e2e_mean is None:
        try:
            augment_benchmark_outputs(benchmark_dir)
            summary = load_json(summary_path)
        except Exception:
            pass
        tpot_by_e2e_mean = extract_summary_tpot_by_e2e_mean(summary)

    maybe_repair_case_manifest_as_success(case_dir, manifest)
    return tpot_by_e2e_mean


def case_dir_has_complete_successful_benchmark(case_dir: Path) -> bool:
    benchmark_dir = case_dir / "benchmark"
    summary_path = benchmark_dir / "summary.json"
    requests_path = benchmark_dir / "requests.jsonl"
    if not summary_path.is_file() or not requests_path.is_file():
        return False

    try:
        summary = load_json(summary_path)
    except Exception:
        return False

    return benchmark_summary_indicates_success(summary)


def latest_successful_case_dir(case_group_dir: Path) -> Path | None:
    if not case_group_dir.is_dir():
        return None

    candidates = sorted((path for path in case_group_dir.iterdir() if path.is_dir()),
                        reverse=True)
    for case_dir in candidates:
        manifest_path = case_dir / "case_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = load_json(manifest_path)
            except Exception:
                manifest = None
            if manifest is not None and manifest.get("status") == "ok":
                return case_dir

        # Interrupted runs can leave a non-ok manifest behind even when the
        # benchmark itself finished cleanly. Accept those directories as
        # historical successes so exact-case reruns still skip them.
        if case_dir_has_complete_successful_benchmark(case_dir):
            return case_dir
    return None


def latest_case_dir(case_group_dir: Path) -> Path | None:
    if not case_group_dir.is_dir():
        return None

    candidates = sorted((path for path in case_group_dir.iterdir() if path.is_dir()),
                        reverse=True)
    return candidates[0] if candidates else None


def latest_successful_case_tpot_by_e2e_mean(case_group_dir: Path) -> tuple[
        Path, float | None] | None:
    case_dir = latest_successful_case_dir(case_group_dir)
    if case_dir is None:
        return None

    manifest_path = case_dir / "case_manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        try:
            manifest = load_json(manifest_path)
        except Exception:
            manifest = None

    recovered_tpot_by_e2e_mean = complete_successful_benchmark_tpot_by_e2e_mean(
        case_dir, manifest=manifest)
    if recovered_tpot_by_e2e_mean is not None:
        return case_dir, recovered_tpot_by_e2e_mean

    summary_path = case_dir / "benchmark" / "summary.json"
    if not summary_path.is_file():
        return case_dir, None

    try:
        summary = load_json(summary_path)
    except Exception:
        return case_dir, None

    return case_dir, extract_summary_tpot_by_e2e_mean(summary)


def latest_case_status(case_group_dir: Path) -> tuple[Path, str | None] | None:
    case_dir = latest_case_dir(case_group_dir)
    if case_dir is None:
        return None

    manifest = load_optional_json(case_dir / "case_manifest.json")
    if manifest is None:
        return case_dir, None
    status = manifest.get("status")
    return case_dir, status if isinstance(status, str) and status else None


def historical_case_group_dirs(
    artifact_root: Path,
    case: ExperimentCase,
    *,
    ignore_bs: bool,
) -> tuple[Path, ...]:
    exact_group_dir = case_artifact_group_dir(artifact_root, case)
    dataset_dir = case_artifact_dataset_dir(artifact_root, case)
    if not dataset_dir.is_dir():
        return (exact_group_dir, )

    scenario_prefix = f"{case_artifact_scenario_match_prefix(case)}-"
    scenario_suffix = f"-{case_artifact_rate_duration_tag(case)}"
    if not ignore_bs:
        scenario_suffix = (f"-{case_artifact_batch_size_tag(case)}"
                           f"{scenario_suffix}")
    matching_group_dirs = tuple(
        sorted((path for path in dataset_dir.iterdir()
                if path.is_dir() and path.name.startswith(scenario_prefix)
                and path.name.endswith(scenario_suffix)),
               key=lambda path: path.name,
               reverse=True))
    return matching_group_dirs or (exact_group_dir, )


def historical_group_dir_match_suffix(
    case: ExperimentCase,
    *,
    matched_group_dir: Path,
    exact_group_dir: Path,
) -> str:
    if matched_group_dir == exact_group_dir:
        return ""

    differences: list[str] = []
    memory_tag = stringify_gpu_memory_utilization(case.gpu_memory_utilization)
    if f"-{memory_tag}-" not in matched_group_dir.name:
        differences.append("mem")

    batch_size_tag = case_artifact_batch_size_tag(case)
    if f"-{batch_size_tag}-" not in matched_group_dir.name:
        differences.append("bs")

    if not differences:
        return f" under {matched_group_dir.name}"
    if len(differences) == 1:
        difference_text = differences[0]
    else:
        difference_text = " and ".join(differences)
    return (f" under {matched_group_dir.name} "
            f"(matched with different {difference_text})")


def latest_successful_case_tpot_by_e2e_mean_for_case(
    artifact_root: Path,
    case: ExperimentCase,
    *,
    ignore_bs: bool,
) -> tuple[Path, float | None] | None:
    exact_group_dir = case_artifact_group_dir(artifact_root, case)
    best_match: tuple[Path, float | None] | None = None
    best_sort_key: tuple[str, int, str] | None = None
    for case_group_dir in historical_case_group_dirs(artifact_root,
                                                     case,
                                                     ignore_bs=ignore_bs):
        candidate = latest_successful_case_tpot_by_e2e_mean(case_group_dir)
        if candidate is None:
            continue
        candidate_sort_key = (
            candidate[0].name,
            int(candidate[0].parent == exact_group_dir),
            candidate[0].parent.name,
        )
        if best_sort_key is None or candidate_sort_key > best_sort_key:
            best_match = candidate
            best_sort_key = candidate_sort_key
    return best_match


def latest_case_status_for_case(
    artifact_root: Path,
    case: ExperimentCase,
    *,
    ignore_bs: bool,
) -> tuple[Path, str | None] | None:
    exact_group_dir = case_artifact_group_dir(artifact_root, case)
    best_match: tuple[Path, str | None] | None = None
    best_sort_key: tuple[str, int, str] | None = None
    for case_group_dir in historical_case_group_dirs(artifact_root,
                                                     case,
                                                     ignore_bs=ignore_bs):
        candidate = latest_case_status(case_group_dir)
        if candidate is None:
            continue
        candidate_sort_key = (
            candidate[0].name,
            int(candidate[0].parent == exact_group_dir),
            candidate[0].parent.name,
        )
        if best_sort_key is None or candidate_sort_key > best_sort_key:
            best_match = candidate
            best_sort_key = candidate_sort_key
    return best_match


def build_historical_skip_state(
    artifact_root: Path,
    selected_cases: list[ExperimentCase],
    *,
    ignore_bs: bool,
) -> tuple[dict[str, str], dict[str, tuple[float, str]]]:
    exact_case_skips: dict[str, str] = {}
    blocked_group_rates: dict[str, tuple[float, str]] = {}

    for case in selected_cases:
        group_key = case_group_key(case)
        exact_group_dir = case_artifact_group_dir(artifact_root, case)
        historical = latest_successful_case_tpot_by_e2e_mean_for_case(
            artifact_root, case, ignore_bs=ignore_bs)
        if historical is not None:
            case_dir, tpot_by_e2e_mean = historical
            case_result_tag = case_dir.name
            exact_reason = (
                f"latest successful result already exists at {case_result_tag}")
            matched_group_dir = case_dir.parent
            exact_reason += historical_group_dir_match_suffix(
                case,
                matched_group_dir=matched_group_dir,
                exact_group_dir=exact_group_dir,
            )
            if tpot_by_e2e_mean is not None:
                exact_reason += f" (tpot_by_e2e.mean={tpot_by_e2e_mean:.3f}ms)"
            exact_case_skips[case.name] = exact_reason

            if tpot_by_e2e_mean is not None:
                if tpot_by_e2e_mean > TPOT_BY_E2E_EARLY_STOP_MS:
                    existing = blocked_group_rates.get(group_key)
                    reason = (
                        "historical latest successful result for "
                        f"rate={stringify_request_rate(case.request_rate)} at "
                        f"{case_result_tag} has "
                        f"tpot_by_e2e.mean={tpot_by_e2e_mean:.3f}ms "
                        f"> {TPOT_BY_E2E_EARLY_STOP_MS:g}ms")
                    reason += historical_group_dir_match_suffix(
                        case,
                        matched_group_dir=matched_group_dir,
                        exact_group_dir=exact_group_dir,
                    )
                    if existing is None or case.request_rate < existing[0]:
                        blocked_group_rates[group_key] = (case.request_rate,
                                                          reason)

        latest_status = latest_case_status_for_case(
            artifact_root,
            case,
            ignore_bs=ignore_bs,
        )
        if latest_status is None:
            continue

        latest_case_dir_path, status = latest_status
        if status not in {"timed_out", "failed"}:
            continue

        matched_group_dir = latest_case_dir_path.parent
        exact_group_dir = case_artifact_group_dir(artifact_root, case)
        status_reason = (
            "historical latest case for "
            f"rate={stringify_request_rate(case.request_rate)} at "
            f"{latest_case_dir_path.name} ended with status={status}")
        status_reason += historical_group_dir_match_suffix(
            case,
            matched_group_dir=matched_group_dir,
            exact_group_dir=exact_group_dir,
        )
        existing = blocked_group_rates.get(group_key)
        if existing is None or case.request_rate < existing[0]:
            blocked_group_rates[group_key] = (case.request_rate, status_reason)

    return exact_case_skips, blocked_group_rates


def block_reason_for_rate(
    blocked_group_rates: Mapping[str, tuple[float, str]],
    case: ExperimentCase,
) -> str | None:
    blocked = blocked_group_rates.get(case_group_key(case))
    if blocked is None:
        return None

    blocked_rate, reason = blocked
    if case.request_rate >= blocked_rate:
        return reason
    return None


def write_run_manifest(
    path: Path,
    *,
    run_name: str,
    created_at: str,
    dry_run: bool,
    keep_going: bool,
    ignore_historical_skips: bool,
    historical_skip_ignore_bs: bool,
    rate_plan: str,
    cases: list[str],
    finished_at: str | None = None,
    historical_exact_case_skips: Mapping[str, str] | None = None,
    historical_blocked_groups: Mapping[str, tuple[float, str]] | None = None,
    runtime_blocked_groups: Mapping[str, tuple[float, str]] | None = None,
    results: list[CaseResult] | None = None,
    aborted: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "run_name": run_name,
        "created_at": created_at,
        "dry_run": dry_run,
        "keep_going": keep_going,
        "ignore_historical_skips": ignore_historical_skips,
        "historical_skip_ignore_bs": historical_skip_ignore_bs,
        "rate_plan": rate_plan,
        "cases": cases,
    }
    if finished_at is not None:
        payload["finished_at"] = finished_at
    if historical_exact_case_skips is not None:
        payload["historical_exact_case_skips"] = historical_exact_case_skips
    if historical_blocked_groups is not None:
        payload["historical_blocked_groups"] = {
            key: {
                "blocked_from_rate": rate,
                "reason": reason,
            }
            for key, (rate, reason) in historical_blocked_groups.items()
        }
    if runtime_blocked_groups is not None:
        payload["runtime_blocked_groups"] = {
            key: {
                "blocked_from_rate": rate,
                "reason": reason,
            }
            for key, (rate, reason) in runtime_blocked_groups.items()
        }
    if results is not None:
        payload["results"] = [jsonify(result) for result in results]
    if aborted is not None:
        payload["aborted"] = dict(aborted)
    write_json(path, payload)


def describe_abort(exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": type(exc).__name__,
    }
    detail = str(exc)
    if detail:
        payload["detail"] = detail
    if isinstance(exc, SystemExit):
        payload["exit_code"] = exc.code
    return payload


def local_ports_for_case(resolved: ResolvedCase) -> tuple[int, ...]:
    return tuple(
        sorted({
            resolved.cluster.master_port,
            resolved.case.data_parallel_rpc_port,
        }))


TCP_STATE_NAMES = {
    "01": "ESTABLISHED",
    "02": "SYN_SENT",
    "03": "SYN_RECV",
    "04": "FIN_WAIT1",
    "05": "FIN_WAIT2",
    "06": "TIME_WAIT",
    "07": "CLOSE",
    "08": "CLOSE_WAIT",
    "09": "LAST_ACK",
    "0A": "LISTEN",
    "0B": "CLOSING",
}


def _iter_proc_net_tcp_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for table_path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        if not table_path.is_file():
            continue
        try:
            lines = table_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 10:
                continue
            rows.append((fields[1], fields[2], fields[3], fields[9]))
    return rows


def _local_address_matches_port(local_address: str, port: int) -> bool:
    try:
        _host_hex, port_hex = local_address.rsplit(":", 1)
    except ValueError:
        return False
    return port_hex.upper() == f"{port:04X}"


def local_tcp_state_counts_for_port(port: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for local_address, _remote_address, state_hex, _inode in _iter_proc_net_tcp_rows():
        if not _local_address_matches_port(local_address, port):
            continue
        state_name = TCP_STATE_NAMES.get(state_hex, state_hex)
        counts[state_name] = counts.get(state_name, 0) + 1
    return counts


def local_tcp_listener_details_for_port(port: int) -> list[str]:
    listener_inodes = {
        inode
        for local_address, _remote_address, state_hex, inode in _iter_proc_net_tcp_rows()
        if state_hex == "0A" and _local_address_matches_port(local_address, port)
    }
    if not listener_inodes:
        return []

    details: list[str] = []
    seen_pids: set[int] = set()
    target_links = {f"socket:[{inode}]" for inode in listener_inodes}
    for proc_path in Path("/proc").iterdir():
        if not proc_path.name.isdigit():
            continue
        pid = int(proc_path.name)
        if pid in seen_pids:
            continue

        fd_path = proc_path / "fd"
        try:
            fd_entries = list(fd_path.iterdir())
        except OSError:
            continue

        matched = False
        for fd_entry in fd_entries:
            try:
                link_target = os.readlink(fd_entry)
            except OSError:
                continue
            if link_target in target_links:
                matched = True
                break
        if not matched:
            continue

        seen_pids.add(pid)
        cmdline_path = proc_path / "cmdline"
        cmdline = ""
        try:
            cmdline = cmdline_path.read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace").strip()
        except OSError:
            cmdline = ""
        if not cmdline:
            try:
                cmdline = f"[{(proc_path / 'comm').read_text(encoding='utf-8').strip()}]"
            except OSError:
                cmdline = "[unknown]"
        details.append(f"pid {pid} ({cmdline})")
    return sorted(details)


def describe_local_port_diagnostics(port: int) -> str:
    listeners = local_tcp_listener_details_for_port(port)
    states = local_tcp_state_counts_for_port(port)
    parts = []
    if listeners:
        parts.append("listeners: " + "; ".join(listeners))
    else:
        parts.append("listeners: none")
    if states:
        parts.append("states: " + ", ".join(
            f"{state}={count}" for state, count in sorted(states.items())))
    else:
        parts.append("states: none")
    return f"{port} [{'; '.join(parts)}]"


def can_bind_local_tcp_port(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Previous cases can leave sockets in TIME_WAIT after a clean
        # shutdown. Treat the port as reusable if we can bind with
        # SO_REUSEADDR, which matches the next listener startup path more
        # closely than a bare bind probe.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", port))
        except OSError:
            return False
    return True


def read_tracked_process_id(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def local_pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def local_process_group_is_running(pgid: int) -> bool:
    if pgid <= 0:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_local_ports_to_clear(
    ports: tuple[int, ...],
    *,
    timeout_sec: float,
    poll_interval_sec: float,
) -> tuple[int, ...]:
    if not ports:
        return ()

    deadline = time.monotonic() + timeout_sec
    while True:
        busy_ports = tuple(
            port for port in ports if not can_bind_local_tcp_port(port))
        if not busy_ports:
            return ()
        if time.monotonic() >= deadline:
            return busy_ports
        time.sleep(poll_interval_sec)


class ManualMultinodeRunner:

    def __init__(self, artifact_root: Path, *, dry_run: bool,
                 keep_going: bool = True,
                 ignore_historical_skips: bool = False,
                 historical_skip_ignore_bs: bool = True,
                 auto_rerun_trend_outliers: bool = False,
                 trend_rerun_max_attempts: int = DEFAULT_TREND_RERUN_MAX_ATTEMPTS,
                 trend_rerun_relative_threshold_pct: float = DEFAULT_TREND_RERUN_RELATIVE_THRESHOLD_PCT,
                 trend_rerun_absolute_threshold_ms: float = DEFAULT_TREND_RERUN_ABSOLUTE_THRESHOLD_MS) -> None:
        self.artifact_root = artifact_root.expanduser().resolve()
        self.dry_run = dry_run
        self.keep_going = keep_going
        self.ignore_historical_skips = ignore_historical_skips
        self.historical_skip_ignore_bs = historical_skip_ignore_bs
        self.auto_rerun_trend_outliers = auto_rerun_trend_outliers
        self.trend_rerun_max_attempts = trend_rerun_max_attempts
        self.trend_rerun_relative_threshold_pct = (
            trend_rerun_relative_threshold_pct)
        self.trend_rerun_absolute_threshold_ms = (
            trend_rerun_absolute_threshold_ms)
        self._active_runtime: ActiveCaseRuntime | None = None
        self._previous_handlers: dict[int, Any] = {}
        self._handling_signal = False

    def log(self, message: str, *, stream: TextIO = sys.stdout) -> None:
        print(f"[runner] {current_iso_timestamp()} {message}",
              file=stream,
              flush=True)

    def log_case(self,
                 case_name: str,
                 message: str,
                 *,
                 stream: TextIO = sys.stdout) -> None:
        self.log(f"{case_name}: {message}", stream=stream)

    def runtime_case_name(self, runtime: ActiveCaseRuntime) -> str:
        return getattr(runtime.resolved.case, "name", "<unnamed-case>")

    def log_waiting(self,
                    case_name: str,
                    message: str,
                    *,
                    next_log_at: float,
                    now: float | None = None,
                    deadline: float | None = None,
                    stream: TextIO = sys.stdout) -> float:
        if now is None:
            now = time.monotonic()
        if now < next_log_at:
            return next_log_at

        if deadline is None:
            rendered_message = message
        else:
            remaining = max(0.0, deadline - now)
            rendered_message = f"{message} (remaining ~{remaining:.0f}s)"
        self.log_case(case_name, rendered_message, stream=stream)
        return now + WAIT_STATUS_LOG_INTERVAL_SEC

    def install_signal_handlers(self) -> None:
        handled_signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            handled_signals.append(signal.SIGHUP)
        for sig in handled_signals:
            self._previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)

    def restore_signal_handlers(self) -> None:
        for sig, handler in self._previous_handlers.items():
            signal.signal(sig, handler)
        self._previous_handlers.clear()

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        if self._handling_signal:
            raise SystemExit(128 + signum)
        self._handling_signal = True
        exit_code = 128 + signum
        signal_name = signal.Signals(signum).name
        try:
            if self._active_runtime is not None:
                case_name = self.runtime_case_name(self._active_runtime)
                self.log_case(case_name,
                              (f"received {signal_name}; cleaning up active "
                               "case and repairing manifest if possible"),
                              stream=sys.stderr)
                with contextlib.suppress(Exception):
                    self.cleanup_case_runtime(self._active_runtime)
                with contextlib.suppress(Exception):
                    self.finalize_active_case_manifest_after_signal(
                        self._active_runtime,
                        signum=signum,
                    )
        finally:
            raise SystemExit(exit_code)

    def finalize_active_case_manifest_after_signal(
            self, runtime: ActiveCaseRuntime, *, signum: int) -> None:
        signal_name = signal.Signals(signum).name
        manifest = load_optional_json(runtime.artifacts.case_manifest_path)
        maybe_finalize_interrupted_case_manifest(
            runtime.artifacts.case_dir,
            manifest=manifest,
            signal_name=signal_name,
            exit_code=128 + signum,
            finished_at=current_iso_timestamp(),
        )
        repaired_manifest = load_optional_json(runtime.artifacts.case_manifest_path)
        if repaired_manifest is not None:
            self.log_case(
                self.runtime_case_name(runtime),
                ("persisted manifest after signal with "
                 f"status={repaired_manifest.get('status')}"),
                stream=sys.stderr,
            )

    def run(self, selected_cases: list[ExperimentCase], run_label: str | None,
            *, rate_plan: str) -> list[CaseResult]:
        run_name = f"{current_run_tag()}__{sanitize_tag(run_label or 'manual_poisson')}"
        run_dir = self.artifact_root / "_runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        run_manifest_path = run_dir / "run_manifest.json"
        created_at = current_iso_timestamp()
        case_names = [case.name for case in selected_cases]
        write_run_manifest(
            run_manifest_path,
            run_name=run_name,
            created_at=created_at,
            dry_run=self.dry_run,
            keep_going=self.keep_going,
            ignore_historical_skips=self.ignore_historical_skips,
            historical_skip_ignore_bs=self.historical_skip_ignore_bs,
            rate_plan=rate_plan,
            cases=case_names,
        )

        results: list[CaseResult] = []
        latest_results_by_case_name: dict[str, CaseResult] = {}
        selected_cases_by_group: dict[str, list[ExperimentCase]] = {}
        for case in selected_cases:
            selected_cases_by_group.setdefault(case_group_key(case), []).append(case)
        exact_case_skips: dict[str, str] = {}
        historical_blocked_groups: dict[str, tuple[float, str]] = {}
        if not self.ignore_historical_skips:
            exact_case_skips, historical_blocked_groups = build_historical_skip_state(
                self.artifact_root,
                selected_cases,
                ignore_bs=self.historical_skip_ignore_bs)
        runtime_blocked_groups: dict[str, tuple[float, str]] = {}
        aborted: dict[str, Any] | None = None
        self.install_signal_handlers()
        try:
            for case in selected_cases:
                historical_group_reason = block_reason_for_rate(
                    historical_blocked_groups, case)
                if historical_group_reason is not None:
                    print(f"[skip] {case.name}: {historical_group_reason}",
                          flush=True)
                    continue

                runtime_group_reason = block_reason_for_rate(
                    runtime_blocked_groups, case)
                if runtime_group_reason is not None:
                    print(f"[skip] {case.name}: {runtime_group_reason}",
                          flush=True)
                    continue

                exact_skip_reason = exact_case_skips.get(case.name)
                if exact_skip_reason is not None:
                    print(f"[skip] {case.name}: {exact_skip_reason}", flush=True)
                    continue

                result = self.run_case(case, run_dir)
                results.append(result)
                latest_results_by_case_name[case.name] = result
                should_stop, stop_reason = should_stop_followup_rates(result)
                if should_stop:
                    reason = stop_reason or "blocked by previous case result"
                    group_key = case_group_key(case)
                    runtime_blocked_groups[group_key] = (case.request_rate,
                                                         reason)
                    print(f"[sweep] stop higher rates for {group_key}: {reason}",
                          flush=True)
                if (result.status not in {"ok", "dry_run", "timed_out"}
                        and not self.keep_going):
                    break

            if self.auto_rerun_trend_outliers:
                self.run_trend_outlier_reruns(
                    selected_cases_by_group,
                    latest_results_by_case_name,
                    results,
                    run_dir,
                )
        except BaseException as exc:
            aborted = describe_abort(exc)
            raise
        finally:
            self.restore_signal_handlers()
            write_run_manifest(
                run_manifest_path,
                run_name=run_name,
                created_at=created_at,
                dry_run=self.dry_run,
                keep_going=self.keep_going,
                ignore_historical_skips=self.ignore_historical_skips,
                historical_skip_ignore_bs=self.historical_skip_ignore_bs,
                rate_plan=rate_plan,
                cases=case_names,
                finished_at=current_iso_timestamp(),
                historical_exact_case_skips=exact_case_skips,
                historical_blocked_groups=historical_blocked_groups,
                runtime_blocked_groups=runtime_blocked_groups,
                results=results,
                aborted=aborted,
            )
        return results

    def run_case(self, case: ExperimentCase, _run_dir: Path) -> CaseResult:
        resolved = resolve_case(case)
        artifacts = prepare_artifact_paths(
            case_artifact_group_dir(self.artifact_root, case),
            resolved.cluster.remote_hosts,
            run_tag=current_run_tag(),
        )
        artifacts.case_dir.mkdir(parents=True, exist_ok=True)
        artifacts.benchmark_dir.mkdir(parents=True, exist_ok=True)

        frontend_argv = build_frontend_argv(resolved, artifacts.benchmark_dir)
        frontend_launch_command = build_local_launch_command(
            resolved,
            artifacts,
            frontend_argv,
        )
        remote_launch_commands = {
            node_rank: build_remote_launch_command(
                resolved,
                artifacts,
                node_rank=node_rank,
            )
            for node_rank in range(1, resolved.cluster.nnodes)
        }
        local_cleanup_command = build_node_cleanup_command(
            resolved=resolved,
            pid_path=artifacts.frontend_pid_path,
            pgid_path=artifacts.frontend_pgid_path,
            include_frontend_pattern=True,
        )
        remote_cleanup_commands = {
            node_rank: build_node_cleanup_command(
                resolved=resolved,
                pid_path=artifacts.rank_pid_paths[node_rank],
                pgid_path=artifacts.rank_pgid_paths[node_rank],
                include_frontend_pattern=False,
            )
            for node_rank in range(1, resolved.cluster.nnodes)
        }

        write_case_command_files(
            resolved,
            artifacts,
            frontend_launch_command=frontend_launch_command,
            remote_launch_commands=remote_launch_commands,
            local_cleanup_command=local_cleanup_command,
            remote_cleanup_commands=remote_cleanup_commands,
        )
        write_case_manifest(
            resolved,
            artifacts,
            frontend_launch_command=frontend_launch_command,
            remote_launch_commands=remote_launch_commands,
            local_cleanup_command=local_cleanup_command,
            remote_cleanup_commands=remote_cleanup_commands,
            status="prepared",
            started_at=None,
            finished_at=None,
            exit_code=None,
            detail=None,
        )

        started_at = current_iso_timestamp()
        write_case_manifest(
            resolved,
            artifacts,
            frontend_launch_command=frontend_launch_command,
            remote_launch_commands=remote_launch_commands,
            local_cleanup_command=local_cleanup_command,
            remote_cleanup_commands=remote_cleanup_commands,
            status="running",
            started_at=started_at,
            finished_at=None,
            exit_code=None,
            detail="launch sequence started",
        )
        print(f"[case] starting {case.name}", flush=True)
        self.log_case(case.name, "precleaning existing processes and ports")
        if self.dry_run:
            finished_at = current_iso_timestamp()
            write_case_manifest(
                resolved,
                artifacts,
                frontend_launch_command=frontend_launch_command,
                remote_launch_commands=remote_launch_commands,
                local_cleanup_command=local_cleanup_command,
                remote_cleanup_commands=remote_cleanup_commands,
                status="dry_run",
                started_at=started_at,
                finished_at=finished_at,
                exit_code=0,
                detail="dry run: commands were written, nothing was executed",
            )
            self.log_case(case.name, "dry run complete; commands were written only")
            print(f"[case] {case.name}: dry_run", flush=True)
            return CaseResult(
                case_name=case.name,
                status="dry_run",
                exit_code=0,
                started_at=started_at,
                finished_at=finished_at,
                case_dir=artifacts.case_dir,
                benchmark_dir=artifacts.benchmark_dir,
                summary_json=artifacts.benchmark_dir / "summary.json",
                detail="dry run",
            )

        runtime = ActiveCaseRuntime(resolved=resolved, artifacts=artifacts)
        self._active_runtime = runtime

        exit_code: int | None = None
        detail = ""
        status = "ok"
        cleanup_error: Exception | None = None
        launched_runtime = False
        try:
            self.run_preclean(resolved, artifacts)
            self.log_case(case.name, "launching remote headless nodes")
            self.launch_headless_nodes(runtime, remote_launch_commands)
            launched_runtime = bool(runtime.headless_nodes)
            self.log_case(case.name, "waiting for remote headless startup")
            self.wait_for_headless_startup(runtime)
            self.log_case(case.name, "launching frontend benchmark process")
            self.launch_frontend(runtime, frontend_launch_command)
            launched_runtime = True
            self.log_case(case.name, "frontend launched; waiting for benchmark completion")
            exit_code = self.wait_for_frontend(runtime)
            self.log_case(case.name,
                          "frontend exited cleanly; waiting for remote headless shutdown")
            self.wait_for_headless_shutdown(runtime)
            self.log_case(case.name,
                          "remote headless nodes exited; waiting for local shutdown")
            self.wait_for_local_shutdown(runtime)
            self.log_case(case.name, "local shutdown finished; augmenting benchmark outputs")
            augment_benchmark_outputs(artifacts.benchmark_dir)
            self.log_case(case.name, "benchmark outputs augmented; validating summary")
            validate_benchmark_summary(artifacts.benchmark_dir)
            detail = "completed"
        except BenchTimeoutError as exc:
            status = "timed_out"
            detail = str(exc)
            exit_code = 124
        except Exception as exc:
            status = "failed"
            detail = str(exc)
            exit_code = 1 if exit_code in (None, 0) else exit_code
        finally:
            try:
                self.log_case(case.name, "running cleanup commands")
                self.cleanup_case_runtime(runtime)
            except Exception as exc:
                cleanup_error = exc
            self._active_runtime = None

        if cleanup_error is not None:
            self.log_case(case.name, f"cleanup reported an error: {cleanup_error}",
                          stream=sys.stderr)
            cleanup_detail = f"cleanup failed: {cleanup_error}"
            if status == "ok":
                status = "failed"
                detail = cleanup_detail
                exit_code = 1 if exit_code in (None, 0) else exit_code
            elif detail:
                detail = f"{detail}\n{cleanup_detail}"
            else:
                detail = cleanup_detail

        if launched_runtime:
            try:
                self.log_case(case.name, "verifying post-cleanup process and port state")
                self.verify_case_cleanup(runtime)
            except Exception as exc:
                self.log_case(case.name,
                              f"post-cleanup verification failed: {exc}",
                              stream=sys.stderr)
                cleanup_detail = str(exc)
                if status == "ok":
                    status = "failed"
                    detail = cleanup_detail
                    exit_code = 1 if exit_code in (None, 0) else exit_code
                elif detail:
                    detail = f"{detail}\n{cleanup_detail}"
                else:
                    detail = cleanup_detail

        finished_at = current_iso_timestamp()
        write_case_manifest(
            resolved,
            artifacts,
            frontend_launch_command=frontend_launch_command,
            remote_launch_commands=remote_launch_commands,
            local_cleanup_command=local_cleanup_command,
            remote_cleanup_commands=remote_cleanup_commands,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            exit_code=exit_code,
            detail=detail,
        )

        self.log_case(case.name, f"final status={status} exit_code={exit_code}")
        print(f"[case] {case.name}: {status}", flush=True)
        return CaseResult(
            case_name=case.name,
            status=status,
            exit_code=exit_code,
            started_at=started_at,
            finished_at=finished_at,
            case_dir=artifacts.case_dir,
            benchmark_dir=artifacts.benchmark_dir,
            summary_json=artifacts.benchmark_dir / "summary.json",
            detail=detail,
        )

    def run_trend_outlier_reruns(
        self,
        selected_cases_by_group: Mapping[str, list[ExperimentCase]],
        latest_results_by_case_name: dict[str, CaseResult],
        results: list[CaseResult],
        run_dir: Path,
    ) -> None:
        attempts_by_case_name: dict[str, int] = {}

        while True:
            pending_candidates: list[TrendOutlierCandidate] = []
            for group_cases in selected_cases_by_group.values():
                for candidate in detect_group_trend_outlier_candidates(
                        group_cases,
                        latest_results_by_case_name,
                        relative_threshold_pct=self.
                        trend_rerun_relative_threshold_pct,
                        absolute_threshold_ms=self.
                        trend_rerun_absolute_threshold_ms):
                    attempts = attempts_by_case_name.get(candidate.case.name, 0)
                    if attempts >= self.trend_rerun_max_attempts:
                        continue
                    pending_candidates.append(candidate)

            if not pending_candidates:
                return

            candidate = pending_candidates[0]
            attempts = attempts_by_case_name.get(candidate.case.name, 0) + 1
            attempts_by_case_name[candidate.case.name] = attempts
            self.log_case(
                candidate.case.name,
                ("trend-outlier rerun "
                 f"{attempts}/{self.trend_rerun_max_attempts}: "
                 f"{TREND_RERUN_METRIC_KEY}.{TREND_RERUN_SUBMETRIC}="
                 f"{candidate.actual_value:.3f}ms, "
                 f"expected≈{candidate.expected_value:.3f}ms, "
                 f"allowed_delta={candidate.allowed_delta_ms:.3f}ms, "
                 f"neighbors=({candidate.lower_case.request_rate:g}, "
                 f"{candidate.upper_case.request_rate:g})"),
            )
            rerun_result = self.run_case(candidate.case, run_dir)
            results.append(rerun_result)
            latest_results_by_case_name[candidate.case.name] = rerun_result
            if rerun_result.status not in {"ok", "dry_run", "timed_out"
                                           } and not self.keep_going:
                return

    def run_preclean(self, resolved: ResolvedCase,
                     artifacts: ArtifactPaths) -> None:
        ports = local_ports_for_case(resolved)
        busy_ports: tuple[int, ...] = ()

        for attempt in range(1, PRESTART_CLEANUP_MAX_ATTEMPTS + 1):
            self.run_local_shell(
                resolved.cluster,
                artifacts.frontend_cleanup_command_path.read_text(
                    encoding="utf-8"),
            )
            for node_rank, host in enumerate(resolved.cluster.remote_hosts,
                                             start=1):
                self.run_remote_shell(
                    resolved.cluster,
                    host,
                    artifacts.rank_cleanup_command_paths[node_rank].read_text(
                        encoding="utf-8"),
                )

            busy_ports = wait_for_local_ports_to_clear(
                ports,
                timeout_sec=PRESTART_CLEANUP_WAIT_SEC,
                poll_interval_sec=PRESTART_CLEANUP_POLL_INTERVAL_SEC,
            )
            if not busy_ports:
                return

            ports_text = ", ".join(str(port) for port in busy_ports)
            diagnostics = "; ".join(
                describe_local_port_diagnostics(port) for port in busy_ports)
            print(
                ("[preclean] local ports still busy after cleanup "
                 f"attempt {attempt}/{PRESTART_CLEANUP_MAX_ATTEMPTS}: "
                 f"{ports_text}"),
                file=sys.stderr,
                flush=True,
            )
            print(f"[preclean] diagnostics: {diagnostics}",
                  file=sys.stderr,
                  flush=True)

        ports_text = ", ".join(str(port) for port in busy_ports)
        diagnostics = "; ".join(
            describe_local_port_diagnostics(port) for port in busy_ports)
        raise RuntimeError("preclean could not free local ports before launch: "
                           f"{ports_text}. diagnostics: {diagnostics}")

    def launch_headless_nodes(self, runtime: ActiveCaseRuntime,
                              remote_launch_commands: dict[int, str]) -> None:
        for node_rank, host in enumerate(runtime.resolved.cluster.remote_hosts,
                                         start=1):
            launch_log_path = runtime.artifacts.rank_launch_log_paths[node_rank]
            launch_log_path.parent.mkdir(parents=True, exist_ok=True)
            launch_log_handle = launch_log_path.open("w", encoding="utf-8")
            ssh_command = build_remote_ssh_command(
                runtime.resolved.cluster,
                host,
                remote_launch_commands[node_rank],
            )
            process = subprocess.Popen(
                ssh_command,
                stdin=subprocess.DEVNULL,
                stdout=launch_log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            runtime.headless_nodes.append(
                NodeRuntime(
                    node_rank=node_rank,
                    host=host,
                    process=process,
                    launch_log_path=launch_log_path,
                    launch_log_handle=launch_log_handle,
                    runtime_log_path=runtime.artifacts.rank_log_paths[node_rank],
                ))

    def wait_for_headless_startup(self, runtime: ActiveCaseRuntime) -> None:
        deadline = time.monotonic() + runtime.resolved.case.start_grace_sec
        while time.monotonic() < deadline:
            for node in runtime.headless_nodes:
                if node.process.poll() is not None:
                    detail = tail_file(node.launch_log_path) or tail_file(
                        node.runtime_log_path)
                    raise RuntimeError(
                        f"rank {node.node_rank} on {node.host} exited during startup.\n{detail}"
                    )
            check_runtime_logs_for_fatal(runtime,
                                         phase="headless startup",
                                         include_frontend=False)
            time.sleep(0.5)

    def launch_frontend(self, runtime: ActiveCaseRuntime,
                        frontend_launch_command: str) -> None:
        launch_log_handle = runtime.artifacts.frontend_launch_log_path.open(
            "w", encoding="utf-8")
        command_argv = shell_command_argv(
            runtime.resolved.cluster.local_shell,
            runtime.resolved.cluster.local_shell_flags,
            frontend_launch_command,
        )
        process = subprocess.Popen(
            command_argv,
            cwd=runtime.resolved.cluster.workdir,
            stdin=subprocess.DEVNULL,
            stdout=launch_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        runtime.frontend = NodeRuntime(
            node_rank=0,
            host="local",
            process=process,
            launch_log_path=runtime.artifacts.frontend_launch_log_path,
            launch_log_handle=launch_log_handle,
            runtime_log_path=runtime.artifacts.frontend_log_path,
        )

    def wait_for_frontend(self, runtime: ActiveCaseRuntime) -> int:
        assert runtime.frontend is not None
        timeout_sec = runtime.resolved.case.max_bench_duration_sec
        deadline = None
        wait_started_at = time.monotonic()
        if timeout_sec is not None:
            deadline = wait_started_at + timeout_sec
        pending_headless_nodes = list(runtime.headless_nodes)
        case_name = self.runtime_case_name(runtime)
        next_log_at = wait_started_at + WAIT_STATUS_LOG_INTERVAL_SEC
        kv_cache_capacity_logged = False
        while True:
            now = time.monotonic()
            exit_code = runtime.frontend.process.poll()
            if exit_code is not None:
                if exit_code != 0:
                    detail = tail_file(
                        runtime.frontend.runtime_log_path) or tail_file(
                            runtime.frontend.launch_log_path)
                    raise RuntimeError(
                        f"frontend exited with code {exit_code}.\n{detail}")
                check_runtime_logs_for_fatal(runtime,
                                             phase="benchmark wait",
                                             include_frontend=True)
                self.log_case(case_name, "frontend process exited with code 0")
                return exit_code

            if not kv_cache_capacity_logged:
                capacity_fields = extract_gpu_kv_cache_capacity_fields(
                    runtime.frontend.runtime_log_path)
                if capacity_fields is not None:
                    self.log_case(
                        case_name,
                        format_gpu_kv_cache_capacity_message(capacity_fields))
                    kv_cache_capacity_logged = True

            if deadline is not None and now >= deadline:
                detail = tail_file(runtime.frontend.runtime_log_path) or tail_file(
                    runtime.frontend.launch_log_path)
                message = ("frontend exceeded "
                           f"max_bench_duration_sec={timeout_sec:g}s "
                           f"({timeout_sec / 60.0:g}min)")
                if detail:
                    message = f"{message}.\n{detail}"
                raise BenchTimeoutError(message)

            still_running_headless_nodes = []
            for node in pending_headless_nodes:
                node_exit_code = node.process.poll()
                if node_exit_code is None:
                    still_running_headless_nodes.append(node)
                    continue
                if node_exit_code != 0:
                    detail = tail_file(node.launch_log_path) or tail_file(
                        node.runtime_log_path)
                    raise RuntimeError(
                        f"rank {node.node_rank} on {node.host} exited early "
                        f"with code {node_exit_code}.\n{detail}")
            pending_headless_nodes = still_running_headless_nodes
            check_runtime_logs_for_fatal(runtime,
                                         phase="benchmark wait",
                                         include_frontend=True)
            pending_ranks = ", ".join(
                str(node.node_rank) for node in pending_headless_nodes) or "none"
            next_log_at = self.log_waiting(
                case_name,
                ("still waiting for frontend benchmark to finish; "
                 f"remote wrappers still running on ranks: {pending_ranks}"),
                next_log_at=next_log_at,
                now=now,
                deadline=deadline,
            )
            time.sleep(1.0)

    def wait_for_headless_shutdown(self, runtime: ActiveCaseRuntime) -> None:
        wait_started_at = time.monotonic()
        deadline = (wait_started_at +
                    runtime.resolved.case.remote_shutdown_grace_sec)
        pending = list(runtime.headless_nodes)
        case_name = self.runtime_case_name(runtime)
        next_log_at = wait_started_at + WAIT_STATUS_LOG_INTERVAL_SEC
        while pending:
            now = time.monotonic()
            if now >= deadline:
                break
            still_pending = []
            pending_statuses: list[str] = []
            for node in pending:
                wrapper_running = node.process.poll() is None
                remote_pid_running = self.remote_pidfile_is_running(
                    runtime.resolved.cluster,
                    node.host,
                    runtime.artifacts.rank_pid_paths[node.node_rank],
                )
                if wrapper_running or remote_pid_running:
                    still_pending.append(node)
                    state_bits = []
                    if wrapper_running:
                        state_bits.append("wrapper")
                    if remote_pid_running:
                        state_bits.append("pidfile")
                    pending_statuses.append(
                        f"rank {node.node_rank}@{node.host} ({'/'.join(state_bits)})")
            pending = still_pending
            if pending:
                next_log_at = self.log_waiting(
                    case_name,
                    "waiting for remote headless shutdown: " +
                    "; ".join(pending_statuses),
                    next_log_at=next_log_at,
                    now=now,
                    deadline=deadline,
                )
                time.sleep(1.0)

        if pending:
            details = []
            for node in pending:
                pid_running = self.remote_pidfile_is_running(
                    runtime.resolved.cluster,
                    node.host,
                    runtime.artifacts.rank_pid_paths[node.node_rank],
                )
                pid_path = runtime.artifacts.rank_pid_paths[node.node_rank]
                detail = f"rank {node.node_rank} on {node.host} did not exit in time."
                if pid_running:
                    detail += f" remote pidfile still points to a live pid at {pid_path}."
                details.append(detail)
            raise RuntimeError("\n".join(details))

    def wait_for_local_shutdown(self, runtime: ActiveCaseRuntime) -> None:
        wait_started_at = time.monotonic()
        deadline = wait_started_at + self.local_shutdown_timeout_sec(runtime)
        details = self.local_shutdown_failures(runtime)
        case_name = self.runtime_case_name(runtime)
        next_log_at = wait_started_at + WAIT_STATUS_LOG_INTERVAL_SEC
        while details:
            now = time.monotonic()
            if now >= deadline:
                break
            next_log_at = self.log_waiting(
                case_name,
                "waiting for local shutdown: " + " | ".join(details),
                next_log_at=next_log_at,
                now=now,
                deadline=deadline,
            )
            time.sleep(LOCAL_SHUTDOWN_POLL_INTERVAL_SEC)
            details = self.local_shutdown_failures(runtime)

        if details:
            raise RuntimeError("node 0 local shutdown did not complete in time.\n" +
                               "\n".join(details))

    def verify_case_cleanup(self, runtime: ActiveCaseRuntime) -> None:
        wait_started_at = time.monotonic()
        deadline = wait_started_at + self.local_shutdown_timeout_sec(runtime)
        details = self.case_cleanup_failures(runtime)
        case_name = self.runtime_case_name(runtime)
        next_log_at = wait_started_at + WAIT_STATUS_LOG_INTERVAL_SEC
        while details:
            now = time.monotonic()
            if now >= deadline:
                break
            next_log_at = self.log_waiting(
                case_name,
                "waiting for post-cleanup verification: " + " | ".join(details),
                next_log_at=next_log_at,
                now=now,
                deadline=deadline,
            )
            time.sleep(LOCAL_SHUTDOWN_POLL_INTERVAL_SEC)
            details = self.case_cleanup_failures(runtime)

        if details:
            raise RuntimeError("post-cleanup verification failed.\n" +
                               "\n".join(details))

    def local_shutdown_timeout_sec(self, runtime: ActiveCaseRuntime) -> float:
        timeout = getattr(runtime.resolved.case, "local_shutdown_grace_sec", None)
        if timeout is not None:
            return timeout
        return getattr(runtime.resolved.case, "remote_shutdown_grace_sec", 30.0)

    def local_shutdown_failures(self, runtime: ActiveCaseRuntime) -> list[str]:
        details: list[str] = []

        pid = read_tracked_process_id(runtime.artifacts.frontend_pid_path)
        if pid is not None and local_pid_is_running(pid):
            details.append("frontend pidfile still points to a live pid at "
                           f"{runtime.artifacts.frontend_pid_path} (pid {pid}).")

        pgid = read_tracked_process_id(runtime.artifacts.frontend_pgid_path)
        if pgid is not None and local_process_group_is_running(pgid):
            details.append("frontend process group still has live processes at "
                           f"{runtime.artifacts.frontend_pgid_path} (pgid {pgid}).")

        busy_ports = tuple(
            port for port in local_ports_for_case(runtime.resolved)
            if not can_bind_local_tcp_port(port))
        if busy_ports:
            diagnostics = "; ".join(
                describe_local_port_diagnostics(port) for port in busy_ports)
            details.append("local ports still busy: " + diagnostics)

        return details

    def case_cleanup_failures(self, runtime: ActiveCaseRuntime) -> list[str]:
        details = list(self.local_shutdown_failures(runtime))
        for node in runtime.headless_nodes:
            pid_path = runtime.artifacts.rank_pid_paths[node.node_rank]
            if self.remote_pidfile_is_running(runtime.resolved.cluster, node.host,
                                              pid_path):
                details.append("rank "
                               f"{node.node_rank} on {node.host} pidfile still "
                               f"points to a live pid at {pid_path}.")
        return details

    def cleanup_case_runtime(self, runtime: ActiveCaseRuntime) -> None:
        case_name = self.runtime_case_name(runtime)
        if runtime.frontend is not None:
            self.log_case(case_name, "terminating frontend wrapper process")
            self.terminate_process(runtime.frontend.process, "frontend")
            runtime.frontend.launch_log_handle.close()
        for node in runtime.headless_nodes:
            self.log_case(case_name,
                          f"terminating rank {node.node_rank} wrapper process")
            self.terminate_process(node.process, f"rank {node.node_rank}")
            node.launch_log_handle.close()

        self.log_case(case_name, "running local cleanup command on node 0")
        self.run_local_shell(
            runtime.resolved.cluster,
            runtime.artifacts.frontend_cleanup_command_path.read_text(
                encoding="utf-8"),
        )
        for node_rank, host in enumerate(runtime.resolved.cluster.remote_hosts,
                                         start=1):
            self.log_case(case_name,
                          f"running remote cleanup command on rank {node_rank} host {host}")
            self.run_remote_shell(
                runtime.resolved.cluster,
                host,
                runtime.artifacts.rank_cleanup_command_paths[node_rank].read_text(
                    encoding="utf-8"),
            )

    def terminate_process(self, proc: subprocess.Popen[str], label: str) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            with contextlib.suppress(Exception):
                os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
        finally:
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.kill()
                with contextlib.suppress(Exception):
                    proc.wait(timeout=5)
            print(f"[cleanup] terminated {label}",
                  file=sys.stderr,
                  flush=True)

    def remote_pidfile_is_running(self, cluster: ClusterSpec, host: str,
                                  pid_path: Path) -> bool:
        ssh_command = build_remote_ssh_command(
            cluster,
            host,
            build_pidfile_probe_command(pid_path),
        )
        completed = subprocess.run(
            ssh_command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return completed.returncode == 0

    def run_local_shell(self, cluster: ClusterSpec, command: str) -> None:
        subprocess.run(
            shell_command_argv(cluster.local_shell, cluster.local_shell_flags,
                               command),
            check=False,
            cwd=cluster.workdir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def run_remote_shell(self, cluster: ClusterSpec, host: str,
                         command: str) -> None:
        ssh_command = build_remote_ssh_command(cluster, host, command)
        subprocess.run(
            ssh_command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run handwritten multi-node offline_poisson_harness cases.")
    parser.add_argument("--list",
                        action="store_true",
                        help="List the configured case names.")
    parser.add_argument(
        "--case",
        action="append",
        help="Run a specific case by name. May be repeated.",
    )
    parser.add_argument(
        "--model",
        action="append",
        help=("Limit configured cases to the specified model. May be "
              "repeated."),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help=("Limit configured cases to the specified dataset. May be "
              "repeated."),
    )
    parser.add_argument(
        "--strategy",
        action="append",
        help=("Limit configured cases to the specified strategy. May be "
              "repeated."),
    )
    parser.add_argument(
        "--case-csv",
        default=None,
        help=("Load cases from a CSV file. CSV row order is preserved and "
              "rows with enabled=0 are ignored."),
    )
    parser.add_argument("--all",
                        action="store_true",
                        help="Run all configured cases sequentially.")
    parser.add_argument(
        "--rate-plan",
        default=DEFAULT_RATE_PLAN,
        choices=sorted(RATE_PLAN_PHASES),
        help=(
            "Request-rate sweep plan to use for --list, --all, and --case "
            "name resolution. 'coarse10' runs only 10-point spacing; "
            "'coarse10_then_mid5' appends 15/25/.../85 after the coarse pass; "
            "'issue01_40_45' targets rates 40 and 45; "
            "'full2p5' uses 2.5-point spacing from 2.5 to 90; "
            "'coarse1_then_mid0p5' combines 1-point coarse rates with 0.5-point "
            "midpoints; "
            "'coarse1_to10_then_mid0p5' limits that combination to rates <= 10."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help="Shared artifact root for run directories.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional suffix for the timestamped run directory.",
    )
    parser.add_argument(
        "--bench-duration-sec",
        type=positive_float,
        default=None,
        help=(
            "Override measured benchmark duration in seconds for selected "
            "cases. Default configured case duration is "
            f"{DEFAULT_SWEEP_BENCH_DURATION_SEC:g}s."
        ),
    )
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Write commands and manifests without execution.")
    parser.add_argument(
        "--frontend-extra-arg",
        action="append",
        default=[],
        metavar="TOKEN",
        help=(
            "Append one extra argv token to every frontend harness command. "
            "Repeat as --frontend-extra-arg=TOKEN for flags and their values "
            "separately. Supports the {benchmark_dir} placeholder."
        ),
    )
    parser.add_argument(
        "--headless-extra-arg",
        action="append",
        default=[],
        metavar="TOKEN",
        help=(
            "Append one extra argv token to every headless-engine harness "
            "command. Repeat as --headless-extra-arg=TOKEN for flags and "
            "their values separately. Supports the {benchmark_dir} placeholder."
        ),
    )
    parser.add_argument(
        "--keep-going",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Continue to later cases even if one case fails. "
              "Use --no-keep-going to stop after the first failed case."),
    )
    parser.add_argument(
        "--historical-skip-ignore-bs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("When checking historical successful results for skip decisions, "
              "treat cases that differ only in bs/max_num_seqs as matches. "
              "Enabled by default; use --no-historical-skip-ignore-bs to "
              "require exact bs matches."),
    )
    parser.add_argument(
        "--ignore-historical-skips",
        action="store_true",
        default=False,
        help=("Ignore all historical skip logic for this run and execute cases "
              "even when prior failures or successful exact matches exist."),
    )
    parser.add_argument(
        "--auto-rerun-trend-outliers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("After the main sweep finishes, detect obvious local trend "
              "outliers in tpot_by_e2e.mean and rerun those rates."),
    )
    parser.add_argument(
        "--trend-rerun-max-attempts",
        type=positive_int,
        default=DEFAULT_TREND_RERUN_MAX_ATTEMPTS,
        help=("Maximum rerun attempts per suspicious rate when "
              "--auto-rerun-trend-outliers is enabled."),
    )
    parser.add_argument(
        "--trend-rerun-relative-threshold-pct",
        type=positive_float,
        default=DEFAULT_TREND_RERUN_RELATIVE_THRESHOLD_PCT,
        help=("Relative tolerance around the interpolated local trend used "
              "to flag rerun candidates."),
    )
    parser.add_argument(
        "--trend-rerun-absolute-threshold-ms",
        type=positive_float,
        default=DEFAULT_TREND_RERUN_ABSOLUTE_THRESHOLD_MS,
        help=("Minimum absolute deviation in milliseconds required to flag "
              "a local trend outlier."),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        experiments = ([] if args.case_csv else build_experiment_matrix(
            args.rate_plan,
            models=args.model,
            datasets=args.dataset,
            strategies=args.strategy,
        ))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    selected_cases = apply_bench_duration_override(
        select_cases(args, experiments),
        args.bench_duration_sec,
    )
    selected_cases = apply_extra_argv_overrides(
        selected_cases,
        frontend_extra_args=tuple(args.frontend_extra_arg),
        headless_extra_args=tuple(args.headless_extra_arg),
    )

    if args.list:
        for case in selected_cases if args.case_csv else experiments:
            print(describe_case(case))
        return

    resolved_rate_plan = (f"case_csv:{Path(args.case_csv).name}"
                          if args.case_csv else args.rate_plan)
    runner = ManualMultinodeRunner(
        Path(args.artifact_root),
        dry_run=args.dry_run,
        keep_going=args.keep_going,
        ignore_historical_skips=args.ignore_historical_skips,
        historical_skip_ignore_bs=args.historical_skip_ignore_bs,
        auto_rerun_trend_outliers=args.auto_rerun_trend_outliers,
        trend_rerun_max_attempts=args.trend_rerun_max_attempts,
        trend_rerun_relative_threshold_pct=args.
        trend_rerun_relative_threshold_pct,
        trend_rerun_absolute_threshold_ms=args.
        trend_rerun_absolute_threshold_ms,
    )
    results = runner.run(selected_cases,
                         args.run_label,
                         rate_plan=resolved_rate_plan)

    failures = [result for result in results if result.status not in {"ok", "dry_run"}]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
