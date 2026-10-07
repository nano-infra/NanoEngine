#!/usr/bin/env python3
"""Run the shared NanoDeploy service driver with Fig. 14 worker networking."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
from typing import Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
AE_ROOT = SCRIPT_DIR.parent
NANO_BENCHMARK = AE_ROOT / "start-e2e" / "nano" / "bench_serving_overhead.py"

WORKER_NETWORK_ENV_NAMES = (
    "GLOO_SOCKET_IFNAME",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_TC",
    "SLIME_VISIBLE_DEVICES",
    "SLIME_GID_INDEX",
    "SLIME_QP_NUM",
)


def merge_worker_runtime_env(
    runtime_env: Mapping[str, object] | None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return a copy of ``runtime_env`` with the validated network variables."""

    source = os.environ if environ is None else environ
    merged = dict(runtime_env or {})
    env_vars = dict(merged.get("env_vars") or {})
    env_vars.update(
        {name: source[name] for name in WORKER_NETWORK_ENV_NAMES if name in source}
    )
    merged["env_vars"] = env_vars
    return merged


def patch_ray_worker_environment() -> None:
    """Inject driver-side network settings into every NanoDeploy Ray actor."""

    from nanodeploy.engine import ray_executor

    original_model_runner = ray_executor.ModelRunner

    class ModelRunnerWithNetworkEnv:
        @staticmethod
        def options(**kwargs):
            kwargs["runtime_env"] = merge_worker_runtime_env(
                kwargs.get("runtime_env")
            )
            return original_model_runner.options(**kwargs)

    ray_executor.ModelRunner = ModelRunnerWithNetworkEnv


def main() -> None:
    if not NANO_BENCHMARK.is_file():
        raise SystemExit(f"NanoDeploy E2E driver not found: {NANO_BENCHMARK}")
    benchmark = runpy.run_path(
        str(NANO_BENCHMARK),
        run_name="fig14_shared_nano_benchmark",
    )
    parsed_args = benchmark["parse_args"]()
    benchmark_main = benchmark["main"]
    benchmark_main.__globals__["parse_args"] = lambda: parsed_args
    patch_ray_worker_environment()
    benchmark_main()


if __name__ == "__main__":
    main()
