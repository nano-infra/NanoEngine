"""Frozen formal profile helpers for LoongServe-style Issue 1% runs.

This module intentionally contains no Ray, CUDA, or NanoDeploy imports so the
formal command-line contract can be tested on CPU-only hosts.
"""

from __future__ import annotations

import argparse
import os
from typing import Any


PROFILE_NAME = "loong_decode_issue001"
ROUTING_STRATEGY = "RoundRobin"
PROXY_ENV_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")

_FROZEN_ENGINE_KWARGS: dict[str, Any] = {
    "enable_ls_decode_core_scheduler": True,
    "ls_decode_profile": PROFILE_NAME,
    "ls_running_max_req_size": 1000,
    "ls_admission_max_tokens_per_pool": "auto",
    "ls_min_comp_bound_decoding_batch_size": 128,
    "ls_decode_initial_kv_dop": 0,
    "ls_decode_enable_memory_scale_up": True,
    "ls_disable_scale_up": False,
    "ls_decode_enable_future_kv_admission": True,
    "ls_kv_consolidation_mode": "execute",
    "ls_kv_consolidation_candidate_util": 0.50,
    "ls_kv_consolidation_target_high_watermark": 0.80,
    "ls_kv_consolidation_stable_steps": 2,
    "ls_kv_consolidation_cooldown_steps": 2,
    "ls_kv_consolidation_check_interval_steps": 1,
    "ls_kv_consolidation_max_source_blocks_per_event": 128,
    "ls_kv_consolidation_migration_chunk_tokens": 64,
    "dummy_bootstrap_token_id": 0,
    "pause_mode": "offload",
    "dp_assignment": "arrival_round_robin",
    "cross_dp_scale_up": False,
    "routing_strategy": ROUTING_STRATEGY,
}

_EXPECTED_RESOLVED_VALUES: dict[str, Any] = {
    "profile": PROFILE_NAME,
    "mode": "decode",
    "dummy_prefill": True,
    "dummy_bootstrap_token_id": 0,
    "max_tokens_min": 1,
    "ignore_eos_required": True,
    "loop_count": 1,
    "scheduler_mode": "centralized",
    "routing_strategy": ROUTING_STRATEGY,
    "dp_assignment": "arrival_round_robin",
    "pause_mode": "offload",
    "cross_dp_scale_up": False,
    "attention_sp": 8,
    "attention_tp": 1,
    "ffn_dp": 1,
    "ffn_tp": 1,
    "kvcache_block_size": 64,
    "fixed_sp_size": 0,
    "enable_dynamic_sp_size": False,
    "sp_backend": "hao_basic",
    "use_dlslime_rpc": True,
    "ls_running_max_req_size": 1000,
    "ls_admission_max_tokens_per_pool_configured": "auto",
    "ls_min_comp_bound_decoding_batch_size": 128,
    "ls_decode_enable_future_kv_admission": True,
    "ls_decode_initial_kv_dop": 0,
    "initial_dop_policy": "min_exact_feasible",
    "ls_disable_scale_up": False,
    "ls_kv_consolidation_mode": "execute",
    "ls_kv_consolidation_candidate_util": 0.50,
    "ls_kv_consolidation_target_high_watermark": 0.80,
    "ls_kv_consolidation_stable_steps": 2,
    "ls_kv_consolidation_cooldown_steps": 2,
    "ls_kv_consolidation_check_interval_steps": 1,
    "ls_kv_consolidation_max_source_blocks_per_event": 128,
    "ls_kv_consolidation_migration_chunk_tokens": 64,
}


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def add_formal_profile_arguments(parser: argparse.ArgumentParser) -> None:
    """Add policy arguments shared by every formal Issue001 entry point."""
    parser.add_argument(
        "--ls-max-num-ooe",
        type=_nonnegative_int,
        required=True,
        help=(
            "Explicit workload-manifest OOE limit. The issue001 profile never "
            "inherits an implicit API or benchmark default."
        ),
    )
    parser.add_argument(
        "--routing-strategy",
        choices=(ROUTING_STRATEGY,),
        default=ROUTING_STRATEGY,
        help="Frozen for manifest clarity; LS DP assignment is arrival round-robin.",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        choices=(0,),
        default=0,
        help=(
            "Formal runs disable request warmup because scheduler state may only "
            "be initialized at engine construction. Use a separate process for "
            "GPU warmup/capture validation."
        ),
    )


def formal_engine_kwargs(
    ls_max_num_ooe: int, *, ls_decode_batch_per_master: int
) -> dict[str, Any]:
    """Return a fresh kwargs dict for the frozen issue001 engine profile."""
    if (
        isinstance(ls_max_num_ooe, bool)
        or not isinstance(ls_max_num_ooe, int)
        or ls_max_num_ooe < 0
    ):
        raise ValueError("ls_max_num_ooe must be an explicit non-negative integer")
    if (
        isinstance(ls_decode_batch_per_master, bool)
        or not isinstance(ls_decode_batch_per_master, int)
        or ls_decode_batch_per_master <= 0
    ):
        raise ValueError("ls_decode_batch_per_master must be a positive integer")
    kwargs = dict(_FROZEN_ENGINE_KWARGS)
    kwargs["ls_max_num_ooe"] = ls_max_num_ooe
    # This legacy ABI value is not a policy branch in the source-aligned path,
    # but it remains explicit and recorded until the constructor ABI is removed.
    kwargs["ls_decode_batch_per_master"] = ls_decode_batch_per_master
    return kwargs


def resolved_manifest(
    config: Any,
    expected_ls_max_num_ooe: int,
    *,
    expected_loop_count: int = 1,
) -> dict[str, Any]:
    """Validate and return the Scheduler-resolved formal manifest fragment."""
    if (
        isinstance(expected_loop_count, bool)
        or not isinstance(expected_loop_count, int)
        or not 1 <= expected_loop_count <= 16
    ):
        raise ValueError("expected_loop_count must be an integer in [1, 16]")
    manifest = dict(config.ls_decode_manifest())
    # These two resolved values still live on Config but are not part of the
    # core manifest fragment yet. Record them here so benchmark artifacts are
    # complete without duplicating a second set of CLI defaults.
    manifest["ls_decode_enable_memory_scale_up"] = (
        config.ls_decode_enable_memory_scale_up
    )
    manifest["ls_decode_batch_per_master"] = config.ls_decode_batch_per_master

    expected = dict(_EXPECTED_RESOLVED_VALUES)
    expected["ls_max_num_ooe"] = expected_ls_max_num_ooe
    expected["loop_count"] = expected_loop_count
    expected["ls_decode_enable_memory_scale_up"] = True
    for key, expected_value in expected.items():
        actual = manifest.get(key)
        if actual != expected_value:
            raise RuntimeError(
                "formal issue001 profile drift: "
                f"{key} resolved to {actual!r}, expected {expected_value!r}"
            )

    resolved_admission_limit = manifest.get("ls_admission_max_tokens_per_pool")
    if (
        isinstance(resolved_admission_limit, bool)
        or not isinstance(resolved_admission_limit, int)
        or resolved_admission_limit <= 0
    ):
        raise RuntimeError(
            "formal issue001 profile drift: "
            "ls_admission_max_tokens_per_pool was not resolved at construction"
        )

    attention_dp = manifest.get("attention_dp")
    if attention_dp not in (1, 2, 4):
        raise RuntimeError(
            "formal issue001 profile drift: attention_dp must be one of 1, 2, 4"
        )
    if manifest.get("ffn_ep") != attention_dp * 8:
        raise RuntimeError(
            "formal issue001 profile drift: ffn_ep must equal attention_dp * 8"
        )
    return manifest


def clear_ray_proxy_env() -> list[str]:
    """Ensure Ray Client/GCS traffic does not inherit an HTTP proxy."""
    cleared: list[str] = []
    for key in PROXY_ENV_KEYS:
        if key in os.environ:
            cleared.append(key)
            os.environ.pop(key, None)
    return cleared
