# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileStrategyDefaults:
    max_num_seqs: int
    gpu_memory_utilization: float


# Strategies with built-in defaults for prepare_custom_lens_case.py.
SUPPORTED_PROFILE_STRATEGIES: tuple[str, ...] = (
    "dp4dcp8",
    "dp8dcp4",
    "dp16cp2",
    "dp32",
    "dp4tp4",
)

STRATEGY_PROFILE_DEFAULTS: dict[str, ProfileStrategyDefaults] = {
    "dp4dcp8": ProfileStrategyDefaults(
        max_num_seqs=1024,
        gpu_memory_utilization=0.85,
    ),
    "dp8dcp4": ProfileStrategyDefaults(
        max_num_seqs=768,
        gpu_memory_utilization=0.85,
    ),
    "dp16cp2": ProfileStrategyDefaults(
        max_num_seqs=384,
        gpu_memory_utilization=0.85,
    ),
    "dp32": ProfileStrategyDefaults(
        max_num_seqs=256,
        gpu_memory_utilization=0.87,
    ),
    "dp4tp4": ProfileStrategyDefaults(
        max_num_seqs=384,
        gpu_memory_utilization=0.85,
    ),
}


def get_strategy_profile_defaults(strategy: str) -> ProfileStrategyDefaults:
    try:
        return STRATEGY_PROFILE_DEFAULTS[strategy]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_PROFILE_STRATEGIES)
        raise ValueError(
            f"No built-in offline profile defaults for strategy '{strategy}'. "
            f"Strategies with defaults: {supported}. Pass --max-num-seqs and "
            "--gpu-memory-utilization explicitly for other strategies."
        ) from exc
