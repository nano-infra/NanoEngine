import dataclasses
from typing import Optional

from dlengine.logging import get_logger

# Initialize logger with DLENGINE namespace
logger = get_logger()


@dataclasses.dataclass
class RunnerConfig:
    max_num_seqs: int | None = None
    dummy_weight: bool = False
    dummy_eplb: bool = False
    enable_eplb: bool = False
    device_comm_backend: str = "nccl"
    # Mega-MoE: opt into deep_gemm.fp8_fp4_mega_moe for the routed
    # decode path. See dlengine/config.py for details.
    use_mega_moe: bool = False
    mega_moe_max_tokens_per_rank: int = 256
    # Per-step host-critical-path timer. Set ``step_timing=True`` to
    # log a phase breakdown of run_from_bytes every
    # ``step_timing_interval`` decode steps. Useful for quantifying
    # the GPU-idle gap between cudaGraphLaunch invocations (the
    # device-signal + pingpong-scheduling target). Driver reads env
    # vars once and threads these into RunnerConfig so Ray actors
    # see the same value regardless of subprocess env propagation.
    step_timing: bool = False
    step_timing_interval: int = 16
    step_timing_rank: int = 0  # -1 = all ranks


# Singleton instance of RunnerConfig
_RUNNER_CONFIG = RunnerConfig()


def get_runner_config() -> RunnerConfig:
    return _RUNNER_CONFIG


def set_runner_config(
    max_num_seqs: int | None = None,
    dummy_weight: Optional[bool] = None,
    dummy_eplb: Optional[bool] = None,
    enable_eplb: Optional[bool] = None,
    use_mega_moe: Optional[bool] = None,
    mega_moe_max_tokens_per_rank: Optional[int] = None,
    step_timing: Optional[bool] = None,
    step_timing_interval: Optional[int] = None,
    step_timing_rank: Optional[int] = None,
):
    global _RUNNER_CONFIG
    _RUNNER_CONFIG = RunnerConfig(
        max_num_seqs=max_num_seqs,
        dummy_weight=dummy_weight,
        dummy_eplb=dummy_eplb,
        enable_eplb=enable_eplb,
        use_mega_moe=bool(use_mega_moe) if use_mega_moe is not None else False,
        mega_moe_max_tokens_per_rank=(
            int(mega_moe_max_tokens_per_rank)
            if mega_moe_max_tokens_per_rank is not None
            else 256
        ),
        step_timing=bool(step_timing) if step_timing is not None else False,
        step_timing_interval=(
            int(step_timing_interval) if step_timing_interval is not None else 16
        ),
        step_timing_rank=(int(step_timing_rank) if step_timing_rank is not None else 0),
    )


def reset_runner_config():
    global _RUNNER_CONFIG
    _RUNNER_CONFIG = RunnerConfig()
