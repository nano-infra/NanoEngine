import dataclasses
from typing import Optional

from nanodeploy.logging import get_logger

# Initialize logger with NANODEPLOY namespace
logger = get_logger()


@dataclasses.dataclass
class RunnerConfig:
    max_num_seqs: int | None = None
    dummy_weight: bool = False
    dummy_eplb: bool = False
    enable_eplb: bool = False
    device_comm_backend: str = "nccl"
    # Mega-MoE: opt into deep_gemm.fp8_fp4_mega_moe for the routed
    # decode path. See nanodeploy/config.py for details.
    use_mega_moe: bool = False
    mega_moe_max_tokens_per_rank: int = 256


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
    )


def reset_runner_config():
    global _RUNNER_CONFIG
    _RUNNER_CONFIG = RunnerConfig()
