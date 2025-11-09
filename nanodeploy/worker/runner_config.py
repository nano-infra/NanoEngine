import dataclasses
from typing import Optional

from nanodeploy.logging import get_logger

# Initialize logger with NANODEPLOY namespace
logger = get_logger()


@dataclasses.dataclass
class RunnerConfig:
    dummy_weight: bool = False
    perfect_eplb: bool = False


# Singleton instance of RunnerConfig
_RUNNER_CONFIG = RunnerConfig()


def get_runner_config() -> RunnerConfig:
    return _RUNNER_CONFIG


def set_runner_config(
    dummy_weight: Optional[bool] = None,
    perfect_eplb: Optional[bool] = None,
):
    global _RUNNER_CONFIG
    _RUNNER_CONFIG = RunnerConfig(dummy_weight=dummy_weight, perfect_eplb=perfect_eplb)


def reset_runner_config():
    global _RUNNER_CONFIG
    _RUNNER_CONFIG = RunnerConfig()
