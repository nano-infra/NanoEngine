from pathlib import Path

import pytest

from nanodeploy.config import Config
from nanodeploy.engine.scheduler import RoutingStrategy, Scheduler


DEEPSEEK_MODEL = Path(
    "/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3"
)


pytestmark = pytest.mark.skipif(
    not (DEEPSEEK_MODEL / "config.json").is_file(),
    reason=f"DeepSeek-V3 config not found at {DEEPSEEK_MODEL}",
)


def make_config(**overrides) -> Config:
    values = {
        "model": str(DEEPSEEK_MODEL),
        "kvcache_block_size": 64,
        "num_kvcache_blocks": 32,
    }
    values.update(overrides)
    return Config(**values)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("RoundRobin", RoutingStrategy.RoundRobin),
        ("LeastBatch", RoutingStrategy.LeastBatch),
        ("LeastCache", RoutingStrategy.LeastCache),
    ],
)
def test_legacy_routing_strategy(name, expected):
    scheduler = Scheduler(make_config(routing_strategy=name))
    assert scheduler.routing_strategy == expected


def test_removed_scheduler_mode_is_not_a_config_field():
    with pytest.raises(TypeError, match="scheduler_mode"):
        Config(  # type: ignore[call-arg]
            model=str(DEEPSEEK_MODEL),
            scheduler_mode="decentralized",
        )
