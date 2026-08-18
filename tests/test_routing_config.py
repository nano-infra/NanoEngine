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


@pytest.mark.parametrize("name", ["VLLMLoadBalance", "random"])
def test_legacy_routing_strategy_rejects_removed_or_unknown_value(name):
    with pytest.raises(ValueError, match="routing_strategy must be one of"):
        make_config(routing_strategy=name)


def test_routing_strategy_binding_only_exports_supported_values():
    assert set(RoutingStrategy.__members__) == {
        "RoundRobin",
        "LeastBatch",
        "LeastCache",
    }


@pytest.mark.parametrize(
    "router_policy",
    ["round_robin", "least_batch", "least_batch_v2", "least_cache"],
)
def test_router_policy_config(router_policy):
    config = make_config(router_policy=router_policy)
    assert config.router_policy == router_policy


def test_router_policy_defaults_to_least_batch():
    config = make_config()
    assert config.router_policy == "least_batch"
    assert config.sp_master_selector == "LeastBatch"


def test_router_policy_config_rejects_unknown_value():
    with pytest.raises(ValueError, match="router_policy must be one of"):
        make_config(router_policy="random")


def test_uniform_random_moe_routing_config():
    config = make_config(
        moe_routing_simulation_strategy="uniform_random",
        seed=0,
    )

    assert config.moe_routing_simulation_strategy == "uniform_random"
    assert config.seed == 0


def test_legacy_perfect_eplb_maps_to_explicit_strategy():
    config = make_config(perfect_eplb=True)

    assert config.moe_routing_simulation_strategy == "perfect_eplb"


def test_uniform_random_rejects_legacy_perfect_eplb():
    with pytest.raises(ValueError, match="perfect_eplb=True conflicts"):
        make_config(
            perfect_eplb=True,
            moe_routing_simulation_strategy="uniform_random",
        )


def test_moe_routing_seed_changes_collective_fingerprint():
    baseline = make_config(
        moe_routing_simulation_strategy="uniform_random",
        seed=0,
    )
    changed = make_config(
        moe_routing_simulation_strategy="uniform_random",
        seed=1,
    )

    assert baseline.collective_fingerprint() != changed.collective_fingerprint()


def test_zmq_worker_transport_requires_hierarchical_scheduler():
    with pytest.raises(
        ValueError,
        match="hierarchical_worker_transport='zmq' requires",
    ):
        make_config(hierarchical_worker_transport="zmq")


def test_worker_transport_rejects_unknown_value():
    with pytest.raises(
        ValueError,
        match="hierarchical_worker_transport must be one of",
    ):
        make_config(hierarchical_worker_transport="socket")


def test_removed_scheduler_mode_is_not_a_config_field():
    with pytest.raises(TypeError, match="scheduler_mode"):
        Config(  # type: ignore[call-arg]
            model=str(DEEPSEEK_MODEL),
            scheduler_mode="decentralized",
        )
