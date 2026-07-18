from dlengine.context_v2.cache import plan_mla_hisparse_capacity


def _plan(*, gpu_cache_budget: int, host_cache_budget: int):
    return plan_mla_hisparse_capacity(
        gpu_cache_budget=gpu_cache_budget,
        host_cache_budget=host_cache_budget,
        max_num_seqs=4,
        device_buffer_size=128,
        block_size=64,
        kv_block_bytes=1_000,
        indexer_block_bytes=250,
    )


def test_mla_hisparse_reserves_buffer_before_sizing_indexer():
    # 4 sequences * (2 Buffer blocks + 1 generation block) * 1000 bytes.
    plan = _plan(gpu_cache_budget=20_000, host_cache_budget=100_000)

    assert plan.hot_blocks == 12
    assert plan.hot_tier_bytes == 12_000
    assert plan.indexer_budget_bytes == 8_000
    assert plan.indexer_blocks == 32
    assert plan.logical_blocks == 32
    assert plan.indexer_tokens == 2_048
    assert plan.logical_tokens == 2_048
    assert plan.limiting_tier == "gpu_indexer"
    assert plan.unused_indexer_budget_bytes == 0
    assert plan.unused_host_budget_bytes == 68_000


def test_mla_hisparse_capacity_is_limited_by_host_memory():
    plan = _plan(gpu_cache_budget=20_000, host_cache_budget=7_999)

    assert plan.indexer_blocks == 32
    assert plan.host_blocks == 7
    assert plan.logical_blocks == 7
    assert plan.host_tokens == 448
    assert plan.logical_tokens == 448
    assert plan.limiting_tier == "host_mla"
    assert plan.unused_indexer_budget_bytes == 6_250
    assert plan.unused_host_budget_bytes == 999


def test_mla_hisparse_capacity_is_limited_by_indexer_memory():
    plan = _plan(gpu_cache_budget=14_000, host_cache_budget=100_000)

    assert plan.indexer_blocks == 8
    assert plan.host_blocks == 100
    assert plan.logical_blocks == 8
    assert plan.limiting_tier == "gpu_indexer"
    assert plan.unused_indexer_budget_bytes == 0
    assert plan.unused_host_budget_bytes == 92_000


def test_mla_hisparse_has_no_capacity_when_buffer_exhausts_hbm():
    plan = _plan(gpu_cache_budget=11_999, host_cache_budget=100_000)

    assert plan.indexer_budget_bytes == 0
    assert plan.indexer_blocks == 0
    assert plan.logical_blocks == 0
    assert plan.buffer_hbm_shortfall_bytes == 1
