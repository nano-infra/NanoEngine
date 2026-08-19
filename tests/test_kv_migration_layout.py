from types import SimpleNamespace

import pytest

from nanodeploy.worker.cache import _plan_kv_migration_ranges


def _block_ctx(
    dispatched_tokens: list[int],
    master_sp_rank: int,
    block_tables: dict[int, list[int]],
):
    return SimpleNamespace(
        num_dispatched_tokens=dispatched_tokens,
        master_sp_idx=master_sp_rank,
        sp_block_table={
            sp_rank: block_tables.get(sp_rank, [])
            for sp_rank in range(len(dispatched_tokens))
        },
    )


def test_pd_migration_repartitions_one_prefill_block_across_sp8():
    remote_ctx = _block_ctx([38], 0, {0: [11]})
    local_ctx = _block_ctx(
        [5, 5, 5, 5, 5, 5, 4, 4],
        0,
        {sp_rank: [100 + sp_rank] for sp_rank in range(8)},
    )

    ranges_per_rank = [
        _plan_kv_migration_ranges(remote_ctx, local_ctx, 64, sp_rank)
        for sp_rank in range(8)
    ]

    assert [sum(item.num_tokens for item in ranges) for ranges in ranges_per_rank] == [
        4,
        5,
        5,
        5,
        5,
        5,
        4,
        4,
    ]
    assert [ranges[0].remote_token_offset for ranges in ranges_per_rank] == [
        0,
        4,
        9,
        14,
        19,
        24,
        29,
        33,
    ]
    assert all(ranges[0].remote_block_id == 11 for ranges in ranges_per_rank)
    assert all(len(ranges) == 1 for ranges in ranges_per_rank)


def test_pd_migration_splits_ranges_at_physical_block_boundaries():
    remote_ctx = _block_ctx([131], 0, {0: [10, 20, 30]})
    local_ctx = _block_ctx([66, 65], 0, {0: [100, 101], 1: [200, 201]})

    rank_zero = _plan_kv_migration_ranges(remote_ctx, local_ctx, 64, 0)
    rank_one = _plan_kv_migration_ranges(remote_ctx, local_ctx, 64, 1)

    assert [item.num_tokens for item in rank_zero] == [64, 1]
    assert [item.remote_block_id for item in rank_zero] == [10, 20]
    assert [item.local_block_id for item in rank_zero] == [100, 101]
    assert [item.num_tokens for item in rank_one] == [63, 1, 1]
    assert [item.remote_block_id for item in rank_one] == [20, 30, 30]
    assert [item.local_block_id for item in rank_one] == [200, 200, 201]


def test_pd_migration_rejects_different_cached_token_counts():
    remote_ctx = _block_ctx([38], 0, {0: [11]})
    local_ctx = _block_ctx([5, 5], 0, {0: [100], 1: [101]})

    with pytest.raises(RuntimeError, match="different cached-token counts"):
        _plan_kv_migration_ranges(remote_ctx, local_ctx, 64, 0)
