from __future__ import annotations

from nanodeploy._cpp import BlockContextSlot, Scheduler, Sequence, prepare_decode_cpp
from tests.sp_routing_oracle import (
    assert_destination_rows_match_packed_receivers,
)


_SP_SIZE = 8
_MAX_NUM_SEQS = 8
_BLOCK_SIZE = 64
_LONG_PROMPT_BUCKET_POLICY = (
    "1:1-512;5:513-768;6:769-1024;7:1025-1280;8:1281-4096"
)
_PROMPT_LENGTHS = (2048, 1792, 1536, 1280, 1024, 768, 512, 500)
_PREFILL_COMPLETION_TOKENS = 1
_EXPECTED_SP_SIZES = (8, 8, 8, 8, 7, 6, 5, 1)


def _make_scheduler() -> Scheduler:
    return Scheduler(
        "pd-decode-bucket-sp8-semantics",
        1,
        _MAX_NUM_SEQS,
        8192,
        16,
        -1,
        1,
        _SP_SIZE,
        4096,
        _BLOCK_SIZE,
        "decode",
        1.0,
        65536,
        "bucket",
        True,
        _LONG_PROMPT_BUCKET_POLICY,
        True,
        "LeastBatch",
        0,
    )


def _schedule_mixed_bucket_batch():
    scheduler = _make_scheduler()
    sequences = []
    for request_index, prompt_length in enumerate(_PROMPT_LENGTHS):
        # The prefill engine produces one completion token before handing the
        # shared Sequence to decode, so bucket selection sees prompt + 1.
        handoff_length = prompt_length + _PREFILL_COMPLETION_TOKENS
        token_base = 10_000 * (request_index + 1)
        sequence = Sequence(
            [token_base + offset for offset in range(handoff_length)],
            1e-5,
            128,
            True,
        )
        sequence.seq_id = 100 + request_index
        scheduler.add(sequence)
        sequences.append(sequence)

    migration = scheduler.schedule()
    assert migration.is_prefill
    assert list(migration.dp_seqs[0]) == sequences
    decode = scheduler.schedule()
    assert not decode.is_prefill
    return scheduler, tuple(sequences), tuple(decode.dp_seqs[0]), decode


def test_long_prompt_bucket_policy_activates_all_requested_sp_sizes():
    """The 500--2K prompt mix must exercise real SP1/5/6/7/8 placement."""

    _, sequences, _, decode = _schedule_mixed_bucket_batch()
    actual_sp_sizes = []
    for sequence in sequences:
        dispatched = list(
            sequence.block_ctx(BlockContextSlot.ACTIVE).num_dispatched_tokens
        )
        assert sum(dispatched) == sequence.num_tokens
        actual_sp_sizes.append(sum(token_count > 0 for token_count in dispatched))

    assert tuple(actual_sp_sizes) == _EXPECTED_SP_SIZES
    histogram = list(decode.sp_size_hist_per_dp[0])
    assert histogram == [0, 1, 0, 0, 0, 1, 1, 1, 4]


def test_mixed_bucket_decode_metadata_keeps_only_participating_rows():
    """Each rank's packed attention rows must match its non-zero KV shards."""

    scheduler, sequences, scheduled, _ = _schedule_mixed_bucket_batch()
    real_sequences = [
        sequence
        for sequence in scheduled
        if not scheduler.worker_state[0].is_control_dummy(sequence)
    ]
    assert real_sequences == list(sequences)

    for sp_rank in range(_SP_SIZE):
        metadata = prepare_decode_cpp(
            list(scheduled),
            sp_rank,
            _SP_SIZE,
            _BLOCK_SIZE,
            _MAX_NUM_SEQS,
        )
        rows_by_master: list[list[int]] = [[] for _ in range(_SP_SIZE)]
        for sequence in scheduled:
            block_ctx = sequence.block_ctx(BlockContextSlot.ACTIVE)
            local_tokens = block_ctx.num_dispatched_tokens[sp_rank]
            if local_tokens > 0:
                rows_by_master[block_ctx.master_sp_idx].append(local_tokens)

        expected_rows = [
            local_tokens
            for master_rows in rows_by_master
            for local_tokens in master_rows
        ]
        assert list(metadata.context_lens_for_attn) == expected_rows
        assert list(metadata.q_offsets) == [
            sum(len(rows) for rows in rows_by_master[:master])
            for master in range(_SP_SIZE + 1)
        ]
        assert metadata.use_sp_a2a


def test_mixed_bucket_destination_rows_have_no_collisions_or_missing_rows():
    """Every sender must write the row in each receiver's packed layout."""

    _, _, scheduled, _ = _schedule_mixed_bucket_batch()
    assert_destination_rows_match_packed_receivers(
        scheduled,
        sp_size=_SP_SIZE,
        max_num_seqs=_MAX_NUM_SEQS,
        block_size=_BLOCK_SIZE,
    )
