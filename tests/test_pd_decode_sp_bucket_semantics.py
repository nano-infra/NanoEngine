from __future__ import annotations

from nanodeploy._cpp import BlockContextSlot, Scheduler, Sequence, prepare_decode_cpp


_SP_SIZE = 8
_MAX_NUM_SEQS = 8
_BLOCK_SIZE = 64
_SHORT_BUCKET_POLICY = (
    "1:1-127;5:128-383;6:384-639;7:640-895;8:896-4096"
)
_PROMPT_LENGTHS = (1024, 768, 512, 256, 64)
_EXPECTED_SP_SIZES = (8, 7, 6, 5, 1)


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
        _SHORT_BUCKET_POLICY,
        True,
        "LeastBatch",
        0,
    )


def _schedule_mixed_bucket_batch():
    scheduler = _make_scheduler()
    sequences = []
    for request_index, prompt_length in enumerate(_PROMPT_LENGTHS):
        token_base = 10_000 * (request_index + 1)
        sequence = Sequence(
            [token_base + offset for offset in range(prompt_length)],
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


def test_short_bucket_policy_activates_all_production_sp_sizes():
    """The compressed policy must exercise real SP1/5/6/7/8 placement."""

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
    assert histogram == [0, 1, 0, 0, 0, 1, 1, 1, 1]


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
