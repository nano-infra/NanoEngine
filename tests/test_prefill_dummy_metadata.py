from nanodeploy._cpp import (
    BlockContextSlot,
    Scheduler,
    Sequence,
    prepare_decode_cpp,
    prepare_prefill_cpp,
)


_ATTENTION_DP = 8
_ATTENTION_SP = 1
_BLOCK_SIZE = 64
_MAX_NUM_SEQS = 8


def _make_scheduler(mode: str = "prefill") -> Scheduler:
    return Scheduler(
        "prefill-test",
        1,
        _MAX_NUM_SEQS,
        4096,
        16,
        -1,
        _ATTENTION_DP,
        _ATTENTION_SP,
        1024,
        _BLOCK_SIZE,
        mode,
        1.0,
        512,
        "legacy",
        False,
        "",
        False,
        "RoundRobin",
        0,
    )


def test_prefill_empty_lanes_do_not_receive_decode_control_dummies():
    scheduler = _make_scheduler()
    sequence = Sequence(list(range(37)), 0.1, 8, True)
    scheduler.add(sequence)

    result = scheduler.schedule()

    assert result.is_prefill
    assert [len(seqs) for seqs in result.dp_sp_seqs] == [1] + [0] * 7

    metadata = prepare_prefill_cpp(
        result.dp_sp_seqs[0],
        0,
        _ATTENTION_SP,
        _BLOCK_SIZE,
        _MAX_NUM_SEQS,
    )
    assert len(metadata.input_ids) == 37
    assert len(metadata.input_ids) == len(metadata.slot_mapping)


def test_ephemeral_prefill_dummy_and_persistent_decode_dummy_are_separate():
    ephemeral = Sequence([0], 1.0, 1, True)
    ephemeral.active("prefill-test", _ATTENTION_SP, 1)
    ephemeral.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx = 0

    prefill_metadata = prepare_prefill_cpp(
        [ephemeral],
        0,
        _ATTENTION_SP,
        _BLOCK_SIZE,
        _MAX_NUM_SEQS,
    )
    assert len(prefill_metadata.input_ids) == 1
    assert len(prefill_metadata.slot_mapping) == 0

    scheduler = _make_scheduler()
    persistent = scheduler.worker_state[0].dummy_seqs[0]
    decode_metadata = prepare_decode_cpp(
        [persistent],
        0,
        _ATTENTION_SP,
        _BLOCK_SIZE,
        _MAX_NUM_SEQS,
    )
    assert len(decode_metadata.input_ids) == 1
    assert len(decode_metadata.slot_mapping) == 1


def test_decode_schedule_still_fills_empty_lanes_with_persistent_dummies():
    scheduler = _make_scheduler(mode="decode")
    sequence = Sequence(list(range(37)), 0.1, 8, True)
    scheduler.add(sequence)

    migration = scheduler.schedule()
    assert migration.is_prefill
    assert [len(seqs) for seqs in migration.dp_sp_seqs] == [1] + [0] * 7

    decode = scheduler.schedule()
    assert not decode.is_prefill
    assert [len(seqs) for seqs in decode.dp_sp_seqs] == [1] * 8
    for dp_idx, seqs in enumerate(decode.dp_sp_seqs):
        metadata = prepare_decode_cpp(
            seqs,
            0,
            _ATTENTION_SP,
            _BLOCK_SIZE,
            _MAX_NUM_SEQS,
        )
        assert len(metadata.input_ids) == len(metadata.slot_mapping) == 1
        if dp_idx > 0:
            assert scheduler.worker_state[dp_idx].is_control_dummy(seqs[0])
