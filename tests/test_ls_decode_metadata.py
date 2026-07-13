from __future__ import annotations

from nanodeploy._cpp import (
    BlockContextSlot,
    SPStateManager,
    Sequence,
    SequenceStatus,
    prepare_decode_cpp,
)


_SP_SIZE = 8
_BLOCK_SIZE = 256
_MAX_NUM_SEQS = 16


def _make_manager() -> SPStateManager:
    return SPStateManager(
        engine_id="ls-metadata-test",
        attention_sp=_SP_SIZE,
        num_kvcache_blocks=32,
        kvcache_block_size=_BLOCK_SIZE,
        max_num_seqs=_MAX_NUM_SEQS,
        max_num_batched_tokens=10_000,
        max_num_recv_seqs=_MAX_NUM_SEQS,
        reserved_blocks_per_req=0.0,
        enable_dynamic_sp_size=False,
        enable_non_uniform_split=False,
        sp_master_selector="RoundRobin",
    )


def _make_allocated_pending_sequence(
    manager: SPStateManager,
) -> Sequence:
    # Four committed prompt tokens followed by one pending dummy token. The
    # initial block is owned by rank 0; rank 1 has no historical KV.
    seq = Sequence([100, 101, 102, 103, 104], 1.0, 16, False)
    seq.active("ls-metadata-test", _SP_SIZE, 1)
    seq.status = SequenceStatus.RUNNING
    seq.num_prompt_tokens = 4

    ctx = seq.block_ctx(BlockContextSlot.ACTIVE)
    ctx.master_sp_idx = 0
    ctx.num_dispatched_tokens = [5, 0, 0, 0, 0, 0, 0, 0]
    ctx.pending_token_present = True
    ctx.pending_token_target_sp = 0
    manager.allocate_ls_initial(seq)
    return seq


def _move_pending_token_to_zero_history_rank() -> tuple[SPStateManager, Sequence]:
    manager = _make_manager()
    seq = _make_allocated_pending_sequence(manager)
    assert manager.reassign_pending_append(seq, 1)
    return manager, seq


def test_pending_reassign_preserves_history_and_allocates_new_rank_slot() -> None:
    manager = _make_manager()
    seq = _make_allocated_pending_sequence(manager)
    ctx = seq.block_ctx(BlockContextSlot.ACTIVE)

    committed_before = [
        seq.committed_context_len(BlockContextSlot.ACTIVE, sp_idx)
        for sp_idx in range(_SP_SIZE)
    ]
    used_before = list(manager.group_used_kv_tokens([seq]))
    old_owner_blocks = list(ctx.sp_block_table[0])

    assert manager.reassign_pending_append(seq, 1)

    assert ctx.master_sp_idx == 1
    assert ctx.pending_token_present is True
    assert ctx.pending_token_target_sp == 1
    assert list(ctx.num_dispatched_tokens[:2]) == [4, 1]
    assert [
        seq.committed_context_len(BlockContextSlot.ACTIVE, sp_idx)
        for sp_idx in range(_SP_SIZE)
    ] == committed_before
    assert list(manager.group_used_kv_tokens([seq])) == used_before
    assert list(ctx.sp_block_table[0]) == old_owner_blocks
    assert len(ctx.sp_block_table[1]) == 1

    meta = prepare_decode_cpp(
        [seq], 1, _SP_SIZE, _BLOCK_SIZE, _MAX_NUM_SEQS
    )
    new_rank_block = ctx.sp_block_table[1][0]
    assert list(meta.input_ids) == [seq.last_token]
    assert list(meta.positions) == [seq.num_tokens - 1]
    assert list(meta.slot_mapping) == [new_rank_block * _BLOCK_SIZE]
    assert list(meta.context_lens_for_attn) == [1]
    assert meta.context_lens_flat[_MAX_NUM_SEQS] == 1
    assert meta.global_context_lens_flat[0] == 4
    assert meta.global_context_lens_flat[_MAX_NUM_SEQS] == 1


def test_pending_reassign_capacity_failure_is_atomic() -> None:
    manager = _make_manager()
    seq = _make_allocated_pending_sequence(manager)

    # Each manager already owns one permanent dummy block. Exhaust rank 1's
    # remaining 31 blocks so a zero-history destination cannot reserve the
    # current input plus the next sampled token.
    filler = Sequence(list(range(31 * _BLOCK_SIZE)), 1.0, 16, False)
    filler.active("ls-metadata-test", _SP_SIZE, 1)
    filler.block_ctx(BlockContextSlot.ACTIVE).num_dispatched_tokens = [
        0,
        31 * _BLOCK_SIZE,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    manager.block_manager[1].allocate_uncached(filler)

    ctx = seq.block_ctx(BlockContextSlot.ACTIVE)
    before = (
        list(ctx.num_dispatched_tokens),
        ctx.pending_token_present,
        ctx.pending_token_target_sp,
        ctx.master_sp_idx,
        [list(ctx.sp_block_table[rank]) for rank in range(_SP_SIZE)],
    )

    assert manager.reassign_pending_append(seq, 1) is False
    assert (
        list(ctx.num_dispatched_tokens),
        ctx.pending_token_present,
        ctx.pending_token_target_sp,
        ctx.master_sp_idx,
        [list(ctx.sp_block_table[rank]) for rank in range(_SP_SIZE)],
    ) == before


def test_native_q_offsets_are_global_and_owner_views_gather_without_overlap() -> None:
    manager = _make_manager()
    seqs = []
    for master in (0, 1):
        seq = Sequence([100, 101, 102, 103, 104], 1.0, 16, False)
        seq.active("ls-metadata-test", _SP_SIZE, 1)
        seq.status = SequenceStatus.RUNNING
        seq.num_prompt_tokens = 4
        ctx = seq.block_ctx(BlockContextSlot.ACTIVE)
        ctx.master_sp_idx = master
        placement = [0] * _SP_SIZE
        placement[master] = 1
        placement[2] = 4
        ctx.num_dispatched_tokens = placement
        ctx.pending_token_present = True
        ctx.pending_token_target_sp = master
        manager.allocate_ls_initial(seq)
        seqs.append(seq)

    metadata = [
        prepare_decode_cpp(seqs, rank, _SP_SIZE, _BLOCK_SIZE, _MAX_NUM_SEQS)
        for rank in range(_SP_SIZE)
    ]
    expected_offsets = [0, 1, 2, 2, 2, 2, 2, 2, 2]
    assert all(list(meta.q_native_offsets) == expected_offsets for meta in metadata)
    for source, meta in enumerate(metadata):
        assert (
            meta.q_native_offsets[source + 1] - meta.q_native_offsets[source]
            == len(meta.input_ids)
        )

    assert list(metadata[0].q_native_slice_fill) == [0]
    assert list(metadata[1].q_native_slice_fill) == [1]
    assert list(metadata[0].q_native_gather_indices) == [0]
    assert list(metadata[1].q_native_gather_indices) == [1]
    assert list(metadata[2].q_native_gather_indices) == [0, 1]
    for meta in metadata:
        gather = list(meta.q_native_gather_indices)
        assert len(gather) == len(meta.context_lens_for_attn)
        assert len(gather) == len(set(gather))
        assert all(0 <= row < expected_offsets[-1] for row in gather)


def test_sole_remote_history_owner_forces_sp_attention_on_every_rank() -> None:
    manager, seq = _move_pending_token_to_zero_history_rank()

    assert list(manager.group_used_kv_tokens([seq])) == [4, 0, 0, 0, 0, 0, 0, 0]
    metadata = [
        prepare_decode_cpp(
            [seq], sp_rank, _SP_SIZE, _BLOCK_SIZE, _MAX_NUM_SEQS
        )
        for sp_rank in range(_SP_SIZE)
    ]

    # use_sp_a2a is a DP-iteration-wide decision: even ranks with no local
    # sequence work must enter the same collective cadence.
    assert all(meta.use_sp_a2a is True for meta in metadata)

    old_owner_meta = metadata[0]
    new_master_meta = metadata[1]
    assert list(old_owner_meta.input_ids) == []
    assert list(old_owner_meta.context_lens_for_attn) == [4]
    assert list(old_owner_meta.q_offsets[:3]) == [0, 0, 1]
    assert list(old_owner_meta.res_slice_get_to_buffer_input) == [0]
    assert list(old_owner_meta.res_slice_fill_to_buffer_input) == [
        _MAX_NUM_SEQS
    ]

    assert list(new_master_meta.input_ids) == [seq.last_token]
    assert list(new_master_meta.context_lens_for_attn) == [1]
    assert list(new_master_meta.q_slice_get) == [0]
