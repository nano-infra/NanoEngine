from __future__ import annotations

from dataclasses import dataclass

from nanodeploy._cpp import BlockContextSlot, Scheduler, Sequence, prepare_decode_cpp
from nanodeploy.worker.sp_graph_policy import (
    build_fixed_sp_graph_layout,
    packed_attention_rows_to_dense,
)


_SP_SIZE = 8
_MAX_NUM_SEQS = 8
_BLOCK_SIZE = 64
_PROMPT_LENGTHS = (24, 31, 38, 45, 52, 59, 66, 73)


@dataclass(frozen=True)
class _DecodeBatch:
    scheduler: Scheduler
    sequences: tuple[Sequence, ...]
    scheduled: tuple[Sequence, ...]


def _make_decode_scheduler() -> Scheduler:
    # Mirrors examples/pd_disagg_deepseek_v3_parallel.py's decode topology:
    # DP1/SP8, fixed_sp_size=8, max_num_seqs=8, and one inner decode step.
    return Scheduler(
        "pd-decode-sp8-semantics",
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
        512,
        "legacy",
        False,
        "",
        False,
        "RoundRobin",
        _SP_SIZE,
    )


def _schedule_fixed_sp_batch(num_requests: int) -> _DecodeBatch:
    scheduler = _make_decode_scheduler()
    sequences = []
    for request_idx, prompt_len in enumerate(_PROMPT_LENGTHS[:num_requests]):
        token_base = 10_000 * (request_idx + 1)
        sequence = Sequence(
            [token_base + token_idx for token_idx in range(prompt_len)],
            1e-5,
            128,
            True,
        )
        sequence.seq_id = 100 + request_idx
        scheduler.add(sequence)
        sequences.append(sequence)

    migration = scheduler.schedule()
    assert migration.is_prefill
    assert list(migration.dp_seqs[0]) == sequences

    decode = scheduler.schedule()
    assert not decode.is_prefill
    return _DecodeBatch(
        scheduler=scheduler,
        sequences=tuple(sequences),
        scheduled=tuple(decode.dp_seqs[0]),
    )


def _metadata_by_rank(batch: _DecodeBatch):
    return [
        prepare_decode_cpp(
            list(batch.scheduled),
            sp_rank,
            _SP_SIZE,
            _BLOCK_SIZE,
            _MAX_NUM_SEQS,
        )
        for sp_rank in range(_SP_SIZE)
    ]


def _matrix(flat_values) -> list[list[int]]:
    values = list(flat_values)
    assert len(values) == _SP_SIZE * _MAX_NUM_SEQS
    return [
        values[row * _MAX_NUM_SEQS : (row + 1) * _MAX_NUM_SEQS]
        for row in range(_SP_SIZE)
    ]


def _binary_mask(matrix: list[list[int]], local_rank: int) -> list[list[int]]:
    return [
        [0 if row_idx == local_rank else int(value != 0) for value in row]
        for row_idx, row in enumerate(matrix)
    ]


def _real_sequences_by_master(batch: _DecodeBatch) -> dict[int, Sequence]:
    result = {}
    for sequence in batch.scheduled:
        if batch.scheduler.worker_state[0].is_control_dummy(sequence):
            continue
        master = sequence.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx
        assert master not in result
        result[master] = sequence
    return result


def _remote_edge_count(batch: _DecodeBatch) -> tuple[int, int]:
    q_edges = 0
    response_edges = 0
    for sp_rank, metadata in enumerate(_metadata_by_rank(batch)):
        q_mask = _binary_mask(_matrix(metadata.global_context_lens_flat), sp_rank)
        response_mask = _binary_mask(_matrix(metadata.context_lens_flat), sp_rank)
        q_edges += sum(sum(row) for row in q_mask)
        response_edges += sum(sum(row) for row in response_mask)
    return q_edges, response_edges


def test_fixed_sp8_batch_changes_fan_in_into_full_all_to_all():
    """Document the semantic delta between the known-good bs=1 and bs=8 runs.

    One real request has one real master: Q fans out over seven links and the
    seven remote partial results fan back into that master.  Eight requests are
    assigned to all eight masters by RoundRobin, so both payloads become a
    complete directed all-to-all with 8 * 7 active remote edges.
    """

    one_request = _schedule_fixed_sp_batch(1)
    eight_requests = _schedule_fixed_sp_batch(8)

    assert set(_real_sequences_by_master(one_request)) == {0}
    assert set(_real_sequences_by_master(eight_requests)) == set(range(_SP_SIZE))
    assert _remote_edge_count(one_request) == (7, 7)
    assert _remote_edge_count(eight_requests) == (56, 56)


def test_fixed_sp8_batch_metadata_preserves_master_local_slot_identity():
    """Validate the exact metadata consumed by Q/Res/LSE DLSlime calls.

    The current eight-prompt script assigns one real request to every SP
    master.  Every prompt is longer than eight tokens, so fixed SP8 gives each
    request KV on every rank.  Consequently each worker must compute the same
    eight logical requests in master-rank order while retaining its own one-row
    sampling batch.
    """

    batch = _schedule_fixed_sp_batch(8)
    by_master = _real_sequences_by_master(batch)
    metadata_by_rank = _metadata_by_rank(batch)

    for sp_rank, metadata in enumerate(metadata_by_rank):
        mastered_sequence = by_master[sp_rank]
        context_lens = _matrix(metadata.context_lens_flat)
        global_context_lens = _matrix(metadata.global_context_lens_flat)

        assert list(metadata.input_ids) == [mastered_sequence.last_token]
        assert list(metadata.positions) == [mastered_sequence.num_tokens - 1]
        assert len(metadata.slot_mapping) == 1
        assert metadata.use_sp_a2a

        # On this worker, attention rows are packed by source master.  With one
        # request per master and all ranks participating, offsets are 0..8.
        assert list(metadata.q_offsets) == list(range(_SP_SIZE + 1))
        assert list(metadata.q_slice_get) == [0]
        assert list(metadata.q_slice_fill) == [sp_rank]
        assert list(metadata.q_copy_mask) == [1]

        expected_local_contexts = [
            by_master[master]
            .block_ctx(BlockContextSlot.ACTIVE)
            .num_dispatched_tokens[sp_rank]
            for master in range(_SP_SIZE)
        ]
        assert list(metadata.context_lens_for_attn) == expected_local_contexts
        for master in range(_SP_SIZE):
            assert context_lens[master] == [expected_local_contexts[master]] + [
                0
            ] * (_MAX_NUM_SEQS - 1)

        dispatched = list(
            mastered_sequence.block_ctx(
                BlockContextSlot.ACTIVE
            ).num_dispatched_tokens
        )
        for participant in range(_SP_SIZE):
            assert global_context_lens[participant] == [dispatched[participant]] + [
                0
            ] * (_MAX_NUM_SEQS - 1)

        expected_remote_attention_rows = [
            master for master in range(_SP_SIZE) if master != sp_rank
        ]
        expected_remote_buffer_slots = [
            master * _MAX_NUM_SEQS
            for master in range(_SP_SIZE)
            if master != sp_rank
        ]

        # The local partial result is patched into source-rank row `sp_rank`.
        assert list(metadata.res_slice_get_to_buffer_output) == [sp_rank]
        assert list(metadata.res_slice_fill_to_buffer_output) == [
            sp_rank * _MAX_NUM_SEQS
        ]
        assert list(metadata.res_to_buffer_output_mask) == [1]

        # Remote partials are written into destination-master-major segments
        # before the transpose all-to-all.
        assert list(metadata.res_slice_get_to_buffer_input) == (
            expected_remote_attention_rows
        )
        assert list(metadata.res_slice_fill_to_buffer_input) == (
            expected_remote_buffer_slots
        )
        assert list(metadata.res_to_buffer_input_mask) == [1] * (_SP_SIZE - 1)

        q_mask = _binary_mask(global_context_lens, sp_rank)
        response_mask = _binary_mask(context_lens, sp_rank)
        for target_rank in range(_SP_SIZE):
            expected = 0 if target_rank == sp_rank else 1
            assert q_mask[target_rank] == [expected] + [0] * (
                _MAX_NUM_SEQS - 1
            )
            assert response_mask[target_rank] == [expected] + [0] * (
                _MAX_NUM_SEQS - 1
            )

        # This is the behavior absent from the known-good one-request case:
        # the same master-local slot (column zero) targets seven different
        # destination masters in the transpose Res/LSE collective.
        assert [
            sum(response_mask[target][slot] for target in range(_SP_SIZE))
            for slot in range(_MAX_NUM_SEQS)
        ] == [_SP_SIZE - 1] + [0] * (_MAX_NUM_SEQS - 1)


def test_fixed_sp8_batch_metadata_roundtrips_distinct_request_identities():
    """Run a CPU semantic oracle over the metadata's Q and response indices.

    Tags stand in for tensors.  This catches slot aliasing that numerical tests
    with identical or rank-only values can miss: every request and every
    participant contribution has a distinct identity.
    """

    batch = _schedule_fixed_sp_batch(8)
    by_master = _real_sequences_by_master(batch)
    metadata_by_rank = _metadata_by_rank(batch)

    # Q: every participant must receive one query from every master, packed in
    # exactly the same order as context_lens_for_attn/block_tables.
    for participant, metadata in enumerate(metadata_by_rank):
        received_queries: list[int | None] = [None] * len(
            metadata.context_lens_for_attn
        )
        for master in range(_SP_SIZE):
            source_metadata = metadata_by_rank[master]
            source_global_lens = _matrix(
                source_metadata.global_context_lens_flat
            )
            source_q_mask = _binary_mask(source_global_lens, master)
            if participant == master:
                destination_row = list(source_metadata.q_slice_fill)[0]
            else:
                assert source_q_mask[participant][0] == 1
                destination_row = list(metadata.q_offsets)[master]

            assert received_queries[destination_row] is None
            received_queries[destination_row] = by_master[master].seq_id

        assert received_queries == [
            by_master[master].seq_id for master in range(_SP_SIZE)
        ]

    # Res/LSE: a participant computes one partial per attention row.  Local
    # master results use the DLSlime local buffer; remote master results use the
    # destination-major transpose input.  Both must land at [participant, slot]
    # on the owning master without cross-request aliasing.
    combined_by_master: dict[int, list[list[tuple[int, int] | None]]] = {
        master: [
            [None for _ in range(_MAX_NUM_SEQS)]
            for _ in range(_SP_SIZE)
        ]
        for master in range(_SP_SIZE)
    }

    for participant, metadata in enumerate(metadata_by_rank):
        attention_sequence_ids = [
            by_master[master].seq_id for master in range(_SP_SIZE)
        ]

        for source_row, local_buffer_index in zip(
            metadata.res_slice_get_to_buffer_output,
            metadata.res_slice_fill_to_buffer_output,
            strict=True,
        ):
            master, slot = divmod(local_buffer_index, _MAX_NUM_SEQS)
            assert master == participant
            combined_by_master[master][participant][slot] = (
                participant,
                attention_sequence_ids[source_row],
            )

        for source_row, transpose_input_index in zip(
            metadata.res_slice_get_to_buffer_input,
            metadata.res_slice_fill_to_buffer_input,
            strict=True,
        ):
            master, slot = divmod(transpose_input_index, _MAX_NUM_SEQS)
            assert master != participant
            assert combined_by_master[master][participant][slot] is None
            combined_by_master[master][participant][slot] = (
                participant,
                attention_sequence_ids[source_row],
            )

    for master in range(_SP_SIZE):
        expected_seq_id = by_master[master].seq_id
        for participant in range(_SP_SIZE):
            assert combined_by_master[master][participant][0] == (
                participant,
                expected_seq_id,
            )
            assert combined_by_master[master][participant][1:] == [None] * (
                _MAX_NUM_SEQS - 1
            )


def test_fixed_sp8_graph_layout_stays_dense_after_eos():
    """Graph padding must preserve dense identities after a request exits.

    Request 0 is replaced by a rank-0-only control dummy.  Ranks 1..7 then
    receive seven packed C++ attention rows, but the fixed graph must keep the
    same eight-row master-major layout.  The absent row gets an explicit Q and
    a legal throw-away attention row; its Res/LSE mask remains zero so it
    cannot enter request 0's reduction.
    """

    batch = _schedule_fixed_sp_batch(8)
    remaining = [
        sequence
        for sequence in batch.scheduled
        if sequence.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx != 0
    ]
    rank_zero_dummy = batch.scheduler.worker_state[0].dummy_seqs[0]
    after_eos = tuple(remaining + [rank_zero_dummy])

    for sp_rank in range(_SP_SIZE):
        metadata = prepare_decode_cpp(
            list(after_eos),
            sp_rank,
            _SP_SIZE,
            _BLOCK_SIZE,
            _MAX_NUM_SEQS,
        )
        layout = build_fixed_sp_graph_layout(
            sp_rank=sp_rank,
            sp_world_size=_SP_SIZE,
            max_num_seqs=_MAX_NUM_SEQS,
            master_bs=1,
        )
        packed_to_dense = packed_attention_rows_to_dense(
            metadata.context_lens_flat,
            sp_world_size=_SP_SIZE,
            max_num_seqs=_MAX_NUM_SEQS,
            master_bs=1,
        )

        assert layout.attention_bs == _SP_SIZE
        assert layout.q_offsets == tuple(range(_SP_SIZE + 1))
        assert layout.q_slice_get == (0,)
        assert layout.q_slice_fill == (sp_rank,)
        assert layout.res_slice_get_to_buffer_output == (sp_rank,)
        assert layout.res_slice_fill_to_buffer_output == (
            sp_rank * _MAX_NUM_SEQS,
        )

        if sp_rank == 0:
            assert len(metadata.context_lens_for_attn) == _SP_SIZE
            assert packed_to_dense == tuple(range(_SP_SIZE))
        else:
            assert len(metadata.context_lens_for_attn) == _SP_SIZE - 1
            assert packed_to_dense == tuple(range(1, _SP_SIZE))

        dense_context_lens = [1] * layout.attention_bs
        for packed_row, dense_row in enumerate(packed_to_dense):
            dense_context_lens[dense_row] = metadata.context_lens_for_attn[
                packed_row
            ]
        assert all(context_len > 0 for context_len in dense_context_lens)

        # Transport every dense Q so no persistent receive-buffer row is
        # stale.  Only real local KV shards return Res/LSE contributions.
        q_mask = [
            0 if destination == sp_rank else 1
            for destination in range(_SP_SIZE)
        ]
        assert sum(q_mask) == _SP_SIZE - 1

        response_mask = [
            int(_matrix(metadata.context_lens_flat)[master][0] > 0)
            for master in range(_SP_SIZE)
        ]
        if sp_rank == 0:
            assert response_mask == [1] * _SP_SIZE
        else:
            assert response_mask == [0] + [1] * (_SP_SIZE - 1)
