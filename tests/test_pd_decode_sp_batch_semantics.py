from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from nanodeploy._cpp import BlockContextSlot, Scheduler, Sequence, prepare_decode_cpp
from nanodeploy.kernels.sp_graph_metadata import update_sp_graph_metadata
from tests.sp_routing_oracle import (
    assert_destination_rows_match_packed_receivers,
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
            if participant == master:
                destination_row = list(source_metadata.q_slice_fill)[0]
            else:
                destination_row = list(
                    source_metadata.q_dst_row_indices_flat
                )[participant * _MAX_NUM_SEQS]
                assert destination_row >= 0

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fixed_sp8_packed_graph_padding_after_eos():
    """Fixed SP keeps packed real rows and pads only Graph-only tails."""

    batch = _schedule_fixed_sp_batch(8)
    assert_destination_rows_match_packed_receivers(
        batch.scheduled,
        sp_size=_SP_SIZE,
        max_num_seqs=_MAX_NUM_SEQS,
        block_size=_BLOCK_SIZE,
    )
    remaining = [
        sequence
        for sequence in batch.scheduled
        if sequence.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx != 0
    ]
    rank_zero_dummy = batch.scheduler.worker_state[0].dummy_seqs[0]
    after_eos = tuple(remaining + [rank_zero_dummy])
    assert_destination_rows_match_packed_receivers(
        after_eos,
        sp_size=_SP_SIZE,
        max_num_seqs=_MAX_NUM_SEQS,
        block_size=_BLOCK_SIZE,
    )

    device = torch.device("cuda")
    for sp_rank in range(_SP_SIZE):
        metadata = prepare_decode_cpp(
            list(after_eos),
            sp_rank,
            _SP_SIZE,
            _BLOCK_SIZE,
            _MAX_NUM_SEQS,
        )
        actual_attn_bs = len(metadata.context_lens_for_attn)
        assert actual_attn_bs == (_SP_SIZE if sp_rank == 0 else _SP_SIZE - 1)

        actual_block_tables = torch.tensor(
            metadata.block_tables_flat,
            dtype=torch.int32,
            device=device,
        ).reshape(actual_attn_bs, metadata.max_num_blocks)
        # Rank 0 also covers piecewise attention, where eager attention uses
        # the exact packed row count and only the master batch is padded.
        graph_attn_bs = (
            actual_attn_bs if sp_rank == 0 else 2 * _SP_SIZE
        )
        graph_master_bs = 2
        graph_vars = {
            "context_lens": torch.tensor(
                metadata.context_lens_flat,
                dtype=torch.int32,
                device=device,
            ).reshape(_SP_SIZE, _MAX_NUM_SEQS),
            "global_context_lens": torch.tensor(
                metadata.global_context_lens_flat,
                dtype=torch.int32,
                device=device,
            ).reshape(_SP_SIZE, _MAX_NUM_SEQS),
            "q_dst_row_indices": torch.full(
                (_SP_SIZE, _MAX_NUM_SEQS),
                -2,
                dtype=torch.int32,
                device=device,
            ),
            "actual_attn_bs": torch.zeros(
                (), dtype=torch.int32, device=device
            ),
            "context_lens_for_attn": torch.zeros(
                graph_attn_bs,
                dtype=torch.int32,
                device=device,
            ),
            "block_tables": torch.full(
                (graph_attn_bs, metadata.max_num_blocks),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            "res_slice_get_to_buffer_output": torch.full(
                (graph_master_bs,), -1, dtype=torch.int32, device=device
            ),
            "res_slice_fill_to_buffer_output": torch.full(
                (graph_master_bs,), -1, dtype=torch.int32, device=device
            ),
            "res_to_buffer_output_mask": torch.zeros(
                graph_master_bs, dtype=torch.int32, device=device
            ),
        }
        graph_vars["context_lens_for_attn"][:actual_attn_bs].copy_(
            torch.tensor(
                metadata.context_lens_for_attn,
                dtype=torch.int32,
                device=device,
            )
        )
        graph_vars["block_tables"][:actual_attn_bs].copy_(actual_block_tables)
        local_result_rows = len(metadata.res_slice_get_to_buffer_output)
        graph_vars["res_slice_get_to_buffer_output"][:local_result_rows].copy_(
            torch.tensor(
                metadata.res_slice_get_to_buffer_output,
                dtype=torch.int32,
                device=device,
            )
        )
        graph_vars["res_slice_fill_to_buffer_output"][:local_result_rows].copy_(
            torch.tensor(
                metadata.res_slice_fill_to_buffer_output,
                dtype=torch.int32,
                device=device,
            )
        )
        graph_vars["res_to_buffer_output_mask"][:local_result_rows].copy_(
            torch.tensor(
                metadata.res_to_buffer_output_mask,
                dtype=torch.int32,
                device=device,
            )
        )
        packed_contexts = graph_vars["context_lens_for_attn"][
            :actual_attn_bs
        ].clone()
        packed_blocks = graph_vars["block_tables"][:actual_attn_bs].clone()

        q_dst_source = torch.tensor(
            metadata.q_dst_row_indices_flat,
            dtype=torch.int32,
            device=device,
        ).reshape(_SP_SIZE, _MAX_NUM_SEQS)
        update_sp_graph_metadata(
            q_dst_source,
            graph_vars["q_dst_row_indices"],
            graph_vars["actual_attn_bs"],
            actual_block_tables,
            graph_vars["context_lens"],
            graph_vars["global_context_lens"],
            graph_vars["context_lens_for_attn"],
            graph_vars["block_tables"],
            graph_vars["res_slice_get_to_buffer_output"],
            graph_vars["res_slice_fill_to_buffer_output"],
            graph_vars["res_to_buffer_output_mask"],
            actual_attn_bs,
            graph_attn_bs,
            1,
            graph_master_bs,
            sp_rank,
            _MAX_NUM_SEQS,
        )
        torch.cuda.synchronize()

        assert torch.equal(graph_vars["q_dst_row_indices"], q_dst_source)
        assert graph_vars["actual_attn_bs"].item() == actual_attn_bs
        torch.testing.assert_close(
            graph_vars["context_lens_for_attn"][:actual_attn_bs],
            packed_contexts,
        )
        torch.testing.assert_close(
            graph_vars["block_tables"][:actual_attn_bs],
            packed_blocks,
        )
        assert torch.equal(
            graph_vars["context_lens_for_attn"][actual_attn_bs:graph_attn_bs],
            torch.ones(
                graph_attn_bs - actual_attn_bs,
                dtype=torch.int32,
                device=device,
            ),
        )
        assert torch.equal(
            graph_vars["block_tables"][actual_attn_bs:graph_attn_bs],
            actual_block_tables[0:1].expand(graph_attn_bs - actual_attn_bs, -1),
        )
        assert graph_vars["context_lens"][sp_rank, 1].item() == 1
        assert graph_vars["global_context_lens"][sp_rank, 1].item() == 1
        assert graph_vars["res_slice_get_to_buffer_output"].tolist() == [
            metadata.res_slice_get_to_buffer_output[0],
            actual_attn_bs if actual_attn_bs < graph_attn_bs else 0,
        ]
        assert graph_vars["res_slice_fill_to_buffer_output"].tolist() == [
            metadata.res_slice_fill_to_buffer_output[0],
            sp_rank * _MAX_NUM_SEQS + 1,
        ]
        assert graph_vars["res_to_buffer_output_mask"].tolist() == [1, 1]
