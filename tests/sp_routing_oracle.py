from __future__ import annotations

from collections.abc import Sequence as SequenceCollection

from nanodeploy._cpp import BlockContextSlot, Sequence, prepare_decode_cpp


def assert_destination_rows_match_packed_receivers(
    scheduled: SequenceCollection[Sequence],
    *,
    sp_size: int,
    max_num_seqs: int,
    block_size: int,
) -> None:
    """Assert every real Q edge writes exactly one receiver-local packed row."""

    sequences_by_master: list[list[Sequence]] = [[] for _ in range(sp_size)]
    for sequence in scheduled:
        master = sequence.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx
        sequences_by_master[master].append(sequence)

    metadata_by_sender = [
        prepare_decode_cpp(
            list(scheduled),
            sender,
            sp_size,
            block_size,
            max_num_seqs,
        )
        for sender in range(sp_size)
    ]

    for sender, metadata in enumerate(metadata_by_sender):
        rows = list(metadata.q_dst_row_indices_flat)
        assert len(rows) == sp_size * max_num_seqs
        for receiver in range(sp_size):
            packed_row = 0
            for source, source_sequences in enumerate(sequences_by_master):
                for local_idx, sequence in enumerate(source_sequences):
                    participates = (
                        sequence.block_ctx(
                            BlockContextSlot.ACTIVE
                        ).num_dispatched_tokens[receiver]
                        > 0
                    )
                    if source == sender:
                        actual = rows[receiver * max_num_seqs + local_idx]
                        expected = (
                            packed_row
                            if participates and receiver != sender
                            else -1
                        )
                        assert actual == expected
                    if participates:
                        packed_row += 1

    for receiver, metadata in enumerate(metadata_by_sender):
        expected_rows: list[tuple[int, int]] = []
        for source, source_sequences in enumerate(sequences_by_master):
            for local_idx, sequence in enumerate(source_sequences):
                if (
                    sequence.block_ctx(
                        BlockContextSlot.ACTIVE
                    ).num_dispatched_tokens[receiver]
                    > 0
                ):
                    expected_rows.append((source, local_idx))

        actual_rows: dict[int, tuple[int, int]] = {}
        for local_idx, packed_row in zip(
            metadata.q_slice_get,
            metadata.q_slice_fill,
            strict=True,
        ):
            assert packed_row not in actual_rows
            actual_rows[packed_row] = (receiver, local_idx)

        for sender, sender_metadata in enumerate(metadata_by_sender):
            if sender == receiver:
                continue
            rows = list(sender_metadata.q_dst_row_indices_flat)
            for local_idx in range(len(sequences_by_master[sender])):
                packed_row = rows[receiver * max_num_seqs + local_idx]
                if packed_row < 0:
                    continue
                assert packed_row not in actual_rows
                actual_rows[packed_row] = (sender, local_idx)

        assert sorted(actual_rows) == list(range(len(expected_rows)))
        assert [actual_rows[row] for row in range(len(expected_rows))] == expected_rows
