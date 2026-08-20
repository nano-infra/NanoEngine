from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class FixedSPGraphLayout:
    """Static index layout for one fixed-full-SP decode Graph bucket.

    Fixed-full-SP graphs execute a dense ``[source_master, local_slot]``
    attention batch.  Runtime metadata is packed and may have fewer rows after
    EOS, so graph replay needs stable indices for the dense transport layout.
    """

    master_bs: int
    attention_bs: int
    q_offsets: tuple[int, ...]
    q_slice_get: tuple[int, ...]
    q_slice_fill: tuple[int, ...]
    res_slice_get_to_buffer_output: tuple[int, ...]
    res_slice_fill_to_buffer_output: tuple[int, ...]
    res_slice_get_to_buffer_input: tuple[int, ...]
    res_slice_fill_to_buffer_input: tuple[int, ...]


def build_fixed_sp_graph_layout(
    *,
    sp_rank: int,
    sp_world_size: int,
    max_num_seqs: int,
    master_bs: int,
) -> FixedSPGraphLayout:
    """Build the dense Q/Res transport layout captured by a fixed SP graph."""

    if sp_world_size <= 1:
        raise ValueError("fixed SP graph layout requires sp_world_size > 1")
    if not 0 <= sp_rank < sp_world_size:
        raise ValueError(f"invalid SP rank {sp_rank} for size {sp_world_size}")
    if not 0 < master_bs <= max_num_seqs:
        raise ValueError(
            f"master_bs must be in [1, {max_num_seqs}], got {master_bs}"
        )

    local_slots = tuple(range(master_bs))
    q_slice_fill = tuple(
        sp_rank * master_bs + slot for slot in local_slots
    )
    remote_masters = tuple(
        master for master in range(sp_world_size) if master != sp_rank
    )

    return FixedSPGraphLayout(
        master_bs=master_bs,
        attention_bs=sp_world_size * master_bs,
        q_offsets=tuple(rank * master_bs for rank in range(sp_world_size + 1)),
        q_slice_get=local_slots,
        q_slice_fill=q_slice_fill,
        res_slice_get_to_buffer_output=q_slice_fill,
        res_slice_fill_to_buffer_output=tuple(
            sp_rank * max_num_seqs + slot for slot in local_slots
        ),
        res_slice_get_to_buffer_input=tuple(
            master * master_bs + slot
            for master in remote_masters
            for slot in local_slots
        ),
        res_slice_fill_to_buffer_input=tuple(
            master * max_num_seqs + slot
            for master in remote_masters
            for slot in local_slots
        ),
    )


def packed_attention_rows_to_dense(
    context_lens_flat: Sequence[int],
    *,
    sp_world_size: int,
    max_num_seqs: int,
    master_bs: int,
) -> tuple[int, ...]:
    """Map C++ packed attention rows into a fixed graph's dense row order."""

    expected = sp_world_size * max_num_seqs
    if len(context_lens_flat) != expected:
        raise ValueError(
            f"context lens has {len(context_lens_flat)} entries, expected {expected}"
        )

    dense_rows = []
    for master in range(sp_world_size):
        row_begin = master * max_num_seqs
        for slot in range(max_num_seqs):
            if int(context_lens_flat[row_begin + slot]) <= 0:
                continue
            if slot >= master_bs:
                raise ValueError(
                    "runtime attention row does not fit the selected graph bucket: "
                    f"master={master} slot={slot} master_bs={master_bs}"
                )
            dense_rows.append(master * master_bs + slot)
    return tuple(dense_rows)
