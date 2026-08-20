from collections.abc import Sequence as SequenceCollection
from typing import Any

from nanodeploy._cpp import BlockContextSlot


def is_full_sp_graph_batch_uniform(
    sequences: SequenceCollection[Any], sp_world_size: int
) -> bool:
    """Return whether every logical decode row participates on every SP rank.

    Fixed-full-SP CUDA Graphs capture the same attention batch size on every
    rank.  A rank-local control dummy (or any partially distributed sequence)
    violates that invariant: some ranks have a real attention row while other
    ranks would replay a zero-length padding row.  This predicate deliberately
    uses only globally shared sequence placement metadata, so every SP rank
    makes the same graph-versus-eager decision.
    """

    if sp_world_size <= 1:
        return True
    if not sequences:
        return False

    for sequence in sequences:
        dispatched = sequence.block_ctx(
            BlockContextSlot.ACTIVE
        ).num_dispatched_tokens
        if len(dispatched) != sp_world_size:
            return False
        if any(int(num_tokens) <= 0 for num_tokens in dispatched):
            return False

    return True
