"""Protocol helpers for the ``dlengine serve`` engine boundary.

The server/engine wire protocol is Rust-owned bincode.  ``Sequence`` remains an
internal compatibility object for the current scheduler, but it is no longer the
object exchanged between the HTTP server and engine process.
"""

from __future__ import annotations

from dlengine._cpp import (
    decode_migration_metadata as _decode_migration_metadata,
    encode_add_request as _encode_add_request,
)


def encode_add_request(
    seq_id: int,
    prompt_token_ids: list[int],
    sampling_params,
    affinity_key: int = 0,
    vision_slots: list[tuple[str, int, int, int, int]] | None = None,
) -> bytes:
    """Encode one server -> engine add request as Rust protocol bytes."""
    return bytes(
        _encode_add_request(
            int(seq_id),
            [int(t) for t in prompt_token_ids],
            sampling_params,
            int(affinity_key),
            vision_slots,
        )
    )


def decode_migration_metadata(data: bytes) -> tuple[int, int]:
    """Read migration seq_id and first generated token without materializing Sequence."""
    seq_id, first_token = _decode_migration_metadata(data)
    return int(seq_id), int(first_token)
