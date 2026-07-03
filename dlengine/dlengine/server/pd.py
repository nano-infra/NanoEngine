"""Protocol helpers for the ``dlengine serve`` engine boundary.

The server/engine wire protocol is Rust-owned bincode.  ``Sequence`` remains an
internal compatibility object for the current scheduler, but it is no longer the
object exchanged between the HTTP server and engine process.
"""

from __future__ import annotations

import base64

from dlengine._cpp import (
    decode_add_requests as _decode_add_requests,
    decode_migration_metadata as _decode_migration_metadata,
    decode_migration_request as _decode_migration_request,
    encode_add_request as _encode_add_request,
    Sequence,
)


def encode_add_request(
    seq_id: int,
    prompt_token_ids: list[int],
    sampling_params,
    affinity_key: int = 0,
) -> bytes:
    """Encode one server -> engine add request as Rust protocol bytes."""
    return bytes(
        _encode_add_request(
            int(seq_id),
            [int(t) for t in prompt_token_ids],
            sampling_params,
            int(affinity_key),
        )
    )


def decode_add_requests(data: bytes) -> list[Sequence]:
    """Decode Rust add request bytes into scheduler compatibility sequences."""
    return _decode_add_requests(data)


def decode_migration(payload: str) -> Sequence:
    """Decode a base64 migration payload into a scheduler compatibility object."""
    return decode_migration_bytes(base64.b64decode(payload))


def decode_migration_bytes(data: bytes) -> Sequence:
    """Decode Rust migration protocol bytes into a scheduler compatibility object."""
    return _decode_migration_request(data)


def decode_migration_metadata(data: bytes) -> tuple[int, int]:
    """Read migration seq_id and first generated token without materializing Sequence."""
    seq_id, first_token = _decode_migration_metadata(data)
    return int(seq_id), int(first_token)
