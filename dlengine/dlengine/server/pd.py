"""PD (prefill-decode) disaggregation helpers for the ``dlengine serve`` path.

The single-process OpenAI server hands a fully-prefilled :class:`Sequence`
from a ``mode="prefill"`` engine to a ``mode="decode"`` engine over HTTP. The
sequence (prompt tokens, sampling params, and the MIGRATE :class:`BlockContext`
that points at the prefill engine's KV blocks) is serialized with the same C++
FlatBuffers codec used by the legacy ZMQ ``engine_server`` migration path
(``serialize`` / ``deserialize``), then base64-encoded so it can ride inside a
JSON ``kv_transfer_params`` field.

The decode engine deserializes it, adds it to the scheduler (decode mode routes
it to ``waiting_migration``), and the engine's existing migrate branch RDMA-pulls
the KV cache from the prefill engine. DLRouter never inspects the payload; it is
an opaque blob relayed from the prefill response to the decode request.
"""

from __future__ import annotations

import base64
import ctypes

from dlengine._cpp import (
    deserialize as _deserialize_cpp,
    Sequence,
    serialize as _serialize_cpp,
)

# 16MB matches the legacy ``_send_migration`` buffer; large enough for long
# prompt histories plus the MIGRATE block-context tables.
_MIGRATION_BUFFER_SIZE = 1024 * 1024 * 16


def serialize_seq(seq: Sequence) -> bytes:
    """Serialize a single (to-be-migrated) :class:`Sequence` to raw bytes."""
    buffer = ctypes.create_string_buffer(_MIGRATION_BUFFER_SIZE)
    ptr = ctypes.addressof(buffer)
    payload_size = _serialize_cpp(ptr, _MIGRATION_BUFFER_SIZE, [seq], False)
    return buffer.raw[:payload_size]


def deserialize_seqs(data: bytes) -> list[Sequence]:
    """Deserialize raw migration bytes back into :class:`Sequence` objects."""
    c_buffer = ctypes.create_string_buffer(data, len(data))
    ptr = ctypes.addressof(c_buffer)
    return _deserialize_cpp(ptr, len(data))


def encode_migration(seq: Sequence) -> str:
    """Serialize ``seq`` and base64-encode it for JSON transport."""
    return base64.b64encode(serialize_seq(seq)).decode("ascii")


def decode_migration(payload: str) -> Sequence:
    """Decode a base64 migration payload into a single :class:`Sequence`."""
    raw = base64.b64decode(payload)
    seqs = deserialize_seqs(raw)
    if not seqs:
        raise ValueError("migration payload did not contain a sequence")
    return seqs[0]
