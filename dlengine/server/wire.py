"""Rust-owned ZMQ IPC protocol helpers."""

from __future__ import annotations

from enum import IntEnum

from dlengine._rust.proto import StepOut
from dlengine._rust.wrapper import export

_native = export(
    (
        "FreeSequences",
        "FreeVisionSlots",
        "Packet",
        "decode_free_sequences",
        "decode_free_vision_slots",
        "decode_packet",
        "encode_free_sequences",
        "encode_free_vision_slots",
        "encode_packet",
    )
)


class SequenceStatus(IntEnum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2
    TO_BE_MIGRATED = 3
    PREFILLING = 4

    @property
    def is_terminal(self) -> bool:
        return self == SequenceStatus.FINISHED


Packet = _native["Packet"]
FreeSequences = _native["FreeSequences"]
FreeVisionSlots = _native["FreeVisionSlots"]


def encode_packet(action: int, payload: bytes) -> bytes:
    return bytes(_native["encode_packet"](int(action), payload))


def decode_packet(data: bytes) -> tuple[int, bytes]:
    packet = _native["decode_packet"](data)
    return int(packet.action), bytes(packet.payload)


def encode_stepout(seq_id: int, token_ids: int | list[int], status: int) -> bytes:
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    return bytes(
        StepOut(
            int(seq_id), [int(token) for token in token_ids], int(status)
        ).to_bytes()
    )


def decode_stepout(data: bytes):
    return StepOut.from_bytes(data)


def encode_free_sequences(seq_ids: list[int], source_engine_id: str = "") -> bytes:
    return bytes(
        _native["encode_free_sequences"](
            [int(seq_id) for seq_id in seq_ids],
            source_engine_id or "",
        )
    )


def decode_free_sequences(data: bytes) -> tuple[list[int], str]:
    msg = _native["decode_free_sequences"](data)
    return [int(seq_id) for seq_id in msg.seq_ids], str(msg.source_engine_id)


def encode_free_vision_slots(
    encoder_engine_id: str,
    slot_indices: list[int],
    source_engine_id: str = "",
) -> bytes:
    return bytes(
        _native["encode_free_vision_slots"](
            str(encoder_engine_id),
            [int(idx) for idx in slot_indices],
            source_engine_id or "",
        )
    )


def decode_free_vision_slots(data: bytes) -> tuple[str, list[int], str]:
    msg = _native["decode_free_vision_slots"](data)
    return (
        str(msg.encoder_engine_id),
        [int(idx) for idx in msg.slot_indices],
        str(msg.source_engine_id),
    )


__all__ = [
    "FreeSequences",
    "FreeVisionSlots",
    "Packet",
    "SequenceStatus",
    "StepOut",
    "decode_free_sequences",
    "decode_free_vision_slots",
    "decode_packet",
    "decode_stepout",
    "encode_free_sequences",
    "encode_free_vision_slots",
    "encode_packet",
    "encode_stepout",
]
