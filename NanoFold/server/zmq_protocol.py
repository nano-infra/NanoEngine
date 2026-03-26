"""ZMQ packet encode/decode for NanoFold.

FlatBuffers schema (NanoSequence/proto/packet.fbs):
  enum Action : byte { ..., FoldRequest = 7, FoldResponse = 8 }
  table ZmqPacket { action: Action; payload: [ubyte]; }

Self-contained: depends only on the `flatbuffers` package (no nanodeploy import).
"""

from __future__ import annotations

import flatbuffers
import flatbuffers.encode
import flatbuffers.number_types
import flatbuffers.packer
import flatbuffers.table

ACTION_FOLD_REQUEST: int = 7
ACTION_FOLD_RESPONSE: int = 8


def encode_packet(action: int, payload: bytes) -> bytes:
    """Encode action + payload bytes into a FlatBuffers ZmqPacket buffer."""
    builder = flatbuffers.Builder(64 + len(payload))
    payload_vec = builder.CreateByteVector(payload)
    # ZmqPacketStart: 2 fields  (slot 0 = action Int8, slot 1 = payload vector)
    builder.StartObject(2)
    builder.PrependInt8Slot(0, action, 0)
    builder.PrependUOffsetTRelativeSlot(
        1, flatbuffers.number_types.UOffsetTFlags.py_type(payload_vec), 0
    )
    pkt = builder.EndObject()
    builder.Finish(pkt)
    return bytes(builder.Output())


def decode_packet(data: bytes) -> tuple[int, bytes]:
    """Decode a FlatBuffers ZmqPacket buffer, returning (action, payload)."""
    n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, data, 0)
    tab = flatbuffers.table.Table(data, n)

    # field 0 (action): vtable slot 0 → byte offset 4 in vtable
    action: int = 0
    o = flatbuffers.number_types.UOffsetTFlags.py_type(tab.Offset(4))
    if o != 0:
        action = tab.Get(flatbuffers.number_types.Int8Flags, o + tab.Pos)

    # field 1 (payload): vtable slot 1 → byte offset 6 in vtable
    payload = b""
    o = flatbuffers.number_types.UOffsetTFlags.py_type(tab.Offset(6))
    if o != 0:
        vec_len = tab.VectorLen(o)
        if vec_len > 0:
            payload = bytes(
                tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
            )

    return action, payload
