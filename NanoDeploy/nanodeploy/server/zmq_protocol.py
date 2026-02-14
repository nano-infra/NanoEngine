"""ZMQ packet encode/decode using FlatBuffers (schema: NanoSequence/proto/packet.fbs).
The entire ZMQ message is a single FlatBuffers buffer containing a ZmqPacket table.
"""

import flatbuffers

from nanodeploy.fbs.ZmqPacket import (
    ZmqPacket,
    ZmqPacketAddAction,
    ZmqPacketAddPayload,
    ZmqPacketEnd,
    ZmqPacketStart,
)


def encode_packet(action: int, payload: bytes) -> bytes:
    builder = flatbuffers.Builder(64 + len(payload))
    payload_vec = builder.CreateByteVector(payload)

    ZmqPacketStart(builder)
    ZmqPacketAddAction(builder, action)
    ZmqPacketAddPayload(builder, payload_vec)
    packet = ZmqPacketEnd(builder)

    builder.Finish(packet)
    return bytes(builder.Output())


def decode_packet(data: bytes) -> tuple[int, bytes]:
    packet = ZmqPacket.GetRootAs(data, 0)
    action = packet.Action()

    payload_len = packet.PayloadLength()
    if payload_len > 0:
        payload = bytes(packet.PayloadAsNumpy())
    else:
        payload = b""

    return action, payload
