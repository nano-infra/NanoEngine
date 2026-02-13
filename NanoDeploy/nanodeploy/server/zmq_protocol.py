"""Binary packet format for ZMQ engine protocol (replaces protobuf StreamPacket).
Layout: seq_id(u64) | action(u32) | payload_len(u32) | payload[payload_len]
"""

import struct

HEADER_SIZE = 16  # 8 + 4 + 4


def encode_packet(seq_id: int, action: int, payload: bytes) -> bytes:
    return struct.pack("<QII", seq_id, action, len(payload)) + payload


def decode_packet(data: bytes) -> tuple[int, int, bytes]:
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Packet too short: {len(data)} bytes")
    seq_id, action, payload_len = struct.unpack("<QII", data[:HEADER_SIZE])
    if len(data) < HEADER_SIZE + payload_len:
        raise ValueError(
            f"Payload truncated: need {payload_len} bytes, "
            f"have {len(data) - HEADER_SIZE}"
        )
    payload = data[HEADER_SIZE : HEADER_SIZE + payload_len]
    return seq_id, action, payload
