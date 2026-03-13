//! ZMQ packet encode/decode using FlatBuffers (schema: NanoSequence/proto/packet.fbs).
//! The entire ZMQ message is a single FlatBuffers buffer containing a ZmqPacket table.

use crate::fbs::{Action, ZmqPacket as FbsZmqPacket, ZmqPacketArgs};
use flatbuffers::FlatBufferBuilder;

/// Convenience wrapper around the FlatBuffers ZmqPacket.
#[derive(Debug, Clone)]
pub struct ZmqPacket {
    pub action: u32,
    pub payload: Vec<u8>,
}

impl ZmqPacket {
    /// Encode into a FlatBuffers byte buffer.
    pub fn encode(&self) -> Vec<u8> {
        let mut builder = FlatBufferBuilder::with_capacity(64 + self.payload.len());
        let payload_vec = builder.create_vector(&self.payload);

        let fbs_action = Action::ENUM_VALUES
            .get(self.action as usize)
            .copied()
            .unwrap_or(Action::StepOut);

        let packet = FbsZmqPacket::create(
            &mut builder,
            &ZmqPacketArgs {
                action: fbs_action,
                payload: Some(payload_vec),
            },
        );

        builder.finish(packet, None);
        builder.finished_data().to_vec()
    }

    /// Decode from a FlatBuffers byte buffer.
    pub fn decode(data: &[u8]) -> anyhow::Result<Self> {
        let packet = flatbuffers::root::<FbsZmqPacket>(data)
            .map_err(|e| anyhow::anyhow!("Invalid FlatBuffers ZmqPacket: {}", e))?;

        let action = packet.action().0 as u32;
        let payload = packet
            .payload()
            .map(|v| v.bytes().to_vec())
            .unwrap_or_default();

        Ok(Self { action, payload })
    }
}
