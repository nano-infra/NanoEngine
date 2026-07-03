//! Packet encode/decode compatible with `dlengine.server.wire`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct ZmqPacket {
    pub action: u32,
    pub payload: Vec<u8>,
}

#[derive(Debug, Deserialize, Serialize)]
struct WirePacket {
    action: u32,
    payload: Vec<u8>,
}

impl ZmqPacket {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(&WirePacket {
            action: self.action,
            payload: self.payload.clone(),
        })
        .expect("serializing packet to JSON cannot fail")
    }

    pub fn decode(data: &[u8]) -> anyhow::Result<Self> {
        let packet: WirePacket = serde_json::from_slice(data)
            .map_err(|e| anyhow::anyhow!("Invalid dlengine wire packet: {}", e))?;
        Ok(Self {
            action: packet.action,
            payload: packet.payload,
        })
    }
}
