//! Binary packet format for ZMQ engine protocol (replaces protobuf StreamPacket).
//! Layout: seq_id(u64) | action(u32) | payload_len(u32) | payload[payload_len]

use std::io::{Cursor, Read};

const HEADER_SIZE: usize = 16; // 8 + 4 + 4

#[derive(Debug, Clone)]
pub struct ZmqPacket {
    pub seq_id: u64,
    pub action: u32,
    pub payload: Vec<u8>,
}

impl ZmqPacket {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_SIZE + self.payload.len());
        buf.extend_from_slice(&self.seq_id.to_le_bytes());
        buf.extend_from_slice(&self.action.to_le_bytes());
        buf.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.payload);
        buf
    }

    pub fn decode(data: &[u8]) -> anyhow::Result<Self> {
        if data.len() < HEADER_SIZE {
            anyhow::bail!("Packet too short: {} bytes", data.len());
        }
        let mut cur = Cursor::new(data);
        let mut seq_id_buf = [0u8; 8];
        cur.read_exact(&mut seq_id_buf)?;
        let seq_id = u64::from_le_bytes(seq_id_buf);

        let mut action_buf = [0u8; 4];
        cur.read_exact(&mut action_buf)?;
        let action = u32::from_le_bytes(action_buf);

        let mut len_buf = [0u8; 4];
        cur.read_exact(&mut len_buf)?;
        let payload_len = u32::from_le_bytes(len_buf) as usize;

        if data.len() < HEADER_SIZE + payload_len {
            anyhow::bail!(
                "Payload truncated: need {} bytes, have {}",
                payload_len,
                data.len().saturating_sub(HEADER_SIZE)
            );
        }
        let mut payload = vec![0u8; payload_len];
        cur.read_exact(&mut payload)?;

        Ok(Self {
            seq_id,
            action,
            payload,
        })
    }
}
