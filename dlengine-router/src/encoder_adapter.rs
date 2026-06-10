//! ZMQ DEALER adapter for EncoderEngine (request-response, JSON over ZmqPacket).
//!
//! Protocol:
//! - Request:  action=5 (EncodeRequest),  payload = JSON `{"messages": [...]}`
//! - Response: action=6 (EncodeResponse), payload = JSON `{"input_ids": [...], "vision_slots": [...]}`

use crate::zmq_packet::ZmqPacket;
use serde::{Deserialize, Serialize};
use std::sync::mpsc;
use std::thread;
use tokio::sync::mpsc as tokio_mpsc;
use tracing::{info, warn};

const ACTION_ENCODE_REQUEST: u32 = 5;
const ACTION_ENCODE_RESPONSE: u32 = 6;

pub struct EncoderAdapter {
    pub uuid: Option<String>,
    request_tx: Option<mpsc::SyncSender<ZmqPacket>>,
    response_rx: Option<tokio_mpsc::UnboundedReceiver<ZmqPacket>>,
    /// Keep the sender alive so the channel doesn't close prematurely.
    _response_tx_keepalive: Option<tokio_mpsc::UnboundedSender<ZmqPacket>>,
    io_thread_handle: Option<thread::JoinHandle<()>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionSlotInfo {
    pub encoder_engine_id: String,
    pub slot_idx: u32,
    pub num_tokens: u32,
    pub hidden_size: u32,
    pub max_tokens_per_slot: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeResponse {
    pub input_ids: Vec<i32>,
    #[serde(default)]
    pub vision_slots: Vec<VisionSlotInfo>,
    #[serde(default)]
    pub error: Option<String>,
}

impl EncoderAdapter {
    pub fn new(_id: String) -> Self {
        Self {
            uuid: None,
            request_tx: None,
            response_rx: None,
            _response_tx_keepalive: None,
            io_thread_handle: None,
        }
    }

    pub async fn connect(&mut self, addr: &str) -> anyhow::Result<()> {
        let endpoint = format!("tcp://{}", addr);
        info!("Connecting to Encoder Engine at {}", endpoint);

        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::DEALER)?;

        socket.set_linger(0)?;
        socket.set_sndtimeo(5000)?;
        socket.set_rcvtimeo(200)?;
        socket.set_reconnect_ivl(100)?;

        socket.connect(&endpoint)?;
        info!("Encoder ZMQ socket connected to {}", endpoint);

        let (send_tx, send_rx) = mpsc::sync_channel::<ZmqPacket>(64);
        self.request_tx = Some(send_tx);

        let (recv_tx, recv_rx) = tokio_mpsc::unbounded_channel::<ZmqPacket>();
        self._response_tx_keepalive = Some(recv_tx.clone());
        self.response_rx = Some(recv_rx);

        let addr_for_log = addr.to_string();
        let io_handle = thread::spawn(move || {
            loop {
                // Try recv
                match socket.recv_bytes(0) {
                    Ok(data) => {
                        if let Ok(packet) = ZmqPacket::decode(&data) {
                            if recv_tx.send(packet).is_err() {
                                info!("Encoder I/O thread: channel closed for {}", addr_for_log);
                                break;
                            }
                        }
                    }
                    Err(zmq::Error::EAGAIN) => {}
                    Err(zmq::Error::ETERM) => {
                        info!(
                            "Encoder I/O thread: context terminated for {}",
                            addr_for_log
                        );
                        break;
                    }
                    Err(e) => {
                        warn!("Encoder ZMQ recv error for {}: {}", addr_for_log, e);
                        break;
                    }
                }

                // Drain send channel
                loop {
                    match send_rx.try_recv() {
                        Ok(packet) => {
                            let data = packet.encode();
                            if let Err(e) = socket.send(&data, 0) {
                                warn!("Encoder ZMQ send error for {}: {}", addr_for_log, e);
                                break;
                            }
                        }
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(mpsc::TryRecvError::Disconnected) => {
                            info!(
                                "Encoder I/O thread: send channel closed for {}",
                                addr_for_log
                            );
                            return;
                        }
                    }
                }
            }
            info!("Encoder I/O thread ended for {}", addr_for_log);
        });

        self.io_thread_handle = Some(io_handle);
        Ok(())
    }

    /// Send an encode request and wait for the response.
    ///
    /// `messages_json` is the serialized JSON payload, e.g.
    /// `{"messages": [{"role": "user", "content": [...]}]}`.
    pub async fn encode(&mut self, messages_json: &[u8]) -> anyhow::Result<EncodeResponse> {
        let tx = self
            .request_tx
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Encoder adapter not connected"))?;

        let packet = ZmqPacket {
            action: ACTION_ENCODE_REQUEST,
            payload: messages_json.to_vec(),
        };
        tx.send(packet)
            .map_err(|_| anyhow::anyhow!("Encoder send channel closed"))?;

        // Wait for response with timeout
        let rx = self
            .response_rx
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Encoder adapter not connected"))?;

        let response = tokio::time::timeout(std::time::Duration::from_secs(60), rx.recv())
            .await
            .map_err(|_| anyhow::anyhow!("Encoder response timeout (60s)"))?
            .ok_or_else(|| anyhow::anyhow!("Encoder response channel closed"))?;

        if response.action != ACTION_ENCODE_RESPONSE {
            return Err(anyhow::anyhow!(
                "Unexpected encoder response action={}",
                response.action
            ));
        }

        let resp: EncodeResponse = serde_json::from_slice(&response.payload).map_err(|e| {
            anyhow::anyhow!(
                "Failed to parse encoder response: {} (payload={})",
                e,
                String::from_utf8_lossy(&response.payload)
            )
        })?;

        if let Some(err) = &resp.error {
            return Err(anyhow::anyhow!("Encoder error: {}", err));
        }

        Ok(resp)
    }
}
