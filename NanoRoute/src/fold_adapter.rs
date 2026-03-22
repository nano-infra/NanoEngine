//! ZMQ DEALER adapter for NanoFold (request-response, JSON over ZmqPacket).
//!
//! Protocol:
//! - Request:  action=7 (FoldRequest),  payload = JSON {"op": "predict"|"embed"|"sample"|"job_status"|"embed_status"|"health", ...}
//! - Response: action=8 (FoldResponse), payload = JSON {"ok": bool, ...}
//!
//! All operations return immediately (jobs are queued server-side; poll via job_status).
//! A Mutex serializes concurrent callers so responses are never mis-routed.

use crate::zmq_packet::ZmqPacket;
use serde::Deserialize;
use std::sync::mpsc;
use std::thread;
use tokio::sync::{mpsc as tokio_mpsc, Mutex};
use tracing::{info, warn};

const ACTION_FOLD_REQUEST: u32 = 7;
const ACTION_FOLD_RESPONSE: u32 = 8;

pub struct FoldAdapter {
    /// Serializes concurrent callers (ZMQ DEALER has no routing envelope).
    lock: Mutex<FoldAdapterInner>,
}

struct FoldAdapterInner {
    request_tx: mpsc::SyncSender<ZmqPacket>,
    response_rx: tokio_mpsc::UnboundedReceiver<ZmqPacket>,
    /// Keep sender alive so channel stays open.
    _response_tx_keepalive: tokio_mpsc::UnboundedSender<ZmqPacket>,
    _io_thread: thread::JoinHandle<()>,
}

/// Parsed header fields from a fold response (full payload kept in `raw`).
#[derive(Debug)]
pub struct FoldResp {
    /// True when the operation succeeded server-side.
    pub ok: bool,
    /// Optional error code (404 = not found on this server; used for fan-out polling).
    pub code: Option<u64>,
    /// Raw JSON bytes of the response for pass-through to the HTTP client.
    pub raw: Vec<u8>,
}

impl FoldAdapter {
    pub async fn connect(addr: &str) -> anyhow::Result<Self> {
        let endpoint = format!("tcp://{}", addr);
        info!("Connecting to NanoFold at {}", endpoint);

        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::DEALER)?;
        socket.set_linger(0)?;
        socket.set_sndtimeo(5000)?;
        socket.set_rcvtimeo(200)?;
        socket.set_reconnect_ivl(100)?;
        socket.connect(&endpoint)?;
        info!("NanoFold ZMQ socket connected to {}", endpoint);

        let (send_tx, send_rx) = mpsc::sync_channel::<ZmqPacket>(64);
        let (recv_tx, recv_rx) = tokio_mpsc::unbounded_channel::<ZmqPacket>();
        let recv_tx_clone = recv_tx.clone();

        let addr_log = addr.to_string();
        let io_handle = thread::spawn(move || loop {
            // Try recv
            match socket.recv_bytes(0) {
                Ok(data) => {
                    if let Ok(pkt) = ZmqPacket::decode(&data) {
                        if recv_tx.send(pkt).is_err() {
                            info!("FoldAdapter I/O thread: channel closed for {}", addr_log);
                            break;
                        }
                    }
                }
                Err(zmq::Error::EAGAIN) => {}
                Err(zmq::Error::ETERM) => {
                    info!(
                        "FoldAdapter I/O thread: context terminated for {}",
                        addr_log
                    );
                    break;
                }
                Err(e) => {
                    warn!("FoldAdapter ZMQ recv error for {}: {}", addr_log, e);
                    break;
                }
            }
            // Drain send channel
            loop {
                match send_rx.try_recv() {
                    Ok(pkt) => {
                        let data = pkt.encode();
                        if let Err(e) = socket.send(&data, 0) {
                            warn!("FoldAdapter ZMQ send error for {}: {}", addr_log, e);
                            break;
                        }
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => return,
                }
            }
        });

        Ok(Self {
            lock: Mutex::new(FoldAdapterInner {
                request_tx: send_tx,
                response_rx: recv_rx,
                _response_tx_keepalive: recv_tx_clone,
                _io_thread: io_handle,
            }),
        })
    }

    /// Send a fold request JSON payload and wait for the response.
    /// `timeout_s` covers the full round-trip.
    pub async fn request(&self, payload_json: &[u8], timeout_s: u64) -> anyhow::Result<FoldResp> {
        let mut inner = self.lock.lock().await;

        let pkt = ZmqPacket {
            action: ACTION_FOLD_REQUEST,
            payload: payload_json.to_vec(),
        };
        inner
            .request_tx
            .send(pkt)
            .map_err(|_| anyhow::anyhow!("FoldAdapter send channel closed"))?;

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_s),
            inner.response_rx.recv(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("FoldAdapter response timeout ({}s)", timeout_s))?
        .ok_or_else(|| anyhow::anyhow!("FoldAdapter response channel closed"))?;

        if response.action != ACTION_FOLD_RESPONSE {
            return Err(anyhow::anyhow!(
                "Unexpected fold response action={}",
                response.action
            ));
        }

        // Parse only the header fields needed for routing decisions.
        #[derive(Deserialize)]
        struct Header {
            ok: bool,
            #[serde(default)]
            code: Option<u64>,
        }
        let header: Header = serde_json::from_slice(&response.payload).map_err(|e| {
            anyhow::anyhow!(
                "Failed to parse fold response: {} (payload={})",
                e,
                String::from_utf8_lossy(&response.payload)
            )
        })?;

        Ok(FoldResp {
            ok: header.ok,
            code: header.code,
            raw: response.payload,
        })
    }
}
