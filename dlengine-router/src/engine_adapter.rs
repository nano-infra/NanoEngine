use crate::zmq_packet::ZmqPacket;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use tokio::sync::{mpsc as tokio_mpsc, Mutex};
use tracing::{info, warn};

const ACTION_STEPOUT: u32 = 0;
const ACTION_ADD_OR_MIGRATE: u32 = 1;
const ACTION_FREE: u32 = 3;
const STATUS_RUNNING: i32 = 1;
const STATUS_FINISHED: i32 = 2;

pub struct EngineAdapter {
    pub request_tx: Option<mpsc::SyncSender<ZmqPacket>>,
    pub pending_requests: Arc<Mutex<HashMap<u64, RequestState>>>,
    pub uuid: Option<String>,
    pub world_size: i32,
    pub num_blocks: i32,
    // Shutdown signal: when dropped, closes the channel to stop reader loop
    pub shutdown_tx: Option<tokio_mpsc::UnboundedSender<()>>,
    // Reader task handle: must be properly awaited during shutdown
    pub reader_handle: Option<tokio::task::JoinHandle<()>>,
    // Keep recv_tx alive to prevent channel from closing prematurely
    pub recv_tx_keepalive: Option<tokio_mpsc::UnboundedSender<ZmqPacket>>,
    // I/O thread handle: must be properly joined during shutdown
    pub io_thread_handle: Option<std::thread::JoinHandle<()>>,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(u32),
    Finished,
    Error(String),
    Migrate(Vec<u8>),
}

pub struct RequestState {
    pub sender: tokio_mpsc::UnboundedSender<StreamEvent>,
    pub accumulated_tokens: Vec<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
struct WireSamplingParams {
    temperature: f64,
    max_tokens: i32,
    ignore_eos: bool,
    return_completion_logprobs: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct WireVisionSlot {
    encoder_engine_id: String,
    slot_idx: i32,
    num_tokens: i32,
    hidden_size: i32,
    max_tokens_per_slot: i32,
}

#[derive(Debug, Deserialize, Serialize)]
struct WireAddRequest {
    seq_id: u64,
    prompt_token_ids: Vec<i32>,
    sampling_params: WireSamplingParams,
    affinity_key: u64,
    vision_slots: Vec<WireVisionSlot>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct WireMigrationRequest {
    seq_id: u64,
    status: i32,
    token_ids: Vec<i32>,
    last_token: i32,
    num_tokens: i32,
    num_prompt_tokens: i32,
    num_checkpointed_tokens: i32,
    num_cached_tokens: i32,
    affinity_key: u64,
    sampling_params: WireSamplingParams,
    completion_logprobs: Vec<f32>,
    active_block_table: Vec<i32>,
    active_block_tables: HashMap<i32, Vec<i32>>,
    active_dispatched_tokens: Vec<i32>,
    active_dp_idx: i32,
    active_group_id: i32,
    active_state_slot: i32,
    active_compressed_block_tables: HashMap<i32, Vec<i32>>,
    active_hisparse_slot: i32,
    migrate_block_table: Vec<i32>,
    migrate_block_tables: HashMap<i32, Vec<i32>>,
    migrate_engine_id: String,
    migrate_num_kvcache_blocks: i32,
    migrate_group_size: i32,
    migrate_dp_idx: i32,
    migrate_group_id: i32,
    migrate_state_slot: i32,
    migrate_compressed_block_tables: HashMap<i32, Vec<i32>>,
    migrate_hisparse_slot: i32,
}

#[derive(Debug, Deserialize)]
struct StepOut {
    seq_id: u64,
    #[serde(default)]
    token_id: Option<u32>,
    #[serde(default)]
    token_ids: Vec<u32>,
    status: i32,
}

#[derive(Debug, Serialize)]
struct FreeSequences {
    seq_ids: Vec<u64>,
    source_engine_id: String,
}

fn encode_add_requests(requests: &[WireAddRequest]) -> anyhow::Result<Vec<u8>> {
    bincode::serialize(requests).map_err(|e| anyhow::anyhow!("failed to encode add request: {e}"))
}

fn decode_migration_seq_id(payload: &[u8]) -> anyhow::Result<u64> {
    let request: WireMigrationRequest = bincode::deserialize(payload)
        .map_err(|e| anyhow::anyhow!("failed to decode migration request: {e}"))?;
    Ok(request.seq_id)
}

fn decode_stepout(payload: &[u8]) -> anyhow::Result<StepOut> {
    serde_json::from_slice(payload).map_err(|e| anyhow::anyhow!("failed to decode stepout: {e}"))
}

fn encode_free_sequences(seq_ids: Vec<u64>, source_engine_id: &str) -> anyhow::Result<Vec<u8>> {
    serde_json::to_vec(&FreeSequences {
        seq_ids,
        source_engine_id: source_engine_id.to_string(),
    })
    .map_err(|e| anyhow::anyhow!("failed to encode free request: {e}"))
}

impl EngineAdapter {
    pub fn new(_id: String) -> Self {
        Self {
            request_tx: None,
            pending_requests: Arc::new(Mutex::new(HashMap::new())),
            uuid: None,
            world_size: 0,
            num_blocks: 0,
            shutdown_tx: None,
            reader_handle: None,
            recv_tx_keepalive: None,
            io_thread_handle: None,
        }
    }

    pub async fn connect(&mut self, addr: &str) -> anyhow::Result<()> {
        let endpoint = format!("tcp://{}", addr);
        info!("Connecting to ZMQ Engine at {}", endpoint);

        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::DEALER)?;

        // Set essential socket options
        socket.set_linger(0)?; // Don't wait on close
        socket.set_sndtimeo(5000)?; // 5s send timeout
        socket.set_reconnect_ivl(100)?; // Reconnect after 100ms

        socket.connect(&endpoint)?;
        info!("ZMQ socket connected to {}", endpoint);

        let (send_tx, send_rx) = mpsc::sync_channel::<ZmqPacket>(256);
        self.request_tx = Some(send_tx);

        let (recv_tx, mut recv_rx) = tokio_mpsc::unbounded_channel::<ZmqPacket>();

        // Store recv_tx to keep the channel alive (even if I/O thread exits)
        self.recv_tx_keepalive = Some(recv_tx.clone());

        // Create shutdown channel to gracefully stop reader loop
        let (shutdown_tx, mut shutdown_rx) = tokio_mpsc::unbounded_channel::<()>();
        let shutdown_tx_for_storage = shutdown_tx.clone();
        self.shutdown_tx = Some(shutdown_tx_for_storage);

        let pending = self.pending_requests.clone();

        let addr_for_log = addr.to_string();
        let addr_for_reader = addr_for_log.clone();

        // Single I/O thread: recv with timeout, drain send channel. ZMQ sockets are not thread-safe.
        socket.set_rcvtimeo(100)?; // 100ms timeout for poll loop
        let recv_tx_for_io = recv_tx.clone();
        let io_thread_handle = thread::spawn(move || {
            loop {
                // Try recv (returns EAGAIN after timeout if no data)
                match socket.recv_bytes(0) {
                    Ok(data) => {
                        if let Ok(packet) = ZmqPacket::decode(&data) {
                            if recv_tx_for_io.send(packet).is_err() {
                                // Channel closed, reader loop stopped - this is a clean shutdown
                                info!(
                                    "ZMQ I/O thread: receiver channel closed for {}",
                                    addr_for_log
                                );
                                break;
                            }
                        }
                    }
                    Err(zmq::Error::EAGAIN) => {
                        // Timeout is normal, continue polling
                    }
                    Err(zmq::Error::ETERM) => {
                        // Context terminated - clean shutdown
                        info!("ZMQ I/O thread: context terminated for {}", addr_for_log);
                        break;
                    }
                    Err(e) => {
                        warn!("ZMQ recv error for {}: {}", addr_for_log, e);
                        break;
                    }
                }

                // Drain send channel
                loop {
                    match send_rx.try_recv() {
                        Ok(packet) => {
                            let data = packet.encode();
                            // Only log important packets (ADD/migration=1, engine_info=2)
                            if packet.action != 0 {
                                info!(
                                    "Sending ZMQ packet: action={}, size={}",
                                    packet.action,
                                    data.len()
                                );
                            }
                            if let Err(e) = socket.send(&data, 0) {
                                if matches!(e, zmq::Error::ETERM) {
                                    info!(
                                        "ZMQ I/O thread: context terminated during send for {}",
                                        addr_for_log
                                    );
                                } else {
                                    warn!("ZMQ send error for {}: {}", addr_for_log, e);
                                }
                                break;
                            }
                        }
                        Err(mpsc::TryRecvError::Empty) => {
                            break; // Done draining
                        }
                        Err(mpsc::TryRecvError::Disconnected) => {
                            info!("ZMQ I/O thread: send channel closed for {}", addr_for_log);
                            return; // Exit the I/O thread entirely
                        }
                    }
                }
            }
            info!("ZMQ I/O thread ended for {}", addr_for_log);
        });

        // Store the I/O thread handle so it can be properly joined during shutdown
        self.io_thread_handle = Some(io_thread_handle);

        // Spawn async reader that processes received packets
        let reader_handle = tokio::spawn(async move {
            info!("EngineAdapter reader loop started for {}", addr_for_reader);

            loop {
                tokio::select! {
                    packet_opt = recv_rx.recv() => {
                        let packet = match packet_opt {
                            Some(p) => p,
                            None => {
                                // Channel closed, exit loop
                                info!("EngineAdapter reader loop: recv channel closed for {}", addr_for_reader);
                                break;
                            }
                        };
                let action = packet.action;
                let payload = packet.payload;

                // Only log migration (action=1) and engine info (action=2) packets
                if action != 0 {
                    info!("Received packet: action={}, payload_size={}", action, payload.len());
                }

                // Action 1: Migration response (Rust-owned bincode payload)
                if action == ACTION_ADD_OR_MIGRATE {
                    let seq_id = match decode_migration_seq_id(&payload) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!("Failed to parse migration payload: {}", e);
                            continue;
                        }
                    };
                    if seq_id > 0 {
                        let mut map = pending.lock().await;
                        let map_size = map.len();
                        if let Some(state) = map.remove(&seq_id) {
                            if state.sender.send(StreamEvent::Migrate(payload)).is_err() {
                                warn!("Migration event send failed (client disconnected) for seq_id={}", seq_id);
                            }
                        } else {
                            warn!("Migration response for seq_id={} not found in pending_requests (map_size={})", seq_id, map_size);
                        }
                    }
                    continue;
                }



                // Action 0: StepOut (token streaming)
                if action == ACTION_STEPOUT {
                    let step_out = match decode_stepout(&payload) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!("Failed to parse StepOut payload: {}", e);
                            continue;
                        }
                    };
                    let seq_id = step_out.seq_id;
                    let status = step_out.status;

                    let tokens: Vec<u32> = if step_out.token_ids.is_empty() {
                        step_out
                            .token_id
                            .filter(|token_id| *token_id > 0)
                            .into_iter()
                            .collect()
                    } else {
                        step_out.token_ids
                    };

                    let mut map = pending.lock().await;
                    if status == STATUS_FINISHED {
                        if let Some(final_state) = map.remove(&seq_id) {
                            // Log finish with total token count
                            let total_tokens = final_state.accumulated_tokens.len() + tokens.len();
                            info!("Sequence {} finished: {} tokens generated, pending_map_size={}", seq_id, total_tokens, map.len());

                            for token_id in &tokens {
                                if final_state.sender.send(StreamEvent::Token(*token_id)).is_err() {
                                    warn!("Sequence {} finish token send failed (client disconnected)", seq_id);
                                    break;
                                }
                            }
                            if final_state.sender.send(StreamEvent::Finished).is_err() {
                                warn!("Sequence {} Finished event send failed (client disconnected)", seq_id);
                            }
                        } else {
                            warn!("Sequence {} finished but not found in pending_requests (map_size={})", seq_id, map.len());
                        }
                    } else if status == STATUS_RUNNING {
                        if let Some(state) = map.get_mut(&seq_id) {
                            // Only log at the beginning (first token)
                            let is_first = state.accumulated_tokens.is_empty();

                            for token_id in tokens {
                                state.accumulated_tokens.push(token_id);
                                if state.sender.send(StreamEvent::Token(token_id)).is_err() {
                                    warn!("Sequence {} token send failed (client disconnected)", seq_id);
                                    break;
                                }
                            }

                            if is_first {
                                info!("Sequence {} started generation", seq_id);
                            }
                        } else {
                            warn!("Sequence {} token received but not found in pending_requests (map_size={})", seq_id, map.len());
                        }
                    }
                }
                    }
                    _ = shutdown_rx.recv() => {
                        // Shutdown signal received, exit loop
                        info!("EngineAdapter reader loop: shutdown signal received for {}", addr_for_reader);
                        break;
                    }
                }
            }

            info!("EngineAdapter reader loop ended for {}", addr_for_reader);
        });

        // Store the reader handle so it can be properly awaited during shutdown
        self.reader_handle = Some(reader_handle);

        Ok(())
    }

    fn send_packet(&self, action: u32, payload: Vec<u8>) -> anyhow::Result<()> {
        if let Some(tx) = &self.request_tx {
            let packet = ZmqPacket { action, payload };
            tx.send(packet)
                .map_err(|_| anyhow::anyhow!("Request channel closed"))?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Not connected"))
        }
    }

    pub async fn send_raw_request(
        &mut self,
        seq_id: u64,
        payload: Vec<u8>,
    ) -> anyhow::Result<tokio_mpsc::UnboundedReceiver<StreamEvent>> {
        let (tx, rx) = tokio_mpsc::unbounded_channel();
        {
            let mut map = self.pending_requests.lock().await;
            map.insert(
                seq_id,
                RequestState {
                    sender: tx,
                    accumulated_tokens: Vec::new(),
                },
            );
        }
        info!(
            "Sending Raw Request (Migration Forward) for Seq {}. Payload Size: {}",
            seq_id,
            payload.len()
        );
        self.send_packet(ACTION_ADD_OR_MIGRATE, payload)?;
        Ok(rx)
    }

    pub async fn send_add_request(
        &mut self,
        seq_id: u64,
        token_ids: &[u32],
        max_tokens: i32,
        temperature: f32,
        ignore_eos: bool,
    ) -> anyhow::Result<tokio_mpsc::UnboundedReceiver<StreamEvent>> {
        self.send_add_request_with_vision(
            seq_id,
            token_ids,
            max_tokens,
            temperature,
            ignore_eos,
            None,
        )
        .await
    }

    pub async fn send_add_request_with_vision(
        &mut self,
        seq_id: u64,
        token_ids: &[u32],
        max_tokens: i32,
        temperature: f32,
        ignore_eos: bool,
        vision_slots_info: Option<&[crate::encoder_adapter::VisionSlotInfo]>,
    ) -> anyhow::Result<tokio_mpsc::UnboundedReceiver<StreamEvent>> {
        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let vision_slots = vision_slots_info
            .unwrap_or_default()
            .iter()
            .map(|slot| WireVisionSlot {
                encoder_engine_id: slot.encoder_engine_id.clone(),
                slot_idx: slot.slot_idx as i32,
                num_tokens: slot.num_tokens as i32,
                hidden_size: slot.hidden_size as i32,
                max_tokens_per_slot: slot.max_tokens_per_slot as i32,
            })
            .collect();
        let request = WireAddRequest {
            seq_id,
            prompt_token_ids: token_ids_i32,
            sampling_params: WireSamplingParams {
                temperature: temperature as f64,
                max_tokens,
                ignore_eos,
                return_completion_logprobs: false,
            },
            affinity_key: 0,
            vision_slots,
        };
        let payload = encode_add_requests(&[request])?;

        let (tx, rx) = tokio_mpsc::unbounded_channel();
        {
            let mut map = self.pending_requests.lock().await;
            map.insert(
                seq_id,
                RequestState {
                    sender: tx,
                    accumulated_tokens: Vec::new(),
                },
            );
        }
        info!(
            "Sending ADD request for seq {} with {} tokens, max_tokens={}",
            seq_id,
            token_ids.len(),
            max_tokens
        );
        self.send_packet(ACTION_ADD_OR_MIGRATE, payload)?;
        Ok(rx)
    }

    pub async fn send_free_request(&mut self, seq_id: u64) -> anyhow::Result<()> {
        let payload = encode_free_sequences(vec![seq_id], "router")?;

        info!("Sending FREE request for seq {} from router", seq_id);
        self.send_packet(ACTION_FREE, payload)?;

        // Remove from pending completely
        let mut map = self.pending_requests.lock().await;
        map.remove(&seq_id);

        Ok(())
    }
}
