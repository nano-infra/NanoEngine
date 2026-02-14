use crate::fbs::{
    SamplingParams, SamplingParamsArgs, Sequence, SequenceArgs, SequenceList, SequenceListArgs,
    SequenceStatus, StepOut,
};
use crate::zmq_packet::ZmqPacket;
use flatbuffers::FlatBufferBuilder;
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use tokio::sync::{mpsc as tokio_mpsc, Mutex};
use tracing::{info, warn};

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
    // Dedicated channel for GetEngineInfo responses (no seq_id needed)
    engine_info_tx: Arc<Mutex<Option<tokio_mpsc::UnboundedSender<Vec<u8>>>>>,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(u32),
    Finished,
    #[allow(dead_code)]
    Error(String),
    Migrate(Vec<u8>),
}

pub struct RequestState {
    pub sender: tokio_mpsc::UnboundedSender<StreamEvent>,
    pub accumulated_tokens: Vec<u32>,
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
            engine_info_tx: Arc::new(Mutex::new(None)),
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
        let engine_info_tx = self.engine_info_tx.clone();
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
                while let Ok(packet) = send_rx.try_recv() {
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

                // Action 1: Migration response (SequenceList payload)
                if action == 1 {
                    let sl = unsafe { flatbuffers::root_unchecked::<SequenceList>(&payload) };
                    let seq_id = sl.sequences().and_then(|seqs| {
                        if seqs.is_empty() {
                            None
                        } else {
                            Some(seqs.get(0).seq_id())
                        }
                    });
                    if let Some(seq_id) = seq_id {
                        if seq_id > 0 {
                            let mut map = pending.lock().await;
                            let map_size = map.len();
                            if let Some(state) = map.remove(&seq_id) {
                                match state.sender.send(StreamEvent::Migrate(payload)) {
                                    Ok(_) => {
                                        info!("[DIAG] Migration event sent OK for seq_id={}, pending_map_size={}", seq_id, map_size - 1);
                                    }
                                    Err(_) => {
                                        warn!("[DIAG] Migration event SEND FAILED (rx dropped = client disconnected) for seq_id={}, pending_map_size={}", seq_id, map_size - 1);
                                    }
                                }
                            } else {
                                warn!("[DIAG] Migration response for seq_id={} but NOT FOUND in pending_requests (map_size={})", seq_id, map_size);
                            }
                        }
                    } else {
                        warn!("[DIAG] Migration response with no seq_id in SequenceList");
                    }
                    continue;
                }

                // Action 2: GetEngineInfo response — deliver via dedicated channel
                if action == 2 {
                    let guard = engine_info_tx.lock().await;
                    if let Some(tx) = guard.as_ref() {
                        let _ = tx.send(payload);
                    }
                    continue;
                }

                // Action 0: StepOut (token streaming)
                if action == 0 {
                    let step_out = unsafe { flatbuffers::root_unchecked::<StepOut>(&payload) };
                    let seq_id = step_out.seq_id();
                    let status = step_out.status();

                    // Extract tokens: prefer token_ids vector over single token_id
                    let tokens: Vec<u32> = step_out.token_ids()
                        .map(|ids| ids.iter().collect())
                        .unwrap_or_else(|| {
                            let token_id = step_out.token_id();
                            if token_id > 0 { vec![token_id] } else { vec![] }
                        });

                    let mut map = pending.lock().await;
                    if status == SequenceStatus::FINISHED {
                        if let Some(final_state) = map.remove(&seq_id) {
                            // Log finish with total token count
                            let total_tokens = final_state.accumulated_tokens.len() + tokens.len();
                            info!("Sequence {} finished: {} tokens generated, pending_map_size={}", seq_id, total_tokens, map.len());

                            for token_id in &tokens {
                                if final_state.sender.send(StreamEvent::Token(*token_id)).is_err() {
                                    warn!("[DIAG] Sequence {} FINISH token send failed (client disconnected)", seq_id);
                                    break;
                                }
                            }
                            if final_state.sender.send(StreamEvent::Finished).is_err() {
                                warn!("[DIAG] Sequence {} FINISHED event send failed (client disconnected)", seq_id);
                            }
                        } else {
                            warn!("[DIAG] Sequence {} FINISHED but NOT FOUND in pending_requests (map_size={})", seq_id, map.len());
                        }
                    } else if matches!(status, SequenceStatus::RUNNING_PREFILL | SequenceStatus::RUNNING_DECODE) {
                        if let Some(state) = map.get_mut(&seq_id) {
                            // Only log at the beginning (first token)
                            let is_first = state.accumulated_tokens.is_empty();

                            for token_id in tokens {
                                state.accumulated_tokens.push(token_id);
                                if state.sender.send(StreamEvent::Token(token_id)).is_err() {
                                    warn!("[DIAG] Sequence {} token send failed (client disconnected), accumulated={}", seq_id, state.accumulated_tokens.len());
                                    break;
                                }
                            }

                            if is_first {
                                info!("Sequence {} started generation", seq_id);
                            }
                        } else {
                            // Only warn for the first occurrence to avoid log spam
                            warn!("[DIAG] Sequence {} token received but NOT FOUND in pending_requests (map_size={})", seq_id, map.len());
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
        self.send_packet(1, payload)?;
        Ok(rx)
    }

    pub async fn send_add_request(
        &mut self,
        seq_id: u64,
        token_ids: &[u32],
        max_tokens: i32,
    ) -> anyhow::Result<tokio_mpsc::UnboundedReceiver<StreamEvent>> {
        let mut builder = FlatBufferBuilder::new();
        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let t_vec = builder.create_vector(&token_ids_i32);

        let sampling_params = SamplingParams::create(
            &mut builder,
            &SamplingParamsArgs {
                temperature: 0.1,
                max_tokens,
                ignore_eos: false,
            },
        );

        let num_tokens = token_ids.len() as i32;
        let last_token = if num_tokens > 0 {
            token_ids_i32[num_tokens as usize - 1]
        } else {
            0
        };

        let seq = Sequence::create(
            &mut builder,
            &SequenceArgs {
                seq_id,
                status: SequenceStatus::WAITING,
                token_ids: Some(t_vec),
                num_tokens,
                num_prompt_tokens: num_tokens,
                num_checkpointed_tokens: num_tokens,
                last_token,
                sampling_params: Some(sampling_params),
                ..Default::default()
            },
        );

        let seqs = builder.create_vector(&[seq]);
        let root = SequenceList::create(
            &mut builder,
            &SequenceListArgs {
                sequences: Some(seqs),
            },
        );

        builder.finish(root, None);
        let payload = builder.finished_data().to_vec();

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
            seq_id, num_tokens, max_tokens
        );
        self.send_packet(1, payload)?;
        Ok(rx)
    }

    pub async fn send_get_engine_info(&mut self) -> anyhow::Result<serde_json::Value> {
        let (tx, mut rx) = tokio_mpsc::unbounded_channel();
        {
            let mut guard = self.engine_info_tx.lock().await;
            *guard = Some(tx);
        }

        self.send_packet(2, vec![])?;

        if let Some(body) = rx.recv().await {
            // Clear the channel after receiving
            {
                let mut guard = self.engine_info_tx.lock().await;
                *guard = None;
            }
            // Parse engine info as JSON
            if let Ok(json_str) = std::str::from_utf8(&body) {
                if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(json_str) {
                    return Ok(json_val);
                }
            }
            Err(anyhow::anyhow!(
                "Failed to parse engine info response as JSON"
            ))
        } else {
            Err(anyhow::anyhow!(
                "Channel closed while waiting for GetEngineInfo response"
            ))
        }
    }
}
