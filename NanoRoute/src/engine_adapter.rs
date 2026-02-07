use crate::fbs::{
    SamplingParams, SamplingParamsArgs, Sequence, SequenceArgs, SequenceList, SequenceListArgs,
    SequenceStatus, StepOut,
};
use crate::zmq_packet::ZmqPacket;
use flatbuffers::FlatBufferBuilder;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
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
    pub next_seq_id: AtomicU64,
    // Shutdown signal: when dropped, closes the channel to stop reader loop
    pub shutdown_tx: Option<tokio_mpsc::UnboundedSender<()>>,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(u32),
    Finished,
    #[allow(dead_code)]
    Error(String),
    Migrate(Vec<u8>),
    P2PResponse(Vec<u8>),
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
            next_seq_id: AtomicU64::new(1),
            shutdown_tx: None,
        }
    }

    pub async fn connect(&mut self, addr: &str) -> anyhow::Result<()> {
        let endpoint = format!("tcp://{}", addr);
        info!("Connecting to ZMQ Engine at {}", endpoint);

        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::DEALER)?;
        socket.set_linger(0)?;
        socket.set_rcvtimeo(5000)?;
        socket.set_sndtimeo(5000)?;
        socket.connect(&endpoint)?;

        let (send_tx, send_rx) = mpsc::sync_channel::<ZmqPacket>(256);
        self.request_tx = Some(send_tx);

        let (recv_tx, mut recv_rx) = tokio_mpsc::unbounded_channel::<ZmqPacket>();

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
        thread::spawn(move || {
            loop {
                // Try recv (returns EAGAIN after timeout if no data)
                match socket.recv_bytes(0) {
                    Ok(data) => {
                        if let Ok(packet) = ZmqPacket::decode(&data) {
                            if recv_tx_for_io.send(packet).is_err() {
                                // Channel closed, reader loop stopped
                                break;
                            }
                        }
                    }
                    Err(zmq::Error::EAGAIN) => {}
                    Err(e) => {
                        warn!("ZMQ recv error for {}: {}", addr_for_log, e);
                        break;
                    }
                }
                // Drain send channel
                while let Ok(packet) = send_rx.try_recv() {
                    let data = packet.encode();
                    if socket.send(&data, 0).is_err() {
                        break;
                    }
                }
            }
            info!("ZMQ I/O thread ended for {}", addr_for_log);
        });

        // Spawn async reader that processes received packets
        tokio::spawn(async move {
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
                let seq_id = packet.seq_id;

                if action == 1 {
                    let sl = unsafe { flatbuffers::root_unchecked::<SequenceList>(&payload) };
                    let extracted_seq_id = sl.sequences().and_then(|seqs| {
                        if seqs.is_empty() {
                            None
                        } else {
                            Some(seqs.get(0).seq_id())
                        }
                    });
                    let effective_id = extracted_seq_id.unwrap_or(seq_id);
                    if effective_id > 0 {
                        let mut map = pending.lock().await;
                        if let Some(state) = map.remove(&effective_id) {
                            let _ = state.sender.send(StreamEvent::Migrate(payload));
                        }
                    }
                    continue;
                }

                // Action 2: GetEngineInfo response
                if action == 2 {
                    let mut map = pending.lock().await;
                    if let Some(state) = map.remove(&seq_id) {
                        let _ = state.sender.send(StreamEvent::P2PResponse(payload));
                    }
                    continue;
                }

                if action == 0 {
                    let step_out = unsafe { flatbuffers::root_unchecked::<StepOut>(&payload) };
                    let seq_id = step_out.seq_id();
                    let token_id = step_out.token_id();
                    let status = step_out.status();

                    let mut map = pending.lock().await;
                    if status == SequenceStatus::FINISHED {
                        if let Some(final_state) = map.remove(&seq_id) {
                            if token_id > 0 {
                                let _ = final_state.sender.send(StreamEvent::Token(token_id));
                            }
                            let _ = final_state.sender.send(StreamEvent::Finished);
                        }
                    } else if status == SequenceStatus::RUNNING_DECODE && token_id > 0 {
                        if let Some(state) = map.get_mut(&seq_id) {
                            state.accumulated_tokens.push(token_id);
                            let _ = state.sender.send(StreamEvent::Token(token_id));
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

        Ok(())
    }

    async fn send_packet(&self, action: u32, seq_id: u64, payload: Vec<u8>) -> anyhow::Result<()> {
        if let Some(tx) = &self.request_tx {
            let packet = ZmqPacket {
                seq_id,
                action,
                payload,
            };
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
        self.send_packet(1, seq_id, payload).await?;
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
        self.send_packet(1, seq_id, payload).await?;
        Ok(rx)
    }

    pub async fn send_get_engine_info(&mut self) -> anyhow::Result<serde_json::Value> {
        let seq_id = self.next_seq_id.fetch_add(1, Ordering::Relaxed);

        let (tx, mut rx) = tokio_mpsc::unbounded_channel();
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

        self.send_packet(2, seq_id, vec![]).await?;

        if let Some(event) = rx.recv().await {
            match event {
                StreamEvent::P2PResponse(body) => {
                    // Parse engine info as JSON
                    if let Ok(json_str) = std::str::from_utf8(&body) {
                        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(json_str) {
                            return Ok(json_val);
                        }
                    }
                    // If not JSON, return error
                    Err(anyhow::anyhow!(
                        "Failed to parse engine info response as JSON"
                    ))
                }
                _ => Err(anyhow::anyhow!(
                    "Unexpected response event for GetEngineInfo"
                )),
            }
        } else {
            Err(anyhow::anyhow!(
                "Channel closed while waiting for GetEngineInfo response"
            ))
        }
    }
}
