use crate::engine_rpc::{engine_service_client::EngineServiceClient, StreamPacket};
use crate::fbs::{
    EngineInfo, EngineInfoArgs, P2PInit, P2PInitArgs, P2PInitResponse, PeerT, SamplingParams,
    SamplingParamsArgs, Sequence, SequenceArgs, SequenceList, SequenceListArgs, SequenceStatus,
    StepOut,
};
use flatbuffers::FlatBufferBuilder;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Endpoint;
use tonic::Request;
use tracing::{error, info};

pub struct EngineAdapter {
    // We send requests via this channel, which pipes into the gRPC stream
    pub request_tx: Option<mpsc::Sender<StreamPacket>>,

    pub pending_requests: Arc<Mutex<HashMap<u64, RequestState>>>,
    pub uuid: Option<String>,
    pub world_size: i32,
    pub num_blocks: i32,
    pub next_seq_id: AtomicU64,
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
    pub sender: mpsc::UnboundedSender<StreamEvent>,
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
        }
    }

    pub async fn connect(&mut self, addr: &str) -> anyhow::Result<()> {
        let uri = format!("http://{}", addr);
        info!("Connecting to gRPC Engine at {}", uri);

        let endpoint = Endpoint::from_shared(uri)?
            .tcp_keepalive(Some(Duration::from_secs(20)))
            .http2_keep_alive_interval(Duration::from_secs(10))
            .connect_timeout(Duration::from_secs(10));

        let channel = endpoint.connect().await?;
        let mut client = EngineServiceClient::new(channel);

        // Create bi-directional stream
        let (tx, rx) = mpsc::channel(128);
        self.request_tx = Some(tx);
        let request_stream = ReceiverStream::new(rx);

        let response = client.interact(Request::new(request_stream)).await?;
        let mut response_stream = response.into_inner();

        // Spawn background reader task
        let pending = self.pending_requests.clone();
        tokio::spawn(async move {
            info!("EngineAdapter reader loop started.");
            while let Ok(Some(packet)) = response_stream.message().await {
                let action = packet.action;
                let payload = packet.payload;
                let seq_id = packet.seq_id;

                // Dispatch Logic (Identical to legacy)

                // Action 1: Migration
                if action == 1 {
                    // Migration Event (Payload is SequenceList serialization)
                    // We need to peek inside to get seq_id
                    let extracted_seq_id = flatbuffers::root::<SequenceList>(&payload)
                        .ok()
                        .and_then(|sl: SequenceList| sl.sequences())
                        .filter(|seqs| !seqs.is_empty())
                        .map(|seqs| seqs.get(0).seq_id());

                    // Prefer extracted ID for migration as it's data plane,
                    // but packet.seq_id should match if python server is correct.
                    let effective_id = extracted_seq_id.unwrap_or(seq_id);

                    if effective_id > 0 {
                        let mut map = pending.lock().await;
                        if let Some(state) = map.remove(&effective_id) {
                            let _ = state.sender.send(StreamEvent::Migrate(payload));
                        }
                    }
                    continue;
                }

                if action == 2 || action == 3 || action == 4 {
                    // Control Responses
                    let mut map = pending.lock().await;
                    if let Some(state) = map.remove(&seq_id) {
                        let _ = state.sender.send(StreamEvent::P2PResponse(payload));
                    }
                    continue;
                }

                if action == 0 {
                    // StepOut (Token)
                    if let Ok(step_out) = flatbuffers::root::<StepOut>(&payload) {
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
            }
            info!("EngineAdapter reader loop ended (Stream closed).");
        });

        Ok(())
    }

    // Internal helper to send Generic Packet
    async fn send_packet(&self, action: u32, seq_id: u64, payload: Vec<u8>) -> anyhow::Result<()> {
        if let Some(tx) = &self.request_tx {
            let packet = StreamPacket {
                seq_id,
                action,
                payload,
            };
            tx.send(packet)
                .await
                .map_err(|_| anyhow::anyhow!("Request Stream Closed"))?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Not connected"))
        }
    }

    // New method to send raw payload (for forwarding migration)
    pub async fn send_raw_request(
        &mut self,
        seq_id: u64,
        payload: Vec<u8>,
    ) -> anyhow::Result<mpsc::UnboundedReceiver<StreamEvent>> {
        // Register pending request
        let (tx, rx) = mpsc::unbounded_channel();
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

        // Action 1: AddRequest
        self.send_packet(1, seq_id, payload).await?;

        Ok(rx)
    }

    pub async fn send_add_request(
        &mut self,
        seq_id: u64,
        token_ids: &[u32],
        max_tokens: i32,
    ) -> anyhow::Result<mpsc::UnboundedReceiver<StreamEvent>> {
        // Application layer serialization
        let mut builder = FlatBufferBuilder::new();
        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let t_vec = builder.create_vector(&token_ids_i32);

        // Create SamplingParams table
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

        // Register pending request
        let (tx, rx) = mpsc::unbounded_channel();
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

        // Action 1: Add Request
        self.send_packet(1, seq_id, payload).await?;

        Ok(rx)
    }

    pub async fn send_get_engine_info(&mut self) -> anyhow::Result<serde_json::Value> {
        // Unique Seq ID
        let seq_id = self.next_seq_id.fetch_add(1, Ordering::Relaxed);

        // Register pending request
        let (tx, mut rx) = mpsc::unbounded_channel();
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

        // Action 2: GetEngineInfo
        self.send_packet(2, seq_id, vec![]).await?;

        // Wait for response from reader loop
        if let Some(event) = rx.recv().await {
            match event {
                StreamEvent::P2PResponse(body) => {
                    let info = flatbuffers::root::<EngineInfo>(&body)?;

                    use serde_json::json;
                    let json_val = json!({
                        "id": info.id(),
                        "role": info.role(),
                        "rank": info.rank(),
                        "world_size": info.world_size(),
                        "num_blocks": info.num_blocks(),
                        "host": info.host(),
                        "port": info.port(),
                        "status": info.status()
                    });

                    Ok(json_val)
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

    pub async fn send_p2p_init(
        &mut self,
        nodes: Vec<(String, String, u16, String, i32, i32)>,
    ) -> anyhow::Result<HashMap<String, PeerT>> {
        let mut builder = FlatBufferBuilder::new();

        let mut node_offsets = Vec::new();
        for (id, host, port, role, ws, nb) in nodes {
            let id_off = builder.create_string(&id);
            let role_off = builder.create_string(&role);
            let host_off = builder.create_string(&host);
            let status_off = builder.create_string("ready");

            node_offsets.push(EngineInfo::create(
                &mut builder,
                &EngineInfoArgs {
                    id: Some(id_off),
                    role: Some(role_off),
                    host: Some(host_off),
                    port: port as i32,
                    world_size: ws,
                    num_blocks: nb,
                    status: Some(status_off),
                    ..Default::default()
                },
            ));
        }

        let nodes_vec = builder.create_vector(&node_offsets);
        let p2p_init = P2PInit::create(
            &mut builder,
            &P2PInitArgs {
                nodes: Some(nodes_vec),
            },
        );

        builder.finish(p2p_init, None);
        let payload = builder.finished_data().to_vec();

        // Unique Seq ID
        let seq_id = self.next_seq_id.fetch_add(1, Ordering::Relaxed);

        // Register pending request
        let (tx, mut rx) = mpsc::unbounded_channel();
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

        // Action 3: P2PInit
        self.send_packet(3, seq_id, payload).await?;

        // Wait for response from reader loop
        if let Some(event) = rx.recv().await {
            match event {
                StreamEvent::P2PResponse(body) => {
                    let p2p_resp = flatbuffers::root::<P2PInitResponse>(&body)
                        .map_err(|e| anyhow::anyhow!("FlatBuffer Error: {:?}", e))?;
                    let mut result: HashMap<String, PeerT> = HashMap::new();
                    if let Some(peers) = p2p_resp.responses() {
                        for i in 0..peers.len() {
                            let peer = peers.get(i);
                            if let Some(id) = peer.remote_id() {
                                result.insert(id.to_string(), peer.unpack());
                            }
                        }
                    }
                    Ok(result)
                }
                _ => Err(anyhow::anyhow!("Unexpected response event for P2P Init")),
            }
        } else {
            Err(anyhow::anyhow!(
                "Channel closed while waiting for P2P Init response"
            ))
        }
    }

    pub async fn send_p2p_connect(&mut self, peers: Vec<PeerT>) -> anyhow::Result<()> {
        use crate::fbs::{P2PConnect, P2PConnectArgs};

        let mut builder = FlatBufferBuilder::new();
        let mut peer_offsets = Vec::new();

        for peer_t in peers {
            peer_offsets.push(peer_t.pack(&mut builder));
        }

        let peers_vec = builder.create_vector(&peer_offsets);
        let p2p_connect = P2PConnect::create(
            &mut builder,
            &P2PConnectArgs {
                peers: Some(peers_vec),
            },
        );

        builder.finish(p2p_connect, None);
        let payload = builder.finished_data().to_vec();

        // Unique Seq ID
        let seq_id = self.next_seq_id.fetch_add(1, Ordering::Relaxed);

        // Register pending request
        let (tx, mut rx) = mpsc::unbounded_channel();
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

        // Action 4: P2PConnect
        self.send_packet(4, seq_id, payload).await?;

        // Wait for Ack
        if let Some(event) = rx.recv().await {
            match event {
                StreamEvent::P2PResponse(_) => Ok(()),
                _ => Err(anyhow::anyhow!("Unexpected response event for P2P Connect")),
            }
        } else {
            Err(anyhow::anyhow!(
                "Channel closed waiting for P2P Connect Ack"
            ))
        }
    }
}
