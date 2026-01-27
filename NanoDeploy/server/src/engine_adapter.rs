use crate::fbs::nanodeploy::sequence::nanodeploy::fbs::{SequenceList, SequenceListArgs, Sequence, SequenceArgs, SequenceStatus, StepOut};
use crate::fbs::nanodeploy::connection::nanodeploy::fbs::{EngineInfo, EngineInfoArgs, P2PInit, P2PInitArgs};
use flatbuffers::FlatBufferBuilder;
use spoke::client::SpokeClient;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tracing::{info, error};

pub struct EngineAdapter {
    pub client: SpokeClient,
    pub pending_requests: Arc<Mutex<HashMap<u64, RequestState>>>,
    pub uuid: Option<String>,
    pub world_size: i32,
    pub num_blocks: i32,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(u32),
    Finished,
    Error(String),
    Migrate(Vec<u8>), // Payload is the full SequenceList FBS
    P2PResponse(Vec<u8>), // JSON payload
}

pub struct RequestState {
    pub sender: mpsc::UnboundedSender<StreamEvent>,
    pub accumulated_tokens: Vec<u32>,
}

impl EngineAdapter {
    pub fn new(id: String) -> Self {
        Self {
            client: SpokeClient::new(id),
            pending_requests: Arc::new(Mutex::new(HashMap::new())),
            uuid: None,
            world_size: 0,
            num_blocks: 0,
        }
    }

    pub async fn connect(&mut self, addr: &str) -> anyhow::Result<()> {
        let mut reader = self.client.connect(addr).await?;

        // Spawn background reader task
        let pending = self.pending_requests.clone();
        tokio::spawn(async move {
            info!("EngineAdapter reader loop started.");
            loop {
                match reader.read_msg().await {
                    Ok((meta, body)) => {
                        // Action 0: StepOut (Default)
                        // Action 1: Migration (SequenceList)

                        if meta.action == 1 {
                             // Migration Event
                             // We need to peek inside to get seq_id to route it
                             // Assuming body is SequenceList -> Sequence
                             use crate::fbs::nanodeploy::sequence::nanodeploy::fbs::root_as_sequence_list;
                             if let Ok(seq_list) = root_as_sequence_list(&body) {
                                  if let Some(seqs) = seq_list.sequences() {
                                      if seqs.len() > 0 {
                                          let seq = seqs.get(0);
                                          let seq_id = seq.seq_id();
                                          info!("Received Action 1 (Migration Candidate) for Seq {}. Payload size: {}", seq_id, body.len());
                                          let mut map = pending.lock().await;
                                          if let Some(state) = map.remove(&seq_id) {
                                              // Send Migrate event with the raw body
                                              let _ = state.sender.send(StreamEvent::Migrate(body.clone()));
                                              // We remove from map because this adapter is done with it.
                                              // The HttpServer will re-add it to the new adapter.
                                          }
                                      }
                                  }
                             }
                             continue;
                        }

                if meta.action == 2 || meta.action == 3 {
                             // Action 2: GetEngineInfo Response
                             // Action 3: P2P Init Response
                             info!("Received Action {} response. Payload size: {}", meta.action, body.len());

                             let mut map = pending.lock().await;
                             // We use seq_id=0 for control channel responses
                             if let Some(state) = map.remove(&0) {
                                  let _ = state.sender.send(StreamEvent::P2PResponse(body.clone()));
                             }
                             continue;
                        }

                         if let Ok(step_out) = flatbuffers::root::<StepOut>(&body) {
                             let seq_id = step_out.seq_id();
                             let token_id = step_out.token_id();
                             let status = step_out.status();

                             let mut map = pending.lock().await;
                             // Check status logic
                             if status == SequenceStatus::FINISHED {
                                 if let Some(final_state) = map.remove(&seq_id) {
                                     if token_id > 0 {
                                         let _ = final_state.sender.send(StreamEvent::Token(token_id));
                                     }
                                     let _ = final_state.sender.send(StreamEvent::Finished);
                                 }
                             } else if status == SequenceStatus::RUNNING_DECODE {
                                 if token_id > 0 {
                                     if let Some(state) = map.get_mut(&seq_id) {
                                         state.accumulated_tokens.push(token_id);
                                         let _ = state.sender.send(StreamEvent::Token(token_id));
                                     }
                                 }
                             }
                        }
                    }
                    Err(e) => {
                        error!("Reader connection error: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    // New method to send raw payload (for forwarding migration)
    pub async fn send_raw_request(&mut self, seq_id: u64, payload: Vec<u8>) -> anyhow::Result<mpsc::UnboundedReceiver<StreamEvent>> {
        // Register pending request
        let (tx, rx) = mpsc::unbounded_channel();
        {
            let mut map = self.pending_requests.lock().await;
            map.insert(seq_id, RequestState {
                sender: tx,
                accumulated_tokens: Vec::new(),
            });
        }

        info!("Sending Raw Request (Migration Forward) for Seq {}. Payload Size: {}", seq_id, payload.len());

        // Action 1: AddRequest (Same as new request, just with populated slots)
        self.client.send_message(1, seq_id, &payload).await?;

        Ok(rx)
    }

    pub async fn send_add_request(&mut self, seq_id: u64, token_ids: &[u32], max_tokens: i32) -> anyhow::Result<mpsc::UnboundedReceiver<StreamEvent>> {
        // Application layer serialization
        let mut builder = FlatBufferBuilder::new();
        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let t_vec = builder.create_vector(&token_ids_i32);

        // Create SamplingParams table
        let sampling_params = crate::fbs::nanodeploy::sequence::nanodeploy::fbs::SamplingParams::create(
            &mut builder,
            &crate::fbs::nanodeploy::sequence::nanodeploy::fbs::SamplingParamsArgs {
                temperature: 0.1,
                max_tokens,
                ignore_eos: false,
            }
        );

        let num_tokens = token_ids.len() as i32;
        let last_token = if num_tokens > 0 { token_ids_i32[num_tokens as usize - 1] } else { 0 };

        let seq = Sequence::create(&mut builder, &SequenceArgs {
            seq_id,
            status: SequenceStatus::WAITING,
            token_ids: Some(t_vec),
            num_tokens,
            num_prompt_tokens: num_tokens,
            num_checkpointed_tokens: num_tokens,
            last_token,
            sampling_params: Some(sampling_params),
            ..Default::default()
        });

        let seqs = builder.create_vector(&[seq]);
        let root = SequenceList::create(&mut builder, &SequenceListArgs {
            sequences: Some(seqs),
        });

        builder.finish(root, None);
        let payload = builder.finished_data();

        // Register pending request
        let (tx, rx) = mpsc::unbounded_channel();
        {
            let mut map = self.pending_requests.lock().await;
            map.insert(seq_id, RequestState {
                sender: tx,
                accumulated_tokens: Vec::new(),
            });
        }

        // Delegate to Generic Spoke (Action 1)
        self.client.send_message(1, seq_id, payload).await?;

        Ok(rx)
    }

    pub async fn send_get_engine_info(&mut self) -> anyhow::Result<serde_json::Value> {
        // Register pending request for seq_id = 0 (Control Channel)
        let (tx, mut rx) = mpsc::unbounded_channel();
        {
            let mut map = self.pending_requests.lock().await;
            map.insert(0, RequestState {
                sender: tx,
                accumulated_tokens: Vec::new(),
            });
        }

        // Action 2: GetEngineInfo
        self.client.send_message(2, 0, &[]).await?;

        // Wait for response from reader loop
        if let Some(event) = rx.recv().await {
            match event {
                StreamEvent::P2PResponse(body) => {
                     let json: serde_json::Value = serde_json::from_slice(&body)?;
                     Ok(json)
                }
                _ => Err(anyhow::anyhow!("Unexpected response event for GetEngineInfo")),
            }
        } else {
             Err(anyhow::anyhow!("Channel closed while waiting for GetEngineInfo response"))
        }
    }

    pub async fn send_p2p_init(&mut self, nodes: Vec<(String, String, u16, String, i32, i32)>) -> anyhow::Result<Vec<u8>> {
        let mut builder = FlatBufferBuilder::new();

        let mut node_offsets = Vec::new();
        for (id, host, port, role, ws, nb) in nodes {
            let id_off = builder.create_string(&id);
            let role_off = builder.create_string(&role);
            let host_off = builder.create_string(&host);
            let status_off = builder.create_string("ready");

            node_offsets.push(EngineInfo::create(&mut builder, &EngineInfoArgs {
                id: Some(id_off),
                role: Some(role_off),
                host: Some(host_off),
                port: port as i32,
                world_size: ws,
                num_blocks: nb,
                status: Some(status_off),
                ..Default::default()
            }));
        }

        let nodes_vec = builder.create_vector(&node_offsets);
        let p2p_init = P2PInit::create(&mut builder, &P2PInitArgs {
            nodes: Some(nodes_vec),
        });

        builder.finish(p2p_init, None);
        let payload = builder.finished_data().to_vec();

        // Register pending request for seq_id = 0 (Control Channel)
        let (tx, mut rx) = mpsc::unbounded_channel();
        {
            let mut map = self.pending_requests.lock().await;
            map.insert(0, RequestState {
                sender: tx,
                accumulated_tokens: Vec::new(),
            });
        }

        // Action 3: P2PInit
        self.client.send_message(3, 0, &payload).await?;

        // Wait for response from reader loop
        if let Some(event) = rx.recv().await {
            match event {
                StreamEvent::P2PResponse(body) => {
                     // Response is now P2PInitResponse (FB)
                     Ok(body)
                }
                _ => Err(anyhow::anyhow!("Unexpected response event for P2P Init")),
            }
        } else {
             Err(anyhow::anyhow!("Channel closed while waiting for P2P Init response"))
        }
    }

    pub async fn send_p2p_connect(&mut self, target_map: HashMap<String, Vec<u8>>) -> anyhow::Result<()> {
        use crate::fbs::nanodeploy::connection::nanodeploy::fbs::{P2PConnect, P2PConnectArgs, Peer, PeerArgs};

        let mut builder = FlatBufferBuilder::new();
        let mut peer_offsets = Vec::new();

        for (target_id, remote_info_bytes) in target_map {
            let id_off = builder.create_string(&target_id);
            let remote_info_off = builder.create_vector(&remote_info_bytes);

            peer_offsets.push(Peer::create(&mut builder, &PeerArgs {
                id: Some(id_off),
                remote_info: Some(remote_info_off),
                ..Default::default()
            }));
        }

        let peers_vec = builder.create_vector(&peer_offsets);
        let p2p_connect = P2PConnect::create(&mut builder, &P2PConnectArgs {
            peers: Some(peers_vec),
        });

        builder.finish(p2p_connect, None);
        let payload = builder.finished_data();

        // Action 4: P2PConnect
        self.client.send_message(4, 0, payload).await?;
        Ok(())
    }
}
