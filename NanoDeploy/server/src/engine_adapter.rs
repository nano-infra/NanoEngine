use crate::fbs::nanodeploy::fbs::nanodeploy::fbs::{SequenceList, SequenceListArgs, Sequence, SequenceArgs, SequenceStatus, StepOut};
use flatbuffers::FlatBufferBuilder;
use spoke::client::SpokeClient;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tracing::{info, error};

pub struct EngineAdapter {
    pub client: SpokeClient,
    pub pending_requests: Arc<Mutex<HashMap<u64, RequestState>>>,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(u32),
    Finished,
    Error(String),
}

pub struct RequestState {
    pub sender: mpsc::UnboundedSender<StreamEvent>,
    pub accumulated_tokens: Vec<u32>, // Keep for potential recovery/debug, though streaming removes need
}

impl EngineAdapter {
    pub fn new(id: String) -> Self {
        Self {
            client: SpokeClient::new(id),
            pending_requests: Arc::new(Mutex::new(HashMap::new())),
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
                    Ok((_meta, body)) => {
                         if let Ok(step_out) = flatbuffers::root::<StepOut>(&body) {
                             let seq_id = step_out.seq_id();
                             let token_id = step_out.token_id();
                             let status = step_out.status();

                             let mut map = pending.lock().await;
                             // Check status logic
                             if status == SequenceStatus::FINISHED {
                                 if let Some(final_state) = map.remove(&seq_id) {
                                     // Send last token if any
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

    pub async fn send_add_request(&mut self, seq_id: u64, token_ids: &[u32], max_tokens: i32) -> anyhow::Result<mpsc::UnboundedReceiver<StreamEvent>> {
        // Application layer serialization
        let mut builder = FlatBufferBuilder::new();
        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let t_vec = builder.create_vector(&token_ids_i32);

        // Create SamplingParams table
        let sampling_params = crate::fbs::nanodeploy::fbs::nanodeploy::fbs::SamplingParams::create(
            &mut builder,
            &crate::fbs::nanodeploy::fbs::nanodeploy::fbs::SamplingParamsArgs {
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

        // Delegate to Generic Spoke
        self.client.send_message(1, seq_id, payload).await?;

        Ok(rx)
    }
}
