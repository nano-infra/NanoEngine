use crate::engine_adapter::StreamEvent;
use crate::engine_manager::EngineManager;
use crate::tokenizer::TokenizerService;
use axum::http::StatusCode;
use axum::response::{sse::Event, IntoResponse, Response, Sse};
use axum::{
    extract::{Json, State},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::trace::TraceLayer;

// Request Payload (OpenAI-compatible)
#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub stream: Option<bool>,
}

// Custom Debug implementation: truncate long messages to first few words
impl ChatCompletionRequest {
    /// Resolve effective max_tokens: max_completion_tokens takes precedence (newer OpenAI field),
    /// falls back to max_tokens (legacy field), then default.
    pub fn effective_max_tokens(&self, default: u32) -> u32 {
        self.max_completion_tokens
            .or(self.max_tokens)
            .unwrap_or(default)
    }
}

impl fmt::Debug for ChatCompletionRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatCompletionRequest")
            .field("model", &self.model)
            .field("messages", &self.messages)
            .field("max_tokens", &self.max_tokens)
            .field("max_completion_tokens", &self.max_completion_tokens)
            .field("stream", &self.stream)
            .finish()
    }
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

// Custom Debug implementation: truncate long content to first 8 words
impl fmt::Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let truncated = if self.content.len() > 50 {
            let words: Vec<&str> = self.content.split_whitespace().take(8).collect();
            format!("{}...", words.join(" "))
        } else {
            self.content.clone()
        };

        f.debug_struct("Message")
            .field("role", &self.role)
            .field("content", &truncated)
            .finish()
    }
}

// Response Payload (Simplified)
#[derive(Serialize, Debug)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
}

#[derive(Serialize, Debug)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: String,
}

// App State
pub struct AppState {
    pub engine_manager: Arc<Mutex<EngineManager>>,
    pub tokenizer: Arc<TokenizerService>,
    pub next_request_id: AtomicU64,
}

// Handler
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    tracing::info!("Received request: {:?}", req);

    let tokenizer = state.tokenizer.clone();

    // Generate unique sequence ID and request ID
    let seq_id = state.next_request_id.fetch_add(1, Ordering::SeqCst);
    let request_id = format!("chatcmpl-{}", seq_id);

    // Get an available engine from manager
    let adapter = {
        let mgr = state.engine_manager.lock().await;
        match mgr.get_next_prefill() {
            Some(a) => a,
            None => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "No prefill engines available",
                )
                    .into_response()
            }
        }
    };

    // Acquire lock briefly to send request
    let rx_result = {
        let mut adapter_guard = adapter.lock().await;

        // Use chat template encoding
        let token_ids = match tokenizer.encode_messages(req.messages.clone()).await {
            Ok(ids) => ids,
            Err(e) => {
                tracing::error!("Encoding error: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Encoding error: {}", e),
                )
                    .into_response();
            }
        };

        let max_tokens = req.effective_max_tokens(16) as i32;
        adapter_guard
            .send_add_request(seq_id, &token_ids, max_tokens)
            .await
    };

    let mut rx = match rx_result {
        Ok(rx) => rx,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Engine error: {}", e),
            )
                .into_response()
        }
    };

    if req.stream.unwrap_or(false) {
        let model_name = req.model.clone();
        let tokenizer = tokenizer.clone();
        let request_id = request_id.clone();

        // Use async-stream macros for clean generator syntax
        let stream = async_stream::stream! {
            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut last_text_len = 0;
            let stream_start = std::time::Instant::now();

            tracing::info!("[DIAG] SSE stream STARTED for seq_id={}", seq_id);

            while let Some(event) = rx.recv().await {
                match event {
                    StreamEvent::Token(id) => {
                        // Engine only sends generated tokens (token_ids[-1] per step),
                        // never prompt echoes.  Every token here is real output.
                        generated_tokens.push(id);
                        // Incremental decoding: decode all generated tokens and take diff
                        if let Ok(full_text) = tokenizer.decode(generated_tokens.clone()).await {
                             let new_len = full_text.len();
                             if new_len > last_text_len {
                                 // Use safe slicing to handle UTF-8 character boundaries
                                 if let Some(delta_str) = full_text.get(last_text_len..) {
                                     let delta = delta_str.to_string();
                                     last_text_len = new_len;

                                     let chunk = serde_json::json!({
                                         "id": request_id,
                                         "object": "chat.completion.chunk",
                                         "created": 1234567890,
                                         "model": model_name,
                                         "choices": [{
                                             "index": 0,
                                             "delta": { "content": delta },
                                             "finish_reason": null
                                         }]
                                     });
                                     yield Ok::<_, std::io::Error>(Event::default().data(chunk.to_string()));
                                 }
                             }
                        }
                    },
                    StreamEvent::Finished => {
                        tracing::info!("[DIAG] SSE stream FINISHED normally for seq_id={}, elapsed={:.1}s, generated_tokens={}", seq_id, stream_start.elapsed().as_secs_f64(), generated_tokens.len());
                        let chunk = serde_json::json!({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": 1234567890,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        });
                        yield Ok(Event::default().data(chunk.to_string()));
                        yield Ok(Event::default().data("[DONE]"));
                        break;
                    },
                    StreamEvent::Error(e) => {
                        tracing::error!("[DIAG] SSE stream ERROR for seq_id={}: {}", seq_id, e);
                         yield Ok(Event::default().event("error").data(e));
                         break;
                    }
                    StreamEvent::Migrate(payload) => {
                        // Reset decode state for new engine — any tokens already
                        // streamed came from the previous engine; the new decode
                        // engine will continue generating fresh tokens.
                        generated_tokens.clear();
                        last_text_len = 0;

                        tracing::info!("[DIAG] Migration triggered for seq_id={}, elapsed={:.1}s. Routing to Decode Engine...", seq_id, stream_start.elapsed().as_secs_f64());
                        let decode_adapter_arc = {
                            let mgr = state.engine_manager.lock().await;
                            mgr.get_next_decode()
                        };

                        if let Some(decode_adapter_arc) = decode_adapter_arc {
                             let mut decode_adapter = decode_adapter_arc.lock().await;
                             match decode_adapter.send_raw_request(seq_id, payload).await {
                                 Ok(new_rx) => {
                                      // SWAP RX channel transparently
                                      rx = new_rx;
                                      tracing::info!("[DIAG] Migration successful for seq_id={}. Resuming stream on Decode Engine.", seq_id);
                                 },
                                 Err(e) => {
                                     tracing::error!("[DIAG] Failed to forward migration for seq_id={}: {}", seq_id, e);
                                     yield Ok(Event::default().event("error").data("Migration Failed"));
                                     break;
                                 }
                             }
                        } else {
                             tracing::error!("[DIAG] No Decode Engine available for migration! seq_id={}", seq_id);
                             yield Ok(Event::default().event("error").data("No Decode Nodes"));
                             break;
                        }
                    }
                }
            }

            // If we reach here via rx channel closing (None), the client likely disconnected
            tracing::warn!("[DIAG] SSE stream ENDED for seq_id={}, elapsed={:.1}s, generated_tokens={}", seq_id, stream_start.elapsed().as_secs_f64(), generated_tokens.len());
        };

        Sse::new(stream).into_response()
    } else {
        // Non-streaming: accumulate all generated tokens
        let mut generated_tokens: Vec<u32> = Vec::new();
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(id) => generated_tokens.push(id),
                StreamEvent::Finished => break,
                StreamEvent::Error(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Engine error: {}", e),
                    )
                        .into_response()
                }
                StreamEvent::Migrate(payload) => {
                    tracing::info!("Migration (Non-Streaming)...");
                    generated_tokens.clear();
                    let decode_adapter_arc = {
                        let mgr = state.engine_manager.lock().await;
                        mgr.get_next_decode()
                    };
                    if let Some(decode_adapter_arc) = decode_adapter_arc {
                        let mut decode_adapter = decode_adapter_arc.lock().await;
                        if let Ok(new_rx) = decode_adapter.send_raw_request(seq_id, payload).await {
                            rx = new_rx;
                        } else {
                            return (StatusCode::INTERNAL_SERVER_ERROR, "Migration Failed")
                                .into_response();
                        }
                    } else {
                        return (StatusCode::SERVICE_UNAVAILABLE, "No Decode Nodes")
                            .into_response();
                    }
                }
            }
        }

        // Engine only sends generated tokens, no prompt echo to skip
        let text = tokenizer.decode(generated_tokens).await.unwrap_or_default();
        Json(ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: req.model,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: text,
                },
                finish_reason: "stop".to_string(),
            }],
        })
        .into_response()
    }
}

async fn health() -> &'static str {
    "OK"
}

pub async fn start_server(
    port: u16,
    engine_manager: Arc<Mutex<EngineManager>>,
    tokenizer: Arc<TokenizerService>,
) {
    // Use timestamp as start ID to avoid collisions on server restart
    // Must fit in u32 for legacy engine protocol
    let start_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs(); // u64, but guarantees < u32::MAX until 2106

    let state = Arc::new(AppState {
        engine_manager,
        tokenizer,
        next_request_id: AtomicU64::new(start_id),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
