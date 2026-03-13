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
use tokio::sync::RwLock;
use tower_http::trace::TraceLayer;

// ── Multimodal Content Types (OpenAI-compatible) ────────────────────

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ImageUrlValue {
    pub url: String,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlValue },
}

/// Message content: either a plain string or a list of content parts
/// (text + image_url) following the OpenAI multimodal API.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract only the text content, joining all text parts.
    /// Image parts are silently ignored (handled by VLEngineServer).
    pub fn text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Whether this content contains any image parts.
    pub fn has_images(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Parts(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. })),
        }
    }
}

// Request Payload (OpenAI-compatible)
#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub ignore_eos: Option<bool>,
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
            .field("temperature", &self.temperature)
            .field("ignore_eos", &self.ignore_eos)
            .finish()
    }
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

impl fmt::Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = self.content.text();
        let truncated = if text.len() > 50 {
            let words: Vec<&str> = text.split_whitespace().take(8).collect();
            format!("{}...", words.join(" "))
        } else {
            text
        };
        f.debug_struct("Message")
            .field("role", &self.role)
            .field("content", &truncated)
            .finish()
    }
}

/// Simplified message for template rendering (always text content).
#[derive(Serialize)]
struct TemplateMessage {
    role: String,
    content: String,
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
    pub tokenizer: Arc<RwLock<Option<Arc<TokenizerService>>>>,
    pub next_request_id: AtomicU64,
}

// ── Pre-flight helpers ───────────────────────────────────────────────

fn check_model(req_model: &str, served_model_dir: &str) -> Option<Response> {
    // Normalize trailing slashes so "/models/Foo/" and "/models/Foo" both match.
    let req = req_model.trim_end_matches('/');
    let served = served_model_dir.trim_end_matches('/');
    if req != served {
        Some(
            (
                StatusCode::NOT_FOUND,
                format!(
                    "Model '{}' not found. This router serves '{}'.",
                    req_model, served_model_dir
                ),
            )
                .into_response(),
        )
    } else {
        None
    }
}

async fn resolve_tokenizer(
    slot: &RwLock<Option<Arc<TokenizerService>>>,
) -> Result<Arc<TokenizerService>, Response> {
    match slot.read().await.as_ref() {
        Some(t) => Ok(t.clone()),
        None => Err((StatusCode::SERVICE_UNAVAILABLE, "Tokenizer not ready").into_response()),
    }
}

async fn check_engine_availability(mgr: &EngineManager, has_images: bool) -> Option<Response> {
    if mgr.get_next_prefill().is_none() {
        return Some(
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No prefill engines available",
            )
                .into_response(),
        );
    }
    if has_images && mgr.get_next_encoder().is_none() {
        return Some(
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No encoder engines available for multimodal request",
            )
                .into_response(),
        );
    }
    None
}

// Handler
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    tracing::info!("Received request: {:?}", req);

    // 1. Tokenizer (lazy — returns 503 until an engine connects and loads it)
    let tokenizer = match resolve_tokenizer(&state.tokenizer).await {
        Ok(t) => t,
        Err(e) => return e,
    };

    // 2. Model check against the directory registered by the engine
    if let Some(err) = check_model(&req.model, tokenizer.model_dir()) {
        return err;
    }

    // 3. Engine availability pre-flight
    let has_images = req.messages.iter().any(|m| m.content.has_images());
    {
        let mgr = state.engine_manager.lock().await;
        if let Some(err) = check_engine_availability(&mgr, has_images).await {
            return err;
        }
    }

    // Generate unique sequence ID and request ID
    let seq_id = state.next_request_id.fetch_add(1, Ordering::SeqCst);
    let request_id = format!("chatcmpl-{}", seq_id);

    // Get an available engine from manager (pre-flight confirmed it exists)
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

        if has_images {
            // ── Multimodal path: send to EncoderEngine first ──
            // pre-flight confirmed encoder exists; keep match for TOCTOU safety
            let encoder_adapter = {
                let mgr = state.engine_manager.lock().await;
                mgr.get_next_encoder()
            };
            let encoder_adapter: Arc<Mutex<crate::encoder_adapter::EncoderAdapter>> =
                match encoder_adapter {
                    Some(a) => a,
                    None => {
                        return (
                            StatusCode::SERVICE_UNAVAILABLE,
                            "No encoder engines available for multimodal request",
                        )
                            .into_response()
                    }
                };

            // Build encode request JSON (forward original messages)
            let encode_req = serde_json::json!({ "messages": &req.messages });
            let encode_json = serde_json::to_vec(&encode_req).unwrap();

            let encode_resp = {
                let mut enc_guard = encoder_adapter.lock().await;
                enc_guard.encode(&encode_json).await
            };
            let encode_resp = match encode_resp {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!("Encoder error: {}", e);
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Encoder error: {}", e),
                    )
                        .into_response();
                }
            };

            let token_ids: Vec<u32> = encode_resp.input_ids.iter().map(|&x| x as u32).collect();
            let max_tokens = req.effective_max_tokens(16) as i32;

            tracing::info!(
                "Encoder returned {} tokens, {} vision_slots for seq_id={}",
                token_ids.len(),
                encode_resp.vision_slots.len(),
                seq_id
            );

            let vision_slots = if encode_resp.vision_slots.is_empty() {
                None
            } else {
                Some(encode_resp.vision_slots)
            };

            adapter_guard
                .send_add_request_with_vision(
                    seq_id,
                    &token_ids,
                    max_tokens,
                    req.temperature.unwrap_or(0.1),
                    req.ignore_eos.unwrap_or(false),
                    vision_slots.as_deref(),
                )
                .await
        } else {
            // ── Text-only path: tokenize locally ──
            let template_messages: Vec<TemplateMessage> = req
                .messages
                .iter()
                .map(|m| TemplateMessage {
                    role: m.role.clone(),
                    content: m.content.text(),
                })
                .collect();
            let token_ids = match tokenizer.encode_messages(template_messages).await {
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
                .send_add_request(
                    seq_id,
                    &token_ids,
                    max_tokens,
                    req.temperature.unwrap_or(0.1),
                    req.ignore_eos.unwrap_or(false),
                )
                .await
        }
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
        let mut active_adapter = adapter.clone();

        // Use async-stream macros for clean generator syntax
        let stream = async_stream::stream! {
            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut last_text_len = 0;
            let stream_start = std::time::Instant::now();
            let mut is_finished = false;

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
                        is_finished = true;
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
                        tracing::error!("SSE stream error for seq_id={}: {}", seq_id, e);
                         yield Ok(Event::default().event("error").data(e));
                         break;
                    }
                    StreamEvent::Migrate(payload) => {
                        // Reset decode state for new engine — any tokens already
                        // streamed came from the previous engine; the new decode
                        // engine will continue generating fresh tokens.
                        generated_tokens.clear();
                        last_text_len = 0;

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

                                      // Update active tracked adapter to route the disconnect signal to the correct place
                                      active_adapter = decode_adapter_arc.clone();

                                 },
                                 Err(e) => {
                                     tracing::error!("Failed to forward migration for seq_id={}: {}", seq_id, e);
                                     yield Ok(Event::default().event("error").data("Migration Failed"));
                                     break;
                                 }
                             }
                        } else {
                             tracing::error!("No Decode Engine available for migration, seq_id={}", seq_id);
                             yield Ok(Event::default().event("error").data("No Decode Nodes"));
                             break;
                        }
                    }
                }
            }

            // If we reach here via rx channel closing (None), the client likely disconnected
            if !is_finished {
                 tracing::warn!("SSE stream disconnected for seq_id={}, elapsed={:.1}s, tokens={}. Sending FREE request.", seq_id, stream_start.elapsed().as_secs_f64(), generated_tokens.len());
                 let mut adapter_guard = active_adapter.lock().await;
                 let _ = adapter_guard.send_free_request(seq_id).await;
            } else {
                 tracing::info!("SSE stream ended for seq_id={}, elapsed={:.1}s, tokens={}", seq_id, stream_start.elapsed().as_secs_f64(), generated_tokens.len());
            }
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
                    content: MessageContent::Text(text),
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
    tokenizer: Arc<RwLock<Option<Arc<TokenizerService>>>>,
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
