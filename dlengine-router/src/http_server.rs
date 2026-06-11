use crate::engine_adapter::StreamEvent;
use crate::engine_manager::{EngineManager, ModelPool};
use crate::tokenizer::TokenizerService;
use crate::tool_parser;
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

// ── Tool / Function Call Types (OpenAI-compatible) ──────────────────

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String, // always "function"
    pub function: FunctionDefinition,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String, // JSON-encoded string (OpenAI spec)
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String, // always "function"
    pub function: FunctionCall,
}

// ── Streaming delta types ────────────────────────────────────────────

#[derive(Serialize, Debug)]
struct DeltaFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    arguments: String,
}

#[derive(Serialize, Debug)]
struct DeltaToolCall {
    index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    call_type: Option<String>,
    function: DeltaFunctionCall,
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
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    #[allow(dead_code)]
    pub tool_choice: Option<serde_json::Value>, // "auto"|"none"|"required"|named obj
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
            .field(
                "tools",
                &self.tools.as_ref().map(|t| format!("[{} tools]", t.len())),
            )
            .finish()
    }
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>, // assistant → tool_calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>, // role="tool" responses
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
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

/// Message passed to the Jinja template renderer.
#[derive(Serialize)]
struct TemplateMessage {
    role: String,
    content: serde_json::Value, // String or Null (for assistant-with-tool-calls)
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<serde_json::Value>>, // [{name, arguments(dict)}]
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

/// Response message for the assistant role.
#[derive(Serialize, Debug)]
pub struct AssistantMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Convert a slice of incoming `Message`s into `TemplateMessage`s for rendering.
fn build_template_messages(messages: &[Message]) -> Vec<TemplateMessage> {
    messages
        .iter()
        .map(|m| {
            let content = match m.role.as_str() {
                "assistant" if m.tool_calls.is_some() => serde_json::Value::Null,
                _ => serde_json::Value::String(m.content.text()),
            };

            // tool_calls: [{name, arguments as dict}] — Qwen template expects a dict,
            // not a JSON-encoded string.
            let tool_calls = m.tool_calls.as_ref().map(|tcs| {
                tcs.iter()
                    .map(|tc| {
                        let args: serde_json::Value =
                            serde_json::from_str(&tc.function.arguments).unwrap_or_else(|e| {
                                tracing::warn!(
                                    "Failed to parse tool call arguments as JSON: {}. Arguments: '{}'",
                                    e,
                                    &tc.function.arguments
                                );
                                serde_json::Value::Object(serde_json::Map::new())
                            });
                        serde_json::json!({
                            "name": tc.function.name,
                            "arguments": args,
                        })
                    })
                    .collect::<Vec<_>>()
            });

            TemplateMessage {
                role: m.role.clone(),
                content,
                tool_calls,
                tool_call_id: m.tool_call_id.clone(),
                name: m.name.clone(),
            }
        })
        .collect()
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
    pub message: AssistantMessage,
    pub finish_reason: String,
}

// App State
pub struct AppState {
    pub engine_manager: Arc<Mutex<EngineManager>>,
    pub next_request_id: AtomicU64,
}

// ── Pre-flight helpers ───────────────────────────────────────────────

async fn resolve_tokenizer(
    slot: &RwLock<Option<Arc<TokenizerService>>>,
) -> Result<Arc<TokenizerService>, Response> {
    match slot.read().await.as_ref() {
        Some(t) => Ok(t.clone()),
        None => Err((StatusCode::SERVICE_UNAVAILABLE, "Tokenizer not ready").into_response()),
    }
}

fn check_engine_availability(pool: &ModelPool, has_images: bool) -> Option<Response> {
    if pool.get_next_prefill().is_none() {
        return Some(
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No prefill engines available",
            )
                .into_response(),
        );
    }
    if has_images && pool.get_next_encoder().is_none() {
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

    let model_key = req.model.trim_end_matches('/').to_string();
    let has_images = req.messages.iter().any(|m| m.content.has_images());

    // Single manager lock: model lookup + engine check + Arc clones, then release
    let (tokenizer_slot, prefill_adapter, encoder_adapter) = {
        let mgr = state.engine_manager.lock().await;
        let pool = match mgr.model_pools.get(&model_key) {
            Some(p) => p,
            None => {
                return (
                    StatusCode::NOT_FOUND,
                    format!(
                        "Model '{}' not found. Available: [{}]",
                        req.model,
                        mgr.available_model_keys().join(", ")
                    ),
                )
                    .into_response()
            }
        };
        if let Some(err) = check_engine_availability(pool, has_images) {
            return err;
        }
        let prefill = pool.get_next_prefill().unwrap(); // safe: checked above
        let encoder = if has_images {
            Some(pool.get_next_encoder().unwrap()) // safe: checked above
        } else {
            None
        };
        (pool.tokenizer_slot.clone(), prefill, encoder)
    }; // manager lock released here

    // Tokenizer check (no manager lock held)
    let tokenizer = match resolve_tokenizer(&tokenizer_slot).await {
        Ok(t) => t,
        Err(e) => return e,
    };

    // Generate unique sequence ID and request ID
    let seq_id = state.next_request_id.fetch_add(1, Ordering::SeqCst);
    let request_id = format!("chatcmpl-{}", seq_id);
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Acquire lock briefly to send request
    let rx_result = {
        let mut adapter_guard = prefill_adapter.lock().await;

        if has_images {
            // ── Multimodal path: send to EncoderEngine first ──
            // pre-flight confirmed encoder exists; encoder_adapter is Some
            let encoder = encoder_adapter.unwrap();

            // Build encode request JSON (forward original messages)
            let encode_req = serde_json::json!({ "messages": &req.messages });
            let encode_json = serde_json::to_vec(&encode_req).unwrap();

            let encode_resp = {
                let mut enc_guard = encoder.lock().await;
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
            let template_messages = build_template_messages(&req.messages);
            let tools_json = req.tools.as_ref().map(|t| {
                serde_json::to_value(t).unwrap_or_else(|e| {
                    tracing::error!("Failed to serialize tools to JSON: {}", e);
                    serde_json::Value::Null
                })
            });
            let token_ids = match tokenizer
                .encode_messages(template_messages, tools_json)
                .await
            {
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
        let mut active_adapter = prefill_adapter.clone();

        // Use async-stream macros for clean generator syntax
        let stream = async_stream::stream! {
            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut last_text_len = 0;
            let stream_start = std::time::Instant::now();
            let mut is_finished = false;
            // Tool call streaming state
            let mut tool_suppress_from: Option<usize> = None; // byte offset in full_text where suppression started
            let mut emitted_tool_calls: u32 = 0;

            while let Some(event) = rx.recv().await {
                match event {
                    StreamEvent::Token(id) => {
                        // Engine only sends generated tokens (token_ids[-1] per step),
                        // never prompt echoes.  Every token here is real output.
                        generated_tokens.push(id);
                        // Incremental decoding: decode all generated tokens and take diff
                        if let Ok(full_text) = tokenizer.decode(generated_tokens.clone()).await {
                            let new_len = full_text.len();

                            // Detect start of a <tool_call> block not yet suppressed.
                            if tool_suppress_from.is_none() {
                                if let Some(tc_rel) = full_text[last_text_len..].find("<tool_call>") {
                                    let tc_start_abs = last_text_len + tc_rel;
                                    // Emit any text before the tool call starts.
                                    if tc_start_abs > last_text_len {
                                        if let Some(delta_str) = full_text.get(last_text_len..tc_start_abs) {
                                            let delta = delta_str.to_string();
                                            let chunk = serde_json::json!({
                                                "id": request_id,
                                                "object": "chat.completion.chunk",
                                                "created": created_at,
                                                "model": model_name,
                                                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": null}]
                                            });
                                            yield Ok::<_, std::io::Error>(Event::default().data(chunk.to_string()));
                                        }
                                    }
                                    last_text_len = tc_start_abs;
                                    tool_suppress_from = Some(tc_start_abs);
                                }
                            }

                            if let Some(suppress_from) = tool_suppress_from {
                                // In tool-call mode: check whether a complete block has arrived.
                                let tool_text = &full_text[suppress_from..];
                                if let Some(end_pos) = tool_parser::find_complete_tool_call_end(tool_text) {
                                    let block = &full_text[suppress_from..suppress_from + end_pos];
                                    let (_, calls) = tool_parser::parse_tool_calls(block);
                                    for tc in calls {
                                        let tc_id = format!("call_{}_{}", seq_id, emitted_tool_calls);
                                        let delta_call = DeltaToolCall {
                                            index: emitted_tool_calls,
                                            id: Some(tc_id),
                                            call_type: Some("function".to_string()),
                                            function: DeltaFunctionCall {
                                                name: Some(tc.name),
                                                arguments: serde_json::to_string(&tc.arguments)
                                                    .unwrap_or_else(|e| {
                                                        tracing::warn!("Failed to serialize tool arguments to JSON string in streaming: {}. Defaulting to empty object.", e);
                                                        "{}".to_string()
                                                    }),
                                            },
                                        };
                                        let delta_call_value = match serde_json::to_value(&delta_call) {
                                            Ok(v) => v,
                                            Err(e) => {
                                                tracing::error!("Failed to serialize DeltaToolCall: {}. Skipping tool call delta.", e);
                                                continue;
                                            }
                                        };
                                        let chunk = serde_json::json!({
                                            "id": request_id,
                                            "object": "chat.completion.chunk",
                                            "created": created_at,
                                            "model": model_name,
                                            "choices": [{"index": 0, "delta": {"tool_calls": [delta_call_value]}, "finish_reason": null}]
                                        });
                                        yield Ok::<_, std::io::Error>(Event::default().data(chunk.to_string()));
                                        emitted_tool_calls += 1;
                                    }
                                    last_text_len = suppress_from + end_pos;
                                    tool_suppress_from = None;
                                }
                                // else: still accumulating inside a tool call — suppress text.
                            } else {
                                // Normal incremental text emission.
                                if new_len > last_text_len {
                                    // Use safe slicing to handle UTF-8 character boundaries
                                    if let Some(delta_str) = full_text.get(last_text_len..) {
                                        let delta = delta_str.to_string();
                                        last_text_len = new_len;
                                        let chunk = serde_json::json!({
                                            "id": request_id,
                                            "object": "chat.completion.chunk",
                                            "created": created_at,
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
                        }
                    },
                    StreamEvent::Finished => {
                        is_finished = true;
                        let finish_reason = if emitted_tool_calls > 0 { "tool_calls" } else { "stop" };
                        let chunk = serde_json::json!({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_at,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason
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
                        tool_suppress_from = None;
                        emitted_tool_calls = 0;

                        let decode_adapter_arc = {
                            let mgr = state.engine_manager.lock().await;
                            mgr.model_pools.get(&model_key).and_then(|p| p.get_next_decode())
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
                        mgr.model_pools
                            .get(&model_key)
                            .and_then(|p| p.get_next_decode())
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
        let raw_text = tokenizer.decode(generated_tokens).await.unwrap_or_default();
        let (content, tool_calls) = tool_parser::parse_tool_calls(&raw_text);
        let (assistant_msg, finish_reason) = if tool_calls.is_empty() {
            (
                AssistantMessage {
                    role: "assistant".to_string(),
                    content: Some(content),
                    tool_calls: None,
                },
                "stop".to_string(),
            )
        } else {
            let oa_calls = tool_calls
                .into_iter()
                .enumerate()
                .map(|(i, tc)| ToolCall {
                    id: format!("call_{}_{}", seq_id, i),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: tc.name,
                        arguments: serde_json::to_string(&tc.arguments).unwrap_or_else(|e| {
                            tracing::warn!(
                                "Failed to serialize tool arguments to JSON string: {}. Defaulting to empty object.",
                                e
                            );
                            "{}".to_string()
                        }),
                    },
                })
                .collect();
            (
                AssistantMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(oa_calls),
                },
                "tool_calls".to_string(),
            )
        };
        Json(ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created: created_at,
            model: req.model,
            choices: vec![Choice {
                index: 0,
                message: assistant_msg,
                finish_reason,
            }],
        })
        .into_response()
    }
}

async fn health() -> &'static str {
    "OK"
}

pub async fn start_server(port: u16, engine_manager: Arc<Mutex<EngineManager>>) {
    // Use timestamp as start ID to avoid collisions on server restart
    // Must fit in u32 for legacy engine protocol
    let start_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs(); // u64, but guarantees < u32::MAX until 2106

    let state = Arc::new(AppState {
        engine_manager,
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
