use axum::{
    extract::{State, Json},
    routing::{post, get},
    Router,
};
use axum::response::{Sse, sse::Event, IntoResponse, Response};
use axum::http::StatusCode;
use tower_http::trace::TraceLayer;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::engine_manager::EngineManager;
use crate::engine_adapter::{EngineAdapter, StreamEvent};
use crate::tokenizer::TokenizerService;
use std::sync::atomic::{AtomicU64, Ordering};

// Request Payload (Simplified OpenAI)
#[derive(Deserialize, Debug)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: Option<u32>,
    pub stream: Option<bool>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
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

    // Generate unique sequence ID
    let seq_id = state.next_request_id.fetch_add(1, Ordering::SeqCst);

    // Get an available engine from manager
    let adapter = {
        let mgr = state.engine_manager.lock().await;
        match mgr.get_next_prefill() {
            Some(a) => a,
            None => return (StatusCode::SERVICE_UNAVAILABLE, "No prefill engines available").into_response(),
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
                 return (StatusCode::INTERNAL_SERVER_ERROR, format!("Encoding error: {}", e)).into_response();
            }
        };

        let max_tokens = req.max_tokens.unwrap_or(16) as i32;
        adapter_guard.send_add_request(seq_id, &token_ids, max_tokens).await
    };

    let mut rx = match rx_result {
        Ok(rx) => rx,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Engine error: {}", e)).into_response(),
    };

    if req.stream.unwrap_or(false) {
        let model_name = req.model.clone();
        let tokenizer = tokenizer.clone();

        // Use async-stream macros for clean generator syntax
        let stream = async_stream::stream! {
            let mut all_tokens = Vec::new();
            let mut last_text_len = 0;

            while let Some(event) = rx.recv().await {
                match event {
                    StreamEvent::Token(id) => {
                        all_tokens.push(id);
                        // Incremental decoding: decode all and take diff
                        if let Ok(full_text) = tokenizer.decode(all_tokens.clone()).await {
                             let new_len = full_text.len();
                             if new_len > last_text_len {
                                 let delta = full_text[last_text_len..].to_string();
                                 last_text_len = new_len;

                                 let chunk = serde_json::json!({
                                     "id": "chatcmpl-stream",
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
                    },
                    StreamEvent::Finished => {
                        let chunk = serde_json::json!({
                            "id": "chatcmpl-stream",
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
                        tracing::error!("Stream error: {}", e);
                         yield Ok(Event::default().event("error").data(e));
                         break;
                    }
                    StreamEvent::Migrate(payload) => {
                        tracing::info!("Migration triggered. Routing to Decode Engine...");
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
                                      tracing::info!("Migration successful. Resuming stream on Decode Engine.");
                                 },
                                 Err(e) => {
                                     tracing::error!("Failed to forward migration: {}", e);
                                     yield Ok(Event::default().event("error").data("Migration Failed"));
                                     break;
                                 }
                             }
                        } else {
                             tracing::error!("No Decode Engine available for migration!");
                             yield Ok(Event::default().event("error").data("No Decode Nodes"));
                             break;
                        }
                    }
                    StreamEvent::P2PResponse(_) => {
                         tracing::error!("Received unexpected P2PResponse in streaming chat request");
                    }
                }
            }
        };

        Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default()).into_response()
    } else {
        // Non-streaming: accumulate
        let mut all_tokens = Vec::new();
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(id) => all_tokens.push(id),
                StreamEvent::Finished => break,
                StreamEvent::Error(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Engine error: {}", e)).into_response(),
                StreamEvent::Migrate(payload) => {
                    tracing::info!("Migration (Non-Streaming)...");
                    let decode_adapter_arc = {
                        let mgr = state.engine_manager.lock().await;
                        mgr.get_next_decode()
                    };
                    if let Some(decode_adapter_arc) = decode_adapter_arc {
                         let mut decode_adapter = decode_adapter_arc.lock().await;
                         if let Ok(new_rx) = decode_adapter.send_raw_request(seq_id, payload).await {
                             rx = new_rx;
                         } else {
                             return (StatusCode::INTERNAL_SERVER_ERROR, "Migration Failed").into_response();
                         }
                    } else {
                         return (StatusCode::SERVICE_UNAVAILABLE, "No Decode Nodes").into_response();
                    }
                }
                StreamEvent::P2PResponse(_) => {
                    tracing::error!("Received unexpected P2PResponse in chat request");
                }
            }
        }

        let text = tokenizer.decode(all_tokens).await.unwrap_or_default();
         Json(ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
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
        }).into_response()
    }
}

async fn health() -> &'static str {
    "OK"
}

pub async fn start_server(
    port: u16,
    engine_manager: Arc<Mutex<EngineManager>>,
    tokenizer: Arc<TokenizerService>
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
