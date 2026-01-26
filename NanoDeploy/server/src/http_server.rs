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
    pub engine_adapter: Arc<Mutex<EngineAdapter>>,
    pub tokenizer: Arc<TokenizerService>,
    pub next_request_id: AtomicU64,
}

// Handler
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    tracing::info!("Received request: {:?}", req);

    let (adapter, tokenizer) = (state.engine_adapter.clone(), state.tokenizer.clone());

    // Generate unique sequence ID
    let seq_id = state.next_request_id.fetch_add(1, Ordering::SeqCst);

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
    engine_adapter: Arc<Mutex<EngineAdapter>>,
    tokenizer: Arc<TokenizerService>
) {
    let state = Arc::new(AppState {
        engine_manager,
        engine_adapter,
        tokenizer,
        next_request_id: AtomicU64::new(1000), // Start from 1000 to avoid engine dummy seqs (<8)
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
