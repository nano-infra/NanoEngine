use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::http::{header::CONTENT_TYPE, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::{
    extract::{Json, State},
    routing::{get, post},
    Router,
};
use futures::StreamExt;
use serde_json::Value;
use tokio::sync::Mutex;
use tower_http::trace::TraceLayer;

use super::engine_manager::{EngineManager, ModelPool};

pub struct AppState {
    pub engine_manager: Arc<Mutex<EngineManager>>,
    pub http_client: reqwest::Client,
}

enum HttpRoute {
    Hybrid(String),
    Pd {
        prefill_url: String,
        decode_url: String,
    },
}

fn request_model(payload: &Value) -> Option<String> {
    payload
        .get("model")
        .and_then(|v| v.as_str())
        .map(|s| s.trim_end_matches('/').to_string())
}

fn request_stream(payload: &Value) -> bool {
    payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn sanitize_http_payload(mut payload: Value) -> Value {
    if let Some(obj) = payload.as_object_mut() {
        obj.retain(|_k, v| !v.is_null());
    }
    payload
}

fn check_engine_availability(pool: &ModelPool) -> Option<Response> {
    if pool.get_next_http_hybrid().is_some() {
        return None;
    }
    if pool.get_next_http_pd_pair().is_some() {
        return None;
    }
    Some(
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "No HTTP DLEngine nodes available",
        )
            .into_response(),
    )
}

async fn upstream_response_to_axum(upstream: reqwest::Response, stream: bool) -> Response {
    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut builder = Response::builder().status(status);

    if stream {
        builder = builder.header(CONTENT_TYPE, "text/event-stream");
        return builder
            .body(Body::from_stream(upstream.bytes_stream()))
            .unwrap_or_else(|e| (StatusCode::BAD_GATEWAY, e.to_string()).into_response());
    }

    let content_type = upstream
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let body = upstream.bytes().await.unwrap_or_default();
    builder
        .header(CONTENT_TYPE, content_type)
        .body(Body::from(body))
        .unwrap_or_else(|e| (StatusCode::BAD_GATEWAY, e.to_string()).into_response())
}

async fn forward_http_payload(
    client: reqwest::Client,
    base_url: String,
    endpoint: &'static str,
    payload: Value,
    stream: bool,
) -> Response {
    let url = format!("{}{}", base_url.trim_end_matches('/'), endpoint);
    match client.post(url).json(&payload).send().await {
        Ok(upstream) => upstream_response_to_axum(upstream, stream).await,
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            format!("HTTP DLEngine upstream error: {}", e),
        )
            .into_response(),
    }
}

async fn free_http_prefill(client: reqwest::Client, prefill_url: String, seq_id: Value) {
    let url = format!("{}/pd/free", prefill_url.trim_end_matches('/'));
    let _ = client
        .post(url)
        .json(&serde_json::json!({ "seq_ids": [seq_id] }))
        .send()
        .await;
}

fn completion_as_sse(completion: Value) -> Response {
    let body = format!("data: {}\n\ndata: [DONE]\n\n", completion);
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .body(Body::from(body))
        .unwrap_or_else(|e| (StatusCode::BAD_GATEWAY, e.to_string()).into_response())
}

async fn forward_http_pd(
    client: reqwest::Client,
    prefill_url: String,
    decode_url: String,
    endpoint: &'static str,
    payload: Value,
    stream: bool,
) -> Response {
    let mut prefill_payload = sanitize_http_payload(payload.clone());
    if let Some(obj) = prefill_payload.as_object_mut() {
        obj.insert("stream".to_string(), Value::Bool(false));
        obj.insert(
            "kv_transfer_params".to_string(),
            serde_json::json!({ "do_remote_decode": true }),
        );
    }

    let prefill_endpoint = format!("{}{}", prefill_url.trim_end_matches('/'), endpoint);
    let prefill_resp = match client
        .post(prefill_endpoint)
        .json(&prefill_payload)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => resp,
        Ok(resp) => return upstream_response_to_axum(resp, false).await,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("HTTP DLEngine prefill error: {}", e),
            )
                .into_response()
        }
    };

    let prefill_info: Value = match prefill_resp.json().await {
        Ok(value) => value,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("Invalid DLEngine prefill response: {}", e),
            )
                .into_response()
        }
    };

    let kv = prefill_info
        .get("kv_transfer_params")
        .and_then(|v| v.as_object());
    let migration = kv.and_then(|m| m.get("migration")).cloned();
    let seq_id = kv.and_then(|m| m.get("seq_id")).cloned();
    let first_token = kv.and_then(|m| m.get("first_token")).cloned();

    let Some(migration) = migration else {
        if prefill_info.get("choices").is_some() {
            return if stream {
                completion_as_sse(prefill_info)
            } else {
                Json(prefill_info).into_response()
            };
        }
        return (
            StatusCode::BAD_GATEWAY,
            "DLEngine prefill response missing migration payload",
        )
            .into_response();
    };
    let Some(seq_id) = seq_id else {
        return (
            StatusCode::BAD_GATEWAY,
            "DLEngine prefill response missing migration seq_id",
        )
            .into_response();
    };

    let mut decode_payload = sanitize_http_payload(payload);
    if let Some(obj) = decode_payload.as_object_mut() {
        let mut kv_transfer_params = serde_json::json!({
            "migration": migration,
            "seq_id": seq_id.clone(),
        });
        if let (Some(params), Some(first_token)) = (kv_transfer_params.as_object_mut(), first_token)
        {
            params.insert("first_token".to_string(), first_token);
        }
        obj.insert("kv_transfer_params".to_string(), kv_transfer_params);
        obj.insert("stream".to_string(), Value::Bool(stream));
    }

    if !stream {
        let response =
            forward_http_payload(client.clone(), decode_url, endpoint, decode_payload, false).await;
        free_http_prefill(client, prefill_url, seq_id).await;
        return response;
    }

    let decode_endpoint = format!("{}{}", decode_url.trim_end_matches('/'), endpoint);
    match client
        .post(decode_endpoint)
        .json(&decode_payload)
        .send()
        .await
    {
        Ok(upstream) => {
            let status =
                StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            if !status.is_success() {
                return upstream_response_to_axum(upstream, false).await;
            }
            let mut bytes_stream = upstream.bytes_stream();
            let free_client = client.clone();
            let free_prefill_url = prefill_url.clone();
            let free_seq_id = seq_id.clone();
            let stream = async_stream::stream! {
                while let Some(chunk) = bytes_stream.next().await {
                    match chunk {
                        Ok(bytes) => yield Ok::<_, std::io::Error>(bytes),
                        Err(e) => {
                            yield Err(std::io::Error::other(e));
                            return;
                        }
                    }
                }
                free_http_prefill(free_client, free_prefill_url, free_seq_id).await;
            };
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from_stream(stream))
                .unwrap_or_else(|e| (StatusCode::BAD_GATEWAY, e.to_string()).into_response())
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            format!("HTTP DLEngine decode error: {}", e),
        )
            .into_response(),
    }
}

async fn route_generation_request(
    State(state): State<Arc<AppState>>,
    Json(mut payload): Json<Value>,
    endpoint: &'static str,
) -> Response {
    let Some(model_key) = request_model(&payload) else {
        return (
            StatusCode::BAD_REQUEST,
            "Missing required string field: model",
        )
            .into_response();
    };
    let stream = request_stream(&payload);

    let route = {
        let mgr = state.engine_manager.lock().await;
        let canonical_model = mgr.resolve_model_key(&model_key).map(str::to_string);
        let pool = match canonical_model
            .as_deref()
            .and_then(|key| mgr.model_pools.get(key))
        {
            Some(pool) => {
                payload["model"] = Value::String(canonical_model.unwrap());
                pool
            }
            None => {
                return (
                    StatusCode::NOT_FOUND,
                    format!(
                        "Model '{}' not found. Available: [{}]",
                        model_key,
                        mgr.available_model_keys().join(", ")
                    ),
                )
                    .into_response()
            }
        };
        if let Some(err) = check_engine_availability(pool) {
            return err;
        }
        if let Some(url) = pool.get_next_http_hybrid() {
            HttpRoute::Hybrid(url.to_string())
        } else {
            let (prefill_url, decode_url) = pool
                .get_next_http_pd_pair()
                .expect("compatible PD pair checked above");
            let prefill_url = prefill_url.to_string();
            let decode_url = decode_url.to_string();
            HttpRoute::Pd {
                prefill_url,
                decode_url,
            }
        }
    };

    match route {
        HttpRoute::Hybrid(url) => {
            tracing::info!("Forwarding request to HTTP DLEngine hybrid node: {}", url);
            forward_http_payload(
                state.http_client.clone(),
                url,
                endpoint,
                sanitize_http_payload(payload),
                stream,
            )
            .await
        }
        HttpRoute::Pd {
            prefill_url,
            decode_url,
        } => {
            tracing::info!(
                "Forwarding request to HTTP DLEngine PD nodes: prefill={} decode={}",
                prefill_url,
                decode_url
            );
            forward_http_pd(
                state.http_client.clone(),
                prefill_url,
                decode_url,
                endpoint,
                payload,
                stream,
            )
            .await
        }
    }
}

async fn chat_completions(state: State<Arc<AppState>>, payload: Json<Value>) -> Response {
    route_generation_request(state, payload, "/v1/chat/completions").await
}

async fn anthropic_messages(state: State<Arc<AppState>>, payload: Json<Value>) -> Response {
    route_generation_request(state, payload, "/v1/messages").await
}

async fn anthropic_count_tokens(
    State(state): State<Arc<AppState>>,
    Json(mut payload): Json<Value>,
) -> Response {
    let Some(model_key) = request_model(&payload) else {
        return (
            StatusCode::BAD_REQUEST,
            "Missing required string field: model",
        )
            .into_response();
    };

    let route = {
        let mgr = state.engine_manager.lock().await;
        let canonical_model = mgr.resolve_model_key(&model_key).map(str::to_string);
        let pool = match canonical_model
            .as_deref()
            .and_then(|key| mgr.model_pools.get(key))
        {
            Some(pool) => {
                payload["model"] = Value::String(canonical_model.unwrap());
                pool
            }
            None => {
                return (
                    StatusCode::NOT_FOUND,
                    format!(
                        "Model '{}' not found. Available: [{}]",
                        model_key,
                        mgr.available_model_keys().join(", ")
                    ),
                )
                    .into_response()
            }
        };
        pool.get_next_http_hybrid()
            .or_else(|| pool.get_next_http_prefill())
            .or_else(|| pool.get_next_http_decode())
            .map(str::to_string)
    };

    let Some(url) = route else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "No HTTP DLEngine nodes available",
        )
            .into_response();
    };

    tracing::info!(
        "Forwarding Anthropic count_tokens request to HTTP DLEngine node: {}",
        url
    );
    forward_http_payload(
        state.http_client.clone(),
        url,
        "/v1/messages/count_tokens",
        sanitize_http_payload(payload),
        false,
    )
    .await
}

async fn health() -> &'static str {
    "OK"
}

async fn models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let manager = state.engine_manager.lock().await;
    let data = manager
        .routable_model_keys()
        .into_iter()
        .map(|model_id| {
            serde_json::json!({
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "dlengine",
            })
        })
        .collect::<Vec<_>>();

    Json(serde_json::json!({
        "object": "list",
        "data": data,
    }))
}

pub async fn start_server(
    port: u16,
    engine_manager: Arc<Mutex<EngineManager>>,
) -> anyhow::Result<()> {
    let state = Arc::new(AppState {
        engine_manager,
        http_client: reqwest::Client::new(),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    if let Err(e) = tokio::signal::ctrl_c().await {
        tracing::warn!("Failed to install Ctrl-C handler: {}", e);
        return;
    }

    tracing::info!("Shutdown signal received, stopping HTTP server");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_pool(hybrid: bool, prefill: bool, decode: bool) -> ModelPool {
        ModelPool {
            http_hybrid_engines: hybrid
                .then(|| "http://hybrid".to_string())
                .into_iter()
                .collect(),
            http_prefill_engines: prefill
                .then(|| "http://prefill".to_string())
                .into_iter()
                .collect(),
            http_decode_engines: decode
                .then(|| "http://decode".to_string())
                .into_iter()
                .collect(),
            http_engine_fabric_placements: Default::default(),
        }
    }

    #[tokio::test]
    async fn models_lists_sorted_routable_model_aliases() {
        let mut manager = EngineManager::new();
        manager
            .model_pools
            .insert("z-hybrid".to_string(), model_pool(true, false, false));
        manager
            .model_pools
            .insert("a-pd".to_string(), model_pool(false, true, true));
        manager
            .model_pools
            .insert("prefill-only".to_string(), model_pool(false, true, false));
        let state = Arc::new(AppState {
            engine_manager: Arc::new(Mutex::new(manager)),
            http_client: reqwest::Client::new(),
        });

        let Json(response) = models(State(state)).await;
        assert_eq!(response["object"], "list");
        let data = response["data"].as_array().unwrap();
        let ids = data
            .iter()
            .map(|model| model["id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["a-pd", "z-hybrid"]);
        for model in data {
            assert_eq!(model["object"], "model");
            assert_eq!(model["owned_by"], "dlengine");
            assert!(model["created"].as_u64().unwrap() > 0);
        }
    }

    #[tokio::test]
    async fn models_returns_an_empty_openai_list_without_routes() {
        let state = Arc::new(AppState {
            engine_manager: Arc::new(Mutex::new(EngineManager::new())),
            http_client: reqwest::Client::new(),
        });

        let Json(response) = models(State(state)).await;
        assert_eq!(response, serde_json::json!({"object": "list", "data": []}));
    }
}
