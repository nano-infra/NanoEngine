mod models;
mod state;

use axum::{
    extract::State,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
// chrono::Utc is no longer needed since timestamp is set in Lua script
// Script is no longer needed, we use EVAL directly
use serde_json::json;
use std::net::SocketAddr;
use tokio::sync::oneshot;
use tower::ServiceBuilder;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::models::*;
use crate::state::{
    AppState, HEARTBEAT_ENGINE_SCRIPT, REGISTER_ENGINE_SCRIPT, UNREGISTER_ENGINE_SCRIPT,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Redis URL: REDIS_URL env (e.g. redis://host:6379) or default localhost
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    tracing::info!("Using Redis URL: {}", redis_url);
    let state = AppState::new(&redis_url)?;
    let app_state = state.clone();

    let app = Router::new()
        .route("/", get(root))
        .route("/start_peer_agent", post(start_peer_agent))
        .route("/query", post(query))
        .route("/init", post(init))
        .route("/connect", post(connect))
        .route("/register_mr", post(register_mr))
        .route("/get_mr_info", post(get_mr_info))
        .route("/get_endpoint_info", post(get_endpoint_info))
        .route("/ack_init", post(ack_init))
        .route("/ack_connect", post(ack_connect))
        .route("/update_endpoint_info", post(update_endpoint_info))
        .route("/cleanup", post(cleanup))
        .route("/get_redis_address", post(get_redis_address))
        .route("/register_engine", post(register_engine))
        .route("/unregister_engine", post(unregister_engine))
        .route("/heartbeat_engine", post(heartbeat_engine))
        .route("/get_engine_info", post(get_engine_info))
        .route("/list_engines", post(list_engines))
        .layer(
            ServiceBuilder::new().layer(
                TraceLayer::new_for_http()
                    .make_span_with(|request: &axum::http::Request<_>| {
                        tracing::info_span!(
                            "http_request",
                            method = %request.method(),
                            uri = %request.uri(),
                        )
                    })
                    .on_request(|request: &axum::http::Request<_>, _span: &tracing::Span| {
                        tracing::info!("Incoming request: {} {}", request.method(), request.uri());
                    })
                    .on_response(
                        |response: &axum::http::Response<_>,
                         latency: std::time::Duration,
                         _span: &tracing::Span| {
                            tracing::info!(
                                "Response sent: status={}, latency={:?}",
                                response.status(),
                                latency
                            );
                        },
                    )
                    .on_failure(
                        |error: tower_http::classify::ServerErrorsFailureClass,
                         latency: std::time::Duration,
                         _span: &tracing::Span| {
                            tracing::error!("Request failed: {:?}, latency={:?}", error, latency);
                        },
                    ),
            ),
        )
        .with_state(app_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    tracing::info!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn root() -> &'static str {
    "NanoCtrl Server Running"
}

async fn query(
    State(state): State<AppState>,
    Json(_body): Json<QueryBody>,
) -> Json<Vec<PeerAgent>> {
    // Basic implementation: Scan for agent:* keys
    let mut conn = match state.redis_client.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::error!("Failed to get Redis connection: {}", e);
            return Json(Vec::new());
        }
    };
    let keys: Vec<String> = redis::cmd("KEYS")
        .arg("agent:*")
        .query_async(&mut conn)
        .await
        .unwrap_or_default();

    let mut agents = Vec::new();
    for key in keys {
        let agent_data: std::collections::HashMap<String, String> = redis::cmd("HGETALL")
            .arg(&key)
            .query_async(&mut conn)
            .await
            .unwrap_or_default();
        // Parse agent_data to PeerAgent, simplified for now
        // Currently just constructing minimal info
        if let (Some(dev), Some(ip)) = (agent_data.get("device"), agent_data.get("addr")) {
            // Assuming key format agent:{name}
            let name = key.strip_prefix("agent:").unwrap_or(&key).to_string();
            agents.push(PeerAgent {
                name,
                device: dev.clone(),
                ib_port: agent_data
                    .get("ib_port")
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(1),
                link_type: agent_data
                    .get("link_type")
                    .cloned()
                    .unwrap_or("RoCE".into()),
                address: ip.clone(),
            });
        }
    }
    Json(agents)
}

async fn start_peer_agent(
    State(state): State<AppState>,
    Json(body): Json<StartPeerAgentBody>,
) -> impl IntoResponse {
    tracing::info!(
        "Received registration request for agent: {} (device={}, ib_port={}, link_type={}, address={})",
        body.alias,
        body.device,
        body.ib_port,
        body.link_type,
        body.address
    );

    let mut conn = match state.redis_client.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::error!("Failed to get Redis connection: {}", e);
            return Json(StartPeerAgentResponse {
                status: "error".to_string(),
                redis_address: "".to_string(),
            });
        }
    };

    let key = format!("agent:{}", body.alias);
    tracing::debug!("Storing agent info in Redis with key: {}", key);
    match redis::cmd("HSET")
        .arg(&key)
        .arg("device")
        .arg(&body.device)
        .arg("ib_port")
        .arg(body.ib_port.to_string())
        .arg("link_type")
        .arg(&body.link_type)
        .arg("addr")
        .arg(&body.address)
        .query_async::<()>(&mut conn)
        .await
    {
        Ok(_) => {
            tracing::info!(
                "Successfully registered peer agent: {} in Redis",
                body.alias
            );
        }
        Err(e) => {
            tracing::error!("Failed to register agent {} in Redis: {}", body.alias, e);
            return Json(StartPeerAgentResponse {
                status: "error".to_string(),
                redis_address: "".to_string(),
            });
        }
    }

    // Extract host:port from redis_url (format: "redis://host:port" or "redis://127.0.0.1:6379")
    // For remote clients, we need to return the public Redis address, not 127.0.0.1
    let redis_address = if state.redis_url.starts_with("redis://") {
        let addr = state
            .redis_url
            .strip_prefix("redis://")
            .unwrap_or(&state.redis_url);
        // If Redis is on localhost (127.0.0.1), check if client is remote
        if (addr.starts_with("127.0.0.1") || addr.starts_with("localhost"))
            && !body.address.starts_with("127.0.0.1")
            && !body.address.starts_with("localhost")
        {
            // Client is remote, need to return server's public IP
            // Use environment variable REDIS_PUBLIC_ADDRESS if set
            if let Ok(public_addr) = std::env::var("REDIS_PUBLIC_ADDRESS") {
                tracing::info!(
                    "Client {} is remote, using REDIS_PUBLIC_ADDRESS={}",
                    body.address,
                    public_addr
                );
                public_addr
            } else {
                // Extract port from addr (format: "127.0.0.1:6379")
                let port = if let Some(colon_pos) = addr.find(':') {
                    &addr[colon_pos + 1..]
                } else {
                    "6379"
                };
                // Try to get server's IP from network interfaces
                // For simplicity, we'll use the server's IP from the client's subnet
                // This is a heuristic: server IP should be on the same network as client
                // We'll try to find a non-loopback interface on the same subnet
                let server_ip = get_server_ip_for_client(&body.address).unwrap_or_else(|| {
                    tracing::warn!(
                        "Cannot determine server IP for remote client {}. \
                        Please set REDIS_PUBLIC_ADDRESS environment variable. \
                        Falling back to 127.0.0.1 (may not work for remote clients).",
                        body.address
                    );
                    "127.0.0.1".to_string()
                });
                format!("{}:{}", server_ip, port)
            }
        } else {
            // Client is local or Redis is not on localhost, return as-is
            addr.to_string()
        }
    } else {
        state.redis_url.clone()
    };

    tracing::info!(
        "Sending response for agent {}: status=ok, redis_address={} (client address: {})",
        body.alias,
        redis_address,
        body.address
    );

    Json(StartPeerAgentResponse {
        status: "ok".to_string(),
        redis_address,
    })
}

// Helper function to get server's IP address for a remote client
// This is a simplified implementation - in production, you might want to use a library
// like `local_ipaddress` or inspect network interfaces directly
fn get_server_ip_for_client(_client_ip: &str) -> Option<String> {
    // For now, we require REDIS_PUBLIC_ADDRESS to be set
    // In the future, we could inspect network interfaces to find the server's IP
    None
}

async fn init(State(state): State<AppState>, Json(body): Json<InitBody>) -> impl IntoResponse {
    let (low, high) = if body.src < body.dst {
        (body.src.clone(), body.dst.clone())
    } else {
        (body.dst.clone(), body.src.clone())
    };

    let conn_key = format!("conn:{}:{}", low, high);
    let lock = state.get_lock(conn_key.clone()).await;
    let _guard = lock.lock().await;

    let mut conn = match state.redis_client.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::error!("Failed to get Redis connection: {}", e);
            return Json(InitResponse {
                status: "error".to_string(),
                message: format!("Failed to connect to Redis: {}", e),
            });
        }
    };

    // Check if connection already initialized or in progress
    let status: Option<String> = redis::cmd("HGET")
        .arg(&conn_key)
        .arg("status")
        .query_async(&mut conn)
        .await
        .unwrap_or(None);

    if let Some(s) = status {
        if s == "initialized" || s == "connected" {
            tracing::info!(
                "Connection {} <-> {} already initialized, returning immediately",
                low,
                high
            );
            return Json(InitResponse {
                status: "ok".to_string(),
                message: "Already initialized".to_string(),
            });
        }
        // If status is "initializing", another init is in progress, wait for it
        if s == "initializing" {
            tracing::info!(
                "Connection {} <-> {} is already initializing, waiting for completion",
                low,
                high
            );
            // Wait for the existing init to complete by checking status periodically
            let mut retries = 100; // 10 seconds
            while retries > 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                let current_status: Option<String> = redis::cmd("HGET")
                    .arg(&conn_key)
                    .arg("status")
                    .query_async(&mut conn)
                    .await
                    .unwrap_or(None);
                if let Some(cs) = current_status {
                    if cs == "initialized" || cs == "connected" {
                        tracing::info!(
                            "Connection {} <-> {} initialized by another request",
                            low,
                            high
                        );
                        return Json(InitResponse {
                            status: "ok".to_string(),
                            message: "Initialized by another request".to_string(),
                        });
                    }
                }
                retries -= 1;
            }
            tracing::warn!(
                "Connection {} <-> {} still initializing after waiting, returning timeout",
                low,
                high
            );
            return Json(InitResponse {
                status: "timeout".to_string(),
                message: "Timeout waiting for concurrent init to complete".to_string(),
            });
        }
    }

    // Set status to initializing
    let _: () = redis::cmd("HSET")
        .arg(&conn_key)
        .arg("status")
        .arg("initializing")
        .arg("qp_num")
        .arg(body.qp_num.to_string())
        .query_async(&mut conn)
        .await
        .unwrap_or_default();

    // Publish init event to both agents' mailboxes (level-triggered)
    // In level-triggered mode, server publishes events and waits for ACKs from both sides
    let event_low = json!({
        "type": "init",
        "src": body.src.clone(),
        "dst": body.dst.clone(),
        "qp_num": body.qp_num,
    });
    let event_high = json!({
        "type": "init",
        "src": body.src.clone(),
        "dst": body.dst.clone(),
        "qp_num": body.qp_num,
    });

    let low_inbox = format!("inbox:{}", low);
    let high_inbox = format!("inbox:{}", high);

    let low_result: Result<(), _> = redis::cmd("LPUSH")
        .arg(&low_inbox)
        .arg(event_low.to_string())
        .query_async(&mut conn)
        .await;

    let high_result: Result<(), _> = redis::cmd("LPUSH")
        .arg(&high_inbox)
        .arg(event_high.to_string())
        .query_async(&mut conn)
        .await;

    tracing::info!(
        "Pushed init events: {} -> {} (inbox:{}), {} -> {} (inbox:{}), results: low={:?}, high={:?}",
        body.src,
        body.dst,
        low_inbox,
        body.src,
        body.dst,
        high_inbox,
        low_result.is_ok(),
        high_result.is_ok()
    );

    tracing::info!(
        "Published init events for connection {} <-> {} (level-triggered, waiting for ACKs). Events sent to inbox:{} and inbox:{}",
        low,
        high,
        low,
        high
    );

    // Level-triggered: create event futures and await ACKs from both agents
    // Create oneshot channels for async ACK waiting
    let (low_tx, low_rx) = oneshot::channel();
    let (high_tx, high_rx) = oneshot::channel();

    // Store senders in state for ack_init to signal
    {
        let mut futures = state.init_futures.lock().await;
        futures.insert(conn_key.clone(), (Some(low_tx), Some(high_tx)));
        tracing::info!(
            "Stored init futures for connection {} <-> {} (waiting for ACKs from {} and {})",
            low,
            high,
            low,
            high
        );
    }

    // Await both ACKs in parallel with timeout
    let timeout = tokio::time::Duration::from_secs(10);
    match tokio::time::timeout(timeout, async {
        let (low_result, high_result) = tokio::join!(low_rx, high_rx);
        // Both channels should receive signals (ignore errors if channel was closed)
        let _ = low_result;
        let _ = high_result;
    })
    .await
    {
        Ok(_) => {
            // Both ACKs received, mark as initialized
            let _: () = redis::cmd("HSET")
                .arg(&conn_key)
                .arg("status")
                .arg("initialized")
                .query_async(&mut conn)
                .await
                .unwrap_or_default();

            // Clean up futures
            {
                let mut futures = state.init_futures.lock().await;
                futures.remove(&conn_key);
            }

            tracing::info!(
                "Init completed for connection {} <-> {} (both ACKs received)",
                low,
                high
            );
            Json(InitResponse {
                status: "ok".to_string(),
                message: "Initialized successfully".to_string(),
            })
        }
        Err(_) => {
            // Timeout: clean up futures
            {
                let mut futures = state.init_futures.lock().await;
                futures.remove(&conn_key);
            }

            tracing::warn!(
                "Init timeout for connection {} <-> {} (waiting for ACKs)",
                low,
                high
            );
            Json(InitResponse {
                status: "timeout".to_string(),
                message: "Timeout waiting for init ACK".to_string(),
            })
        }
    }
}

async fn connect(
    State(state): State<AppState>,
    Json(body): Json<ConnectBody>,
) -> impl IntoResponse {
    let (low, high) = if body.src < body.dst {
        (body.src.clone(), body.dst.clone())
    } else {
        (body.dst.clone(), body.src.clone())
    };

    let conn_key = format!("conn:{}:{}", low, high);
    let lock = state.get_lock(conn_key.clone()).await;
    let _guard = lock.lock().await;

    let mut conn = match state.redis_client.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::error!("Failed to get Redis connection: {}", e);
            return Json(ConnectResponse {
                status: "error".to_string(),
                message: format!("Failed to connect to Redis: {}", e),
            });
        }
    };

    // Check if already connected
    let status: Option<String> = redis::cmd("HGET")
        .arg(&conn_key)
        .arg("status")
        .query_async(&mut conn)
        .await
        .unwrap_or(None);

    if let Some(s) = status {
        if s == "connected" {
            return Json(ConnectResponse {
                status: "ok".to_string(),
                message: "Already connected".to_string(),
            });
        }
    }

    // Set status to connecting
    let _: () = redis::cmd("HSET")
        .arg(&conn_key)
        .arg("status")
        .arg("connecting")
        .query_async(&mut conn)
        .await
        .unwrap_or_default();

    // Publish connect event (level-triggered)
    // In level-triggered mode, server publishes events and waits for connection to complete
    let event_low = json!({
        "type": "connect",
        "src": body.src.clone(),
        "dst": body.dst.clone(),
    });
    let event_high = json!({
        "type": "connect",
        "src": body.src.clone(),
        "dst": body.dst.clone(),
    });

    let _: () = redis::cmd("LPUSH")
        .arg(format!("inbox:{}", low))
        .arg(event_low.to_string())
        .query_async(&mut conn)
        .await
        .unwrap_or_default();

    let _: () = redis::cmd("LPUSH")
        .arg(format!("inbox:{}", high))
        .arg(event_high.to_string())
        .query_async(&mut conn)
        .await
        .unwrap_or_default();

    tracing::info!(
        "Published connect events for connection {} <-> {} (level-triggered, waiting for connection)",
        low,
        high
    );

    // Level-triggered: create event future and await connection ACK
    let (tx, rx) = oneshot::channel();

    // Store sender in state for ack_connect to signal
    {
        let mut futures = state.connect_futures.lock().await;
        futures.insert(conn_key.clone(), tx);
    }

    // Await connection ACK with timeout
    let timeout = tokio::time::Duration::from_secs(5);
    match tokio::time::timeout(timeout, rx).await {
        Ok(Ok(_)) => {
            // Connection ACK received
            // Clean up future
            {
                let mut futures = state.connect_futures.lock().await;
                futures.remove(&conn_key);
            }

            tracing::info!("Connect completed for connection {} <-> {}", low, high);
            Json(ConnectResponse {
                status: "ok".to_string(),
                message: "Connected successfully".to_string(),
            })
        }
        Ok(Err(_)) | Err(_) => {
            // Timeout or channel closed: clean up future
            {
                let mut futures = state.connect_futures.lock().await;
                futures.remove(&conn_key);
            }

            tracing::warn!("Connect timeout for connection {} <-> {}", low, high);
            Json(ConnectResponse {
                status: "timeout".to_string(),
                message: "Timeout waiting for connect ACK".to_string(),
            })
        }
    }
}

async fn register_mr(
    State(state): State<AppState>,
    Json(body): Json<RegisterMrBody>,
) -> impl IntoResponse {
    tracing::info!(
        "Registering MR: agent={}, mr_name={}, addr={}, length={}, rkey={}, lkey={}",
        body.agent_name,
        body.mr_name,
        body.addr,
        body.length,
        body.rkey,
        body.lkey
    );

    let mut conn = match state.redis_client.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::error!("Failed to get Redis connection: {}", e);
            return Json(RegisterMrResponse {
                status: "error".to_string(),
            });
        }
    };

    let mr_key = format!("mr:{}:{}", body.agent_name, body.mr_name);
    let mr_info = json!({
        "addr": body.addr,
        "length": body.length,
        "rkey": body.rkey,
        "lkey": body.lkey,
    });

    let _: () = redis::cmd("SET")
        .arg(&mr_key)
        .arg(mr_info.to_string())
        .query_async(&mut conn)
        .await
        .unwrap_or_default();

    tracing::info!(
        "Registered MR: {} for agent: {}",
        body.mr_name,
        body.agent_name
    );
    Json(RegisterMrResponse {
        status: "ok".to_string(),
    })
}

async fn get_mr_info(
    State(state): State<AppState>,
    Json(body): Json<GetMrInfoBody>,
) -> impl IntoResponse {
    let cache_key = format!("{}:{}", body.dst, body.mr_name);

    // Check cache first
    {
        let cache = state.mr_info_cache.lock().await;
        if let Some((cached_at, cached)) = cache.get(&cache_key) {
            if cached_at.elapsed().as_secs() < state::MR_INFO_CACHE_TTL_SECS {
                return Json(cached.clone());
            }
        }
    }

    let mut conn = match state.redis_client.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::error!("Failed to get Redis connection: {}", e);
            return Json(GetMrInfoResponse { mr_info: None });
        }
    };

    let mr_key = format!("mr:{}:{}", body.dst, body.mr_name);
    let mr_info_str: Option<String> = redis::cmd("GET")
        .arg(&mr_key)
        .query_async(&mut conn)
        .await
        .unwrap_or(None);

    let response = if let Some(s) = mr_info_str {
        if let Ok(mr_json) = serde_json::from_str::<serde_json::Value>(&s) {
            if let (Some(addr), Some(length), Some(rkey), Some(lkey)) = (
                mr_json["addr"].as_u64(),
                mr_json["length"].as_u64(),
                mr_json["rkey"].as_u64(),
                mr_json["lkey"].as_u64(),
            ) {
                GetMrInfoResponse {
                    mr_info: Some(MrInfo {
                        addr,
                        length: length as usize,
                        rkey: rkey as u32,
                        lkey: lkey as u32,
                    }),
                }
            } else {
                GetMrInfoResponse { mr_info: None }
            }
        } else {
            GetMrInfoResponse { mr_info: None }
        }
    } else {
        GetMrInfoResponse { mr_info: None }
    };

    // Store in cache
    {
        let mut cache = state.mr_info_cache.lock().await;
        cache.insert(cache_key, (std::time::Instant::now(), response.clone()));
    }

    Json(response)
}

async fn get_endpoint_info(
    State(state): State<AppState>,
    Json(body): Json<GetEndpointInfoBody>,
) -> impl IntoResponse {
    let (low, high) = if body.src < body.dst {
        (body.src.clone(), body.dst.clone())
    } else {
        (body.dst.clone(), body.src.clone())
    };

    let conn_key = format!("conn:{}:{}", low, high);
    let mut conn = match state.redis_client.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::error!("Failed to get Redis connection: {}", e);
            return Json(GetEndpointInfoResponse {
                endpoint_info: None,
            });
        }
    };

    // Determine which endpoint info to get
    // body.src is the requester, body.dst is whose endpoint info we want
    // If body.dst == low, we want low_endpoint_info
    // If body.dst == high, we want high_endpoint_info
    let field_name = if body.dst == low {
        "low_endpoint_info"
    } else {
        "high_endpoint_info"
    };

    tracing::info!(
        "Getting endpoint info: src={}, dst={}, conn_key={}, field_name={}",
        body.src,
        body.dst,
        conn_key,
        field_name
    );

    let endpoint_info_str: Option<String> = redis::cmd("HGET")
        .arg(&conn_key)
        .arg(field_name)
        .query_async(&mut conn)
        .await
        .unwrap_or(None);

    if let Some(s) = endpoint_info_str {
        if let Ok(endpoint_info) = serde_json::from_str::<serde_json::Value>(&s) {
            tracing::info!("Found endpoint info for {} -> {}", body.src, body.dst);
            return Json(GetEndpointInfoResponse {
                endpoint_info: Some(endpoint_info),
            });
        } else {
            tracing::warn!(
                "Failed to parse endpoint info JSON for {} -> {}",
                body.src,
                body.dst
            );
        }
    } else {
        tracing::warn!(
            "Endpoint info not found for {} -> {} (conn_key={}, field_name={})",
            body.src,
            body.dst,
            conn_key,
            field_name
        );
    }

    Json(GetEndpointInfoResponse {
        endpoint_info: None,
    })
}

async fn ack_init(
    State(state): State<AppState>,
    Json(body): Json<AckInitBody>,
) -> impl IntoResponse {
    let (low, high) = if body.src < body.dst {
        (body.src.clone(), body.dst.clone())
    } else {
        (body.dst.clone(), body.src.clone())
    };

    let conn_key = format!("conn:{}:{}", low, high);
    let mut conn = match state.redis_client.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::error!("Failed to get Redis connection: {}", e);
            return Json(AckResponse {
                status: "error".to_string(),
            });
        }
    };

    // Store endpoint info
    let field_name = if body.src == low {
        "low_endpoint_info"
    } else {
        "high_endpoint_info"
    };

    let _: () = redis::cmd("HSET")
        .arg(&conn_key)
        .arg(field_name)
        .arg(body.endpoint_info.to_string())
        .query_async(&mut conn)
        .await
        .unwrap_or_default();

    // Signal the init future that this ACK was received
    let mut futures = state.init_futures.lock().await;
    if let Some((low_tx, high_tx)) = futures.get_mut(&conn_key) {
        let _ack_sent = if body.src == low {
            // Low agent sent ACK
            if let Some(tx) = low_tx.take() {
                let result = tx.send(());
                tracing::info!(
                    "Received init ACK from low agent {} for connection {} <-> {} (channel send: {})",
                    body.src,
                    low,
                    high,
                    if result.is_ok() { "ok" } else { "error" }
                );
                result.is_ok()
            } else {
                tracing::warn!(
                    "Received duplicate init ACK from low agent {} for connection {} <-> {}",
                    body.src,
                    low,
                    high
                );
                false
            }
        } else {
            // High agent sent ACK
            if let Some(tx) = high_tx.take() {
                let result = tx.send(());
                tracing::info!(
                    "Received init ACK from high agent {} for connection {} <-> {} (channel send: {})",
                    body.src,
                    low,
                    high,
                    if result.is_ok() { "ok" } else { "error" }
                );
                result.is_ok()
            } else {
                tracing::warn!(
                    "Received duplicate init ACK from high agent {} for connection {} <-> {}",
                    body.src,
                    low,
                    high
                );
                false
            }
        };

        // Check if both ACKs have been sent (both senders are None)
        if low_tx.is_none() && high_tx.is_none() {
            // Both ACKs received, can clean up (but keep for status check)
            tracing::info!(
                "Both init ACKs received for connection {} <-> {}",
                low,
                high
            );
        }
    } else {
        tracing::warn!(
            "Received init ACK from {} for connection {} <-> {} but no future found (may have timed out or already completed)",
            body.src,
            low,
            high
        );
    }

    Json(AckResponse {
        status: "ok".to_string(),
    })
}

async fn ack_connect(
    State(state): State<AppState>,
    Json(body): Json<AckConnectBody>,
) -> impl IntoResponse {
    let (low, high) = if body.src < body.dst {
        (body.src.clone(), body.dst.clone())
    } else {
        (body.dst.clone(), body.src.clone())
    };

    let conn_key = format!("conn:{}:{}", low, high);
    let mut conn = match state.redis_client.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::error!("Failed to get Redis connection: {}", e);
            return Json(AckResponse {
                status: "error".to_string(),
            });
        }
    };

    // Track ACKs: use a counter or check if both sides have ACKed
    // For simplicity, we'll use a counter in Redis
    let ack_field = if body.src == low {
        "low_connect_ack"
    } else {
        "high_connect_ack"
    };

    let _: () = redis::cmd("HSET")
        .arg(&conn_key)
        .arg(ack_field)
        .arg("1")
        .query_async(&mut conn)
        .await
        .unwrap_or_default();

    // Check if both ACKs received
    let low_ack: Option<String> = redis::cmd("HGET")
        .arg(&conn_key)
        .arg("low_connect_ack")
        .query_async(&mut conn)
        .await
        .unwrap_or(None);
    let high_ack: Option<String> = redis::cmd("HGET")
        .arg(&conn_key)
        .arg("high_connect_ack")
        .query_async(&mut conn)
        .await
        .unwrap_or(None);

    if low_ack.is_some() && high_ack.is_some() {
        // Both ACKs received, mark as connected and signal the future
        let _: () = redis::cmd("HSET")
            .arg(&conn_key)
            .arg("status")
            .arg("connected")
            .query_async(&mut conn)
            .await
            .unwrap_or_default();

        // Signal the connect future
        let mut futures = state.connect_futures.lock().await;
        if let Some(tx) = futures.remove(&conn_key) {
            let _ = tx.send(());
            tracing::info!(
                "Both connect ACKs received for connection {} <-> {}",
                low,
                high
            );
        }
    }

    Json(AckResponse {
        status: "ok".to_string(),
    })
}

async fn update_endpoint_info(
    State(state): State<AppState>,
    Json(body): Json<UpdateEndpointInfoBody>,
) -> impl IntoResponse {
    let mut conn = match state.redis_client.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::error!("Failed to get Redis connection: {}", e);
            return Json(AckResponse {
                status: "error".to_string(),
            });
        }
    };

    let key = format!("agent:{}", body.agent_name);
    let _: () = redis::cmd("HSET")
        .arg(&key)
        .arg("endpoint_info")
        .arg(body.endpoint_info.to_string())
        .query_async(&mut conn)
        .await
        .unwrap_or_default();

    Json(AckResponse {
        status: "ok".to_string(),
    })
}

async fn cleanup(
    State(state): State<AppState>,
    Json(body): Json<CleanupBody>,
) -> impl IntoResponse {
    let mut conn = match state.redis_client.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::error!("Failed to get Redis connection: {}", e);
            return Json(CleanupResponse {
                status: "error".to_string(),
                message: format!("Failed to connect to Redis: {}", e),
            });
        }
    };

    let agent_name = body.agent_name;
    tracing::info!("Cleaning up agent: {}", agent_name);

    // 1. Delete agent registration
    let agent_key = format!("agent:{}", agent_name);
    let _: () = redis::cmd("DEL")
        .arg(&agent_key)
        .query_async(&mut conn)
        .await
        .unwrap_or_default();

    // 2. Delete agent's inbox
    let inbox_key = format!("inbox:{}", agent_name);
    let _: () = redis::cmd("DEL")
        .arg(&inbox_key)
        .query_async(&mut conn)
        .await
        .unwrap_or_default();

    // 3. Delete all MRs for this agent
    let mr_pattern = format!("mr:{}:*", agent_name);
    let mr_keys: Vec<String> = redis::cmd("KEYS")
        .arg(&mr_pattern)
        .query_async(&mut conn)
        .await
        .unwrap_or_default();
    if !mr_keys.is_empty() {
        let _: () = redis::cmd("DEL")
            .arg(&mr_keys)
            .query_async(&mut conn)
            .await
            .unwrap_or_default();
    }

    // 4. Find and clean up connections involving this agent
    // Also notify peer agents to clean up their side
    let conn_pattern = "conn:*";
    let conn_keys: Vec<String> = redis::cmd("KEYS")
        .arg(conn_pattern)
        .query_async(&mut conn)
        .await
        .unwrap_or_default();

    let mut peers_to_notify = std::collections::HashSet::new();

    for conn_key in &conn_keys {
        // Format: conn:{low}:{high}
        let parts: Vec<&str> = conn_key.split(':').collect();
        if parts.len() == 3 {
            let low = parts[1];
            let high = parts[2];

            if low == agent_name {
                // This agent is the low side, notify high side
                peers_to_notify.insert(high.to_string());
                // Delete the connection
                let _: () = redis::cmd("DEL")
                    .arg(conn_key)
                    .query_async(&mut conn)
                    .await
                    .unwrap_or_default();
            } else if high == agent_name {
                // This agent is the high side, notify low side
                peers_to_notify.insert(low.to_string());
                // Delete the connection
                let _: () = redis::cmd("DEL")
                    .arg(conn_key)
                    .query_async(&mut conn)
                    .await
                    .unwrap_or_default();
            }
        }
    }

    // 5. Notify peer agents to clean up their side (对等清理)
    for peer in peers_to_notify {
        let cleanup_event = json!({
            "type": "cleanup",
            "peer": agent_name,
        });
        let _: () = redis::cmd("LPUSH")
            .arg(format!("inbox:{}", peer))
            .arg(cleanup_event.to_string())
            .query_async(&mut conn)
            .await
            .unwrap_or_default();
        tracing::info!(
            "Notified peer {} to clean up connection with {}",
            peer,
            agent_name
        );
    }

    tracing::info!("Cleanup completed for agent: {}", agent_name);
    Json(CleanupResponse {
        status: "ok".to_string(),
        message: format!("Cleaned up agent: {}", agent_name),
    })
}

async fn get_redis_address(
    State(state): State<AppState>,
    Json(_body): Json<GetRedisAddressBody>,
) -> impl IntoResponse {
    // Return the full Redis URL (e.g., "redis://127.0.0.1:6379")
    // This allows NanoRouter to connect to Redis for dynamic service discovery
    tracing::info!("Returning Redis URL: {}", state.redis_url);
    Json(GetRedisAddressResponse {
        status: "ok".to_string(),
        redis_address: state.redis_url.clone(), // Return full URL, not just host:port
    })
}

async fn register_engine(
    State(state): State<AppState>,
    Json(body): Json<RegisterEngineBody>,
) -> impl IntoResponse {
    tracing::info!(
        "Registering engine: id={}, role={}, world_size={}, num_blocks={}, host={}, port={}, peer_addrs={:?}",
        body.engine_id,
        body.role,
        body.world_size,
        body.num_blocks,
        body.host,
        body.port,
        body.peer_addrs
    );

    let mut conn = match state.redis_client.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::error!("Failed to get Redis connection: {}", e);
            return Json(RegisterEngineResponse {
                status: "error".to_string(),
                message: format!("Failed to connect to Redis: {}", e),
            });
        }
    };
    let engine_key = format!("engine:{}", body.engine_id);
    let revision_key = "nano_meta:engine_revision";
    let channel = "nano_events:engine_update";

    tracing::debug!("Storing engine info in Redis with key: {}", engine_key);

    // Prepare payload JSON (for Lua script's cjson.decode)
    let zmq_address = format!("tcp://{}:{}", body.host, body.port);
    let payload = json!({
        "id": body.engine_id,
        "role": body.role,
        "host": body.host,
        "port": body.port,
        "zmq_address": zmq_address,
        "world_size": body.world_size,
        "num_blocks": body.num_blocks,
        "peer_addrs": body.peer_addrs,
    });

    // Prepare engine info JSON (for Redis hash storage)
    let engine_info = json!({
        "id": body.engine_id,
        "role": body.role,
        "world_size": body.world_size,
        "num_blocks": body.num_blocks,
        "host": body.host,
        "port": body.port,
        "peer_addrs": body.peer_addrs,
    });

    // Use Lua script to atomically: HSET + EXPIRE + INCR + PUBLISH
    // Note: Using EVAL directly since MultiplexedConnection doesn't implement ConnectionLike for Script
    let _revision: i64 = match redis::cmd("EVAL")
        .arg(REGISTER_ENGINE_SCRIPT)
        .arg(3) // number of keys
        .arg(&engine_key)
        .arg(revision_key)
        .arg(channel)
        .arg(&body.engine_id)
        .arg(&body.role)
        .arg(&body.host)
        .arg(body.port.to_string())
        .arg(body.world_size.to_string())
        .arg(body.num_blocks.to_string())
        .arg(serde_json::to_string(&body.peer_addrs).unwrap_or_default())
        .arg(engine_info.to_string())
        .arg(payload.to_string()) // Event payload JSON
        .arg(state::ENGINE_TTL_SECS.to_string()) // TTL
        .query_async::<i64>(&mut conn)
        .await
    {
        Ok(rev) => {
            tracing::info!(
                "Successfully registered engine: {} in Redis (revision: {}, TTL: {}s). Event published atomically.",
                body.engine_id,
                rev,
                state::ENGINE_TTL_SECS
            );
            rev
        }
        Err(e) => {
            tracing::error!(
                "Failed to register engine {} in Redis: {}",
                body.engine_id,
                e
            );
            return Json(RegisterEngineResponse {
                status: "error".to_string(),
                message: format!("Failed to register engine: {}", e),
            });
        }
    };

    Json(RegisterEngineResponse {
        status: "ok".to_string(),
        message: format!("Engine {} registered successfully", body.engine_id),
    })
}

async fn get_engine_info(
    State(state): State<AppState>,
    Json(body): Json<GetEngineInfoBody>,
) -> impl IntoResponse {
    tracing::info!("Querying engine info for: {}", body.engine_id);

    let mut conn = match state.redis_client.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::error!("Failed to get Redis connection: {}", e);
            return Json(GetEngineInfoResponse {
                status: "error".to_string(),
                engine_info: None,
            });
        }
    };

    let key = format!("engine:{}", body.engine_id);
    let engine_info_str: Option<String> = redis::cmd("HGET")
        .arg(&key)
        .arg("info")
        .query_async(&mut conn)
        .await
        .unwrap_or(None);

    if let Some(info_str) = engine_info_str {
        if let Ok(engine_info) = serde_json::from_str::<serde_json::Value>(&info_str) {
            tracing::info!("Found engine info for: {}", body.engine_id);
            return Json(GetEngineInfoResponse {
                status: "ok".to_string(),
                engine_info: Some(engine_info),
            });
        } else {
            tracing::warn!("Failed to parse engine info JSON for: {}", body.engine_id);
        }
    } else {
        tracing::warn!("Engine info not found for: {}", body.engine_id);
    }

    Json(GetEngineInfoResponse {
        status: "not_found".to_string(),
        engine_info: None,
    })
}

async fn list_engines(
    State(state): State<AppState>,
    Json(_body): Json<ListEnginesBody>,
) -> impl IntoResponse {
    tracing::info!("Listing all registered engines");

    let mut conn = match state.redis_client.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::error!("Failed to get Redis connection: {}", e);
            return Json(ListEnginesResponse {
                status: "error".to_string(),
                engines: Vec::new(),
            });
        }
    };

    // Get all engine keys
    let keys: Vec<String> = redis::cmd("KEYS")
        .arg("engine:*")
        .query_async(&mut conn)
        .await
        .unwrap_or_default();

    let mut engines = Vec::new();
    for key in keys {
        // Extract engine_id from key (format: "engine:{engine_id}")
        if let Some(_engine_id) = key.strip_prefix("engine:") {
            // Get the info field
            let engine_info_str: Option<String> = redis::cmd("HGET")
                .arg(&key)
                .arg("info")
                .query_async(&mut conn)
                .await
                .unwrap_or(None);

            if let Some(info_str) = engine_info_str {
                if let Ok(engine_info) = serde_json::from_str::<serde_json::Value>(&info_str) {
                    engines.push(engine_info);
                }
            }
        }
    }

    tracing::info!("Found {} registered engines", engines.len());
    Json(ListEnginesResponse {
        status: "ok".to_string(),
        engines,
    })
}

async fn unregister_engine(
    State(state): State<AppState>,
    Json(body): Json<UnregisterEngineBody>,
) -> impl IntoResponse {
    tracing::info!("Unregistering engine: {}", body.engine_id);

    let mut conn = match state.redis_client.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::error!("Failed to get Redis connection: {}", e);
            return Json(UnregisterEngineResponse {
                status: "error".to_string(),
                message: format!("Failed to connect to Redis: {}", e),
            });
        }
    };
    let engine_key = format!("engine:{}", body.engine_id);
    let revision_key = "nano_meta:engine_revision";
    let channel = "nano_events:engine_update";

    // Use Lua script to atomically: DEL + INCR + PUBLISH
    // Note: MultiplexedConnection doesn't implement ConnectionLike, so we use EVAL directly
    match redis::cmd("EVAL")
        .arg(UNREGISTER_ENGINE_SCRIPT)
        .arg(3) // number of keys
        .arg(&engine_key)
        .arg(revision_key)
        .arg(channel)
        .arg(&body.engine_id)
        .query_async::<i64>(&mut conn)
        .await
    {
        Ok(rev) => {
            if rev == 0 {
                // Engine not found
                tracing::warn!("Engine {} not found in Redis", body.engine_id);
                return Json(UnregisterEngineResponse {
                    status: "not_found".to_string(),
                    message: format!("Engine {} not found in Redis", body.engine_id),
                });
            }
            tracing::info!(
                "Successfully unregistered engine: {} from Redis (revision: {}). Event published atomically.",
                body.engine_id,
                rev
            );
        }
        Err(e) => {
            tracing::error!(
                "Failed to unregister engine {} from Redis: {}",
                body.engine_id,
                e
            );
            return Json(UnregisterEngineResponse {
                status: "error".to_string(),
                message: format!("Failed to unregister engine: {}", e),
            });
        }
    };

    Json(UnregisterEngineResponse {
        status: "ok".to_string(),
        message: format!("Engine {} unregistered successfully", body.engine_id),
    })
}

async fn heartbeat_engine(
    State(state): State<AppState>,
    Json(body): Json<HeartbeatEngineBody>,
) -> impl IntoResponse {
    tracing::debug!("Heartbeat for engine: {}", body.engine_id);

    let mut conn = match state.redis_client.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::error!("Failed to get Redis connection: {}", e);
            return Json(HeartbeatEngineResponse {
                status: "error".to_string(),
                message: format!("Failed to connect to Redis: {}", e),
            });
        }
    };

    let engine_key = format!("engine:{}", body.engine_id);

    // Use Lua script to refresh TTL only (no event, no revision increment)
    match redis::cmd("EVAL")
        .arg(HEARTBEAT_ENGINE_SCRIPT)
        .arg(1) // number of keys
        .arg(&engine_key)
        .arg(state::ENGINE_TTL_SECS.to_string()) // TTL
        .query_async::<i64>(&mut conn)
        .await
    {
        Ok(result) => {
            if result == 1 {
                tracing::debug!(
                    "Heartbeat successful for engine: {} (TTL refreshed)",
                    body.engine_id
                );
                Json(HeartbeatEngineResponse {
                    status: "ok".to_string(),
                    message: format!("Heartbeat successful for engine {}", body.engine_id),
                })
            } else {
                tracing::warn!("Engine {} not found in Redis for heartbeat", body.engine_id);
                Json(HeartbeatEngineResponse {
                    status: "not_found".to_string(),
                    message: format!(
                        "Engine {} not found. Please register first.",
                        body.engine_id
                    ),
                })
            }
        }
        Err(e) => {
            tracing::error!("Failed to refresh TTL for engine {}: {}", body.engine_id, e);
            Json(HeartbeatEngineResponse {
                status: "error".to_string(),
                message: format!("Failed to refresh TTL: {}", e),
            })
        }
    }
}
