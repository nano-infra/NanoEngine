mod models;
mod state;

use axum::{
    extract::State,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use std::net::SocketAddr;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::models::*;
use crate::state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let redis_url = "redis://127.0.0.1:6379"; // Could be from env
    let state = AppState::new(redis_url)?;
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
    let mut conn = state
        .redis_client
        .get_multiplexed_async_connection()
        .await
        .unwrap();
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
    let mut conn = state
        .redis_client
        .get_multiplexed_async_connection()
        .await
        .unwrap();

    let key = format!("agent:{}", body.alias);
    let _: () = redis::cmd("HSET")
        .arg(&key)
        .arg("device")
        .arg(&body.device)
        .arg("ib_port")
        .arg(body.ib_port.to_string())
        .arg("link_type")
        .arg(&body.link_type)
        .arg("addr")
        .arg(&body.address)
        .query_async(&mut conn)
        .await
        .unwrap_or_default();

    tracing::info!("Registered peer agent: {}", body.alias);
    Json(json!({"status": "ok"}))
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

    let mut conn = state
        .redis_client
        .get_multiplexed_async_connection()
        .await
        .unwrap();

    // Check if connection already initialized
    let status: Option<String> = redis::cmd("HGET")
        .arg(&conn_key)
        .arg("status")
        .query_async(&mut conn)
        .await
        .unwrap_or(None);

    if let Some(s) = status {
        if s == "initialized" || s == "connected" {
            return Json(InitResponse {
                status: "ok".to_string(),
                message: "Already initialized".to_string(),
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

    // Publish init event to both agents' mailboxes
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

    // Wait for ACK from both agents (simplified: wait for status update)
    // In production, this should use Redis pub/sub or polling
    let mut retries = 100; // Increased retries for more time
    while retries > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        let status: Option<String> = redis::cmd("HGET")
            .arg(&conn_key)
            .arg("status")
            .query_async(&mut conn)
            .await
            .unwrap_or(None);

        // Check if both endpoint infos are present
        let low_info: Option<String> = redis::cmd("HGET")
            .arg(&conn_key)
            .arg("low_endpoint_info")
            .query_async(&mut conn)
            .await
            .unwrap_or(None);
        let high_info: Option<String> = redis::cmd("HGET")
            .arg(&conn_key)
            .arg("high_endpoint_info")
            .query_async(&mut conn)
            .await
            .unwrap_or(None);

        if low_info.is_some() && high_info.is_some() {
            // Both ACKs received, mark as initialized
            let _: () = redis::cmd("HSET")
                .arg(&conn_key)
                .arg("status")
                .arg("initialized")
                .query_async(&mut conn)
                .await
                .unwrap_or_default();
            return Json(InitResponse {
                status: "ok".to_string(),
                message: "Initialized successfully".to_string(),
            });
        }

        if let Some(s) = status {
            if s == "initialized" {
                return Json(InitResponse {
                    status: "ok".to_string(),
                    message: "Initialized successfully".to_string(),
                });
            }
        }
        retries -= 1;
    }

    Json(InitResponse {
        status: "timeout".to_string(),
        message: "Timeout waiting for init ACK".to_string(),
    })
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

    let mut conn = state
        .redis_client
        .get_multiplexed_async_connection()
        .await
        .unwrap();

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

    // Publish connect event
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

    // Wait for connection to complete
    let mut retries = 50;
    while retries > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
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
                    message: "Connected successfully".to_string(),
                });
            }
        }
        retries -= 1;
    }

    Json(ConnectResponse {
        status: "timeout".to_string(),
        message: "Timeout waiting for connect ACK".to_string(),
    })
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

    let mut conn = state
        .redis_client
        .get_multiplexed_async_connection()
        .await
        .unwrap();

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
    let mut conn = state
        .redis_client
        .get_multiplexed_async_connection()
        .await
        .unwrap();

    let mr_key = format!("mr:{}:{}", body.dst, body.mr_name);
    let mr_info_str: Option<String> = redis::cmd("GET")
        .arg(&mr_key)
        .query_async(&mut conn)
        .await
        .unwrap_or(None);

    if let Some(s) = mr_info_str {
        if let Ok(mr_json) = serde_json::from_str::<serde_json::Value>(&s) {
            if let (Some(addr), Some(length), Some(rkey), Some(lkey)) = (
                mr_json["addr"].as_u64(),
                mr_json["length"].as_u64(),
                mr_json["rkey"].as_u64(),
                mr_json["lkey"].as_u64(),
            ) {
                return Json(GetMrInfoResponse {
                    mr_info: Some(MrInfo {
                        addr,
                        length: length as usize,
                        rkey: rkey as u32,
                        lkey: lkey as u32,
                    }),
                });
            }
        }
    }

    Json(GetMrInfoResponse { mr_info: None })
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
    let mut conn = state
        .redis_client
        .get_multiplexed_async_connection()
        .await
        .unwrap();

    // Determine which endpoint info to get
    let field_name = if body.dst == low {
        "low_endpoint_info"
    } else {
        "high_endpoint_info"
    };

    let endpoint_info_str: Option<String> = redis::cmd("HGET")
        .arg(&conn_key)
        .arg(field_name)
        .query_async(&mut conn)
        .await
        .unwrap_or(None);

    if let Some(s) = endpoint_info_str {
        if let Ok(endpoint_info) = serde_json::from_str::<serde_json::Value>(&s) {
            return Json(GetEndpointInfoResponse {
                endpoint_info: Some(endpoint_info),
            });
        }
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
    let mut conn = state
        .redis_client
        .get_multiplexed_async_connection()
        .await
        .unwrap();

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

    // Check if both endpoints have been initialized
    let low_info: Option<String> = redis::cmd("HGET")
        .arg(&conn_key)
        .arg("low_endpoint_info")
        .query_async(&mut conn)
        .await
        .unwrap_or(None);
    let high_info: Option<String> = redis::cmd("HGET")
        .arg(&conn_key)
        .arg("high_endpoint_info")
        .query_async(&mut conn)
        .await
        .unwrap_or(None);

    if low_info.is_some() && high_info.is_some() {
        let _: () = redis::cmd("HSET")
            .arg(&conn_key)
            .arg("status")
            .arg("initialized")
            .query_async(&mut conn)
            .await
            .unwrap_or_default();
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
    let mut conn = state
        .redis_client
        .get_multiplexed_async_connection()
        .await
        .unwrap();

    // Mark as connected (simplified: assume both sides ACK)
    let _: () = redis::cmd("HSET")
        .arg(&conn_key)
        .arg("status")
        .arg("connected")
        .query_async(&mut conn)
        .await
        .unwrap_or_default();

    Json(AckResponse {
        status: "ok".to_string(),
    })
}

async fn update_endpoint_info(
    State(state): State<AppState>,
    Json(body): Json<UpdateEndpointInfoBody>,
) -> impl IntoResponse {
    let mut conn = state
        .redis_client
        .get_multiplexed_async_connection()
        .await
        .unwrap();

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
    let mut conn = state
        .redis_client
        .get_multiplexed_async_connection()
        .await
        .unwrap();

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
