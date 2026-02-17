//! NanoCtrl: Control plane server for NanoInfra
//!
//! NanoCtrl is stateless and supports multiple scopes sharing the same instance.
//! Scope is determined by clients (NanoRoute, EngineServer, peer_agent) via
//! `NANOCTRL_SCOPE` env var.

mod config;
mod error;
mod handlers;
mod models;
mod net;
mod redis_repo;
mod state;

use axum::{
    routing::{get, post},
    Router,
};
use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use tower::ServiceBuilder;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::config::AppConfig;
use crate::redis_repo::{LuaScripts, RedisRepo};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("NANOCTRL_RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();
    tracing::info!("Loading configuration from {:?}", args.config);
    let mut config = AppConfig::load_from_file(&args.config)?;

    // Env override for Redis URL (optional)
    if let Ok(url) = std::env::var("NANOCTRL_REDIS_URL") {
        config.redis.url = url;
    }

    tracing::info!("Using Redis URL: {}", config.redis.url);

    // Load Lua scripts from external files
    let scripts = LuaScripts::load()?;
    tracing::info!("Loaded Lua scripts from lua/ directory");

    // Create Redis repository with connection pool
    let repo = RedisRepo::new(&config.redis.url, scripts)?;
    tracing::info!("Redis connection pool initialized");

    // Warm up: verify Redis is reachable
    {
        tracing::info!("Warming up Redis connection...");
        let mut conn = repo.conn().await?;
        let _: String = redis::cmd("PING").query_async(&mut *conn).await?;
        tracing::info!("Redis connection established successfully");
    }

    let app = Router::new()
        // Health
        .route("/", get(handlers::util::root))
        // Peer agent
        .route("/start_peer_agent", post(handlers::peer::start_peer_agent))
        .route("/query", post(handlers::peer::query))
        .route("/cleanup", post(handlers::peer::cleanup))
        // RDMA
        .route(
            "/v1/desired_topology/:agent_id",
            post(handlers::rdma::set_desired_topology),
        )
        .route("/register_mr", post(handlers::rdma::register_mr))
        .route("/get_mr_info", post(handlers::rdma::get_mr_info))
        // Engine
        .route("/register_engine", post(handlers::engine::register_engine))
        .route(
            "/unregister_engine",
            post(handlers::engine::unregister_engine),
        )
        .route(
            "/heartbeat_engine",
            post(handlers::engine::heartbeat_engine),
        )
        .route("/get_engine_info", post(handlers::engine::get_engine_info))
        .route("/list_engines", post(handlers::engine::list_engines))
        // Utility
        .route(
            "/get_redis_address",
            post(handlers::util::get_redis_address),
        )
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
                        tracing::debug!("Incoming request: {} {}", request.method(), request.uri());
                    })
                    .on_response(
                        |response: &axum::http::Response<_>,
                         latency: std::time::Duration,
                         _span: &tracing::Span| {
                            tracing::info!(
                                status = %response.status(),
                                latency_us = latency.as_micros(),
                                "api done"
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
        .with_state(repo);

    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port)
        .parse()
        .map_err(|e| {
            anyhow::anyhow!(
                "Invalid server address {}:{}: {}",
                config.server.host,
                config.server.port,
                e
            )
        })?;
    tracing::info!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
