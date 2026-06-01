mod config;
mod engine_manager;
mod engine_watcher;
mod zmq_packet;
#[allow(warnings)]
pub mod fbs {
    #[allow(clippy::all)]
    mod sequence_generated {
        include!(concat!(env!("OUT_DIR"), "/sequence_generated.rs"));
    }

    #[allow(clippy::all)]
    mod packet_generated {
        include!(concat!(env!("OUT_DIR"), "/packet_generated.rs"));
    }

    pub use self::packet_generated::nanodeploy::fbs::*;
    pub use self::sequence_generated::nanodeploy::fbs::*;
}

mod encoder_adapter;
mod engine_adapter;
mod http_server;
mod tokenizer;
mod tool_parser;

use clap::Parser;
use config::AppConfig;
use std::path::PathBuf;
use tracing::{debug, error, info};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nanoroute=info,tower_http=info".into()),
        )
        .init();

    let args = Args::parse();
    info!("Loading configuration from {:?}", args.config);
    let config = AppConfig::load_from_file(&args.config)?;

    debug!("Configuration loaded.");

    // Phase 1: Engine Manager & Connect
    // Get NanoCtrl address from config (required for dynamic discovery)
    let ctrl_address = match &config.engine {
        config::EngineConfig::Unified { ctrl_address, .. } => ctrl_address.clone(),
        config::EngineConfig::Disaggregated { ctrl_address, .. } => ctrl_address.clone(),
    };

    let ctrl_address = match ctrl_address {
        Some(addr) => addr,
        None => {
            error!("ctrl_address is required for dynamic service discovery");
            return Err(anyhow::anyhow!(
                "ctrl_address must be configured in config.toml"
            ));
        }
    };

    debug!("Using dlslime-ctrl at: {}", ctrl_address);

    // Get Redis URL from dlslime-ctrl (once per process; if you see 3x in the log, check for 3 router instances)
    // Read scope from config, fall back to DLSLIME_CTRL_SCOPE env var.
    let config_scope = match &config.engine {
        config::EngineConfig::Unified { ctrl_scope, .. } => ctrl_scope.clone(),
        config::EngineConfig::Disaggregated { ctrl_scope, .. } => ctrl_scope.clone(),
    };
    let redis_scope = config_scope.or_else(|| std::env::var("DLSLIME_CTRL_SCOPE").ok());
    let engine_mgr = engine_manager::EngineManager::with_scope(redis_scope);
    let redis_url = match engine_mgr.get_redis_url_from_ctrl(&ctrl_address).await {
        Ok(url) => {
            debug!("Retrieved Redis URL from NanoCtrl: {}", url);
            url
        }
        Err(e) => {
            error!("Failed to get Redis URL from NanoCtrl: {}", e);
            return Err(anyhow::anyhow!(
                "Failed to get Redis URL from NanoCtrl: {}. Make sure NanoCtrl is running and accessible.",
                e
            ));
        }
    };

    // Start dynamic service discovery (only mode)
    debug!(
        "Starting dynamic service discovery with Redis: {}",
        redis_url
    );
    let engine_manager = match engine_mgr
        .start_dynamic_discovery(redis_url, Some(ctrl_address.clone()))
        .await
    {
        Ok(manager_arc) => {
            debug!("Dynamic service discovery started successfully.");
            // Log engine counts
            let manager = manager_arc.lock().await;
            let (prefill_count, decode_count, encoder_count) = manager.total_engine_counts();
            let model_keys: Vec<String> = manager
                .available_model_keys()
                .iter()
                .map(|s| s.to_string())
                .collect();
            drop(manager);
            info!(
                "Connected engines: {} prefill, {} decode, {} encoder, models: {:?}",
                prefill_count, decode_count, encoder_count, model_keys
            );
            manager_arc
        }
        Err(e) => {
            error!("Failed to start dynamic service discovery: {}", e);
            return Err(anyhow::anyhow!(
                "Failed to start dynamic service discovery: {}. Please check Redis connection and NanoCtrl status.",
                e
            ));
        }
    };

    // P2P mesh is handled by NanoCtrl microservice

    // Phase 2: Start HTTP Server
    info!("Starting HTTP Server on port {}", config.server.port);
    http_server::start_server(config.server.port, engine_manager).await;

    Ok(())
}
