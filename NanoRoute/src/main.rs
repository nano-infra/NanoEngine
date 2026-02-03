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
    mod connection_generated {
        include!(concat!(env!("OUT_DIR"), "/connection_generated.rs"));
    }

    pub use self::connection_generated::nanodeploy::fbs::*;
    pub use self::sequence_generated::nanodeploy::fbs::*;
}

mod engine_adapter;
mod http_server;
mod tokenizer;

use clap::Parser;
use config::AppConfig;
use std::path::PathBuf;
use tracing::{error, info, warn};
// use crate::engine_adapter::EngineAdapter; // Removed
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    #[arg(long)]
    tokenizer_path: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nanodeploy_server=info,tower_http=info".into()),
        )
        .init();

    let args = Args::parse();
    info!("Loading configuration from {:?}", args.config);
    let mut config = AppConfig::load_from_file(&args.config)?;

    if let Some(tp) = args.tokenizer_path {
        info!("Overriding tokenizer path with: {}", tp);
        config.tokenizer.path = tp;
    }

    info!("Configuration loaded.");

    // Phase 1: Engine Manager & Connect
    // Get NanoCtrl address from config (required for dynamic discovery)
    let nanoctrl_address = match &config.engine {
        config::EngineConfig::Unified {
            nanoctrl_address, ..
        } => nanoctrl_address.clone(),
        config::EngineConfig::Disaggregated {
            nanoctrl_address, ..
        } => nanoctrl_address.clone(),
    };

    let nanoctrl_address = match nanoctrl_address {
        Some(addr) => addr,
        None => {
            error!("nanoctrl_address is required for dynamic service discovery");
            return Err(anyhow::anyhow!(
                "nanoctrl_address must be configured in config.toml"
            ));
        }
    };

    info!("Using NanoCtrl at: {}", nanoctrl_address);

    // Get Redis URL from NanoCtrl
    let engine_mgr = engine_manager::EngineManager::new();
    let redis_url = match engine_mgr
        .get_redis_url_from_nanoctrl(&nanoctrl_address)
        .await
    {
        Ok(url) => {
            info!("Retrieved Redis URL from NanoCtrl: {}", url);
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
    info!(
        "Starting dynamic service discovery with Redis: {}",
        redis_url
    );
    let engine_manager = match engine_mgr
        .start_dynamic_discovery(redis_url, Some(nanoctrl_address.clone()))
        .await
    {
        Ok(manager_arc) => {
            info!("Dynamic service discovery started successfully.");
            // Log engine counts
            let manager = manager_arc.lock().await;
            let prefill_count = manager.prefill_engines.len();
            let decode_count = manager.decode_engines.len();
            drop(manager);
            info!(
                "Connected engines: {} prefill, {} decode",
                prefill_count, decode_count
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

    // Phase 2: Tokenizer
    info!("Initializing Tokenizer...");
    let mut tokenizer_service = tokenizer::TokenizerService::new(&config.tokenizer.path);
    if let Err(e) = tokenizer_service.load().await {
        error!("Failed to load tokenizer: {}", e);
        // Continue without tokenizer? Or fail?
        // For now, let's log and continue, maybe usage will fail gracefully.
    }

    // Phase 3: Start HTTP Server
    let tokenizer_service_arc = Arc::new(tokenizer_service);

    info!("Starting HTTP Server on port {}", config.server.port);
    http_server::start_server(config.server.port, engine_manager, tokenizer_service_arc).await;

    Ok(())
}
