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

    debug!("Using NanoCtrl at: {}", nanoctrl_address);

    // Get Redis URL from NanoCtrl (once per process; if you see 3x in NanoCtrl log, check for 3 router instances)
    // Read scope from config, fall back to NANOCTRL_SCOPE env var
    let config_scope = match &config.engine {
        config::EngineConfig::Unified { nanoctrl_scope, .. } => nanoctrl_scope.clone(),
        config::EngineConfig::Disaggregated { nanoctrl_scope, .. } => nanoctrl_scope.clone(),
    };
    let redis_scope = config_scope.or_else(|| std::env::var("NANOCTRL_SCOPE").ok());
    let engine_mgr = engine_manager::EngineManager::with_scope(redis_scope);
    let redis_url = match engine_mgr
        .get_redis_url_from_nanoctrl(&nanoctrl_address)
        .await
    {
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
        .start_dynamic_discovery(redis_url, Some(nanoctrl_address.clone()))
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
