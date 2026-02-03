mod config;
mod engine_manager;
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
    let mut engine_mgr = engine_manager::EngineManager::new();

    // This replaces launch and manual connect loop
    info!("Connecting to Engines (Manual Config)...");
    match engine_mgr.connect_all(&config.engine).await {
        Ok(_) => {
            info!("Engine connection completed.");
        }
        Err(e) => {
            error!("Failed to connect to engines: {}", e);
            warn!("Continuing without engines - HTTP server will start but requests may fail");
        }
    }

    // Log engine counts
    let prefill_count = engine_mgr.prefill_engines.len();
    let decode_count = engine_mgr.decode_engines.len();
    info!(
        "Connected engines: {} prefill, {} decode",
        prefill_count, decode_count
    );

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
    let engine_manager = Arc::new(Mutex::new(engine_mgr));
    let tokenizer_service_arc = Arc::new(tokenizer_service);

    info!("Starting HTTP Server on port {}", config.server.port);
    http_server::start_server(config.server.port, engine_manager, tokenizer_service_arc).await;

    Ok(())
}
