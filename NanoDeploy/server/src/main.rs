mod config;
mod engine_manager;
mod tokenizer;
mod engine_adapter;
mod fbs;
mod http_server;

use clap::Parser;
use config::AppConfig;
use std::path::PathBuf;
use tracing::{info, error};
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
    info!("Connecting to Engines...");
    if let Err(e) = engine_mgr.connect_all(&config.engine).await {
        error!("Failed to connect to engines: {}", e);
        return Err(e);
    }
    info!("All Engines connected.");

    // Phase 1.5: P2P Handshake
    info!("Initializing P2P Mesh...");
    if let Err(e) = engine_mgr.initialize_p2p_mesh(&config.engine).await {
         error!("Failed to initialize P2P mesh: {}", e);
         return Err(e);
    }
    info!("P2P Mesh Initialized.");

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
