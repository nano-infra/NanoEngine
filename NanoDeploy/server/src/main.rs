mod config;
mod engine_manager;
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
use tracing::{error, info};
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

    if let Some(etcd_config) = config.etcd.clone() {
        info!("Starting Etcd Discovery...");
        // To start watch, we need the Arc<Mutex<Manager>>.
        // But we construct it later for HttpServer.
        // We can wrap it now.
        let mgr_arc = Arc::new(Mutex::new(engine_mgr));
        engine_manager::EngineManager::start_etcd_watch(mgr_arc.clone(), etcd_config).await;

        // Wait a bit for initial discovery? Or just proceed.
        // Since discovery is async, engines might not be ready immediately.
        // But the server can start listening.
        // We reassign engine_mgr to keep the old variable name meaningful if needed,
        // but actually we should just use mgr_arc from now on.

        // However, the rest of the code expects `engine_mgr` (the struct) to perform specialized init.
        // But we moved ownership to Arc.
        // We can clone the arc for the HTTP server provided we change the code below.

        // Refactoring: Let's change the flow slightly.

        // If we use Etcd, we skip manual connect_all.
        // And we skip P2P init done by server, because engines do it themselves via Etcd.

        info!("Etcd Discovery started. Skipping manual connect and P2P init.");

        // Phase 2: Tokenizer
        info!("Initializing Tokenizer...");
        let mut tokenizer_service = tokenizer::TokenizerService::new(&config.tokenizer.path);
        if let Err(e) = tokenizer_service.load().await {
            error!("Failed to load tokenizer: {}", e);
        }
        let tokenizer_service_arc = Arc::new(tokenizer_service);

        // Phase 3: Start HTTP Server
        info!("Starting HTTP Server on port {}", config.server.port);
        http_server::start_server(config.server.port, mgr_arc, tokenizer_service_arc).await;

        return Ok(());
    }

    // Legacy/Manual Mode

    // This replaces launch and manual connect loop
    info!("Connecting to Engines (Manual Config)...");
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
