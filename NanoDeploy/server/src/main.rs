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
use crate::engine_adapter::EngineAdapter;
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

    // Phase 1: Engine Manager & Launch
    let mut engine_mgr = engine_manager::EngineManager::new();
    if let Err(e) = engine_mgr.launch_engine(&config.engine).await {
        error!("Failed to launch engine: {}", e);
        return Err(e);
    }

    // Phase 1: Connection (Placeholder IP - in real system, Engine reports IP or we assign port)
    // For now, assume Engine binds to a known port or we wait for logic.
    // Spec says: "Rust sends Ping".
    // We will try to connect to a default port (e.g. 5000) for demo

    info!("Waiting for Engine to startup...");
    // tokio::time::sleep(std::time::Duration::from_secs(2)).await; // Moved to manager

    // Phase 1: Connect
    // Phase 1: Connect
    // Phase 1: Connect
    let engine_host = config.server.engine_host.unwrap_or_else(|| "127.0.0.1".to_string());
    let engine_port = config.server.engine_port.unwrap_or(5000);
    let engine_addr = format!("{}:{}", engine_host, engine_port);
    info!("Connecting to Engine at {}", engine_addr);

    let mut client = EngineAdapter::new("rust_router".to_string());
    loop {
        match client.connect(&engine_addr).await {
            Ok(_) => {
                info!("Connected to Engine!");
                break;
            }
            Err(e) => {
                error!("Failed to connect to Engine: {}. Retrying in 2s...", e);
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
        }
    }

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
    let engine_adapter = Arc::new(Mutex::new(client));
    let tokenizer_service_arc = Arc::new(tokenizer_service);

    info!("Starting HTTP Server on port {}", config.server.port);
    http_server::start_server(config.server.port, engine_manager, engine_adapter, tokenizer_service_arc).await;

    // info!("Server is running. Press Ctrl+C to stop.");
    // tokio::signal::ctrl_c().await?;
    // info!("Shutting down...");

    Ok(())
}
