pub mod engine_manager;
pub mod http_server;

use clap::Parser;
use pyo3::prelude::*;
use tracing::{debug, error, info};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 3001)]
    port: u16,

    #[arg(long, default_value = "http://127.0.0.1:4479")]
    ctrl_address: String,

    #[arg(long)]
    ctrl_scope: Option<String>,
}

fn normalize_ctrl_address(addr: &str) -> String {
    if addr.starts_with("http://") || addr.starts_with("https://") {
        addr.to_string()
    } else {
        format!("http://{}", addr)
    }
}

async fn run(args: Args) -> anyhow::Result<()> {
    let ctrl_address = normalize_ctrl_address(&args.ctrl_address);

    debug!("Using dlslime-ctrl at: {}", ctrl_address);

    let ctrl_scope = args
        .ctrl_scope
        .or_else(|| std::env::var("DLSLIME_CTRL_SCOPE").ok());
    let engine_mgr = engine_manager::EngineManager::with_scope(ctrl_scope);

    debug!("Starting dynamic service discovery from NanoCtrl API");
    let engine_manager = match engine_mgr
        .start_dynamic_discovery(Some(ctrl_address.clone()))
        .await
    {
        Ok(manager_arc) => {
            debug!("Dynamic service discovery started successfully.");
            let manager = manager_arc.lock().await;
            let (prefill_count, decode_count, encoder_count) = manager.total_engine_counts();
            let (hybrid_count, http_prefill_count, http_decode_count) =
                manager.total_http_role_counts();
            let model_keys: Vec<String> = manager
                .available_model_keys()
                .iter()
                .map(|s| s.to_string())
                .collect();
            drop(manager);
            info!(
                "Connected DLEngine HTTP nodes: {} hybrid, {} prefill, {} decode; route counts: {} prefill, {} decode, {} encoder; models: {:?}",
                hybrid_count,
                http_prefill_count,
                http_decode_count,
                prefill_count,
                decode_count,
                encoder_count,
                model_keys
            );
            manager_arc
        }
        Err(e) => {
            error!("Failed to start dynamic service discovery: {}", e);
            return Err(anyhow::anyhow!(
                "Failed to start dynamic service discovery: {}. Please check NanoCtrl status.",
                e
            ));
        }
    };

    info!("Starting HTTP Server on port {}", args.port);
    http_server::start_server(args.port, engine_manager).await?;

    Ok(())
}

pub fn run_from_iter<I, T>(args: I) -> anyhow::Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .try_init();

    let args = Args::parse_from(args);
    tokio::runtime::Runtime::new()?.block_on(run(args))
}

#[pyfunction]
#[allow(dead_code)]
fn run_router(py: Python<'_>, args: Vec<String>) -> PyResult<()> {
    let argv = std::iter::once("dlengine-router".to_string()).chain(args);
    #[allow(deprecated)]
    py.allow_threads(|| run_from_iter(argv))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[allow(dead_code)]
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_router, m)?)?;
    Ok(())
}
