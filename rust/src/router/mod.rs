pub mod engine_manager;
pub mod http_server;

use clap::Parser;
use pyo3::prelude::*;
use std::fs;
use std::process::{Command, Stdio};
use tracing::{debug, error, info};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<CommandKind>,
    #[arg(short, long, global = true, default_value_t = 3001)]
    port: u16,

    #[arg(long, global = true, default_value = "http://127.0.0.1:4479")]
    ctrl_address: String,

    #[arg(long, global = true)]
    ctrl_scope: Option<String>,

    /// Run in background (accepts true/yes/1).
    #[arg(long, global = true, value_name = "BOOL", default_value = "false", action = clap::ArgAction::Set)]
    daemonize: String,
}

#[derive(clap::Subcommand, Debug, Clone)]
enum CommandKind {
    Start,
    Status,
    Stop,
}

fn runtime_dir() -> std::path::PathBuf {
    std::env::var_os("DLENGINE_ROUTER_RUNTIME_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp/dlengine-router"))
}
fn pid_file() -> std::path::PathBuf {
    runtime_dir().join("dlengine-router.pid")
}
fn read_pid() -> Option<u32> {
    fs::read_to_string(pid_file()).ok()?.trim().parse().ok()
}
fn running(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
fn daemonize(args: &Args) -> anyhow::Result<()> {
    fs::create_dir_all(runtime_dir())?;
    if let Some(pid) = read_pid() {
        if running(pid) {
            println!("dlengine-router is already running (pid={pid})");
            return Ok(());
        }
    }
    let exe = std::env::var_os("DLENGINE_ROUTER_EXECUTABLE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let e = std::env::current_exe()
                .unwrap_or_else(|_| std::path::PathBuf::from("dlengine-router"));
            if e.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.contains("python"))
                .unwrap_or(false)
            {
                std::path::PathBuf::from("dlengine-router")
            } else {
                e
            }
        });
    let mut a = vec![
        "--port".into(),
        args.port.to_string(),
        "--ctrl-address".into(),
        args.ctrl_address.clone(),
    ];
    if let Some(s) = &args.ctrl_scope {
        a.extend(["--ctrl-scope".into(), s.clone()]);
    }
    let c = Command::new(exe)
        .args(a)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    fs::write(pid_file(), c.id().to_string())?;
    println!("dlengine-router started (pid={})", c.id());
    Ok(())
}
fn ops(args: &Args, c: CommandKind) -> anyhow::Result<()> {
    match c {
        CommandKind::Start => daemonize(args),
        CommandKind::Status => {
            if let Some(p) = read_pid() {
                if running(p) {
                    println!("dlengine-router is running (pid={p})");
                    let url = format!("http://127.0.0.1:{}/status", args.port);
                    if let Ok(out) = Command::new("curl").args(["-fsS", &url]).output() {
                        if out.status.success() {
                            println!("{}", String::from_utf8_lossy(&out.stdout));
                        } else {
                            println!("health: unavailable");
                        }
                    }
                    return Ok(());
                }
            }
            println!("dlengine-router is not running");
            Ok(())
        }
        CommandKind::Stop => {
            if let Some(p) = read_pid() {
                if running(p) {
                    Command::new("kill")
                        .args(["-TERM", &p.to_string()])
                        .status()?;
                    println!("dlengine-router stopped (pid={p})");
                }
                let _ = fs::remove_file(pid_file());
            } else {
                println!("dlengine-router is not running");
            }
            Ok(())
        }
    }
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
    if let Some(command) = args.command.as_ref() {
        return ops(&args, command.clone());
    }
    if matches!(args.daemonize.as_str(), "true" | "yes" | "1") {
        return daemonize(&args);
    }
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
