use crate::config::EngineConfig;
use crate::engine_adapter::EngineAdapter;

use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

pub struct EngineManager {
    // We share adapters via Arc<Mutex> because multiple threads (http server) might access them
    pub prefill_engines: Vec<Arc<Mutex<EngineAdapter>>>,
    pub decode_engines: Vec<Arc<Mutex<EngineAdapter>>>,
}

impl Default for EngineManager {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineManager {
    pub fn new() -> Self {
        Self {
            prefill_engines: Vec::new(),
            decode_engines: Vec::new(),
        }
    }

    pub async fn connect_all(&mut self, config: &EngineConfig) -> anyhow::Result<()> {
        match config {
            EngineConfig::Unified {
                host,
                port,
                nanoctrl_address,
            } => {
                if let Some(nanoctrl_addr) = nanoctrl_address {
                    // Query engines from NanoCtrl
                    info!("Querying engines from NanoCtrl at {}", nanoctrl_addr);
                    match self.list_engines_from_nanoctrl(nanoctrl_addr).await {
                        Ok(engines) => {
                            let engines_count = engines.len();
                            info!("Found {} engines from NanoCtrl", engines_count);
                            let mut connected_count = 0;
                            for engine_info in engines {
                                // Parse port - could be u64 or string
                                let port_num = engine_info["port"]
                                    .as_u64()
                                    .or_else(|| {
                                        engine_info["port"]
                                            .as_str()
                                            .and_then(|s| s.parse::<u64>().ok())
                                    })
                                    .unwrap_or(0);

                                if let (Some(host_str), Some(role)) =
                                    (engine_info["host"].as_str(), engine_info["role"].as_str())
                                {
                                    if port_num == 0 {
                                        warn!(
                                            "Skipping engine with invalid port: {:?}",
                                            engine_info
                                        );
                                        continue;
                                    }
                                    // Handle 0.0.0.0 host - use localhost instead
                                    let connect_host = if host_str == "0.0.0.0" {
                                        "127.0.0.1"
                                    } else {
                                        host_str
                                    };
                                    let addr = format!("{}:{}", connect_host, port_num);
                                    let engine_id = engine_info["id"].as_str().unwrap_or("unknown");
                                    info!(
                                        "Attempting to connect to {} engine {} at {}",
                                        role, engine_id, addr
                                    );
                                    let mut adapter = EngineAdapter::new(engine_id.to_string());

                                    match adapter.connect(&addr).await {
                                        Ok(_) => {
                                            if let Some(id) = engine_info["id"].as_str() {
                                                adapter.uuid = Some(id.to_string());
                                            }
                                            if let Some(ws) = engine_info["world_size"].as_u64() {
                                                adapter.world_size = ws as i32;
                                            }
                                            if let Some(nb) = engine_info["num_blocks"].as_u64() {
                                                adapter.num_blocks = nb as i32;
                                            }

                                            let adapter = Arc::new(Mutex::new(adapter));
                                            if role == "prefill" {
                                                self.prefill_engines.push(adapter);
                                                info!(
                                                    "Connected to prefill engine {} at {}:{}",
                                                    engine_id, connect_host, port_num
                                                );
                                                connected_count += 1;
                                            } else if role == "decode" {
                                                self.decode_engines.push(adapter);
                                                info!(
                                                    "Connected to decode engine {} at {}:{}",
                                                    engine_id, connect_host, port_num
                                                );
                                                connected_count += 1;
                                            } else {
                                                // Unified or hybrid
                                                self.prefill_engines.push(adapter.clone());
                                                self.decode_engines.push(adapter);
                                                info!(
                                                    "Connected to {} engine {} at {}:{}",
                                                    role, engine_id, connect_host, port_num
                                                );
                                                connected_count += 1;
                                            }
                                        }
                                        Err(e) => {
                                            warn!(
                                                "Failed to connect to {} engine {} at {}: {}",
                                                role, engine_id, addr, e
                                            );
                                        }
                                    }
                                } else {
                                    warn!(
                                        "Skipping engine with missing host or role: {:?}",
                                        engine_info
                                    );
                                }
                            }
                            info!(
                                "Successfully connected to {}/{} engines from NanoCtrl",
                                connected_count, engines_count
                            );
                        }
                        Err(e) => {
                            error!("Failed to query engines from NanoCtrl: {}", e);
                            anyhow::bail!("Failed to query engines from NanoCtrl: {}", e);
                        }
                    }
                } else {
                    // Fallback to static config
                    info!("Connecting to Unified Engine at {}:{}", host, port);
                    let addr = format!("{}:{}", host, port);
                    let mut adapter = EngineAdapter::new(format!("unified-{}", port));
                    adapter.connect(&addr).await?;

                    let adapter = Arc::new(Mutex::new(adapter));
                    self.prefill_engines.push(adapter.clone());
                    self.decode_engines.push(adapter);
                }
            }
            EngineConfig::Disaggregated {
                prefill,
                decode,
                nanoctrl_address,
            } => {
                if let Some(nanoctrl_addr) = nanoctrl_address {
                    // Query engines from NanoCtrl
                    info!("Querying engines from NanoCtrl at {}", nanoctrl_addr);
                    match self.list_engines_from_nanoctrl(nanoctrl_addr).await {
                        Ok(engines) => {
                            let engines_count = engines.len();
                            info!("Found {} engines from NanoCtrl", engines_count);
                            let mut connected_count = 0;
                            for engine_info in engines {
                                // Parse port - could be u64 or string
                                let port_num = engine_info["port"]
                                    .as_u64()
                                    .or_else(|| {
                                        engine_info["port"]
                                            .as_str()
                                            .and_then(|s| s.parse::<u64>().ok())
                                    })
                                    .unwrap_or(0);

                                if let (Some(host_str), Some(role)) =
                                    (engine_info["host"].as_str(), engine_info["role"].as_str())
                                {
                                    if port_num == 0 {
                                        warn!(
                                            "Skipping engine with invalid port: {:?}",
                                            engine_info
                                        );
                                        continue;
                                    }
                                    // Handle 0.0.0.0 host - use localhost instead
                                    let connect_host = if host_str == "0.0.0.0" {
                                        "127.0.0.1"
                                    } else {
                                        host_str
                                    };
                                    let addr = format!("{}:{}", connect_host, port_num);
                                    let engine_id = engine_info["id"].as_str().unwrap_or("unknown");
                                    info!(
                                        "Attempting to connect to {} engine {} at {}",
                                        role, engine_id, addr
                                    );
                                    let mut adapter = EngineAdapter::new(engine_id.to_string());

                                    match adapter.connect(&addr).await {
                                        Ok(_) => {
                                            if let Some(id) = engine_info["id"].as_str() {
                                                adapter.uuid = Some(id.to_string());
                                            }
                                            if let Some(ws) = engine_info["world_size"].as_u64() {
                                                adapter.world_size = ws as i32;
                                            }
                                            if let Some(nb) = engine_info["num_blocks"].as_u64() {
                                                adapter.num_blocks = nb as i32;
                                            }

                                            let adapter = Arc::new(Mutex::new(adapter));
                                            if role == "prefill" {
                                                self.prefill_engines.push(adapter);
                                                info!(
                                                    "Connected to prefill engine {} at {}:{}",
                                                    engine_id, connect_host, port_num
                                                );
                                                connected_count += 1;
                                            } else if role == "decode" {
                                                self.decode_engines.push(adapter);
                                                info!(
                                                    "Connected to decode engine {} at {}:{}",
                                                    engine_id, connect_host, port_num
                                                );
                                                connected_count += 1;
                                            }
                                        }
                                        Err(e) => {
                                            warn!(
                                                "Failed to connect to {} engine {} at {}: {}",
                                                role, engine_id, addr, e
                                            );
                                        }
                                    }
                                } else {
                                    warn!(
                                        "Skipping engine with missing host or role: {:?}",
                                        engine_info
                                    );
                                }
                            }
                            info!(
                                "Successfully connected to {}/{} engines from NanoCtrl",
                                connected_count, engines_count
                            );
                        }
                        Err(e) => {
                            error!("Failed to query engines from NanoCtrl: {}", e);
                            anyhow::bail!("Failed to query engines from NanoCtrl: {}", e);
                        }
                    }
                } else {
                    // Fallback to static config
                    for (i, node) in prefill.iter().enumerate() {
                        info!(
                            "Connecting to Prefill Engine #{} at {}:{}",
                            i, node.host, node.port
                        );
                        let addr = format!("{}:{}", node.host, node.port);
                        let mut adapter = EngineAdapter::new(format!("prefill-{}", i));
                        adapter.connect(&addr).await?;

                        // Fetch real UUID and specs
                        if let Ok(info) = adapter.send_get_engine_info().await {
                            if let Some(id) = info["id"].as_str() {
                                adapter.uuid = Some(id.to_string());
                                info!("Prefill Engine #{} UUID: {}", i, id);
                            }
                            if let Some(ws) = info["world_size"].as_i64() {
                                adapter.world_size = ws as i32;
                            }
                            if let Some(nb) = info["num_blocks"].as_i64() {
                                adapter.num_blocks = nb as i32;
                            }
                        }

                        self.prefill_engines.push(Arc::new(Mutex::new(adapter)));
                    }

                    for (i, node) in decode.iter().enumerate() {
                        info!(
                            "Connecting to Decode Engine #{} at {}:{}",
                            i, node.host, node.port
                        );
                        let addr = format!("{}:{}", node.host, node.port);
                        let mut adapter = EngineAdapter::new(format!("decode-{}", i));
                        adapter.connect(&addr).await?;

                        // Fetch real UUID and specs
                        if let Ok(info) = adapter.send_get_engine_info().await {
                            if let Some(id) = info["id"].as_str() {
                                adapter.uuid = Some(id.to_string());
                                info!("Decode Engine #{} UUID: {}", i, id);
                            }
                            if let Some(ws) = info["world_size"].as_i64() {
                                adapter.world_size = ws as i32;
                            }
                            if let Some(nb) = info["num_blocks"].as_i64() {
                                adapter.num_blocks = nb as i32;
                            }
                        }

                        self.decode_engines.push(Arc::new(Mutex::new(adapter)));
                    }
                }
            }
        }
        Ok(())
    }

    // Helper to get a round-robin engine (simplest scheduler)
    pub fn get_next_prefill(&self) -> Option<Arc<Mutex<EngineAdapter>>> {
        self.prefill_engines.first().cloned()
    }

    pub fn get_next_decode(&self) -> Option<Arc<Mutex<EngineAdapter>>> {
        self.decode_engines.first().cloned()
    }

    /// List all engines from NanoCtrl
    pub async fn list_engines_from_nanoctrl(
        &self,
        nanoctrl_address: &str,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let client = reqwest::Client::new();
        let url = format!("{}/list_engines", nanoctrl_address);

        let body = serde_json::json!({});

        let response = client.post(&url).json(&body).send().await?;

        if response.status().is_success() {
            let result: serde_json::Value = response.json().await?;
            if let Some(status) = result.get("status").and_then(|s| s.as_str()) {
                if status == "ok" {
                    if let Some(engines) = result.get("engines").and_then(|e| e.as_array()) {
                        let engines: Vec<serde_json::Value> = engines.clone();
                        info!("Found {} engines from NanoCtrl", engines.len());
                        return Ok(engines);
                    }
                }
            }
        }

        Err(anyhow::anyhow!("Failed to list engines from NanoCtrl"))
    }
}
