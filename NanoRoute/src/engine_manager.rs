use crate::config::EngineConfig;
use crate::engine_adapter::EngineAdapter;
use crate::engine_watcher::{EngineEvent, EnginePayload, EngineWatcher};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

pub struct EngineManager {
    // We share adapters via Arc<Mutex> because multiple threads (http server) might access them
    pub prefill_engines: Vec<Arc<Mutex<EngineAdapter>>>,
    pub decode_engines: Vec<Arc<Mutex<EngineAdapter>>>,
    redis_key_prefix: String,
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
            redis_key_prefix: "".to_string(), // Empty prefix to match NanoCtrl default
        }
    }

    #[allow(dead_code)]
    pub async fn connect_all(&mut self, config: &EngineConfig) -> anyhow::Result<()> {
        match config {
            EngineConfig::Unified {
                host,
                port,
                nanoctrl_address,
                redis_url: _,
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
                redis_url: _,
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

    /// Get Redis URL from NanoCtrl API
    pub async fn get_redis_url_from_nanoctrl(
        &self,
        nanoctrl_address: &str,
    ) -> anyhow::Result<String> {
        let client = reqwest::Client::new();
        let url = format!("{}/get_redis_address", nanoctrl_address);

        let response = client
            .post(&url)
            .json(&serde_json::json!({}))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to get Redis URL from NanoCtrl: {}",
                response.status()
            ));
        }

        let result: serde_json::Value = response.json().await?;
        if let Some(redis_url) = result["redis_address"].as_str() {
            // Ensure it's a full URL (add redis:// if missing)
            let redis_url = if redis_url.starts_with("redis://") {
                redis_url.to_string()
            } else {
                format!("redis://{}", redis_url)
            };
            Ok(redis_url)
        } else {
            Err(anyhow::anyhow!(
                "Invalid response from NanoCtrl: missing redis_address"
            ))
        }
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

    /// Get current revision from Redis
    async fn get_current_revision(&self, redis_url: &str) -> anyhow::Result<i64> {
        let client = redis::Client::open(redis_url)?;
        let mut conn = client.get_multiplexed_async_connection().await?;
        let revision_key = format!("{}:nano_meta:engine_revision", self.redis_key_prefix);
        let revision: Option<i64> = redis::cmd("GET")
            .arg(&revision_key)
            .query_async(&mut conn)
            .await?;
        let revision = revision.unwrap_or(0);
        Ok(revision)
    }

    /// Load snapshot from Redis directly
    async fn load_snapshot_from_redis(&mut self, redis_url: &str) -> anyhow::Result<i64> {
        let client = redis::Client::open(redis_url)?;
        let mut conn = client.get_multiplexed_async_connection().await?;

        // Scan all engine:* keys with scope prefix
        let pattern = format!("{}:engine:*", self.redis_key_prefix);
        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(&pattern)
            .query_async(&mut conn)
            .await?;

        for key in keys {
            let engine_info_str: Option<String> = redis::cmd("HGET")
                .arg(&key)
                .arg("info")
                .query_async(&mut conn)
                .await?;

            if let Some(info_str) = engine_info_str {
                if let Ok(engine_info) = serde_json::from_str::<serde_json::Value>(&info_str) {
                    if let Err(e) = self.add_engine_from_info(engine_info).await {
                        warn!("Failed to add engine from snapshot: {}", e);
                    }
                }
            }
        }

        // Get current revision
        let revision = self.get_current_revision(redis_url).await?;
        Ok(revision)
    }

    /// Add engine from engine info JSON
    async fn add_engine_from_info(&mut self, engine_info: serde_json::Value) -> anyhow::Result<()> {
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
                return Err(anyhow::anyhow!("Invalid port"));
            }

            let connect_host = if host_str == "0.0.0.0" {
                "127.0.0.1"
            } else {
                host_str
            };
            let addr = format!("{}:{}", connect_host, port_num);
            let engine_id = engine_info["id"].as_str().unwrap_or("unknown");

            let mut adapter = EngineAdapter::new(engine_id.to_string());

            adapter.connect(&addr).await?;

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

            match role {
                "prefill" => {
                    self.prefill_engines.push(adapter);
                    info!("Added prefill engine {} from snapshot", engine_id);
                }
                "decode" => {
                    self.decode_engines.push(adapter);
                    info!("Added decode engine {} from snapshot", engine_id);
                }
                _ => {
                    // hybrid or unified
                    self.prefill_engines.push(adapter.clone());
                    self.decode_engines.push(adapter);
                    info!("Added {} engine {} from snapshot", role, engine_id);
                }
            }
        }

        Ok(())
    }

    /// Start dynamic service discovery
    /// Note: This method consumes self and returns Arc<Mutex<Self>> for concurrent access
    pub async fn start_dynamic_discovery(
        mut self,
        redis_url: String,
        nanoctrl_address: Option<String>,
    ) -> anyhow::Result<Arc<Mutex<Self>>> {
        // Redis key prefix: Use empty prefix to match NanoCtrl default behavior
        // The automatic prefix generation was removed to fix key mismatch issues
        // See: TROUBLESHOOTING_GUIDE.md Issue #3 - Mangled Redis Prefix
        info!(
            "Using Redis key prefix: '{}' (matches NanoCtrl default)",
            self.redis_key_prefix
        );
        // Step 1: Load snapshot (full sync)
        // Strategy: Load from both Redis and NanoCtrl API, merge results
        // This ensures we get all engines even if some have expired TTL in Redis

        let mut initial_revision = 0i64;
        let mut redis_engines = 0;

        // First, try to load from Redis
        match self.load_snapshot_from_redis(&redis_url).await {
            Ok(rev) => {
                initial_revision = rev;
                redis_engines = self.prefill_engines.len() + self.decode_engines.len();
                info!(
                    "Loaded {} engines from Redis snapshot (revision={})",
                    redis_engines, rev
                );
            }
            Err(e) => {
                warn!("Failed to load snapshot from Redis: {}", e);
            }
        }

        // Then, if NanoCtrl address is provided, also query from API and merge
        if let Some(addr) = &nanoctrl_address {
            match self.list_engines_from_nanoctrl(addr).await {
                Ok(api_engines) => {
                    info!(
                        "NanoCtrl API reports {} engines, Redis snapshot has {} engines",
                        api_engines.len(),
                        redis_engines
                    );

                    // Collect existing engine IDs from snapshot (use async lock)
                    let mut existing_ids = std::collections::HashSet::new();
                    for adapter in &self.prefill_engines {
                        let guard = adapter.lock().await;
                        if let Some(uuid) = &guard.uuid {
                            existing_ids.insert(uuid.clone());
                        }
                    }
                    for adapter in &self.decode_engines {
                        let guard = adapter.lock().await;
                        if let Some(uuid) = &guard.uuid {
                            existing_ids.insert(uuid.clone());
                        }
                    }

                    // Add any engines from API that might be missing from Redis snapshot
                    let mut added_count = 0;
                    for engine_info in api_engines {
                        let engine_id = engine_info["id"].as_str().unwrap_or("unknown").to_string();
                        if !existing_ids.contains(&engine_id) {
                            if let Err(e) = self.add_engine_from_info(engine_info.clone()).await {
                                warn!(
                                    "Failed to add engine {} from NanoCtrl API: {}",
                                    engine_id, e
                                );
                            } else {
                                added_count += 1;
                                info!("Added missing engine {} from NanoCtrl API", engine_id);
                            }
                        }
                    }

                    if added_count > 0 {
                        info!("Added {} missing engines from NanoCtrl API", added_count);
                    }

                    // Update revision from Redis if API query succeeded
                    if let Ok(rev) = self.get_current_revision(&redis_url).await {
                        initial_revision = rev;
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to query engines from NanoCtrl API (using Redis snapshot only): {}",
                        e
                    );
                }
            }
        }

        info!(
            "Snapshot loaded: {} prefill engines, {} decode engines, revision={}",
            self.prefill_engines.len(),
            self.decode_engines.len(),
            initial_revision
        );

        // Step 2: Start watcher with scoped prefix
        let redis_prefix = self.redis_key_prefix.clone();
        let (watcher, mut event_rx): (EngineWatcher, mpsc::UnboundedReceiver<EngineEvent>) =
            EngineWatcher::new(redis_url.clone(), initial_revision, redis_prefix);
        let watcher_handle = tokio::spawn(async move {
            if let Err(e) = watcher.start().await {
                error!("Watcher task error: {}", e);
            }
        });

        // Step 3: Process events in a separate task
        let manager_arc = Arc::new(Mutex::new(self));
        let nanoctrl_addr_clone = nanoctrl_address.clone();
        let redis_url_clone = redis_url.clone();

        let manager_arc_clone = manager_arc.clone();
        tokio::spawn(async move {
            while let Some(event) = event_rx.recv().await {
                let mut manager = manager_arc_clone.lock().await;
                match event {
                    EngineEvent::Add {
                        engine_id,
                        payload,
                        revision,
                    } => {
                        info!(
                            "Processing ADD event: engine_id={}, revision={}",
                            engine_id, revision
                        );
                        if let Err(e) = manager.handle_add_engine(payload).await {
                            error!("Failed to add engine {}: {}", engine_id, e);
                        } else {
                            info!("Successfully added engine: {}", engine_id);
                        }
                    }
                    EngineEvent::Remove {
                        engine_id,
                        revision,
                    } => {
                        info!(
                            "Processing REMOVE event: engine_id={}, revision={}",
                            engine_id, revision
                        );
                        if let Err(e) = manager.handle_remove_engine(&engine_id).await {
                            error!("Failed to remove engine {}: {}", engine_id, e);
                        } else {
                            info!("Successfully removed engine: {}", engine_id);
                        }
                    }
                    EngineEvent::Update {
                        engine_id,
                        payload,
                        revision,
                    } => {
                        info!(
                            "Processing UPDATE event: engine_id={}, revision={}",
                            engine_id, revision
                        );
                        if let Err(e) = manager.handle_update_engine(&engine_id, payload).await {
                            error!("Failed to update engine {}: {}", engine_id, e);
                        } else {
                            info!("Successfully updated engine: {}", engine_id);
                        }
                    }
                    EngineEvent::GapDetected {
                        expected_revision,
                        actual_revision,
                    } => {
                        warn!(
                            "Gap detected: expected={}, actual={}, triggering full sync",
                            expected_revision, actual_revision
                        );
                        if let Err(e) = manager
                            .handle_gap_detected(nanoctrl_addr_clone.as_deref(), &redis_url_clone)
                            .await
                        {
                            error!("Failed to handle gap: {}", e);
                        }
                    }
                    EngineEvent::ReconnectRequired => {
                        warn!("Reconnect required, triggering full sync");
                        if let Err(e) = manager
                            .handle_gap_detected(nanoctrl_addr_clone.as_deref(), &redis_url_clone)
                            .await
                        {
                            error!("Failed to handle reconnect: {}", e);
                        }
                    }
                }
            }
            // If event stream ends, watcher task should have ended too
            drop(watcher_handle);
        });

        Ok(manager_arc)
    }

    async fn handle_add_engine(&mut self, payload: EnginePayload) -> anyhow::Result<()> {
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAY: Duration = std::time::Duration::from_secs(2);

        // IMPORTANT: Remove existing engine with same ID first to avoid duplicates
        // This handles the case where engine restarts and we get a new ADD event
        info!(
            "Adding engine: {} (role: {}), checking for existing instance...",
            payload.id, payload.role
        );
        if let Err(e) = self.handle_remove_engine(&payload.id).await {
            // It's ok if engine doesn't exist, just log
            info!("No existing engine {} to remove: {}", payload.id, e);
        }

        // Handle host "0.0.0.0" -> "127.0.0.1" conversion
        let zmq_addr = if payload.zmq_address.starts_with("tcp://0.0.0.0:") {
            payload
                .zmq_address
                .replace("tcp://0.0.0.0:", "tcp://127.0.0.1:")
        } else {
            payload.zmq_address.clone()
        };

        let addr = zmq_addr.strip_prefix("tcp://").unwrap_or(&zmq_addr);

        for attempt in 1..=MAX_RETRIES {
            let mut adapter = EngineAdapter::new(payload.id.clone());

            match adapter.connect(addr).await {
                Ok(_) => {
                    adapter.uuid = Some(payload.id.clone());
                    adapter.world_size = payload.world_size as i32;
                    adapter.num_blocks = payload.num_blocks as i32;

                    let adapter = Arc::new(Mutex::new(adapter));

                    match payload.role.as_str() {
                        "prefill" => {
                            self.prefill_engines.push(adapter);
                            info!(
                                "Added prefill engine: {} (total prefill: {})",
                                payload.id,
                                self.prefill_engines.len()
                            );
                        }
                        "decode" => {
                            self.decode_engines.push(adapter);
                            info!(
                                "Added decode engine: {} (total decode: {})",
                                payload.id,
                                self.decode_engines.len()
                            );
                        }
                        _ => {
                            // hybrid or unified
                            self.prefill_engines.push(adapter.clone());
                            self.decode_engines.push(adapter);
                            info!(
                                "Added {} engine: {} (total prefill: {}, decode: {})",
                                payload.role,
                                payload.id,
                                self.prefill_engines.len(),
                                self.decode_engines.len()
                            );
                        }
                    }
                    return Ok(());
                }
                Err(e) => {
                    if attempt == MAX_RETRIES {
                        error!(
                            "Failed to connect to engine {} after {} attempts: {}",
                            payload.id, MAX_RETRIES, e
                        );
                        return Err(anyhow::anyhow!("Connection failed: {}", e));
                    }
                    warn!(
                        "Failed to connect to engine {} (attempt {}/{}): {}, retrying...",
                        payload.id, attempt, MAX_RETRIES, e
                    );
                    tokio::time::sleep(RETRY_DELAY * attempt).await;
                }
            }
        }

        unreachable!()
    }

    async fn handle_remove_engine(&mut self, engine_id: &str) -> anyhow::Result<()> {
        info!("Removing engine: {}", engine_id);

        // Remove from prefill_engines and cleanup pending requests
        let prefill_before = self.prefill_engines.len();
        let mut removed_prefill = false;
        self.prefill_engines.retain(|adapter| {
            let mut adapter_guard = futures::executor::block_on(adapter.lock());
            let should_keep = adapter_guard.uuid.as_deref() != Some(engine_id);
            if !should_keep {
                removed_prefill = true;
                info!(
                    "Removed prefill engine: {} (cleaning up and shutting down)",
                    engine_id
                );

                // Step 1: Close request channel to stop I/O thread
                drop(adapter_guard.request_tx.take());

                // Step 2: Cleanup pending requests
                let pending = adapter_guard.pending_requests.clone();
                futures::executor::block_on(async {
                    let mut map = pending.lock().await;
                    for (seq_id, state) in map.drain() {
                        let _ =
                            state
                                .sender
                                .send(crate::engine_adapter::StreamEvent::Error(format!(
                                    "Engine {} disconnected",
                                    engine_id
                                )));
                        info!(
                            "Cleaned up pending request seq_id={} for removed engine {}",
                            seq_id, engine_id
                        );
                    }
                });

                // Step 3: Send shutdown signal to stop reader loop
                if let Some(shutdown_tx) = &adapter_guard.shutdown_tx {
                    let _ = shutdown_tx.send(());
                }
            }
            should_keep
        });
        let prefill_after = self.prefill_engines.len();

        // Remove from decode_engines and cleanup pending requests
        let decode_before = self.decode_engines.len();
        let mut removed_decode = false;
        self.decode_engines.retain(|adapter| {
            let mut adapter_guard = futures::executor::block_on(adapter.lock());
            let should_keep = adapter_guard.uuid.as_deref() != Some(engine_id);
            if !should_keep {
                removed_decode = true;
                info!(
                    "Removed decode engine: {} (cleaning up and shutting down)",
                    engine_id
                );

                // Step 1: Close request channel to stop I/O thread
                drop(adapter_guard.request_tx.take());

                // Step 2: Cleanup pending requests
                let pending = adapter_guard.pending_requests.clone();
                futures::executor::block_on(async {
                    let mut map = pending.lock().await;
                    for (seq_id, state) in map.drain() {
                        let _ =
                            state
                                .sender
                                .send(crate::engine_adapter::StreamEvent::Error(format!(
                                    "Engine {} disconnected",
                                    engine_id
                                )));
                        info!(
                            "Cleaned up pending request seq_id={} for removed engine {}",
                            seq_id, engine_id
                        );
                    }
                });

                // Step 3: Send shutdown signal to stop reader loop
                if let Some(shutdown_tx) = &adapter_guard.shutdown_tx {
                    let _ = shutdown_tx.send(());
                }
            }
            should_keep
        });
        let decode_after = self.decode_engines.len();

        if removed_prefill || removed_decode {
            info!(
                "Engine removal complete: engine_id={}, prefill: {}->{}, decode: {}->{}",
                engine_id, prefill_before, prefill_after, decode_before, decode_after
            );
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Engine {} not found in prefill or decode lists",
                engine_id
            ))
        }
    }

    async fn handle_update_engine(
        &mut self,
        engine_id: &str,
        payload: EnginePayload,
    ) -> anyhow::Result<()> {
        // Remove old connection
        self.handle_remove_engine(engine_id).await?;
        // Add new connection
        self.handle_add_engine(payload).await?;
        info!("Updated engine: {}", engine_id);
        Ok(())
    }

    async fn handle_gap_detected(
        &mut self,
        nanoctrl_address: Option<&str>,
        redis_url: &str,
    ) -> anyhow::Result<()> {
        warn!("Gap detected, performing full sync");
        // Clear existing connections
        self.prefill_engines.clear();
        self.decode_engines.clear();

        // Reload snapshot
        if let Some(addr) = nanoctrl_address {
            let engines = self.list_engines_from_nanoctrl(addr).await?;
            for engine_info in engines {
                if let Err(e) = self.add_engine_from_info(engine_info).await {
                    warn!("Failed to add engine from full sync: {}", e);
                }
            }
        } else {
            self.load_snapshot_from_redis(redis_url).await?;
        }

        info!(
            "Full sync completed: {} prefill engines, {} decode engines",
            self.prefill_engines.len(),
            self.decode_engines.len()
        );
        Ok(())
    }
}
