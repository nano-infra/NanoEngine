use crate::encoder_adapter::EncoderAdapter;
use crate::engine_adapter::EngineAdapter;
use crate::engine_watcher::{EngineEvent, EnginePayload, EngineWatcher};
use crate::tokenizer::TokenizerService;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Per-model engine pool: owns engine adapters and a lazily-loaded tokenizer slot.
pub struct ModelPool {
    pub prefill_engines: Vec<Arc<Mutex<EngineAdapter>>>,
    pub decode_engines: Vec<Arc<Mutex<EngineAdapter>>>,
    pub encoder_engines: Vec<Arc<Mutex<EncoderAdapter>>>,
    /// Lazily loaded tokenizer; None until an engine with model_path arrives.
    pub tokenizer_slot: Arc<RwLock<Option<Arc<TokenizerService>>>>,
}

impl ModelPool {
    fn new() -> Self {
        Self {
            prefill_engines: Vec::new(),
            decode_engines: Vec::new(),
            encoder_engines: Vec::new(),
            tokenizer_slot: Arc::new(RwLock::new(None)),
        }
    }

    pub fn get_next_prefill(&self) -> Option<Arc<Mutex<EngineAdapter>>> {
        self.prefill_engines.first().cloned()
    }

    pub fn get_next_decode(&self) -> Option<Arc<Mutex<EngineAdapter>>> {
        self.decode_engines.first().cloned()
    }

    pub fn get_next_encoder(&self) -> Option<Arc<Mutex<EncoderAdapter>>> {
        self.encoder_engines.first().cloned()
    }
}

/// Parsed fields from an engine info JSON value.
struct ParsedEngineInfo {
    engine_id: String,
    role: String,
    connect_addr: String,
    world_size: i32,
    num_blocks: i32,
}

pub struct EngineManager {
    /// Per-model engine pools, keyed by normalized model_dir (no trailing slash).
    pub model_pools: HashMap<String, ModelPool>,
    /// Reverse map: engine_id → model_key. Used for O(1) removal.
    engine_model_map: HashMap<String, String>,
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
            model_pools: HashMap::new(),
            engine_model_map: HashMap::new(),
            redis_key_prefix: "".to_string(),
        }
    }

    pub fn with_scope(scope: Option<String>) -> Self {
        Self {
            model_pools: HashMap::new(),
            engine_model_map: HashMap::new(),
            redis_key_prefix: scope.unwrap_or_default(),
        }
    }

    // ─── Helpers ──────────────────────────────────────────────────────

    fn normalize_model_key(path: &str) -> String {
        path.trim_end_matches('/').to_string()
    }

    pub fn total_engine_counts(&self) -> (usize, usize, usize) {
        self.model_pools.values().fold((0, 0, 0), |acc, p| {
            (
                acc.0 + p.prefill_engines.len(),
                acc.1 + p.decode_engines.len(),
                acc.2 + p.encoder_engines.len(),
            )
        })
    }

    /// Returns sorted list of registered model keys for 404 messages / diagnostics.
    pub fn available_model_keys(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self.model_pools.keys().map(|s| s.as_str()).collect();
        keys.sort();
        keys
    }

    /// Parse engine info JSON into structured fields.
    /// Handles port as u64 or string, and rewrites 0.0.0.0 → 127.0.0.1.
    fn parse_engine_info(info: &serde_json::Value) -> anyhow::Result<ParsedEngineInfo> {
        let port_num = info["port"]
            .as_u64()
            .or_else(|| info["port"].as_str().and_then(|s| s.parse::<u64>().ok()))
            .unwrap_or(0);

        let host_str = info["host"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing host"))?;
        let role = info["role"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing role"))?;

        if port_num == 0 {
            return Err(anyhow::anyhow!("Invalid port"));
        }

        let connect_host = if host_str == "0.0.0.0" {
            "127.0.0.1"
        } else {
            host_str
        };

        Ok(ParsedEngineInfo {
            engine_id: info["id"].as_str().unwrap_or("unknown").to_string(),
            role: role.to_string(),
            connect_addr: format!("{}:{}", connect_host, port_num),
            world_size: info["world_size"].as_u64().unwrap_or(0) as i32,
            num_blocks: info["num_blocks"].as_u64().unwrap_or(0) as i32,
        })
    }

    /// Insert an adapter into the correct engine pool based on role.
    fn insert_engine_by_role(
        &mut self,
        adapter: Arc<Mutex<EngineAdapter>>,
        role: &str,
        engine_id: &str,
        model_key: &str,
    ) {
        let pool = self
            .model_pools
            .entry(model_key.to_string())
            .or_insert_with(ModelPool::new);
        match role {
            "prefill" => {
                pool.prefill_engines.push(adapter);
                info!(
                    "Added prefill engine: {} for model {} (total: {})",
                    engine_id,
                    model_key,
                    pool.prefill_engines.len()
                );
            }
            "decode" => {
                pool.decode_engines.push(adapter);
                info!(
                    "Added decode engine: {} for model {} (total: {})",
                    engine_id,
                    model_key,
                    pool.decode_engines.len()
                );
            }
            "encoder" => {
                warn!(
                    "insert_engine_by_role called for encoder role; use insert_encoder instead. engine_id={}",
                    engine_id
                );
            }
            _ => {
                // hybrid or unified — add to both pools
                pool.prefill_engines.push(adapter.clone());
                pool.decode_engines.push(adapter);
                info!(
                    "Added {} engine: {} for model {} (prefill: {}, decode: {})",
                    role,
                    engine_id,
                    model_key,
                    pool.prefill_engines.len(),
                    pool.decode_engines.len()
                );
            }
        }
        self.engine_model_map
            .insert(engine_id.to_string(), model_key.to_string());
    }

    /// Insert an encoder adapter into the encoder pool.
    fn insert_encoder(
        &mut self,
        adapter: Arc<Mutex<EncoderAdapter>>,
        engine_id: &str,
        model_key: &str,
    ) {
        let pool = self
            .model_pools
            .entry(model_key.to_string())
            .or_insert_with(ModelPool::new);
        pool.encoder_engines.push(adapter);
        info!(
            "Added encoder engine: {} for model {} (total: {})",
            engine_id,
            model_key,
            pool.encoder_engines.len()
        );
        self.engine_model_map
            .insert(engine_id.to_string(), model_key.to_string());
    }

    // ─── NanoCtrl API ────────────────────────────────────────────────

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

        let mut body = serde_json::json!({});
        if !self.redis_key_prefix.is_empty() {
            body["scope"] = serde_json::Value::String(self.redis_key_prefix.clone());
        }

        let response = client.post(&url).json(&body).send().await?;

        if response.status().is_success() {
            let result: serde_json::Value = response.json().await?;
            if let Some(status) = result.get("status").and_then(|s| s.as_str()) {
                if status == "ok" {
                    if let Some(engines) = result.get("engines").and_then(|e| e.as_array()) {
                        let engines: Vec<serde_json::Value> = engines.clone();
                        debug!("Found {} engines from NanoCtrl", engines.len());
                        return Ok(engines);
                    }
                }
            }
        }

        Err(anyhow::anyhow!("Failed to list engines from NanoCtrl"))
    }

    // ─── Redis snapshot ──────────────────────────────────────────────

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
                if let Ok(mut engine_info) = serde_json::from_str::<serde_json::Value>(&info_str) {
                    // `info` JSON may predate the model_path field; read the
                    // dedicated `model_path` hash field as a direct fallback.
                    if engine_info["model_path"].is_null() {
                        let model_path: Option<String> = redis::cmd("HGET")
                            .arg(&key)
                            .arg("model_path")
                            .query_async(&mut conn)
                            .await
                            .unwrap_or(None);
                        if let Some(mp) = model_path {
                            if !mp.is_empty() {
                                engine_info["model_path"] = serde_json::Value::String(mp);
                            }
                        }
                    }

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

    // ─── Engine lifecycle ────────────────────────────────────────────

    /// Add engine from engine info JSON (used by snapshot and NanoCtrl API paths).
    async fn add_engine_from_info(&mut self, engine_info: serde_json::Value) -> anyhow::Result<()> {
        let parsed = Self::parse_engine_info(&engine_info)?;

        let model_path = match engine_info["model_path"].as_str() {
            Some(p) if !p.is_empty() => p,
            _ => {
                warn!(
                    "Engine {} has no model_path — skipping (unroutable)",
                    parsed.engine_id
                );
                return Ok(());
            }
        };
        let model_key = Self::normalize_model_key(model_path);

        if parsed.role == "encoder" {
            let mut adapter = EncoderAdapter::new(parsed.engine_id.clone());
            adapter.connect(&parsed.connect_addr).await?;
            adapter.uuid = Some(parsed.engine_id.clone());
            let adapter = Arc::new(Mutex::new(adapter));
            self.insert_encoder(adapter, &parsed.engine_id, &model_key);
        } else {
            let mut adapter = EngineAdapter::new(parsed.engine_id.clone());
            adapter.connect(&parsed.connect_addr).await?;
            adapter.uuid = Some(parsed.engine_id.clone());
            adapter.world_size = parsed.world_size;
            adapter.num_blocks = parsed.num_blocks;
            let adapter = Arc::new(Mutex::new(adapter));
            self.insert_engine_by_role(adapter, &parsed.role, &parsed.engine_id, &model_key);
        }

        self.maybe_spawn_tokenizer_load(Some(model_path));
        Ok(())
    }

    /// Load initial engine set from Redis snapshot + NanoCtrl API merge.
    /// Returns the initial revision number for the watcher.
    async fn load_initial_engines(
        &mut self,
        redis_url: &str,
        nanoctrl_address: Option<&str>,
    ) -> anyhow::Result<i64> {
        let mut initial_revision = 0i64;
        let mut redis_engines = 0;

        // First, try to load from Redis
        match self.load_snapshot_from_redis(redis_url).await {
            Ok(rev) => {
                initial_revision = rev;
                redis_engines = self.engine_model_map.len();
                debug!(
                    "Loaded {} engines from Redis snapshot (revision={})",
                    redis_engines, rev
                );
            }
            Err(e) => {
                warn!("Failed to load snapshot from Redis: {}", e);
            }
        }

        // Then, if NanoCtrl address is provided, also query from API and merge
        // (bootstrap: 1x list_engines per router start)
        if let Some(addr) = nanoctrl_address {
            match self.list_engines_from_nanoctrl(addr).await {
                Ok(api_engines) => {
                    debug!(
                        "NanoCtrl API reports {} engines, Redis snapshot has {} engines",
                        api_engines.len(),
                        redis_engines
                    );

                    // Collect existing engine IDs from reverse map
                    let existing_ids: std::collections::HashSet<String> =
                        self.engine_model_map.keys().cloned().collect();

                    // Build set of engine IDs reported by NanoCtrl API (source of truth)
                    let api_ids: std::collections::HashSet<String> = api_engines
                        .iter()
                        .filter_map(|e| e["id"].as_str().map(|s| s.to_string()))
                        .collect();

                    // Remove stale engines from snapshot that are NOT in API
                    // This handles the race where Redis still had a stale key when
                    // we took the snapshot, but NanoCtrl has since cleaned it up.
                    let stale_ids: Vec<String> = existing_ids
                        .iter()
                        .filter(|id| !api_ids.contains(*id))
                        .cloned()
                        .collect();
                    for stale_id in &stale_ids {
                        info!(
                            "Removing stale engine {} (in Redis snapshot but not in NanoCtrl API)",
                            stale_id
                        );
                        if let Err(e) = self.handle_remove_engine(stale_id).await {
                            warn!("Failed to remove stale engine {}: {}", stale_id, e);
                        }
                    }
                    if !stale_ids.is_empty() {
                        info!("Removed {} stale engines from snapshot", stale_ids.len());
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
                    if let Ok(rev) = self.get_current_revision(redis_url).await {
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

        let (p, d, enc) = self.total_engine_counts();
        debug!(
            "Initial engines loaded: {} prefill, {} decode, {} encoder across {} model(s), revision={}",
            p, d, enc, self.model_pools.len(), initial_revision
        );

        Ok(initial_revision)
    }

    // ─── Dynamic discovery ───────────────────────────────────────────

    /// Start dynamic service discovery
    /// Note: This method consumes self and returns Arc<Mutex<Self>> for concurrent access
    pub async fn start_dynamic_discovery(
        mut self,
        redis_url: String,
        nanoctrl_address: Option<String>,
    ) -> anyhow::Result<Arc<Mutex<Self>>> {
        debug!(
            "Using Redis key prefix: '{}' (matches NanoCtrl default)",
            self.redis_key_prefix
        );

        // Step 1: Load snapshot + merge API data
        let initial_revision = self
            .load_initial_engines(&redis_url, nanoctrl_address.as_deref())
            .await?;

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

        // Step 4: Periodic resync to evict TTL-expired/crashed engines.
        // Redis key expiry is silent (no Pub/Sub event), so the watcher alone can't detect it.
        // Every 90s we query NanoCtrl's live list and remove any engine no longer present.
        // (TTL = 60s, so a crashed engine's key is gone within 60s; 90s gives margin.)
        if nanoctrl_address.is_some() {
            let manager_arc_resync = manager_arc.clone();
            let nanoctrl_addr_resync = nanoctrl_address.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(90));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                interval.tick().await; // skip the immediate first tick
                loop {
                    interval.tick().await;
                    let mut manager = manager_arc_resync.lock().await;
                    if let Err(e) = manager
                        .handle_periodic_sync(nanoctrl_addr_resync.as_deref())
                        .await
                    {
                        warn!("Periodic sync failed: {}", e);
                    }
                }
            });
        }

        Ok(manager_arc)
    }

    // ─── Event handlers ──────────────────────────────────────────────

    /// Spawn a background task to lazily load the tokenizer from an engine's model path.
    /// Routes to the correct per-model pool. No-op if already loaded or model_path is None/empty.
    fn maybe_spawn_tokenizer_load(&mut self, model_path: Option<&str>) {
        let Some(path) = model_path else { return };
        if path.is_empty() {
            return;
        }
        let model_key = Self::normalize_model_key(path);
        let pool = self
            .model_pools
            .entry(model_key)
            .or_insert_with(ModelPool::new);
        TokenizerService::spawn_load(pool.tokenizer_slot.clone(), path.to_string());
    }

    async fn handle_add_engine(&mut self, payload: EnginePayload) -> anyhow::Result<()> {
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAY: Duration = std::time::Duration::from_secs(2);

        // Guard: skip engines without a model_path (unroutable; old protocol)
        let model_path = match payload.model_path.as_deref() {
            Some(p) if !p.is_empty() => p,
            _ => {
                warn!(
                    "Engine {} has no model_path — skipping (unroutable)",
                    payload.id
                );
                return Ok(());
            }
        };
        let model_key = Self::normalize_model_key(model_path);

        // IMPORTANT: Remove existing engine with same ID first to avoid duplicates
        // This handles the case where engine restarts and we get a new ADD event
        info!(
            "Adding engine: {} (role: {}), checking for existing instance...",
            payload.id, payload.role
        );
        if let Err(e) = self.handle_remove_engine(&payload.id).await {
            // It's ok if engine doesn't exist, just log
            info!("No existing engine {} to remove: {}", payload.id, e);
        } else {
            // Give ZMQ time to fully clean up the old socket before reconnecting
            // This prevents "Address already in use" and connection state issues
            info!(
                "Waiting 500ms for ZMQ socket cleanup before reconnecting to {}",
                payload.id
            );
            tokio::time::sleep(Duration::from_millis(500)).await;
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

        // Encoder role uses EncoderAdapter (simpler request-response ZMQ)
        if payload.role == "encoder" {
            for attempt in 1..=MAX_RETRIES {
                let mut adapter = EncoderAdapter::new(payload.id.clone());
                match adapter.connect(addr).await {
                    Ok(_) => {
                        adapter.uuid = Some(payload.id.clone());
                        let adapter = Arc::new(Mutex::new(adapter));
                        self.insert_encoder(adapter, &payload.id, &model_key);
                        self.maybe_spawn_tokenizer_load(Some(model_path));
                        return Ok(());
                    }
                    Err(e) => {
                        if attempt == MAX_RETRIES {
                            error!(
                                "Failed to connect to encoder engine {} after {} attempts: {}",
                                payload.id, MAX_RETRIES, e
                            );
                            return Err(anyhow::anyhow!("Connection failed: {}", e));
                        }
                        warn!(
                            "Failed to connect to encoder engine {} (attempt {}/{}): {}, retrying...",
                            payload.id, attempt, MAX_RETRIES, e
                        );
                        tokio::time::sleep(RETRY_DELAY * attempt).await;
                    }
                }
            }
            unreachable!()
        }

        for attempt in 1..=MAX_RETRIES {
            let mut adapter = EngineAdapter::new(payload.id.clone());

            match adapter.connect(addr).await {
                Ok(_) => {
                    adapter.uuid = Some(payload.id.clone());
                    adapter.world_size = payload.world_size as i32;
                    adapter.num_blocks = payload.num_blocks as i32;

                    let adapter = Arc::new(Mutex::new(adapter));
                    self.insert_engine_by_role(adapter, &payload.role, &payload.id, &model_key);
                    self.maybe_spawn_tokenizer_load(Some(model_path));
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

    /// Helper function to cleanup an adapter's resources
    fn cleanup_adapter(
        adapter_guard: &mut EngineAdapter,
        engine_id: &str,
    ) -> (
        Option<tokio::task::JoinHandle<()>>,
        Option<std::thread::JoinHandle<()>>,
    ) {
        // Step 1: Close recv channel to stop async reader
        drop(adapter_guard.recv_tx_keepalive.take());

        // Step 2: Close request channel to stop I/O thread
        drop(adapter_guard.request_tx.take());

        // Step 3: Cleanup pending requests
        let pending = adapter_guard.pending_requests.clone();
        futures::executor::block_on(async {
            let mut map = pending.lock().await;
            for (_seq_id, state) in map.drain() {
                let _ = state
                    .sender
                    .send(crate::engine_adapter::StreamEvent::Error(format!(
                        "Engine {} disconnected",
                        engine_id
                    )));
            }
        });

        // Step 4: Send shutdown signal to stop reader loop
        if let Some(shutdown_tx) = &adapter_guard.shutdown_tx {
            let _ = shutdown_tx.send(());
        }

        // Step 5: Extract handles for later cleanup
        (
            adapter_guard.reader_handle.take(),
            adapter_guard.io_thread_handle.take(),
        )
    }

    async fn handle_remove_engine(&mut self, engine_id: &str) -> anyhow::Result<()> {
        info!("Removing engine: {}", engine_id);

        // Look up which model pool owns this engine (O(1) via reverse map)
        let model_key = self
            .engine_model_map
            .remove(engine_id)
            .ok_or_else(|| anyhow::anyhow!("Engine {} not found", engine_id))?;

        let pool = self
            .model_pools
            .get_mut(&model_key)
            .ok_or_else(|| anyhow::anyhow!("Model pool '{}' missing", model_key))?;

        // Collect reader handles and I/O thread handles to await after retain
        let mut reader_handles = Vec::new();
        let mut io_thread_handles = Vec::new();

        // Remove from prefill_engines and cleanup pending requests
        let prefill_before = pool.prefill_engines.len();
        let mut removed_prefill = false;
        pool.prefill_engines.retain(|adapter| {
            let mut adapter_guard = futures::executor::block_on(adapter.lock());
            let should_keep = adapter_guard.uuid.as_deref() != Some(engine_id);
            if !should_keep {
                removed_prefill = true;
                info!("Removing prefill engine: {}", engine_id);
                let (reader_handle, io_handle) =
                    Self::cleanup_adapter(&mut adapter_guard, engine_id);
                if let Some(h) = reader_handle {
                    reader_handles.push(h);
                }
                if let Some(h) = io_handle {
                    io_thread_handles.push(h);
                }
            }
            should_keep
        });
        let prefill_after = pool.prefill_engines.len();

        // Remove from decode_engines and cleanup pending requests
        let decode_before = pool.decode_engines.len();
        let mut removed_decode = false;
        pool.decode_engines.retain(|adapter| {
            let mut adapter_guard = futures::executor::block_on(adapter.lock());
            let should_keep = adapter_guard.uuid.as_deref() != Some(engine_id);
            if !should_keep {
                removed_decode = true;
                info!("Removing decode engine: {}", engine_id);
                let (reader_handle, io_handle) =
                    Self::cleanup_adapter(&mut adapter_guard, engine_id);
                if let Some(h) = reader_handle {
                    reader_handles.push(h);
                }
                if let Some(h) = io_handle {
                    io_thread_handles.push(h);
                }
            }
            should_keep
        });
        let decode_after = pool.decode_engines.len();

        // Remove from encoder_engines
        let encoder_before = pool.encoder_engines.len();
        let mut removed_encoder = false;
        pool.encoder_engines
            .retain(|adapter: &Arc<Mutex<EncoderAdapter>>| {
                let adapter_guard = futures::executor::block_on(adapter.lock());
                let should_keep = adapter_guard.uuid.as_deref() != Some(engine_id);
                if !should_keep {
                    removed_encoder = true;
                    info!("Removing encoder engine: {}", engine_id);
                    // EncoderAdapter cleanup: drop channels to stop I/O thread
                    // (handled automatically when adapter is dropped after retain)
                }
                should_keep
            });
        let encoder_after = pool.encoder_engines.len();

        // Await all reader tasks
        for handle in reader_handles {
            if let Err(e) = tokio::time::timeout(Duration::from_secs(2), handle).await {
                warn!("Reader task timeout for engine {}: {:?}", engine_id, e);
            }
        }

        // Await all I/O threads
        for handle in io_thread_handles {
            let join_result = tokio::task::spawn_blocking(move || handle.join()).await;
            if let Err(e) = join_result {
                warn!("I/O thread join failed for engine {}: {:?}", engine_id, e);
            }
        }

        if removed_prefill || removed_decode || removed_encoder {
            info!(
                "Engine removal complete: engine_id={}, model={}, prefill: {}→{}, decode: {}→{}, encoder: {}→{}",
                engine_id, model_key, prefill_before, prefill_after, decode_before, decode_after,
                encoder_before, encoder_after
            );
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Engine {} not found in prefill, decode, or encoder lists for model '{}'",
                engine_id,
                model_key
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

    /// Periodic sync: diff local pool against NanoCtrl live list and remove stale engines.
    ///
    /// This is the only way to detect engines that died without calling /unregister_engine
    /// (crash, SIGKILL, etc.). Their Redis key expires via TTL, but key expiry does NOT
    /// publish a REMOVE event — so the Pub/Sub watcher never fires for them.
    async fn handle_periodic_sync(&mut self, nanoctrl_address: Option<&str>) -> anyhow::Result<()> {
        let Some(addr) = nanoctrl_address else {
            return Ok(());
        };

        let live_engines = self.list_engines_from_nanoctrl(addr).await?;
        let live_ids: std::collections::HashSet<String> = live_engines
            .iter()
            .filter_map(|e| e["id"].as_str().map(|s| s.to_string()))
            .collect();

        // Collect local engine IDs from reverse map (O(1), no async lock needed)
        let local_ids: std::collections::HashSet<String> =
            self.engine_model_map.keys().cloned().collect();

        // Remove engines present locally but absent from NanoCtrl (TTL-expired or crashed)
        let stale: Vec<String> = local_ids.difference(&live_ids).cloned().collect();
        for stale_id in &stale {
            error!(
                "Engine heartbeat lost: {} no longer in NanoCtrl (TTL expired or process crashed)",
                stale_id
            );
            if let Err(e) = self.handle_remove_engine(stale_id).await {
                warn!("Failed to remove stale engine {}: {}", stale_id, e);
            }
        }

        if !stale.is_empty() {
            let (tp, td, enc) = self.total_engine_counts();
            warn!(
                "Periodic sync evicted {} engine(s) — pool now: {} prefill, {} decode, {} encoder across {} model(s)",
                stale.len(), tp, td, enc, self.model_pools.len()
            );
            for (key, pool) in &self.model_pools {
                if pool.prefill_engines.is_empty() && pool.decode_engines.is_empty() {
                    error!(
                        "All engines gone for model '{}' — requests will return 503",
                        key
                    );
                }
            }
        }

        Ok(())
    }

    async fn handle_gap_detected(
        &mut self,
        nanoctrl_address: Option<&str>,
        redis_url: &str,
    ) -> anyhow::Result<()> {
        warn!("Gap detected, performing full sync");
        // Clear engine vectors per pool (preserve tokenizer slots)
        for pool in self.model_pools.values_mut() {
            pool.prefill_engines.clear();
            pool.decode_engines.clear();
            pool.encoder_engines.clear();
        }
        self.engine_model_map.clear();

        // Reload from NanoCtrl or Redis
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

        let (p, d, enc) = self.total_engine_counts();
        info!(
            "Full sync completed: {} prefill engines, {} decode engines, {} encoder engines across {} model(s)",
            p, d, enc, self.model_pools.len()
        );
        Ok(())
    }
}
