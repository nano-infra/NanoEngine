use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tracing::{debug, info, warn};

#[derive(Default)]
pub struct ModelPool {
    pub http_hybrid_engines: Vec<String>,
    pub http_prefill_engines: Vec<String>,
    pub http_decode_engines: Vec<String>,
}

impl ModelPool {
    pub fn has_http_route(&self) -> bool {
        !self.http_hybrid_engines.is_empty()
            || (!self.http_prefill_engines.is_empty() && !self.http_decode_engines.is_empty())
    }

    pub fn get_next_http_hybrid(&self) -> Option<&str> {
        self.http_hybrid_engines.first().map(String::as_str)
    }

    pub fn get_next_http_prefill(&self) -> Option<&str> {
        self.http_prefill_engines.first().map(String::as_str)
    }

    pub fn get_next_http_decode(&self) -> Option<&str> {
        self.http_decode_engines.first().map(String::as_str)
    }
}

struct HttpEngineInfo {
    engine_id: String,
    role: String,
    url: String,
}

pub struct EngineManager {
    pub model_pools: HashMap<String, ModelPool>,
    engine_model_map: HashMap<String, String>,
    http_engine_urls: HashMap<String, String>,
    ctrl_scope: String,
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
            http_engine_urls: HashMap::new(),
            ctrl_scope: String::new(),
        }
    }

    pub fn with_scope(scope: Option<String>) -> Self {
        Self {
            ctrl_scope: scope.unwrap_or_default(),
            ..Self::new()
        }
    }

    pub fn available_model_keys(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self.model_pools.keys().map(String::as_str).collect();
        keys.sort();
        keys
    }

    pub fn routable_model_keys(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self
            .model_pools
            .iter()
            .filter_map(|(key, pool)| pool.has_http_route().then_some(key.as_str()))
            .collect();
        keys.sort();
        keys
    }

    pub fn total_engine_counts(&self) -> (usize, usize, usize) {
        let (hybrid, prefill, decode) = self.total_http_role_counts();
        (prefill + hybrid, decode + hybrid, 0)
    }

    pub fn total_http_role_counts(&self) -> (usize, usize, usize) {
        let mut hybrid = HashSet::new();
        let mut prefill = HashSet::new();
        let mut decode = HashSet::new();
        for pool in self.model_pools.values() {
            hybrid.extend(pool.http_hybrid_engines.iter().cloned());
            prefill.extend(pool.http_prefill_engines.iter().cloned());
            decode.extend(pool.http_decode_engines.iter().cloned());
        }
        (hybrid.len(), prefill.len(), decode.len())
    }

    fn normalize_model_key(path: &str) -> String {
        path.trim().trim_end_matches('/').to_string()
    }

    fn entity_id(info: &serde_json::Value) -> Option<String> {
        info.get("entity_id")
            .or_else(|| info.get("id"))
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
    }

    fn parse_http_engine_info(info: &serde_json::Value) -> Option<HttpEngineInfo> {
        if info.get("kind").and_then(|v| v.as_str()) != Some("dlengine") {
            return None;
        }
        let endpoint = info.get("endpoint")?.as_object()?;
        let protocol = endpoint
            .get("protocol")
            .and_then(|v| v.as_str())
            .unwrap_or("http");
        if protocol != "http" && protocol != "https" {
            return None;
        }
        let host = endpoint.get("host")?.as_str()?;
        let port = endpoint.get("port")?.as_u64()?;
        let connect_host = if host == "0.0.0.0" { "127.0.0.1" } else { host };
        let role = info
            .get("metadata")
            .and_then(|v| v.as_object())
            .and_then(|m| m.get("role"))
            .and_then(|v| v.as_str())
            .unwrap_or("hybrid")
            .to_ascii_lowercase();
        let engine_id = Self::entity_id(info).unwrap_or_else(|| "unknown".to_string());
        Some(HttpEngineInfo {
            engine_id,
            role,
            url: format!("{}://{}:{}", protocol, connect_host, port),
        })
    }

    fn model_aliases_from_metadata(
        metadata: &serde_json::Map<String, serde_json::Value>,
    ) -> Vec<String> {
        let mut aliases = Vec::new();
        let mut add = |value: Option<&str>| {
            let Some(value) = value else {
                return;
            };
            let key = Self::normalize_model_key(value);
            if !key.is_empty() && !aliases.iter().any(|x| x == &key) {
                aliases.push(key);
            }
        };

        add(metadata.get("served_model_name").and_then(|v| v.as_str()));
        if let Some(model_path) = metadata.get("model_path").and_then(|v| v.as_str()) {
            add(Some(model_path));
            add(model_path.trim_end_matches('/').rsplit('/').next());
        }
        aliases
    }

    fn insert_http_engine(&mut self, info: &HttpEngineInfo, model_key: &str) {
        let pool = self.model_pools.entry(model_key.to_string()).or_default();
        let engines = match info.role.as_str() {
            "prefill" => &mut pool.http_prefill_engines,
            "decode" => &mut pool.http_decode_engines,
            _ => &mut pool.http_hybrid_engines,
        };
        if !engines.contains(&info.url) {
            engines.push(info.url.clone());
        }
        info!(
            "Added HTTP {} engine: {} -> {} for model {}",
            info.role, info.engine_id, info.url, model_key
        );
        self.engine_model_map
            .insert(info.engine_id.clone(), model_key.to_string());
        self.http_engine_urls
            .insert(info.engine_id.clone(), info.url.clone());
    }

    pub async fn list_engines_from_ctrl(
        &self,
        ctrl_address: &str,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let client = reqwest::Client::new();
        let url = format!("{}/list_entities", ctrl_address.trim_end_matches('/'));
        let mut body = serde_json::json!({ "entity_type": "service" });
        if !self.ctrl_scope.is_empty() {
            body["scope"] = serde_json::Value::String(self.ctrl_scope.clone());
        }

        let response = client.post(url).json(&body).send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to list NanoCtrl entities: {}",
                response.status()
            ));
        }

        let result: serde_json::Value = response.json().await?;
        if result.get("status").and_then(|s| s.as_str()) != Some("ok") {
            return Err(anyhow::anyhow!(
                "NanoCtrl list_entities returned non-ok status"
            ));
        }
        let Some(entities) = result.get("entities").and_then(|e| e.as_array()) else {
            return Ok(Vec::new());
        };
        let engines = entities
            .iter()
            .filter(|entity| entity.get("kind").and_then(|v| v.as_str()) == Some("dlengine"))
            .cloned()
            .collect::<Vec<_>>();
        debug!(
            "Found {} DLEngine HTTP entities from NanoCtrl",
            engines.len()
        );
        Ok(engines)
    }

    async fn add_engine_from_info(&mut self, engine_info: serde_json::Value) -> anyhow::Result<()> {
        let http_info = Self::parse_http_engine_info(&engine_info)
            .ok_or_else(|| anyhow::anyhow!("not a DLEngine HTTP entity"))?;
        let metadata = engine_info
            .get("metadata")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow::anyhow!("HTTP DLEngine entity missing metadata"))?;
        let model_path = metadata
            .get("model_path")
            .and_then(|v| v.as_str())
            .filter(|p| !p.is_empty())
            .ok_or_else(|| anyhow::anyhow!("HTTP DLEngine entity missing model_path"))?;

        let aliases = Self::model_aliases_from_metadata(metadata);
        if aliases.is_empty() {
            self.insert_http_engine(&http_info, &Self::normalize_model_key(model_path));
        } else {
            for alias in aliases {
                self.insert_http_engine(&http_info, &alias);
            }
        }
        self.engine_model_map.insert(
            http_info.engine_id.clone(),
            Self::normalize_model_key(model_path),
        );
        Ok(())
    }

    async fn load_initial_engines(&mut self, ctrl_address: Option<&str>) -> anyhow::Result<()> {
        let Some(addr) = ctrl_address else {
            return Err(anyhow::anyhow!("ctrl_address is required"));
        };
        let engines = self.list_engines_from_ctrl(addr).await?;
        for engine_info in engines {
            let engine_id = Self::entity_id(&engine_info).unwrap_or_else(|| "unknown".to_string());
            if let Err(e) = self.add_engine_from_info(engine_info).await {
                warn!(
                    "Failed to add engine {} from NanoCtrl API: {}",
                    engine_id, e
                );
            } else {
                info!("Added DLEngine node {} from NanoCtrl API", engine_id);
            }
        }
        Ok(())
    }

    pub async fn start_dynamic_discovery(
        mut self,
        ctrl_address: Option<String>,
    ) -> anyhow::Result<Arc<Mutex<Self>>> {
        debug!(
            "Using NanoCtrl scope: '{}'",
            if self.ctrl_scope.is_empty() {
                "(default)"
            } else {
                &self.ctrl_scope
            }
        );
        self.load_initial_engines(ctrl_address.as_deref()).await?;
        let manager_arc = Arc::new(Mutex::new(self));

        if ctrl_address.is_some() {
            let manager_arc_resync = manager_arc.clone();
            let ctrl_addr_resync = ctrl_address.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(5));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                interval.tick().await;
                loop {
                    interval.tick().await;
                    let mut manager = manager_arc_resync.lock().await;
                    if let Err(e) = manager
                        .handle_periodic_sync(ctrl_addr_resync.as_deref())
                        .await
                    {
                        warn!("Periodic sync failed: {}", e);
                    }
                }
            });
        }

        Ok(manager_arc)
    }

    async fn handle_remove_engine(&mut self, engine_id: &str) -> anyhow::Result<()> {
        let Some(url) = self.http_engine_urls.remove(engine_id) else {
            return Err(anyhow::anyhow!("Engine {} not found", engine_id));
        };
        for pool in self.model_pools.values_mut() {
            pool.http_hybrid_engines.retain(|u| u != &url);
            pool.http_prefill_engines.retain(|u| u != &url);
            pool.http_decode_engines.retain(|u| u != &url);
        }
        self.engine_model_map.remove(engine_id);
        info!("Removed HTTP DLEngine node: {} -> {}", engine_id, url);
        Ok(())
    }

    async fn handle_periodic_sync(&mut self, ctrl_address: Option<&str>) -> anyhow::Result<()> {
        let Some(addr) = ctrl_address else {
            return Ok(());
        };
        let live_engines = self.list_engines_from_ctrl(addr).await?;
        let live_ids: HashSet<String> = live_engines.iter().filter_map(Self::entity_id).collect();
        let local_ids: HashSet<String> = self.engine_model_map.keys().cloned().collect();

        let mut added = 0usize;
        for engine_info in live_engines {
            let Some(engine_id) = Self::entity_id(&engine_info) else {
                continue;
            };
            if !local_ids.contains(&engine_id) {
                if let Err(e) = self.add_engine_from_info(engine_info).await {
                    warn!(
                        "Failed to add live DLEngine {} from NanoCtrl: {}",
                        engine_id, e
                    );
                } else {
                    added += 1;
                    info!("Discovered live DLEngine from NanoCtrl: {}", engine_id);
                }
            }
        }

        let stale: Vec<String> = local_ids.difference(&live_ids).cloned().collect();
        for stale_id in &stale {
            warn!("Engine heartbeat lost: {} no longer in NanoCtrl", stale_id);
            if let Err(e) = self.handle_remove_engine(stale_id).await {
                warn!("Failed to remove stale engine {}: {}", stale_id, e);
            }
        }

        if added > 0 || !stale.is_empty() {
            let (tp, td, enc) = self.total_engine_counts();
            let (hybrid, http_prefill, http_decode) = self.total_http_role_counts();
            let message = format!(
                "NanoCtrl sync changed nodes: added={}, removed={} - HTTP roles: {} hybrid, {} prefill, {} decode; route counts: {} prefill, {} decode, {} encoder; models={:?}",
                added, stale.len(), hybrid, http_prefill, http_decode, tp, td, enc, self.available_model_keys()
            );
            if stale.is_empty() {
                info!("{}", message);
            } else {
                warn!("{}", message);
            }
        }

        Ok(())
    }
}
