use crate::config::{EngineConfig, EtcdConfig};
use crate::engine_adapter::EngineAdapter;
// Fix import path for flatbuffers generated code
// Hierarchy: fbs.rs (mod nanodeploy -> mod connection) -> include!(...) -> mod nanodeploy -> mod fbs
use crate::fbs::nanodeploy::connection::nanodeploy::fbs::EngineInfo;
use etcd_client::{Client, EventType, GetOptions, WatchOptions};

use futures::stream::StreamExt; // For WatchStream iteration
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

pub struct EngineManager {
    // We share adapters via Arc<Mutex> because multiple threads (http server) might access them
    pub prefill_engines: Vec<Arc<Mutex<EngineAdapter>>>,
    pub decode_engines: Vec<Arc<Mutex<EngineAdapter>>>,
    // Track discovered nodes to avoid duplicates or reconnects
    // Map: NodeUUID -> Adapter
    pub known_nodes: HashMap<String, Arc<Mutex<EngineAdapter>>>,
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
            known_nodes: HashMap::new(),
        }
    }

    pub async fn connect_all(&mut self, config: &EngineConfig) -> anyhow::Result<()> {
        match config {
            EngineConfig::Unified { host, port } => {
                info!("Connecting to Unified Engine at {}:{}", host, port);
                let addr = format!("{}:{}", host, port);
                let mut adapter = EngineAdapter::new(format!("unified-{}", port));
                adapter.connect(&addr).await?;

                let adapter = Arc::new(Mutex::new(adapter));
                self.prefill_engines.push(adapter.clone());
                self.decode_engines.push(adapter);
            }
            EngineConfig::Disaggregated { prefill, decode } => {
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
        Ok(())
    }

    // Helper to get a round-robin engine (simplest scheduler)
    pub fn get_next_prefill(&self) -> Option<Arc<Mutex<EngineAdapter>>> {
        self.prefill_engines.first().cloned()
    }

    pub fn get_next_decode(&self) -> Option<Arc<Mutex<EngineAdapter>>> {
        self.decode_engines.first().cloned()
    }

    pub async fn initialize_p2p_mesh(&mut self, config: &EngineConfig) -> anyhow::Result<()> {
        let mut all_nodes = Vec::new();
        let mut decode_uuids = Vec::new();
        let mut prefill_uuids = Vec::new();

        if let EngineConfig::Disaggregated {
            prefill: _,
            decode: _,
        } = config
        {
            // 1. Gather all info using UUIDs
            for (i, engine) in self.prefill_engines.iter().enumerate() {
                let adapter = engine.lock().await;
                let uuid = adapter
                    .uuid
                    .clone()
                    .unwrap_or_else(|| format!("prefill-{}", i));
                prefill_uuids.push(uuid.clone());
            }
            // Actually, let's rebuild all_nodes list with UUIDs.
            if let EngineConfig::Disaggregated { prefill, decode } = config {
                for (i, node) in prefill.iter().enumerate() {
                    let adapter = self.prefill_engines[i].lock().await;
                    let uuid = adapter
                        .uuid
                        .clone()
                        .unwrap_or_else(|| format!("prefill-{}", i));
                    all_nodes.push((
                        uuid,
                        node.host.clone(),
                        node.port,
                        "prefill".to_string(),
                        adapter.world_size,
                        adapter.num_blocks,
                    ));
                }
                for (i, node) in decode.iter().enumerate() {
                    let adapter = self.decode_engines[i].lock().await;
                    let uuid = adapter
                        .uuid
                        .clone()
                        .unwrap_or_else(|| format!("decode-{}", i));
                    decode_uuids.push(uuid.clone());
                    all_nodes.push((
                        uuid,
                        node.host.clone(),
                        node.port,
                        "decode".to_string(),
                        adapter.world_size,
                        adapter.num_blocks,
                    ));
                }
            }

            // 1. Send P2PInit to ALL nodes & Collect Info
            info!("Broadcasting P2PInit to all {} nodes...", all_nodes.len());

            let mut node_responses: HashMap<String, Vec<u8>> = HashMap::new();

            for engine in &self.prefill_engines {
                let mut adapter = engine.lock().await;
                let my_uuid = adapter.uuid.clone().unwrap_or_default();
                let resp = adapter.send_p2p_init(all_nodes.clone()).await?;
                node_responses.insert(my_uuid, resp);
            }
            for engine in &self.decode_engines {
                let mut adapter = engine.lock().await;
                let my_uuid = adapter.uuid.clone().unwrap_or_default();
                let resp = adapter.send_p2p_init(all_nodes.clone()).await?;
                node_responses.insert(my_uuid, resp);
            }

            // 2. Connect Prefill -> Decode
            info!(
                "Instructing Prefill nodes to connect to {} Decode nodes...",
                decode_uuids.len()
            );
            for engine in &self.prefill_engines {
                let my_uuid = {
                    let adapter = engine.lock().await;
                    adapter.uuid.clone().unwrap_or_default()
                };
                let mut target_map = HashMap::new();

                for target_uuid in &decode_uuids {
                    if let Some(target_resp_bytes) = node_responses.get(target_uuid) {
                        use crate::fbs::nanodeploy::connection::nanodeploy::fbs::P2PInitResponse;
                        if let Ok(p2p_resp) =
                            flatbuffers::root::<P2PInitResponse>(target_resp_bytes)
                        {
                            if let Some(peers) = p2p_resp.responses() {
                                for i in 0..peers.len() {
                                    let peer = peers.get(i);
                                    if peer.id() == Some(my_uuid.as_str()) {
                                        if let Some(local_info) = peer.local_info() {
                                            target_map
                                                .insert(target_uuid.clone(), local_info.to_vec());
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                if !target_map.is_empty() {
                    let mut adapter = engine.lock().await;
                    adapter.send_p2p_connect(target_map).await?;
                }
            }

            // 3. Connect Decode -> Prefill
            info!(
                "Instructing Decode nodes to connect to {} Prefill nodes...",
                prefill_uuids.len()
            );
            for engine in &self.decode_engines {
                let my_uuid = {
                    let adapter = engine.lock().await;
                    adapter.uuid.clone().unwrap_or_default()
                };
                let mut target_map = HashMap::new();

                for target_uuid in &prefill_uuids {
                    if let Some(target_resp_bytes) = node_responses.get(target_uuid) {
                        use crate::fbs::nanodeploy::connection::nanodeploy::fbs::P2PInitResponse;
                        if let Ok(p2p_resp) =
                            flatbuffers::root::<P2PInitResponse>(target_resp_bytes)
                        {
                            if let Some(peers) = p2p_resp.responses() {
                                for i in 0..peers.len() {
                                    let peer = peers.get(i);
                                    if peer.id() == Some(my_uuid.as_str()) {
                                        if let Some(local_info) = peer.local_info() {
                                            target_map
                                                .insert(target_uuid.clone(), local_info.to_vec());
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                if !target_map.is_empty() {
                    let mut adapter = engine.lock().await;
                    adapter.send_p2p_connect(target_map).await?;
                }
            }
        }
        Ok(())
    }

    /// Start watching Etcd for node updates
    pub async fn start_etcd_watch(manager: Arc<Mutex<EngineManager>>, config: EtcdConfig) {
        let manager_clone = manager.clone();
        tokio::spawn(async move {
            let prefix = format!("/nanodeploy/mesh/{}/nodes/", config.cluster_id);
            info!("Starting Etcd discovery on prefix: {}", prefix);

            // Connect to Etcd
            let mut client = match Client::connect([&config.address], None).await {
                Ok(c) => c,
                Err(e) => {
                    error!("Failed to connect to Etcd at {}: {}", config.address, e);
                    return;
                }
            };

            // 1. Initial Scan
            match client
                .get(prefix.clone(), Some(GetOptions::new().with_prefix()))
                .await
            {
                Ok(resp) => {
                    let kvs = resp.kvs();
                    info!("Etcd scan found {} existing nodes.", kvs.len());
                    for kv in kvs {
                        let val = kv.value();
                        EngineManager::handle_node_update(&manager_clone, val).await;
                    }
                }
                Err(e) => error!("Etcd initial scan failed: {}", e),
            }

            // 2. Watch Loop
            match client
                .watch(prefix, Some(WatchOptions::new().with_prefix()))
                .await
            {
                Ok((_watcher, mut stream)) => {
                    info!("Etcd watch established.");
                    while let Some(resp_result) = stream.next().await {
                        match resp_result {
                            Ok(resp) => {
                                for event in resp.events() {
                                    match event.event_type() {
                                        EventType::Put => {
                                            if let Some(kv) = event.kv() {
                                                EngineManager::handle_node_update(
                                                    &manager_clone,
                                                    kv.value(),
                                                )
                                                .await;
                                            }
                                        }
                                        EventType::Delete => {
                                            if let Some(kv) = event.kv() {
                                                let key = kv.key_str().unwrap_or_default();
                                                warn!("Node deleted: {}. Removal not fully implemented.", key);
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Etcd watch stream error: {}", e);
                                break;
                            }
                        }
                    }
                }
                Err(e) => error!("Failed to start Etcd watch: {}", e),
            }
        });
    }

    async fn handle_node_update(manager: &Arc<Mutex<EngineManager>>, val: &[u8]) {
        // Parse FlatBuffer
        let info = match flatbuffers::root::<EngineInfo>(val) {
            Ok(i) => i,
            Err(e) => {
                error!("Failed to parse EngineInfo from Etcd: {}", e);
                return;
            }
        };
        let uuid = info.id().unwrap_or_default();
        let role = info.role().unwrap_or_default();
        let host = info.host().unwrap_or_default();
        let port = info.port() as u16;
        let status = info.status().unwrap_or("ready"); // Default to ready if missing

        // We connect to the "external" host/port.
        let addr = format!("{}:{}", host, port);

        let mut mgr = manager.lock().await;

        // 1. Ensure Connection
        if !mgr.known_nodes.contains_key(uuid) {
            info!(
                "Discovered new node: {} ({}) at {} [status={}]",
                uuid, role, addr, status
            );
            let mut adapter = EngineAdapter::new(uuid.to_string());
            adapter.uuid = Some(uuid.to_string());
            adapter.world_size = info.world_size();
            adapter.num_blocks = info.num_blocks();

            if let Err(e) = adapter.connect(&addr).await {
                error!("Failed to connect to discovered node {}: {}", uuid, e);
                return;
            }
            mgr.known_nodes
                .insert(uuid.to_string(), Arc::new(Mutex::new(adapter)));
        } else {
            // Update metadata if needed? For now just log
            // info!("Received update for node: {} [status={}]", uuid, status);
        }

        let adapter_arc = mgr.known_nodes.get(uuid).unwrap().clone();

        // 2. Manage Routing Lists based on Status
        if status == "ready" {
            if role == "prefill" {
                // Check if already in list
                let exists = mgr.prefill_engines.iter().any(|a| {
                    // We can't easily check UUID without locking.
                    // But we have the Arc pointer equality!
                    // If we inserted the SAME Arc from known_nodes, PtrEq works.
                    Arc::ptr_eq(a, &adapter_arc)
                });
                if !exists {
                    info!("Node {} is ready. Adding to Prefill pool.", uuid);
                    mgr.prefill_engines.push(adapter_arc);
                }
            } else if role == "decode" {
                let exists = mgr
                    .decode_engines
                    .iter()
                    .any(|a| Arc::ptr_eq(a, &adapter_arc));
                if !exists {
                    info!("Node {} is ready. Adding to Decode pool.", uuid);
                    mgr.decode_engines.push(adapter_arc);
                }
            }
        } else {
            // Remove from pool if present (e.g. went back to initializing or unhealthy)
            if role == "prefill" {
                mgr.prefill_engines
                    .retain(|a| !Arc::ptr_eq(a, &adapter_arc));
            } else if role == "decode" {
                mgr.decode_engines.retain(|a| !Arc::ptr_eq(a, &adapter_arc));
            }
        }
    }
}
