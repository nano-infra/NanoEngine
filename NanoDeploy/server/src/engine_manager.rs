use crate::config::EngineConfig;
use crate::engine_adapter::EngineAdapter;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;
use serde_json::Value;

pub struct EngineManager {
    // We share adapters via Arc<Mutex> because multiple threads (http server) might access them
    // effectively, though usually the Scheduler will be the sole owner.
    // For now, let's keep them as Arc<Mutex<...>> for compatibility with current main.rs style
    prefill_engines: Vec<Arc<Mutex<EngineAdapter>>>,
    decode_engines: Vec<Arc<Mutex<EngineAdapter>>>,
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
                    info!("Connecting to Prefill Engine #{} at {}:{}", i, node.host, node.port);
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
                    info!("Connecting to Decode Engine #{} at {}:{}", i, node.host, node.port);
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
        // TODO: Round robin state
        self.prefill_engines.first().cloned()
    }

    pub fn get_next_decode(&self) -> Option<Arc<Mutex<EngineAdapter>>> {
        // TODO: Round robin state. For now just pick first.
        self.decode_engines.first().cloned()
    }

    pub async fn initialize_p2p_mesh(&mut self, config: &EngineConfig) -> anyhow::Result<()> {
        let mut all_nodes = Vec::new();
        let mut decode_uuids = Vec::new();
        let mut prefill_uuids = Vec::new();

        if let EngineConfig::Disaggregated { prefill: _, decode: _ } = config {
            // 1. Gather all info using UUIDs
            for (i, engine) in self.prefill_engines.iter().enumerate() {
                let adapter = engine.lock().await;
                let uuid = adapter.uuid.clone().unwrap_or_else(|| format!("prefill-{}", i));
                prefill_uuids.push(uuid.clone());
                // We don't have the external host/port easily here, but we can get it from config if needed.
                // However, P2PInit payload 'host'/'port' are used for RDMA address resolution if needed.
                // For now, let's just pass the ones used for connection.
            }
            // Actually, let's rebuild all_nodes list with UUIDs.
            // We need to match adapter back to config for host/port.
            if let EngineConfig::Disaggregated { prefill, decode } = config {
                 for (i, node) in prefill.iter().enumerate() {
                     let adapter = self.prefill_engines[i].lock().await;
                     let uuid = adapter.uuid.clone().unwrap_or_else(|| format!("prefill-{}", i));
                     all_nodes.push((uuid, node.host.clone(), node.port, "prefill".to_string(), adapter.world_size, adapter.num_blocks));
                 }
                 for (i, node) in decode.iter().enumerate() {
                     let adapter = self.decode_engines[i].lock().await;
                     let uuid = adapter.uuid.clone().unwrap_or_else(|| format!("decode-{}", i));
                     decode_uuids.push(uuid.clone());
                     all_nodes.push((uuid, node.host.clone(), node.port, "decode".to_string(), adapter.world_size, adapter.num_blocks));
                 }
            }

            // 1. Send P2PInit to ALL nodes & Collect Info
            info!("Broadcasting P2PInit to all {} nodes...", all_nodes.len());

            // Map: NodeUUID -> (Map: TargetUUID -> Info)
            let mut node_responses: HashMap<String, serde_json::Map<String, Value>> = HashMap::new();

            // Prefill Init
            for engine in &self.prefill_engines {
                 let mut adapter = engine.lock().await;
                 let my_uuid = adapter.uuid.clone().unwrap_or_default();
                 let resp = adapter.send_p2p_init(all_nodes.clone()).await?;
                 if let Value::Object(map) = resp {
                     node_responses.insert(my_uuid, map);
                 }
            }
            // Decode Init
            for engine in &self.decode_engines {
                 let mut adapter = engine.lock().await;
                 let my_uuid = adapter.uuid.clone().unwrap_or_default();
                 let resp = adapter.send_p2p_init(all_nodes.clone()).await?;
                 if let Value::Object(map) = resp {
                     node_responses.insert(my_uuid, map);
                 }
            }

            // 2. Connect Prefill -> Decode
            info!("Instructing Prefill nodes to connect to {} Decode nodes...", decode_uuids.len());
            for engine in &self.prefill_engines {
                let my_uuid = {
                    let adapter = engine.lock().await;
                    adapter.uuid.clone().unwrap_or_default()
                };
                let mut target_map = HashMap::new();

                for target_uuid in &decode_uuids {
                    if let Some(target_resp) = node_responses.get(target_uuid) {
                        if let Some(info) = target_resp.get(&my_uuid) {
                            target_map.insert(target_uuid.clone(), info.clone());
                        }
                    }
                }

                if !target_map.is_empty() {
                    let mut adapter = engine.lock().await;
                    adapter.send_p2p_connect(target_map).await?;
                }
            }

            // 3. Connect Decode -> Prefill
            info!("Instructing Decode nodes to connect to {} Prefill nodes...", prefill_uuids.len());
            for engine in &self.decode_engines {
                let my_uuid = {
                    let adapter = engine.lock().await;
                    adapter.uuid.clone().unwrap_or_default()
                };
                 let mut target_map = HashMap::new();

                for target_uuid in &prefill_uuids {
                     if let Some(target_resp) = node_responses.get(target_uuid) {
                        if let Some(info) = target_resp.get(&my_uuid) {
                            target_map.insert(target_uuid.clone(), info.clone());
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
}
