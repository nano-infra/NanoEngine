use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, error, info, warn};

/// Engine event types
#[derive(Debug, Clone)]
pub enum EngineEvent {
    Add {
        engine_id: String,
        payload: EnginePayload,
        revision: i64,
    },
    Remove {
        engine_id: String,
        revision: i64,
    },
    Update {
        engine_id: String,
        payload: EnginePayload,
        revision: i64,
    },
    GapDetected {
        expected_revision: i64,
        actual_revision: i64,
    },
    ReconnectRequired,
}

/// Engine payload from event
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EnginePayload {
    pub id: String,
    pub role: String,
    pub host: String,
    pub port: u32,
    pub zmq_address: String,
    pub world_size: u32,
    pub num_blocks: u32,
    pub peer_addrs: Vec<String>,
    #[serde(default)]
    pub model_path: Option<String>, // None = old engine without this field
}

/// Event deduplicator to handle duplicate messages
struct EventDeduplicator {
    processed_revisions: HashSet<i64>,
    max_window_size: usize,
}

impl EventDeduplicator {
    fn new(max_window_size: usize) -> Self {
        Self {
            processed_revisions: HashSet::new(),
            max_window_size,
        }
    }

    fn is_duplicate(&mut self, revision: i64) -> bool {
        if self.processed_revisions.contains(&revision) {
            return true;
        }
        self.processed_revisions.insert(revision);

        // Limit window size to prevent memory leak
        if self.processed_revisions.len() > self.max_window_size {
            let min_revision = *self.processed_revisions.iter().min().unwrap();
            self.processed_revisions.remove(&min_revision);
        }

        false
    }
}

/// Engine Watcher for Redis Pub/Sub
pub struct EngineWatcher {
    redis_url: String,
    initial_revision: i64,
    event_tx: mpsc::UnboundedSender<EngineEvent>,
    deduplicator: Arc<Mutex<EventDeduplicator>>,
    redis_key_prefix: String,
}

impl EngineWatcher {
    pub fn new(
        redis_url: String,
        initial_revision: i64,
        redis_key_prefix: String,
    ) -> (Self, mpsc::UnboundedReceiver<EngineEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            Self {
                redis_url,
                initial_revision,
                event_tx: tx,
                deduplicator: Arc::new(Mutex::new(EventDeduplicator::new(1000))),
                redis_key_prefix,
            },
            rx,
        )
    }

    /// Start the background watcher task
    pub async fn start(mut self) -> anyhow::Result<()> {
        let mut retry_count = 0;
        const MAX_RETRIES: u32 = 10;
        const RETRY_DELAY: Duration = Duration::from_secs(5);

        loop {
            match self.run_subscription_loop().await {
                Ok(_) => {
                    // Normal exit (shouldn't happen)
                    warn!("Subscription loop exited unexpectedly");
                    break;
                }
                Err(e) => {
                    retry_count += 1;
                    if retry_count > MAX_RETRIES {
                        error!("Max retries reached, giving up");
                        return Err(e);
                    }
                    warn!(
                        "Subscription loop error (retry {}/{}): {}, reconnecting in {:?}",
                        retry_count, MAX_RETRIES, e, RETRY_DELAY
                    );

                    // Notify that reconnect is required (triggers full sync)
                    let _ = self.event_tx.send(EngineEvent::ReconnectRequired);

                    tokio::time::sleep(RETRY_DELAY).await;
                }
            }
        }

        Ok(())
    }

    async fn run_subscription_loop(&mut self) -> anyhow::Result<()> {
        use futures::StreamExt;

        let client = redis::Client::open(self.redis_url.as_str())?;
        // Note: For Pub/Sub, we need a dedicated connection (not multiplexed)
        // get_async_connection is deprecated but still needed for pubsub
        #[allow(deprecated)]
        let conn = client.get_async_connection().await?;
        let mut pubsub = conn.into_pubsub();

        // Subscribe to channel with scoped prefix
        let channel = format!("{}:nano_events:engine_update", self.redis_key_prefix);
        pubsub.subscribe(&channel).await?;
        debug!("Subscribed to {}", channel);

        // Get current revision at subscription time (before converting to stream)
        let subscription_revision = self.get_current_revision_direct().await?;
        let mut last_seen_revision = self.initial_revision;
        let mut first_message = true;

        debug!(
            "Subscription established: initial_revision={}, subscription_revision={}",
            self.initial_revision, subscription_revision
        );

        // Convert pubsub to stream
        let mut stream = pubsub.into_on_message();

        loop {
            tokio::select! {
                msg_opt = stream.next() => {
                    let msg = match msg_opt {
                        Some(msg) => msg,
                        None => {
                            warn!("Message stream ended");
                            break;
                        }
                    };

                    let payload: String = msg.get_payload()?;
                    if let Err(e) = self.handle_message(
                        payload,
                        &mut last_seen_revision,
                        &mut first_message,
                        subscription_revision,
                    ).await {
                        error!("Error handling message: {}", e);
                        // Continue processing next message, don't exit loop
                    }
                }
            }
        }

        Ok(())
    }

    async fn handle_message(
        &self,
        payload: String,
        last_seen_revision: &mut i64,
        first_message: &mut bool,
        _subscription_revision: i64,
    ) -> anyhow::Result<()> {
        let event: serde_json::Value = serde_json::from_str(&payload)?;

        let event_type = event["event_type"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing event_type"))?;
        let engine_id = event["engine_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing engine_id"))?
            .to_string();
        let revision = event["revision"]
            .as_i64()
            .ok_or_else(|| anyhow::anyhow!("Missing revision"))?;
        let timestamp = event["timestamp"].as_i64().unwrap_or(0);

        info!(
            "Received event: type={}, engine_id={}, revision={}, timestamp={}, last_seen_revision={}",
            event_type, engine_id, revision, timestamp, last_seen_revision
        );

        // Check for duplicates
        {
            let mut dedup = self.deduplicator.lock().await;
            if dedup.is_duplicate(revision) {
                warn!(
                    "Duplicate event detected: revision={}, type={}, engine_id={}",
                    revision, event_type, engine_id
                );
                return Ok(());
            }
        }

        // Gap detection: if this is the first message, check for gap
        if *first_message {
            *first_message = false;
            if revision > *last_seen_revision + 1 {
                let gap = revision - *last_seen_revision - 1;
                warn!(
                    "Gap detected! Expected revision {}, got {}. Missing {} events. Triggering full sync.",
                    *last_seen_revision + 1,
                    revision,
                    gap
                );

                // Send Gap event, trigger full sync
                let _ = self.event_tx.send(EngineEvent::GapDetected {
                    expected_revision: *last_seen_revision + 1,
                    actual_revision: revision,
                });
            }
        } else {
            // Check for non-contiguous revisions (message loss detection)
            if revision != *last_seen_revision + 1 {
                warn!(
                    "Revision gap detected: expected {}, got {}",
                    *last_seen_revision + 1,
                    revision
                );
                // Could trigger full sync here too, but for now just log
            }
        }

        *last_seen_revision = revision;

        // Handle event by type
        match event_type {
            "ADD" => {
                let payload_json = event["payload"].clone();
                let engine_payload: EnginePayload = serde_json::from_value(payload_json)?;
                info!(
                    "Sending ADD event to manager: engine_id={}, revision={}",
                    engine_id, revision
                );
                let _ = self.event_tx.send(EngineEvent::Add {
                    engine_id,
                    payload: engine_payload,
                    revision,
                });
            }
            "REMOVE" => {
                info!(
                    "Sending REMOVE event to manager: engine_id={}, revision={}",
                    engine_id, revision
                );
                let _ = self.event_tx.send(EngineEvent::Remove {
                    engine_id,
                    revision,
                });
            }
            "UPDATE" => {
                let payload_json = event["payload"].clone();
                let engine_payload: EnginePayload = serde_json::from_value(payload_json)?;
                info!(
                    "Sending UPDATE event to manager: engine_id={}, revision={}",
                    engine_id, revision
                );
                let _ = self.event_tx.send(EngineEvent::Update {
                    engine_id,
                    payload: engine_payload,
                    revision,
                });
            }
            _ => {
                warn!("Unknown event_type: {}", event_type);
            }
        }

        Ok(())
    }

    async fn get_current_revision_direct(&self) -> anyhow::Result<i64> {
        // Get current revision using a separate connection
        let client = redis::Client::open(self.redis_url.as_str())?;
        let mut conn = client.get_multiplexed_async_connection().await?;
        let revision_key = format!("{}:nano_meta:engine_revision", self.redis_key_prefix);
        let revision: Option<i64> = redis::cmd("GET")
            .arg(&revision_key)
            .query_async(&mut conn)
            .await?;
        let revision = revision.unwrap_or(0);
        Ok(revision)
    }
}
