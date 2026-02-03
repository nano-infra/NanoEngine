use redis::Client;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::oneshot;
use tokio::sync::Mutex;

/// Cache TTL for get_mr_info (seconds)
pub const MR_INFO_CACHE_TTL_SECS: u64 = 10;

#[derive(Clone)]
pub struct AppState {
    pub redis_client: Client,
    pub redis_url: String,
    pub locks: Arc<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
    // Event futures: conn_key -> (low_ack_sender, high_ack_sender)
    pub init_futures:
        Arc<Mutex<HashMap<String, (Option<oneshot::Sender<()>>, Option<oneshot::Sender<()>>)>>>,
    pub connect_futures: Arc<Mutex<HashMap<String, oneshot::Sender<()>>>>,
    /// Cache for get_mr_info: key "dst:mr_name" -> (cached_at, response)
    pub mr_info_cache: Arc<Mutex<HashMap<String, (Instant, crate::models::GetMrInfoResponse)>>>,
}

impl AppState {
    pub fn new(redis_url: &str) -> anyhow::Result<Self> {
        let client = Client::open(redis_url)?;
        Ok(Self {
            redis_client: client,
            redis_url: redis_url.to_string(),
            locks: Arc::new(Mutex::new(HashMap::new())),
            init_futures: Arc::new(Mutex::new(HashMap::new())),
            connect_futures: Arc::new(Mutex::new(HashMap::new())),
            mr_info_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub async fn get_lock(&self, key: String) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.locks.lock().await;
        locks
            .entry(key)
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    }
}
