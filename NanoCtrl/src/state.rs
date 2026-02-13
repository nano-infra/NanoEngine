use redis::Client;

/// Engine TTL in seconds (for heartbeat mechanism)
/// Engine must send heartbeat every 15 seconds to keep alive
/// TTL is set to 60 seconds to allow 4 missed heartbeats before expiration
pub const ENGINE_TTL_SECS: usize = 60;

/// Lua script for atomically registering engine + increment revision + publish event + set TTL
///
/// KEYS[1] = engine key (e.g., "engine:{id}")
/// KEYS[2] = revision key (e.g., "nano_meta:engine_revision")
/// KEYS[3] = pubsub channel (e.g., "nano_events:engine_update")
/// ARGV[1..8] = engine fields (id, role, host, port, world_size, num_blocks, peer_addrs, info)
/// ARGV[9] = event payload JSON string (complete payload object)
/// ARGV[10] = TTL in seconds
///
/// Returns: new revision number
pub const REGISTER_ENGINE_SCRIPT: &str = r#"
    local engine_key = KEYS[1]
    local revision_key = KEYS[2]
    local channel = KEYS[3]
    local ttl = tonumber(ARGV[10])

    -- 1. Write engine data to hash
    redis.call('HSET', engine_key,
        'id', ARGV[1],
        'role', ARGV[2],
        'host', ARGV[3],
        'port', ARGV[4],
        'world_size', ARGV[5],
        'num_blocks', ARGV[6],
        'peer_addrs', ARGV[7],
        'info', ARGV[8]
    )

    -- 2. Set Expiration (Heartbeat mechanism - prevents zombie nodes)
    redis.call('EXPIRE', engine_key, ttl)

    -- 3. Atomically increment global revision
    local new_revision = redis.call('INCR', revision_key)

    -- 4. Construct and publish event (using cjson for JSON manipulation)
    local payload_data = cjson.decode(ARGV[9])
    local event = {
        event_type = 'ADD',
        engine_id = ARGV[1],
        timestamp = redis.call('TIME')[1],  -- Server time (seconds since epoch)
        revision = new_revision,
        payload = payload_data
    }
    local event_json = cjson.encode(event)

    -- 5. Publish event atomically (within the same transaction)
    redis.call('PUBLISH', channel, event_json)

    return new_revision
"#;

/// Lua script for atomically unregistering engine + increment revision + publish event
///
/// KEYS[1] = engine key (e.g., "engine:{id}")
/// KEYS[2] = revision key (e.g., "nano_meta:engine_revision")
/// KEYS[3] = pubsub channel (e.g., "nano_events:engine_update")
/// ARGV[1] = engine_id
///
/// Returns: new revision number, or 0 if engine not found
pub const UNREGISTER_ENGINE_SCRIPT: &str = r#"
    local engine_key = KEYS[1]
    local revision_key = KEYS[2]
    local channel = KEYS[3]
    local engine_id = ARGV[1]

    -- 1. Check if engine exists
    local exists = redis.call('EXISTS', engine_key)

    if exists == 0 then
        return 0
    end

    -- 2. Delete engine
    redis.call('DEL', engine_key)

    -- 3. Atomically increment global revision
    local new_revision = redis.call('INCR', revision_key)

    -- 4. Construct and publish REMOVE event
    local event = {
        event_type = 'REMOVE',
        engine_id = engine_id,
        timestamp = redis.call('TIME')[1],  -- Server time
        revision = new_revision,
        payload = nil
    }
    local event_json = cjson.encode(event)

    -- 5. Publish event atomically
    redis.call('PUBLISH', channel, event_json)

    return new_revision
"#;

/// Lua script for heartbeat: only refresh TTL, no event, no revision increment
///
/// KEYS[1] = engine key (e.g., "engine:{id}")
/// ARGV[1] = TTL in seconds
///
/// Returns: 1 if engine exists and TTL refreshed, 0 if engine not found
pub const HEARTBEAT_ENGINE_SCRIPT: &str = r#"
    local engine_key = KEYS[1]
    local ttl = tonumber(ARGV[1])

    -- 1. Check if engine exists
    local exists = redis.call('EXISTS', engine_key)

    if exists == 0 then
        return 0
    end

    -- 2. Refresh TTL only (no data change, no event, no revision increment)
    redis.call('EXPIRE', engine_key, ttl)

    return 1
"#;

#[derive(Clone)]
pub struct AppState {
    /// Redis client (MultiplexedConnection is Clone and handles connection pooling)
    pub redis_client: Client,
    pub redis_url: String,
}

impl AppState {
    pub fn new(redis_url: &str) -> anyhow::Result<Self> {
        let client = Client::open(redis_url)?;
        Ok(Self {
            redis_client: client,
            redis_url: redis_url.to_string(),
        })
    }

    /// Build Redis key prefix from scope (provided by client in API requests)
    /// Returns scope string if present, or empty string if no scope
    pub fn build_key_prefix(&self, scope: Option<&str>) -> String {
        scope.map(|s| s.to_string()).unwrap_or_default()
    }

    /// Generate scoped engine key
    pub fn engine_key(&self, engine_id: &str, scope: Option<&str>) -> String {
        let prefix = self.build_key_prefix(scope);
        if prefix.is_empty() {
            format!("engine:{}", engine_id)
        } else {
            format!("{}:engine:{}", prefix, engine_id)
        }
    }

    /// Generate scoped revision key
    pub fn revision_key(&self, scope: Option<&str>) -> String {
        let prefix = self.build_key_prefix(scope);
        if prefix.is_empty() {
            "nano_meta:engine_revision".to_string()
        } else {
            format!("{}:nano_meta:engine_revision", prefix)
        }
    }

    /// Generate scoped events channel
    pub fn events_channel(&self, scope: Option<&str>) -> String {
        let prefix = self.build_key_prefix(scope);
        if prefix.is_empty() {
            "nano_events:engine_update".to_string()
        } else {
            format!("{}:nano_events:engine_update", prefix)
        }
    }
}
