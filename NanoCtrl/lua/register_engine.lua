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
