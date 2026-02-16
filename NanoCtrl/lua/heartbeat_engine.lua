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
