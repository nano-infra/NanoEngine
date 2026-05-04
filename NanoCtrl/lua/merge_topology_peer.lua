-- Atomically merge one caller into a target peer's desired topology spec.
--
-- KEYS[1] = peer spec key (spec:topology:{target})
-- ARGV[1] = caller agent id to merge into target_peers
-- ARGV[2] = scope to set when creating/updating a scoped spec, or ""

local existing = redis.call('GET', KEYS[1])
local spec

if existing then
    local ok, decoded = pcall(cjson.decode, existing)
    if ok and type(decoded) == 'table' then
        spec = decoded
    else
        spec = {target_peers = {}}
    end
else
    spec = {target_peers = {}}
end

if type(spec.target_peers) ~= 'table' then
    spec.target_peers = {}
end

local already = false
for _, p in ipairs(spec.target_peers) do
    if p == ARGV[1] then
        already = true
        break
    end
end

if not already then
    table.insert(spec.target_peers, ARGV[1])
end

if ARGV[2] ~= '' and (spec.scope == nil or spec.scope == cjson.null) then
    spec.scope = ARGV[2]
end

redis.call('SET', KEYS[1], cjson.encode(spec))
return already and 0 or 1
