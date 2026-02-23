# Troubleshooting Guide: Prefill-Decode Disaggregation

This guide documents specific issues encountered during development and their solutions.

## Issue 1: SIGSEGV in FlatBuffers Serialization

### Symptoms

```
[Traceback] PC: @ nanodeploy::fbs::CreateBlockContext(flatbuffers::FlatBufferBuilder&, nanodeploy::BlockContext const*)
Stack trace (most recent call last):
  File "llm_engine.py", line 542, in step
```

### Root Cause

In `sequence.h`, the `migrate()` method was setting ACTIVE BlockContext to `nullptr`:

```cpp
int32_t migrate() {
    ensure_slot(BlockContextSlot::MIGRATE);
    ensure_slot(BlockContextSlot::ACTIVE);
    get_slot(BlockContextSlot::MIGRATE) = std::move(get_slot(BlockContextSlot::ACTIVE));
    get_slot(BlockContextSlot::ACTIVE) = nullptr;  // ❌ BUG
    return 0;
}
```

When FlatBuffers tried to serialize the sequence, it called `Pack()` which dereferenced the null pointer in `CreateBlockContext()`, causing a segmentation fault.

### Solution

Create an empty BlockContext instead of nullptr:

```cpp
int32_t migrate() {
    ensure_slot(BlockContextSlot::MIGRATE);
    ensure_slot(BlockContextSlot::ACTIVE);
    get_slot(BlockContextSlot::MIGRATE) = std::move(get_slot(BlockContextSlot::ACTIVE));
    get_slot(BlockContextSlot::ACTIVE) = std::make_unique<BlockContext>();  // ✅ FIX
    return 0;
}
```

**File:** `/mnt/nvme1n1/ml_research/majinming/src/NanoInfra/NanoSequence/nanosequence/csrc/sequence/sequence.h:121`

### Prevention

- Always ensure pointers are valid before serialization
- Use smart pointers (unique_ptr/shared_ptr) to avoid null dereferences
- Add assertions to validate state before operations

______________________________________________________________________

## Issue 2: Redis Key Prefix Mismatch

### Symptoms

```
Timeout waiting for peers... Connected: set()
peer_endpoints: {}
```

### Root Cause

NanoCtrl and PeerAgent were using different Redis key prefixes:

- **NanoCtrl**: Used `default:` prefix
- **PeerAgent**: Used empty string prefix

This caused PeerAgent to look for `spec:topology:*` while NanoCtrl wrote `default:spec:topology:*`.

### Investigation Steps

1. Check Redis keys:

```bash
redis-cli KEYS "*"
```

Output showed:

```
default:spec:topology:prefill_to_decode
default:status:topology:prefill_to_decode
```

2. Check PeerAgent code:

```python
# PeerAgent was using:
self.redis_key_prefix = ""  # Empty string

# But NanoCtrl was using:
let prefix = "default";
```

### Solution

Changed both to use empty prefix:

**NanoCtrl** (`src/state.rs:146`):

```rust
pub fn new(redis_url: &str, redis_key_prefix: Option<String>) -> anyhow::Result<Self> {
    let client = Client::open(redis_url)?;
    let prefix = redis_key_prefix.unwrap_or_else(|| "".to_string());  // Changed from "default"
    // ...
}
```

**PeerAgent** (`dlslime/peer_agent.py:209`):

```python
self.redis_key_prefix = ""  # Explicitly empty
```

### Prevention

- Document Redis key schema clearly
- Use centralized configuration for key prefixes
- Add integration tests that verify key naming

______________________________________________________________________

## Issue 3: Mangled Redis Prefix

### Symptoms

```
Redis keys: 0nano_0nano_0nano_0nano_3000:engine:prefill
```

### Root Cause

NanoCtrl `main.rs` had a bug in prefix generation:

```rust
let redis_key_prefix = std::env::var("REDIS_KEY_PREFIX").ok().or_else(|| {
    let addr = std::env::var("SERVER_ADDRESS").unwrap_or_else(|_| "0.0.0.0:3000".to_string());
    Some(addr.replace([':', '.', '-'], "_").replace("_", "nano_"))  // ❌ BUG
});
```

The `.replace("_", "nano_")` was replacing ALL underscores, including those created by the first replace:

- `0.0.0.0:3000` → `0_0_0_0_3000` → `0nano_0nano_0nano_0nano_3000`

### Solution

Removed automatic prefix generation entirely:

```rust
let redis_key_prefix = std::env::var("REDIS_KEY_PREFIX").ok();  // ✅ FIX
```

**File:** `/mnt/nvme1n1/ml_research/majinming/src/NanoInfra/NanoCtrl/src/main.rs:36-39`

### Prevention

- Avoid chained string replacements that can interact
- Test prefix generation with various inputs
- Use explicit configuration rather than auto-generation

______________________________________________________________________

## Issue 4: list_engines Prefix Stripping Bug

### Symptoms

```
/list_engines returns: {"status": "ok", "engines": []}
```

But Redis shows:

```
:engine:prefill
:engine:decode
```

### Root Cause

The `list_engines` function was using wrong prefix for key stripping:

```rust
for key in keys {
    if let Some(_engine_id) = key.strip_prefix("engine:") {  // ❌ WRONG
        // ...
    }
}
```

Should have been:

```rust
let engine_prefix = format!("{}:engine:", state.redis_key_prefix);
for key in keys {
    if let Some(_engine_id) = key.strip_prefix(&engine_prefix) {  // ✅ CORRECT
        // ...
    }
}
```

### Solution

**File:** `/mnt/nvme1n1/ml_research/majinming/src/NanoInfra/NanoCtrl/src/state.rs:826-829`

```rust
let engine_prefix = format!("{}:engine:", state.redis_key_prefix);

for key in keys {
    if let Some(_engine_id) = key.strip_prefix(&engine_prefix) {
        // Parse engine data
    }
}
```

### Prevention

- Use consistent prefix construction everywhere
- Add unit tests for key parsing
- Log warnings when keys are filtered out unexpectedly

______________________________________________________________________

## Issue 5: TypeError - BlockLocation Not Subscriptable

### Symptoms

```python
TypeError: 'nanodeploy._nanodeploy_cpp.BlockLocation' object is not subscriptable
```

At line:

```python
if source_block_idx[0] != sp_idx:  # ❌ Error here
```

### Root Cause

`BlockLocation` is a FlatBuffers struct, not a Python tuple. The code was treating it as a subscriptable sequence.

### Solution

**Step 1:** Add pybind11 binding for BlockLocation

**File:** `/mnt/nvme1n1/ml_research/majinming/src/NanoInfra/NanoSequence/nanosequence/csrc/bind/sequence_binding.cpp:41-50`

```cpp
py::class_<fbs::BlockLocation>(m, "BlockLocation")
    .def(py::init<>())
    .def(py::init<int, int>())
    .def_property_readonly("first", [](const fbs::BlockLocation& bl) {
        return bl.first();
    })
    .def_property_readonly("second", [](const fbs::BlockLocation& bl) {
        return bl.second();
    })
    .def("__repr__", [](const fbs::BlockLocation& bl) {
        return "BlockLocation(first=" + std::to_string(bl.first()) +
               ", second=" + std::to_string(bl.second()) + ")";
    });
```

**Step 2:** Update all code to use property access

**File:** `/mnt/nvme1n1/ml_research/majinming/src/NanoInfra/NanoDeploy/nanodeploy/context/cache.py`

```python
# Before:
if source_block_idx[0] != sp_idx:
    continue
remote_rank = (seq.dp_idx(BlockContextSlot.MIGRATE) * migrate_ctx.attention_sp
               + remote_block_idx[0])
assignment = (peer_alias, kv_idx, layer_idx, remote_block_idx[1], source_block_idx[1])

# After:
if source_block_idx.first != sp_idx:
    continue
remote_rank = (seq.dp_idx(BlockContextSlot.MIGRATE) * migrate_ctx.attention_sp
               + remote_block_idx.first)
assignment = (peer_alias, kv_idx, layer_idx, remote_block_idx.second, source_block_idx.second)
```

**Files Changed:**

- `nanodeploy/context/cache.py:374, 382, 398-399`
- `nanodeploy/endpoint/rpc_endpoint.py` (if applicable)

### Prevention

- Always bind FlatBuffers structs explicitly with pybind11
- Add `__repr__` methods for better debugging
- Document the interface in Python type hints

______________________________________________________________________

## Issue 6: Empty peer_endpoints from list_engines

### Symptoms

```
Peer endpoints details: {}
```

### Root Cause

The `_fetch_peer_endpoints_from_nanoctrl()` method was filtering out engines with empty peer_addrs:

```python
for eng in engines:
    engine_id = eng.get("id")
    peer_addrs = eng.get("peer_addrs", [])
    if engine_id and peer_addrs:  # ❌ Skips engines with empty peer_addrs
        peer_endpoints[engine_id] = peer_addrs
```

But engines register before they have peer connections, so `peer_addrs` can be legitimately empty.

### Solution

**File:** `/mnt/nvme1n1/ml_research/majinming/src/NanoInfra/NanoDeploy/nanodeploy/engine/llm_engine.py:274`

```python
for eng in engines:
    engine_id = eng.get("id")
    peer_addrs = eng.get("peer_addrs", [])
    if engine_id:  # ✅ Include all engines
        peer_endpoints[engine_id] = peer_addrs
```

### Prevention

- Don't filter data unnecessarily
- Empty lists are valid data
- Add logging to show filtered vs. total count

______________________________________________________________________

## Issue 7: Invalid FlatBuffer Size

### Symptoms

```
RuntimeError: Invalid FlatBuffer size: -1 (buffer capacity: 1048576)
```

### Root Cause

RPC endpoint was reading `imm_data()` without validation, and sometimes received invalid sizes.

### Solution

**File:** `/mnt/nvme1n1/ml_research/majinming/src/NanoInfra/NanoDeploy/nanodeploy/endpoint/rpc_endpoint.py:186-191`

```python
size = future.imm_data()
logger.debug(f"Received sequences size: {size} bytes")

if size <= 0 or size > buffer.numel():
    raise RuntimeError(
        f"Invalid FlatBuffer size: {size} (buffer capacity: {buffer.numel()})"
    )

return deserialize(buffer_ptr, size)
```

### Prevention

- Always validate data sizes before deserialization
- Add upper and lower bounds checks
- Log the actual size received for debugging

______________________________________________________________________

## Issue 8: Engine Key Deleted by TTL

### Symptoms

Engine registers successfully but disappears from Redis after ~60 seconds.

### Root Cause

Redis keys have TTL of 60 seconds, but heartbeat wasn't running or was failing.

### Solution

Ensure heartbeat thread is running:

```python
def _start_heartbeat_thread(self):
    """Background thread to refresh NanoCtrl registration."""
    def heartbeat_loop():
        while True:
            time.sleep(30)  # Half of TTL
            try:
                self._register_with_nanoctrl()
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")

    thread = threading.Thread(target=heartbeat_loop, daemon=True)
    thread.start()
```

### Prevention

- Heartbeat interval should be \< TTL/2
- Add exponential backoff on heartbeat failure
- Monitor heartbeat health

______________________________________________________________________

## Debugging Checklist

When encountering timeout or connection issues:

1. **Check Redis Keys**

```bash
redis-cli KEYS "*"
redis-cli GET ":engine:prefill"
redis-cli GET ":spec:topology:prefill_to_decode"
```

2. **Check NanoCtrl API**

```bash
curl -X POST http://localhost:3000/list_engines
```

3. **Check Logs**

```bash
tail -f log.log | grep -E "(peer_endpoints|timeout|RDMA)"
```

4. **Verify Network**

```bash
# Check RDMA devices
ibv_devices

# Check connectivity
ping <peer-ip>
```

5. **Check Compilation**

```bash
# Ensure all instances compiled with same version
cd NanoSequence && pip install -e . --force-reinstall
cd NanoDeploy && pip install -e . --force-reinstall
cd DLSlime && pip install -e . --force-reinstall
```

6. **Verify Configuration**

```python
# Check config values
print(f"nanoctrl_address: {config.nanoctrl_address}")
print(f"engine_id: {config.engine_id}")
print(f"peer_ports: {engine.peer_ports}")
```

______________________________________________________________________

## Performance Tuning

### RDMA Buffer Sizes

Default: 1 MB per endpoint

Increase for large sequences:

```python
server = RPCServerEndpoint(buffer_size=4*1024*1024, world_size=2)
```

### Cache TTL

Peer endpoints cache: 10 seconds

```python
_PEER_ENDPOINTS_CACHE_TTL = 10.0  # Increase to reduce NanoCtrl load
```

Engine registration TTL: 60 seconds

```rust
const ENGINE_TTL: usize = 60;  // Increase for slower heartbeat
```

### Logging Levels

Production settings:

```python
# Reduce verbosity
logger.setLevel(logging.INFO)  # Change to WARNING in prod

# Remove per-block migration logs (already done)
# logger.debug(f"Migration batch {i}/{num_batches}...")  # Removed
```

______________________________________________________________________

## Common Gotchas

1. **Compilation Mismatch**: Always recompile all packages after C++ changes
2. **Ray Actor Names**: Use unique names to avoid conflicts
3. **GPU Memory**: Prefill + Decode on same GPU may OOM
4. **Port Conflicts**: Ensure peer_ports don't overlap with other services
5. **Redis Persistence**: Use AOF or snapshots to preserve topology on restart
6. **FlatBuffers Version**: Ensure same FlatBuffers version across all components

______________________________________________________________________

## Success Indicators

When everything is working:

```
✅ Logs show: "Registered with NanoCtrl successfully"
✅ Logs show: "Peer endpoints details: {'prefill': [...], 'decode': [...]}"
✅ Logs show: "RDMA link prefill_to_decode: alive"
✅ Logs show: "Sequences migrated: 1, Blocks migrated: X"
✅ Token generation produces correct output
✅ Performance: Prefill ~1900 tok/s, Decode ~300 tok/s
```

Example working log:

```
[INFO] Registered with NanoCtrl successfully: prefill
[DEBUG] Peer endpoints details: {'decode': ['192.168.1.10:50051']}
[INFO] RDMA link prefill_to_decode: alive, latency: 2.1 ms
[INFO] Migrating 1 sequence(s) to decode
[INFO] Generated 512 tokens in 1.67s (306.3 tok/s)
```
