# Migration Guide: Manual to Automatic Peer Discovery

This guide helps you migrate from manual `set_peer_info()` calls to automatic peer discovery via NanoCtrl.

## What's Changed

### Before (Manual Setup)

```python
import ray
from nanodeploy.server.llm_component import LLMComponent

# Start engines
prefill = LLMComponent.options(num_gpus=1).remote(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_disaggregated_prefill=True,
)

decode = LLMComponent.options(num_gpus=1).remote(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_disaggregated_prefill=False,
)

# ❌ Manual peer setup required
prefill_info = ray.get(prefill.get_engine_info.remote())
decode_info = ray.get(decode.get_engine_info.remote())
ray.get(prefill.set_peer_info.remote(decode_info))
ray.get(decode.set_peer_info.remote(prefill_info))

# Now ready to generate
```

### After (Automatic Discovery)

```python
import ray
from nanodeploy.server.llm_component import LLMComponent

# Start engines with nanoctrl_address
prefill = LLMComponent.options(num_gpus=1).remote(
    model="meta-llama/Llama-3.1-8B-Instruct",
    nanoctrl_address="localhost:3000",  # ✅ Add this
    enable_disaggregated_prefill=True,
)

decode = LLMComponent.options(num_gpus=1).remote(
    model="meta-llama/Llama-3.1-8B-Instruct",
    nanoctrl_address="localhost:3000",  # ✅ Add this
    enable_disaggregated_prefill=False,
)

# ✅ No manual setup needed! Engines auto-discover each other
# Ready to generate immediately
```

## Migration Steps

### Step 1: Start NanoCtrl

You need a running NanoCtrl instance:

```bash
# Install dependencies
cd /path/to/NanoCtrl
cargo build --release

# Start NanoCtrl
export REDIS_URL=redis://localhost:6379
./target/release/nanoctrl --server-address 0.0.0.0:3000
```

### Step 2: Update Engine Configuration

Add `nanoctrl_address` to your engine configuration:

```python
from nanodeploy import EngineConfig

config = EngineConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    nanoctrl_address="localhost:3000",  # ✅ Add this line
    enable_disaggregated_prefill=True,
    # ... rest of config
)
```

Or if using LLMComponent directly:

```python
engine = LLMComponent.options(num_gpus=1).remote(
    model="meta-llama/Llama-3.1-8B-Instruct",
    nanoctrl_address="localhost:3000",  # ✅ Add this parameter
    enable_disaggregated_prefill=True,
)
```

### Step 3: Remove Manual Peer Setup

Delete the manual `set_peer_info()` calls:

```python
# ❌ Remove these lines:
# prefill_info = ray.get(prefill.get_engine_info.remote())
# decode_info = ray.get(decode.get_engine_info.remote())
# ray.get(prefill.set_peer_info.remote(decode_info))
# ray.get(decode.set_peer_info.remote(prefill_info))
```

### Step 4: Recompile (If Using C++ Components)

If you've updated the C++ code:

```bash
cd /path/to/NanoSequence
pip install -e . --force-reinstall

cd /path/to/NanoDeploy
pip install -e . --force-reinstall

cd /path/to/DLSlime
pip install -e . --force-reinstall
```

### Step 5: Test

Run your test script:

```bash
python pd_disagg.py
```

You should see logs like:

```
[INFO] Registered with NanoCtrl successfully: prefill
[INFO] Registered with NanoCtrl successfully: decode
[DEBUG] Peer endpoints details: {'decode': ['192.168.1.10:50051']}
```

## Configuration Options

### NanoCtrl Address Formats

The `nanoctrl_address` parameter accepts:

- `localhost:3000` - Local NanoCtrl
- `192.168.1.100:3000` - Remote NanoCtrl by IP
- `nanoctrl.example.com:3000` - Remote NanoCtrl by hostname

### Engine ID

By default, engines use auto-generated IDs. To specify custom IDs:

```python
config = EngineConfig(
    engine_id="my_prefill_engine",  # Custom ID
    nanoctrl_address="localhost:3000",
    # ...
)
```

Custom IDs are useful for:

- Identifying engines in logs
- Manual peer setup (if needed)
- Monitoring and debugging

### Peer Ports

Engines expose RDMA endpoints on specific ports. By default, ports are assigned automatically. To specify custom ports:

```python
config = EngineConfig(
    peer_ports=[50051, 50052],  # Custom ports
    nanoctrl_address="localhost:3000",
    # ...
)
```

## Backward Compatibility

### Still Need Manual Setup?

If you can't use NanoCtrl, manual setup still works:

```python
# Don't set nanoctrl_address
engine = LLMComponent.options(num_gpus=1).remote(
    model="meta-llama/Llama-3.1-8B-Instruct",
    # nanoctrl_address not set
    enable_disaggregated_prefill=True,
)

# Manual setup
prefill_info = ray.get(prefill.get_engine_info.remote())
ray.get(decode.set_peer_info.remote(prefill_info))
```

### Hybrid Setup

You can mix automatic and manual:

```python
# Automatic discovery for most peers
config = EngineConfig(
    nanoctrl_address="localhost:3000",
)

# Manual override for specific peer
ray.get(engine.set_peer_info.remote(special_peer_info))
```

Manual peers take precedence over automatic discovery.

## Troubleshooting

### Issue: "Failed to connect to NanoCtrl"

**Check:**

1. Is NanoCtrl running? `curl http://localhost:3000/health`
2. Is the address correct? Use `localhost:3000`, not `http://localhost:3000`
3. Is Redis running? `redis-cli PING`

**Solution:**

```bash
# Start Redis
redis-server

# Start NanoCtrl
cd /path/to/NanoCtrl
cargo run -- --server-address 0.0.0.0:3000
```

### Issue: "Timeout waiting for peers"

**Check:**

1. Are both engines registered? `curl -X POST http://localhost:3000/list_engines`
2. Are peer_addrs populated?
3. Is Redis accessible to both engines?

**Solution:**

```bash
# Check Redis keys
redis-cli KEYS "*engine*"

# Should see:
# :engine:prefill
# :engine:decode

# Check engine data
redis-cli GET ":engine:prefill"
```

### Issue: "peer_endpoints is empty"

**Check:**

1. Did you wait for registration? (It's async)
2. Is heartbeat running? Check logs for "Heartbeat failed"
3. Are engines TTL-expired? Redis key TTL is 60 seconds

**Solution:**

```python
# Add a small delay after engine creation
import time
time.sleep(1)  # Let registration complete

# Then proceed with generation
```

### Issue: "Compilation errors after upgrade"

**Solution:**

```bash
# Clean rebuild
cd /path/to/NanoSequence
rm -rf build dist *.egg-info
pip install -e . --force-reinstall --no-cache-dir

cd /path/to/NanoDeploy
rm -rf build dist *.egg-info
pip install -e . --force-reinstall --no-cache-dir

cd /path/to/DLSlime
rm -rf build dist *.egg-info
pip install -e . --force-reinstall --no-cache-dir
```

## Performance Considerations

### Caching

Automatic discovery caches results for 10 seconds. This means:

- **Advantage**: Reduced load on NanoCtrl
- **Disadvantage**: Up to 10-second delay to discover new peers

To disable caching (not recommended):

```python
# In llm_engine.py
_PEER_ENDPOINTS_CACHE_TTL = 0  # Disable cache
```

### Heartbeat Overhead

Engines send heartbeat every 30 seconds:

- **Network overhead**: ~1 KB per heartbeat
- **CPU overhead**: Negligible (~0.01% CPU)
- **Latency impact**: None (runs in background thread)

### Registration Latency

First generation after engine start:

- **Without NanoCtrl**: 0ms (manual setup is synchronous)
- **With NanoCtrl**: ~50-100ms (includes HTTP round-trip)

Subsequent generations:

- **Without NanoCtrl**: 0ms
- **With NanoCtrl**: ~0ms (cached)

## Best Practices

### 1. Always Set nanoctrl_address

Even for single-engine setups, it enables monitoring:

```python
config = EngineConfig(
    nanoctrl_address="localhost:3000",  # Always set this
)
```

### 2. Use Custom Engine IDs in Production

For better debugging and monitoring:

```python
config = EngineConfig(
    engine_id=f"prefill_{region}_{replica_id}",
    nanoctrl_address="nanoctrl.prod.example.com:3000",
)
```

### 3. Monitor NanoCtrl Health

Add health checks in your orchestration:

```bash
# Kubernetes liveness probe
curl -f http://nanoctrl:3000/health || exit 1
```

### 4. Set Explicit Peer Ports

To avoid port conflicts:

```python
config = EngineConfig(
    peer_ports=[50051 + replica_id * 10 for _ in range(num_ports)],
)
```

### 5. Use Redis Persistence

To survive restarts:

```bash
# In redis.conf
appendonly yes
appendfsync everysec
```

## Rollback Plan

If automatic discovery causes issues, you can rollback:

### Immediate Rollback (No Code Change)

Simply don't set `nanoctrl_address`:

```python
# Remove or comment out nanoctrl_address
engine = LLMComponent.options(num_gpus=1).remote(
    model="...",
    # nanoctrl_address="localhost:3000",  # Disabled
    enable_disaggregated_prefill=True,
)

# Use manual setup
ray.get(engine.set_peer_info.remote(peer_info))
```

### Full Rollback (Git Revert)

```bash
# Find commit before automatic discovery
git log --oneline | grep "automatic peer discovery"

# Revert to before that commit
git revert <commit-hash>

# Rebuild
pip install -e . --force-reinstall
```

## Migration Checklist

- [ ] NanoCtrl is running and accessible
- [ ] Redis is running with appropriate persistence settings
- [ ] Added `nanoctrl_address` to all engine configurations
- [ ] Removed manual `set_peer_info()` calls
- [ ] Recompiled all C++ components (if changed)
- [ ] Tested in development environment
- [ ] Monitored logs for registration success
- [ ] Verified RDMA connections are established
- [ ] Tested generation with automatic migration
- [ ] Performance is acceptable
- [ ] Prepared rollback plan

## Further Reading

- [PREFILL_DECODE_DISAGGREGATION.md](PREFILL_DECODE_DISAGGREGATION.md) - Architecture overview
- [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md) - Detailed issue analysis
- [NanoCtrl API Documentation](../NanoCtrl/API.md) - REST API reference

## Support

If you encounter issues:

1. Check [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
2. Review logs with: `tail -f log.log | grep -E "(ERROR|WARNING|peer)"`
3. Check Redis: `redis-cli MONITOR`
4. Check NanoCtrl: `curl -X POST http://localhost:3000/list_engines`

For bugs or feature requests, please open an issue with:

- Full error message and stack trace
- NanoCtrl version and configuration
- Engine configuration
- Redis key dump: `redis-cli KEYS "*"`
