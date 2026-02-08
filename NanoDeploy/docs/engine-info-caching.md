# Engine Info Caching and Fetching Refactoring

## Overview

This document describes the refactoring of engine information fetching and caching logic, which improves separation of concerns, reduces API overhead, and implements efficient on-demand fetching with caching.

## Motivation

Previously, the peer endpoint fetching logic was scattered across multiple components:

- `LLMEngine` handled fetching peer endpoints from NanoCtrl
- Lifecycle management (registration, heartbeat) was in `LLMEngine`
- Cache logic was split between fetch and migrate methods
- The `peer_endpoints` parameter was passed through multiple layers but never used
- Used heavy `/list_engines` API even when only specific engines were needed
- Fetched data every step even when sequences shared the same engines

## Changes

### 1. Moved Peer Info Fetching to CacheContext

**Before:** `LLMEngine` owned the `_fetch_peer_endpoints_from_nanoctrl()` method
**After:** `CacheContext` owns `_fetch_engine_info_from_nanoctrl()` method

This follows the single responsibility principle - `CacheContext` manages all cache-related operations.

### 2. Expanded from peer_endpoints to Full engine_info

**Before:** Only fetched `peer_endpoints` (list of addresses)
**After:** Fetches complete `engine_info` dict with all engine metadata

Benefits:

- More extensible for future features
- Single source of truth for engine metadata
- Reduces need for multiple API calls

### 3. Removed Unused peer_endpoints Parameter

The `peer_endpoints` parameter was passed through multiple layers but never actually used:

- Removed from `CacheContext.migrate()`
- Removed from `ModelRunner.migrate()`
- Removed from `RayExecutor.migrate()`
- Removed from `LLMEngine.step()`

### 4. Moved Lifecycle Management to LLMComponent

**Before:** `LLMEngine` handled registration, heartbeat, and unregistration
**After:** `LLMComponent` handles all lifecycle management

Moved methods:

- `_register_with_nanoctrl()`
- `_unregister_from_nanoctrl()`
- `_start_heartbeat()`
- `_heartbeat_to_nanoctrl()`
- `shutdown()`

This improves modularity - `LLMEngine` focuses on inference, `LLMComponent` handles server lifecycle.

### 5. Implemented True On-Demand Fetching

**Before:** Would call fetch method on every step
**After:** Only fetches when sequences actually need migration

```python
# Collect target engine IDs from sequences
target_engine_ids = set()
for seqs in dp_seqs:
    for seq in seqs:
        if getattr(seq, "is_to_be_migrated", False):
            ctx = seq.block_ctx(BlockContextSlot.MIGRATE)
            eid = getattr(ctx, "engine_id", None)
            if eid:
                target_engine_ids.add(eid)

# Only fetch if we have target engines
if not target_engine_ids:
    logger.debug("No target engine_ids found, skipping migration")
    return

# Get engine_info (handles caching internally)
engine_info_map = self._fetch_engine_info_from_nanoctrl(target_engine_ids)
```

### 6. Used Lightweight API Endpoint

**Before:** Used `/list_engines` to fetch all engines
**After:** Use `/get_engine_info` to fetch specific engines

```python
# Fetch missing engines from NanoCtrl
url = f"http://{self.nanoctrl_address}/get_engine_info"

for engine_id in missing_ids:
    try:
        response = client.post(url, json={"engine_id": engine_id})
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "ok":
            engine_info = data.get("engine_info", {})
            if engine_info:
                fetched_map[engine_id] = engine_info
    except Exception as e:
        logger.error(f"Error fetching engine_info for {engine_id}: {e}")
        continue
```

Benefits:

- Reduces network overhead
- Scales better when number of engines grows
- Only fetches what's needed

### 7. Set Cache TTL to Infinity

**Before:** Cache had a finite TTL
**After:** Cache TTL set to `float('inf')` (never expires)

```python
# Cache TTL for engine_info from NanoCtrl (seconds)
# Engine registration rarely changes, cache forever by default (inf means never expire)
# To refresh, call invalidate_engine_info_cache() or restart the engine
_ENGINE_INFO_CACHE_TTL = float('inf')
```

Rationale:

- Engine registration rarely changes during runtime
- Eliminates unnecessary cache misses and refetches
- Explicit invalidation method provided: `invalidate_engine_info_cache()`

### 8. Consolidated Cache Logic for Better Locality

**Before:** Cache checking split between `migrate()` and fetch method
**After:** ALL cache logic consolidated in `_fetch_engine_info_from_nanoctrl()`

The fetch method now handles:

1. Check cache for requested engine IDs
2. Identify missing IDs
3. Fetch only missing IDs from NanoCtrl
4. Update cache with new results
5. Return combined cached + fetched results

```python
def _fetch_engine_info_from_nanoctrl(self, engine_ids: set[str]) -> dict[str, dict]:
    """Get engine_info for specified engine_ids (cache + fetch if needed).

    This method handles all caching logic: checks cache, identifies missing IDs,
    fetches only missing ones from NanoCtrl, and updates cache.

    Uses the lightweight /get_engine_info endpoint instead of /list_engines.
    """
    if not engine_ids:
        return {}

    # Check cache and identify missing IDs
    engine_info_map = {}
    missing_ids = engine_ids

    if self._engine_info_cache is not None:
        cached_at, cached = self._engine_info_cache
        if time.time() - cached_at < _ENGINE_INFO_CACHE_TTL:
            # Get cached results
            engine_info_map = {
                eid: info for eid, info in cached.items() if eid in engine_ids
            }
            missing_ids = engine_ids - cached.keys()

            if not missing_ids:
                logger.debug(
                    f"All {len(engine_ids)} engines found in cache, no fetch needed"
                )
                return engine_info_map
            else:
                logger.debug(
                    f"Cache hit for {len(engine_info_map)} engines, fetching {len(missing_ids)} missing: {missing_ids}"
                )

    # Fetch missing engines and update cache...
```

## Architecture

### Before

```
LLMEngine
├── _fetch_peer_endpoints_from_nanoctrl()
├── _peer_endpoints_cache
├── _register_with_nanoctrl()
├── _heartbeat_to_nanoctrl()
└── step() -> executor.migrate(dp_sp_seqs, peer_endpoints)
    └── RayExecutor.migrate(dp_seqs, peer_endpoints)
        └── ModelRunner.migrate(seqs, peer_endpoints)
            └── CacheContext.migrate(seqs, peer_endpoints)  # parameter unused!
```

### After

```
LLMComponent
├── _register_with_nanoctrl()
├── _start_heartbeat()
├── _heartbeat_to_nanoctrl()
└── shutdown()

LLMEngine
└── step() -> executor.migrate(dp_sp_seqs)
    └── RayExecutor.migrate(dp_seqs)
        └── ModelRunner.migrate(seqs)
            └── CacheContext.migrate(seqs)
                └── _fetch_engine_info_from_nanoctrl(target_engine_ids)
                    ├── Check cache for all requested IDs
                    ├── Identify missing IDs
                    ├── Fetch only missing from /get_engine_info
                    ├── Update cache
                    └── Return combined results
```

## Performance Benefits

1. **Reduced API Calls**: Cache hit rate approaches 100% for stable engine configurations
2. **Lower Network Overhead**: Only fetch specific engines, not all engines
3. **Better Scalability**: Performance doesn't degrade as number of engines grows
4. **Eliminates Redundant Fetches**: Sequences sharing same engines don't trigger multiple fetches

## Usage

### Normal Operation (Automatic)

The caching is completely transparent - no changes needed to existing code:

```python
# In LLMEngine.step()
self.executor.migrate(dp_sp_seqs)  # CacheContext handles everything
```

### Manual Cache Invalidation

If you need to refresh engine info (e.g., after engine redeployment):

```python
from nanodeploy.context.cache import get_cache_context

# Force refresh on next fetch
get_cache_context().invalidate_engine_info_cache()
```

## API Endpoints Used

### /get_engine_info (New)

Lightweight endpoint to fetch single engine info:

**Request:**

```json
{
  "engine_id": "uuid-string"
}
```

**Response:**

```json
{
  "status": "ok",
  "engine_info": {
    "id": "uuid-string",
    "role": "decode",
    "world_size": 8,
    "num_blocks": 1024,
    "peer_addrs": ["tcp://...", "tcp://..."]
  }
}
```

### /list_engines (Deprecated for this use case)

Heavy endpoint that returns all engines - no longer used for migration.

## Future Improvements

1. **Automatic Invalidation**: Could invalidate cache when receiving engine down signals
2. **Per-Engine TTL**: Different TTLs for different engine types if needed
3. **Background Refresh**: Proactively refresh cache before TTL expires
4. **Metrics**: Track cache hit rate and fetch latencies

## Related Files

- `NanoDeploy/nanodeploy/context/cache.py` - Main caching implementation
- `NanoDeploy/nanodeploy/engine/llm_engine.py` - Simplified migration call
- `NanoDeploy/nanodeploy/server/llm_component.py` - Lifecycle management
- `NanoDeploy/nanodeploy/worker/model_runner.py` - Parameter removal
- `NanoDeploy/nanodeploy/engine/ray_executor.py` - Parameter removal

## Summary

This refactoring achieves:

- ✅ Better separation of concerns
- ✅ Improved code locality and maintainability
- ✅ Reduced API overhead and network traffic
- ✅ Efficient caching with near-zero fetch overhead for stable configs
- ✅ Cleaner parameter passing without unused arguments
- ✅ Better modularity with lifecycle management separation
