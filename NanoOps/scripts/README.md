# NanoOps Cleanup Scripts

This directory contains cleanup utilities to help reset NanoOps state before testing.

## Quick Cleanup

### Option 1: Using CLI command (Recommended)

```bash
# Clean up with defaults (demo, demo2, test sessions)
nanoctrl cleanup

# Clean specific sessions only
nanoctrl cleanup --sessions "mysession,test1,test2"

# Clean without killing processes
nanoctrl cleanup --no-kill-processes

# Custom Redis URL
nanoctrl cleanup --redis-url redis://localhost:6379
```

### Option 2: Using bash script

```bash
# Run the cleanup script
./scripts/cleanup.sh

# Or with custom Redis
REDIS_HOST=localhost REDIS_PORT=6379 ./scripts/cleanup.sh
```

### Option 3: Using Python directly

```python
from nanoops.cleanup import cleanup_all

# Clean everything
cleanup_all(
    redis_url="redis://10.102.97.179:6379",
    sessions=["demo", "demo2", "test"],
    kill_processes_flag=True,
)
```

## What Gets Cleaned

1. **Processes**

   - `engine_server` processes (NanoDeploy engines)
   - `nanoroute` processes (NanoRoute instances)

2. **Redis Keys**

   - Session data: `{session_id}:*`
   - Stale peer_agents: `*:agent:*`
   - Unscoped engine keys: `engine:*`
   - Old scope data: `JimyMa:*` (or other specified scopes)

3. **Ray Jobs** (optional, manual in bash script)

   - Can stop running Ray jobs if needed

## When to Use

Run cleanup before starting a fresh test session if you encounter:

- "Session already exists" errors
- Stale peer_agent connection errors
- ZMQ packet corruption errors
- Ray worker ID mismatch warnings
- Old engines appearing in `list_engines` output

## Example Workflow

```bash
# 1. Clean up from previous tests
nanoctrl cleanup

# 2. Start fresh session
nanoctrl start --session-id fresh \
  --redis-url redis://10.102.97.179:6379 \
  --ray-address http://10.102.97.179:8265 \
  --nanoctrl-address http://10.102.97.179:3000

# 3. Configure and spawn
nanoctrl set --session-id fresh \
  --model /models/qwen3-235B-Instruct-2507-FP8 \
  --prefill-attention-dp 8 --prefill-ffn-ep 8 \
  --decode-attention-dp 8 --decode-ffn-ep 8

nanoctrl spawn prefill --session-id fresh
nanoctrl spawn decode --session-id fresh
nanoctrl spawn route --session-id fresh

# 4. Wait for ready
nanoctrl wait --session-id fresh --timeout 300
```

## Notes

- The cleanup command is **safe** - it only removes NanoOps-related resources
- Redis cleanup uses pattern matching to target specific keys
- Process killing uses `-9` (SIGKILL) to ensure processes are stopped
- Ray job cleanup is optional and must be confirmed in the bash script
