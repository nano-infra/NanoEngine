#!/bin/bash
# ============================================
# NanoOps Cleanup Script
# ============================================
# Cleans up stale processes, Ray jobs, and Redis keys
# Run this before starting a fresh test session

set -e

REDIS_HOST="${REDIS_HOST:-10.102.97.179}"
REDIS_PORT="${REDIS_PORT:-6379}"

echo "============================================"
echo "NanoOps Cleanup Script"
echo "============================================"
echo "Redis: $REDIS_HOST:$REDIS_PORT"
echo ""

# ============================================
# Step 1: Kill stale processes
# ============================================
echo "=== Step 1: Killing stale processes ==="
echo ""

echo "Killing engine_server processes..."
pkill -9 -f "engine_server" 2>/dev/null && echo "  ✓ Killed engine_server processes" || echo "  ℹ No engine_server processes found"

echo "Killing nanoroute processes..."
pkill -9 -f "nanoroute.*config" 2>/dev/null && echo "  ✓ Killed nanoroute processes" || echo "  ℹ No nanoroute processes found"

echo ""

# ============================================
# Step 2: Clean Redis keys
# ============================================
echo "=== Step 2: Cleaning Redis keys ==="
echo ""

# Function to delete keys by pattern
delete_keys() {
    local pattern=$1
    local count=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT --scan --pattern "$pattern" 2>/dev/null | wc -l)

    if [ "$count" -gt 0 ]; then
        redis-cli -h $REDIS_HOST -p $REDIS_PORT --scan --pattern "$pattern" 2>/dev/null | \
            xargs -r -L 100 redis-cli -h $REDIS_HOST -p $REDIS_PORT DEL >/dev/null 2>&1
        echo "  ✓ Deleted $count keys matching: $pattern"
    else
        echo "  ℹ No keys found matching: $pattern"
    fi
}

echo "Cleaning demo* session keys..."
delete_keys "demo:*"
delete_keys "demo2:*"
delete_keys "test*:*"

echo ""
echo "Cleaning old scope keys..."
delete_keys "JimyMa:*"

echo ""
echo "Cleaning stale agent keys (all scopes)..."
delete_keys "*:agent:*"

echo ""
echo "Cleaning stale engine keys (unscoped)..."
delete_keys "engine:*"

echo ""

# ============================================
# Step 3: Optional - Stop Ray jobs
# ============================================
echo "=== Step 3: Ray jobs cleanup (optional) ==="
echo ""

read -p "Do you want to stop all Ray jobs? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Stopping Ray jobs..."

    # Try to list and stop jobs
    ray list jobs 2>/dev/null | grep -E "RUNNING|PENDING" | awk '{print $1}' | while read job_id; do
        if [ -n "$job_id" ] && [ "$job_id" != "Job" ]; then
            echo "  Stopping job: $job_id"
            ray job stop "$job_id" 2>/dev/null || true
        fi
    done

    echo "  ✓ Stopped Ray jobs"
else
    echo "  ℹ Skipped Ray jobs cleanup"
fi

echo ""

# ============================================
# Step 4: Verify cleanup
# ============================================
echo "=== Step 4: Verification ==="
echo ""

echo "Remaining processes:"
ENGINE_COUNT=$(ps aux | grep -E "engine_server|nanoroute" | grep -v grep | wc -l)
if [ "$ENGINE_COUNT" -eq 0 ]; then
    echo "  ✓ No engine/route processes running"
else
    echo "  ⚠ Found $ENGINE_COUNT engine/route processes still running:"
    ps aux | grep -E "engine_server|nanoroute" | grep -v grep | head -5
fi

echo ""
echo "Redis keys by scope:"
for scope in demo demo2 test JimyMa; do
    count=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT KEYS "${scope}:*" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "  ⚠ $scope: $count keys remaining"
    else
        echo "  ✓ $scope: clean"
    fi
done

echo ""
AGENT_COUNT=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT KEYS "*:agent:*" 2>/dev/null | wc -l)
if [ "$AGENT_COUNT" -eq 0 ]; then
    echo "  ✓ No stale agent keys"
else
    echo "  ⚠ Found $AGENT_COUNT agent keys (might be from active sessions)"
fi

echo ""

# ============================================
# Summary
# ============================================
echo "============================================"
echo "Cleanup Complete!"
echo "============================================"
echo ""
echo "You can now start a fresh session:"
echo ""
echo "  nanoctrl start --session-id fresh \\"
echo "    --redis-url redis://$REDIS_HOST:$REDIS_PORT \\"
echo "    --ray-address http://10.102.97.179:8265 \\"
echo "    --nanoctrl-address http://10.102.97.179:3000"
echo ""
echo "  nanoctrl set --session-id fresh \\"
echo "    --model /models/qwen3-235B-Instruct-2507-FP8 \\"
echo "    --prefill-attention-dp 8 --prefill-ffn-ep 8 \\"
echo "    --decode-attention-dp 8 --decode-ffn-ep 8"
echo ""
echo "  nanoctrl spawn prefill --session-id fresh"
echo "  nanoctrl spawn decode --session-id fresh"
echo "  nanoctrl spawn route --session-id fresh"
echo ""
echo "  nanoctrl wait --session-id fresh --timeout 300"
echo ""
echo "============================================"
