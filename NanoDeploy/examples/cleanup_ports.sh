#!/bin/bash

# Port ranges used by RDMA tests
PREFILL_PORTS=(10000 10001 10002 10003 10004 10005 10006 10007)
DECODE_PORTS=(11000 11001 11002 11003 11004 11005 11006 11007)

echo "Cleaning up RDMA test ports..."

for port in "${PREFILL_PORTS[@]}" "${DECODE_PORTS[@]}"; do
    pid=$(lsof -t -i :$port)
    if [ ! -z "$pid" ]; then
        echo "Killing process $pid using port $port"
        kill -9 $pid 2>/dev/null
    fi
done

# Also kill any leftover ray workers if they are stuck
# (Caution: this might kill other ray workers on the machine)
# pkill -9 -f "ray::TestWorker"

echo "Done."
