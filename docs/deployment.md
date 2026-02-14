# Deployment Guide

This guide covers deploying NanoInfra with disaggregated prefill/decode using **NanoCtrl** (service discovery), **NanoRoute** (load balancer), and **NanoDeploy** (inference engine).

## Prerequisites

- NVIDIA GPUs with CUDA 12.1+ (SM90+ for FlashMLA / Flash Attention 3)
- RDMA-capable NICs (for multi-node KV cache migration)
- Python 3.10+, Ray, Redis, Rust 1.70+

```bash
pip install torch transformers accelerate
pip install ray redis zmq flatbuffers httpx pydantic jsonargparse
```

## Step 1: Start Redis & NanoCtrl

```bash
redis-server --bind 0.0.0.0 --port 6379 --protected-mode no

cd NanoCtrl && ./target/release/nanoctrl \
  --host 0.0.0.0 --port 8080 --redis-url redis://127.0.0.1:6379 --ttl 60
```

## Step 2: Start Ray Cluster

```bash
# Head node
ray start --head --port=6379 --dashboard-host=0.0.0.0

# Worker nodes
ray start --address=<head-node-ip>:6379
```

## Step 3: Start Engine Servers

```bash
# Prefill engine (Node A)
python -m nanodeploy.server.engine_server \
  --config engine_config.yaml \
  --mode prefill --host 10.1.16.4 --port 6001 \
  --nanoctrl_address 10.1.16.1:8080 --ray_address 10.1.16.1:6379

# Decode engine (Node B)
python -m nanodeploy.server.engine_server \
  --config engine_config.yaml \
  --mode decode --host 10.1.16.5 --port 6002 \
  --nanoctrl_address 10.1.16.1:8080 --ray_address 10.1.16.1:6379
```

## Step 4: Start NanoRoute

```bash
cd NanoRoute && ./target/release/nanoroute \
  --host 0.0.0.0 --port 38080 \
  --nanoctrl-url http://10.1.16.1:8080 \
  --routing-strategy round-robin
```

## Step 5: Send Requests

```bash
curl -X POST http://localhost:38080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-v3", "prompt": "Hello!", "max_tokens": 64}'
```
