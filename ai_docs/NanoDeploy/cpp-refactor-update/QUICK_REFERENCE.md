# Quick Reference Guide - C++ Refactor Update

## Table of Contents

- [Common Commands](#common-commands)
- [Configuration Examples](#configuration-examples)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)
- [API Examples](#api-examples)

## Common Commands

### Building Components

```bash
# Build NanoSequence (C++ library)
cd NanoSequence
mkdir -p build && cd build
cmake .. && make -j$(nproc)
cd ../..

# Build NanoCtrl (Rust)
cd NanoCtrl
cargo build --release
cd ..

# Build NanoRoute (Rust)
cd NanoRoute
cargo build --release
cd ..

# Install NanoDeploy (Python + C++)
cd NanoDeploy
pip install -e .
cd ..
```

### Starting Services

```bash
# 1. Redis (required for NanoCtrl)
redis-server --bind 0.0.0.0 --port 6379 --protected-mode no

# 2. NanoCtrl (service discovery)
./NanoCtrl/target/release/nanoctrl \
  --host 0.0.0.0 \
  --port 8080 \
  --redis-url redis://127.0.0.1:6379 \
  --ttl 60

# 3. Ray Head Node
ray start --head --port=6379 --dashboard-host=0.0.0.0

# 4. Prefill Engine
python -m nanodeploy.server.engine_server \
  --config config.yaml \
  --mode prefill \
  --host 0.0.0.0 \
  --port 6001 \
  --nanoctrl_address 127.0.0.1:8080

# 5. Decode Engine
python -m nanodeploy.server.engine_server \
  --config config.yaml \
  --mode decode \
  --host 0.0.0.0 \
  --port 6002 \
  --nanoctrl_address 127.0.0.1:8080

# 6. NanoRoute (load balancer)
./NanoRoute/target/release/nanoroute \
  --host 0.0.0.0 \
  --port 38080 \
  --nanoctrl-url http://127.0.0.1:8080 \
  --routing-strategy round-robin
```

### Checking Status

```bash
# Check Redis
redis-cli ping

# Check NanoCtrl
curl http://localhost:8080/list_engines

# Check specific role engines
curl http://localhost:8080/get_engine/prefill
curl http://localhost:8080/get_engine/decode

# Check Ray cluster
ray status

# Check NanoRoute health
curl http://localhost:38080/health
```

### Stopping Services

```bash
# Stop engines (Ctrl+C in terminal)

# Stop Ray
ray stop

# Stop NanoCtrl/NanoRoute (Ctrl+C)

# Stop Redis
redis-cli shutdown
```

## Configuration Examples

### Single Node (Development)

**Config File** (`config_dev.yaml`):

```yaml
# Model
model: "/path/to/model"
max_model_len: 8192
max_num_batched_tokens: 8192

# Scheduler
max_num_seqs: 128
loop_count: 16

# KV Cache
kvcache_block_size: 256
num_kvcache_blocks: 8000
gpu_memory_utilization: 0.9

# Parallelism (4 GPUs)
attention_dp: 2
attention_sp: 1
attention_tp: 2
ffn_dp: 2
ffn_ep: 1
ffn_tp: 2

# Network
host: "0.0.0.0"
port: 6001
ray_address: "127.0.0.1:6379"
nanoctrl_address: "127.0.0.1:8080"

# Logging
log_level: "INFO"
```

**Launch Commands**:

```bash
# Terminal 1: Control Plane
redis-server --port 6379 &
./NanoCtrl/target/release/nanoctrl --port 8080 &

# Terminal 2: Ray
ray start --head

# Terminal 3: Prefill Engine
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m nanodeploy.server.engine_server \
  --config config_dev.yaml --mode prefill --port 6001

# Terminal 4: Decode Engine
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m nanodeploy.server.engine_server \
  --config config_dev.yaml --mode decode --port 6002

# Terminal 5: Load Balancer
./NanoRoute/target/release/nanoroute --port 38080
```

### Multi-Node (Production)

**Config File** (`config_prod.yaml`):

```yaml
# Model
model: "/path/to/large-model"
max_model_len: 16384
max_num_batched_tokens: 16384

# Scheduler
max_num_seqs: 256
loop_count: 32

# KV Cache
kvcache_block_size: 256
num_kvcache_blocks: 15000
gpu_memory_utilization: 0.9

# Parallelism (8 GPUs per node)
attention_dp: 2
attention_sp: 2
attention_tp: 2
ffn_dp: 2
ffn_ep: 2
ffn_tp: 2

# Network (will be overridden by --host)
port: 6001
ray_address: "10.1.16.1:6379"
nanoctrl_address: "10.1.16.1:8080"

# Logging
log_level: "WARNING"
```

**Node 1 (Control Node - 10.1.16.1)**:

```bash
# Redis
redis-server --bind 0.0.0.0 --port 6379 --protected-mode no

# NanoCtrl
./NanoCtrl/target/release/nanoctrl \
  --host 0.0.0.0 --port 8080 \
  --redis-url redis://127.0.0.1:6379

# Ray Head
ray start --head --node-ip-address=10.1.16.1 \
  --port=6379 --dashboard-host=0.0.0.0

# NanoRoute
./NanoRoute/target/release/nanoroute \
  --host 0.0.0.0 --port 38080 \
  --nanoctrl-url http://10.1.16.1:8080
```

**Node 2 (Prefill - 10.1.16.4)**:

```bash
# Join Ray
ray start --address=10.1.16.1:6379

# Prefill Engine
python -m nanodeploy.server.engine_server \
  --config config_prod.yaml \
  --mode prefill \
  --host 10.1.16.4 \
  --port 6001 \
  --nanoctrl_address 10.1.16.1:8080 \
  --ray_address 10.1.16.1:6379
```

**Node 3 (Decode - 10.1.16.5)**:

```bash
# Join Ray
ray start --address=10.1.16.1:6379

# Decode Engine
python -m nanodeploy.server.engine_server \
  --config config_prod.yaml \
  --mode decode \
  --host 10.1.16.5 \
  --port 6002 \
  --nanoctrl_address 10.1.16.1:8080 \
  --ray_address 10.1.16.1:6379
```

**Node 4 (Decode - 10.1.16.6)**:

```bash
# Join Ray
ray start --address=10.1.16.1:6379

# Decode Engine
python -m nanodeploy.server.engine_server \
  --config config_prod.yaml \
  --mode decode \
  --host 10.1.16.6 \
  --port 6003 \
  --nanoctrl_address 10.1.16.1:8080 \
  --ray_address 10.1.16.1:6379
```

## Troubleshooting

### Engine Not Registering

**Symptom**: Engine starts but doesn't appear in `/list_engines`

**Diagnosis**:

```bash
# 1. Check NanoCtrl is running
curl http://<nanoctrl-host>:8080/list_engines

# 2. Check network connectivity
ping <nanoctrl-host>
telnet <nanoctrl-host> 8080

# 3. Check engine logs
# Look for "Successfully registered with NanoCtrl" or error messages
```

**Solutions**:

- Ensure `nanoctrl_address` matches actual NanoCtrl host:port
- Check firewall rules: `sudo ufw allow 8080/tcp`
- Verify Redis is running: `redis-cli ping`
- Check NanoCtrl logs for registration attempts

### ZMQ Connection Refused

**Symptom**: "Connection refused" errors, requests not reaching engines

**Diagnosis**:

```bash
# 1. Check engine is listening
netstat -tlnp | grep <engine-port>

# 2. Test ZMQ connectivity
nc -zv <engine-host> <engine-port>

# 3. Check engine startup logs
# Look for "ZMQ Connect: tcp://..." line
```

**Solutions**:

**For Localhost Mode**:

```bash
# Always use --host 0.0.0.0 (will auto-use 127.0.0.1 for ZMQ)
python -m nanodeploy.server.engine_server \
  --host 0.0.0.0 \
  --port 6001
```

**For Distributed Mode**:

```bash
# Use actual node IP
python -m nanodeploy.server.engine_server \
  --host 10.1.16.4 \
  --port 6001
```

**Firewall**:

```bash
# Allow engine ports
sudo ufw allow 6001:6010/tcp
```

### Segmentation Faults

**Symptom**: Engine crashes with `SIGSEGV` during operation

**Diagnosis**:

```bash
# Check GPU memory
nvidia-smi

# Run with debug symbols
gdb --args python -m nanodeploy.server.engine_server ...
```

**Solutions**:

1. **Update to latest version** (includes BlockContext fixes)
2. **Reduce memory usage**:
   ```yaml
   gpu_memory_utilization: 0.8  # Reduce from 0.9
   num_kvcache_blocks: 10000    # Reduce from 15000
   ```
3. **Check CUDA version compatibility**:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   nvcc --version
   ```

### Request Hanging

**Symptom**: Curl request hangs indefinitely, no response

**Diagnosis**:

```bash
# 1. Check engine status
curl http://<nanoctrl>:8080/list_engines

# 2. Check NanoRoute logs (look for ZMQ errors)

# 3. Check engine logs (look for request reception)

# 4. Test direct engine connection
# (requires ZMQ client)
```

**Solutions**:

- Ensure both prefill and decode engines are running
- Check NanoRoute has discovered engines (see logs)
- Verify engines are handling `RUNNING_PREFILL` and `RUNNING_DECODE` statuses
- Check for deadlocks in engine logs

### High Latency

**Symptom**: Time to First Token (TTFT) > 1 second

**Diagnosis**:

```bash
# Check network latency
ping <decode-engine-host>

# Check RDMA status (if using)
ibstatus

# Check GPU utilization
nvidia-smi dmon -s u
```

**Solutions**:

1. **Enable RDMA for multi-node** (if not already enabled)
2. **Reduce batch size**:
   ```yaml
   max_num_batched_tokens: 8192  # Reduce from 16384
   ```
3. **Check network bandwidth**:
   ```bash
   iperf3 -s  # On one node
   iperf3 -c <server-ip>  # On another node
   ```
4. **Use fewer decode engines** (reduce routing overhead)

### Memory Leak

**Symptom**: GPU memory increasing over time

**Diagnosis**:

```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Check for unreleased sequences in logs
```

**Solutions**:

1. **Restart engines periodically** (temporary workaround)
2. **Check for stuck sequences**:
   ```python
   # Add monitoring
   logger.info(f"Running sequences: {len(scheduler.running_queue)}")
   ```
3. **Ensure proper sequence cleanup** (check `is_finished()` logic)

## Performance Tuning

### Latency Optimization

**Goal**: Minimize Time to First Token (TTFT)

**Configuration**:

```yaml
# Reduce batch size
max_num_batched_tokens: 4096  # Lower = faster prefill

# Reduce concurrent sequences
max_num_seqs: 64  # Lower = faster scheduling

# Increase scheduling frequency
loop_count: 8  # Lower = more frequent scheduling
```

**Deployment**:

- Use fewer decode engines (1-2 is optimal for latency)
- Colocate prefill and decode engines (same node) if possible
- Enable RDMA for multi-node

### Throughput Optimization

**Goal**: Maximize tokens/second

**Configuration**:

```yaml
# Increase batch size
max_num_batched_tokens: 32768  # Higher = more throughput

# Increase concurrent sequences
max_num_seqs: 512  # Higher = more batching

# Reduce scheduling frequency
loop_count: 64  # Higher = less overhead
```

**Deployment**:

- Use more decode engines (4-8 for high throughput)
- Increase data parallelism (higher `attention_dp`, `ffn_dp`)
- Use load balancing strategy: `least-batch`

### Memory Optimization

**Goal**: Minimize GPU memory usage

**Configuration**:

```yaml
# Reduce KV cache blocks
num_kvcache_blocks: 8000  # Lower = less memory

# Use sequence parallelism
attention_sp: 4  # Higher = split memory across GPUs

# Reduce memory utilization
gpu_memory_utilization: 0.8  # Lower = more headroom
```

**Deployment**:

- Use more GPUs with smaller memory footprint per GPU
- Enable expert parallelism for MoE models (`ffn_ep > 1`)

### Network Optimization (Multi-Node)

**Goal**: Minimize KV cache migration latency

**RDMA Configuration**:

```bash
# Check RDMA device
ibv_devices

# Check RDMA bandwidth
ib_send_bw -d mlx5_0 -g 0  # Server
ib_send_bw -d mlx5_0 -g 0 <server-ip>  # Client
```

**DLSlime Configuration**:

```python
# Increase QP count for higher bandwidth
SLIME_QP_NUM=8 python -m nanodeploy.server.engine_server ...

# Enable GPUDirect RDMA
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## API Examples

### Non-Streaming Completion

```bash
curl -X POST http://localhost:38080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen",
    "prompt": "Explain quantum computing in simple terms.",
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  }'
```

**Response**:

```json
{
  "id": "req-1234567890",
  "object": "text_completion",
  "created": 1234567890,
  "model": "Qwen",
  "choices": [{
    "index": 0,
    "text": " Quantum computing uses quantum mechanics principles...",
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 120,
    "total_tokens": 128
  }
}
```

### Streaming Completion

```bash
curl -X POST http://localhost:38080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen",
    "prompt": "Write a haiku about AI.",
    "max_tokens": 64,
    "stream": true
  }'
```

**Response** (Server-Sent Events):

```
data: {"id":"req-1234567890","choices":[{"index":0,"text":"Silicon","finish_reason":null}]}

data: {"id":"req-1234567890","choices":[{"index":0,"text":" dreams","finish_reason":null}]}

data: {"id":"req-1234567890","choices":[{"index":0,"text":" awake","finish_reason":null}]}

data: {"id":"req-1234567890","choices":[{"index":0,"text":"\n","finish_reason":null}]}

data: {"id":"req-1234567890","choices":[{"index":0,"text":"Processing","finish_reason":null}]}

...

data: {"id":"req-1234567890","choices":[{"index":0,"text":"","finish_reason":"stop"}]}

data: [DONE]
```

### Python Client Example

```python
import requests

url = "http://localhost:38080/v1/completions"
headers = {"Content-Type": "application/json"}
payload = {
    "model": "Qwen",
    "prompt": "Hello, world!",
    "max_tokens": 64,
    "temperature": 0.7
}

# Non-streaming
response = requests.post(url, json=payload, headers=headers)
print(response.json()["choices"][0]["text"])

# Streaming
payload["stream"] = True
response = requests.post(url, json=payload, headers=headers, stream=True)
for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith("data: "):
            data = line[6:]
            if data != "[DONE]":
                import json
                chunk = json.loads(data)
                text = chunk["choices"][0]["text"]
                print(text, end="", flush=True)
print()
```

### Using OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:38080/v1",
    api_key="dummy"  # Not used but required
)

# Non-streaming
completion = client.completions.create(
    model="Qwen",
    prompt="Hello, world!",
    max_tokens=64,
    temperature=0.7
)
print(completion.choices[0].text)

# Streaming
stream = client.completions.create(
    model="Qwen",
    prompt="Hello, world!",
    max_tokens=64,
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].text, end="", flush=True)
print()
```

## Monitoring Commands

### System Health

```bash
# Check all services
ps aux | grep -E 'redis|nanoctrl|nanoroute|engine_server|ray'

# Check GPU utilization
nvidia-smi dmon -s u -c 10

# Check network traffic
iftop -i eth0  # Or your network interface

# Check disk I/O
iostat -x 1
```

### Performance Metrics

```bash
# NanoRoute metrics (future feature)
curl http://localhost:38080/metrics

# Engine-level metrics
# (Check engine logs for throughput/latency stats)

# Ray dashboard
# Open http://<ray-head>:8265 in browser
```

### Log Analysis

```bash
# Engine logs
tail -f /path/to/engine.log | grep -E 'TTFT|throughput|latency'

# NanoRoute logs
tail -f /path/to/nanoroute.log | grep -E 'request|response|error'

# NanoCtrl logs
tail -f /path/to/nanoctrl.log | grep -E 'register|heartbeat'
```

______________________________________________________________________

**Quick Reference Version**: 1.0
**Last Updated**: February 2026
**For More Details**: See [README.md](./README.md) and [ARCHITECTURE.md](./ARCHITECTURE.md)
