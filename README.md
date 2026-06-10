# DLEngine: LLM Inference with Prefill-Decode Disaggregation and Wide Expert Parallelism

## 📦 Components

| Component                            | Language   | Description          | Key Features                                                                                    |
| ------------------------------------ | ---------- | -------------------- | ----------------------------------------------------------------------------------------------- |
| [dlengine](./dlengine)               | Python/C++ | LLM inference engine | Prefill/decode engines, KV cache management, continuous batching, Ray-based distributed workers |
| [dlengine-router](./dlengine-router) | Rust       | HTTP load balancer   | OpenAI-compatible API, tool calls, routing strategies, engine discovery                         |

## 🧠 Supported Models

| Model         | Component   | Architecture          |
| ------------- | ----------- | --------------------- |
| DeepSeek-V3   | dlengine    | MLA + MoE             |
| DeepSeek-V3.2 | dlengine    | MLA + MoE + NSA       |
| DeepSeek-V4   | dlengine    | MLA + MoE + DSA + SWA |
| GLM-5         | dlengine    | MLA + MoE + NSA       |
| Kimi-K2       | dlengine    | MLA + MoE             |
| Qwen3         | dlengine    | GQA (Dense)           |
| Qwen3-MoE     | dlengine    | GQA + MoE             |
| Qwen3.5-MoE   | dlengine    | GQA + GDN + MoE       |
| Qwen3-VL      | dlengine.vl | GQA + MoE + ViT       |

## ✨ Key Features

| Feature                                            | Description                                                                                 |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| ✅ **Chunked Prefill**                             | Split long prompts into chunks to overlap with decode batches.                              |
| ✅ **Continuous Batching**                         | Dynamic request scheduling with paged KV cache.                                             |
| ✅ **CUDA Graph**                                  | Captured decode kernels for low-latency token generation.                                   |
| ✅ **Encoder-Prefill-Decode (EPD) Disaggregation** | Separate encoder, prefill and decode across GPU nodes with GPUDirect RDMA KV migration.     |
| ✅ **FP8 KV Cache**                                | Float8 (E4M3) paged KV cache, ~50% memory reduction.                                        |
| ✅ **Gated Delta Net (GDN)**                       | Linear attention for Qwen3.5-MoE hybrid full/linear layers.                                 |
| ✅ **Multi-head Latent Attention (MLA)**           | Compressed KV cache with low-rank projection for DeepSeek-V3 family.                        |
| ✅ **Multi-Token Prediction (MTP)**                | Speculative decoding with model-native MTP heads.                                           |
| ✅ **Native Sparse Attention (NSA)**               | FP8 sparse decode with block-level indexing for DeepSeek-V3.2.                              |
| ✅ **Node Discovery**                              | Automatic engine registration and heartbeat via the DLSlime control plane (`dlslime-ctrl`). |
| ✅ **Prefix Caching**                              | Reuse KV cache of shared prompt prefixes across requests.                                   |
| ✅ **Tensor Parallelism (TP)**                     | Split weight matrices across GPUs for large model inference.                                |
| ✅ **Wide Expert Parallelism**                     | MoE EP across all GPUs with attention data-parallel (`attention_dp × ffn_ep`).              |

## 🏗️ Architecture

```mermaid
graph TB
    Client[Client Layer<br/>HTTP Requests / OpenAI SDK]
    Route[dlengine-router<br/>Rust/HTTP<br/>Load Balancer]
    VL[dlengine.vl<br/>Vision Encoder]
    Prefill[Prefill Engine<br/>Python/C++]
    Decode[Decode Engine<br/>Python/C++]
    Ctrl[dlslime-ctrl<br/>Redis<br/>Service Registry<br/>from DLSlime]

    Client -->|HTTP| Route
    Route -->|ZMQ| VL
    Route -->|ZMQ| Prefill
    Route -->|ZMQ| Decode
    VL -->|RDMA<br/>Embeddings| Prefill
    Prefill -->|RDMA<br/>KV Migration| Decode
    VL -->|Register/Heartbeat| Ctrl
    Prefill -->|Register/Heartbeat| Ctrl
    Decode -->|Register/Heartbeat| Ctrl
    Route -->|Engine Discovery| Ctrl
```

## 🚀 Installation

### Docker Development Image

A prebuilt CUDA 12.8 development image bundles all build dependencies (PyTorch,
DeepEP/DeepGEMM/FlashMLA/FlashInfer, flash-attn, DLSlime, Rust toolchain), plus an
optional variant with **3FS USRBIO** support. See **[`docker/README.md`](./docker/README.md)**
for the pinned dependency versions, build/run commands, and the 3FS image.

The DeepSeek kernels require SM90+ (NVIDIA Hopper) GPUs. To install the key dependencies
manually instead of using the image:

```bash
cd DeepEP && pip install .
cd DeepGEMM && pip install .
cd FlashMLA && pip install .
pip install flashinfer-python==0.6.9
pip install dlslime==0.1.16
```

### One-liner: install everything

```bash
pip install ".[all]"
```

### Install individual components

```bash
pip install ".[dlengine]"   # DLEngine inference engine only
pip install ".[dlenginevl]" # DLEngine + vision-language extras (dlengine.vl subpackage)
```

> The control-plane server (`dlslime-ctrl`) and its Python client (`dlslime.ctrl.NanoCtrlClient`)
> now live in the [DLSlime](https://github.com/DeepLink-org/DLSlime) repo. Install them via:
>
> ```bash
> pip install dlslime           # PeerAgent + NanoCtrlClient (data-plane wheel)
> pip install dlslime-ctrl      # Rust control-plane server binary
> # or, from a DLSlime checkout: pip install -e ./dlslime ./dlslime-ctrl
> ```

### For developers

```bash
# Build DLEngine C++ extensions in-place (GPU kernels ship in dlengine.kernel)
cd dlengine && pip install -e . && cd ..

# Build dlengine-router (Rust)
cd dlengine-router && cargo build --release && cd ..

# Build dlslime-ctrl (Rust) from the DLSlime checkout
cd /path/to/DLSlime/dlslime-ctrl && cargo build --release && cd -
```

## Quick Start: LLM Inference

Prefill-Decode disaggregation splits prompt processing (prefill) and token generation (decode) across separate GPU nodes connected via RDMA.

### Prerequisites

- 2 nodes with NVIDIA GPUs (SM90+ for FP8), RDMA-capable NICs
- Redis, Ray cluster, Rust toolchain

#### 1. Start Ray

```bash
# Node 0 (head)
ray start --head --port=7078 --dashboard-host=0.0.0.0

# Node 1 (multi-node only)
ray start --address <node0-ip>:7078
```

### Offline mode

Batch generation without HTTP serving.

#### Single node (no dlslime-ctrl needed)

```bash
python dlengine/examples/non_disagg.py \
    --model /models/Qwen3-235B-A22B \
    --ray_address <node0-ip>:7078 \
    --master_address <node0-ip>:6006 \
    --attention_dp 8 --ffn_ep 8 \
    --kvcache_block_size 256 \
    --prompt "1+1=?" --max_tokens 128
```

#### PD disaggregated (2 nodes)

##### 2. Start Redis + dlslime-ctrl

```bash
redis-server --bind 0.0.0.0 --port 6379
dlslime-ctrl server --redis-url redis://127.0.0.1:6379
```

##### 3. Launch engines

```bash
python dlengine/examples/disagg.py \
    --model /models/Qwen3-235B-A22B \
    --ray_address <node0-ip>:7078 \
    --ctrl_address <node0-ip>:4479 \
    --attention_dp 8 --ffn_ep 8 \
    --prefill.master_address <node0-ip>:6006 \
    --decode.master_address <node1-ip>:6006
```

### Single-node serving (`dlengine serve`)

For single-node hybrid deployment (prefill + decode in one process), use the
`dlengine serve` command. It runs the engine in-process and exposes an
OpenAI-compatible HTTP API directly, in the spirit of `vllm serve` — no
dlengine-router and no ZMQ engine servers required:

```bash
# Same Config flags as engine_server.py (--host/--port bind HTTP for serve)
dlengine serve /path/to/model \
  --host 0.0.0.0 --port 8100 \
  --served-model-name Qwen3-4B \
  --ray_address 127.0.0.1:7078
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/completions`,
`POST /v1/chat/completions` (streaming and non-streaming).

```bash
curl http://127.0.0.1:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B", "messages": [{"role": "user", "content": "Hello"}]}'
```

To make the node discoverable by a router (e.g. DLRouter) via dlslime-ctrl,
point it at a running control plane; the server then registers its HTTP
endpoint (entity kind `dlengine`) and keeps a heartbeat:

```bash
# Control plane (Redis + dlslime-ctrl)
redis-server --bind 0.0.0.0 --port 6379 &
dlslime-ctrl server --redis-url redis://127.0.0.1:6379 &

dlengine serve /path/to/model \
  --host 0.0.0.0 --port 8100 \
  --served-model-name Qwen3-4B \
  --ctrl-address 127.0.0.1:4479
```

#### PD disaggregation with `dlengine serve` + DLRouter

`dlengine serve` also supports Prefill-Decode disaggregation over HTTP.
Launch one engine with `--mode prefill` and another with `--mode decode`
(both pointed at the same dlslime-ctrl). They register their role with the
control plane and connect their PeerAgents for KV migration. A PD-aware
gateway such as [DLRouter](https://github.com/DeepLink-org/DLRouter) then
orchestrates the two-stage handoff: it asks the prefill node to process the
prompt and return an opaque KV migration payload (via `kv_transfer_params`),
hands that payload to the decode node, which RDMA-pulls the KV cache and
streams the completion, and finally releases the prefill-side KV blocks
(`POST /pd/free`).

```bash
# Control plane (Redis + dlslime-ctrl)
redis-server --bind 0.0.0.0 --port 6379 &
dlslime-ctrl server --redis-url redis://127.0.0.1:6379 &

# Prefill node
dlengine serve /path/to/model \
  --host 0.0.0.0 --port 8101 \
  --served-model-name Qwen3-4B \
  --mode prefill \
  --ctrl-address 127.0.0.1:4479 \
  --ray_address 127.0.0.1:7078

# Decode node
dlengine serve /path/to/model \
  --host 0.0.0.0 --port 8102 \
  --served-model-name Qwen3-4B \
  --mode decode \
  --ctrl-address 127.0.0.1:4479 \
  --ray_address 127.0.0.1:7078
```

DLRouter discovers both nodes (entity kind `dlengine`) via dlslime-ctrl,
maps their roles to `PREFILL`/`DECODE`, and serves an OpenAI-compatible API.
Point clients at DLRouter; requests transparently flow prefill → KV
migration → decode. When the prefill node fully answers a request on its own
(e.g. the first sampled token is EOS), the completion is returned directly
without a decode handoff. Prefill and decode engines may co-locate on the
same node when GPU resources allow.

### Online mode

ZMQ engine servers with OpenAI-compatible HTTP API via dlengine-router.

##### 2. Start Redis + dlslime-ctrl

```bash
redis-server --bind 0.0.0.0 --port 6379
dlslime-ctrl server --redis-url redis://127.0.0.1:6379
```

##### 3. Start dlengine-router

```bash
cd dlengine-router && cargo run --release    # edit config.toml to set ctrl_address
```

##### 4. Launch engines

```bash
# Terminal 1 — Decode engine
python dlengine/dlengine/server/engine_server.py \
    --model /models/Qwen3-235B-A22B \
    --mode decode \
    --ray_address <node0-ip>:7078 \
    --ctrl_address <node0-ip>:4479 \
    --ctrl_scope nanoctrl-0 \
    --master_address <node1-ip>:6006 \
    --host <node0-ip> --port 6001 \
    --attention_dp 8 --ffn_ep 8 \
    --kvcache_block_size 64 \
    --max_num_batched_tokens 16384 --max_model_len 16384

# Terminal 2 — Prefill engine
python dlengine/dlengine/server/engine_server.py \
    --model /models/Qwen3-235B-A22B \
    --mode prefill \
    --ray_address <node0-ip>:7078 \
    --ctrl_address <node0-ip>:4479 \
    --ctrl_scope nanoctrl-0 \
    --master_address <node0-ip>:6006 \
    --host <node0-ip> --port 6002 \
    --attention_dp 8 --ffn_ep 8 \
    --kvcache_block_size 64 \
    --max_num_batched_tokens 16384 --max_model_len 16384
```

##### 5. Send requests

```bash
curl http://<node0-ip>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/models/Qwen3-235B-A22B", "messages": [{"role": "user", "content": "Hello"}]}'
```

______________________________________________________________________

## 📄 License

See individual component [license](./LICENSE).

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/JimyMa/NanoDeploy/issues)
- **Documentation**: Check component READMEs
