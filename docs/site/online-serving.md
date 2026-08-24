# Production Serving: Ray + DLEngine PD + Router

This example shows the production serving path for GLM-5.2-FP8 with recurrent MTP: a PP8 prefill engine, an attention-DP8/EP8 HiSparse decode engine, and `dlengine-router` serving both OpenAI and Anthropic APIs.

### What each component does

| Component                | Responsibility                                                                                                                                                                         |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ray cluster**          | Manages cluster GPU resources, places DLEngine workers, and provides the distributed execution backend. Ray does not route client requests.                                            |
| **dlslime-ctrl + Redis** | Provides node/service discovery and the control plane: registration, heartbeat/liveness, scope isolation, and coordination metadata such as RDMA peers. It does not execute inference. |
| **dlengine serve**       | Runs the inference data plane. Prefill consumes prompts and produces KV state; decode pulls that state and generates tokens.                                                           |
| **dlengine-router**      | Exposes the public OpenAI/Anthropic HTTP API, discovers healthy engines through dlslime-ctrl, and orchestrates the prefill-to-decode request flow.                                     |

In short, **Ray answers “where do the GPU workers run?”**, while **dlslime-ctrl answers “which service nodes are alive and how do they find each other?”**. DLEngine performs the model computation, and the router connects clients to that compute data plane.

### Request flow

```text
OpenAI / Anthropic / Claude Code / OpenCode client
                  │
                  ▼
        dlengine-router :3001
          │             │
          │ prompt      │ decode + stream
          ▼             ▼
  prefill engine ──RDMA KV──► decode engine
          │                     │
          └── register/heartbeat┴──► dlslime-ctrl :4479 ──► Redis :16379
                  GPU workers are allocated and placed by Ray :7078
```

The concrete topology below requires 8 GPUs for prefill and 8 GPUs for decode. Adjust `pp`, `attention_dp`, and `ffn_ep` together only within combinations accepted by runtime validation.

### Prerequisites

- NVIDIA Hopper-class GPUs and RDMA connectivity for the FP8/RDMA path.
- The same DLEngine build, model checkpoint, and dependency versions on every Ray node.
- Ray, Redis, `dlslime-ctrl`, and `dlengine-router` installed.
- TCP connectivity for Ray (`7078` in this example), Redis (`16379`), dlslime-ctrl (`4479`), engine HTTP endpoints (`8101/8102`), and the router (`3001`).

### 1. Define deployment values

Run these exports in every shell that launches a DLEngine or router process:

```bash
export HEAD_IP=<ray-head-ip>
export PREFILL_IP=<prefill-coordinator-ip>
export DECODE_IP=<decode-coordinator-ip>
export MODEL_PATH=/path/to/GLM-5.2-FP8
export SERVED_MODEL=GLM-5.2-FP8

export RAY_ADDRESS="${HEAD_IP}:7078"
export CTRL_ADDRESS="http://${HEAD_IP}:4479"
```

`SERVED_MODEL` must match on both engines and in every client request. The prefill and decode coordinators may be the same host if the cluster has enough GPUs.

### 2. Start the Ray resource cluster

On the head node:

```bash
ray start --head \
  --node-ip-address "${HEAD_IP}" \
  --port 7078 \
  --dashboard-host 0.0.0.0
```

On every worker node:

```bash
ray start --address "${HEAD_IP}:7078"
```

Verify that all nodes and GPUs joined the same resource pool:

```bash
ray status --address "${RAY_ADDRESS}"
```

When either `dlengine serve` process starts, it connects to this address and asks Ray to place the requested PP/DP/EP workers across the available GPUs.

### 3. Start the discovery and control plane

Run the ordinary host deployment on the control-plane node:

```bash
redis-server \
  --daemonize yes \
  --port 16379 \
  --bind 0.0.0.0 \
  --protected-mode no

export REDIS_PUBLIC_ADDRESS="${HEAD_IP}:16379"
dlslime-ctrl start --redis-url redis://127.0.0.1:16379
```

Verify both processes:

```bash
redis-cli -h 127.0.0.1 -p 16379 ping
dlslime-ctrl status
curl -sS http://127.0.0.1:4479/
```

DLEngine nodes register their HTTP endpoint, model name, `prefill`/`decode` role, and heartbeat with dlslime-ctrl. The router watches this registry instead of relying on a static engine list. Redis stores the control-plane state; model tensors and KV cache do not pass through Redis. `REDIS_PUBLIC_ADDRESS` tells remote DLSlime PeerAgents which reachable Redis endpoint to use, while dlslime-ctrl itself connects over loopback.

`--protected-mode no` exposes Redis to the network; use it only on a trusted/private deployment network and restrict port `16379` at the firewall. For container deployment, advertised Redis addresses, and external-Redis configuration, follow the official [DLSlime Docker deployment guide](https://github.com/DeepLink-org/DLSlime/blob/main/docker/README.md).

### 4. Launch the DLEngine inference data plane

Both services connect to the same Ray cluster and control plane. `dlengine serve` starts the engine, asks Ray to place its workers, binds an HTTP endpoint, registers its role with dlslime-ctrl, and maintains a heartbeat.

On the prefill coordinator:

```bash
dlengine serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL}" \
  --host 0.0.0.0 \
  --port 8101 \
  --mode prefill \
  --ray_address "${RAY_ADDRESS}" \
  --ctrl_address "${CTRL_ADDRESS}" \
  --executor_backend dlslime \
  --pp 8 \
  --attention_dp 1 \
  --ffn_ep 1 \
  --max_model_len 1048576 \
  --max_num_batched_tokens 16384 \
  --pp_prefill_scheduler_depth 0 \
  --num_speculative_tokens 5 \
  --gpu_memory_utilization 0.75 \
  2>&1 | tee prefill_log.log
```

Here `max_num_batched_tokens=16384` is the per-stage prefill microbatch size. Scheduler depth is independent of transport in-flight depth. The default `pp_prefill_scheduler_depth=0` keeps at least one microbatch per stage admitted and, for long prompts, expands the scheduler window to at most 64 consecutive microbatches without turning the entire prompt into one GPU forward.

On the decode coordinator:

```bash
dlengine serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL}" \
  --host 0.0.0.0 \
  --port 8102 \
  --mode decode \
  --ray_address "${RAY_ADDRESS}" \
  --ctrl_address "${CTRL_ADDRESS}" \
  --executor_backend dlslime \
  --attention_dp 8 \
  --ffn_ep 8 \
  --max_model_len 1048576 \
  --max_num_batched_tokens 2048 \
  --num_speculative_tokens 5 \
  --enable_hisparse true \
  --host_utilization_per_device 64 \
  --hisparse_device_buffer_size 12288 \
  --gpu_memory_utilization 0.7 \
  --enforce_eager false \
  2>&1 | tee decode_log.log
```

For GLM recurrent MTP, both PD roles must use `num_speculative_tokens=5`. The decode hot buffer must hold at least `(5 + 1) × index_topk = 12288` entries per request. For an ordinary non-MTP PD deployment, omit `num_speculative_tokens` and size the HiSparse hot buffer for that model's non-speculative attention policy.

Check the direct engine endpoints and their service-registry entries:

```bash
curl -sS "http://${PREFILL_IP}:8101/health"
curl -sS "http://${DECODE_IP}:8102/health"

curl -sS "${CTRL_ADDRESS}/list_entities" \
  -H "Content-Type: application/json" \
  -d '{"entity_type":"service","kind":"dlengine"}'
```

The registry response should contain one `prefill` and one `decode` endpoint for `GLM-5.2-FP8`.

### 5. Start the public API gateway

The router watches dlslime-ctrl for engine registration and heartbeat changes. It may start before or after the engines; discovery is refreshed dynamically.

```bash
RUST_LOG=info dlengine-router \
  --port 3001 \
  --ctrl-address "${CTRL_ADDRESS}"
```

Verify the public gateway:

```bash
curl -sS http://127.0.0.1:3001/health
curl -sS http://127.0.0.1:3001/v1/models
```

### 6. Send an OpenAI-compatible request

```bash
curl http://127.0.0.1:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.2-FP8",
    "messages": [{"role": "user", "content": "Explain pipeline parallelism."}],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": false
  }'
```

The router selects the model pool, sends the prompt to prefill, passes the returned migration metadata to decode, streams or returns the decode result, and releases the prefill-side KV allocation.

### 7. Use Claude Code through the Anthropic API

`dlengine-router` exposes `POST /v1/messages` and `POST /v1/messages/count_tokens`, so Claude Code can use the same PD deployment without an extra protocol adapter:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:3001 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_MODEL=GLM-5.2-FP8 \
ANTHROPIC_SMALL_FAST_MODEL=GLM-5.2-FP8 \
claude --model GLM-5.2-FP8
```

Run the command from the project directory that Claude Code should work on. `ANTHROPIC_API_KEY=dummy` is a placeholder for this local gateway; add real authentication at the network or gateway layer before exposing the endpoint outside a trusted environment.

### 8. Use OpenCode through the OpenAI-compatible API

OpenCode can connect directly to the router’s OpenAI-compatible `/v1` endpoint. Add a provider to `~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "dlengine/GLM-5.2-FP8",
  "provider": {
    "dlengine": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local DLEngine",
      "options": {
        "baseURL": "http://127.0.0.1:3001/v1"
      },
      "models": {
        "GLM-5.2-FP8": {
          "name": "GLM-5.2-FP8"
        }
      }
    }
  }
}
```

Run OpenCode from the project directory:

```bash
opencode
```

The top-level `model` selects the configured model by default. To override it for one invocation, use `opencode -m dlengine/GLM-5.2-FP8`. The provider model name must match the `--served-model-name` registered by both DLEngine roles. Keep the router on loopback or add authentication before exposing it outside a trusted network.

### Operational checks

- **Ray has insufficient GPUs:** run `ray status --address "${RAY_ADDRESS}"` and confirm the total resources match the requested PP/DP/EP topology.
- **Router reports model not found:** verify that both engines use exactly the same `--served-model-name`, `--ctrl_address`, and optional `--ctrl_scope`.
- **Router has only one PD role:** query `/list_entities` and inspect the registered `metadata.role` and heartbeat.
- **Engine is healthy but unreachable through the router:** ensure the advertised coordinator IP and ports `8101/8102` are reachable from the router host.
- **KV migration fails:** verify RDMA/NIC configuration and that prefill and decode use the same dlslime-ctrl/Redis service.
- **Multiple jobs share one control plane:** give both engines the same `--ctrl_scope <job-name>` and start the router with `--ctrl-scope <job-name>`.
