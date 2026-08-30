# Transport-neutral P/D deployment

DLEngine can migrate cache state between separate prefill and decode services through DLSlime. The same deployment flow supports automatic transport selection: compatible CUDA Fabric placements use NVLink, while deployments with no Fabric placement metadata on either role continue to use RDMA.

## Components and responsibilities

| Component                        | Responsibility                                                                                                           |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Ray                              | Places and starts DLEngine workers. It is not the inference data plane.                                                  |
| Redis and `dlslime-ctrl`         | Store service discovery, topology, heartbeat, and memory-region metadata. They do not carry cache tensors.               |
| DLEngine prefill/decode services | Allocate caches, publish worker placement, execute inference, and request cache migration.                               |
| DLSlime PeerAgent                | Discovers topology, owns exportable Fabric allocations, selects a compatible transport, and performs named-region reads. |
| `dlengine-router`                | Discovers HTTP services and pairs only transport-compatible prefill and decode roles.                                    |

The control plane may be shared by multiple deployments when each deployment uses a distinct `--ctrl_scope`.

## Prerequisites

- Every worker can read the same model checkpoint path.
- Ray workers can reach the Ray head and control-plane endpoints.
- Redis and `dlslime-ctrl` are reachable through addresses advertised to every worker.
- The installed DLSlime build provides topology resources, `transport="auto"`, PeerAgent-owned Fabric allocation, and named-region I/O.
- For Fabric transport, each worker exposes one membership-ready GPU and a usable IMEX channel. The participating prefill and decode workers belong to a common Fabric domain and share an accessible IMEX channel.
- For RDMA transport, both roles omit Fabric placement and expose compatible RDMA resources.

Do not mix a Fabric-advertising role with a role that publishes no Fabric placement. The router deliberately treats one-sided metadata as incompatible.

## Start the control plane

Define portable deployment values in each coordinator shell:

```bash
export MODEL_PATH="/path/to/model"
export SERVED_MODEL="model-alias"
export RAY_ADDRESS="ray://ray-head.example:10001"
export CTRL_ADDRESS="http://control.example:4479"
export CTRL_SCOPE="deployment-name"
```

Start Redis and `dlslime-ctrl` on the control-plane host. Use deployment-specific authentication and network policy in production.

```bash
export REDIS_PUBLIC_ADDRESS="control.example:16379"
dlslime-ctrl start --redis-url redis://127.0.0.1:16379

dlslime-ctrl status
curl -sS http://127.0.0.1:4479/
```

Start or join the Ray cluster before launching either inference role. Confirm that the expected workers are alive with `ray status`.

## Launch prefill and decode

Choose PP, attention DP, and FFN EP values appropriate for the model and available worker count. The two roles may use different parallel layouts only when the model's cache-migration contract supports that layout.

Prefill coordinator:

```bash
dlengine serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL}" \
  --host 0.0.0.0 \
  --port 8101 \
  --mode prefill \
  --ray_address "${RAY_ADDRESS}" \
  --ctrl_address "${CTRL_ADDRESS}" \
  --ctrl_scope "${CTRL_SCOPE}" \
  --executor_backend dlslime \
  --pp "${PREFILL_PP}" \
  --attention_dp "${PREFILL_DP}" \
  --ffn_ep "${PREFILL_EP}"
```

Decode coordinator:

```bash
dlengine serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL}" \
  --host 0.0.0.0 \
  --port 8102 \
  --mode decode \
  --ray_address "${RAY_ADDRESS}" \
  --ctrl_address "${CTRL_ADDRESS}" \
  --ctrl_scope "${CTRL_SCOPE}" \
  --executor_backend dlslime \
  --attention_dp "${DECODE_DP}" \
  --ffn_ep "${DECODE_EP}"
```

When Fabric is ready, DLEngine asks the PeerAgent to allocate KV and Indexer storage from exportable CUDA Fabric memory. Otherwise it retains ordinary device allocation. Cache regions use stable names such as `kv_cache` and `indexer_cache`; transfer submission validates local and remote offsets against the published region lengths.

## Placement publication and routing

Each engine service publishes a resource with this shape:

```json
{
  "schema_version": 1,
  "placements": [
    {
      "rank": 0,
      "peer_agent_id": "engine-id:0",
      "gpu_uuid": "GPU-UUID",
      "cluster_uuid": "fabric-cluster-uuid",
      "clique_id": 0,
      "fabric_domain_id": "fabric-cluster-uuid:0",
      "topology_epoch": 1,
      "membership_ready": true,
      "imex_channel_ids": [0]
    }
  ]
}
```

An engine either publishes one valid placement per worker or an empty placement list. Partial publication and multiple domains inside one engine fail registration. The router applies the following compatibility rule:

| Prefill placement | Decode placement                      | Result                                                   |
| ----------------- | ------------------------------------- | -------------------------------------------------------- |
| Both empty        | Both empty                            | Pair through the existing RDMA path.                     |
| Non-empty         | Common Fabric domain and IMEX channel | Pair; PeerAgent selects the compatible Fabric transport. |
| Non-empty         | Disjoint domain or IMEX channels      | Do not pair.                                             |
| Empty             | Non-empty, or the reverse             | Do not pair.                                             |

Hybrid services remain routable independently of P/D placement matching.

Start the router with the same control-plane scope:

```bash
RUST_LOG=info dlengine-router \
  --port 3001 \
  --ctrl-address "${CTRL_ADDRESS}" \
  --ctrl-scope "${CTRL_SCOPE}"
```

## Verify the deployment

Check Ray, direct services, and router discovery before sending generation traffic:

```bash
ray status
curl -sS "http://prefill.example:8101/health"
curl -sS "http://decode.example:8102/health"
curl -sS "http://router.example:3001/v1/models"
```

Inspect the service registry and confirm both HTTP entities copy the resource published by their corresponding engine entity:

```bash
curl -sS "${CTRL_ADDRESS}/list_entities" \
  -H 'Content-Type: application/json' \
  -d '{"entity_type":"service"}'
```

For a Fabric deployment, verify that every placement is membership-ready, the placement count equals the engine worker count, IMEX channel lists are non-empty, and the selected prefill/decode placements share both a Fabric domain and an IMEX channel. For RDMA, verify that both placement lists are empty.

Then send a small request through the router:

```bash
curl -sS http://router.example:3001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"model-alias","messages":[{"role":"user","content":"Reply with OK."}],"temperature":0,"max_tokens":8}'
```

## Troubleshooting

- **No model appears at the router:** confirm that both roles use the same served model, control address, and scope. Inspect service heartbeats and role metadata.
- **Prefill and decode are healthy but not paired:** compare `resource.placements`. One-sided, partial, or transport-incompatible Fabric metadata is intentionally rejected.
- **Fabric allocation is not enabled:** confirm that topology reports exactly one membership-ready visible GPU, an IMEX channel, and a DLSlime build with Fabric allocation support.
- **Automatic transport fails:** inspect both PeerAgent topology resources. Automatic selection requires a common Fabric domain and IMEX channel, or compatible RDMA resources, and does not silently fall back to TCP.
- **Named-region transfer is rejected:** compare the published region names and lengths on both roles. Cache layouts, block size, attention sharding, and model architecture must be compatible.
- **A Ray worker cannot load the model:** mount the checkpoint at the same path on every worker. Startup fails early when a worker sees no safetensors shards.
- **Control-plane state is stale:** confirm heartbeats, then restart only the affected service. Keep Redis, `dlslime-ctrl`, engines, and router in the same scope.
