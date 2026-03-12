# Plan: Replace SGLang with NanoInfra (NanoDeploy + NanoRoute + DLSlime) in slime

## Context

slime uses SGLang as its inference backend for full online RL training (GRPO/PPO). The goal is a complete replacement using NanoInfra so the two codebases share one stack. This requires bridging several critical gaps:

| SGLang feature                                  | NanoInfra status                                        |
| ----------------------------------------------- | ------------------------------------------------------- |
| `/generate` (raw token IDs → tokens + logprobs) | ❌ Only `/v1/chat/completions`                          |
| Per-token log probabilities                     | ❌ `logprobs` field in SamplingParams exists but unused |
| `/pause_generation` / `/continue_generation`    | ❌ Not implemented                                      |
| `/flush_cache`                                  | ❌ Not implemented                                      |
| Weight update (NCCL collective)                 | ❌ Replaced by DLSlime RDMA (see Stream C)              |
| `/release_memory_occupation` / resume           | ❌ Not implemented (Phase 2)                            |
| Load-balancing router (sglang_router)           | ✅ NanoRoute + NanoCtrl already do this                 |
| PD disaggregation                               | ✅ NanoInfra already supports it                        |
| RDMA infra (DLSlime)                            | ✅ PeerAgent + NanoCtrl + Redis already in place        |

______________________________________________________________________

## Architecture

```
slime Training Process                     NanoInfra Inference
┌────────────────────────────┐             ┌──────────────────────────────┐
│  Actor/Critic (FSDP /      │  ①DLSlime   │  NanoDeploy ModelRunner      │
│  Megatron)                 │  RDMA READ  │  Workers (TP/DP ranks)       │
│                            │ ◄──pull─────│                              │
│  PeerAgent "trainer"       │             │  PeerAgent "worker_{id}_{r}" │
│  register_memory_region()  │             │  (reads its TP slice)        │
│  [once at init]            │             └────────────┬─────────────────┘
│                            │                          │ ZMQ (FlatBuffers)
│  Redis: PUBLISH            │                          ▼
│  "weights_ready:v{N}"      │             NanoRoute (Rust)
└────────────────────────────┘
         │
         │ ② HTTP POST /generate
         │   (input_ids, return_logprob)
         └──────────────────────────────► NanoRoute → ZMQ → NanoDeploy
                                          ◄── tokens + logprobs ──────────
```

**Why DLSlime instead of NCCL:**

- NCCL broadcast is **two-sided** — trainer rank 0 must actively participate in every `dist.broadcast()` call, blocking training until all engines receive
- DLSlime RDMA READ is **one-sided** — trainer registers params once at init; workers independently pull when signaled; trainer doesn't participate during the actual transfer
- After an in-place optimizer step (`Adam`, `SGD`), `param.data_ptr()` doesn't change → **no re-registration needed between steps**
- Workers read exactly their TP shard via `remote_offset` in the assignment tuple → no AllGather broadcast overhead

**Two channels in slime:**

- **Inference**: `slime → HTTP POST /generate → NanoRoute → ZMQ → NanoDeploy`
- **Weight sync**: `trainer signals Redis → each NanoDeploy worker pulls via RDMA READ`

______________________________________________________________________

## Work Streams

### Stream A — NanoDeploy: Log Probabilities *(critical)*

**Files to modify:**

- `NanoDeploy/nanodeploy/layers/sampler.py`
- `NanoSequence/proto/sequence.fbs` — `StepOut` table
- Regenerate FlatBuffers Python bindings in `NanoDeploy/nanodeploy/fbs/`
- `NanoRoute/src/engine_adapter.rs` (read `logprobs` from StepOut, pass to response)

**Changes:**

1. **`sampler.py`**: After computing sampled token, compute its log probability:

   ```python
   log_probs = torch.log_softmax(logits.float(), dim=-1)
   token_logprob = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
   return tokens, token_logprob   # return both
   ```

2. **`sequence.fbs` (StepOut)**: Add `logprobs: [float];` parallel to `token_ids`:

   ```flatbuffers
   table StepOut {
     seq_id: uint64;
     token_ids: [uint];
     logprobs: [float];      // ← NEW: parallel array to token_ids
     status: SequenceStatus;
   }
   ```

3. **Engine step loop** (wherever StepOut is built): populate `logprobs` from sampler output.

4. **NanoRoute `engine_adapter.rs`**: extract `logprobs` from StepOut and accumulate per request.

______________________________________________________________________

### Stream B — NanoDeploy: Pause / Resume / Flush *(critical)*

These are needed by each NanoDeploy worker to briefly gate inference during the RDMA weight pull.

**Files to modify:**

- `NanoDeploy/nanodeploy/engine/llm_engine.py`
- `NanoDeploy/nanodeploy/engine/scheduler.py` (may need `_CppScheduler` extension)
- `NanoDeploy/nanodeploy/llm_component.py` — expose as Ray RPC methods

**Changes:**

1. Add `self._paused: bool = False` flag to `LLMEngine`.
2. `pause()`: set `_paused = True`; scheduler skips promoting waiting → running.
3. `resume()`: set `_paused = False`.
4. `flush_cache()`: abort all waiting sequences; poll until running queue is empty.
5. Expose all three as methods on `LLMComponent`.

______________________________________________________________________

### Stream C — DLSlime RDMA Weight Sync *(critical — replaces NCCL)*

**Protocol:**

```
INIT (once at training start):

  [slime trainer rank 0]
    agent = start_peer_agent(alias="trainer", server_url=nanoctrl_url, scope=scope)
    for name, param in model.named_parameters():
        agent.register_memory_region(
            mr_name=f"w:{name}",
            ptr=param.data_ptr(),
            length=param.numel() * param.element_size()
        )
    # Wait for all NanoDeploy workers to connect
    agent.wait_for_peers(all_worker_aliases)

  [each NanoDeploy ModelRunner worker, TP rank r, TP size T]
    agent = start_peer_agent(alias=f"worker_{engine_id}_{r}", server_url=nanoctrl_url, scope=scope)
    agent.set_desired_topology(target_peers=["trainer"], symmetric=True)
    agent.wait_for_peers(["trainer"])

    # Build assignment list ONCE — reused every update
    assignments = []
    for name, param in model.named_parameters():
        mr_info = agent.get_mr_info("trainer", f"w:{name}")
        remote_h = agent.register_remote_memory_region("trainer", f"w:{name}", mr_info)
        local_h  = agent.register_memory_region(f"local_w:{name}", param.data_ptr(),
                                                  param.numel() * param.element_size())
        full_numel  = param.numel() * T          # full param size on trainer
        shard_bytes = param.numel() * param.element_size()
        remote_off  = r * shard_bytes            # this worker's slice offset
        assignments.append((local_h, remote_h, 0, remote_off, shard_bytes))

WEIGHT UPDATE (each training iteration):

  [slime trainer rank 0]
    # Params updated in-place by optimizer → same data_ptr → no re-registration
    redis.publish(f"{scope}:weight_ready", f"{version}:trainer")

  [each NanoDeploy worker, subscribed to Redis channel]
    on "weight_ready" event:
      self.pause()             # Stream B: stop new requests
      self.flush_cache()       # Stream B: drain in-flight
      endpoint = agent.get_endpoint("trainer")
      endpoint.read(assign=assignments).wait()   # RDMA READ all param shards
      # params updated in-place on GPU — no copy needed, RDMA writes directly
      self.resume()
      redis.publish(f"{scope}:weight_loaded", f"{version}:{agent.alias}")

  [slime trainer rank 0]
    # Wait for N_workers confirmations from Redis before continuing
    wait_for_confirmations(version, expected_count=num_workers)
```

**New files:**

- `NanoDeploy/nanodeploy/weight_sync/dlslime_weight_sync.py` — worker-side PeerAgent init, subscription loop, RDMA pull, confirmation publish
- `slime/backends/nanoinfra_utils/trainer_weight_server.py` — trainer-side PeerAgent init, MR registration, version signaling, confirmation wait

**Files to modify:**

- `NanoDeploy/nanodeploy/llm_component.py` — call `dlslime_weight_sync.init()` at startup; subscription runs as a background thread inside the worker
- `NanoDeploy/nanodeploy/worker/model_runner.py` — expose `param.data_ptr()` map for MR registration; weights updated in-place by RDMA (no explicit copy)

**TP sharding note:** If the trainer uses TP (Megatron), trainer rank 0 first AllGathers its own TP shards before registering. This is one AllGather at init time (not every step), because the MR just points to the memory — and after AllGather the full param lives in a new buffer registered once. Each training step the optimizer updates that buffer in-place.

______________________________________________________________________

### Stream D — NanoRoute: `/generate` Endpoint *(critical)*

**File to modify:**

- `NanoRoute/src/http_server.rs`

**New request struct (SGLang-compatible):**

```rust
#[derive(Deserialize)]
struct GenerateRequest {
    input_ids: Vec<u32>,
    sampling_params: GenerateSamplingParams,  // temperature, top_p, top_k, max_new_tokens, stop_token_ids, ...
    return_logprob: Option<bool>,
    image_data: Option<Vec<String>>,
}
```

**Response (SGLang-compatible — slime parser unchanged):**

```json
{
  "text": "generated text",
  "meta_info": {
    "output_token_logprobs": [[-0.5, 100], [-0.3, 201], ...],
    "finish_reason": "stop"
  }
}
```

**Implementation:**

- Skip tokenizer; build `Sequence` FlatBuffer directly from `input_ids`
- Route through same prefill→migration→decode flow as `/v1/chat/completions`
- Decode generated token IDs → text via tokenizer (for `text` field)
- Collect `logprobs` from StepOut stream (requires Stream A)
- `return_logprob: false` → omit `output_token_logprobs` (saves bandwidth)

______________________________________________________________________

### Stream E — slime: New NanoInfra Backend *(critical)*

New directory: `slime/backends/nanoinfra_utils/`

#### `nanoinfra_engine.py` — drop-in for `sglang_engine.py`

```python
class NanoInfraEngine(RayActor):
    def init(self, nanoroute_ip, nanoroute_port, nanoctrl_url, config: Config, ...):
        self.llm_component = LLMComponent(config)  # NanoDeploy Ray actor
        # LLMComponent auto-registers with NanoCtrl → NanoRoute discovers via Redis
        self.nanoroute_url = f"http://{nanoroute_ip}:{nanoroute_port}"

    # Generation: just returns the NanoRoute URL — rollout code does HTTP directly
    def get_generate_url(self) -> str:
        return f"{self.nanoroute_url}/generate"

    # Control: direct Ray RPC (no HTTP overhead)
    def health_generate(self):   requests.get(f"{self.nanoroute_url}/health")
    def flush_cache(self):       ray.get(self.llm_component.flush_cache.remote())
    def pause_generation(self):  ray.get(self.llm_component.pause.remote())
    def continue_generation(self): ray.get(self.llm_component.resume.remote())
    # Weight sync: handled by DLSlime background thread — no explicit RPC needed
```

#### `nanoinfra_rollout.py` — drop-in for `sglang_rollout.py`

Two-line diff from `sglang_rollout.py`:

- URL base: `nanoroute_ip:nanoroute_port` (same `/generate` path and payload)
- Response parser: **identical** — `output["meta_info"]["output_token_logprobs"]`

RL algorithm code (`rollout_log_probs`, importance sampling) is **unchanged**.

#### `trainer_weight_server.py` — replaces `update_weight_from_distributed.py`

Called by slime trainer after each optimizer step:

```python
class TrainerWeightServer:
    def init(self, model, nanoctrl_url, scope, worker_aliases):
        self.agent = start_peer_agent(alias="trainer", server_url=nanoctrl_url, scope=scope)
        self._register_all_params(model)  # one-time MR registration
        self.agent.wait_for_peers(worker_aliases)
        self._version = 0

    def update_weights(self):
        # Params already updated in-place by optimizer — nothing to copy
        self._version += 1
        self._redis.publish(f"{scope}:weight_ready", f"{self._version}:trainer")
        self._wait_for_confirmations(self._version, len(worker_aliases))
```

#### `arguments.py` — CLI args

- `--nanoinfra-nanoroute-ip/port`
- `--nanoinfra-nanoctrl-address`, `--nanoinfra-scope`
- `--nanoinfra-tensor-parallel-size`, `--nanoinfra-data-parallel-size`

#### Integration hook in `slime/ray/rollout.py`

Add `--rollout-engine {sglang|nanoinfra}` flag.

______________________________________________________________________

### Stream F — Memory Offload *(Phase 2 — colocated mode)*

Required for `--colocate` (training + inference share the same GPU pool).

**`release_memory_occupation()`**: move model params + KV cache to CPU pinned memory.
**`resume_memory_occupation(tags)`**: reload from CPU. Tags: `["weights"]`, `["kv_cache", "cuda_graph"]`.

Files: `nanodeploy/worker/model_runner.py`, `nanodeploy/context/cache.py`.

*Note: When memory offload is active, `param.data_ptr()` changes on GPU reload → DLSlime MRs must be re-registered after `resume_memory_occupation(["weights"])`. Handle this in `dlslime_weight_sync.py`.*

______________________________________________________________________

## Implementation Order

| Order | Stream                      | Dependency           |
| ----- | --------------------------- | -------------------- |
| 1     | A — Log probabilities       | —                    |
| 2     | B — Pause/flush control     | —                    |
| 3     | D — NanoRoute `/generate`   | A                    |
| 4     | C — DLSlime weight sync     | B                    |
| 5     | E — slime NanoInfra backend | C, D                 |
| 6     | F — Memory offload          | E working end-to-end |

Streams A, B, D can be implemented in parallel. Stream C requires B. Stream E requires C and D.

______________________________________________________________________

## Critical File Map

| File                                                       | Change                                                         |
| ---------------------------------------------------------- | -------------------------------------------------------------- |
| `NanoDeploy/nanodeploy/layers/sampler.py`                  | Return `(tokens, token_logprobs)`                              |
| `NanoSequence/proto/sequence.fbs`                          | Add `logprobs: [float]` to `StepOut`                           |
| `NanoDeploy/nanodeploy/engine/llm_engine.py`               | Add `pause/resume/flush_cache`                                 |
| `NanoDeploy/nanodeploy/llm_component.py`                   | Expose pause/flush as Ray RPC; init DLSlime weight sync thread |
| `NanoDeploy/nanodeploy/worker/model_runner.py`             | Expose `param.data_ptr()` map for MR registration              |
| `NanoDeploy/nanodeploy/weight_sync/dlslime_weight_sync.py` | **NEW**: worker-side RDMA weight pull                          |
| `NanoRoute/src/http_server.rs`                             | Add `POST /generate` handler                                   |
| `NanoRoute/src/engine_adapter.rs`                          | Extract `logprobs` from StepOut                                |
| `slime/backends/nanoinfra_utils/nanoinfra_engine.py`       | **NEW**: Ray actor wrapping LLMComponent                       |
| `slime/backends/nanoinfra_utils/nanoinfra_rollout.py`      | **NEW**: rollout via NanoRoute `/generate`                     |
| `slime/backends/nanoinfra_utils/trainer_weight_server.py`  | **NEW**: trainer PeerAgent + Redis signaling                   |
| `slime/backends/nanoinfra_utils/arguments.py`              | **NEW**: CLI args                                              |
| `slime/ray/rollout.py`                                     | Add `--rollout-engine nanoinfra` branch                        |

______________________________________________________________________

## Branch / PR Workflow

Repo: `/mnt/nvme1n1/ml_research/majinming/src/NanoInfra`
Design doc destination: `ai_docs/NanoDeploy/slime_rl_integration.md`

Implementation streams (A–F) are on separate branches off `feature/nanodeploy_vl`.

______________________________________________________________________

## Verification

1. **Log probs unit test**: Single NanoDeploy forward pass → verify `StepOut.logprobs` matches `F.log_softmax(logits)[token_id]`.
2. **`/generate` endpoint**: `curl POST /generate -d '{"input_ids":[1,2,3],...,"return_logprob":true}'` → verify `meta_info.output_token_logprobs` is returned.
3. **DLSlime weight sync**: Run `trainer_weight_server` + 2 NanoDeploy workers; after `update_weights()`, verify each worker's model params match trainer params.
4. **TP shard correctness**: With TP=2, verify worker 0 gets first half and worker 1 gets second half of each parameter.
5. **Integration test**: `slime/train.py --rollout-engine nanoinfra` on Qwen3-7B for 5 rollout steps; verify training loss decreases and weight version increments.
6. **Logprob equivalence**: Compare NanoInfra `/generate` logprobs vs SGLang on the same prompt; values should match within `1e-3`.
