# Rust Server + Router: PD Disaggregation Analysis

## 1. Background

The "Prefill-Decode (PD) Disaggregation" architecture separates the LLM inference process into two specialized stages:

1. **Prefill**: Processing the input prompt (computationally dense, bandwidth-bound).
2. **Decode**: Generating the output tokens one by one (memory bandwidth-bound, latency-sensitive).

By decoupling these workloads, we can scale them independently and optimize hardware utilization (e.g., using different GPU types or parallelism strategies for each).

## 2. Reference Implementation Analysis (`pd_disagg.py`)

The reference Python implementation (`examples/pd_disagg.py`) demonstrates the orchestration logic using Ray.

**Adaptation Note**: In our Rust implementation, we will **IGNORE** the `Ray` specific mechanisms (like `as_remote`, `ray.get`).

- **Ray**: Uses a centralized Driver to spawn Actors (`as_remote`) and manage object refs.
- **NanoDeploy Server**: Will use a **Manual Spawn + Server Handshake** model. Operators start `engine_server.py` manually (or via k8s), and the Rust Server connects to them via TCP to trigger the logic equivalent to `p2p_init`/`p2p_connect`.

### 2.1 Key Components & Interactions

The Python script orchestrates two `LLM` instances (wrappers around `LLMEngine`):

1. **Dual Engine Initialization**:

   - **Prefill Engine**: Configured with `mode="prefill"`, `enforce_eager=True`, `loop_count=1`. It does *not* run the decode loop.
   - **Decode Engine**: Configured with `mode="decode"`. It runs the continuous batching decode loop.

2. **P2P Handshake (Topology Discovery)**:
   Before processing requests, the two engines must establish direct communication paths for KV Cache transfer.

   - **Step 1: Gather Meta**: The coordinator fetches `engine_id`, `num_kv_blocks`, and `attn_world_size` from both engines.
   - **Step 2: P2P Init**: The coordinator calls `p2p_init` on each engine, passing the *peer's* metadata. This likely prepares the shared memory or network transport.
     - `prefill.p2p_init(decode_meta)` -> returns `prefill_endpoints`
     - `decode.p2p_init(prefill_meta)` -> returns `decode_endpoints`
   - **Step 3: P2P Connect**: The coordinator calls `p2p_connect` on each engine with the *peer's endpoint info*. This finalizes the connection.

3. **Request Lifecycle (Migration Flow)**:

   - **Submission**: Requests are first added to the **Prefill Engine** (`prefill.add_request`).
   - **Prefill Execution**: `prefill.generate()` is called. It runs the prefill phase and returns `migrated_seqs` (sequences marked `is_to_be_migrated=True`).
     - *Crucial*: The KV Cache for these sequences is now resident on the Prefill Engine's GPU.
   - **Decode Handoff**: The coordinator takes the `migrated_seqs` and adds them to the **Decode Engine** (`decode.add_request`).
     - *Implied Magic*: The `decode.add_request` call (or internal logic) triggers the physical transfer of KV Cache from Prefill to Decode nodes using the established P2P link.
     - Alternatively, the migration might be lazy/on-demand.
   - **Decode Execution**: `decode.generate()` is called. It processes the requests until completion (`is_finished=True`).
   - **Cleanup**: The coordinator calls `prefill.free_to_be_migrated(migrated_seqs)` to release the KV Cache on the Prefill Engine.

## 3. Rust Server Adaptation Strategy

To replace the Python coordinator script with a Rust Server, we must map these operations to **Spoke IPC** commands.

### 3.1 Mapping Python Methods to Spoke Actions

We need to extend the Spoke Protocol to support these new operations.

| Python Method         | Proposed Spoke Action / Logic | Description                                                                    |
| :-------------------- | :---------------------------- | :----------------------------------------------------------------------------- |
| `get_engine_id()`     | `Handshake/Init`              | standard Init response should include ID & block info.                         |
| `p2p_init(...)`       | `Action::P2PInit (0x??)`      | Send peer metadata to Engine.                                                  |
| `p2p_connect(...)`    | `Action::P2PConnect (0x??)`   | Send peer endpoints to Engine.                                                 |
| `add_request(...)`    | `Action::AddRequest (0x??)`   | Already exists. Need `SequenceList` FBS support.                               |
| `generate()`          | `Action::Step (0x??)`         | Trigger a step. In Rust, this is likely an event loop, not explicit step call. |
| `free_to_be_migrated` | `Action::Free (0x??)`         | Release specific sequence resources.                                           |

### 3.2 Router Responsibility Changes

The Rust Router must now strictly efficiently manage the lifecycle states:

1. **Topology Manager**: On startup, connect to all configured engines, determine roles (Prefill vs Decode), and orchestrate the `P2PInit` + `P2PConnect` handshake between them.
2. **Disaggregated Scheduler**:
   - Incoming requests -> **Prefill Queue**.
   - Dispatch to **Prefill Engine**.
   - Receive `StreamPush` with status `Migrated`/`PrefillDone`.
   - **Immediately** Dispatch logic to **Decode Queue**.
   - Send `Migrate` (or `AddRequest` with `migrated_from` flag) to **Decode Engine**.
   - Wait for Decoder Acknowledgement? (Or rely on async stream).
   - Send `Free` to Prefill Engine (Cleanup).

## 4. Risks & Open Questions

1. **KV Transfer Latency**: How long does migration take? Does it block the Scheduler?
   - *Analysis*: P2P transfer is usually GPU-Direct RDMA. It should be fast but non-zero. The Router shouldn't block; it should be async.
2. **Split Brain**: What if Prefill succeeds but Decode fails (OOM)?
   - *Analysis*: Need robust error handling. If Decode rejects, we might need to requeue to another Decode node or abort.
3. **Protocol Definition**: The exact payload for `P2PInit` and `P2PConnect` needs to be defined in `flatbuffers`.

## 5. Conclusion

Migrating `pd_disagg.py` logic to Rust is feasible but requires a significant upgrade to the `Router` state machine. It turns the Router from a simple Load Balancer into a **Distributed Transaction Coordinator** for the KV Cache lifecycle.
