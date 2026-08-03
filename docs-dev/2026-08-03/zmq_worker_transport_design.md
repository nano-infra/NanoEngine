# Hierarchical worker ZMQ transport design

Date: 2026-08-03 UTC

Status: `DRAFT` — implementation is blocked until the review in
`zmq_worker_transport_review.md` reaches `GO`.

## 1. Baseline and repository state

The immutable NanoDeploy baseline for this work is:

```text
c3bb461bb3bd10a947e147148a14eb9b6c798dff
c3bb461 fix: balance KV-gated decentralized admission
2026-07-31 13:55:44 +0000
```

The local vLLM reference is read-only and currently points to:

```text
/mnt/nvme1n1/ml_research/linbinbin1/vllm-main-July
a5d19cbb95872c4b426c06735733568542fa33db
```

At design start, the NanoDeploy worktree already contained user-owned changes:

```text
 M scripts/sp_ablation/start_bench.sh
?? bench_logs/
```

They are not part of this change and must not be modified, staged, or committed.

## 2. Why this is a surgical transport change

The completed `c3bb461` rate-50, DP2 x SP8 run on Ray
`10.102.206.14:8776` finished 18,000/18,000 requests in 452.184 seconds. Its
request TPOT P50/P90/P99 was 85.611/88.708/92.596 ms, while the real-token
execution-boundary TPOT-with-queue P50/P90/P99 was 84.525/87.093/87.967 ms.

The measured hierarchical worker boundary still submits eight Ray actor calls
per LocalEngine and quantum. Mean actor submission is about 5.65 ms per
engine-quantum, versus about 1.43 ms for the centralized 16-worker fan-out.
This difference is at most roughly 0.26 ms/token after division by the 16-token
quantum, and it includes Python per-rank command construction. Worker finish to
`ray.get` is already effectively equal to the centralized path (about 2.54 ms
versus 2.52 ms per quantum).

Therefore ZMQ is not expected to fix the remaining load-balance problem or to
produce a multi-millisecond/token gain by itself. The justified target is the
bounded hot-path overhead and jitter caused by repeated Ray task submission,
ObjectRef creation, and result materialization. The change must not disturb the
existing DLSlime/RDMA sequence path or GPU collectives.

## 3. vLLM reference and transferable rules

The relevant reference is vLLM's V1 Ray executor, not its complete frontend
transport:

- `vllm/v1/executor/ray_executor_v2.py`: Ray creates and places actors, then
  calls one long-lived `run.remote()` per worker. The returned ObjectRefs are
  retained as liveness sentinels.
- `vllm/v1/executor/multiproc_executor.py`: the worker busy loop receives a
  command from a persistent message queue, executes it, and sends a success or
  failure response.
- `vllm/distributed/device_communicators/shm_broadcast.py`: explicit readiness
  handshakes prevent PUB/SUB slow-join loss; local small messages use shared
  memory and remote messages use ZMQ.
- `vllm/v1/engine/core.py`: long-lived ROUTER/DEALER sockets use explicit
  identities and an initial ready message.
- `vllm/utils/network_utils.py`: sockets set explicit high-water marks, buffer
  sizes, identities, bind/connect direction, and linger behavior.

The rules copied into NanoDeploy are:

1. Ray remains the lifecycle and placement plane.
2. A worker has one long-lived execution loop, not one Ray task per quantum.
3. Startup is not READY until every data-plane peer completes a handshake.
4. Every response carries status and request identity; failures are fail-stop.
5. Persistent-loop ObjectRefs are health sentinels, not result carriers.

NanoDeploy will not copy vLLM's shared-memory broadcast queue wholesale. The
NanoDeploy LocalEngine and all of its workers are already strict-packed on one
host, each rank needs a distinct command, and DLSlime already owns the large
sequence transfer.

## 4. Scope

### In scope for phase 1

- Hierarchical `LocalExecutor` to `ModelRunner` per-quantum command and result
  control traffic.
- A single persistent worker method started once through Ray.
- One duplex IPC ROUTER/DEALER channel per LocalEngine and its local workers.
- Versioned, non-executable MessagePack envelopes.
- Startup readiness, bounded waits, identity/order validation, failure
  propagation, graceful stop, and Ray liveness sentinels.
- An explicit `ray`/`zmq` configuration switch for rollback and controlled A/B.
- Transport-specific execution-boundary metrics without breaking the existing
  Ray metric names on the Ray path.

### Explicitly out of scope

- Sequence/block-table transport. `RPCServerEndpoint.send_seqs()` and
  `RPCClientEndpoint.recv_seqs()` remain DLSlime/RDMA.
- CUDA, NCCL, Gloo, SP, EP, model, scheduler, admission, and routing behavior.
- The legacy centralized `RayExecutor`.
- Frontend to LocalEngine request/event traffic. That can become a later ZMQ
  phase after the worker path is validated independently.
- Worker restart, command replay, or transparent retry after GPU execution may
  have begun.
- Copying or modifying the external vLLM tree.

## 5. Current and target critical paths

Current hierarchical quantum:

```text
LocalEngine event-loop thread
  -> 8 x ModelRunner.run.remote(command metadata)
  -> DLSlime send_seqs(sequence payloads)
  -> ray.get(8 ObjectRefs)
  -> validate/rebuild WorkerDecodeResult
```

Target ZMQ quantum:

```text
LocalEngine event-loop thread (sole ROUTER socket owner)
  -> ROUTER sends one versioned command to each rank
  -> DLSlime send_seqs(sequence payloads)
  -> ROUTER polls and receives one versioned response from each rank
  -> validate/rebuild WorkerDecodeResult

ModelRunner actor (one persistent run_zmq_loop.remote() sentinel)
  -> DEALER receives its command
  -> existing ModelRunner.run(enable_rpc=True) receives DLSlime payload
  -> existing 16-loop GPU decode
  -> DEALER sends success or failure response
```

There is exactly one in-flight quantum per LocalEngine and per worker. A new
command is never sent until every response for the previous command has been
validated.

## 6. Socket topology and ownership

Each LocalEngine gets one private IPC endpoint:

```text
ipc:///tmp/nanodeploy-zmq-e<engine>-<uuid>/worker.sock
```

The containing directory is created with mode `0700`. A UUID makes stale-path
collision impossible; normal shutdown removes the socket and directory on a
best-effort basis.

- The LocalEngine event-loop thread creates, binds, polls, uses, and closes one
  `zmq.ROUTER` socket. No other thread touches that socket.
- Each `ModelRunner.run_zmq_loop()` creates, connects, polls, uses, and closes
  one `zmq.DEALER` socket. No other actor method touches that socket.
- DEALER identity contains protocol epoch, engine ID, and global rank and is
  also validated against the decoded payload.
- `LINGER=0`; send and receive high-water marks are small and bounded because
  only READY, one command/result, and STOP may be outstanding.
- All blocking operations use a monotonic absolute deadline. Socket polling is
  used instead of unbounded `recv()`.

IPC is selected instead of TCP because the existing placement contract already
proves that a LocalEngine and its workers share a node. This also avoids port
allocation races and remote exposure. Initialization fails if that placement
contract is not satisfied.

## 7. Protocol

Encoding uses `msgspec.msgpack`, with `pyzmq` and `msgspec` declared as direct
project dependencies. Unlike pickle, decoding an untrusted or malformed frame
cannot execute Python code.

All message types include:

- `protocol_version` (initially `1`);
- `deployment_epoch` (a random UUID for this LocalExecutor instance);
- `engine_id`;
- `global_rank`.

Messages are tagged structs:

| Direction | Type | Required additional fields |
| --- | --- | --- |
| worker -> engine | `ready` | worker rank and protocol version |
| engine -> worker | `decode` | wave ID, quantum ID, wall-clock send timestamp, optional execution trace context, diagnostics flag |
| worker -> engine | `success` | wave ID, quantum ID, token rows, worker end timestamp, optional trace, optional diagnostic |
| worker -> engine | `failure` | wave ID, quantum ID, exception type, bounded message, bounded traceback |
| engine -> worker | `stop` | reason |
| worker -> engine | `stopped` | acknowledgement |

Validation occurs before payload use:

1. ROUTER identity is known and maps to the expected rank.
2. Protocol version, epoch, engine ID, and rank match the active deployment.
3. Message kind is legal in the current state.
4. Wave and quantum match the only in-flight command for that rank.
5. Exactly one terminal response arrives per expected rank.
6. Existing token-row, mastered-request, trace, and diagnostic validation still
   runs unchanged in `LocalExecutor`.

Malformed, duplicate, stale, future, or wrong-rank messages are fatal protocol
errors. They are never ignored or retried.

## 8. Cross-transport ordering invariant

Command metadata and sequence payload use different transports, so their
pairing must not depend on arrival timing alone. The invariant is:

1. LocalExecutor sends one ZMQ `decode` command to every worker.
2. Only after all sends are accepted into the local ZMQ queues does it call
   DLSlime `send_seqs()`.
3. A worker receives exactly one command, then calls the existing blocking
   `recv_seqs()` exactly once.
4. LocalExecutor waits for and validates all results before advancing the
   quantum ID or sending another command.

If DLSlime data arrives before the worker begins `recv_seqs()`, it waits in the
existing endpoint. If the ZMQ command arrives first, the worker blocks in the
existing endpoint. Because there is only one in-flight pair and no retry, the
two streams cannot cross quantum boundaries. Wave/quantum validation on the
result detects any implementation violation before scheduler postprocess.

This invariant is the reason phase 1 does not pipeline multiple quantums and
does not use independent PUSH/PULL queues without rank identities.

## 9. Lifecycle and readiness

### Startup

1. Ray creates workers and performs all existing model, cache, CUDA graph, and
   DLSlime initialization.
2. LocalEngine validates global ranks, local ranks, fingerprints, GPU ownership,
   and strict single-node placement as it does today.
3. For ZMQ mode, LocalExecutor creates the private endpoint metadata and calls
   a short Ray configuration method on each worker. That method stores only
   immutable transport configuration; it does not create a ZMQ socket.
4. LocalExecutor starts one `run_zmq_loop.remote()` per worker and retains all
   ObjectRefs.
5. The LocalEngine event-loop thread creates and binds the ROUTER. Workers
   create their DEALER sockets inside their persistent methods and send READY.
6. The LocalEngine initialization call does not return `EngineReady` until the
   event-loop thread has validated READY from every expected rank.

The socket is created by the thread that uses it. A startup event transfers
only success/failure state between the LocalEngine actor-method thread and its
event-loop thread; it never transfers socket ownership.

### Normal shutdown

1. LocalEngine stops accepting new event-loop work and lets the current
   synchronous quantum finish or time out.
2. The event-loop thread sends `stop` to all connected workers and collects
   best-effort `stopped` acknowledgements under a short bounded deadline.
3. It closes the ROUTER/context and removes its private IPC directory.
4. Persistent run ObjectRefs should then complete normally. DeploymentManager
   retains its existing final `ray.kill(..., no_restart=True)` cleanup as a
   backstop.

### Failure shutdown

A worker exception is encoded as `failure`, sent if possible, and then
re-raised so the persistent Ray ObjectRef also completes exceptionally.
LocalEngine records a fatal failure and stops. It does not postprocess partial
tokens and does not attempt the next quantum.

## 10. Timeouts and health

- READY uses `startup_timeout_s`.
- Decode result collection uses `quantum_timeout_s` as one absolute deadline
  for the entire rank set, not a full timeout per rank.
- On ZMQ timeout or send failure, LocalExecutor performs a nonblocking
  `ray.wait()` over the fixed persistent-loop ObjectRefs and resolves any
  completed ref to expose the original actor exception. It then fails the
  LocalEngine even if no sentinel has completed yet.
- Frontend health polling may perform the same fixed-set, nonblocking sentinel
  check. Normal quantum completion performs no Ray call.
- Ray actor restart is disabled by the current default; a reconnected identity
  is not accepted as transparent recovery.

There is intentionally no decode retry. A timeout cannot prove whether GPU/KV
state was mutated, so replay could advance a request twice.

## 11. Configuration and rollback

Add:

```python
hierarchical_worker_transport: Literal["ray", "zmq"] = "ray"
```

Validation permits `zmq` only with `scheduler_arch="hierarchical"`, decode mode,
DLSlime RPC, and the existing strict-packed topology. The selected transport is
part of `collective_fingerprint()`.

The initial default remains `ray`. Benchmark launch supports
`NANODEPLOY_HIER_WORKER_TRANSPORT=zmq` so Ray and ZMQ can be compared from the
same commit and reverted without code changes. Default promotion is a separate
decision after correctness and performance gates pass.

## 12. Metrics

Ray mode preserves all current fields and meanings.

ZMQ mode records:

- `worker_command_send_latency_ms`;
- `send_seqs_latency_ms`;
- `worker_result_wait_latency_ms`;
- `executor_until_worker_result_ms`;
- `worker_observed_critical_ms`;
- `worker_finish_to_result_ms`;
- `worker_finish_skew_ms`.

The old `actor_submit_latency_ms`, `ray_get_latency_ms`,
`executor_until_ray_get_ms`, and `worker_finish_to_ray_get_ms` are not populated
with misleading ZMQ values. Aggregators already accept sparse metric names.
Benchmark output must record the selected transport.

## 13. Implementation slices

1. Add protocol structs, codec validation, deadline helpers, and ROUTER-side
   transport in a new focused module under `nanodeploy/engine/`.
2. Add worker-side configuration and persistent loop to `ModelRunner`, reusing
   the existing `run(enable_rpc=True)` body unchanged.
3. Add the configuration switch and direct dependencies.
4. Integrate LocalExecutor startup, ZMQ execution branch, health sentinel, and
   shutdown while leaving its Ray branch intact.
5. Add the LocalEngine READY barrier and event-loop-owned activation/cleanup.
6. Expose the benchmark environment switch and transport name in artifacts.

No C++ file is expected to change in phase 1.

## 14. Test plan

### CPU/unit tests

- MessagePack round trip for every message type.
- Reject unsupported version, wrong epoch/engine/rank, invalid kind, malformed
  bytes, oversized failure text, stale/future quantum, duplicate rank, and
  missing rank.
- Local IPC ROUTER/DEALER startup handshake with multiple fake workers.
- Commands may be sent before DLSlime payload; results may arrive in arbitrary
  rank order but are returned in topology order.
- Worker success and failure propagation.
- One absolute deadline across all ranks.
- Graceful STOP and idempotent cleanup.
- Ray fallback behavior remains covered by the existing LocalExecutor tests.
- Config validation and fingerprint change for `ray` versus `zmq`.
- LocalEngine does not report READY before the transport barrier and surfaces
  startup errors.

### Repository regression

Run at minimum:

```bash
python -m pytest tests/test_hierarchical_control_plane.py
python -m pytest tests/test_hierarchical_contract.py \
  tests/test_hierarchical_serving_ingress.py \
  tests/test_routing_config.py
```

### GPU/Ray gates

GPU work requires explicit elevated permission. Before launch:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export SLIME_QP_NUM=4
```

Use Ray `10.102.206.14:8776`, the same request stream, commit, nodes, DP2 x SP8
topology, batch size, and QP count for both arms:

1. short correctness/smoke in `ray` mode;
2. identical short smoke in `zmq` mode;
3. full rate-50/18,000 A/B only after both smokes pass.

Correctness acceptance:

- 18,000/18,000 success, zero failure/rejection attributable to transport;
- identical request IDs and output-token counts for deterministic inputs;
- no protocol mismatch, timeout, actor death, or leaked placement group;
- trace and quantum-diagnostic modes each pass at least one smoke;
- clean normal shutdown and fail-stop worker-kill test.

Performance acceptance:

- report TPOT and TPOT-with-queue mean/P50/P90/P99, runtime, throughput, and
  transport boundary metrics;
- ZMQ must reduce command/result control overhead without regressing worker
  critical time;
- no claim of an end-to-end win unless same-machine A/B confidence intervals or
  repeated runs separate it from run-to-run noise;
- if ZMQ regresses TPOT-with-queue P99 by more than 1% or adds instability,
  retain Ray as default and stop rollout.

## 15. Rejected alternatives

- **Replace DLSlime with ZMQ:** moves large, already optimized sequence data to
  a slower serialization path and changes too many variables.
- **PUB/SUB broadcast:** commands differ by rank; slow-join requires a more
  complex subscription barrier; a dropped subscriber would be hard to
  attribute.
- **One PUSH/PULL pair per direction:** lacks the explicit peer identity and
  single-socket rank multiplexing useful for validation and failure reporting.
- **Create the ROUTER in `initialize()` and use it in the event-loop thread:**
  violates ZeroMQ socket thread-affinity rules.
- **Use pickle:** faster to prototype but permits code execution during decode;
  MessagePack handles the required primitives.
- **Retry on timeout:** unsafe because GPU/KV mutation may already have occurred.
- **Flip the default immediately:** prevents a controlled same-commit fallback
  before GPU validation.

## 16. Design review gate

Implementation may begin only after a separate review verifies:

- socket thread ownership and startup ordering;
- cross-transport command/DLSlime pairing;
- exact rank/wave/quantum validation;
- partial send/result behavior and no unsafe retry;
- persistent Ray actor concurrency and shutdown behavior;
- health sentinel behavior;
- bounded queues, deadlines, message sizes, and error text;
- backward-compatible Ray path and metrics;
- testability without CUDA/Ray cluster access;
- realistic performance upside relative to the measured boundary.

