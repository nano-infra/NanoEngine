# vLLM V1 Dispatch Reference

## Scope

This note compares NanoDeploy's hierarchical `least_batch` path with local
vLLM commit `a5d19cbb9`. The goal is to remove the rate-50 admission RPC
collapse while retaining NanoDeploy's global queue: requests are sent to a
LocalEngine only when a fresh load snapshot indicates admission capacity.

## What vLLM Does

For API-to-EngineCore dispatch, vLLM separates transport acceptance from
scheduler admission:

1. `AsyncLLM` registers frontend output state, chooses one DP EngineCore, and
   awaits only the ZMQ send.
2. A dedicated EngineCore I/O thread deserializes requests into an in-process
   input queue. The core loop drains that queue before the next engine step.
3. `Scheduler.add_request()` appends the request to its waiting queue and
   records `QUEUED`; it does not require the request to fit in the next GPU
   batch.
4. `SCHEDULED` is recorded only when the scheduler moves the request to
   running. vLLM reports queue time as `SCHEDULED - QUEUED`.

For internal DP load balancing, each EngineCore publishes `[waiting, running]`
counts through a coordinator. Frontends consume the newest snapshot, choose
the minimum `waiting * 4 + running` score, locally increment the selected
waiting count between roughly 100 ms updates, rotate tie-breaking, and keep the
request sticky to that engine. There is no per-request load query, placement
ACK, cross-engine admission fallback, or global retry.

vLLM also multiplexes outputs through a persistent output task. It does not
poll a separate RPC for every event type. Its ZMQ high-water marks are
unbounded, so NanoDeploy should copy the asynchronous queueing model but add
explicit overload limits.

### Ray-specific behavior

vLLM RayExecutorV2 uses Ray to place and own long-lived worker actors, then
starts one `run.remote()` busy loop per actor. Scheduler inputs and model
outputs travel through shared-memory or TCP `MessageQueue` instances rather
than per-step Ray actor RPCs. Each long-lived `run()` ObjectRef is retained as
a liveness sentinel. A background monitor calls one `ray.wait()` over that
fixed worker set with a five-second timeout.

The older Ray executor also avoids per-ref polling: it fans a collective call
out to all workers and then performs one `ray.get(refs)`, or wraps the complete
ref list in one future for non-blocking execution.

The transferable rule is therefore not “increase Ray actor concurrency.” It is
“keep ObjectRef cardinality bounded by workers or batches, never by requests.”

## NanoDeploy Gap

NanoDeploy already has most required primitives:

- `try_admit_batch()` atomically evaluates a candidate batch;
- `LoadSnapshot` carries waiting, running, pending ingress, and KV state.

The `least_batch` path launches one blocking `admit_add()` actor call per
request, eagerly moves the global queue into `_pending_ingress`, and calls
`ray.wait([handle], timeout=0)` for every pending request on every frontend
poll. A deferred request can then create a second-DP fallback and a later
global retry. ObjectRefs and retries therefore grow with offered load rather
than DP count.

## Recommended Design

### Phase 1: Global queue with bounded batch flights

- Keep requests in global FIFO until a new `LoadSnapshot` indicates an engine
  can attempt admission.
- Permit at most one `_AdmissionBatchFlight` per engine. Select a bounded
  candidate batch from the global head and issue one
  `admit_batch.remote(commands, load_generation)` call.
- Poll all active batch refs with one `ray.wait(refs,
  num_returns=len(refs), timeout=0)`, then call one `ray.get(ready_refs)`.
  With DP=2, the wait set can never exceed two refs.
- Accepted requests become owned by the selected engine. Deferred requests
  return to global FIFO and remain blocked until an admission-relevant load
  generation advances.
- Do not immediately fallback to another DP in the same generation. Retry
  limits apply to batches/generations, not individual poll iterations.

The LocalEngine batch actor method may still wait for its single-writer loop,
but it occupies only one actor concurrency slot and creates only one mailbox
entry per engine. `max_concurrency` is no longer the admission window.

### Phase 2: Capacity notification and event consolidation

- Initially, fetch all DP cached load snapshots together at the existing
  100 ms cadence. Include an admission-capacity epoch or generation in every
  snapshot so stale credits cannot launch duplicate batches.
- If 100 ms wakeup granularity is material, keep one long-poll state-change
  ObjectRef per engine. Re-arm it after completion and wait over the fixed DP
  set. This is the Ray analogue of vLLM's persistent MessageQueue.
- Consolidate add results, first-token, first-schedule, terminal, health, and
  load into one frontend event batch per engine rather than separate actor RPCs.
- A later MessageQueue/TCP control channel can remove Ray from the hot path
  completely, following RayExecutorV2, but is not required for the first A/B.

### Phase 3: Preserve metric boundaries

Record these milestones:

- global queued: request entered the frontend global FIFO;
- batch dispatched: capacity snapshot caused a batch RPC;
- admission committed: the request obtained SP/KV placement;
- first scheduled/model token/terminal.

Primary arrival TTFT and TPOT-with-queue must include all real waiting.
Report global queue time separately from batch RPC time and local command queue
time. Split cadence wait from capacity/contention wait if the fixed decode-loop
admission cadence should be excluded from a centralized-scheduler comparison.
Never treat batch dispatch or mailbox receipt as bootstrap readiness.

## Validation

Run current-code-compatible rate 20 and rate 50 tests. Require:

- 18,000 requests dispatched without growing dispatch lag;
- normal-flow same-generation fallback count remains zero;
- outstanding admission refs never exceed the number of DP engines;
- P50/P90/P99 global queue, batch RPC, local command queue, TTFT,
  TPOT-with-queue, and decode ITL are reported;
- queue capacity produces controlled backpressure or rejection, not a retry
  storm.

After the blocking path is removed, return `LocalEngineCore.max_concurrency`
to 32 (or lower) and A/B it separately; it should no longer be a throughput
control knob.
