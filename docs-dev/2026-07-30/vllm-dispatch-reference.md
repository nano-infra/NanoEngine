# vLLM V1 Dispatch Reference

## Scope

This note compares NanoDeploy's hierarchical `least_batch` path with local
vLLM commit `a5d19cbb9`. The goal is to remove the rate-50 admission RPC
collapse without hiding scheduler queueing time.

## What vLLM Does

vLLM separates transport acceptance from scheduler admission:

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

## NanoDeploy Gap

NanoDeploy already has most required primitives:

- `enqueue_add_batch()` quickly reserves and queues requests;
- `_drain_ingress()` batches local `scheduler.add()` calls;
- `LocalScheduler.add()` retains requests in `WAITING_ADMISSION`;
- `LoadSnapshot` carries waiting, running, pending ingress, and KV state.

The `least_batch` path bypasses those advantages. It calls blocking
`admit_add()`, waits for the single-writer loop to attempt immediate placement,
polls every Ray object reference separately, removes deferred requests from the
local waiting queue, and retries them on another DP. Separately, frontend
polling issues distinct actor RPCs for add results, first-token events,
first-schedule events, terminal events, health, and load.

## Recommended Design

### Phase 1: Sticky, queue-preserving routing

- Score each DP from `waiting`, `running`, and frontend tentative assignments;
  start with vLLM's `waiting * 4 + running` policy.
- Batch new commands per DP and use the existing `enqueue_add_batch()` fast
  path.
- Once routed, retain the request in that LocalScheduler's waiting queue.
- Remove `admission_deferred` fallback and global retry from normal flow.
- Keep fallback only for engine failure. Treat hard validation failure as a
  rejection and queue-capacity exhaustion as explicit overload/backpressure.

### Phase 2: O(DP), not O(request), control-plane work

- Replace per-object `ray.wait([ref], timeout=0)` with one batched `ray.wait`
  over outstanding batch handles.
- Consolidate LocalEngine events and the latest load snapshot into one
  `drain_frontend_events()` call per DP per frontend tick. A later persistent
  long-poll/event channel can further approximate vLLM's ZMQ output task.

### Phase 3: Preserve metric boundaries

Record three separate milestones:

- ingress receipt: target LocalEngine has durably queued the command;
- local queued: LocalScheduler has created the waiting record;
- admission/scheduled: the request first obtains SP/KV placement.

Primary arrival TTFT and TPOT-with-queue must include all real waiting.
Additionally report `queued -> scheduled` locally, split into cadence wait and
capacity/contention wait if the fixed decode-loop admission cadence should be
excluded from a centralized-scheduler comparison. Never use the fast ingress
receipt as bootstrap readiness.

## Validation

Run current-code-compatible rate 20 and rate 50 tests. Require:

- 18,000 requests dispatched without growing dispatch lag;
- normal-flow fallback and global retry counts remain zero;
- outstanding control-plane handles scale with DP count/batches, not requests;
- P50/P90/P99 ingress receipt, queued-to-scheduled, TTFT, TPOT-with-queue, and
  decode ITL are reported;
- queue capacity produces controlled backpressure or rejection, not a retry
  storm.

After the blocking path is removed, return `LocalEngineCore.max_concurrency`
to 32 (or lower) and A/B it separately; it should no longer be a throughput
control knob.
