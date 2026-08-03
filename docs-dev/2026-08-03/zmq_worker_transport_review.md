# Review: hierarchical worker ZMQ transport

Date: 2026-08-03 UTC

Reviewed design: `zmq_worker_transport_design.md`

Outcome: **GO for the opt-in phase-1 implementation.** This is not approval to
make ZMQ the default or to replace DLSlime. Default promotion remains gated on
same-commit GPU/Ray A/B results.

## 1. Review method

The review was performed as an adversarial architecture pass against:

- current NanoDeploy startup, execution, failure, metrics, and shutdown code;
- vLLM `a5d19cbb95872c4b426c06735733568542fa33db` persistent Ray worker and
  MessageQueue implementation;
- ZeroMQ socket thread-affinity, identity routing, slow-start, high-water-mark,
  linger, and timeout behavior;
- the measured `c3bb461` worker boundary and the realistic performance ceiling;
- deterministic failure tables for startup, partial fan-out, DLSlime failure,
  worker exception, missing/stale result, and shutdown.

Two executable design probes were also run with the installed
`pyzmq 27.1.0` and `msgspec 0.19.0`:

1. tagged `msgspec.Struct` union round trips passed; the probe confirmed that
   unknown fields are accepted unless `forbid_unknown_fields=True`, so that
   setting is now mandatory;
2. an IPC DEALER connected and sent READY before the ROUTER bound, then
   completed READY/decode/success/stop identity routing. The sandbox disallowed
   IPC bind, so the same pure-CPU `/tmp` probe was rerun with host permission and
   passed. Its temporary directory was removed.

## 2. Blocking findings and resolutions

### R1 — partial ROUTER fan-out was underspecified

Severity: blocker

A ROUTER cannot atomically send eight rank-specific commands. If sends to ranks
0-3 succeed and rank 4 fails, ranks 0-3 may block in DLSlime `recv_seqs()`.
Retrying or continuing with a partial rank set is invalid for SP/EP collectives.

Resolution recorded in the design:

- do not call `send_seqs()` after any command-send failure;
- fail the complete LocalEngine;
- do not retry;
- rely on bounded shutdown followed by DeploymentManager's Ray kill backstop to
  release workers already waiting in DLSlime.

The same rule covers a partial DLSlime send. No partial token set reaches
postprocess.

Status: resolved.

### R2 — the metric schema would report a misleading zero

Severity: blocker for meaningful A/B

Current `LoadSnapshot` and `LLM.hierarchical_metrics()` only aggregate fields
named `ray_get_*`. Merely adding ZMQ-only boundary keys would make production
telemetry appear to have zero result-wait latency.

Resolution recorded in the design:

- add transport-neutral command-submit and result-wait boundary fields;
- add result-wait total/max/mean through LocalExecutor, LoadSnapshot, and the
  hierarchical aggregator;
- preserve existing Ray keys as Ray-mode aliases only;
- record the selected transport in benchmark output.

Status: resolved.

### R3 — schema decoding was not strict enough by default

Severity: high

The executable probe showed that a normal msgspec struct decoder silently
ignores unknown map fields. That weakens versioning and typo detection.

Resolution recorded in the design:

- every tagged protocol struct uses `forbid_unknown_fields=True`;
- explicit pre-decode and socket message-size limits are required;
- command and response maxima are 1 MiB and 16 MiB;
- exception text and traceback are bounded.

Status: resolved.

### R4 — connect-before-bind needed a bounded readiness rule

Severity: high

Workers start their persistent Ray methods before the LocalEngine event-loop
thread owns and binds the ROUTER. Default ZMQ queuing could otherwise make a
READY send appear successful before a peer exists or block indefinitely.

Resolution recorded in the design:

- worker DEALER uses `IMMEDIATE=1`;
- it polls for `POLLOUT` under the absolute startup deadline before READY;
- LocalEngine is not `EngineReady` until every expected identity and READY
  payload is validated.

The IPC proof confirmed that the intended connect-before-bind sequence works.

Status: resolved.

## 3. Correctness review

| Area | Decision | Reason |
| --- | --- | --- |
| Socket ownership | pass | ROUTER is created/used/closed only by the existing LocalEngine event-loop thread; DEALER only by the persistent worker method. |
| Rank identity | pass | ROUTER identity plus payload engine/rank/epoch validation prevents response substitution. |
| Quantum ordering | pass | one in-flight command per rank, command-before-RDMA ordering, and wave/quantum response checks prevent cross-quantum pairing. |
| GPU state semantics | pass | no retry after send, timeout, or worker failure; partial tokens are never committed. |
| Existing execution body | pass | the persistent loop calls the existing `ModelRunner.run(enable_rpc=True)` body, limiting behavior change to transport. |
| Scheduler contract | pass | LocalExecutor still emits the same ordered `WorkerDecodeResult` values after existing token/request/trace validation. |
| DP coordination | pass with residual risk | ZMQ does not change Gloo consensus. A failed DP may leave its peer waiting until the existing timeout and deployment teardown. |
| Ray fallback | pass | explicit opt-in branch leaves the current `.run.remote()`/`ray.get()` path available from the same commit. |

## 4. Lifecycle and concurrency review

`ModelRunner` is a single-concurrency synchronous Ray actor. A persistent method
therefore prevents later normal actor methods from running. This is acceptable
only under these implementation constraints:

1. model, cache, CUDA graph, identity, DLSlime, and ZMQ configuration calls all
   finish before `run_zmq_loop.remote()`;
2. after the loop starts, commands and graceful STOP use ZMQ, not actor methods;
3. final resource cleanup still has DeploymentManager `ray.kill` as a backstop;
4. the persistent ObjectRef is retained and checked, never discarded.

LocalEngine has a separate actor-method concurrency pool and one explicit
single-writer scheduler/event-loop thread. Initialization must use a thread
event so the actor-method call can wait for ROUTER readiness without touching
the socket. Shutdown must join that event-loop thread before actor kill.

Decision: pass, provided tests cover READY-before-return and initialization
failure propagation.

## 5. Failure table

| Failure point | Observable behavior | Required action | Retry? |
| --- | --- | --- | --- |
| worker configuration Ray RPC | startup exception | DeploymentManager closes actors/PGs | no |
| missing/invalid READY | startup deadline/protocol error | LocalEngine fails before `EngineReady` | no |
| command encode/size check | local protocol error | send nothing, fail LocalEngine | no |
| command send after some ranks | partial fan-out error | do not send RDMA; tear down all workers | no |
| DLSlime partial send | endpoint exception | fail LocalEngine; kill possibly blocked ranks | no |
| worker decode exception | `failure`, then exceptional sentinel | fail immediately; preserve bounded traceback | no |
| malformed success | protocol error | discard whole quantum and fail | no |
| stale/future/duplicate result | protocol error | discard whole quantum and fail | no |
| worker disappears before result | timeout or ROUTER error; sentinel may be ready | resolve ready sentinel for root cause, fail | no |
| timeout with live sentinel | ambiguous GPU/KV state | fail and tear down | no |
| normal STOP ACK missing | bounded shutdown timeout | Ray kill backstop | no |

The table has no state in which scheduler postprocess receives fewer than all
expected ranks.

## 6. Security and resource review

- MessagePack avoids pickle code execution.
- Private `0700` IPC directories constrain local peer access.
- Epoch and rank identity prevent stale actor traffic from being accepted.
- Strict schemas, maximum frame sizes, bounded HWM, absolute deadlines, and
  bounded failure text limit memory and wait amplification.
- `LINGER=0` prevents shutdown from hanging on queued messages.
- UUID paths avoid collision after hard actor death; normal cleanup removes the
  path. A hard kill may leave an empty private `/tmp` directory, which is an
  operational residue rather than a correctness hazard.

Decision: pass for a trusted single-user inference node. This is not a
multi-tenant authenticated network protocol.

## 7. Performance review

The design removes, per LocalEngine quantum:

- eight Ray actor task submissions and ObjectRefs;
- one `ray.get()` result materialization over those refs;
- Ray serialization/control scheduling for the worker response.

It adds:

- eight small MessagePack command encodes and ROUTER sends;
- eight result decodes and ROUTER receives;
- one long-lived Ray ObjectRef per worker;
- no additional sequence-payload copy and no change to GPU collectives.

The measured upper bound is modest: the centralized/hierarchical actor-submit
difference was only about 4.2 ms/quantum, or 0.26 ms/token before subtracting
the ZMQ work. Worker finish-to-result Ray overhead is already near the
centralized value. The likely win is lower CPU jitter and bounded Ray metadata,
not a large TPOT shift.

Decision: implementation cost is justified as an isolated opt-in A/B, but any
claim that ZMQ solves the remaining end-to-end gap would be unsupported.

## 8. Testability review

The transport/protocol module must not import CUDA/model code. A worker-side
loop should accept an execution callback so codec, handshake, success, failure,
timeout, and STOP behavior can be exercised with CPU threads or processes.
ModelRunner supplies the callback that invokes its existing decode method.

Existing fake-worker LocalExecutor tests remain on the default Ray path. New
ZMQ tests must use explicit transport configuration and may inject an endpoint
or context to avoid depending on cluster placement.

Decision: pass.

## 9. Residual risks accepted for phase 1

1. ZMQ and DLSlime are independent transports. The one-in-flight invariant
   makes pairing safe, but it deliberately prevents quantum pipelining.
2. A worker failure inside NCCL/EP may require Ray kill to release peers; this
   is already true of fatal worker failures on the Ray path.
3. IPC behavior still needs a real Ray multi-process smoke; the thread probe is
   directional evidence, not deployment proof.
4. Result MessagePack CPU cost may consume most of the small Ray-submit saving.
5. A hard kill may leave a private UUID `/tmp` directory.

None of these risks requires changing scheduler or model semantics, and every
one is contained by the default-Ray rollback switch.

## 10. Review decision

All design blockers found in this pass are resolved in the design document.
The implementation may proceed under these gates:

- phase 1 remains opt-in with `ray` as default;
- no C++ or external vLLM edit;
- no DLSlime payload replacement;
- protocol and transport CPU tests pass before any GPU run;
- implementation review repeats the failure table against the actual diff;
- ZMQ default promotion requires separate same-commit GPU/Ray A/B approval.

