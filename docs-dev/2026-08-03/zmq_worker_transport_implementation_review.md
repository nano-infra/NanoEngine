# Implementation review: hierarchical worker ZMQ transport

Date: 2026-08-03 UTC

Implementation commit:

```text
9618d529254344e2de956110b2fe8dd85b045d08
9618d52 feat: add persistent ZMQ worker transport
```

Design baseline:

```text
c3bb461bb3bd10a947e147148a14eb9b6c798dff
```

Status: **opt-in deployment GO; default promotion NO-GO pending the remaining
comparative/failure gates.** ZMQ remains opt-in and Ray remains the default.

## Scope confirmation

The committed implementation matches the reviewed phase-1 boundary:

- Ray still creates, places, configures, monitors, and kills workers.
- One persistent `run_zmq_loop.remote()` ObjectRef is retained per worker.
- LocalEngine owns one event-loop-thread ROUTER; each persistent worker method
  owns one DEALER.
- ZMQ carries only command metadata and token/diagnostic results.
- DLSlime still carries all `Sequence`/block-table payloads.
- Existing model execution, CUDA/NCCL/Gloo, scheduling, admission, routing, and
  legacy centralized execution are unchanged.
- No C++ or external vLLM file changed.

The pre-existing user worktree entries remain outside every commit:

```text
 M scripts/sp_ablation/start_bench.sh
?? bench_logs/
```

## Actual implementation map

- `nanodeploy/engine/worker_transport.py`
  - strict tagged MessagePack protocol;
  - private IPC endpoint creation and cleanup;
  - ROUTER/DEALER identity and READY barrier;
  - one-in-flight quantum state machine;
  - bounded frame sizes, HWM, deadlines, failure text, STOP/ACK;
  - fail-stop partial fan-out and worker exception handling.
- `nanodeploy/engine/local_executor.py`
  - Ray/ZMQ branch;
  - command-before-DLSlime ordering;
  - persistent ObjectRef liveness checks;
  - ordered response conversion into the unchanged result validation path;
  - transport-neutral and legacy-Ray boundary metrics.
- `nanodeploy/worker/model_runner.py`
  - immutable pre-loop configuration;
  - persistent client callback that reuses the existing
    `run(enable_rpc=True)` method.
- `nanodeploy/engine/local_engine.py`
  - startup event that prevents `EngineReady` before ZMQ READY;
  - event-loop-owned activation and shutdown;
  - frontend health sentinel checks.
- config/contract/metrics/benchmark files
  - opt-in `hierarchical_worker_transport` field and fingerprint;
  - neutral result-wait aggregation;
  - `NANODEPLOY_HIER_WORKER_TRANSPORT=zmq` benchmark selection and artifact
    field;
  - direct `pyzmq>=25.0.0` and `msgspec>=0.18.0` dependencies.

## Failure-table re-review against code

| Case | Actual behavior | Review |
| --- | --- | --- |
| encode/schema failure before fan-out | all frames are encoded/validated before first send | pass |
| rank N send fails after earlier sends | server marks failed and reports sent ranks; LocalExecutor never reaches DLSlime send | pass |
| DLSlime send fails | server is marked failed; event loop fails; no result postprocess | pass |
| worker callback fails | bounded failure response is attempted, persistent method re-raises | pass |
| stale/wrong/duplicate result | strict identity/epoch/rank/wave/quantum state rejects it | pass |
| missing result | one absolute quantum deadline fails the LocalEngine | pass |
| worker ObjectRef completes | health or error-path nonblocking `ray.wait` exposes failure | pass |
| startup peer missing | READY deadline releases LocalEngine startup event with failure | pass |
| normal shutdown | event-loop owner sends STOP, waits up to five seconds, then Ray kill remains a backstop | pass |

There is no code path that calls scheduler postprocess with a partial rank set or
retries an ambiguous quantum.

## Thread/concurrency re-review

- `ZmqWorkerServer._require_active()` asserts the ROUTER owner thread before
  every command/result operation.
- The ROUTER is created inside `LocalEngineCore._event_loop()`, not in the Ray
  actor-method thread.
- Worker DEALER creation and use are entirely inside `run_zmq_loop()`.
- All short Ray actor methods finish before the persistent single-concurrency
  method starts.
- Socket cross-thread misuse has an explicit regression test.

Decision: pass.

## CPU validation

Executed on commit content before commit:

```bash
python -m pytest -q \
  tests/test_worker_transport.py \
  tests/test_hierarchical_control_plane.py \
  tests/test_hierarchical_contract.py \
  tests/test_hierarchical_serving_ingress.py \
  tests/test_routing_config.py
```

Result: `121 passed in 4.61s`.

The nine transport tests cover:

- strict codec round trips and unknown-field rejection;
- oversized command rejection;
- connect-before-bind READY and identity routing;
- topology-ordered results across multiple quantums and a new wave;
- worker execution failure with no retry;
- wrong deployment epoch;
- skipped quantum rejection;
- cross-thread socket rejection;
- one absolute READY timeout;
- fatal partial command fan-out with the sent-rank set reported.

Additional standalone check:

```bash
python tests/test_sequence_proxy.py
```

Result: `[ok] C++ proxy containers behave as mutable views`.

`python -m pip check` reported unrelated pre-existing environment conflicts
(`pycairo`, Torch version constraints, and `outlines-core`). The new direct
dependencies are installed as `pyzmq 27.1.0` and `msgspec 0.19.0`; no failure
was attributed to this change.

## GPU/Ray validation

Validation ran from tree `d59eb84af585696f71cd5fb4abe6314417fd7856`.
The transport code in that tree is the implementation committed at `9618d52`;
the intervening commit only added this review document. Both proxy families
were unset and `SLIME_QP_NUM=4` was present in the driver environment.

Common production-shaped configuration:

```text
Ray: 10.102.206.14:8776
topology: DP=2, SP=8, EP=16, TP=1 (two nodes, 16 GPUs)
batch size: 192
request rate: 50/s
loop count: 16
router: least_batch_v2 / LeastBatch master selection
backend: hao_basic, full CUDA graph
dataset: sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv
```

### Same-commit 512-request smoke

Both arms used the same 512 inputs and enabled hierarchical quantum
diagnostics. Both exited zero with `512/512` successes and no failure or
rejection.

| Metric | Ray | ZMQ | ZMQ minus Ray |
| --- | ---: | ---: | ---: |
| runtime (s) | 77.880 | 78.826 | +0.946 |
| TPOT-with-queue mean (ms) | 46.334 | 46.513 | +0.179 |
| TPOT-with-queue P50 (ms) | 46.649 | 46.791 | +0.142 |
| TPOT-with-queue P90 (ms) | 47.470 | 47.551 | +0.081 |
| TPOT-with-queue P99 (ms) | 47.650 | 48.027 | +0.377 (+0.79%) |
| dispatch TPOT P99 (ms) | 49.243 | 49.315 | +0.072 |
| command submit, two-engine mean (ms/quantum) | 1.402 | 0.494 | -0.907 (-64.7%) |
| finish-to-result, two-engine mean (ms/quantum) | 0.874 | 0.382 | -0.492 (-56.3%) |

The end-to-end difference is below the review's 1% P99 stop threshold. The ZMQ
arm's executor critical path was about 7.3 ms/quantum slower in this short run,
while both isolated control-boundary measurements improved materially. This is
consistent with GPU/run-to-run variation rather than a ZMQ control regression;
one short sample is not sufficient to claim an end-to-end improvement.

Artifacts:

```text
bench_logs/zmq_worker_transport_smoke_d59eb84_20260803/ray/.../20260803_141734.summary.json
bench_logs/zmq_worker_transport_smoke_d59eb84_20260803/zmq/.../20260803_141302.summary.json
```

### ZMQ 18,000-request full run

At the user's direction, only the ZMQ full arm was completed. It exited zero,
cleanly drained the long-request tail, and released its placement groups.

| Metric | Result |
| --- | ---: |
| successful / total / failed | 18,000 / 18,000 / 0 |
| ingress / scheduler rejected | 0 / 0 |
| runtime | 450.284 s |
| output throughput | 23,988.42 token/s |
| TPOT-with-queue mean / P50 / P90 / P99 | 80.410 / 83.315 / 86.156 / 87.294 ms |
| dispatch TPOT mean / P50 / P90 / P99 | 81.786 / 84.455 / 87.666 / 91.541 ms |
| TTFT mean / P50 / P90 / P99 | 1955.181 / 1935.047 / 2535.211 / 3432.001 ms |
| goodput at TPOT-with-queue < 100 ms | 18,000 / 18,000 (100%) |

The full log contains no `Traceback`, `RayTaskError`, transport exception,
protocol mismatch, timeout, or actor-death report. After normal shutdown,
`ray status` reported `0.0/16.0 GPU`, no pending demand, and no placement-group
reservation. No local `nanodeploy-zmq-*` IPC directory remained.

The completed artifact is:

```text
bench_logs/zmq_worker_transport_ab_d59eb84_20260803/zmq/.../20260803_142551.summary.json
```

For direction only, the earlier Ray-only baseline at `c3bb461` completed the
same 18,000-input workload in 452.184 s with TPOT-with-queue
mean/P50/P90/P99 `81.464/84.525/87.093/87.967` ms. The ZMQ run is lower by
`1.054/1.210/0.937/0.673` ms and is 1.899 s faster, but the hashes differ and
the runs were not paired. These numbers therefore do not establish causality.

A same-commit Ray full arm was started, then explicitly cancelled during CUDA
graph capture when the user requested a ZMQ-only full run. It submitted no
benchmark requests and is excluded from all comparisons. CUDA workers emitted
signal diagnostics as they were interrupted; cleanup completed and the cluster
returned to `0/16` GPUs.

## Final review decision

Real multi-process IPC visibility, persistent actor/CUDA execution, live
DLSlime ordering, multi-node operation, production result sizes, normal
shutdown, and a full ZMQ workload have now passed. The phase-1 ZMQ transport is
accepted for opt-in use via:

```bash
NANODEPLOY_HIER_WORKER_TRANSPORT=zmq
```

Ray remains the default. Default promotion is still blocked on:

- a completed same-commit, full-size Ray/ZMQ comparison (preferably repeated);
- one GPU execution-trace smoke;
- an explicit live worker-kill/fail-stop validation.

This preserves the reviewed rollback path and avoids an end-to-end performance
claim unsupported by paired full-size evidence.
