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

Status: **CPU/control-plane GO; GPU/Ray deployment validation pending.** ZMQ is
still opt-in and Ray remains the default.

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

## Remaining deployment gates

CPU tests cannot validate:

- IPC visibility between real strict-packed Ray actor processes;
- the interaction of a persistent actor method with CUDA/NCCL execution;
- real DLSlime command-before-payload timing;
- multi-node DP2 x SP8 shutdown/failure behavior;
- ZMQ serialization cost versus Ray under production result sizes.

The next step is a GPU/Ray smoke from commit `9618d52`, with proxies unset and
`SLIME_QP_NUM=4`. Run identical Ray and ZMQ arms before any full 18,000-request
A/B. Default promotion remains blocked until those results are reviewed.

