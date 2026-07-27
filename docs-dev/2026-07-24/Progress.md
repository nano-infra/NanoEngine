# Decentralized NanoDeploy Implementation Progress

Updated: 2026-07-24 (DP2SP4 lifecycle validation)

## Goal

Implement `docs-dev/2026-07-23/decentralized_nanodeploy_proposal.md` using
`/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3` as the model.

## Completed

- Inventoried the legacy decentralized scheduler path and current centralized engine.
- Completed and committed Phase 0 as
  `2f94869 Implement hierarchical scheduler phase 0 contracts`.
- Completed and committed the Phase 1/2-capable runtime control plane as
  `94cdaa9 Implement hierarchical runtime control plane`.
- Introduced the `scheduler_arch=legacy_global|hierarchical` configuration contract.
- Added hierarchical topology validation and deterministic rank slicing.
- Added wire/control contracts for Add, Abort, Finish, Load, worker decode results,
  and fixed 16-step decode batches.
- Removed the old `scheduler_mode=decentralized` implementation and command-line
  options.
- Refactored the C++ scheduler API to expose admission and decode planning.
- Added deterministic persistent control dummies with reserved KV blocks.
- Added bootstrap-token accounting so the internal bootstrap token is excluded from
  client-visible completion tokens.
- Added a Python `LocalScheduler` single-writer wrapper for a DP engine.
- Removed the worker-side random emergency dummy fallback.
- Updated examples and scripts to use `scheduler_arch`.
- Added CPU tests covering config/topology/contracts/dummy admission/decode planning,
  bootstrap accounting, final-step overrun, and abort.
- Followed the required install procedure with proxy variables enabled and elevated
  permissions: `pip install -v -e .` completed successfully and installed
  `nanodeploy 0.2.0`.
- Repeated the same exact elevated editable-install procedure after the final Phase 0
  C++ accounting fix; it completed successfully.
- Verified that the active C++ extension is loaded from the editable installation and
  moved a stale source-tree extension aside to
  `/tmp/nanodeploy-stale-local-extension-20260724/`.
- Passed the Phase 0 CPU regression suite (41 tests).
- Implemented the initial Phase 1 control plane:
  - `RequestRouter` with round-robin dispatch, queue-full retry, sticky ownership,
    owner-validated finish, abort routing, and load snapshots.
  - `DecodeCoordinator` with READY/fingerprint validation, wave numbering, and
    wakeup-race handling.
  - `LocalExecutor` and `LocalEngineCore` with DP-local workers, one single-writer
    scheduling loop, fixed 16-quantum waves, Gloo consensus, prioritized aborts,
    event emission, and fail-fast behavior.
  - `DeploymentManager` with exact per-DP placement groups, global KV-block
    consensus, zero-cache-before-READY, coordinator wiring, health polling, and
    cleanup.
  - Hierarchical frontend integration in `LLMEngine`.
- Added Phase 1 control-plane tests. The combined CPU suite currently passes 52
  tests; the standalone Sequence proxy check, `compileall`, and
  `git diff --check` also pass.
- Hardened the initial runtime implementation after review:
  - each placement group now has exact 1-GPU worker bundles plus a separate CPU
    control bundle;
  - pending placement groups and partially launched worker sets are tracked for
    startup-timeout cleanup;
  - worker global/local ranks, one-GPU ownership, node placement, and worker/engine
    config fingerprints are all checked before READY;
  - waiting abort events publish immediately, in-flight abort remains
    quantum-boundary, and request IDs may safely collide with internal dummy IDs;
  - shutdown refuses to destroy Gloo from another thread if the event loop did not
    stop within the configured timeout;
  - execution tracing is opt-in, validates per-rank 16-forward order, and can be
    drained for integration validation without imposing steady-state timing overhead;
  - LoadSnapshot and frontend aggregation expose useful/raw/dummy work, preemption,
    queue delay, stage latency, and coordination counters.
- Re-ran the exact mandated command on the finalized control-plane checkout with all
  four proxy variables and elevated permissions:
  `pip install -v -e .` completed successfully at 09:04 UTC. Post-install tests
  remained 52/52, the installed binary extension reports the new scheduler API, and
  elevated imports of `LLMEngine`, `DeploymentManager`, and the Ray `ModelRunner`
  actor succeeded.
- Verified under elevated execution that this node has eight H200 GPUs and that the
  new frontend/manager/runner modules import successfully.
- Connected to the existing two-node Ray cluster at `10.102.206.14:7789`, confirmed
  16 total GPUs, and verified fail-fast placement-group cleanup after a deliberately
  low-memory smoke attempt.
- Ran a proxy-unset, CPU-only Ray actor smoke test against the live cluster after the
  Phase 1 commit. The real `DecodeCoordinator` completed READY registration,
  broadcast wave 1 to both fake engines, preserved a racing wakeup, then broadcast
  wave 2 to both engines. All temporary actors were killed in `finally`.
- Closed the post-commit ADD legality gap with a placement-aware, exclusive-capacity
  probe that reuses the configured C++ Scheduler/SP policy. Probe state is cached by
  request shape and its logical block pool grows only as far as the request requires.
  The concrete SP4 counterexample (120-token prompt, 16-token completion, two service
  blocks per rank) is now rejected by ADD instead of being accepted and crashing at
  admission. The correction is committed as
  `ac07a54 Enforce exact hierarchical admission capacity`.
- Changed transient admission failures to remain in `WAITING_ADMISSION`: if the
  current placement fails the exact padded-lifetime check or lacks one bootstrap-token
  allocation, its prompt blocks are released and it is retried later.
- Added regressions for illegal exclusive placement, legal distributed placement, and
  temporary bootstrap-capacity deferral. The post-fix CPU suite now passes 55 tests.
- Re-ran the exact requested elevated install at 09:22 UTC with all four proxy
  variables: `pip install -v -e .` completed successfully, reinstalled
  `nanodeploy 0.2.0`, and the post-install extension/API check plus all 55 tests,
  Sequence proxy test, `compileall`, and `git diff --check` passed.
- Began the real single-node DP2SP4/EP8 integration test after all eight local H200
  GPUs became available. The first run at `gpu_memory_utilization=0.60` launched all
  eight model runners and initialized SP/Gloo, then failed cleanly before READY
  because the full DeepSeek-V3 weights exceeded that KV-cache budget
  (`num_gpu_blocks=-4869`). Deployment cleanup removed the placement groups and
  actors without touching unrelated processes.
- Re-ran DP2SP4 at `gpu_memory_utilization=0.90`. All eight ranks reported a positive,
  consistent KV capacity of 5163 blocks and both DP-local engine actors launched,
  but endpoint initialization failed inside `LocalExecutor`:
  `worker.init_rpc_endpoint.remote(server_info)` was rejected by Ray's client-side
  signature check as `TypeError: too many positional arguments`.
- Ruled out stale Ray worker reuse with both a deployment-unique runtime token and a
  full `ray stop --force`/fresh-head restart. Exact in-actor diagnostics showed the
  cause: Ray 2.51 deserialized each nested `ModelRunner` handle with a generic
  `(**kwargs)` signature even though the driver-side handle retained the concrete
  method signatures. Positional calls were therefore rejected before task dispatch.
- Fixed LocalEngine-to-ModelRunner calls to use the concrete parameter names for both
  endpoint setup and decode. This is accepted by both concrete and generic
  `(**kwargs)` Ray handle metadata. A CPU regression simulates a kwargs-only nested
  actor handle and checks the complete endpoint/decode invocation.
- Added an opt-in `--hierarchical-execution-trace` example flag that captures the
  integration trace and validates that every global rank sees the same ordered
  wave/quantum set with exactly 16 inner forwards.
- Switched integration testing to the user-specified single-node Ray head at
  `10.102.243.60:8776`. Because launcher-owned daemons are reaped when a normal
  command exits, tests use a persistent
  `ray start --head --port 8776 --block` session. Every GPU test starts with
  `ray stop --force` and a fresh head.
- Completed all three single-node 8-H200 topology smoke tests with DeepSeek-V3:
  - DP2SP4/EP8 completed 2/2 requests and 32 useful completion tokens with no
    preemption; all eight ranks had 5163 KV blocks.
  - DP8SP1/EP8 completed 8/8 requests and validated 56 trace records across eight
    ranks and seven globally identical 16-forward steps.
  - DP1SP8/EP8 completed 2/2 requests and validated 16 trace records across eight
    ranks and two globally identical 16-forward steps.
  All deployments cleaned up to 0/8 Ray GPU use and 0 MiB on every local GPU.
- Included `ALL_PROXY` and `all_proxy` in the in-process Ray proxy guard; Ray
  operations now remove all six common proxy variables and restore them afterward.
- Re-ran the exact elevated install at 09:55 UTC with the four required HTTP(S)
  proxy variables: `pip install -v -e .` successfully rebuilt and reinstalled
  `nanodeploy 0.2.0`. The post-install suite passes 56 tests, the Sequence proxy
  check and `compileall` pass, and the installed scheduler exposes `admit` and
  `plan_decode`.
- Committed the nested Ray handle repair, all-proxy Ray guard, trace CLI, and
  regression as `816aecb Fix nested Ray worker calls and validate GPU traces`.
- Added and ran `examples/hierarchical_lifecycle.py` against DP2SP4/EP8 in one
  continuous deployment:
  - a 5-token request owned by DP0 completed in one global step; ranks 0-3 were
    `real_or_mixed`, ranks 4-7 were `all_control_dummy`, and final-quantum overrun
    committed exactly five useful tokens;
  - after completion, 0.5 seconds of polling/load reporting produced no new quantum
    or trace, proving global idle pause;
  - two 32-token round-robin requests were owned by different engines, overlapped in
    at least one global step, completed after three globally identical steps, and
    produced 24 valid rank trace records;
  - a 128-token request was aborted during a later in-flight quantum, returned
    `abort_pending`, emitted one `ABORTED` terminal, committed only the 16-token
    prefix from the prior completed quantum, and produced 16 valid trace records.
  Every lifecycle trace passed the 8-rank, ordered 16-forward validation.
- Stopped the local Ray head after the lifecycle run as requested. No raylet or GCS
  process remains, and all eight GPUs report 0 MiB used.
- Committed the reproducible lifecycle validator as
  `21f2998 Add hierarchical lifecycle validation`.

## Current Phase

The CPU/control-plane implementation, exact ADD legality correction, nested Ray
handle fix, and all three single-node topology smoke cases are complete. Next
actions:

1. Run the remaining Phase 1 GPU cases: KV-pressure preemption/re-admission and one
   representative CUDA Graph configuration. The single/dual-engine, finish/reuse,
   idle pause/wakeup, final overrun, and in-flight abort lifecycle cases now pass.
2. Run the Phase 2 multi-node liveness/fail-fast cases once a second 8-GPU node is
   joined to `10.102.243.60:8776` with the current editable installation.
3. Run matched legacy/hierarchical benchmarks at least three times per selected
   topology, compare medians against the proposal thresholds, and only then make
   profiling-driven optimizations.

## Known Constraints

- All future GPU commands require elevated permissions.
- Ray operations must run with proxy variables unset.
- For every GPU integration test, run `ray stop --force`, then start a fresh local
  head at `10.102.243.60:8776`; under the command runner the start command must use
  `--block` in a persistent session. Reconfirm GPU ownership and Ray state before
  each test.
- The current node has eight H200 GPUs and all were free after the final test. Ray
  is currently stopped.
  Never signal a PID until GCS proves that it belongs to a completed deployment.
- The local 8776 cluster currently has only one 8-GPU node, so Phase 2 cannot run
  until another correctly installed node joins it.
- The remote Ray node has eight free GPUs and sees the checkout, but its installed
  extension predates the new scheduler API. The required remote command
  `pip install -v -e .` was attempted with all four mandated proxy variables and
  failed before compilation because `127.0.0.1:15409` is not listening on that
  remote host. A follow-up read-only probe confirmed no remote proxy environment
  variables and no listener on port 15409.
- An elevated socket check confirmed that the working proxy on the current node is
  bound only to `127.0.0.1:15409`, so it is not directly reachable from the remote
  Ray node.
- Do not substitute `--no-build-isolation`, copy extension binaries, or otherwise
  bypass the user's required editable-install procedure.
- The Ray dashboard/metrics exporter agent does not become available in these fresh
  heads and emits warnings. GCS, placement, collectives, execution, cleanup, and
  trace validation all succeeded; no dependency was added to silence monitoring.
- Pre-existing user changes (`.gitignore`, `Agents.md`, and proposal documents) must
  remain untouched.
