# Hierarchical sync control-plane optimization progress

- Recorded at: 2026-08-26T08:16:11Z
- Objective: reduce repeated LocalScheduler load traversal and overlap the
  required LocalEngine leader consensus with synchronous decode planning.
- Scope: pure synchronous hierarchical scheduling only. The 16-forward
  quantum, worker execution path, routing policy, and result semantics remain
  unchanged.

## Implemented

- `LocalScheduler.plan_decode()` now accumulates active master, receiver, and
  dispatched-token load while building its existing frozen per-rank batch.
- The resulting `LoadSnapshot` is frozen into `LocalDecodeBatch` and reused by
  the LocalEngine cached Router load and pre-execute qdiag record.
- Active event-loop iterations no longer perform the former loop-head and
  post-admit full load refreshes. Idle ingress publication and the
  postprocess-state refresh remain.
- LocalEngine consensus is split into asynchronous start and explicit finish.
  The single Gloo `MAX all_reduce` starts after admission, overlaps
  `plan_decode()` and cached snapshot publication, and is always waited before
  executor submission.
- The existing wave/quantum min/max validation and global unfinished OR remain
  in the same collective.
- qdiag now distinguishes exposed `consensus_wait_ms`,
  `consensus_overlap_window_ms`, and launch-to-completion
  `consensus_total_ms`; the two-node analyzer reports all three.

## CPU/control-plane validation

- `python3 -m pytest tests/test_hierarchical_contract.py -q`: 37 passed.
- `python3 -m pytest tests/test_local_executor_result_path.py
  tests/test_analyze_2node_qdiag_ab.py -q`: 4 passed before the analyzer
  extension; the combined post-extension CPU run is 41 passed.
- Elevated `python3 -m pytest tests/test_hierarchical_control_plane.py -q`:
  51 passed, including the explicit event-loop ordering test.
- The new ordering test verifies
  `consensus start -> plan -> frozen refresh -> consensus wait -> executor`.
- Relevant Python compilation and `git diff --check` pass.

## Stop boundary

No GPU serving experiment or benchmark has been run. After final CPU review and
commit, the next required evidence is a same-commit two-node sync qdiag run on
the reinstalled DLSlime build. Per user instruction, record this checkpoint and
stop before starting that GPU experiment.
