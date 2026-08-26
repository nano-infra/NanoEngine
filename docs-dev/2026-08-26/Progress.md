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

## Two-node rate-50 launch command (not run)

This is the full non-smoke validation: two interleaved H/C pairs, four runs in
the order `hierarchical -> central -> hierarchical -> central`. Each run sends
15,000 requests at rate 50 for 300 seconds. The implementation under test is
commit `f9fbe24`; the later Progress-only commit does not change runtime code.

Run from the Ray head/driver node:

```bash
cd /mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-July

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export SLIME_VISIBLE_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export SLIME_GID_INDEX=3
export SLIME_QP_NUM=4
export NANODEPLOY_HIER_RESULT_FASTPATH=1
export NANODEPLOY_HIER_WORKER_TRANSPORT=ray

RAY_ADDR=10.102.243.60:6380 \
MASTER_ADDR=10.102.243.60:29500 \
RATE=50 \
DURATION_SECONDS=300 \
NUM_REQUESTS=15000 \
SMOKE=0 \
CONTINUE_ON_ERROR=0 \
RUN_ORDER="hierarchical central hierarchical central" \
RUN_TAG="sync_control_overlap_rate50_$(date -u +%Y%m%d_%H%M%S)" \
/bin/bash scripts/run_2node_rate40_qdiag_ab.sh
```

The runner writes all artifacts under `bench_logs/$RUN_TAG/`. On success,
`run_manifest.tsv` must contain four successful rows. It then automatically
generates `comparison.json` and `report.html`; the new hierarchical qdiag rows
must contain `consensus_wait_ms`, `consensus_overlap_window_ms`, and
`consensus_total_ms`.

The exact command was validated with `DRY_RUN=1` at 2026-08-26T08:22:46Z. It
expanded to four stages with the expected DP2/SP8, batch-192, loop-count-16,
qdiag, address, rate, duration, and request-count arguments. No Ray connection
or GPU workload was started.
