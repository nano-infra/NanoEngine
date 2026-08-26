# Scheduler overhead profiler

`profile_scheduler_scalability.py` measures the unmodified production C++
`Scheduler.schedule()` implementation without starting Ray or using GPUs. One
logical node represents eight logical GPUs.

The default sweep covers:

- logical nodes: 4, 8, 16, 32 (32--256 logical GPUs);
- active batch per GPU: 32, 64, 128;
- no SP, fixed SP8, dynamic 1% SP8, and dynamic 5% SP8.

The default decode quantum is 16 steps, matching the production serving
configuration. Override it with `--loop-count` when evaluating another
configuration.

Dynamic scenarios use NanoDeploy's production DeepSeek-V3 bucket policy. Short
requests contain 1,024 tokens and select SP1; long requests contain 428,033
tokens and therefore fall in the policy's SP8 interval
(`428033--1048576`). `dynamic_sp8_1pct` and `dynamic_sp8_5pct` control the
fraction of these long requests. `no_sp` uses a DP-only topology, while
`fixed_sp8` forces every request to use SP8.

Run the full sweep with:

```bash
python3 scripts/scheduler_overhead/profile_scheduler_scalability.py
```

Results are written as JSON and CSV beneath
`bench_logs/scheduler_overhead/<UTC timestamp>/`. Each case reports two
separate scheduler costs:

- bulk admission, including GPU selection and SP-degree/placement decisions
  for all waiting requests in the configured batch;
- steady-state decode scheduling after those requests are running.

Scheduler construction, request object creation and queue insertion, Ray/RDMA
communication, and GPU execution are excluded. Bulk admission is repeated ten
times by default; use `--admission-iterations` to change the sample count.

For a quick smoke run:

```bash
python3 scripts/scheduler_overhead/profile_scheduler_scalability.py \
  --logical-nodes 1 \
  --batch-sizes 2 \
  --admission-iterations 1 \
  --warmup-iterations 1 \
  --iterations 3 \
  --output-dir /tmp/nanodeploy-scheduler-overhead-smoke
```

The context lengths affect CPU block allocation and placement only; no
attention kernel is executed. Custom lengths can be supplied through
`--short-context-len` and `--long-context-len`, but dynamic lengths must remain
inside the SP1 and SP8 intervals of the production policy.

## Hierarchical scheduler counterpart

`profile_hierarchical_scheduler_scalability.py` runs the matching workload
through the production hierarchical control-plane contract:

1. `RequestRouter` performs global least-batch routing and frontend admission
   planning against real empty `LocalScheduler` load snapshots;
2. every destination validates and commits the Router-selected reservations
   with `LocalScheduler.commit_planned_batch()`;
3. every LocalScheduler executes steady-state `admit`, load snapshots,
   `plan_decode`, first-forward marking, and `postprocess` with contract-valid
   fake worker results.

It does not start Ray or request a GPU. Sequence construction, fake worker
result construction, transport, RDMA, ModelRunner, CUDA, and kernels are
excluded from the timers.

The workload is directly aligned with the legacy profiler. For fixed SP8, one
logical node owns one SP8 LocalScheduler and `BS/GPU=32` means 256 requests in
that LocalScheduler, balanced to 32 master requests per rank. For no-SP, one
logical node owns eight SP1 LocalSchedulers with 32 requests each. Both cases
therefore contain 256 total requests per logical node.

Run a small contract smoke test with:

```bash
CUDA_VISIBLE_DEVICES= python3 \
  scripts/scheduler_overhead/profile_hierarchical_scheduler_scalability.py \
  --logical-nodes 1 \
  --batch-sizes 1 \
  --scenarios fixed_sp8 \
  --admission-iterations 1 \
  --warmup-iterations 1 \
  --iterations 3 \
  --output-dir /tmp/nanodeploy-hierarchical-scheduler-overhead-smoke
```

Run the matrix matching the legacy profiler with:

```bash
CUDA_VISIBLE_DEVICES= python3 \
  scripts/scheduler_overhead/profile_hierarchical_scheduler_scalability.py \
  --logical-nodes 4,8,16,32 \
  --batch-sizes 32,64,128 \
  --scenarios no_sp,fixed_sp8,dynamic_sp8_1pct,dynamic_sp8_5pct \
  --admission-iterations 10 \
  --warmup-iterations 10 \
  --iterations 100
```

The output files are `hierarchical_scheduler_overhead.json` and
`hierarchical_scheduler_overhead.csv`. Important fields are:

- `admission_ms.router_plan_receipt_ms`: centralized RequestRouter planning,
  batching, ownership, and immediately-ready receipt CPU time;
- `admission_ms.local_commit_critical_ms`: slowest LocalScheduler planned
  commit, assuming local commits execute on separate nodes;
- `local_phase_ms`: pooled LocalScheduler phase samples;
- `modelled_parallel_quantum_ms`: maximum local scheduler CPU time for each
  logical quantum, used as the ideal distributed critical path;
- `aggregate_local_cpu_ms_per_global_quantum`: sum of all local CPU work, which
  is useful for capacity accounting but is not latency;
- `weak_scaling_efficiency_vs_smallest_node`: aggregate local quantum
  throughput relative to ideal scaling from the smallest matching case.

The harness executes LocalSchedulers serially on one CPU host. Values named
`modelled_parallel_*` are derived from the maximum local cost and do not include
multi-host synchronization or transport. NanoDeploy only whitelists complete
hierarchical deployments at 1, 2, and 4 logical nodes. Larger counts are
labelled `logical_independent_local_scheduler_replica_model`; they measure CPU
algorithm scaling and must not be presented as deployable topologies.

For a legacy/hierarchical comparison, match `scenario`, `logical_nodes`,
`batch_size_per_gpu`, contexts, loop count, warmup, and iterations. Compare
legacy `admission_mean_ms` with the hierarchical admission phase breakdown, and
legacy steady-state `mean_ms` with hierarchical
`modelled_parallel_quantum_ms.mean`. Keep the latter labelled as a modelled
critical path rather than measured distributed wall time.
