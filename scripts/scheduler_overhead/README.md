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
2. every destination stages its requests before the measured region, then
   validates and commits the Router-selected reservations with
   `LocalScheduler.commit_planned_batch()`;
3. every LocalScheduler executes steady-state native C++
   `Scheduler.schedule()` plus the contract bookkeeping (load snapshots,
   first-forward marking, and `postprocess`) with contract-valid fake worker
   results.

It does not start Ray or request a GPU. Sequence construction, fake worker
result construction, transport, RDMA, ModelRunner, CUDA, and kernels are
excluded from the timers.

The timing contract is explicit: centralized admission measures C++
`Scheduler.schedule()` with its waiting queue already populated; decentralized
admission stages the equivalent waiting queues first, then measures the local
planned commit (including native KV allocation) on the slowest independent
LocalScheduler. Decode uses one native C++ `Scheduler.schedule()` call on both
architectures. Router planning/receipts and Python contract bookkeeping are
reported separately so they cannot be mistaken for the scheduler-only metric.

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
- `admission_ms.scheduler_admission_critical_ms`: primary scheduler-only
  admission boundary: slowest planned LocalScheduler commit after queue
  staging, assuming local commits execute on separate nodes;
- `admission_ms.modelled_admission_critical_ms`: full Router planning, receipt,
  queue staging, and local commit control-plane path, retained for diagnosis;
- `local_phase_ms`: pooled LocalScheduler phase samples;
- `modelled_parallel_quantum_ms`: maximum native C++ `Scheduler.schedule()` time
  for each logical quantum, used as the ideal distributed scheduler critical
  path;
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

## BS/GPU=128 centralized/decentralized comparison

`profile_bs128_centralized_decentralized.py` runs the fixed per-GPU workload
through both profilers and writes a side-by-side comparison. It uses only the
complete production topology counts `1,2,4` (8, 16, and 32 logical GPUs), the
four scenarios above, and `BS/GPU=128`. Each case measures bulk admission and
steady-state decode; the expanded report therefore contains 48 architecture/
phase result cells. The script is CPU-only and does not start Ray or request a
GPU. Hierarchical critical-path fields remain a modelled parallel path because
the underlying harness executes LocalSchedulers serially.

Run the comparison with:

```bash
python3 scripts/scheduler_overhead/profile_bs128_centralized_decentralized.py \
  --logical-nodes 1,2,4 \
  --scenarios no_sp,fixed_sp8,dynamic_sp8_1pct,dynamic_sp8_5pct \
  --batch-size-per-gpu 128 \
  --admission-iterations 10 --warmup-iterations 10 --iterations 100 \
  --output-dir bench_logs/scheduler_overhead/<run>/rjob0_cpu
```

The output consists of `comparison_bs128.json`, the raw side-by-side CSV,
`result_cells.csv` (one row per expanded result cell), `report.html`, and a
short `README.md`. Transport, Ray, RDMA, consensus, and GPU execution are not
part of this primary comparison and must be reported separately.

To match a large logical-scale table such as the 256-GPU row, pass
`--allow-modelled-topologies` and use logical node count 32 (32 x 8 = 256
logical GPUs). This is an independent LocalScheduler CPU model, not a real
32-node deployment; the generated record carries that topology scope.

### Cross-host comparison

After two completed runs, `compare_bs128_hosts.py` validates that their
matrices match and writes per-case host ratios (`host_b / host_a`) plus median
and range summaries:

```bash
python3 scripts/scheduler_overhead/compare_bs128_hosts.py \
  --host-a bench_logs/.../rjob0_cpu \
  --host-b bench_logs/.../rjob3_cpu \
  --label-a rjob0 --label-b rjob3 \
  --output-dir bench_logs/.../host_comparison_rjob0_rjob3
```

The comparison report keeps 3-node results out of scope and preserves the
large-scale topology caveat above.

## BS/GPU=128 mixed admission + decode comparison

`profile_bs128_mixed_admission_decode.py` adds the workload that combines a
running decode population with a newly arriving admission population.  It
keeps `BS/GPU=128` requests in decode, advances them through one decode
quantum, and only then injects the admission cohort.  The admission-to-decode
ratio is defined against the already-running decode cohort and is rounded per
SP8 LocalScheduler:

- `dynamic_sp8_1pct`: 31 new admissions per 1,024 decoding requests (about
  3:100);
- `dynamic_sp8_5pct`: 10 new admissions per 1,024 decoding requests (about
  1:100).

The default matrix is exactly `1,2,4,32` logical nodes × the two dynamic
policies.  Nodes 1/2/4 are complete topology CPU models; node 32 is an
independent LocalScheduler replica model.  The central and decentralized
primary admission metrics use matching scheduler-only boundaries, while
Router planning/receipts and Python contract phases are retained as
diagnostics.  The report records both the target ratio and the integer ratio
actually used in each case.

Run it remotely (after installing the repository on the target host) with:

```bash
CUDA_VISIBLE_DEVICES= \
python3 scripts/scheduler_overhead/profile_bs128_mixed_admission_decode.py \
  --logical-nodes 1,2,4,32 \
  --scenarios dynamic_sp8_1pct,dynamic_sp8_5pct \
  --admission-iterations 10 --warmup-iterations 10 --iterations 100 \
  --output-dir bench_logs/scheduler_overhead/<run>/mixed_bs128
```

The output is `mixed_admission_decode.json`,
`mixed_admission_decode.csv`, `report.html`, and a run `README.md`.  The HTML table shows
centralized/decentralized admission mean, decode mean, decode P99, request
counts, and the measured ratio; JSON additionally contains aggregate local
CPU, Router, and full control-plane diagnostic timings.  This mixed profiler requires the rebuilt
native `commit_planned_sequences` extension from the C++ optimization; it
fails fast instead of silently falling back to the old Python commit path.
