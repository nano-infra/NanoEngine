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
