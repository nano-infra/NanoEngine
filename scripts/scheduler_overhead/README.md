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

The 64- and 512-token synthetic contexts keep the CPU benchmark compact while
exercising the production SP1 and SP8 placement paths. They are not intended to
model attention execution time; custom lengths can be supplied through
`--short-context-len` and `--long-context-len`.
