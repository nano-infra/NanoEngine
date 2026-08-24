# Scheduler overhead profiler

`profile_scheduler_scalability.py` measures the unmodified production C++
`Scheduler.schedule()` implementation without starting Ray or using GPUs. One
logical node represents eight logical GPUs.

The default sweep covers:

- logical nodes: 4, 8, 16, 32 (32--256 logical GPUs);
- active batch per GPU: 32, 64, 128;
- no SP, fixed SP8, dynamic 1% SP8, and dynamic 5% SP8.

Run the full sweep with:

```bash
python3 scripts/scheduler_overhead/profile_scheduler_scalability.py
```

Results are written as JSON and CSV beneath
`bench_logs/scheduler_overhead/<UTC timestamp>/`. Setup, request admission,
Ray/RDMA communication, and GPU execution are explicitly excluded from the
reported steady-state decode latency.

For a quick smoke run:

```bash
python3 scripts/scheduler_overhead/profile_scheduler_scalability.py \
  --logical-nodes 1 \
  --batch-sizes 2 \
  --warmup-iterations 1 \
  --iterations 3 \
  --output-dir /tmp/nanodeploy-scheduler-overhead-smoke
```

The 64- and 512-token synthetic contexts keep the CPU benchmark compact while
exercising the production SP1 and SP8 placement paths. They are not intended to
model attention execution time; custom lengths can be supplied through
`--short-context-len` and `--long-context-len`.
