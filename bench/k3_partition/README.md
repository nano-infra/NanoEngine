# K3 partitioning experiments

Run `python bench/k3_partition/analyze_model_costs.py` to reproduce the
shape-derived FLOP, cache, and expanded-K/V traffic values stored in
`results/model_costs.json`.

The collective microbenchmark uses BF16 hidden-state tensors with K3 hidden
width 7168. It measures NCCL all-reduce, reduce-scatter, all-gather, and
all-to-all on one NVLink-connected B300 node at world sizes 2 and 4.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  bench/k3_partition/benchmark_collectives.py \
  --output bench/k3_partition/results/collectives_tp2.csv
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  bench/k3_partition/benchmark_collectives.py \
  --output bench/k3_partition/results/collectives_tp4.csv
MPLBACKEND=Agg python bench/k3_partition/plot_collectives.py
```

Each point uses 5 warm-up and 20 measured iterations. CUDA events measure the
average device-side duration. `logical_payload_gbps` divides the complete
logical hidden-state tensor size by latency; it is an application-facing rate,
not topology-adjusted NCCL bus bandwidth. The local experiment characterizes
intra-node communication only and does not claim to measure the target
DP2/attention-TP8/EP16 multi-node topology.
