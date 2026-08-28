# Decentralized control-plane profiler

`profile_decentralized_control_plane.py` is a dedicated CPU-only profiler for
the hierarchical scheduler. It does not import or invoke the centralized
Ray/DLSLime benchmark.

It measures two production-shaped paths:

1. the asynchronous eight-`int64` Gloo consensus between LocalEngine leaders,
   including overlap and exposed wait;
2. one persistent Ray `FrontendEventBatch` flight per LocalEngine, immediate
   re-arm of ready engines, and batched receipt processing by the production
   `RequestRouter` using the same ready-flight grouping as `LLMEngine`.

The actors request `num_gpus=0`. ModelRunner, CUDA, kernels, DLSLime/RDMA,
Router-to-LocalEngine request ingress, and worker collectives are excluded.

## Environment

Ray head and workers must load the same current `NanoDeploy-July` checkout and
compiled extension. Do not use the frozen e124 isolated `PYTHONPATH` for this
profiler.

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH=/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-July
export GLOO_SOCKET_IFNAME=<CONTROL_INTERFACE>
cd /tmp
```

Neither the `SLIME_*` variables nor GPU approval are required for this
CPU-only profiler.

## Two-node run

Start or join a dedicated two-node Ray cluster first, then run:

```bash
python3 /mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-July/scripts/decentralized_control_plane/profile_decentralized_control_plane.py \
  --ray-address <HEAD_IP>:<GCS_PORT> \
  --node-counts 1,2 \
  --components consensus,events \
  --batch-sizes 32,64,128 \
  --event-mixes load,mixed \
  --scaling-modes strong,weak \
  --overlap-work-ms 0,0.25,1 \
  --warmup-iterations 20 \
  --iterations 500 \
  --output-dir /tmp/nanodeploy-decentralized-control-plane-two-node
```

## Four-node run

```bash
python3 /mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-July/scripts/decentralized_control_plane/profile_decentralized_control_plane.py \
  --ray-address <HEAD_IP>:<GCS_PORT> \
  --node-counts 1,2,4 \
  --components consensus,events \
  --batch-sizes 32,64,128 \
  --event-mixes load,mixed \
  --scaling-modes strong,weak \
  --overlap-work-ms 0,0.25,1 \
  --warmup-iterations 20 \
  --iterations 500 \
  --output-dir /tmp/nanodeploy-decentralized-control-plane-four-node
```

Use `--consensus-straggler-delays-ms 0,0.5,2,5` or
`--event-straggler-delays-ms 0,1,5` for controlled late-engine experiments.
Use repeated `--node-ip` arguments to make physical node order explicit.

Outputs:

- `decentralized_control_plane.json`
- `decentralized_control_plane.csv`

For consensus, the paper-facing critical-path metric is
`consensus_exposed_wait_critical_ms_*`; divide by the 16-step quantum only
when reporting a per-decode-step value. For frontend events, report both
`critical_flight_age_ms_*` and `router_cpu_ms_per_global_quantum_*`; the Ray
flight and Router CPU costs must not be blindly summed when they overlap.
The reported payload byte count is an untimed representative Python-pickle
size, not a claim about Ray's physical wire-byte accounting.
