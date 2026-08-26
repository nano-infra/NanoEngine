# Decentralized CPU scalability profilers

This directory measures the hierarchical control plane without requesting a
GPU resource or creating a CUDA context. The three profilers isolate:

1. `profile_router_cpu.py`: production `RequestRouter` least-batch planning,
   SP8 reservations, batching, and immediately-ready receipt processing;
2. `profile_frontend_ray_cpu.py`: Router-to-LocalEngine ZMQ Sequence ingress
   and LocalEngine-to-Router Ray `FrontendEventBatch` return;
3. `profile_local_scheduler_ray_cpu.py`: production `LocalScheduler` planned
   commit, steady-state admit, SP8 load snapshots, decode planning, and
   postprocessing with contract-valid fake worker results.

All Ray actors declare `num_gpus=0` and validate that Ray assigned no GPU
accelerator IDs. Sequence construction, actor startup, model weights,
ModelRunner, worker collectives, DLSLime/RDMA, CUDA, and GPU kernels are outside
the measured scheduler scope.

## Three-node topology

The current cluster has three physical nodes. Router and frontend profiles run
the physical `1,2,3` node matrix. NanoDeploy does not have a deployable DP3SP8
topology: only DP1SP8, DP2SP8, and DP4SP8 are whitelisted. Therefore the
three-node LocalScheduler case launches the first three independent engines of
a DP4SP8 configuration and labels the record
`topology_scope=partial_dp4_cpu_scaling_only`. It measures local scheduler CPU
scaling, but it is not a claim that DP3SP8 can be deployed. One- and two-node
records are complete production topologies.

## Environment

The same current `NanoDeploy-July` Python package and compiled extension must be
visible on every Ray node. Do not use the isolated e124 transport benchmark's
`PYTHONPATH` for these profilers.

Before starting or joining Ray on every node:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export CUDA_VISIBLE_DEVICES=
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTHONPATH=/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-July:${PYTHONPATH:-}
```

The supplied cluster address is `10.102.252.174:6380`. The scripts connect to
that cluster; they do not start or stop it. `SLIME_VISIBLE_DEVICES`,
`SLIME_GID_INDEX`, and `SLIME_QP_NUM` are not needed because this suite does not
run DLSLime or RDMA.

## Run

From the repository root, first run the small end-to-end smoke matrix:

```bash
bash scripts/decentralized_scalability/run_3node_matrix.sh smoke
```

Then run the complete strong/weak matrix:

```bash
bash scripts/decentralized_scalability/run_3node_matrix.sh full
```

The wrapper defaults to `RAY_ADDRESS=10.102.252.174:6380`. Override the address
or output location when needed:

```bash
RAY_ADDRESS=10.102.252.174:6380 \
OUTPUT_ROOT=/tmp/nanodeploy-decentralized-cpu-full \
bash scripts/decentralized_scalability/run_3node_matrix.sh full
```

Each component writes JSON and CSV files beneath its output directory. The JSON
metadata records selected nodes, actor placement, requested/assigned GPU
resources, measurement boundaries, and all arguments.

## Interpretation

- Router throughput is a single-driver ceiling; compare
  `throughput_retention_vs_one_engine` rather than expecting linear speedup.
- Frontend weak scaling should keep per-message receipt/event latency stable
  while aggregate item throughput rises.
- LocalScheduler weak scaling uses fixed batch per engine. Its
  `weak_scaling_efficiency_vs_one_node` is aggregate scheduler quantum
  throughput divided by `N * one-node throughput`.
- Strong LocalScheduler scaling treats `--batch-sizes` as the total request
  batch divided across the selected engines. Weak scaling treats it as the
  batch on every engine.

Do not use the one-iteration smoke numbers as performance results; they only
validate placement, imports, contracts, output, and zero-GPU allocation.
