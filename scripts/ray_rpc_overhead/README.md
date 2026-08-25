# Ray RPC Scalability Profiler

`profile_ray_rpc_scalability.py` has two input modes:

1. The default control-only mode submits the empty `dp_seqs` used by the
   production DLSlime decode path.
2. `--ray-sequence-length N` sends `BS/GPU` real NanoDeploy `Sequence` objects
   containing `N` tokens through Ray to reproduce the former Ray input path.

In both modes each worker returns `BS/GPU * loop_count` sampled token IDs and
the driver waits for and deserializes every result with `ray.get`. The timed
round trip includes input serialization and transfer, actor execution, output
serialization and transfer, driver deserialization, and fan-in. Sequence and
output construction happen before timing.

The profiler uses real Ray actors but no ModelRunner, model weights, GPUs, or
DLSlime endpoints. By default all logical workers are colocated in a fresh
isolated Ray instance. It can also connect to a dedicated external Ray cluster
and hard-pin a fixed number of actors per node. Multi-node mode measures real
Ray serialization, network result transfer, and fan-in, but still does not
measure the DLSlime/RDMA data path. Local Ray runtime sockets and logs are
placed under a short, unique `/tmp/nd-ray-*` directory to avoid the Linux
Unix-socket path length limit.

Run the production-sized sweep:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy \
  python3 scripts/ray_rpc_overhead/profile_ray_rpc_scalability.py \
  --logical-workers 32,64,128,256 \
  --batch-sizes 32,64,128 \
  --loop-count 16 \
  --warmup-iterations 10 \
  --iterations 100 \
  --output-dir /tmp/nanodeploy-ray-rpc-overhead
```

The primary metric is `roundtrip_mean_ms`: one complete per-quantum Ray
submission and result collection. `roundtrip_mean_ms_per_decode_step` is that
value divided by `loop_count`; the actual RPC is issued once per quantum, not
once per decode step.

To measure full 20K-token request Sequences through Ray, add:

```bash
--ray-sequence-length 20000
```

This is a Ray Sequence-transfer baseline, not the current DLSlime input path.
The JSON reports the estimated pickle bytes per worker and across all workers.

For a dedicated two-node Ray cluster with eight workers on each host, connect
to the head GCS address and request hard placement:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy \
  python3 scripts/ray_rpc_overhead/profile_ray_rpc_scalability.py \
  --ray-address 10.0.0.1:8776 \
  --logical-workers 16 \
  --workers-per-node 8 \
  --batch-sizes 32,64,128 \
  --loop-count 16 \
  --ray-sequence-length 20000 \
  --warmup-iterations 10 \
  --iterations 100 \
  --output-dir /tmp/nanodeploy-ray-rpc-two-node
```

The run fails if fewer than two live nodes are available or any actor is not
placed on its assigned node. The JSON records every actor's hostname and Ray
node ID so that the topology can be audited with the timing results.

## Ray versus DLSLime

`profile_ray_vs_dlslime.py` compares full-Sequence Ray transport against the
current NanoDeploy path using the production `RPCServerEndpoint` and
`RPCClientEndpoint` classes. The DLSLime mode includes native Sequence
serialization, RDMA write/receive, native deserialization, the empty Ray
control command, token-result transfer through Ray, and `ray.get` fan-in.
All selected batch sizes for one transport run contiguously, so large Ray
Sequence transfers are not interleaved between DLSLime cases. DLSLime endpoint
setup is outside the timed interval, and configured warmup iterations run
before each measured batch-size case.

Run it from the Ray head/driver with the DLSLime driver environment configured:

```bash
export SLIME_VISIBLE_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export SLIME_GID_INDEX=3
export SLIME_QP_NUM=4
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy \
  python3 scripts/ray_rpc_overhead/profile_ray_vs_dlslime.py \
  --ray-address 10.0.0.1:8776 \
  --logical-workers 16 \
  --workers-per-node 8 \
  --batch-sizes 32,64,128 \
  --sequence-length 8000 \
  --loop-count 16 \
  --transports ray,dlslime \
  --warmup-iterations 10 \
  --iterations 100 \
  --output-dir /tmp/nanodeploy-ray-vs-dlslime
```
