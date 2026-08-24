# Ray Decode Control RPC Scalability Profiler

`profile_ray_rpc_scalability.py` measures the Ray portion that remains in the
production DLSlime decode path:

1. The driver submits `run.remote([], False, True, send_timestamp)` to every
   logical worker.
2. Each worker immediately returns `BS/GPU * loop_count` sampled token IDs and
   its completion timestamp.
3. The driver waits for and deserializes all results with `ray.get`.

The profiler uses real Ray actors but no ModelRunner, model weights, GPUs, or
DLSlime endpoints. All logical workers are colocated in a fresh isolated Ray
instance. Consequently, it measures single-node Ray actor submission,
serialization, result transfer, and fan-in scaling; it does not simulate
multi-node Ray or RDMA bandwidth. Ray runtime sockets and logs are placed under
a short, unique `/tmp/nd-ray-*` directory to avoid the Linux Unix-socket path
length limit.

Run the production-sized sweep:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
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
