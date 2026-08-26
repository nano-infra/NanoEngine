# Three-node decentralized CPU scalability results

## Objective

Measure the CPU-only scalability of the hierarchical control plane in three
isolated parts:

1. production `RequestRouter` least-batch planning and receipt processing;
2. Router-to-LocalEngine ZMQ Sequence ingress and LocalEngine-to-Router Ray
   frontend-event communication;
3. production `LocalScheduler` planned commit and steady-state decode
   scheduling.

The experiment intentionally excludes model weights, GPU workers, CUDA,
DLSLime/RDMA, collectives, and model kernels.

## Run configuration

- Run time: 2026-08-26 08:43--08:54 UTC
- Ray address: `10.102.252.174:6380`
- Physical nodes: `10.102.243.60`, `10.102.252.174`, `10.102.98.154`
- Logical topology: 1, 2, and 3 engines, with fixed SP8 per engine
- Scaling modes: strong and weak
- Batch sizes: 32, 64, and 128
- Prompt lengths: 32 and 8000 tokens
- Router trials per case: 5
- Sequence-ingress trials per case: 5
- Ray-event warmup/measured iterations: 20/500
- LocalScheduler warmup/measured iterations: 20/500
- GPU request per actor: 0
- Observed Ray GPU assignment: empty on every actor

Command:

```bash
bash scripts/decentralized_scalability/run_3node_matrix.sh full
```

All correctness checks passed. This includes one-time Sequence transfer,
positive ingress receipts, empty Router ingress flights, no scheduler commit in
the isolated ingress case, valid SP8 load snapshots, equal quantum advancement,
and no scheduler preemption.

## Summary

The table below focuses on the production-like 8000-token prompt. Throughput
ratios use the median measurement for the three-node case divided by the
corresponding one-node measurement.

| Three-node result relative to one node | Batch 32 | Batch 64 | Batch 128 |
|---|---:|---:|---:|
| Router strong-scaling speedup | 1.70x | 1.41x | 1.22x |
| Router weak-scaling throughput retained | 66.1% | 57.9% | 51.2% |
| ZMQ Sequence ingress strong-scaling speedup | 2.00x | 1.75x | 1.62x |
| ZMQ Sequence ingress weak-scaling efficiency | 25.9% | 25.2% | 24.4% |
| Ray frontend-event weak-scaling efficiency | 71.4% | 72.4% | 67.4% |
| LocalScheduler weak-scaling efficiency | 81.5% | 82.2% | 82.3% |
| LocalScheduler strong-scaling global-step speedup | 1.34x | 1.57x | 1.74x |

### Load balancer

The centralized Router is the clearest scaling limit. For an 8000-token
prompt, its weak-scaling throughput changes as follows while the input burst
grows from 4096 to 12,288 requests:

| Batch | One engine | Two engines | Three engines |
|---|---:|---:|---:|
| 32 | 5,844 req/s | 4,584 req/s | 3,864 req/s |
| 64 | 8,606 req/s | 6,267 req/s | 4,986 req/s |
| 128 | 11,180 req/s | 7,561 req/s | 5,727 req/s |

This profiler uses an immediately-ready fake transport, so these numbers
isolate Router submission, SP8 least-batch planning, batching, ownership state,
and receipt processing. They do not include network or scheduler work.

### Router-to-LocalEngine Sequence ingress

With a fixed total of 4096 long-prompt requests, three-node strong-scaling
speedup is 2.00x, 1.75x, and 1.62x for batches 32, 64, and 128. Weak scaling is
poor because all requests are still submitted and polled by one Router driver.

For batch 32, weak-scaling throughput is
`3,352 -> 2,926 -> 2,604 req/s`, while the median of per-trial receipt p99 rises
from `7.43 -> 15.35 -> 22.82 ms`. For batch 64, receipt p99 rises from
`9.46 -> 16.51 -> 27.98 ms`.

Server-side Sequence deserialization remains comparatively stable. For batch
64 it is approximately 0.53 ms per message at both one and three nodes. This
points to centralized submission, polling, in-flight bookkeeping, and queueing
as the primary source of the ingress degradation rather than Sequence
deserialization alone.

Batch-128 receipt-tail measurements are noisy: the five per-trial p99 samples
have roughly 20--25 ms standard deviation. More ingress repeats are required
before treating those tail-latency values as stable.

### LocalEngine-to-Router Ray events

This direction scales substantially better. In weak scaling:

| Events per engine | One node | Two nodes | Three nodes | Three-node p99 |
|---|---:|---:|---:|---:|
| 32 | 46.2k events/s | 72.2k events/s | 98.8k events/s | 1.49 ms |
| 64 | 70.4k events/s | 124.8k events/s | 153.0k events/s | 1.55 ms |
| 128 | 107.1k events/s | 175.9k events/s | 216.6k events/s | 2.17 ms |

The event profiler uses production-shaped `FrontendEventBatch` and
`LoadSnapshot` values and consumes ready results with `ray.wait`.

### LocalScheduler

The independent LocalSchedulers have the best weak-scaling efficiency. With
8000-token prompts:

| Local batch per engine | One node | Two nodes | Three nodes | Efficiency |
|---|---:|---:|---:|---:|
| 32 | 2,603 quantum/s | 4,273 quantum/s | 6,361 quantum/s | 81.5% |
| 64 | 1,657 quantum/s | 2,558 quantum/s | 4,089 quantum/s | 82.2% |
| 128 | 969 quantum/s | 1,600 quantum/s | 2,394 quantum/s | 82.3% |

For fixed-total-batch strong scaling, global scheduler-step speedup at three
nodes is 1.34x, 1.57x, and 1.74x for total batches 32, 64, and 128.

NanoDeploy does not expose a deployable DP3SP8 configuration. The three-node
LocalScheduler cases therefore launch the first three independent engines of a
DP4SP8 configuration and are labelled
`topology_scope=partial_dp4_cpu_scaling_only`. They measure local scheduler CPU
scaling, not production DP3SP8 deployability. The one- and two-node cases use
complete production topologies.

## Conclusion

The isolated results point to the centralized RequestRouter and its
Router-to-LocalEngine ingress loop as the first scalability target. The local
schedulers retain about 82% weak-scaling efficiency, and the Ray frontend-event
return path retains about 67--72%. The ingress path retains only 73--78% of its
one-node absolute throughput at three nodes, corresponding to approximately
24--26% efficiency relative to ideal linear weak scaling.

A follow-up steady-state experiment should use distributed load generators or
sharded Routers with a fixed-duration, rate-controlled workload. The current
weak-scaling ingress experiment deliberately grows a single-driver request
burst from 4096 to 12,288 requests, which is useful for locating the centralized
bottleneck but is not an end-to-end serving-capacity measurement.

## Raw artifacts

- [Router JSON](../../bench_logs/decentralized_cpu_scalability/20260826_084337/router/router_cpu_scalability.json)
- [Router CSV](../../bench_logs/decentralized_cpu_scalability/20260826_084337/router/router_cpu_scalability.csv)
- [Sequence ingress JSON](../../bench_logs/decentralized_cpu_scalability/20260826_084337/frontend/frontend_ingress_cpu_scalability.json)
- [Sequence ingress CSV](../../bench_logs/decentralized_cpu_scalability/20260826_084337/frontend/frontend_ingress_cpu_scalability.csv)
- [Ray frontend-event JSON](../../bench_logs/decentralized_cpu_scalability/20260826_084337/frontend/frontend_events_cpu_scalability.json)
- [Ray frontend-event CSV](../../bench_logs/decentralized_cpu_scalability/20260826_084337/frontend/frontend_events_cpu_scalability.csv)
- [LocalScheduler JSON](../../bench_logs/decentralized_cpu_scalability/20260826_084337/scheduler/local_scheduler_cpu_scalability.json)
- [LocalScheduler CSV](../../bench_logs/decentralized_cpu_scalability/20260826_084337/scheduler/local_scheduler_cpu_scalability.csv)
