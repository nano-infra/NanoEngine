# Bug: SIGSEGV in cudaGraphLaunch — DeepEP clean_low_latency_buffer placement

## Symptom

All model workers on the remote node crash with a segfault during the first decode
step after prefill:

```
Fatal Python error: Segmentation fault
  File "model_runner.py", line N in run_model   ← graph.replay()
    at::cuda::CUDAGraph::replay()
    cudaGraphLaunch
```

Two failure modes were observed depending on where `transition_to_low_latency()` was
called:

1. **Without any call before graph replay** → SIGSEGV because the RDMA buffer is dirty.
2. **With the call placed right before `graph.replay()`** → SIGSEGV on the *receiving*
   nodes due to a race between NVSHMEM RDMA writes and CUDA graph execution.

## Root cause

### Why clean is needed at all

DeepEP uses a shared RDMA buffer for both normal (prefill, high-throughput) dispatch and
low-latency (decode) dispatch.  The low-latency kernels expect certain regions of the
RDMA buffer to be zero-initialised on entry — they are used as arrival flags
(`ready_flag == 0` means "no token yet").

After a normal dispatch/combine cycle (prefill), those regions contain non-zero values
left by the normal-mode protocol.  The first `low_latency_dispatch` kernel would read
those stale flags, misinterpret them as "token already arrived", and dereference a
garbage pointer → SIGSEGV.

`clean_low_latency_buffer` zeros exactly those two regions.

### Why placing it right before `graph.replay()` is also wrong

`clean_low_latency_buffer` launches a single CUDA kernel that:

```c++
// 1. barrier — all EP ranks synchronise via NVSHMEM IBGDA
nvshmemx_barrier_all_block();
// 2. memset the two dirty regions to 0
for (i ...) clean_0[i] = 0;
for (i ...) clean_1[i] = 0;
// 3. barrier again — make sure every peer sees the clean state
nvshmemx_barrier_all_block();
```

`nvshmemx_barrier_all_block()` is a **device-side NVSHMEM collective**.  It fires IBGDA
(InfiniBand GPU Direct Async) write operations to symmetric memory on every peer rank.
These RDMA writes travel over the network asynchronously.

When `transition_to_low_latency()` is placed immediately before `graph.replay()`:

```
rank 179:  clean kernel → graph.replay() starts
rank 183:  clean kernel → graph.replay() starts
                 ↕
  RDMA writes from rank 179's barrier arrive at rank 183
  while rank 183 is already inside graph.replay()
```

The arriving RDMA writes land in the NVSHMEM symmetric address space of rank 183.  If
those addresses overlap with, or share the IBGDA channel state used by, the
`low_latency_dispatch` kernels captured inside the graph, the graph execution is
corrupted → SIGSEGV in `cudaGraphLaunch` on the receiving node.

This explains why only the *remote* node (183) crashed: it was the RDMA target of the
barrier writes issued by the local node (179).

## Fix

Move `transition_to_low_latency()` to the **end of the prefill forward pass** inside
`run_model`, guarded by `if is_prefill:`.

```python
# model_runner.py — run_model(), prefill branch
hidden = self.model(input_ids, positions)
if is_prefill:
    ExpertContext.get_instance().transition_to_low_latency()   # ← here
return self.model.compute_logits(hidden)
```

At this point the clean kernel and both of its `nvshmemx_barrier_all_block()` calls
execute on the compute stream as part of the prefill step.  Before the engine advances
to decode, the Ray executor collects results from all workers (a full round-trip), which
acts as an implicit cross-rank synchronisation.  By the time `graph.replay()` is invoked
on any rank, the IBGDA writes from the barrier have already settled on all peers.

The misplaced call that was sitting just before `graph.replay()` is removed.

## Call-site summary

| Location                                       | When called                        | Action                                                                                                                      |
| ---------------------------------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `experts.py` `_compute_prefill_ep`             | start of each prefill expert layer | `transition_to_normal()` — marks mode as normal                                                                             |
| `experts.py` `_compute_decode_ep`              | start of each decode expert layer  | `transition_to_low_latency()` — no-op in graph mode (mode already "low_latency"); cleans in eager mode if mode was "normal" |
| `model_runner.py` `run_model` (prefill branch) | once, after full prefill forward   | `transition_to_low_latency()` — cleans the buffer on the prefill stream, before decode graph replay                         |

## Files changed

- `NanoDeploy/nanodeploy/worker/model_runner.py`
