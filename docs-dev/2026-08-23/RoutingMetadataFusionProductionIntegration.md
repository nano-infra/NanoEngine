# Routing Metadata Fusion Production Integration

Date: 2026-08-23

## Status

The fused SP routing-metadata update is now used by the actual CUDA Graph
inference path. The production PyTorch/unfused helpers were removed; there is
no serving fallback or compatibility branch.

Implementation commit:

```text
3215d167fc97d164f5c09ba6a66dcc36f08455ee
feat: fuse production SP graph metadata updates
```

## Implementation

`nanodeploy/kernels/sp_graph_metadata.py` contains the one-launch Triton
kernel and its warmed compiled runner. It performs the full q-destination copy,
writes the actual attention batch scalar, and materializes attention/master
Graph padding.

`nanodeploy/worker/sp_graph_policy.py` calls the fused update after copying the
real metadata prefixes. The former `copy_graph_q_dst_rows`,
`copy_graph_actual_attn_bs`, and `materialize_sp_graph_padding` production
implementations were deleted.

`nanodeploy/worker/model_runner.py` warms the fused kernel after allocating the
persistent full and piecewise CUDA Graph buffers, so Triton compilation is not
on the serving iteration path.

The conservative copy of the complete dummy block-table row remains unchanged.
No C++ source changed, so an extension rebuild was not required.

## Correctness and operation count

The production kernel passed the prototype's 54-case exact oracle matrix. All
nine complete persistent metadata tensors matched the historical implementation
for padding variants, SP ranks 0/1/7, and block-table widths 1/17/1384.

A warmed Issue1 profiler invocation reported one
`update_sp_graph_metadata_kernel` CUDA operation. The historical reference had
ten CUDA operations.

Focused tests passed:

```text
python3 -m pytest \
  tests/test_sp_graph_runtime_overhead.py \
  tests/test_pd_decode_sp_batch_semantics.py -q
18 passed in 2.64s
```

The one-node SP8 dynamic-routing test also passed on eight H200 GPUs:

```text
torchrun --nproc_per_node=8 -m pytest -q \
  tests/test_hao_dynamic_sp_q_routing.py -s
8 ranks passed
```

## Production component benchmark

The benchmark used one H200, 200 warmup calls, seven repeats, and 2,000 calls
per repeat. The representative Issue1 shape was
`(M_actual, N_actual)=(70,76)` padded to `(M_graph, N_graph)=(80,80)`, with
SP=8 and block-table width 1384.

| Case | Scope | Device P50 | Device P95 | Host P50 | Host P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| no padding | routing | 11.188 us | 12.224 us | 11.813 us | 12.436 us |
| Issue1 padded | routing | 10.666 us | 11.907 us | 11.686 us | 12.206 us |
| no padding | all Graph metadata | 174.975 us | 185.417 us | 172.553 us | 185.296 us |
| Issue1 padded | all Graph metadata | 182.366 us | 199.921 us | 187.093 us | 197.836 us |

Against the historical-helper reference measured in Phase 1, routing device
P50 changes from 11.392 to 11.188 us without padding and from 55.756 to
10.666 us for Issue1 padding. The padded routing component is 5.23x faster;
the no-padding path does not regress.

The generated result is retained locally at:

```text
bench_logs/graph_runtime_overhead/metadata_fusion_production.json
```

Its manifest identifies commit `3215d167fc97d164f5c09ba6a66dcc36f08455ee`,
PyTorch `2.10.0+cu129`, CUDA runtime `12.9`, Python `3.12.13`, and an NVIDIA
H200. This is a routing-component result, not a 5.23x end-to-end serving speedup.

## Review

The final review was limited to safety, performance, and logic. No blocking
issues were found. The kernel writes only the same selected destinations as the
removed implementation, uses explicit tensor strides, runs on the active CUDA
stream, and stays at one launch after initialization.
