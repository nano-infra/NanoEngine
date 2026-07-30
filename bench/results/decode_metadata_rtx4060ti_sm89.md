# Decode Metadata Transport Decision

Issue: #231
Baseline commit: `22358e3988d5509bd6810bb6e92741ef6a5f5f48`

## Environment

- GPU: NVIDIA GeForce RTX 4060 Ti 16 GB (SM 8.9)
- Driver: 580.159.03
- CUDA: 12.8
- PyTorch: 2.10.0+cu128
- Warmup/measured iterations: 100/1000 per case

The exact command and complete P50/P99/mean measurements are stored in
`decode_metadata_rtx4060ti_sm89.json`.

## GPU P50

| Batch | Context | Payload | Mapped UVA | Aggregated H2D + kernel | Per-field Torch |
| ----: | ------: | ------: | ---------: | ----------------------: | --------------: |
|     1 |    1024 |   224 B |   17.41 us |                33.79 us |        68.54 us |
|     8 |    1024 |   912 B |   17.54 us |                33.79 us |        68.61 us |
|    16 |    1024 |  1744 B |   18.43 us |                34.78 us |        68.61 us |
|     1 |    4096 |   416 B |   17.60 us |                34.78 us |        70.66 us |
|     8 |    4096 |  2448 B |   17.66 us |                34.78 us |        70.66 us |
|    16 |    4096 |  4816 B |   18.24 us |                34.03 us |        70.66 us |

Mapped UVA reduced median P50 GPU time by 48.3% relative to the aggregated
copy path across the six cases. It also avoided the per-field baseline's
59.76-120.86 us P50 host preparation cost.

## Decision

Proceed with mapped-host UVA for the production decode metadata slab and unpack
kernel. Keep the flat layout compatible with a future aggregated-copy transport
if measurements on another platform overturn this result, but do not add a
runtime production fallback.

The benchmark synchronizes each iteration so the mapped payload can be safely
rewritten. This makes P99 sensitive to host scheduling noise. CUDA Graph
inter-replay idle gap and end-to-end throughput remain integration-stage Nsight
and Qwen3.5 acceptance checks; this microbenchmark only decides the metadata
transport.
