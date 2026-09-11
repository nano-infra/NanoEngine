# KDA performance evaluation

- GPU: NVIDIA B300 SXM6 AC
- Torch/CUDA: 2.11.0+cu130 / 13.0
- Shape: H=96, K=128, V=128, BF16
- Timing: 3 warmups, 20 measured iterations (CUDA Events)
- Prefix lengths are aligned down to block size 64.

## Scope

This is a kernel-level evaluation of the exact chunk-prefill and packed-decode
functions called by `FlashInferKda`. A hit means that the recurrent prefix state
already exists; only the fresh suffix is timed. Current production scheduling
disables cross-request prefix caching for cache plans containing GDN/KDA, so the
hit-rate sweep is a controlled what-if evaluation rather than current end-to-end
request behavior.

## Summary

- Cases: 21 prefill, 4 decode.
- Maximum measured speedup: 12.24x at logical length 16384, effective hit 98.83%.

See `kda_results.csv` for all percentiles and throughput values.
