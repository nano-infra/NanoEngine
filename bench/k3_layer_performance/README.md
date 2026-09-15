# K3 Layer-Component Performance

This directory contains the first B300 performance sweep used by Chapter 7 of the K3 analysis. It separates production-shape kernel pipelines from complete attention-component paths; it does not claim full Decoder Layer (attention + FFN) or end-to-end model latency.

## Recorded experiments

- `results/kda_chunk_decode.csv`: KDA Prefill chunk and Decode batch sweeps.
- `results/mla_cache_total_*.json`: MLA cached-prefix Prefill with 2K, 8K, and 16K fresh chunk, mixed FP8/BF16 latent cache, and a 128K prefix split.
- `benchmark_mla_full_breakdown.py` and `results/mla_full_breakdown.csv`: separate the MLA cached-prefix kernel pipeline from the complete MLA attention path at 1M context, sweeping 1K through 16K fresh chunks.
- `results/dense_ffn.csv`: K3 single dense SwiGLU FFN, with Prefill chunk and Decode batch sweeps.
- `results/megamoe_chunk.csv`: single-rank production MXFP4 MegaMoE with 896 experts, Top-16 routing, latent width 3584, and intermediate width 3072.
- `plot_results.py`: regenerates the documentation figures from the raw data.

The measurements were collected on one NVIDIA B300 SXM6 GPU after warm-up with CUDA-event timing. MegaMoE uses a 16K-token preallocated capacity and reports its persistent workspace separately from steady forward activation.

## Regenerate figures

```bash
python bench/k3_layer_performance/plot_results.py
```

The next pass should add production paged-cache MLA Decode, MegaMoE Decode batch scaling, and Nsight counters for a formal roofline classification.
