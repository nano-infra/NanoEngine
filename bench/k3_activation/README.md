# K3 activation peak-memory benchmark

This benchmark measures incremental CUDA peak memory for K3-shaped operator
cores. Inputs, weights, and synthetic persistent cache/state are allocated
before the measurement baseline. The current targets are the KDA recurrence
core and the non-absorbed MLA Prefill attention core.

```bash
CUDA_VISIBLE_DEVICES=0 python bench/k3_activation/benchmark_activation.py
```

The benchmark uses BF16 activations and the dimensions from
`/mnt/public/Kimi-K3/config.json`: 96 heads, KDA head dimension 128, and MLA
Q/K and V head dimensions 192 and 128. It is not an end-to-end server-memory
measurement.

The maximum setting is a 1M-token resident context and a 16K-token Prefill
chunk. The persistent cache/state is synthetic and excluded from the
incremental activation peak. The MLA measurement is currently fresh-only: its
16K queries attend the 16K fresh tokens, not the entire 1M-token cache.

MoE is intentionally measured with a local backend in this chapter. The
distributed MegaMoE experiment belongs to the later partitioning analysis.

Run the cached-prefix MLA maximum separately because it expands a 1M-token
latent cache and executes 16K queries against the full history:

```bash
CUDA_VISIBLE_DEVICES=0 python bench/k3_activation/benchmark_mla_cached.py
```


Run the FP8 cached-prefix split sweep with `0` for the unsplit control. The
production default is 131072 tokens and the same environment variable is
forwarded to Ray workers.

```bash
for split in 16384 32768 65536 131072 262144 0; do
  CUDA_VISIBLE_DEVICES=0 python bench/k3_activation/benchmark_mla_prefix_split.py \
    --prefix-chunk-size "$split"
done
MPLBACKEND=Agg python bench/k3_activation/plot_results.py
```

The benchmark uses a 1M-token total context, a 1008K-token packed mixed-FP8/BF16
cached prefix, and 16K fresh queries. It runs one warm-up before measuring one
steady forward with CUDA events. Incremental peak includes cache restoration,
K/V expansion, attention, and online output/LSE merging; weights and persistent
packed cache are allocated before the baseline.


Measure the production DeepGEMM MegaMoE kernel without EP communication:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 \
  bench/k3_activation/benchmark_megamoe_ws1.py
```

This benchmark runs NanoDeploy's `MegaMoEExperts` wrapper with `ffn_ep=1` and all 896 K3 experts. It separates the persistent 16K-capacity symmetric buffer (both DeepGEMM logical bytes and CUDA free-memory delta) from first-forward and steady allocation peaks.
