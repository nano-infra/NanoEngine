# K3 cache-performance benchmarks

The KDA sweep uses the K3 shape (96 heads, key/value width 128), BF16 state,
and includes a device-to-device SSM-slot restore inside every CUDA-event timing
interval. It measures the recurrence core, not projections or the full layer.

```bash
CUDA_VISIBLE_DEVICES=0 python bench/kda_eval/benchmark_kda.py \
  --output-dir bench/k3_cache_performance/results \
  --heads 96 --kdim 128 --vdim 128 \
  --lengths 64,256,1024,4096,16384 \
  --hit-rates 0,0.5,0.75,0.9,0.95,0.99 \
  --decode-batch-sizes 1,8,32,128 \
  --include-state-restore --warmup 3 --repeats 20
```

The MLA sweep fixes logical context at 64K and changes the cached-prefix/fresh-
suffix split. It uses the mixed FP8/BF16 cache layout and the production 128K
prefix chunk. Cache restoration, latent K/V expansion, attention, and online
merge are timed; input projections and cache lookup are outside the interval.

```bash
for fresh in 65536 32768 16384 4096 512; do
  CUDA_VISIBLE_DEVICES=0 python bench/k3_activation/benchmark_mla_prefix_split.py \
    --total-context 65536 --fresh-chunk "$fresh" \
    --prefix-chunk-size 131072 --warmup 2 \
    --output-dir bench/k3_cache_performance/results
done
MPLBACKEND=Agg python bench/k3_cache_performance/plot_results.py
```

The primary serving experiment fixes total context at 1M, measures all 64
consecutive 16K chunks, and derives remaining-Prefill latency at aligned cache
boundaries:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python \
  bench/k3_cache_performance/benchmark_1m_serving.py --warmup 1
MPLBACKEND=Agg python bench/k3_cache_performance/plot_results.py
```

`cache_1m_serving.csv` is the source for the chapter's primary KDA and MLA
figures. The older fixed-16K/fixed-64K data are microbenchmark controls.
