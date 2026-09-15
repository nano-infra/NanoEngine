# KDA evaluation

完整 Layer 架构、Mermaid 图和源码位置见
[`docs/blogs/dlengine-kda-evaluation.md`](../../docs/blogs/dlengine-kda-evaluation.md)。

Run on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python bench/kda_eval/benchmark_kda.py
python bench/kda_eval/plot_results.py bench/kda_eval/results
```

The benchmark uses the KDA shape implied by the GLM fused projections in this
repository: 12 heads with 128-dimensional key/value heads. Override `--heads`,
`--kdim`, and `--vdim` when evaluating another checkpoint.

`hit_rate` is an effective recurrent-state hit: the prefix state is already
available and only the uncached suffix is included in the timed region.
