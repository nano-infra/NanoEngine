# Shared FlashMLA measurement engine

`benchmark_flashmla.py` measures standalone FlashMLA decode with CUDA Graph
timing. It imports no vLLM code. The caller must explicitly provide either a
total-token list or a sequence-length list, plus batch sizes.

Fig. 3 supplies its regular grid:

```bash
python3 microbench/attention/benchmark_flashmla.py \
  --total_tokens 65536,131072,196608,262144,393216,524288,655360,786432,917504,1048576 \
  --batch_sizes 1,128,1024 \
  --num_heads 128 --head_dim 576 --v_head_dim 512 \
  --rep-ms 200 --bench-repeats 10 --warmup 10 \
  --output <fig3-attention.csv> --skip-plot
```

Fig. 5 supplies the unique sequence lengths extracted from its E2E snapshot
and fixes batch size to one:

```bash
python3 microbench/attention/benchmark_flashmla.py \
  --seq_lens <comma-separated-Fig5-sequence-lengths> \
  --batch_sizes 1 \
  --num_heads 128 --head_dim 576 --v_head_dim 512 \
  --rep-ms 200 --bench-repeats 10 --warmup 10 \
  --output <fig5-attention.csv> --skip-plot
```

Normally users call `fig3/attention/run_attention.sh` or
`fig5/reproduce_fig5.py`; these figure entry points select inputs, check the
environment, validate the CSV, and invoke this shared engine.
