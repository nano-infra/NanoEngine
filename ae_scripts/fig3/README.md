# Reproducing Figure 3

Figure 3 measures FlashMLA decode latency versus total sequence length in
panel (a), and DeepEP dispatch/combine latency versus per-GPU batch size in
panel (b).

## Resource and time estimate

Both workflows require an allocation of 4 nodes with 8 H200 GPUs per node.
Attention uses 1 GPU on node 0, while DeepEP uses all 32 GPUs.

| Workflow | Attention | DeepEP | Expected total |
| --- | ---: | ---: | ---: |
| Full paper reproduction | 2–3 minutes | about 30 minutes | about 35 minutes |
| Quick trend check | about 1 minute | about 10 minutes | about 12 minutes |

Run all commands from the `ae_scripts` directory. The software environment
is described in the root [`README.md`](../README.md).

## 1. Full paper reproduction

### 1.1 Run Attention

```bash
export RUN_ID=ae_run1

bash fig3/attention/run_attention.sh \
  --gpu 0 \
  --run-id "$RUN_ID"
```

Use a new `RUN_ID` for each fresh run. Results are written to
`fig3/results/attention/${RUN_ID}/`.

### 1.2 Run DeepEP

Run once on node 0; the launcher starts the other nodes over SSH:

```bash
export RUN_ID=ae_run1

python3 fig3/deepep/launch_deepep.py \
  --run-id "$RUN_ID"
```

Results are written to `fig3/results/deepep/${RUN_ID}_4nodes/`. To use another
allocation, add `--remote-hosts <worker-1> <worker-2> <worker-3>`.

## 2. Optional: Quick trend check

This samples fewer points for a trend check and does not replace the full
experiment. Run Attention on one GPU:

```bash
export RUN_ID=ae_quick1

bash fig3/attention/run_attention_quick.sh \
  --gpu 0 \
  --run-id "$RUN_ID"
```

Then run the reduced DeepEP sweep on node 0:

```bash
export RUN_ID=ae_quick1

python3 fig3/deepep/launch_deepep.py \
  --run-id "$RUN_ID" \
  --quick
```

## 3. Plot measured results

Set `RUN_ID` to the completed full or quick run:

```bash
export RUN_ID=ae_run1
export NUM_NODES=4

python3 fig3/plot_fig3.py \
  --mla-data-dir "fig3/results/attention/${RUN_ID}" \
  --mla-pattern "external_flashmla_cudagraph_total_tokens_${RUN_ID}.csv" \
  --deepep-data-dir "fig3/results/deepep/${RUN_ID}_${NUM_NODES}nodes" \
  --deepep-pattern 'node*_summary_rank*.csv' \
  --output-dir "fig3/results/from_fresh/${RUN_ID}" \
  --output-name "fig3_${RUN_ID}" \
  --save-png
```

For a quick run, add `--no-smooth` because the sampled curves contain fewer
points.
