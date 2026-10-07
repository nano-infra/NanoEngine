# Reproducing Figure 5

Figure 5 uses three independent vLLM E2E runs. Panels (a) and (b) extract
per-rank inputs for standalone DeepEP and Attention measurements; panel (c)
plots its E2E HoL time series directly.

Run every command from the `ae_scripts` directory. Use a new `RUN_ID` for
each fresh run so that results from different runs are not mixed. See the root
[`README.md`](../README.md) for shared environment and multi-node requirements.

## Resource and time estimate

Reserve 4 nodes with 8 H200 GPUs per node for the full workflow.

| Stage | GPUs | Expected time |
| --- | ---: | ---: |
| Three E2E source cases | 32 H200 GPUs | 40–45 minutes total |
| DeepEP sweep | 32 H200 GPUs | about 10 minutes |
| Attention microbenchmark | 1 H200 GPU | 8–10 minutes |

The full reproduction takes about 60–65 minutes.

## How the three panels are produced

| Panel | Source | Measurement |
|---|---|---|
| (a) MoE imbalance | `deepep` run, 60%-progress rank snapshot | DeepEP latency for each rank's running-request count |
| (b) Attention imbalance | `attention` run, 60%-progress rank snapshot | FlashMLA latency for each rank's KV token count |
| (c) HoL behavior | `hol` run | HoL demand and free KV blocks over time |

`--fig5-case all` runs the three cases sequentially. The file
`rank_snapshot_time60.json` denotes 60% experiment progress, not 60 seconds.

## 1. Full paper reproduction

The paper configuration uses DP=32 on four 8-GPU nodes. Complete the shared
multi-node setup in the root README before starting.

### 1.1 Generate the three E2E logs on node 0

```bash
export RUN_ID=ae_run1
export NUM_NODES=4

python3 fig5/launch_vllm_e2e.py \
  --num-nodes "$NUM_NODES" \
  --fig5-case all \
  --run-id "$RUN_ID" \
  --ignore-historical-skips
```

The launcher generates
`fig5/results/e2e/fig5_${NUM_NODES}node_${RUN_ID}_env.sh` after all three cases
succeed. To use different workers, add
`--remote-hosts <worker-1> <worker-2> <worker-3>`.

If one case fails, rerun that case with the same `RUN_ID`, for example
`--fig5-case hol`; completed cases are retained.

### 1.2 Extract the 32-rank states for panels (a) and (b)

```bash
export RUN_ID=ae_run1
export NUM_NODES=4
export NUM_RANKS=$((NUM_NODES * 8))
source "fig5/results/e2e/fig5_${NUM_NODES}node_${RUN_ID}_env.sh"

python3 fig5/extract_vllm_rank_snapshot.py \
  "$ATTENTION_LOG" \
  --rank-count "$NUM_RANKS" \
  --output-dir "fig5/results/snapshots/${RUN_ID}/attention"

python3 fig5/extract_vllm_rank_snapshot.py \
  "$DEEPEP_LOG" \
  --rank-count "$NUM_RANKS" \
  --output-dir "fig5/results/snapshots/${RUN_ID}/deepep"
```

Each command must report a complete 32-rank snapshot.

### 1.3 Measure DeepEP for panel (a)

Run once on node 0; the launcher starts the workers over SSH:

```bash
export RUN_ID=ae_run1
export NUM_NODES=4
source "fig5/results/e2e/fig5_${NUM_NODES}node_${RUN_ID}_env.sh"
python3 fig5/deepep/launch_deepep.py \
  --snapshot "fig5/results/snapshots/${RUN_ID}/deepep/rank_snapshot_time60.json" \
  --num-nodes "$NUM_NODES" \
  --master-addr "$MASTER_ADDR" \
  --max-cases 0 \
  --run-id "$RUN_ID"
```

Results are written under `fig5/results/deepep/${RUN_ID}_${NUM_NODES}nodes/`.

### 1.4 Measure Attention and assemble all three panels on node 0

After the DeepEP launcher finishes, run:

```bash
export RUN_ID=ae_run1
export NUM_NODES=4
source "fig5/results/e2e/fig5_${NUM_NODES}node_${RUN_ID}_env.sh"

python3 fig5/reproduce_fig5.py \
  --gpu 0 \
  --attention-snapshot \
    "fig5/results/snapshots/${RUN_ID}/attention/rank_snapshot_time60.json" \
  --deepep-snapshot \
    "fig5/results/snapshots/${RUN_ID}/deepep/rank_snapshot_time60.json" \
  --deepep-data-dir "fig5/results/deepep/${RUN_ID}_${NUM_NODES}nodes" \
  --hol-input "$HOL_LOG" \
  --hol-engines 8-15 \
  --result-root "fig5/results/from_fresh/${RUN_ID}/data" \
  --output "fig5/results/from_fresh/${RUN_ID}/fig5"
```

Expected outputs:

```text
fig5/results/from_fresh/ae_run1/fig5.pdf
fig5/results/from_fresh/ae_run1/fig5.png
fig5/results/from_fresh/ae_run1/data/
```

Small numerical differences from the paper result are expected.
