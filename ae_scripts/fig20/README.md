# Fig. 20: Effect of Dual Batch Overlap

Figure 20 compares vLLM, vLLM with Dual Batch Overlap (DBO), and NanoDeploy
on the ShareGPT4o and Issue1% workloads. It reports mean and P99
time-per-output-token (TPOT).

Run all commands from the `ae_scripts` directory after completing the
shared setup in the root [`README.md`](../README.md).

## Resource and time estimate

The paper configuration uses four 8-GPU H200 nodes.

| Configuration | GPUs | Send time per rate | Expected time per 5-rate sweep |
| --- | ---: | ---: | ---: |
| Paper reproduction | 32 H200 | 600 seconds | about 1.25 hours |

Startup and cleanup time are included in the estimates.

## 1. Run the DBO sweeps

The launcher requires a fresh run name. The two workload commands below use
the same name so their outputs remain together. The documented default name is
`ae_fig20_full`; choose a new name when changing experiment settings.

### Paper configuration

Run ShareGPT4o:

```bash
RUN_NAME=ae_fig20_full

python3 fig20/run_fig20.py \
  --run-name "$RUN_NAME" \
  --mode dbo \
  --num-nodes 4 \
  --bench-duration-sec 600 \
  --max-request-tokens 1000000
```

Then run Issue1% with the same run name:

```bash
python3 fig20/run_fig20.py \
  --run-name "$RUN_NAME" \
  --mode dbo \
  --num-nodes 4 \
  --bench-duration-sec 600 \
  --max-request-tokens 1000000 \
  --dataset issue01_random
```

This configuration uses `DP32+EP32`, `TP=1`, `DCP=1`, `least_batch`,
`max_num_seqs=192`, and DBO decode-token threshold 2. ShareGPT4o uses rates
40, 60, 80, 100, and 120 req/s; Issue1% uses rates 10, 20, 30, 40, and
50 req/s.

`--mode non-dbo` runs only the baseline; `--mode both` runs DBO and non-DBO
sequentially with identical workload settings.

## Outputs and resuming

Each run is isolated below its name:

```text
fig20/results/dbo/$RUN_NAME/
├── run_config.json
├── _case_csv/
├── _filtered_datasets/
├── _runs/
└── DPSK/
```

Non-DBO results use the same layout below
`fig20/results/non_dbo/$RUN_NAME/`. With `--mode both`, the launcher creates
both roots.

Rerunning a compatible command skips completed cases. The launcher rejects an
existing run name when the requested topology, duration, token limit, rates,
or other recorded settings differ. Use a fresh `RUN_NAME` after changing any
of them.

A successful case reports `ok` and contains `benchmark/summary.json` and
`benchmark/requests.jsonl`. If a case fails, inspect its `case_manifest.json`,
`frontend.log`, and available `rank*.log` files.

## 2. Generate Figure 20

Generate the paper figure only from a completed four-node, 600-second run.
The parser reuses the supplied Fig. 12 non-DBO vLLM and NanoDeploy results and
reads the new DBO results from the selected run directory.

```bash
RUN_NAME=ae_fig20_full

python3 fig20/plot_dbo_tpot_comparison.py \
  --dbo-root "fig20/results/dbo/$RUN_NAME" \
  --data-output fig20/dbo_tpot_comparison.tsv \
  --data-only

python3 fig20/plot_dbo_tpot_comparison.py \
  --dataset issue01_random \
  --dbo-root "fig20/results/dbo/$RUN_NAME" \
  --data-output fig20/dbo_tpot_comparison_issue1.tsv \
  --data-only

python3 fig20/plot_vllm_tbo_paper.py
```

The final command writes `fig20/fig20.pdf` and `fig20/fig20.png`. The two TSV
files retain the selected metrics and their input paths for inspection.
