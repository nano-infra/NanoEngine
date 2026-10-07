# Fig. 16: NanoDeploy Ablation

Figure 16 evaluates four NanoDeploy configurations on DeepSeek-V3 with the
Issue 1% workload. All commands run from the `ae_scripts` directory.

## Resource and time estimate

| GPUs | Expected time |
| ---: | ---: |
| 32 H200 GPUs | about 3–5 hours |

The experiment uses four 8-GPU nodes, `DP4-SP8-EP32`, LeastBatch routing, and
request rates 10, 20, 30, and 40. Each of the 16 cases sends requests for 600
seconds.

## Files

| Path | Description |
| --- | --- |
| `fig16/reproduce_fig16.py` | Runs the 16 cases, validates results, and plots the figure |
| `fig16/bench_serving_overhead.py` | Local NanoDeploy request driver and metric collector |

Both scripts use the `nanodeploy` package installed in the active environment.
They do not call scripts from a separate NanoDeploy checkout.

## Run

Start a Ray cluster containing four active 8-GPU nodes, then run:

```bash
export RUN_ID=ae_fig16_full
export NUM_NODES=4

python3 fig16/reproduce_fig16.py run \
  --ray-address 10.102.252.174:6380 \
  --master-address 10.102.252.174:29500 \
  --num-nodes "$NUM_NODES" \
  --run-id "$RUN_ID"
```

`NUM_NODES=4` is the paper configuration. A smaller value can be used for a
reduced functional test, but its results are not directly comparable with the
paper. Use a new `RUN_ID` after changing experiment settings.

The launcher checks the installed NanoDeploy API, model, dataset, and Ray
resources before starting. It runs the four ablations at all four rates and
writes:

```text
fig16/results/$RUN_ID/
├── config.json
├── sweep_summary.tsv
└── */rate_*/
    ├── command.txt
    ├── driver.log
    └── itl_samples.jsonl
```

After all 16 cases succeed, the launcher creates `fig16/fig16.pdf` and
`fig16/fig16.png`. Use `--model-path` or `--dataset-path` only when selecting
different inputs.

Rerunning the command with the same `--run-id` resumes the run. Cases whose
status is `ok` and whose command, log, and ITL output are complete are skipped.
The launcher rejects a reused run ID when experiment settings differ; choose
a fresh ID after changing rates, topology, or other settings. A change to a
variant's GPU memory utilization alone does not require a fresh ID; completed
cases are still checked against their recorded commands and re-run if the
command changed.

## Run selected rates

Use `--rates` to run all four configurations at only the requested rates. For
example, the following command runs the four-way comparison at 30 requests/s:

```bash
export RUN_ID=ae_fig16_rate30
export NUM_NODES=4

python3 fig16/reproduce_fig16.py run \
  --ray-address 10.102.252.174:6380 \
  --master-address 10.102.252.174:29500 \
  --num-nodes "$NUM_NODES" \
  --run-id "$RUN_ID" \
  --rates 30 \
  --max-num-seqs 256
```

One rate runs four cases and normally takes about 45–75 minutes on 32 H200
GPUs. Multiple rates may be supplied, for example `--rates 20 40`. Subset-run
plots are written inside that run's `fig16/results/$RUN_ID/` directory, so
they do not overwrite the full figure. `--max-num-seqs` controls the maximum
concurrent sequences and CUDA Graph batch size; its default remains 256.

## Plot existing results

```bash
export RUN_ID=ae_fig16_full

python3 fig16/reproduce_fig16.py plot \
  --run-id "$RUN_ID"
```

The plot command reads `fig16/results/$RUN_ID/sweep_summary.tsv`. Full-sweep
plots use `fig16/fig16` as the output stem; selected-rate plots are written to
`fig16/results/$RUN_ID/fig16`. Use `--output-root` only when the run used a
non-default results root.
