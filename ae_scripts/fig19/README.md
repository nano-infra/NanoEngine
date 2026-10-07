# Reproducing Figure 19

Figure 19 shows how the number of requests at each context-parallel (CP) size
changes during a DeepSeek-V3 serving run, together with the corresponding
decode all-to-all latency.

Run every command below from the `ae_scripts` directory. Complete the
shared container, model, Ray, and DLSlime setup in the root
[`README.md`](../README.md) first.

## Expected running time

| Stage | Hardware | Expected time |
|---|---|---:|
| Generate a fresh routing trace | Four nodes, 8 GPUs per node | About 10 min, plus model startup and drain |
| Recommended: approximate routing replay | One node, 8 GPUs | About 5 min, including log extraction |
| Optional: replay all unique routing patterns | One node, 8 GPUs | About 30 min |

The number of extracted patterns and occupied approximation intervals depends
on the newly generated routing trace. The recommended replay measures a small
representative subset; the complete replay remains available for evaluators
who need every extracted pattern measured separately.

## 1. Generate the routing trace

This stage runs the paper workload on four eight-GPU nodes and records the
`mode='decode_a2a_masks'` entries consumed by the replay stage.

On node 0, launch the run:

```bash
export RUN_NAME=ae_fig19
python3 fig19/run_fig19_e2e.py \
  --run-name "$RUN_NAME" \
  --num-nodes 4 \
  --send-duration-sec 600
```

To validate the input paths and print the resolved serving command without
using GPUs, append `--dry-run`; this check is optional. After the real run,
confirm that the `rate_60` row in
`bench_logs/e2e/$RUN_NAME/run_summary.tsv` reports `ok`. Then select the routing
log for the remaining steps:

```bash
export DECODE_LOG="bench_logs/e2e/$RUN_NAME/rate_60/driver.log"
```

The launcher refuses to overwrite an existing run directory. Set a new
`RUN_NAME` when repeating Step 1. On an allocation with different input or Ray
paths, use the corresponding options shown by
`python3 fig19/run_fig19_e2e.py --help`.


Steps 2 and 3 run unchanged on one eight-GPU node.

## 2. Recommended: Run the approximate replay

Use one node with exactly eight visible GPUs.

Choose a new output directory and run the staircase DLSlime replay:

```bash
export RESULT_DIR="fig19/results/${RUN_NAME}_approx"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 fig19/collect_fig19_data.py \
  --decode-log "$DECODE_LOG" \
  --output-dir "$RESULT_DIR" \
  --approximate \
  --bucket-step 2 \
  --samples-per-bucket 2 \
  --lse-samples 16
```

The benchmark imports the CUDA-enabled DLSlime installation from the active
Python environment configured in the root README.

The approximation groups Q and residual patterns by two-row intervals of
`(max_send_rows, max_recv_rows)`, keeping zero in a separate interval. It
measures the two most frequent patterns in each occupied interval and assigns
their mean latency to the interval. For LSE, it measures 16
traffic-stratified patterns and uses their median latency. An offline comparison
against an archived complete replay gave about 1.3 us mean absolute error and
3.1 us P95 absolute error for total latency. A new trace may produce different
case counts and approximation error. This is a fast approximation, not a
formal latency upper bound.

A successful run creates:

```text
$RESULT_DIR/
├── cases/
│   ├── benchmark_q_cases.json
│   ├── benchmark_res_cases.json
│   └── benchmark_lse_cases.json
├── approximation_plan.json
├── approximation_case_mapping.csv
├── approximation_summary.json
├── iter_case_mapping.csv
├── q_latency_dataset.csv
├── res_latency_dataset.csv
├── lse_latency_dataset.csv
├── per_iter_latencies.csv
├── per_iter_cp_size_hist.csv
├── extract_summary.json
└── pipeline_summary.json
```

The compact approximation plan and the three representative-case files avoid
writing the much larger JSON files containing every unique pattern. Those
files are written only by the optional complete replay.

## Optional: Replay every routing pattern

To measure every unique Q, residual, and LSE pattern instead of using the
recommended approximation, choose a separate output directory and omit the
approximation options:

```bash
export RESULT_DIR="fig19/results/${RUN_NAME}_full"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 fig19/collect_fig19_data.py \
  --decode-log "$DECODE_LOG" \
  --output-dir "$RESULT_DIR"
```

The number of unique patterns depends on the new routing trace. The 30-minute
estimate assumes a trace with a similar number of unique patterns; a trace with
substantially more patterns will take longer because the complete replay
measures every pattern separately.

## 3. Plot Figure 19

Plot the two materialized per-iteration CSVs produced in Step 2:

```bash
python3 fig19/plot_fig19.py "$RESULT_DIR" \
  --output fig19/fig19
```

The command validates the CSV columns and DP-rank coverage, reports the
selected decode-iteration window, and writes:

```text
fig19/fig19.pdf
fig19/fig19.png
```

## Expected result

The upper panel should show the cluster-wide request counts for CP sizes 5, 6,
7, and 8 as stacked bars. CP=1 should appear as a dashed line on the right
axis. The lower panel should show the median total all-to-all latency across
DP0--DP3, with the minimum-to-maximum range as a shaded band.

Exact latency values can vary with GPU clocks, system load, CUDA, PyTorch, and
DLSlime versions. The expected qualitative result is that dynamic CP sizes
coexist throughout the steady-state window while decode all-to-all latency
varies across iterations and DP ranks.

## Measurement details

The serving log supplies routing tensors and the CP-size timeline; it does not
supply the latency plotted in the lower panel. The recommended path maps the
measured staircase values back to every decode iteration. The optional full
path instead measures every unique Q, residual, and LSE pattern. Both paths use
DLSlime `AllToAllBuffer` and produce the same plotting inputs.

The paper measurement configuration is:

```text
dtype: bfloat16
query heads: 128
Q/K head dimension: 576
V head dimension: 512
execution mode: CUDA Graph
preamble: all-reduce
warmups: 5
repeats: 4
profiled iterations: 10
inner graph iterations: 20
latency measurement: mean repeated P50 duration of the all-to-all CUDA kernel on each active rank, then the maximum across active ranks
```

For each payload, the reported latency is the average of the repeat-level
kernel P50 values on each active rank, followed by the maximum across active
ranks. The plotted value is:

```text
total_us = q_us + res_us + lse_us
```

Thus, `total_us` is the sum of three independently replayed communication
kernels rather than an end-to-end interval from the serving log.
