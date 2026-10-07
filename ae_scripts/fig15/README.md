# Reproducing Figure 15

Figure 15 profiles NanoDeploy and vLLM with the per-rank request distribution
from one representative decode step. The complete workflow first runs the E2E
test, extracts the profiler inputs, and then profiles both systems.

## Expected running time

| Stage | Expected time |
|---|---:|
| E2E input generation | About 45 min |
| NanoDeploy profile | About 15 min |
| vLLM profile | About 30 min |
| Analyze and plot | About 5 min |

The complete workflow takes about 95 minutes after the distributed environment
is ready. This estimate excludes Ray setup.

Run every command from the `ae_scripts` directory. Complete the shared
container, model, Ray, and SSH setup in the root [`README.md`](../README.md)
first.

## 1. Generate the profiler inputs

On `h200-rjob0`, run:

```bash
export RUN_ID=ae_fig15_run1
export NUM_NODES=4

python3 fig15/run_nano_e2e_inputs.py \
  --num-nodes "$NUM_NODES" \
  --run-id "$RUN_ID"
```

`NUM_NODES=4` is the paper configuration. `NUM_NODES=2` runs a reduced
16-GPU check and scales the E2E load with the cluster size. Use a new
`RUN_ID` for every fresh run; shell assignments must not contain spaces around
`=`. In the reduced configuration, the vLLM profiler defaults to DCP only;
the four-node configuration runs the complete DCP and pure-DP matrix.

The command runs the `short`, `issue01`, and `issue05` workloads sequentially
on the selected cluster. For each workload, it selects one middle decode step
and saves the per-rank request lengths in both profiler input formats:

```text
fig15/e2e_results/nano/$RUN_ID/inputs/
├── nano/<workload>/processed_input_3d.json
└── vllm/<workload>.json
```

The E2E launcher does not start Ray. Complete the shared environment and Ray
setup in the root README before running it.

## 2. Profile NanoDeploy and vLLM

Use the same `RUN_ID` and `NUM_NODES` as the E2E step. Run the two profiling
commands sequentially on `h200-rjob0`:

```bash
export RUN_ID=ae_fig15_run1
export NUM_NODES=4

python3 fig15/run_nano_profiling.py \
  --num-nodes "$NUM_NODES" \
  --input-root "fig15/e2e_results/nano/$RUN_ID/inputs/nano" \
  --datasets short issue01 issue05 \
  --run-id "$RUN_ID"

python3 fig15/run_vllm_profiling.py \
  --num-nodes "$NUM_NODES" \
  --lens-root "fig15/e2e_results/nano/$RUN_ID/inputs/vllm" \
  --datasets short issue01 issue05 \
  --run-id "$RUN_ID"
```

## 3. Analyze and plot

After both profiling commands finish, run:

```bash
export RUN_ID=ae_fig15_run1
export NUM_NODES=4

python3 fig15/analyze_and_plot.py \
  --nano-run "fig15/profiling_results/nano/$RUN_ID" \
  --vllm-run "fig15/profiling_results/vllm/$RUN_ID" \
  --output-dir "fig15/results/${RUN_ID}_${NUM_NODES}nodes" \
  --figure-output \
    "fig15/results/${RUN_ID}_${NUM_NODES}nodes/fig15_latency_breakdown"
```

The command discovers all trace directories from the two manifests, validates
the rank count recorded by `NUM_NODES`, parses the traces, and writes:

```text
fig15/results/${RUN_ID}_${NUM_NODES}nodes/
├── fig15_latency_breakdown.pdf
└── fig15_latency_breakdown.png
```

No timestamped directory, track ID, or trace path needs to be entered manually.
Append `--dry-run` to validate and display the discovered plan without parsing.
