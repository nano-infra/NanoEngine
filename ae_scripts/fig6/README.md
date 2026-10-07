# Reproducing Figure 6

Figure 6 breaks down decode attention latency into Attention Computation and
CP Communication. The script measures DP, CP2, CP4, and CP8 across eight
sequence lengths from 8K to 1M, for 32 points in total.

## Resource and time estimate

| Workflow | Hardware | Expected total |
| --- | ---: | ---: |
| Full paper reproduction | 1 node × 8 H200 GPUs | about 10 minutes |

Run all commands from the `ae_scripts` directory.

## Run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 fig6/reproduce_fig6.py
```

The script runs the CUDA Graph benchmarks, collects kernel data with Nsight
Systems, validates all 32 points, and generates:

```text
fig6/fig6.pdf
fig6/fig6.png
```

Raw results are saved under a new directory for each run:

```text
fig6/results/run_*/
├── logs/
├── nsys/
└── online_event/
```

The default configuration uses BF16, 10 warmup iterations, 3 CUDA Graph
warmup iterations, and 20 measured replays.

For other options:

```bash
python3 fig6/reproduce_fig6.py --help
```
