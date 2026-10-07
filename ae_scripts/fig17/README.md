# Fig. 17: NanoDeploy Scheduling Overhead

Figure 17 compares pure DP (`DP32-SP1`) with bucket-based dynamic SP
(`DP4-SP8`) at batch sizes 32, 64, 96, 128, 160, 192, and 256. Dynamic SP uses
the `deepseek_v3` bucket preset.

Each point uses a burst of 128-input/128-output-token requests after warmup.
These short requests select SP=1; `DP4-SP8` is the available dynamic topology.
The figure reports post-warmup P50 model execution, scheduling, and data-transfer
latency, together with total overhead as a percentage of ITL.

## Resource and time estimate

Reserve 4 nodes with 8 H200 GPUs per node for the full workflow.

| Stage | GPUs | Expected time |
| --- | ---: | ---: |
| Full 14-point reproduction | 32 H200 GPUs | about 45–75 minutes |
| Selected two-point comparison | 32 H200 GPUs | about 8–10 minutes |

The points run sequentially. Start the four-node Ray cluster described in the
repository README before running either workflow.

## Full reproduction

```bash
python3 fig17/reproduce_fig17.py
```

The script runs all 14 experiment points and generates:

```text
fig17/results/run_*/
├── logs/
├── fig17.pdf
├── fig17.png
└── manifest.json
```

To inspect the raw scheduling-overhead entries, use:

```bash
rg "'sch_ovhd':.*'post_sch_ovhd':" fig17/results/<run-id>/logs
```

## Run selected points

Run both configurations at one batch size:

```bash
python3 fig17/reproduce_fig17.py \
  --batch-sizes 32 \
  --strategies DP32,DP4SP8
```

This still requires four nodes and generates a two-bar figure.
