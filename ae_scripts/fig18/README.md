# Fig. 18: DLSlime vs. NCCL Collective Communication

Figure 18 compares the intra-node all-to-all collective communication
performance of DLSlime and NCCL. Run all commands from the `ae_scripts`
directory.

## Resource and time estimate

Reserve 1 node with 8 H200 GPUs for the full workflow.

| Stage | GPUs | Expected time |
| --- | ---: | ---: |
| Full reproduction | 8 H200 GPUs | about 30 seconds |

The complete benchmark, validation, and figure generation normally finish in
less than one minute.

## Full paper reproduction

Run the complete experiment and generate the figure:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 fig18/reproduce_fig18_dlslime.py
```

One invocation:

1. checks that eight GPUs are visible;
2. launches one process per GPU and measures DLSlime and NCCL at
   batch sizes 2, 4, 8, 16, 32, 64, and 128;
3. verifies the correctness of the results; and
4. renders:

```text
fig18/fig18_dlslime.pdf
fig18/fig18_dlslime.png
```
