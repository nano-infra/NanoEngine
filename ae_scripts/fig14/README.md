# Fig. 14: Load Balance and Head-of-Line Blocking

Fig. 14 uses five independent four-node, 32-GPU service runs. The launcher
reads the experiment settings from [`fig14_cases.csv`](fig14_cases.csv).

| Panel | System | Dataset | Routing Policy | Rate |
|---|---|---|---|---:|
| (a) | NanoDeploy `DP4-SP8` | Issue 1% | LeastBatch | 25 |
| (a) | vLLM DP32 | Issue 1% | LeastBatch | 25 |
| (a) | vLLM DP32 | Issue 1% | LeastCache | 60 |
| (b) | NanoDeploy `DP4-SP8` | Issue 5% | LeastBatch | 30 |
| (b) | vLLM DP32 | Issue 5% | LeastBatch  | 17 |

Run all commands from the `ae_scripts` directory after completing the
shared environment and four-node setup in the root [`README.md`](../README.md).

## Resource and time estimate

All five service cases run sequentially on 4 nodes with 8 H200 GPUs per node.

| System | Expected time |
|---|---:|
| NanoDeploy, 2 cases | about 25 minutes total |
| vLLM, 3 cases | about 15 minutes each (45 minutes total) |
| Total | about 70 minutes |

## Reproduce

Choose a fresh run ID and run the two NanoDeploy cases and three vLLM cases
sequentially:

```bash
export RUN_ID=ae_fig14_1
export NUM_NODES=4

python3 fig14/run_fig14.py \
  --num-nodes "$NUM_NODES" \
  --run-id "$RUN_ID"
```

All input paths, expanded commands, child manifests, and logs are recorded
below `fig14/results/$RUN_ID/`. Use `--systems nano` or
`--systems vllm` to run only one system. Cluster, model, workdir, and
individual-case overrides are listed by `python3 fig14/run_fig14.py --help`.

Rerunning the same command with the same `RUN_ID` resumes the workflow.
NanoDeploy `lb`/`hol` cases and vLLM cases whose manifests and required output
files show a complete successful run are skipped. Failed or unfinished cases
are run again. If the settings recorded for an existing case differ from the
new command, the launcher rejects the reuse of that `RUN_ID`.

The main outputs are:

```text
fig14/results/$RUN_ID/
├── manifest.json
├── nano/
│   ├── manifest.json
│   ├── lb/driver.log
│   └── hol/driver.log
└── vllm/
    ├── lb_least_batch/
    ├── lb_least_cache/
    └── hol_least_batch/
```

## Plot

The plotter resolves the five completed service logs from the run directory:

```bash
export RUN_ID=ae_fig14_1
python3 fig14/plot_fig14_from_service_logs.py \
  --run-root "fig14/results/$RUN_ID" \
  --output fig14/fig14
```

This writes `fig14/fig14.pdf` and `fig14/fig14.png`.
