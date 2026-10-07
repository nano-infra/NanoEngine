# 4-node E2E experiments

This directory owns only the reusable service experiments and their raw
artifacts. Figure-specific processing does not belong here.

```text
start-e2e/{system}/             start a service experiment
        │
        └── frontend.log + rank*.log + case_manifest.json + benchmark/
                               │
fig*/                          select the figure's time point, derive data,
                               run microbenchmarks, and draw the figure
```

The vLLM launcher is under [`vllm/`](vllm/README.md), and the NanoDeploy
launcher is under [`nano/`](nano/README.md). Both launch chains live in this
repository; each system checkout is used only as the implementation under test,
and their scripts and parameters remain separate.

For Fig. 5 specifically, extracting a rank snapshot and converting it to token
counts or batch sizes is implemented by
`fig5/extract_vllm_rank_snapshot.py`, not by `start-e2e`.
