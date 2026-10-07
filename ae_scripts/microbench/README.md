# Shared operator microbenchmarks

This directory contains measurement engines shared by multiple figures:

```text
microbench/
├── attention/
│   ├── README.md
│   └── benchmark_flashmla.py
└── deepep/
    ├── README.md
    ├── run_one_low_latency.py
    ├── run_low_latency_sweep.sh
    └── parse_low_latency_logs.py
```

The figure directories own presets and data transformations:

- Fig. 3 defines a regular scaling sweep and averages its selected DeepEP
  representative ranks.
- Fig. 5 extracts per-rank E2E load, deduplicates the observed inputs, runs the
  corresponding points, and maps latency back to the original ranks.

Do not put E2E launchers or plotting code here. Conversely, figure directories
should not carry private copies of the FlashMLA or DeepEP measurement engines.
