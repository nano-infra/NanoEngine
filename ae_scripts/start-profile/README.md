# Shared profiler launchers

This directory owns the reusable launch step for torch-profiler experiments.
It deliberately contains no figure dataset, KV-cache distribution, analysis
rule, or plotting code.

| System | Launcher | Input format |
|---|---|---|
| NanoDeploy | `run_nano_profiling.py` | `processed_input_3d.json` |
| vLLM | `run_vllm_profiling.py` | flat JSON array of prompt lengths |

Inputs are always passed as repeated `--input NAME=PATH` arguments. This keeps
the launch mechanism common while each `fig*/` directory owns its workloads.
Both launchers support `--dry-run` and print every expanded per-case launch
command.

The NanoDeploy launcher connects to the AE cluster's default Ray head at
`10.102.252.174:6380`. Use `--ray-address` or
`PROFILE_NANO_RAY_ADDRESS` only to override it. The independent torch
distributed address must be supplied with `--master-address` or
`PROFILE_NANO_MASTER_ADDRESS`.

The NanoDeploy profiler implementation is stored in
`start-profile/nano_dummy_prefill_profile.py`. It imports the installed
`nanodeploy` package and does not read scripts from a NanoDeploy source
checkout. The same package version must be installed on every Ray node.

The vLLM launcher delegates topology and SSH setup to the existing
`benchmarks/offline_dp_profile/start_multinode_offline_profile.sh` in the vLLM
checkout. Therefore the named cluster (default `4node_h200`) must already be
configured there for the current allocation.

For a figure-specific CUDA Graph limit, repeat
`--strategy-max-num-seqs STRATEGY=N` for the selected strategies. An omitted
strategy continues to use the vLLM offline profiler's existing shared default;
the launcher does not replace those defaults globally.

See `../fig13/README.md` for a complete Fig. 13 invocation. These shared
launchers only collect profiles; the figure-owned parser turns the traces into
an auditable plotting CSV.
