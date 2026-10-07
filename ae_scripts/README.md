# ASPLOS artifact-evaluation scripts

## Pinned dependencies

DeepEP experiments use `deep_ep` **1.2.1+73b6ea4**, built from DeepEP commit
`73b6ea4a439ba03a695563f9fd242c8e4b02b37c`. The matching upstream
`test_low_latency.py` benchmark and helpers are included under
`microbench/deepep/third_party/deepep/`; no external DeepEP source checkout is
required at runtime.

NanoDeploy experiments require the `nanodeploy` Python package to be installed
in the active environment on every Ray node. Profiler orchestration is included
in `start-profile/nano_dummy_prefill_profile.py`; no external NanoDeploy script
directory is used. Verify the package in each container with:

```bash
python3 -c "import nanodeploy; print(nanodeploy.__file__)"
```

The Fig. 15 merged-trace parser additionally uses `ijson` to stream profiler
JSON files without loading multi-gigabyte documents into memory. Verify it in
the postprocessing environment with:

```bash
python3 -c "import ijson; print(ijson.__version__)"
```

Fig. 18 and Fig. 19 use DLSlime's CUDA intra-node collectives. The active
Python environment must contain a DLSlime build compiled with CUDA and
`BUILD_INTRA_OPS=ON`; the default PyPI build does not enable these operators.
Verify the exact API required by the AE scripts with:

```bash
python3 -c "import dlslime; print(dlslime.__file__); print(dlslime.AllToAllBuffer, dlslime.KernelImpl.Basic)"
```

## Shared model directories

Model checkpoints, datasets, and external checkouts are configured in
[`paths.env`](paths.env) (copy [`paths.env.example`](paths.env.example) and fill
it in). The keys used by the scripts are:

| Key | Contents |
|---|---|
| `AE_DPSK_MODEL` | DeepSeek-V3 checkpoint |
| `AE_KIMI_MODEL` | Kimi-K2-Instruct-0905 checkpoint |
| `AE_DATASET_ROOT` | dataset root |
| `AE_VLLM_ROOT` | vLLM checkout |

Figure-specific instructions may use `--model-path <model-path>` when the model
location is user-selectable. For multi-node experiments, every path in
`paths.env` must resolve to the same location on all nodes.

## AE container

Run the artifact commands inside the pre-created `ae_merged` container. From
the `ae_scripts` directory on an allocated host, enter the container at the
working directory with:

```bash
docker exec -it --workdir "$PWD" ae_merged zsh
```

For an SSH-launched multi-node workflow, configure an endpoint for every
worker that enters its container directly. From node 0, verify each worker
endpoint with:

```bash
ssh -t <worker-ssh-host> zsh
```

## NanoDeploy Ray cluster

NanoDeploy experiments use `10.102.252.174:6380` as the default Ray head
address. This default applies to the NanoDeploy E2E and profiling launchers;
Fig. 5 and the other vLLM workflows use their documented SSH and
multiprocessing launch paths instead of Ray. Override the address with the
figure launcher's `--ray-address` option or documented environment variable
only when using a different allocation.

For a fresh allocation, enter the `ae_merged` container on every node. Start
the head on `10.102.252.174` with:

```bash
ray start --head --node-ip-address=10.102.252.174 --port=6380 \
  --num-gpus=8 --include-dashboard=false --disable-usage-stats
```

Join each worker with the same copy-paste command:

```bash
ray start --address=10.102.252.174:6380 \
  --num-gpus=8 --disable-usage-stats
```

On the head, verify the allocation before launching an experiment:

```bash
ray status --address=10.102.252.174:6380
```

The two-node quick tests require two active nodes and 16 GPUs. Four-node paper
reproductions require four active nodes and 32 GPUs.

## Basic test

After preparing exactly two eight-GPU nodes as described above, run the
functional kick-the-tires test from the `ae_scripts` directory:

```bash
python3 basic-test/run_basic_test.py
```

It runs NanoDeploy DP2-CP8-DCP and vLLM DP2-TP8-DCP8 sequentially on the
Issue 1% workload for 60 seconds each. See
[`basic-test/README.md`](basic-test/README.md)
for address overrides, expected output, and retained artifacts.

The wrapper applies the cluster's validated eight-HCA RoCE defaults, including
GID index 3, to the NanoDeploy process. Preserve these defaults on the supplied
allocation; override them through environment variables only if the cluster
operator provides different interface or GID settings.

## Multi-node service requirements

The Figure 5 E2E paths require the same repository, vLLM checkout, model,
request-length dataset, and Python environment at the same paths on every
node. Node 0 must be able to reach every worker over SSH, and the address
passed as `MASTER_ADDR` must be reachable from the worker containers. The
two-node quick test uses 8 GPUs per node. Its postprocessing additionally
requires CUDA PyTorch, Triton, `flash_mla`, and the pinned `deep_ep` build
described above.

All four machines in the AE allocation share the `/vllm` mount and the shared
research filesystem. Repository changes, generated environment files,
snapshots, logs, and results written below these paths are immediately
visible on every node; no manual synchronization is required.

The repository separates service experiments, reusable operator measurements,
and figure-specific processing:

```text
assets/                    shared fonts and other plotting assets
ae_utils/                  shared helpers used by the AE scripts
start-e2e/                 launch end-to-end systems and retain raw artifacts
start-profile/             launch reusable profiler runs for supplied inputs
microbench/                reusable Attention and DeepEP measurement engines
fig*/                      select figure inputs, validate data, and plot
archive/                   retired scripts; never used by active workflows
```

For Fig. 3 and Fig. 5, the ownership boundary is:

```text
microbench/attention/benchmark_flashmla.py
    ├── Fig. 3 supplies a regular total-token × batch-size grid
    └── Fig. 5 supplies sequence lengths observed in a vLLM E2E snapshot

microbench/deepep/run_low_latency_sweep.sh
    ├── Fig. 3 supplies its regular token sweep
    └── Fig. 5 supplies batch sizes observed across E2E DP ranks
```

The shared scripts know how to measure an explicitly supplied input. They do
not choose a paper figure's E2E case, snapshot, token grid, aggregation rule,
or plotting style. Those choices remain documented and implemented under the
corresponding `fig*/` directory.

When a script is fully superseded, move the retired implementation to
`archive/` and remove all active references to it. Do not archive a figure
preset, validator, or plotter while the figure workflow still calls it.

See [`microbench/README.md`](microbench/README.md),
[`fig3/README.md`](fig3/README.md), and [`fig5/README.md`](fig5/README.md).
