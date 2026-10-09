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

## Shared model and dataset paths

Model checkpoints, datasets, and external checkouts are configured in
[`paths.env`](paths.env) (copy [`paths.env.example`](paths.env.example) and fill
it in). The keys used by the scripts are:

| Key | Contents |
|---|---|
| `AE_DPSK_MODEL` | DeepSeek-V3 checkpoint |
| `AE_KIMI_MODEL` | Kimi-K2-Instruct-0905 checkpoint |
| `AE_DATASET_ROOT` | dataset root with the figure-specific subdirectory layout described below |
| `AE_DATASET_SHAREGPT4O` | directory containing the ShareGPT-4o CSV |
| `AE_DATASET_MIXLONG_0326` | directory containing the Issue 1% and Issue 5% CSVs |
| `AE_DATASET_MADHA` | full path to the GitHub Issues CSV |
| `AE_VLLM_ROOT` | vLLM checkout |

Figure-specific instructions may use `--model-path <model-path>` when the model
location is user-selectable. For multi-node experiments, every path in
`paths.env` must resolve to the same location on all nodes.

## Included workload traces

The four preprocessed request-length traces used for the ShareGPT-4o,
Issue 1%, Issue 5%, and GitHub Issues workloads are included in
[`dataset/`](dataset/). These workload names describe the input traces and
are independent of the model selected for an experiment. No additional
dataset download or regeneration is needed to obtain these four CSVs.

| Workload | File under `dataset/` | Requests | Composition |
|---|---|---:|---|
| ShareGPT-4o | [`sharegpt4o-mixed-random-60k.csv`](dataset/sharegpt4o-mixed-random-60k.csv) | 60,000 | Short-context requests |
| Issue 1% | [`sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv`](dataset/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv) | 60,000 | 59,400 short + 600 long requests |
| Issue 5% | [`sharegpt4o-random_geminiissue_r0.05_n60000_60k.csv`](dataset/sharegpt4o-random_geminiissue_r0.05_n60000_60k.csv) | 60,000 | 57,000 short + 3,000 long requests |
| GitHub Issues | [`Gemini_Issues_Stats_rename-shuffle.csv`](dataset/Gemini_Issues_Stats_rename-shuffle.csv) | 2,776 | Long-context requests |

The files occupy 2,992,270 bytes in total (approximately 2.85 MiB,
uncompressed). Each row specifies one request using `prompt_len` and
`output_len`, both measured in tokens. The CSVs contain request-length
metadata rather than the original prompt or response text. The mixed traces
also contain `total_len` and `type` (`short` or `long`). The GitHub Issues
trace additionally contains `type`, `query_id`, `num_total_tokens`, and
`pd_ratio`.

The row counts and mixture ratios above describe the supplied files before
any experiment-specific sampling or filtering. For example, the two-node
basic test filters Issue 1% requests to fit its KV-cache capacity; its
run-local input is therefore a subset of the supplied trace. See the
figure-specific README for the request count and filtering used in a run.

### Configure the bundled traces

The CSVs are stored directly under `dataset/`. Figures 12, 15, and 20 also
look for the original `sharegpt-4o/`, `sharegpt-4o-mixlong-0326/`, and `madha/`
subdirectories below `AE_DATASET_ROOT`. On a fresh checkout, run the following
once from `ae_scripts/` to create relative file links for that layout:

```zsh
mkdir -p dataset/sharegpt-4o dataset/sharegpt-4o-mixlong-0326 dataset/madha
ln -s ../sharegpt4o-mixed-random-60k.csv dataset/sharegpt-4o/
ln -s ../sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv dataset/sharegpt-4o-mixlong-0326/
ln -s ../sharegpt4o-random_geminiissue_r0.05_n60000_60k.csv dataset/sharegpt-4o-mixlong-0326/
ln -s ../Gemini_Issues_Stats_rename-shuffle.csv dataset/madha/
```

After copying `paths.env.example` to `paths.env`, set the following entries.
Replace `/absolute/path/to/NanoDeploy/ae_scripts` with the absolute path to
your checkout, visible at the same location on every worker:

```text
AE_DATASET_ROOT=/absolute/path/to/NanoDeploy/ae_scripts/dataset
AE_DATASET_SHAREGPT4O=/absolute/path/to/NanoDeploy/ae_scripts/dataset/sharegpt-4o
AE_DATASET_MIXLONG_0326=/absolute/path/to/NanoDeploy/ae_scripts/dataset/sharegpt-4o-mixlong-0326
AE_DATASET_MADHA=/absolute/path/to/NanoDeploy/ae_scripts/dataset/madha/Gemini_Issues_Stats_rename-shuffle.csv
```

Use literal absolute paths in `paths.env`: the Python path loader does not
expand shell variables or `~`. The three directory entries and the one CSV
entry above cover these four supplied traces. Other optional workloads in
`paths.env.example`, such as `AE_DATASET_0110`, require their own inputs.

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
