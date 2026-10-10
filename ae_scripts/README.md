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

## Modified vLLM baseline

The vLLM baselines require our modified **vLLM 0.18.0**, including the dispatch
and profiling changes used by the AE scripts:

- Repository: [FirwoodLin/vllm](https://github.com/FirwoodLin/vllm).
- Branch: [`ae-repro-clean`](https://github.com/FirwoodLin/vllm/tree/ae-repro-clean).
- Pinned commit: [`5dcf9eec3c55309824a89b996bc57687d4aa4ac8`](https://github.com/FirwoodLin/vllm/commit/5dcf9eec3c55309824a89b996bc57687d4aa4ac8).

The [AE Dockerfile](docker/Dockerfile) installs this pinned revision
automatically. The following manual steps are for an existing compatible
environment. Run them inside that environment using `zsh`. The installation
below targets Linux x86_64 with Python 3.12, PyTorch 2.10.0+cu129, and CUDA 12.9,
matching the `vllm/vllm-openai:v0.18.0` base environment. The build dependencies
listed in the fork's `pyproject.toml` must already be installed.

### Obtain the pinned source

Choose a fresh checkout path visible at the same absolute location on every
worker. Clone and check out the fixed commit once on the shared filesystem:

```zsh
VLLM_SRC=/absolute/path/to/vllm-ae
VLLM_COMMIT=5dcf9eec3c55309824a89b996bc57687d4aa4ac8

git clone --depth 1 --single-branch --branch ae-repro-clean \
  https://github.com/FirwoodLin/vllm.git "$VLLM_SRC"
git -C "$VLLM_SRC" fetch --depth 1 origin "$VLLM_COMMIT"
git -C "$VLLM_SRC" checkout --detach "$VLLM_COMMIT"
test "$(git -C "$VLLM_SRC" rev-parse HEAD)" = "$VLLM_COMMIT"
```

### Build and install a normal wheel

Reuse the matching native components from the official vLLM 0.18.0 CUDA wheel
while packaging the modified Python source. This uses the fork's
`VLLM_USE_PRECOMPILED` build support and avoids recompiling CUDA kernels.
Download the base wheel once, then build the modified wheel:

```zsh
VLLM_BUILD_DIR="$(mktemp -d)"
curl --fail --location --retry 3 \
  https://github.com/vllm-project/vllm/releases/download/v0.18.0/vllm-0.18.0-cp38-abi3-manylinux_2_31_x86_64.whl \
  --output "$VLLM_BUILD_DIR/vllm-base-precompiled.whl"

VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION="$VLLM_BUILD_DIR/vllm-base-precompiled.whl" \
VLLM_TARGET_DEVICE=cuda \
VLLM_VERSION_OVERRIDE=0.18.0 \
python3 -m pip wheel --no-index --no-build-isolation --no-deps \
  --wheel-dir "$VLLM_BUILD_DIR" "$VLLM_SRC"

python3 -m pip install --no-index --no-deps --force-reinstall \
  "$VLLM_BUILD_DIR"/vllm-0.18.0-*.whl
```

This installs a regular package and replaces the existing vLLM without changing
its dependencies; it does not use editable installation. Build once, then run
the final installation command in the active Python environment of **every
worker container**. If `VLLM_BUILD_DIR` is local to the build node, copy the
generated wheel to a shared directory or to each worker, and set
`VLLM_BUILD_DIR` to the directory containing that wheel before installing.

Run this check from `ae_scripts/`, outside the vLLM source checkout:

```zsh
python3 -c "import vllm; print(vllm.__version__); print(vllm.__file__)"
```

The version should be `0.18.0`, and the imported module should be in the active
environment's installed packages. Create a separate runtime working directory
at the same absolute path on every worker, so the source checkout does not
shadow the installed package:

```zsh
mkdir -p /absolute/path/to/vllm-workdir
export VLLM_WORKDIR=/absolute/path/to/vllm-workdir
```

Set the same literal absolute path in `ae_scripts/paths.env`:

```text
AE_VLLM_ROOT=/absolute/path/to/vllm-workdir
```

## Shared model and dataset paths

Model checkpoints, datasets, and external checkouts are configured in
[`paths.env`](paths.env) (copy [`paths.env.example`](paths.env.example) and fill
it in). The keys used by the scripts are:

| Key | Contents |
|---|---|
| `AE_DPSK_MODEL` | DeepSeek-V3 checkpoint |
| `AE_KIMI_MODEL` | Kimi-K2-Instruct-0905 checkpoint directory, used directly |
| `AE_DATASET_ROOT` | dataset root with the figure-specific subdirectory layout described below |
| `AE_DATASET_SHAREGPT4O` | directory containing the ShareGPT-4o CSV |
| `AE_DATASET_MIXLONG_0326` | directory containing the Issue 1% and Issue 5% CSVs |
| `AE_DATASET_MADHA` | full path to the GitHub Issues CSV |
| `AE_VLLM_ROOT` | vLLM working directory; `/opt/ae/vllm-workdir` in the AE image |

Configure only the model and dataset paths needed by the selected experiments.
The shared vLLM runner and Fig. 12 resolve paths for selected workloads only;
unused Qwen and `1k1k` inputs can remain unset. Set `AE_KIMI_MODEL` to the actual
checkpoint directory (including a snapshot directory when using the Hugging
Face cache); no cache-root key or fixed snapshot suffix is required.

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

Build the environment from the supplied [Dockerfile](docker/Dockerfile).
See [docker/README.md](docker/README.md) for the pinned source inputs and build
prerequisites. From `ae_scripts/` on the host, prepare a fresh build context
and build the image:

```zsh
python3 docker/prepare_context.py \
  --output /absolute/path/to/ae-image-context

docker build --platform linux/amd64 \
  -t nanodeploy-ae:vllm0180 /absolute/path/to/ae-image-context
```

Build once and distribute the same image to every allocated node through a
registry or `docker save` / `docker load`. Each host needs NVIDIA Container
Toolkit, a compatible NVIDIA driver, and the allocation's RDMA devices.

Create an `ae_merged` container on each host. Replace the three paths below:
`SHARED_ROOT` must contain the experiment checkout, datasets, and writable
results directory; `MODEL_ROOT` contains the checkpoints. Add mounts for any
other paths used by `paths.env` or the worker SSH setup. Use identical absolute
paths on every node. The launch options below target the supplied AE cluster:

```zsh
SHARED_ROOT=/absolute/path/to/shared-workspace
MODEL_ROOT=/absolute/path/to/model-storage
AE_SCRIPTS_DIR=/absolute/path/to/NanoDeploy/ae_scripts

docker run -d --name ae_merged \
  --gpus all --network host --ipc host --privileged \
  --cap-add SYS_ADMIN --cap-add SYS_PTRACE --ulimit memlock=-1:-1 \
  --mount "type=bind,src=$SHARED_ROOT,dst=$SHARED_ROOT" \
  --mount "type=bind,src=$MODEL_ROOT,dst=$MODEL_ROOT,readonly" \
  --workdir "$AE_SCRIPTS_DIR" \
  --entrypoint /usr/bin/tail nanodeploy-ae:vllm0180 -f /dev/null

docker exec -it --workdir "$AE_SCRIPTS_DIR" ae_merged zsh
```

The image installs NanoDeploy and the pinned modified vLLM automatically.
For this image, set `AE_VLLM_ROOT=/opt/ae/vllm-workdir` in `paths.env`;
`VLLM_WORKDIR` is already set to that directory in the image. Keep runtime
working directories separate from vLLM source checkouts. Inside each
container, check the package versions and required native imports with:

```zsh
python3 /opt/ae/verify_environment.py
```

For an SSH-launched multi-node workflow, configure an endpoint for every
worker that runs commands inside its container, with the same mounted paths
and Python environment. A host SSH login alone does not enter the container.
From node 0's container, verify each configured worker endpoint with:

```zsh
ssh -t <worker-ssh-host> zsh
```

Confirm that the resulting shell is inside the intended worker's `ae_merged`
container. The image does not configure worker SSH access or start Ray;
prepare SSH access for your allocation and follow the next section for Ray.

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

The Figure 5 E2E paths require the same repository, vLLM working directory, model,
request-length dataset, and Python environment at the same paths on every
node. Node 0 must be able to reach every worker over SSH, and the address
passed as `MASTER_ADDR` must be reachable from the worker containers. The
two-node quick test uses 8 GPUs per node. Its postprocessing additionally
requires CUDA PyTorch, Triton, `flash_mla`, and the pinned `deep_ep` build
described above.

Mount the shared research filesystem at the same absolute paths on every
worker. Repository changes, generated environment files, snapshots, logs,
and results written there are then visible on every node. If using node-local
storage, synchronize the required inputs and configuration before launching
a multi-node run.

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
