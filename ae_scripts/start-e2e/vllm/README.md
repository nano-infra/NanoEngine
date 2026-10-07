# vLLM 4-node E2E launcher

This directory is a self-contained copy of the vLLM E2E launch chain. It does
not import the runner under `vllm-two-batch-overlap` or the older
`vllm-v0180/offline_bench` directory.

| File | Responsibility |
|---|---|
| `launch_vllm_e2e.py` | Portable entry point and paper presets |
| `manual_multinode_poisson_runner.py` | SSH orchestration, cleanup, manifests, artifact layout |
| `offline_poisson_harness.py` | Start the vLLM frontend/headless engines and issue Poisson requests |
| `offline_profile_strategy_defaults.py` | Strategy defaults required by the runner |

The only code imported outside this directory is the vLLM checkout selected by
`--vllm-workdir`; the model, dataset, Python/CUDA environment, SSH aliases, and
shared artifact filesystem remain environment inputs.

## Fig. 5 source experiments

All three presets use 4 nodes × 8 H200 GPUs, DP=32, TP=1, DCP=1, EP enabled,
FlashMLA Attention, `deepep_low_latency`, `max_num_seqs=256`, 32 warm-up
requests, a 600-second Poisson trace, and `max_model_len=1,000,000`.

| Fig. 5 use | Exact vLLM dispatch flag | Rate | KV memory | Requests | Raw field consumed later |
|---|---|---:|---:|---:|---|
| Attention imbalance | `waiting_x4_plus_running` | 30 | 0.90 | 18,000 | per-rank KV usage |
| DeepEP imbalance | `least_cache` | 30 | 0.87 | 18,000 | per-rank running requests |
| HoL blocking | `waiting_x4_plus_running` | 45 | 0.90 | 27,000 | queue/KV time series |

The paper calls `waiting_x4_plus_running` “least batch.” The rate-30 historical
run omitted the CLI flag because this was the launcher default; the new preset
writes the equivalent flag explicitly so the manifest is unambiguous.

## 1. Configure a new environment

Run the launcher on node 0. It reaches the other three nodes over SSH. The AE
directory, vLLM checkout, model, dataset, and artifact root must be visible at
the same absolute paths on all four nodes.

```bash
cd <path-to>/NanoDeploy-July/ae_scripts

export VLLM_4NODE_H200_MASTER_ADDR=<node-0-IP-reachable-by-workers>
export VLLM_4NODE_REMOTE_HOSTS=<worker1,worker2,worker3>
export VLLM_WORKDIR=<path-to-vllm-checkout>
export VLLM_E2E_MODEL_PATH=<path-to-DeepSeek-V3-checkpoint>
export VLLM_E2E_DATASET_PATH=<path-to-issue01-random-csv>
export VLLM_E2E_ARTIFACT_ROOT=<shared-output-directory>
```

For the four-node H200 cluster, do not forward the head node's expanded shell
environment to every worker. By default, the runner sources `/root/.zshrc`
independently inside the head container and inside each worker container. The
path is the same, but each node may define its own `PATH`, `LD_LIBRARY_PATH`,
NCCL, and NVSHMEM settings. This is the same environment handling used by
`start_multinode_offline_profile_dbo.sh`. `--ssh-config` can replace the
default `/root/.ssh/config`.

First generate and inspect all four launch commands without starting anything:

```bash
python3 start-e2e/vllm/launch_vllm_e2e.py \
  --preset fig5 --fig5-case attention \
  --dry-run --ignore-historical-skips
```

Inspect `case_manifest.json`, `frontend.command.sh`, and `rank{1,2,3}.command.sh`
in the printed output directory. In particular, verify `master_addr`, all SSH
hosts, model/dataset paths, and the Python environment before removing
`--dry-run`.

## 2. Start an experiment

Run the three source cases separately so each output and failure is easy to
identify:

```bash
# Attention source: least-batch implementation, rate 30
python3 start-e2e/vllm/launch_vllm_e2e.py \
  --preset fig5 --fig5-case attention \
  --ignore-historical-skips

# DeepEP source: least-cache, rate 30
python3 start-e2e/vllm/launch_vllm_e2e.py \
  --preset fig5 --fig5-case deepep \
  --ignore-historical-skips

# HoL source: least-batch implementation, rate 45
python3 start-e2e/vllm/launch_vllm_e2e.py \
  --preset fig5 --fig5-case hol \
  --ignore-historical-skips
```

`--fig5-case all` also runs all three sequentially. Use it only when one long
launcher process is desirable.

For a non-paper case, omit `--preset` and set, for example,
`--dispatch-policy`, `--request-rates`, `--gpu-memory-utilization`,
`--max-num-seqs`, and `--bench-duration-sec` explicitly. See `--help` for the
full interface.

## 3. Raw outputs and the boundary with figures

Each case directory contains at least:

```text
case_manifest.json       resolved settings, topology, status, exact commands
frontend.command.sh      node-0 command
rank{1,2,3}.command.sh   worker commands
frontend.log             all DP-engine periodic statistics
rank{1,2,3}.log          worker startup/runtime logs
benchmark/               request-level and aggregate E2E results
```

`start-e2e/vllm` stops here. For Fig. 5, pass the appropriate `frontend.log`
to `fig5/extract_vllm_rank_snapshot.py`; that script owns rank alignment,
snapshot selection, and token/batch conversion.

## Historical Fig. 5 sources

The released data came from the following case directories under the vLLM
checkout's `offline_bench/manual_multinode` directory:

```text
# Attention, rate 30
DPSK/issue01_random/dp32-mem90-bs256-rate30-dur600/20260404-121436

# DeepEP, rate 30
DPSK/issue01_random/dp32-dispatch_least_cache-mem87-bs256-rate30-dur600/20260407-033129

# HoL, rate 45
DPSK/issue01_random/dp32-mem90-bs256-rate45-dur600/20260406-034407
```

The historical DeepEP service case eventually hit an OOM and its manifest is
`failed`; Fig. 5 uses the 60% snapshot recorded before that failure. A new run
should be treated as valid only if the requested snapshot exists and all 32
ranks are complete; the Fig. 5 extractor checks that condition.
