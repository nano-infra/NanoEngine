# SP8 Minimal Pure Decode Profile Handoff

## Purpose

Run the minimum 8-GPU profile needed to decide the provisional SP<=8 decode
scheduler policy.

This is a pure decode benchmark:

```text
dummy_prefill=True
dummy_weight=True
mode=decode
loop_count=1
cuda_graph_mode=full
SP=8
EP=8
d_attn in {1,2,4,8}
```

Do not run prefill.

## Important

All GPU/Ray/CUDA commands must be run with elevated permission.

Before profiling, check the cluster is clean:

```bash
ray status
nvidia-smi
```

Expected:

```text
Ray: 0/16 GPU in use, or at least one full 8-GPU node free
local nvidia-smi: no large residual memory allocations
```

If `nvidia-smi` shows large memory usage with no process listed, do not start the
profile. Clear the stale GPU/Ray state first.

One failed attempt with `--gpu-memory-utilization 0.90` hit NVSHMEM
`cuMemCreate failed` during worker initialization, so start with `0.70`. Increase
only if the runner reports too few KV blocks for the target point.

## Profile Matrix

Run B=16 first. It is the minimum matrix that maps to the current data and can
answer whether scale-up is needed while `d=1` fits.

Comparable memory-fit points:

| B | L | W_attn | d values | purpose |
|---:|---:|---:|---|---|
| 16 | 16384 | 262144 | 1,2,4,8 | mid context sanity |
| 16 | 40960 | 655360 | 1,2,4,8 | long context, >=640K |
| 16 | 53248 | 851968 | 1,2,4,8 | near single-rank capacity |

Memory-forced point:

| B | L | W_attn | d values | purpose |
|---:|---:|---:|---|---|
| 16 | 65536 | 1048576 | 2,4,8 | d=1 should be skipped/OOM |

If time permits, add B=32:

| B | L | W_attn | d values |
|---:|---:|---:|---|
| 32 | 20480 | 655360 | 1,2,4,8 |
| 32 | 32768 | 1048576 | 2,4,8 |

## Commands

### 1. B=16, d=1-fit comparable points

```bash
python docs-dev/loongserve_decode_profile_runner.py \
  --sp 8 \
  --ep 8 \
  --batches 16 \
  --lengths 16384,40960,53248 \
  --dops 1,2,4,8 \
  --warmup 3 \
  --steps 10 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 131072 \
  --out docs-dev/profile-results/loongserve_sp8_min_b16_fit_20260709.jsonl
```

### 2. B=16, memory-forced point

```bash
python docs-dev/loongserve_decode_profile_runner.py \
  --sp 8 \
  --ep 8 \
  --batches 16 \
  --lengths 65536 \
  --dops 2,4,8 \
  --warmup 3 \
  --steps 10 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 131072 \
  --out docs-dev/profile-results/loongserve_sp8_min_b16_1m_20260709.jsonl
```

### 3. Optional B=32 extension

```bash
python docs-dev/loongserve_decode_profile_runner.py \
  --sp 8 \
  --ep 8 \
  --batches 32 \
  --lengths 20480 \
  --dops 1,2,4,8 \
  --warmup 3 \
  --steps 10 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 131072 \
  --out docs-dev/profile-results/loongserve_sp8_min_b32_fit_20260709.jsonl
```

```bash
python docs-dev/loongserve_decode_profile_runner.py \
  --sp 8 \
  --ep 8 \
  --batches 32 \
  --lengths 32768 \
  --dops 2,4,8 \
  --warmup 3 \
  --steps 10 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 131072 \
  --out docs-dev/profile-results/loongserve_sp8_min_b32_1m_20260709.jsonl
```

## What To Check

For each CSV/JSONL row, verify:

```text
batch_size_scheduled == B
sp_size_hist matches the requested d
occupied and append ranks have size d
sp_send_counts/sp_recv_counts outside active ranks are zero if node-local comm is expected
step_p90_ms is stable relative to p50
```

Expected current decision rule if results match earlier SP16 data:

```text
if d=1 fits:
    choose d=1
elif d=8 fits:
    choose d=8
elif d=4 fits:
    choose d=4
elif d=2 fits:
    choose d=2
else:
    no normal SP<=8 plan
```

## Deliverables

Save raw outputs under:

```text
docs-dev/profile-results/
```

Required files:

```text
loongserve_sp8_min_b16_fit_20260709.jsonl
loongserve_sp8_min_b16_fit_20260709.csv
loongserve_sp8_min_b16_1m_20260709.jsonl
loongserve_sp8_min_b16_1m_20260709.csv
```

Optional:

```text
loongserve_sp8_min_b32_fit_20260709.jsonl
loongserve_sp8_min_b32_fit_20260709.csv
loongserve_sp8_min_b32_1m_20260709.jsonl
loongserve_sp8_min_b32_1m_20260709.csv
```

After the run, paste or summarize this table:

| B | L | W_attn | d=1 | d=2 | d=4 | d=8 | best viable |
|---:|---:|---:|---:|---:|---:|---:|---:|

Use `step_p90_ms` for threshold decisions and keep `step_mean_ms` as secondary
context.
