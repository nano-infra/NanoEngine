# Routing Metadata Fusion P/D Serving Validation

Date: 2026-08-23 UTC

> **Superseded for text validation.** This run used the Issue1% length CSV,
> which creates random token prompts and does not retain generated text. It is
> useful as a length-workload performance run, but it is not the requested
> ShareGPT text validation. See
> `RoutingMetadataFusionPDShareGPTValidation.md` for the corrected run.

## Result

PASS. The production fused SP graph metadata path completed a two-node,
real-weight P/D serving run without request failures or CUDA, NCCL, Ray, or
Python runtime errors. All 101 scheduled requests completed, both executors
shut down cleanly, and Ray reported all 16 GPUs free after the run.

The implementation under test was commit `3215d16` (`feat: fuse production SP
graph metadata updates`), with documentation commit `e92b976` at the benchmark
HEAD.

## Topology and workload

- Ray GCS: `10.102.243.60:6380`
- Prefill node: `10.102.98.154`, 8 H200 GPUs
- Decode node: `10.102.243.60`, 8 H200 GPUs
- Driver RDMA environment:
  - `SLIME_VISIBLE_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7`
  - `SLIME_GID_INDEX=3`
  - `SLIME_QP_NUM=4`
- Benchmark: `examples/bench_pd_serving.py`
- Parallel engine builder:
  `examples/pd_disagg_deepseek_v3_parallel.py::build_engines_parallel`
- Model: local DeepSeek-V3 snapshot, real weights
- Decode topology: `bucket-sp8`
- Dynamic SP policy:
  `1:1024-63488;5:63489-210944;6:210945-399360;7:399361-428032;8:428033-1048576`
- Arrival window: 120 seconds
- Target request rate: 1 request/second, Poisson, seed 0
- Sampled arrivals: 101 requests, or 0.842 requests/second
- Warmup: 8 P/D requests
- Maximum model length: 1,000,000 tokens
- Maximum sequences: 8

The CSV was:

```text
/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv
```

The default 910,000-token request filter reduced it from 60,000 to 59,955
rows. The eligible data contains 555 `long` rows and 59,400 `short` rows.

The script consumes eligible CSV rows in order. The 101 requests scheduled in
this two-minute run were all `short`; their prompt lengths ranged from 86 to
12,644 tokens and output lengths from 125 to 2,054 tokens. The first eligible
`long` row is at zero-based index 307. Consequently, this run validates the
Issue1% workload configuration and the real distributed P/D/fused-routing path,
but it does not validate an actually sampled `long` issue request.

## Command

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export SLIME_VISIBLE_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export SLIME_GID_INDEX=3
export SLIME_QP_NUM=4
python3 -u examples/bench_pd_serving.py \
  --duration 120 \
  --request-rate 1 \
  --dataset csv \
  --csv-path /mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv \
  --model-path /mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/e815299b0bcbac849fa540c768ef21845365c9eb \
  --ray-address 10.102.243.60:6380 \
  --prefill-master-address 10.102.98.154:6006 \
  --decode-master-address 10.102.243.60:6006 \
  --decode-topology bucket-sp8 \
  --dynamic-sp-bucket-policy '1:1024-63488;5:63489-210944;6:210945-399360;7:399361-428032;8:428033-1048576' \
  --max-model-len 1000000 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --warmup-requests 8 \
  --itl-log-path bench_logs/pd_issue1_rate1_2min_parallel_20260823_1340/itl_samples.jsonl \
  --text-output-path '' \
  --no-tqdm
```

## Initialization

- Prefill engine initialization: 364.57 seconds
- Decode engine initialization: 368.26 seconds
- Parallel initialization wall time: 368.26 seconds
- Decode graph capture per worker: 4 local graphs and 8 SP graphs
- P/D KV-transfer endpoints: connected
- P/D warmup: 36.54 seconds

## Serving results

| Metric | Result |
| --- | ---: |
| Requests sent/completed | 101 / 101 |
| Input tokens | 42,824 |
| Output tokens | 64,118 |
| Injection window | 120.00 s |
| Drain time | 162.34 s |
| Total measured time | 282.34 s |
| Throughput | 227.10 tokens/s |
| Average TTFT | 12,014.81 ms |
| Average E2E latency | 82.98 s |
| TPOT average | 100.04 ms/token |
| TPOT P50 / P90 / P95 / P99 | 98.97 / 106.11 / 107.49 / 108.99 ms/token |
| ITL with decode queue average | 110.39 ms/token |
| ITL with queue P50 / P90 / P95 / P99 | 101.37 / 134.77 / 141.05 / 148.31 ms/token |
| Goodput, queued TPOT under 100 ms | 1 / 101 (0.99%) |

The strict goodput SLO is below the measured queued-token latency median, so
the low goodput is consistent with the reported latency distribution rather
than a request failure.

## Artifacts and cleanup

- Full log:
  `bench_logs/pd_issue1_rate1_2min_parallel_20260823_1340/run.log`
- Per-request ITL samples (101 JSONL records):
  `bench_logs/pd_issue1_rate1_2min_parallel_20260823_1340/itl_samples.jsonl`
- Process exit code: 0
- Post-run Ray status: two active nodes, no failures or pending demands,
  `0.0/16.0 GPU` in use
