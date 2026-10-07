# Fig. 13: Per-layer phase breakdown

Fig. 13 compares NanoDeploy DCP with four vLLM DP/DCP configurations under
four controlled KV-cache distributions. Profile collection uses the reusable
launchers in [`../start-profile`](../start-profile/README.md); this directory
owns the distributions, trace parser, plotting data, and plotter.

## Resource and time estimate

Reserve 4 nodes with 8 H200 GPUs per node for the full workflow.

| Stage | GPUs | Expected time |
| --- | ---: | ---: |
| NanoDeploy profiling (4 cases) | 32 H200 GPUs | 8–10 minutes |
| vLLM profiling (16 cases) | 32 H200 GPUs | 50–55 minutes |

The GPU experiments take about 60–65 minutes in total.

## Workloads

Every case uses four nodes with eight GPUs per node. Each GPU has 64 short
requests with a 2,048-token KV cache. The four cases add 1, 3, 5, or 7 long
requests per node, each with a 512 Ki-token KV cache.

The same ordered lengths are stored in the format required by each system:

```text
kv-distributions/
├── nano/<case>/processed_input_3d.json
├── vllm/<case>.json
└── summary.json
```

Regenerate or verify them with:

```bash
python3 fig13/generate_kv_distributions.py
python3 fig13/generate_kv_distributions.py --check
```

## Four-node paper reproduction

Complete the four-node setup in the root README, then run the following steps in
order from the `ae_scripts` directory. The measured allocation used `h200-rjob0` as
the head node and `h200-rjob1`, `h200-rjob2`, and `h200-rjob3` as workers.

### 1. Collect NanoDeploy profiles

This command uses the Ray head at `10.102.252.174:6380` and reproduces
`DP4-SP8`, `hao_basic`, the `long_short` policy, and threshold 100,000:

```bash
python3 start-profile/run_nano_profiling.py \
  --ray-address 10.102.252.174:6380 \
  --master-address 10.102.252.174:29500 \
  --output-root fig13/profiling_results/nano \
  --input mix_1x512k_pernode_plus_64x2048_pergpu=fig13/kv-distributions/nano/mix_1x512k_pernode_plus_64x2048_pergpu/processed_input_3d.json \
  --input mix_3x512k_pernode_plus_64x2048_pergpu=fig13/kv-distributions/nano/mix_3x512k_pernode_plus_64x2048_pergpu/processed_input_3d.json \
  --input mix_5x512k_pernode_plus_64x2048_pergpu=fig13/kv-distributions/nano/mix_5x512k_pernode_plus_64x2048_pergpu/processed_input_3d.json \
  --input mix_7x512k_pernode_plus_64x2048_pergpu=fig13/kv-distributions/nano/mix_7x512k_pernode_plus_64x2048_pergpu/processed_input_3d.json \
  --config dp4sp8 --sp-backend hao_basic --sp-size-policy long_short \
  --long-request-sp-threshold 100000 \
  --max-num-seqs 80 --max-num-recv-seqs 16 --max-num-send-seqs 16
```

The 128 traces are written below `fig13/profiling_results/nano/`.

### 2. Collect vLLM profiles

The `4node_h200` preset uses `h200-rjob0` as the head node and
`h200-rjob1/2/3` as workers. Run all four strategies and distributions:

```bash
python3 start-profile/run_vllm_profiling.py \
  --artifact-root fig13/profiling_results/vllm \
  --strategies dp32 dp16cp2 dp8dcp4 dp4dcp8 \
  --input mix_1x512k_pernode_plus_64x2048_pergpu=fig13/kv-distributions/vllm/mix_1x512k_pernode_plus_64x2048_pergpu.json \
  --input mix_3x512k_pernode_plus_64x2048_pergpu=fig13/kv-distributions/vllm/mix_3x512k_pernode_plus_64x2048_pergpu.json \
  --input mix_5x512k_pernode_plus_64x2048_pergpu=fig13/kv-distributions/vllm/mix_5x512k_pernode_plus_64x2048_pergpu.json \
  --input mix_7x512k_pernode_plus_64x2048_pergpu=fig13/kv-distributions/vllm/mix_7x512k_pernode_plus_64x2048_pergpu.json \
  --cluster 4node_h200 --model deepseek_v3_1024k \
  --dispatch-policy least_batch --routing-mode explicit_rank_replay \
  --strategy-max-num-seqs dp4dcp8=528 \
  --strategy-max-num-seqs dp8dcp4=272 \
  --strategy-max-num-seqs dp16cp2=144 \
  --strategy-max-num-seqs dp32=80 \
  --warmup-requests 32 --profile-delay-iterations 32
```

The 512 traces are written below `fig13/profiling_results/vllm/`.

Both commands accept `--dry-run` to print every expanded launch without using
GPUs.

### 3. Parse profiles

Parse all 20 cases after both profile collections finish:

```bash
python3 fig13/parse_phase_breakdown_profiles.py matrix \
  --nano-root fig13/profiling_results/nano \
  --vllm-root fig13/profiling_results/vllm \
  --output fig13/phase_breakdown_plot_data.generated.csv \
  --trace-summary fig13/phase_breakdown_trace_summary.generated.csv
```

The command writes 20 rows to
`fig13/phase_breakdown_plot_data.generated.csv` and 640 rows to
`fig13/phase_breakdown_trace_summary.generated.csv`, excluding CSV headers. It
rejects incomplete or mixed trace sets.

The checked-in [`phase_breakdown_plot_data.csv`](phase_breakdown_plot_data.csv)
is the paper's manually read reference. The generated file is deliberately
separate: it uses automated trace aggregation instead of selecting one layer
visually, so small numerical differences from the reference are expected.

### 4. Plot

```bash
python3 fig13/visualize_phase_breakdown_by_long_node.py \
  --input fig13/phase_breakdown_plot_data.generated.csv \
  --output fig13/fig13.generated
```

The command creates `fig13/fig13.generated.png` and
`fig13/fig13.generated.pdf`. The plotter also accepts a non-empty subset of the
20 rows for pipeline debugging; the paper reproduction requires all rows.

## CP Cost definition

`CP Cost` measures the communication introduced by context parallelism. Use
the duration of CUDA events whose trace category is `kernel`; do not use the
CPU or user-annotation events with similar names. Match the following stable
substrings in `traceEvents[].name`:

| System/configuration | Included operation | Kernel-name substring |
|---|---|---|
| vLLM DCP (`dp16cp2`, `dp8dcp4`, `dp4dcp8`) | All-Gather | `ncclDevKernel_AllGather_RING_LL` |
| vLLM DCP (`dp16cp2`, `dp8dcp4`, `dp4dcp8`) | Send/receive | `ncclDevKernel_SendRecv` |
| vLLM `dp16cp2` | All-Reduce | `two_shot_all_reduce_kernel_inplace` |
| vLLM `dp8dcp4`, `dp4dcp8` | All-Reduce | `multimem_all_reduce_kernel` |
| NanoDeploy `nano dcp` | dLSLIME all-to-all | `dlslime::intranode_alltoall_kernel` |

Within each selected layer and profiler iteration, calculate vLLM `CP Cost` as
the sum of the AG, send/receive, and strategy-specific AR kernel durations.
Calculate NanoDeploy `CP Cost` as the sum of the dLSLIME all-to-all kernel
durations. Record the result in microseconds in the CSV `cp_cost` column.
For vLLM `dp32`, record `0` because pure DP does not use context parallelism.
