# K3 partitioning experiments

Run `python bench/k3_partition/analyze_model_costs.py` to reproduce the
shape-derived FLOP, cache, and expanded-K/V traffic values stored in
`results/model_costs.json`.

The collective microbenchmark uses BF16 hidden-state tensors with K3 hidden
width 7168. It measures NCCL all-reduce, reduce-scatter, all-gather, and
all-to-all on one NVLink-connected B300 node at world sizes 2 and 4.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  bench/k3_partition/benchmark_collectives.py \
  --output bench/k3_partition/results/collectives_tp2.csv
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  bench/k3_partition/benchmark_collectives.py \
  --output bench/k3_partition/results/collectives_tp4.csv
MPLBACKEND=Agg python bench/k3_partition/plot_collectives.py
```

Each point uses 5 warm-up and 20 measured iterations. CUDA events measure the
average device-side duration. `logical_payload_gbps` divides the complete
logical hidden-state tensor size by latency; it is an application-facing rate,
not topology-adjusted NCCL bus bandwidth. The local experiment characterizes
intra-node communication only and does not claim to measure the target
DP2/attention-TP8/EP16 multi-node topology.

## Chapter 8: 16-GB200 evaluation (2026-09-08)

The GB200 experiments use four Ray nodes with four GPUs each. Each worker owns
one GPU; contiguous groups of 2/4 stay within a node, while 8/16 span nodes.
Ray launches actors only; CUDA events measure the actual device work. A driver
releases its actors on completion or failure. Run one driver at a time; every
stage uses all 16 GPUs and reports the maximum participating-rank latency in
the article. The initial stages use synthetic weights; the extended stages below load real checkpoint components or the complete model.

From the repository root:

```bash
ray status
python bench/k3_partition/benchmark_ray.py --stages baseline graphs
python bench/k3_partition/benchmark_ray.py --stages kda \
  --output-dir bench/k3_partition/results/gb200_graph_kda
python bench/k3_partition/benchmark_ray.py --stages cp \
  --output-dir bench/k3_partition/results/gb200_graph_cp
python bench/k3_partition/benchmark_ray.py --stages moe \
  --output-dir bench/k3_partition/results/gb200_graph_moe
python bench/k3_partition/benchmark_ray.py --stages transition \
  --output-dir bench/k3_partition/results/gb200_transition
python bench/k3_partition/benchmark_ray.py --stages validation \
  --output-dir bench/k3_partition/results/gb200_validation
python bench/k3_partition/audit_checkpoint.py
python bench/k3_partition/analyze_joint.py
python bench/k3_partition/summarize_gb200.py
```

Use a different `--output-dir` to preserve the recorded results. These commands
use the currently installed PyTorch/CUDA/FlashAttention/DeepGEMM and local
NanoDeploy code; changing any of those can change performance. The 2026-09-08
run used PyTorch 2.11.0+cu130 and CUDA 13.0. Per-worker software versions, device
memory and local topology are saved in each environment JSON. The initial
`gb200/environment.json` belongs to the first baseline run.

| Data | Timed boundary and method |
| --- | --- |
| `gb200/roofline.csv` | 256 MiB HBM copy (read+write byte accounting), plus BF16 `[M,7168] @ [7168,12288/TP]` GEMMs. All GPUs run independently. Copy: 5 warmups/100 iterations; GEMMs: 5/30. |
| `gb200/collectives.csv` | Eager NCCL all-reduce, RS, AG, equal-split all-to-all and directly timed RS→AG. 5 warmups, 3 trials × 30 iterations. RS/AG have nonzero ownership checks. |
| `gb200/graphs.csv` | Same collective matrix; 8 operations per captured graph, 5 warmups, 3 trials × 30 replays, divided by 8. Small rows are TP-padded. `logical_bytes` is the complete tensor for RS/AG/AR; all-to-all uses that many bytes **per rank**. It is not link bus bandwidth. |
| `gb200_graph_kda/kda.csv` | Full current KDA, including output all-reduce, at TP1/2/4/8/16. Eager: median of 15 iterations after 3 warmups, state reset outside timing. Decode graph: 3 warmups/30 replays. Prefill graph is intentionally blank: its current convolution path contains CPU `.item()` synchronization. Zero inputs/weights give stable zero recurrent state during replay. |
| `gb200_graph_cp/cp.csv` | Expanded cached-KV FlashAttention plus fused FP32 LSE merge; 3 warmups/10 eager iterations and 3/20 graph replays. `tp` means **effective head TP**, not nested projection TP. No Q exchange, compressed-cache restore, KV expansion or fresh causal triangle is timed. One-query rows are not production absorbed Decode. |
| `gb200_graph_moe/moe.csv` | Production pre-dispatch + fused MXFP4 MegaMoE, including dispatch/combine, but excluding router/shared/latent projections. Three trials of 5 warmups/30 iterations or graph replays. Includes extreme hot-expert routing and uneven source ownership. Blank dynamic memory for source-placement-only rows means unmeasured, not zero. |
| `gb200_transition/transition.csv` | Same routed branch with optional source redistribution and inverse mapping, including activation, expert-ID and score transfers. Three trials of 5 warmups/30 graph replays. Nonzero row IDs and routing metadata verify exact ownership. |
| `gb200_validation/validation.csv` | Fused CP merge checks against FP64 composition: Q=1/5/33, noncontiguous LSE, large logits, masked ranks and fully masked rows, across CP2/4/8/16. |
| `gb200/joint_model.json` | Weight/cache/20-GiB-allowance capacity estimates and phase-specific algorithmic FLOPs. CP admission counts are design estimates, not OOM-tested limits. |
| `gb200/checkpoint_inventory.json` | File-size versus safetensors-header audit. It does not read or validate all tensor payload bytes. |

`gb200/kda.csv`, `gb200/cp.csv` and `gb200/moe.csv` retain the initial pilot
sweeps. The pilot CP merge used eager PyTorch packing, later replaced by the
preallocated Triton merge in `cp_merge.py`. The pilot MoE used fewer repetitions
and contained launch/timing outliers; the report uses the later three-trial
graph/eager sweep. Do not mix pilot and final measurements when comparing
execution modes.

The current DeepGEMM MegaMoE allocator must use PyTorch symmetric-memory backend
`NCCL` on this multi-node job. The default `CUDA` backend rejected overlapping
local device ordinals across nodes. The harness selects `NCCL` before its first
symmetric allocation; the full-serving harness selects it through its explicit Ray worker hook.
No packages are installed. Group-specific symmetric buffers stay live and are reused; their
reported incremental reservations are not layer-count multipliers.

`component_kernels.py` contains experimental harnesses, not new production CP or
expert-TP support. `cp_merge.py` retains the correct log-sum-exp weighting and
uses two collectives plus two Triton kernels with preallocated buffers. The
transition experiment moves only routed latent rows and metadata, preserving
the original hidden/shared branch ownership. Integrating either optimization
into serving requires cache/scheduler/graph and residual ownership work.

The supplied complete checkpoint is `/hgpfs/Kimi-K3`: all 96 indexed shards
cover their declared payload lengths. `gb200/hgpfs_checkpoint_inventory.json`
records that audit. `gb200/checkpoint_inventory.json` is the historical audit
of a different, incomplete `/mnt/mnt/public/Kimi-K3` directory; it does not
characterize the supplied model.

## Extended phase/context and real-weight experiments

Use module execution from the repository root. These stages load real
checkpoint components and use the installed native kernels:

```bash
python -m bench.k3_partition.benchmark_ray --stages checkpoint \
  --output-dir bench/k3_partition/results/gb200_checkpoint
python -m bench.k3_partition.benchmark_ray --stages mla \
  --output-dir bench/k3_partition/results/new_mla
python -m bench.k3_partition.benchmark_ray --stages equal_mla \
  --output-dir bench/k3_partition/results/new_equal_mla
python -m bench.k3_partition.benchmark_ray --stages real_moe equal_kda \
  --output-dir bench/k3_partition/results/new_real_moe
python -m bench.k3_partition.benchmark_ray --stages paged_cp cold \
  --output-dir bench/k3_partition/results/new_paged_cp_cold
python -m bench.k3_partition.benchmark_ray --stages expert_tp \
  --output-dir bench/k3_partition/results/new_expert_tp
python -m bench.k3_partition.benchmark_ray --world-size 8 --stages graphs kda \
  --output-dir bench/k3_partition/results/new_world8
python -m bench.k3_partition.analyze_serving
python -m bench.k3_partition.summarize_extended
python -m bench.k3_partition.plot_serving
python -m bench.k3_partition.full_serving --tp 8 --memory-utilization .88
python -m bench.k3_partition.full_serving --tp 16 --memory-utilization .88 \
  --contexts 5 8192 131072 1048560 --output-dir bench/k3_partition/results/new_full_tp16
python -m bench.k3_partition.full_serving --tp 4 --memory-utilization .91 \
  --contexts 5 8192 131072 1048560 --output-dir bench/k3_partition/results/new_full_tp4
```

`--world-size` supports 1/2/4/8/16 for the generic/attention stages. The legacy
CP/MoE matrix assumes 16 ranks. `--gpu-offset` selects an allocation offset in
sorted-node order; disjoint small worlds used offsets 0/8/12/14 for world
8/4/2/1. This allocates disjoint resources, not exclusive network isolation.
New environment files include benchmark/runtime source SHA-256 fingerprints.

| Data | Interpretation |
| --- | --- |
| `gb200_checkpoint/checkpoint.csv` | All 24 real MLA layers, BF16/FP8 cache, causal reference versus chunked Prefill and four Decode steps, graph/eager match and dummy-cache preservation. 48 layer/dtype rows. |
| `gb200_mla/mla.csv` | Combined complete MLA layer matrix, 350 points × 16 ranks. Checkpoint layer 3; synthetic latent histories on distinct pages. Prefill: 3 trials × 3 eager forwards; Decode: 3 trials × 20 graph replays. Includes Q/KV projections, cache write, restore/expansion or paged absorbed attention, gate/value/output projections and output all-reduce. |
| `gb200_equal_mla/equal_mla.csv` | Additional Decode local batches 2/4/16 at all five TP sizes and six contexts, both cache formats. Use with the complete MLA matrix to compare the same global request population. |
| `gb200_real_moe/equal_kda.csv` | Additional KDA local Decode batches 2/4/16/64 at all five TP sizes; same graph procedure as the initial KDA sweep. |
| `gb200_world{1,2,4,8}` | Actual independent NCCL worlds, including their TP subgroup sweep. Do not confuse these isolated components with full-model placement feasibility. |
| `gb200_mla_final/paged_cp.csv` | 426 points × 16 ranks: native paged absorbed MLA, Q all-gather/packing, FP32 LSE merge and original-head restoration; three trials × 20 graph replays. Excludes Q/KV/value/output projections and cache append. Nonzero small cases checked against independent dense attention. |
| `gb200_real_moe/real_moe.csv` | Complete checkpoint FFN layer 1: real router, shared experts, latent projections/norm and packed routed experts, including overlap and dispatch/combine. Synthetic normal inputs; not a production-traffic routing histogram. EP4/8/16, source and batch sweeps, three trials × 20 graph replays. |
| `gb200_cold/cold.csv` | Sequential cold real-weight MLA layer Prefill, every chunk appended from empty cache, TP4/8/16, BF16/FP8, endpoints 8K/128K/1M; extra TP16 4K/16K chunks. One sweep per point, 352 per-rank rows. Last-chunk split/unsplit output checked at each endpoint. |
| `gb200_expert_tp/expert_tp.csv` | Native routed expert TP1/2/4 across 16 GPUs, real checkpoint weights, balanced/hot routes, equal global inputs, input gather and output reduce-scatter included. TP4 pads logical width 768 to physical 1024; charge its extra storage. |
| `gb200_expert_tp_unpadded` | Initial native expert-TP comparison: real checkpoint weights, EP16 versus EP8×TP2, exact token-ownership conversion and output comparison. EP4×TP4's 768-wide intermediate failed a native TMA-alignment requirement; failure and valid preceding rows retained. |
| `serving_model` | B200/GB200/B300 hardware-budget sensitivity, worlds 1/2/4/8/16/32, context and cache-format capacity grid, plus cold/suffix/Decode arithmetic. Analytical, not all combinations executed. |
| `gb200_full_serving`, `gb200_full_tp4`, `gb200_full_tp16` | Full-checkpoint startup and sequential context sweeps at attention TP8, TP4 and TP16, respectively, all EP16. Read `result.json` for actual completion status, timing and any error; prior failed attempts are retained separately. |

`gb200_mla/mla.csv` combines TP2/4/8/16 from
`gb200_mla_fixed_partial/mla.csv` and TP1 from
`gb200_mla_final/mla_tp1.csv`. The former already includes bounded BF16 prefix
expansion and 24→32 query-head padding; the latter additionally pads 96→128
for the installed large-batch kernel. The initial `gb200_mla_before` preserves
TP8/16 timings before the BF16-prefix fix. `gb200_mla_final/equal_mla.csv` is an
initial TP1-only sweep; the complete equal-load dataset is `gb200_equal_mla`.
Progress directories are recovery snapshots, not additional independent trials.

Current K3 `auto` cache is BF16, 1,152 bytes/token/MLA layer. Explicit Blackwell
FP8 uses raw 576-byte rows. The historical 656-byte mixed cache is separate;
new capacity tables do not silently substitute it for either runtime format.
The 20-GiB allowance in analytical tables is a planning parameter, not a
full-model measured reserve or a `gpu_memory_utilization` setting.

The full-serving harness selects NCCL symmetric memory in a Ray worker setup
hook and limits CPU threads. It loads all 93 layers, uses TP8/DP2/EP16, requests
a maximum context of 1M and explicitly selects FP8 cache. The later run sets
GPU memory utilization to 0.88 after the 0.95 run OOMed during long Prefill.
Timing includes client/driver/worker execution; the first execution of each
length may include shape-specific JIT. Inputs repeat a short token sequence,
so this is a capacity/execution test, not a long-text reasoning-quality test.
The harness counts completions only after the prompt is fully consumed: the
current offline helper also exposes intermediate Prefill emissions, which are
recorded separately to avoid treating them as generated continuation tokens.

All three full-model layouts completed 1,048,560 prompt tokens plus eight true
continuation tokens; maximum context is 1,048,576, not a fixed request length.
These are single-run capacity/execution measurements, not p99, concurrent
saturation or long-document accuracy results. TP4 uses utilization .91, TP8/16
use .88. Retained failures show the original lazy NCCL communicator registration
crash and the .95-utilization Prefill OOM. The runtime now primes the CUDA
communicator once before first MegaMoE symmetric-buffer registration.

The MLA runtime fixes pad unsupported 24/96 query heads to 32/128 and bound
BF16 prefix expansion on FA3/FA4, preserving the existing fallback on other
backends. The final checkpoint/generation runs and component results include
the supported-kernel changes. Run the relevant regression checks on Blackwell:

```bash
python -m pytest -q tests/test_mla_decode_head_padding.py \
  tests/test_mla_prefix_chunk.py tests/test_mla_cache_dtype.py \
  tests/test_megamoe_backend.py tests/test_kimi_k3_mega_capacity.py
python -m mkdocs build --strict -f docs/mkdocs.yml --site-dir /tmp/k3-evaluation-docs
```

`results/validation_final/manifest.json` indexes final datasets with SHA-256
hashes, current source fingerprints and verification outcomes. Its source
hashes describe the final tree; historical run fingerprints remain in each
run's environment metadata. Verification logs preserve the initial two-value
FP8 tolerance failure and the subsequent passing norm-bounded regression.
All 62 distinct selected regression cases were verified across the initial
suite and the four-case rerun; the strict documentation build also passed.
