# Ascend NPU Profiler — Design Walkthrough

## Overview

NanoInfra supports timeline profiling on both CUDA and Ascend NPU hardware.
On Ascend, the standard `torch.profiler` with `ProfilerActivity.CUDA` cannot
capture NPU kernel execution. We use `torch_npu.profiler` with
`ProfilerActivity.NPU` instead, matching the approach used by vllm-ascend.

## Files

| File | What changed |
|------|-------------|
| `nanodeploy/config.py:67-70` | Config fields: `enable_profiler`, `profiler_start_step`, `profiling_step`, `profiler_dir` |
| `nanodeploy/worker/model_runner.py:202-232` | Init: device-aware profiler creation (`npu` vs `cuda`) |
| `nanodeploy/worker/model_runner.py:297-331` | `_create_npu_profiler()` method |
| `nanodeploy/worker/model_runner.py:1091-1151` | Run loop: `.start()` / `.step()` / `.stop()` (shared by both backends) |

## Config

```python
# nanodeploy/config.py
enable_profiler: bool = False          # toggle profiling
profiler_start_step: int = 16          # which decode step to start
profiling_step: int = 32               # how many steps to capture
profiler_dir: str = "./profiler_res"   # output directory
```

CLI usage:
```bash
python examples/non_disagg.py \
  --enable_profiler \
  --profiler_start_step 16 \
  --profiling_step 32 \
  --profiler_dir ./profiler_res \
  --backend_type ascend \
  --model /datapool/models/Qwen3-235B-A22B \
  ...
```

## Initialization (model_runner.py:202-232)

```python
self.run_count = 0
self.profiler = None
if getattr(config, "enable_profiler", False):
    ...
    if self._dev == "npu":
        self.profiler = self._create_npu_profiler(profiler_dir, rank)
    else:
        self.profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            ...
        )
```

The branch on `self._dev` is the only divergence point. Both paths produce an
object with the same `.start()` / `.step()` / `.stop()` interface, so the run
loop is unchanged.

## NPU Profiler Setup (model_runner.py:301-331)

```python
def _create_npu_profiler(self, profiler_dir: str, rank: int):
    import torch_npu

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
        gc_detect_threshold=None,
    )

    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=None,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            profiler_dir,
            worker_name=f"{self.engine_id}_rank_{rank}",
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        experimental_config=experimental_config,
    )
```

### Key config choices (aligned with vllm-ascend)

| Parameter | Value | Why |
|-----------|-------|-----|
| `ProfilerActivity.NPU` | — | Captures NPU kernel launches and AiCore execution (replaces CUDA) |
| `ProfilerLevel.Level1` | — | Operator-level timeline without per-kernel hardware counters (low overhead) |
| `data_simplification=True` | — | Reduces trace file size by omitting redundant metadata |
| `with_stack=False` | — | Python stack capture is expensive; disabled to keep overhead low |
| `profile_memory=False` | — | Memory tracking adds overhead; enable only when debugging OOM |
| `msprof_tx=False` | — | MSPTI-based dumping not needed for timeline analysis |
| `aic_metrics=AiCoreNone` | — | Skip hardware counter collection (Level1 is sufficient for timeline) |
| `ExportType.Text` | — | Generates ascend_pt format, convertible to Chrome trace |

### CUDA profiler (unchanged)

For GPU runs, the original `torch.profiler.profile` with `ProfilerActivity.CUDA`
is used with `record_shapes=True`, `profile_memory=True`, `with_stack=True` —
heavier but acceptable on CUDA.

## Run Loop (model_runner.py:1091-1151)

```python
for i in range(loop_count):
    # --- start ---
    if self.profiler and self.run_count == self.profiler_start_step:
        self.profiler.start()

    # ... prepare inputs, run_model, sample, all_reduce ...

    # --- step / stop ---
    if self.profiler and self.run_count >= self.profiler_start_step:
        if self.run_count < self.profiler_end_step:
            self.profiler.step()
        if self.run_count == self.profiler_end_step - 1:
            self.profiler.stop()

    self.run_count += 1
```

Both `torch.profiler` and `torch_npu.profiler` expose the same context-manager
and manual-control API, so no code change is needed here.

### What gets captured per step

One profiler step captures one full decode iteration:
1. `prepare_decode_bytes` — input tensor construction
2. `run_model` — full model forward (attention + MoE + layernorm)
   - If graph capture is enabled: `graph.replay()` (the graph replay itself
     shows up as a single NPU op; expand "AscendCL" lane for kernel detail)
   - If eager: individual ops visible directly
3. `sampler` — logit sampling
4. `dist.all_reduce` — TP synchronization

## Output & Viewing

### Output directory structure

```
profiler_res/
├── {engine_id}_rank_0.{timestamp}.pt.trace.json    # tensorboard format
├── {engine_id}_rank_1.{timestamp}.pt.trace.json
├── ...
└── ASCEND_PROFILER_OUTPUT/       # ascend_pt raw data (Level1)
    ├── PROF_XXX/
    │   ├── device_0/
    │   ├── host/
    │   └── mindstudio_profiler_output/
    └── ...
```

### Option 1: MindStudio Insight (recommended for Ascend)

```bash
# Install MindStudio Insight (comes with CANN toolkit)
# Open the PROF_XXX directory directly
msprof --export=on --output=profiler_res/ASCEND_PROFILER_OUTPUT/PROF_XXX
```

This gives the most detailed NPU timeline: AiCore utilization, memory bandwidth,
inter-op gaps, HCCL collective duration.

### Option 2: TensorBoard

```bash
pip install tensorboard torch-tb-profiler
tensorboard --logdir=./profiler_res --port=6007
```

Navigate to the **PyTorch Profiler** tab → **Trace** view. Shows CPU + NPU
timeline side by side. Less detail than MindStudio but more familiar.

### Option 3: Chrome trace (manual conversion)

```python
import torch_npu
torch_npu.profiler.profiler.analyse(
    profiler_res_path="profiler_res/ASCEND_PROFILER_OUTPUT/PROF_XXX"
)
```

Then open `chrome://tracing` and load the generated JSON.

## What to Look For in the Timeline

### Decode latency breakdown (Qwen3-235B-A22B, EP=8)

| Component | Expected % | What to check |
|-----------|-----------|---------------|
| Attention (npu_incre_flash_attention) | ~30-40% | Look for KV pre-gather overhead before the flash attention kernel |
| MoE dispatch (AllGather + argsort + grouped_matmul) | ~30-40% | Check AllGather duration, argsort gap, grouped_matmul kernel time |
| HCCL collectives (AllGather, ReduceScatter, AllReduce) | ~15-20% | Look for long waits / uneven load across ranks |
| Stream sync / graph overhead | ~5-10% | Any `synchronize()` calls showing up as idle gaps |
| Sampling + misc | ~5% | Should be negligible |

### Red flags

- **Long gaps between kernels**: Host-side Python overhead or stream sync
- **`synchronize()` in graph replay path**: `model_runner.py:1014` has a sync
  before `graph.replay()` — this is a known latency bottleneck
- **AiCpu fallback ops**: If `argsort` or other ops show "AiCpu" instead of
  "AiCore", they're running on the slow scalar processor. The float32 cast
  for argsort is specifically to avoid this.
- **Uneven HCCL times across ranks**: Indicates load imbalance in EP dispatch

## Comparison with vllm-ascend

| Aspect | NanoInfra | vllm-ascend |
|--------|-----------|-------------|
| Profiler library | `torch_npu.profiler` | `torch_npu.profiler` |
| Activities | CPU + NPU | CPU + NPU |
| ExperimentalConfig | Level1, Text, no AiCore metrics | Level1, Text, no AiCore metrics |
| Trigger mechanism | Step-based (start_step + duration) | API-based (`/start_profile`, `/stop_profile`) |
| Output handler | tensorboard_trace_handler | tensorboard_trace_handler |
| `with_stack` | False | False (configurable) |
| `profile_memory` | False | Configurable |
| MS Service Profiler | Not implemented | Supported (symbol YAML + msserviceprofiler) |

The core profiler setup is identical. The main difference is the trigger:
NanoInfra uses step-count-based start/stop (simpler, no API server needed),
while vllm-ascend supports dynamic start/stop via HTTP endpoints.

## Tuning the Profiler

### Capture more detail (higher overhead)

```python
# In _create_npu_profiler(), change:
profiler_level=torch_npu.profiler.ProfilerLevel.Level2,  # hardware counters
aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,  # pipeline util
profile_memory=True,   # memory allocation tracking
with_stack=True,       # Python call stacks
```

### Capture less (minimal overhead)

```python
profiler_level=torch_npu.profiler.ProfilerLevel.Level0,  # minimal
data_simplification=True,
record_shapes=False,
```

### Profile specific steps

Adjust via CLI:
```bash
--profiler_start_step 50   # skip warmup iterations
--profiling_step 10        # capture only 10 steps
```
