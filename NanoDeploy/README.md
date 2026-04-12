## Configuration Reference

### Engine Parameters

| Parameter                | Type  | Default            | Description                                |
| ------------------------ | ----- | ------------------ | ------------------------------------------ |
| `model`                  | str   | Required           | Model path or HuggingFace ID               |
| `mode`                   | str   | `"hybrid"`         | Engine mode: `prefill`, `decode`, `hybrid` |
| `host`                   | str   | `"0.0.0.0"`        | Bind address                               |
| `port`                   | int   | `5000`             | ZMQ port                                   |
| `max_model_len`          | int   | `16384`            | Maximum sequence length                    |
| `max_num_batched_tokens` | int   | `16384`            | Max tokens per batch                       |
| `max_num_seqs`           | int   | `256`              | Max concurrent sequences                   |
| `kvcache_block_size`     | int   | `256`              | KV cache block size (64 for MLA models)    |
| `gpu_memory_utilization` | float | `0.9`              | GPU memory usage fraction                  |
| `enforce_eager`          | bool  | `False`            | Disable CUDA Graph (for debugging)         |
| `ray_address`            | str   | `"127.0.0.1:6379"` | Ray cluster address                        |
| `nanoctrl_address`       | str   | `None`             | NanoCtrl address for PD disaggregation     |
| `log_level`              | str   | `"CRITICAL"`       | Logging level                              |
| `enable_profiler`        | bool  | `False`            | Enable torch profiler                      |
| `profiler_start_step`    | int   | `40`               | Step number to start profiling             |
| `profiling_step`         | int   | `16`               | Number of steps to profile                 |
| `profiler_dir`           | str   | `"./profiler_res"` | Output directory for profiler traces       |
