## Configuration Reference

### Engine Parameters

| Parameter                | Type  | Default            | Description                                                          |
| ------------------------ | ----- | ------------------ | -------------------------------------------------------------------- |
| `model`                  | str   | Required           | Model path or HuggingFace ID                                         |
| `mode`                   | str   | `"hybrid"`         | Engine mode: `prefill`, `decode`, `hybrid`                           |
| `host`                   | str   | `"0.0.0.0"`        | Bind address                                                         |
| `port`                   | int   | `0`                | Service port (`0` lets the OS allocate an available port)            |
| `max_model_len`          | int   | `16384`            | Maximum sequence length                                              |
| `max_num_batched_tokens` | int   | `16384`            | Max tokens per batch                                                 |
| `max_num_seqs`           | int   | `256`              | Max concurrent sequences                                             |
| `kvcache_block_size`     | int   | `256`              | KV cache block size (64 for MLA models)                              |
| `gpu_memory_utilization` | float | `0.9`              | GPU memory usage fraction                                            |
| `enforce_eager`          | bool  | `False`            | Disable CUDA Graph (for debugging)                                   |
| `hardware_backend`       | str   | `"auto"`           | Hardware backend: `auto`, `blackwell`, `hopper`, or `gpu_generic`    |
| `ray_address`            | str   | `"auto"`           | Ray cluster address (`auto` discovers a local cluster)               |
| `master_address`         | str   | `None`             | Optional legacy rendezvous override; normally discovered from rank 0 |
| `ctrl_address`           | str   | `None`             | dlslime-ctrl HTTP address for PD disaggregation                      |
| `log_level`              | str   | `"CRITICAL"`       | Logging level                                                        |
| `profiler_dir`           | str   | `"./profiler_res"` | Output root for runtime profiler traces                              |

When `port` is left at `0`, DLEngine binds an OS-assigned port and logs the
resulting bind and advertise endpoints. NanoCtrl registration uses the
advertise endpoint, never the wildcard bind address.

### Runtime Profiling

A running OpenAI server can delimit a profiling window through its management
API:

```bash
curl -X POST http://127.0.0.1:<port>/start_profiler \
  -H 'content-type: application/json' \
  -d '{"trace_name":"baseline"}'

# Send the inference requests to capture.

curl -X POST http://127.0.0.1:<port>/stop_profiler
```

`trace_name` accepts 1-64 letters, numbers, dots, underscores, or hyphens.
Traces from every worker are written below
`<profiler_dir>/<trace_name>/`, and the stop response lists the files. Restrict
these management endpoints to trusted networks in production.

Profiling starts only through `/start_profiler` and continues until
`/stop_profiler`; step-based automatic profiling is not supported.
