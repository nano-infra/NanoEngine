# Original NanoDeploy loop1 enforce eager 两机 A/B

日期：2026-07-17

## 结论

在此前 full CUDA Graph + worker-level `CUDA_LAUNCH_BLOCKING=1` 于正式阶段
154 秒、2,157 / 7,200 请求失败的相同 Original NanoDeploy loop1 配置上，仅增加
`--enforce-eager` 后，运行稳定推进到 10 分 16 秒、5,224 / 7,200 请求，未出现
`CUDA illegal memory access`、`torch.AcceleratorError` 或 Python traceback。

本轮按用户要求手动中止，exit code 130，不是运行故障。因此不能宣称 7,200 请求
完整通过，但已经满足以下更强的局部 A/B 条件：

- 稳定时间达到 graph 失败窗口的约 4 倍；
- 完成请求数达到 graph 失败点的约 2.4 倍；
- KV 利用率一度达到 100%，发生 scheduler preemption 后仍继续推进；
- 中止时 KV 已回落到约 77.5%，没有 CUDA 或 worker crash。

这给出了高置信证据：loop1 illegal memory 依赖 full CUDA Graph
capture/replay 路径，而不是一般的 loop1 eager forward、KV 压力、请求 metadata
传输或 `.tolist()` 同步本身。具体是图捕获内容、动态地址/metadata 生命周期，还是
某个 graph 内 kernel 的 replay 约束，仍需进一步细分。

## A/B 配置

除 `--enforce-eager` 外，保持上一轮有效 blocking 诊断配置：

- 7,200 requests，20 req/s，同一 CSV；
- Attention `DP2 × SP8 × TP1`，FFN `EP16`；
- `loop_count=1`；
- `max_num_seqs=256`，`gpu_memory_limit_gb=141`；
- `sp_backend=hao_basic`；
- legacy dynamic SP，`long_short_sp8`；
- non-uniform split 开启，fixed SP 关闭；
- centralized scheduler，LeastBatch routing；
- 16 个 Ray ModelRunner 均通过 `runtime_env` 获得
  `CUDA_LAUNCH_BLOCKING=1`。

命令行配置打印的 `cuda_graph_mode=full` 是 parser 默认值；
`enforce_eager=True` 会跳过 graph capture，并在 `run_model` 中直接执行 model
forward。本轮日志没有 CUDA Graph capture/replay 阶段，初始化后直接进入 eager
warmup，符合预期。

## 运行结果

- eager warmup：256 requests，85.45 秒，成功；
- 正式阶段开始：09:14:34 UTC；
- 手动中止：正式阶段 10 分 16 秒；
- 最后进度：5,224 / 7,200（73%）；
- 平均已完成请求 latency：约 268.25 秒；
- exit code：130（用户要求后发送 Ctrl-C）；
- CUDA illegal / AcceleratorError / traceback：0；
- JSONL：未生成，benchmark 仅在完整结束后写出。

压力最高阶段日志显示：

```text
min_free_blocks=0
max_kv_util_pct=100.00
Preemption happens for seq_id=...
```

系统随后继续完成数千个请求。手动中止前最后一个完整 engine step 为：

```text
itl=408.99 ms
total_batch_size=1978
max_sp_batch_size=140
min_free_blocks=3160
max_kv_util_pct=77.50
sp_size_hist_global={1: 1960, 8: 18}
waiting_reqs=0
```

中止后 Ray 两节点均保持 active，资源为 `0 / 16 GPU`，无 pending demand。

## 与 graph blocking 诊断对照

| 项目 | full graph + blocking | enforce eager + blocking |
| --- | ---: | ---: |
| 结束性质 | CUDA illegal failure | 用户手动中止 |
| 正式阶段时间 | 154 秒 | 616 秒 |
| 完成请求 | 2,157 | 5,224 |
| 峰值 KV 压力 | 约 34% | 100% |
| 首个同步点 | `graph.replay()` illegal | 无 CUDA 错误 |

由于本轮同时保留 worker-level blocking，差异主要就是 graph replay 与 eager
forward。结果把后续定位优先级明确放在 CUDA Graph 路径：先检查 graph capture
期间的动态 tensor/metadata 地址和 replay 生命周期，再对 graph 内 hao all-to-all、
FlashMLA、Triton indexed copy 与结果合并 kernel 做分段 A/B。

## 产物

- `nanodeploy_original_loop1_enforce_eager_cuda_launch_blocking_issue001_2node_dp2sp8_r20_6min_20260717.log`：本轮完整到手动中止的日志；
- 本报告。

日志受仓库 `*.log` ignore 规则保护，只保留在当前工作区。
