# Original NanoDeploy loop1 CUDA_LAUNCH_BLOCKING 两机诊断

日期：2026-07-17

## 结论

在与 Original NanoDeploy loop16 成功运行对齐的两机配置上，仅保持
`loop_count=1` 并启用 `CUDA_LAUNCH_BLOCKING=1` 后，确认 CUDA illegal memory
发生在 full CUDA Graph 的 `graph.replay()` 内部：

```text
nanodeploy/worker/model_runner.py:801 in run_model
    graph.replay()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

有效诊断由 head 节点 global rank 6 同步报错，正式阶段运行 154 秒，完成
2,157 / 7,200 请求。错误不再延迟到 `torch.cat(...).T.tolist()` 才暴露，说明
此前的 `.tolist()` 和 NCCL watchdog 都只是异步错误的观察点，不是最初故障点。

`CUDA_LAUNCH_BLOCKING` 只能把整个 CUDA Graph replay 作为一个同步单元，因此本轮
把范围收敛到了捕获图内部，但仍不能仅靠 Python traceback 区分图中的 FlashMLA、
Triton indexed copy、hao all-to-all 或结果合并 kernel。

## 配置对齐

除诊断环境变量外，配置与此前两次 Original loop1 完全一致：

- 7,200 requests，20 req/s，固定 CSV 和 seed；
- Attention `DP2 × SP8 × TP1`，FFN `EP16`；
- `loop_count=1`；
- `max_num_seqs=256`，`max_num_recv_seqs=32`；
- `gpu_memory_limit_gb=141`，每 rank 14,045 KV blocks；
- full CUDA Graph，`sp_backend=hao_basic`；
- legacy dynamic SP，`long_short_sp8`；
- non-uniform split 开启，fixed SP 关闭；
- centralized scheduler，LeastBatch routing；
- LS decode core scheduler 与 KV consolidation 关闭。

运行前后 Ray 都显示两节点 active、`0 / 16 GPU` 占用。

## Ray 环境变量传播校验

第一次尝试只在 benchmark driver 命令前设置：

```bash
CUDA_LAUNCH_BLOCKING=1 python ...
```

但 Ray 集群是预先启动的，driver 环境不会自动传播到已有 Ray worker。运行后通过
Ray remote task 查询得到：

```text
(head_node_id, None)
```

因此第一次尝试不是有效的 blocking 诊断；它在 174 秒、2,653 / 7,200 时由
global rank 3 失败，错误仍在 `.tolist()` 暴露。该轮只能算第三次普通 loop1
复现，不能用于判断 `CUDA_LAUNCH_BLOCKING` 的效果。

随后在 `RayExecutor` 的 per-actor `runtime_env.env_vars` 转发列表中加入
`CUDA_LAUNCH_BLOCKING`。有效重跑的初始化日志确认：

```text
Rank 12 worker CUDA_LAUNCH_BLOCKING='1'
Rank 15 worker CUDA_LAUNCH_BLOCKING='1' [repeated 15x across cluster]
```

即 16 个 ModelRunner 均实际收到该变量。

## 有效诊断结果

- CUDA Graph capture：20 个 local graph + 60 个 SP graph，成功；
- warmup：256 requests，26.35 秒，成功；
- 正式阶段最后进度：2,157 / 7,200 @ 154 秒；
- exit code：1；
- 首个同步 fatal：head global rank 6；
- 同步失败 API：`torch.cuda.CUDAGraph.replay()`；
- JSONL：未生成，driver 只在完整完成后写出。

最后一个完整 step：

```text
itl=72.50 ms
total_batch_size=947
max_sp_batch_size=65
min_free_blocks=9270
max_kv_util_pct=34.00
sp_size_hist_global={1: 942, 6: 1, 8: 4}
waiting_reqs=0
```

随后发送约 3.27 MB RPC metadata，下一次 rank 6 graph replay 同步报 illegal
memory。故障时仍有大量 KV 空间，没有 waiting，再次排除 KV exhaustion 和 RPC
endpoint 容量不足。

## 诊断含义

本轮新增了两个高置信判断：

1. illegal memory 的最初故障范围在 full CUDA Graph replay 内，而不是输出采样结果
   转 CPU 的 `.tolist()`；
2. NCCL watchdog、TCPStore reset 和 Ray worker crash 都是 CUDA context 已损坏后的
   级联现象。

尚未确定的是 graph 内部的具体 kernel。下一步若继续定位，最小 A/B 是同配置
`enforce_eager`：若 eager 能稳定越过多个历史失败窗口，则进一步确认 CUDA Graph
capture/replay 与动态 metadata 的组合；若 eager 仍失败，则应在 hao all-to-all、
FlashMLA 与 Triton copy 阶段之间插入同步边界做二分。

## 产物

- `nanodeploy_original_loop1_cuda_launch_blocking_worker_env_issue001_2node_dp2sp8_r20_6min_20260717.log`：有效 worker-level blocking 完整日志；
- `nanodeploy_original_loop1_cuda_launch_blocking_issue001_2node_dp2sp8_r20_6min_20260717.log`：无效 driver-only 尝试日志，仅作传播问题记录；
- 本报告。

两个日志受仓库 `*.log` ignore 规则保护，只保留在当前工作区。
