# LS-Decode-Core 单机 8 卡预检

## 目的

在启动 `4 DP x 8 SP / EP32` 正式实验前，先用 `1 DP x 8 SP / EP8` 验证 LoongServe-style Decode 的端到端执行链路。

单机 8 卡已经包含一个完整的 8-SP allocation domain，因此能够覆盖：

- admission-time uniform KV placement；
- source-style threshold greedy multi-master planning；
- compute/memory scale-up 与自然 master scale-down；
- zero-history new master 的 pending-token 重指派；
- `hao_basic` Q/O/LSE SP all-to-all；
- dummy participants 和固定 collective cadence。

它不覆盖 4 个独立 DP domain 的并发、跨 group/DP 负载分布或 EP32 性能。因此 EP8 的 ITL、MoE dispatch/combine 开销和 master crossover 不能作为正式实验结论。

## 支持拓扑

```text
attention_dp = 1
attention_sp = 8
attention_tp = 1
ffn_dp       = 1
ffn_ep       = 8
ffn_tp       = 1
```

其余 LS-Decode-Core 约束保持不变：

```text
mode = decode
dummy_prefill = true
scheduler_mode = centralized
loop_count = 1
use_dlslime_rpc = true
sp_backend = hao_basic
fixed_sp_size = 0
```

## 最小测试矩阵

1. 单 master：`num_seqs <= ls_decode_batch_per_master`，验证 admission 和连续 Decode。
2. compute scale-up：`num_seqs = ls_decode_batch_per_master + 1`，预期至少两个 masters，并触发 zero-history new master。
3. 多 master：`num_seqs = 2 * ls_decode_batch_per_master + 1`，预期 source-style slices 为 `[T, T, 1]`（容量充足时）。
4. 多轮运行：至少执行两个真实 Decode iterations，确认 pending token 每轮只 append 一次且 collective 不 hang。

## 通过标准

- 8 个 workers 全部完成初始化；
- admission 日志记录合法的 `initial_kv_dops`、`initial_kv_ranks` 和 uniform `prompt_kv_tokens`；
- Decode 日志中的 `master_dops`、`master_batch_sizes` 和 `iteration_master_assignments` 符合阈值规划；
- 存在 remote KV owner 时进入 SP all-to-all，且无 shape/assert/NCCL/DLSLIME error；
- `historical_kv_migration_bytes` 恒为 0；
- 至少两个真实 Decode iterations 完成，无 hang、preemption、block leak 或负计数。

## 结果解释

单机预检通过后，才进入 `4 DP x 8 SP / EP32` 的空 DP collective、跨 DP 并发和正式性能测试。若全量 DeepSeek-V3 在 EP8 下因每卡 expert 权重或 KV cache 容量不足而 OOM，应改用保持 MLA/DeepEP 合法维度的 reduced DeepSeek-V3 fixture 做功能预检；不能把 EP8 OOM 解释为 scheduler correctness 失败。
