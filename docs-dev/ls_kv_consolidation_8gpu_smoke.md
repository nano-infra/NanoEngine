# LS KV Consolidation 单机 8-GPU Smoke

日期：2026-07-15

环境：

- Ray：`10.102.243.60:8776`
- master：`10.102.243.60:28776`
- 单节点 `10.102.243.60`，8 × H200
- topology：Attention `DP1 × SP8 × TP1`，FFN `EP8`
- model：DeepSeek-V3，dummy weight，eager execution

## 1. 快速 execute correctness

为了在短请求内覆盖完整的多次自动收缩，使用：

```text
initial_kv_dop = 8
candidate_util = 0.50
target_high_watermark = 0.80
stable_steps = 1
cooldown_steps = 0
check_interval_steps = 1
max_source_blocks_per_event = 16
migration_chunk_tokens = 64
prompt_len = 4096
max_tokens = 8
```

结果：

- admission placement 为 `8` ranks，每 rank `512` KV tokens；
- 连续完成 7 次真实 KV P2P maintenance，allocation `8→7→6→5→4→3→2→1`；
- 每次移动 `512` tokens、8 physical ranges；
- maintenance wall time 分别为约 `648/421/467/424/411/448/410 ms`；
- 收缩后完成 7 个真实 Decode steps，请求输出 8 tokens 并正常 drain；
- 无 Ray、NCCL、CUDA、scheduler transaction 或 block ownership 错误。

详细结果：

- `ls_kv_consolidation_execute_smoke_8gpu.json`
- `ls_kv_consolidation_execute_smoke_8gpu.jsonl`

## 2. 默认时间阈值验证

保持已接入的默认时间阈值：

```text
candidate_util = 0.50
target_high_watermark = 0.80
stable_steps = 32
cooldown_steps = 64
check_interval_steps = 8
```

使用 `initial_kv_dop=8`、`prompt_len=4096`、`max_tokens=72` 的单请求长 Decode。结果：

- stable window 达到 32 后，调度器继续受 scale-up cooldown 限制；
- cooldown 满足后继续等待 8-step 检查点；
- 在 scheduler step 72（结果文件 `step_idx=71`）触发一次 maintenance；
- 事件 telemetry 为 `stable_steps=71`、`source_rank=1`、`target_dop=1`；
- 搬迁 `512` tokens、16 physical ranges，wall time 约 `125 ms`；
- allocation 成功从 `8→7`，下一个真实 Decode 使用 `kv_dop=7` 并完成请求；
- 总计 1 prefill、71 decode、1 maintenance，输出 72 tokens，正常 drain。

详细结果：

- `ls_kv_consolidation_default_thresholds_smoke_8gpu.json`
- `ls_kv_consolidation_default_thresholds_smoke_8gpu.jsonl`

## 3. 清理检查

两轮结束后 8 个 ModelRunner actors 均正常终止。Ray 报告 `GPU: 8.0` 全部可用，相关 placement groups 状态均为 `REMOVED`。

以上是 correctness smoke，不是性能结论；dummy weight、单请求和 eager execution 的延迟不能用于生产容量或收益阈值标定。
