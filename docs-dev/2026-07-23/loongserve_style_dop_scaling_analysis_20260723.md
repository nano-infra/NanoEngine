# LoongServe-style DoP 扩缩容条件与日志分析

## 范围

- 日志：
  `docs-dev/2026-07-23/ls_style_loop16_dp2sp8_r20_diag01_3.log`
- 配置：attention DP2×SP8、`loop_count=16`、request rate 20、
  每个 rank 12,372 个 KV blocks、block size 64。
- 本轮真正用于 LS Decode master 规划的 compute threshold 是
  `ls_min_comp_bound_decoding_batch_size=128`。命令和 manifest 中虽然还有
  `ls_decode_batch_per_master=64`，但当前 core scheduler 仅保存该字段，
  调用 planner 时传入的是前者。

## 先区分两层 DoP

1. **master/compute DoP**：某一轮真正承担 query、生成新 KV 的 master
   rank 数，即日志中的 `master_dops`。
2. **KV/allocation DoP**：一个 group 历史 KV 所在、由 group 持有的 rank
   数，即本轮日志中的 `kv_dops` / `len(rank_allocations)`。

master DoP 每轮重算，可以只复用 group 已经持有的 passive rank；这种变化
不会修改 KV allocation，也不会产生 `scale_reason=compute`。KV DoP 的增加
需要 admission/group merge 或 planner 真正取得新的 idle rank；KV DoP 的
减少则在 rank 非空时需要 consolidation。

## master/compute DoP 扩张与缩小

### 扩张条件

Decode planner 的物理 compute scale-up 条件是：

```text
存在本 DP 内 graph-idle rank
且 floor(real_batch_size / current_allocation_dop) > 128
```

满足时逐个加入 idle rank，直到整数除法结果不再大于 128，或已没有 idle
rank。这里的 128 不是严格的 per-master batch 上限；如果没有空闲 rank，
或 append/receiver 约束造成偏斜，单 master batch 可以超过 128。

随后 source-greedy 在已有 allocation 中选择实际 masters。目标 chunk 为
`max(remaining / remaining_candidates, 128)`，但还会被以下条件截断：

- master 必须能容纳当前 pending input 和本轮 16 个 output token；
- 每个 master request 还保留 1 个 KV block headroom；
- 每个 rank 的 remote-attention receiver 数不能超过 128；
- 每个 master 的 metadata batch 不能超过 256。

因此 master DoP 也可能在 `scale_reason=none` 时增加：只是激活已有的
passive KV rank，而非增加 allocation。例如 group 195 从 batch 126、
master DoP1 变为 batch 134、master DoP2，allocation 始终为 `[2, 4]`；
新 master rank 2 被记录为 `reused_passive_master_ranks`。

反过来，即使 batch 小于 128，KV/receiver 容量也可能要求多个 masters。
group 68 在 batch 51 时使用两个 masters；后续 rank 7 已使用
12,348/12,372 blocks。group 134 在 batch 89 时甚至使用三个 masters，
其中两个 rank 分别已经使用 12,358 和 12,295 blocks。

### 缩小条件

master DoP 没有单独的 scale-down transaction。每轮重新跑 source-greedy，
当更少的 master 已足够覆盖 batch、append capacity 和 receiver capacity
时，自然不再选择其余 passive ranks。

典型例子是 group 225：

- batch 129：masters `[4, 2]`，batch split `[128, 1]`；
- batch 127：master `[4]`，batch split `[127]`；
- KV allocation 始终是 `[2, 4]`，所以这只是 compute DoP2→1，不是 KV
  DoP2→1。

本轮 3,818 个 group-step 中，master DoP 分布为 D1 3,315、D2 312、
D3 176、D4 15；有 349 个 group-step 满足 `master_dop < kv_dop`。

## KV/allocation DoP 扩张

### Admission 初始 placement

`ls_decode_initial_kv_dop=0` 表示从 D1 开始逐级尝试，选择第一个 exact
feasible DoP。需要同时满足：

- 新 prompt/bootstrap KV 和每请求 1 block headroom 能放下；
- 首轮 receiver load 不超过 128；
- master metadata 不超过 256；
- 开启的 future-KV gate 判断整批请求的未来峰值能被本 DP 的 SP8 pool
  覆盖。

本轮 531 个 admission batch 的 `planned_kv_dop` 为：

- D1：513；
- D2：16；
- D3：2。

D2/D3 batch 几乎都由 53 万至 97 万 token 的超长 prompt 触发。例如
seq=4853（727,218 tokens）选择了 ranks `[5, 1]`、planned DoP2。

### CAPACITY_APPEND / group merge

当 admission batch 的 KV token 需求大于所有完全 idle ranks 的总容量时，
进入 `CAPACITY_APPEND`。调度器按 slack 从大到小借用已有 group，直到覆盖
容量缺口，把 donor group 的 sequences 和 rank allocation 合并进新 group。

本轮：

- 531 个 admission batch 中，18 个 `STANDALONE`，513 个
  `CAPACITY_APPEND`；
- 513 个 `CAPACITY_APPEND` 中，有 226 个新 group 的首次实际 allocation
  DoP 大于该 admission 自己的 `planned_kv_dop`，差值来自 donor ranks；
- 最明显的 group 532：admission 自身 planned D1/rank `[0]`，但合并 donor
  后首次 iteration 已持有 `[0, 5, 2, 1, 7]`，即 KV DoP5；当时 batch 332、
  master DoP4。

所以这次 KV DoP 增长的主要来源不是 Decode planner 的 `compute/memory/
receiver` scale-up，而是 admission 的 CAPACITY_APPEND group merge。

### Decode mandatory safety 和 planner scale-up

代码还有两条物理扩张路径：

1. 每轮 admission 前计算
   `slack = allocated_capacity_tokens - used_tokens - running_requests`。
   如果 slack 为负，先合并有正 slack 的 donor group，再逐个加入本 DP
   的 unallocated ranks，直到 slack 非负。
2. Decode planner 如果当前 allocation 无法为本轮 16-token chunk 找到
   append/receiver 可行 assignment，则在启用 memory scale-up 时加入 idle
   rank；receiver owner-local quota 或 metadata 容量也可能要求加 rank。

但本轮所有 3,818 个 `scale_reasons` 都是 `none`，所有
`new_master_ranks` 也为空；同一 group 的 allocation 没有观察到向上变化。
因此日志没有证据表明本轮真正走过 Decode planner 的 compute/memory/
receiver 加 rank，观测到的扩张可由 admission merge 解释。

此外，`scale_reasons` 只覆盖 Decode planner 加 rank，不覆盖 admission
merge 和 mandatory-safety merge，不能用它单独判断整套系统是否扩过 KV
DoP。

## KV/allocation DoP 缩小

### 空 rank 直接回收

如果 allocated rank 已经没有 group KV，且不是任何 running sequence 的
active master 或 pending target，mandatory safety 会直接从 allocation
删除它，不需要搬 KV。

### 非空 rank：KV consolidation

非空 rank 只有满足以下条件才成为自动 consolidation candidate：

1. `participants` 是 allocation 中 `used_blocks > 0` 的 ranks，且没有尚未
   回收的空 allocation rank；
2. 目标 DoP 从 1 开始，在
   `floor(real_batch / target_dop) > 128` 时递增；
3. `participants.size() > target_dop`；
4. group block utilization
   `sum(used_blocks) / sum(total_blocks_on_participants) < 0.50`；
5. 至少有一个 source rank 既不是任何 sequence 的 active master，也不是
   pending-token target；优先选 KV 最少的 source；
6. 相同 target DoP、相同 member IDs、相同 allocation 连续稳定 2 个
   scheduler steps；
7. 距离最近一次 scale-up 和最近一次 consolidation 都至少 2 steps；
8. 本配置每 1 step 检查一次；
9. 单次 source rank 的迁移 block 数不超过 128；
10. 迁移后每个 retained rank 的物理 utilization 不超过 0.80，且新布局
    仍满足 KV capacity 和 receiver≤128。

每次 transaction 只释放一个 source rank，而不是一次直接跳到 target
DoP，所以大 group 会连续执行多次。

本轮 346 个 Decode iteration 中，308 次是 `no_candidate`，38 次显示
`stable_window`；这些可见 Decode 记录的 candidate 都是
`stable_steps=1`，说明 drain 过程中成员持续完成、candidate identity
反复被重置。真正执行的 maintenance 记录均达到 `stable_steps=2`。

最后 06:31:48–06:31:59 共执行 7 次：

- group 525：KV DoP2→1，一次；
- group 531：KV DoP3→1，两次；
- group 532：KV DoP5→1，四次。

七次 consolidation 总 stall 约 669.69 ms，全部发生在 drain 尾部小 batch、
极低 utilization 阶段。group 532 在 batch 12 时仍是 master DoP1/KV
DoP5，随后才按 5→4→3→2→1 逐 rank 收缩。

## 结论

当前策略是明显非对称的：

- **compute master DoP 响应快**：每轮重算，可立即激活/停用 group 已持有
  的 passive ranks；
- **KV DoP 扩张积极**：超长请求的 exact placement、CAPACITY_APPEND
  donor merge、memory safety 都可以增加 group allocation；
- **KV DoP 缩小保守且滞后**：有 KV 的 rank 必须低于 50% group
  utilization、稳定 2 steps，并通过迁移预算和 80% high-watermark 后，
  才能一次释放一个 rank。

这正是日志中“大部分轮次 master DoP1，但 KV 仍分散在 D2–D5”现象的直接
原因。`loop_count=16` 只进入 append capacity 计算（pending input + 16 个
outputs），不会把 compute threshold 从 128 改为 16 或 64。
