# LS-Decode T=128 16 卡 6 分钟测试与调度假阴性修复建议

日期：2026-07-16

## 1. 结论

将 `ls_decode_batch_per_master` 从 8 提高到 128，显著延后了 receiver/owner 抖动，但没有修复根因：首次 `append capacity cannot cover remaining requests` 从旧运行的正式 workload 约 20 秒延后到约 250 秒。新运行最终仍进入 `preempt -> recovery admission -> same plan failure` 循环。

这次失败是可形式化复现的调度假阴性：当前 source-greedy 的连续 prefix 顺序无法找到 assignment，但同一请求集合存在满足 master、receiver 和 append 约束的计划。它不是 KV blocks 物理不足，也不是 consolidation 导致。

建议采用两层修复：

1. 热路径保留现有 source-greedy；失败后按 current master / KV owner 稳定分桶，复用原 chunk 和 T 逻辑进行低成本 repair。
2. repair 仍失败时，用 owner-local quota 的 lower-bound matching 做精确可行性兜底。只有精确求解证明不可行后，才允许进入 scale-up、等待或抢占。

同时必须给 recovery 增加 state epoch/backoff，避免真正容量不足时继续形成活锁。

## 2. 测试配置与结果

- 2 nodes，16 H200 GPUs；Attention `DP2 x SP8`，FFN `EP16`；
- Issue 1% CSV，7,200 requests，20 req/s，360 秒发送窗口；
- `max_num_seqs=256`，`max_num_recv_seqs=128`；
- `ls_decode_batch_per_master=128`；
- KV consolidation `execute`，参数为 `candidate_util=0.50`、`high_watermark=0.80`、`stable/cooldown/check=2/2/1`；
- CUDA Graph full；开启 `NANODEPLOY_LOG_MODEL_FORWARD_TIMING=1` 和 verbose LS telemetry。

日志：`docs-dev/2026-07-16/ls_style_issue001_2node_dp2sp8_r20_t128_6min_20260716.log`。

运行时间线：

- 08:57:06 UTC：正式 workload 开始；warmup 32/32 完成；
- 09:01:17 UTC：首次 plan failure，正式运行约 250 秒，完成约 3,169 请求；
- 6 分钟发送窗口内完成 3,414/7,200，窗口内平均完成速率约 9.48 req/s，最近一次平均 latency 为 82.72 秒；
- drain 到 09:09 elapsed 时完成 3,708/7,200，平均 latency 约 95.03 秒；
- 此时同一批 recovery request 仍被反复 admission/preempt，ETA 已达到数小时，因此主动终止；
- 无 Traceback、RuntimeError、CUDA error、OOM 或 segfault；由于非自然结束，没有 JSONL 和 `Benchmark Results`。

故障统计：

- 1,934 次 plan failure / preemption；
- 1,235 个 unique preempted sequences；
- pending 峰值 3,142 requests / 1,655 batches；
- consolidation candidate/action 均为 0。

Actual-forward CUDA event 统计：

| 区间 | 样本数 | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| 首次失败前 | 1,874 | 114.40 ms | 121.05 ms | 147.08 ms | 242.86 ms |
| 首次失败后 | 377 | 152.67 ms | 148.63 ms | 217.95 ms | 239.45 ms |

## 3. 首次调度假阴性的精确证据

失败前最后一个成功 step：

- `total_batch_size=1941`；
- `max_sp_batch_size=128`；
- `min_free_blocks=415`；
- `max_kv_util_pct=97.05%`；
- SP owner DoP histogram 为 `{1: 1613, 2: 328}`。

DP1 当时有：

- group 3：313 requests，masters `[0, 1, 7]`，chunks `[128, 128, 57]`；
- group 7：127 requests，master 2；
- 新到达 3 requests 的初始 KV owner 都是 rank 2，使 group 7 增至 130，随后因 T=128 与 group 3 合并；
- 合并后共有 443 requests，candidate 顺序为 `[2, 0, 1, 7]`。

当前连续-prefix greedy 的重放结果：

- rank 2 取 128；
- rank 0 受 receiver prefix 限制只能取 98；
- rank 1 只能取 55；
- rank 7 的 prefix capacity 变成 0；
- 尚余 162 requests，于是返回 `append capacity cannot cover remaining requests`。

但把请求按 current master 稳定分桶后，完全相同的 candidate 和 T 逻辑可得到 chunks `[128, 128, 128, 59]`，443 个请求全部分配，最大 receiver count 只有 103，且 append/metadata validator 可以通过。

因此：receiver 128 是需要满足的硬约束，但不是物理资源不足；真正错误是 request order + 连续 prefix 贪心产生了可行性假阴性。415 个最小 free blocks 也足以容纳该合法计划的 pending append/reservation。

## 4. 修复方案

### 4.1 第一层：低风险 stable-owner repair

在 `SPStateManager::plan_iteration_masters_source_greedy` 内把现有 while 抽成：

```text
run_attempt(planning_order)
```

流程：

1. 先用 identity order 执行当前算法，保持现有正常路径和 scale-up 行为不变；
2. 当无 extra rank 可加入且 attempt 硬失败时，生成新的稳定顺序：
   - 按 candidate 顺序，将 current master 等于该 candidate 的请求放入对应 bucket；
   - current master 不在 allocation 时，选择 committed KV 最多的 candidate owner；
   - bucket 内保持原 admission/request 顺序；
3. 用新顺序再跑同一套 `receiver_prefix_capacity`、append capacity、target chunk 和 T 逻辑；
4. `sequence_master_ranks` 必须映射回原始 stable request index；
5. 返回前仍调用现有 `validate_iteration_master_plan`。

该 repair 只走失败路径，复杂度仍为 `O(batch x SP^2)`，SP 最大只有 8；它已能解开本次真实 443-request 反例，并保持 `[128,128,128,59]` 的 T=128 chunk 形状。

可以再增加一个 validated sticky fallback：直接保留每条 sequence 的 current master，按 candidate 聚合 master loads；只有现有 validator 完全通过才采用。它可保护“合并前各子组都合法”的常见场景。

stable-owner repair 不是数学完备算法，失败不能作为抢占的最终依据。

### 4.2 第二层：owner-local quota 精确 matching

对请求 `i`，记历史 KV owner 集为 `O_i`，master assignment 为 `a_i`。对 rank `r`：

```text
owner_count[r] = count(i where r in O_i)
self_count[r]  = count(i where r in O_i and a_i == r)
recv[r]        = owner_count[r] - self_count[r]
```

所以 receiver 约束可精确化为：

```text
self_count[r] >= L[r]
L[r] = max(0, owner_count[r] - max_num_recv_seqs)
```

这不再需要模拟连续 prefix。实现可使用 lower-bound bipartite matching/circulation：

- 每个 request 必须分配到一个 candidate master；
- owner edge 为 rank 提供一个 local quota 单位；
- 每个 rank 至少满足 `L[r]`，并受 nominal chunk/load、decode metadata 和 append-safe capacity 上限约束；
- 第一轮固定原算法生成的 nominal chunk profile `q`，从而保持 T 和 active master 数；
- 若 `q[r] < L[r]` 或固定 q 不可行，再允许 quota-aware 调整 load，必要时激活 passive/extra owner rank；
- 输出后必须通过现有 exact validator。

owner 集可以压缩成 8-bit mask，最多只有 256 类；每类再按 previous master/append cost 细分。这样 exact fallback 的图规模基本不随 2,000 个请求线性膨胀。建议只在 repair 失败、group merge/active-set 改变或 receiver 接近上限时执行，目标 p99 小于 1 ms；不要在每 token 热路径运行 successive-shortest-path min-cost flow。

分配目标按以下优先级优化：

1. 满足 receiver、metadata、append 等硬约束；
2. 保持 previous master；
3. 优先选择已有 KV owner，避免每 token 扩散新的 owner；
4. 最小化 remote edges 和 load deviation。

### 4.3 Append capacity

对 rank `r`，令 `F_r` 为当前 free blocks 加可回收 reservation；请求 `i` 在 `r` 上的本轮 block cost 为：

```text
b_ir = ceil((committed_ir + 2) / block_size)
       - ceil(committed_ir / block_size)
```

assignment 必须满足：

```text
sum_i(x_ir * b_ir) + ceil(reserved_blocks_per_req * load_r) <= F_r
```

快路径可以精确维护；matching 使用 worst-case block cost 得到安全 count cap，最后由 `validate_iteration_master_plan` 做权威检查。安全包络不通过应标为 append/memory pressure，并尝试 memory scale-up 或等待，不能误报成 receiver 不可行。

### 4.4 只有精确证明不可行后才能抢占

当前 generic failure 会直接进入 merge/preempt。应拆分 failure kind：

- `receiver_greedy_failed`；
- `receiver_proven_infeasible`；
- `append_capacity`；
- `decode_metadata_capacity`；
- `restart_limit`。

启发式失败只允许进入 repair/exact fallback；不能直接进入 victim selection。exact fallback 证明不可行后，优先等待/scale-up；确需抢占时，victim 应针对超限 receiver 或 append rank 选择。

### 4.5 Recovery 活锁保护

即使 planner 修好，真实过载仍可能发生。recovery batch 不应无条件 `push_front` 后下一 step 立即重入。至少增加：

- `blocked_on_capacity_epoch` 或 receiver/owner generation；
- 只有 request finish、rank allocation、free-block bucket 或 owner topology 改变时才解除；
- 或使用有上限的指数 backoff，并将 recovery 放到普通 pending 后方；
- 相同 state fingerprint 的 exact failure 结果应缓存，禁止同一 step 进行 9 次等价 preempt/re-admit。

## 5. 回归与验收

1. 在 `tests/test_ls_decode_planner.py` 固化一个小型确定性反例：现有 prefix greedy 失败，stable-owner repair/matching 成功，validator 通过；
2. 固化本次 443-request owner-mask 快照，要求输出总 load 443、最大 receiver 不超过 128，且不产生 preemption；
3. 增加 Hall-violation 负例，只有 exact matching 可标记 `receiver_proven_infeasible`；
4. 保留现有 `B=T/T+1/2T+1` chunk 边界语义；receiver 强制激活 owner 时允许显式标记 `receiver_forced` 小 chunk；
5. 覆盖 append 0/1 block cost、真实 KV 不足、memory scale-up 和 validator failure；
6. 连续数百 step 验证 master stickiness 和 owner DoP 不再无界扩散；
7. 对 batch 64/256/1024/2048、稀疏/密集 owner masks 记录 fast/repair/exact path 的 p50/p99 调度耗时。

## 6. 推荐实施顺序

1. 先提交 stable-owner repair、structured failure reason 和小型回归反例；
2. 加 recovery state epoch/backoff，先切断活锁；
3. 加 owner-mask lower-bound matching，作为最终可行性裁判；
4. 用本次同配置重新跑 16 卡 6 分钟，验收首次 failure、preemption、pending、owner DoP 和 actual-forward；
5. planner 正确后再扫描 T=64/96/128，T 不应再承担掩盖 receiver bug 的职责。
