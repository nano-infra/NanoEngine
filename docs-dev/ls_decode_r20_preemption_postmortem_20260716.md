# LS-Decode-Core rate 20 频繁抢占复盘

日期：2026-07-16

## 1. 结论摘要

两节点 `DP2 × SP8 / EP16`、Issue 1%、`rate=20` 运行中出现的早期频繁抢占，主要是 NanoDeploy 当前 LS-Decode-Core 调度实现中的 receiver 约束、贪心 master 分配和立即恢复机制共同造成的调度抖动；它不是开始阶段全局 KV cache 用满，也不是 LoongServe multi-master 思想的固有限制。

随后进行的 600-request 定向诊断复跑已经确认：首次失败由 `max_num_recv_seqs=128` 的 receiver capacity 触发；当时集群真实 KV block 利用率只有 4.59%，并且存在 receiver 最大值仅 121 的均衡合法 assignment。当前 source-greedy 没有找到该 assignment，随后进入每轮 9 次抢占和立即恢复的活锁。

同时，`rate=20` 对这份数据集提供的长期负载很高。随着等待队列持续增长，后期仍可能出现真实的计算或 KV 容量压力。因此需要区分：

- 开始约 20 秒出现的重复抢占：NanoDeploy 调度/恢复问题；
- 持续注入后的排队和后期容量压力：工作负载过载叠加 admission 缺少未来 KV 保护；
- consolidation：可能放大 owner/receiver 热点，但当前日志不足以证明它是首次故障的直接原因。

本报告分析的日志为 `docs-dev/ls_style_issue001_2node_dp2sp8_r20_6min.log`。该运行在正式 workload 约 187 秒时中止，没有最终 JSONL 或 `Benchmark Results`，不能作为完整性能结果。

## 2. 测试配置

- 2 nodes，16 H200 GPUs；
- Attention `DP2 × SP8 × TP1`，FFN `EP16`；
- `max_num_seqs=256`，`max_num_recv_seqs=128`；
- 每 rank 14,044 个 KV blocks，block size 64；
- `ls_decode_initial_kv_dop=0`，自动选择初始 KV DoP；
- `ls_decode_batch_per_master=8`；
- consolidation `execute`，`stable/cooldown/check=2/2/1`；
- Issue 1% CSV，7,200 requests，`20 req/s`，360 秒 offered-load window；
- routing 为 `LeastBatch`。

## 3. 日志取证

### 3.1 首次抢占时不是全局 KV 容量耗尽

正式 workload 开始约 20 秒后首次出现：

```text
LS-Decode-Core plan failed for group=2: append capacity cannot cover remaining requests
Preemption happens for seq_id=379
```

使用相同 seed 重放到达序列，在该时刻共有 434 个请求到达、2 个完成、432 个 outstanding。前 434 个请求的 prompt 合计为 9,835 个 64-token blocks；16 rank 总容量为：

```text
14,044 × 16 = 224,704 blocks
9,835 / 224,704 = 4.38%
```

即使把这些 prompt 全部偏置到一个 8-rank DP，prompt block 总量也只相当于该 DP 容量的约 8.75%。因此可以高置信排除首次故障是全局 KV cache 或整个 DP 的物理 blocks 用尽。

等待队列中的请求尚未 admission，也没有分配 KV；不能用累计到达请求的总 prompt 直接推断运行态 KV 已耗尽。

### 3.2 抢占表现为 recovery thrashing

日志中共出现：

- 1,432 次 preemption；
- 仅 65 个不同的 sequence；
- group 3 失败 1,397 次，group 2 失败 35 次；
- `seq_id=436` 被抢占 123 次，`437` 被抢占 88 次，`441/445` 各 87 次；
- 高频 victim 多数只有 100–280 prompt tokens，即 2–5 个 blocks。

正式请求的 CSV 映射是 `csv_row_zero_based = seq_id - 48`：scheduler 创建 16 个 dummy sequence，warmup 再创建 32 个 sequence。

同一批很小的请求被反复清空和恢复，释放的 KV 很少，也没有解除 receiver owner 热点。这是活性问题，而不是 1,432 次相互独立的 OOM。

## 4. 代码机制分析

### 4.1 错误原因被合并

`SPStateManager::plan_iteration_masters_source_greedy` 只有在下列两个条件同时满足时，才认为一个 rank 可以接收后续请求：

1. `estimate_pending_append_capacity(...) > 0`；
2. `receiver_prefix_capacity(...) > 0`。

任一条件失败，最终都可能返回同一句：

```text
append capacity cannot cover remaining requests
```

代码位置：`csrc/nanodeploy/scheduler/sp_state_manager.cpp:487-537`。

因此旧日志中的错误文本不能证明物理 KV blocks 不足。第 9 节的诊断复跑通过逐 rank blocks/receiver 快照进一步确认，这次故障的直接触发条件是 receiver capacity。

### 4.2 source-greedy 可能拒绝实际可行的分配

当前 planner：

- 按 rank 的 group 聚合 KV 排序；
- 按固定 sequence 顺序分配连续 prefix/chunk；
- 一个 rank 取过一个 chunk 后通过 `candidate_begin` 向前推进；
- 不按每条 sequence 的 KV owners 做 matching，不跳过暂时不适配的请求，也不回溯。

代码位置：`csrc/nanodeploy/scheduler/sp_state_manager.cpp:464-570`。

例如一个 owner 持有约 200 条请求的 prompt KV，若 8 个 master 近似平均分配，该 owner 需要接收约 175 条 remote request，超过 `max_num_recv_seqs=128`。实际可以让 owner 自己承担至少 72 条请求，把 remote receiver 数降到 128；但一次性平均切块的贪心不一定找到这个分配。

旧日志缺少故障瞬间的 per-sequence owner snapshot，单靠旧日志无法形式化证明存在合法 matching。第 9 节通过重放 admission 和逐轮 master assignment 重建了精确 owner 集合，并找到了满足所有约束的均衡 assignment，已经确认本次 source-greedy 存在 false negative。

### 4.3 victim 选择没有针对瓶颈

单一 group 规划失败后，scheduler 直接选择 group 中最后一个 `RUNNING` request 抢占，而不是选择：

- 在瓶颈 receiver rank 上贡献最多 remote edge 的请求；
- 占用瓶颈 rank 最多 blocks 的请求；
- 抢占后能让下一次规划变为可行的最小 victim 集合。

代码位置：`csrc/nanodeploy/scheduler/scheduler.cpp:2067-2084`。

这解释了为什么大量 2–5-block 的短请求被反复选中，却不能解除热点。

### 4.4 recovery 没有 backoff，形成闭环

抢占会清空 sequence 的运行态 KV/生成进度，建立 singleton recovery batch，并 `push_front` 到 pending 队首。下一次 schedule 可以在容量和 receiver 形状没有变化时立即重新 admission，随后再次进入同一失败状态。

代码位置：`csrc/nanodeploy/scheduler/scheduler.cpp:2365-2435`。

当前没有 capacity-generation/receiver-generation 条件、blocked recovery queue 或跨 step backoff；因此会形成：

```text
plan failed -> preempt small newest request -> recovery push_front
-> immediate re-admission/merge -> same plan failed
```

### 4.5 固定拓扑造成局部资源不可替代

每个 running group 固定属于一个 DP，不能跨 DP 使用空闲 SP participant。`LeastBatch` 平衡请求数量，而不是 prompt tokens、KV blocks 或 receiver edges，因此可能出现一个 DP/owner 已达到 receiver 上限，另一 DP 或其他 ranks 仍有大量空闲 blocks。

设计约束见 `docs-dev/loongserve_style_scheduler_design.md:118-127`。

## 5. 与原版 LoongServe 的区别

NanoDeploy 当前实现是共享 NanoDeploy data/model plane 的 LS-Decode-Core scheduler-policy baseline，并不是完整 LoongServe reproduction。

原版 LoongServe 有两层重要保护：

1. admission 按请求最大剩余输出估算未来 KV，可能触发 eviction 时先让请求排队；见 `LoongServe/paper-tex-src/sections/design.tex:77-80` 和 `LoongServe/loongserve/longserve_server/router/req_queue.py:54-78`；
2. decode 容量不足时，先 merge batch、利用全局 idle instances 和 memory scale-up，仍无法满足才 pause/offload；见 `LoongServe/loongserve/longserve_server/router/manager.py:844-970`。

NanoDeploy 当前 initial admission 主要预算 prompt、pending token 和固定 `reserved_blocks_per_req=1`，没有按 `max_tokens` 预留完整未来 KV。原版 LoongServe 也有清空 KV 后重算的 fallback，开源实现并非完全没有缺陷；但它通常受 future-aware admission/backpressure 保护。因此本轮在低 KV 占用下持续重入，不能归因于 LoongServe multi-master 的核心设计。

## 6. 负载因素

前 7,200 条请求平均输出长度约 598.9 tokens，`rate=20` 对应约：

```text
20 × 598.9 = 11,978 output tokens/s
```

这是很高的 offered load，完成速率长期低于到达速率时，排队必然增长。它可以解释后期拥塞，却不能解释正式开始约 20 秒、全局 prompt blocks 仅约 4.38% 时发生的循环抢占。

## 7. Consolidation 判断

本轮 consolidation 使用 `execute` 和非常积极的 `2/2/1` 阈值，理论上可能把 KV 压到少数 owner rank，放大 receiver 热点或 scale-down/scale-up 抖动。

但旧运行关闭了详细 NanoDeploy 日志，缺少 consolidation action、迁移前后 rank blocks 和 receiver counts，不能确认首次失败前是否实际执行过有效迁移。新的诊断复跑记录到零次 consolidation action，已排除 consolidation 是该次复现的首因；它在其他运行中是否会成为放大器，仍需单独 A/B。

## 8. 诊断复跑方案

第一轮采用相同模型、数据、topology、rate 和 consolidation 设置，但把 offered-load window 缩短到 30 秒（600 requests）。它是相同 seed 下原 workload 的精确前缀：上一轮约 20 秒首次失败，最后一个采样到达约 28.18 秒，因此仍有约 8 秒的故障观测窗口。该运行只用于捕获首次失败和后续 recovery 循环，不用于报告吞吐或尾延迟；若没有复现，再延长到 60 秒（1,200 requests）。

开启：

```bash
export NANODEPLOY_LOG_DECODE_STEP_DETAIL=1
export NANODEPLOY_LOG_MODEL_FORWARD_TIMING=1
```

并传入：

```text
--verbose-nanodeploy-logs
```

预期采集：

- 每步 `free_blocks[dp][sp]`；
- 每步 `sp_recv_counts[dp][sp]`；
- group rank allocation、master assignment、KV blocks 和 pending append blocks；
- preempted sequence IDs 和 failure reason；
- consolidation candidate/action/reason；
- 16 rank CUDA-event 汇总的 `model_forward_gpu_ms`，以及 driver 侧 `model_runner_duration_ms`/step ITL；
- pending batch 数量、age 和 admission attempts。

重点判据：若首次失败前后所有相关 rank 仍有大量 free blocks，同时某 owner 的 receiver count 接近 128，并出现同一 recovery request 反复 admission/preempt，则可以直接确认 receiver/recovery thrashing；若 receiver 未接近上限，则需要用该次 placement snapshot 进一步检查 planner false-negative 或 reservation 计算。

## 9. 诊断复跑结果

诊断日志：`docs-dev/ls_style_issue001_2node_dp2sp8_r20_diag30s_verbose_forward_20260716.log`。

测试使用原配置和相同 seed 的前 600 个请求，最后一个采样到达为 28.17699 秒。正式 workload 于 08:08:27 开始，首次失败于 08:08:50 出现，即约 23 秒。发送窗口结束且活锁证据已完整采集后，在 08:09:16 主动停止 drain；因此该运行没有最终 JSONL，也不作为性能结果。

### 9.1 Receiver 是直接触发条件

首次失败前最后一个成功 decode step 中，两个 group 分别为 173 和 168 个请求，均已使用 8 个 ranks。逐 rank 状态为：

```text
DP0 free_blocks    = [13844, 13909, 13850, 13859,  8745, 12010, 13854, 13858]
DP0 recv_counts    = [  128,    65,    68,   104,    47,    70,    87,   111]

DP1 free_blocks    = [13812, 13714, 13921, 13810, 13668, 13811, 13810, 13926]
DP1 recv_counts    = [  116,    49,    62,    79,   108,    88,   126,    54]
```

由此得到：

- DP0 KV block 利用率 7.50%；
- DP1 KV block 利用率 1.67%；
- 集群整体 KV block 利用率 4.59%；
- 最紧张的单 rank 仍有 `8,745 / 14,044 = 62.27%` blocks 空闲；
- receiver 已分别达到 128 和 126。

失败调度中，group 3 在合入 48 个 pending requests 后共有 216 个请求。对应 DP 的任意 rank 至少仍有 13,668 个 free blocks；即使保守地给 216 个请求各计一个 pending append block 和一个 reservation block，也只需要不超过 432 blocks。因此 `estimate_pending_append_capacity(...)` 对候选 rank 不可能为零，联合候选谓词中只剩 `receiver_prefix_capacity(...)` 可以导致失败。

这确认了 generic `append capacity cannot cover remaining requests` 实际由 receiver cap 触发，而不是 KV blocks 用尽。

### 9.2 存在合法均衡 assignment，确认 planner false negative

诊断日志记录了每个 batch 的 initial KV ranks 和每轮 sequence-to-master assignment。由于本轮没有 consolidation action，可按时间顺序取这些 ranks 的并集，精确重建每条 active sequence 的 KV owners。重建出的上一轮 receiver counts 与日志矩阵完全一致：

```text
reconstructed = [116, 49, 62, 79, 108, 88, 126, 54]
logged        = [116, 49, 62, 79, 108, 88, 126, 54]
```

216 个候选请求在 8 个 ranks 上的 owner counts 为：

```text
[140, 70, 91, 101, 129, 111, 147, 109]
```

将每条请求分配给它已有 KV 的一个 owner，并在满足 receiver lower bound 后平衡 master load，可以得到：

```text
master_load   = [26, 26, 26, 26, 26, 26, 26, 34]
receiver_load = [114, 44, 65, 75, 103, 85, 121, 75]
```

所有 master batch 均不小于 `batch_per_master=8`，最大只有 34；所有 receiver 均不超过 128，最大为 121；append blocks 也有充足余量。这是一个满足当前物理和 metadata 约束的合法计划。

因此此次失败不是请求集合本身不可调度，而是当前连续 prefix/chunk、无回溯的 source-greedy 没有找到已存在的合法 assignment，造成资源未充分利用。

### 9.3 立即恢复形成确定的活锁

主动停止前共记录：

- 113 个包含失败的调度 iteration；
- 每个 iteration 达到 `attention_sp + 1 = 9` 次重试/抢占；
- 合计 1,017 次 preemption，但只有 12 个 unique victims；
- `seq_id=430..435` 各被抢占 113 次，`427..429` 各 112 次，`436..438` 各 1 次；
- 1,012 次 recovery admission；
- pending batch 峰值 172，pending request 峰值 221；
- oldest pending batch age 达到 1,203 steps，最大 admission attempts 达到 1,204。

日志直接显示 recovery request 被 `push_front` 后，在同一 group 和几乎相同资源状态下再次 admission。该数据确认了 `preempt -> recovery push_front -> immediate re-admission -> same failure` 的 liveness bug。

### 9.4 Consolidation 在该复现中没有执行

333 个 decode iterations 中：

- `kv_consolidation_decision_reason=no_candidate`：149 次；
- `pending_admission_not_proven`：184 次；
- consolidation action：0 次。

所以虽然配置为 aggressive `execute(2/2/1)`，本次早期 receiver failure 与 recovery thrashing 并不是 consolidation migration 造成的。

### 9.5 Actual forward 已成功采集

`NANODEPLOY_LOG_MODEL_FORWARD_TIMING=1` 共得到 326 个正式 workload CUDA-event 样本。该时间覆盖每个 worker 的实际 `run_model`，包括模型内 collective，不包括调度、RPC prepare 和 sampling；distributed critical path 取每轮 16 ranks 中的最慢 rank。

```text
                              首次失败前        首次失败后
critical-path forward mean      93.2 ms           111.4 ms
critical-path forward p50       94.3 ms           113.4 ms
rank-mean forward mean          82.7 ms           100.6 ms
rank-min forward mean           45.1 ms            54.7 ms
planner latency mean             1.05 ms            6.14 ms
driver scheduler overhead mean   3.30 ms           25.05 ms
step ITL mean                   103.9 ms           141.0 ms
```

详细日志会扰动 driver wall time，因此 scheduler overhead 和 step ITL 只用于故障诊断；CUDA-event actual forward 仍可用于区分模型执行与调度/恢复开销。数据表明抢占循环不仅没有释放真实 KV 瓶颈，还把平均调度开销提高约 7.6 倍，并使 step ITL 明显恶化。

## 10. 修复优先级

1. 将 block append、receiver overflow、metadata batch limit 等失败原因拆开，并在失败时保存逐 rank 快照；
2. recovery 增加 capacity/receiver epoch 或 backoff，禁止无状态变化的立即重入；
3. master 分配改为 receiver-aware、owner-aware 的 matching/flow 或带回溯的分配；
4. victim 按实际瓶颈贡献选择，而不是无条件选择最后一个 request；
5. admission 引入未来 KV/backpressure，避免把长期不可持续负载全部放入 running 状态；
6. routing 同时考虑 KV blocks、长请求和 receiver edges；
7. 完成上述修复后，再用 consolidation `off/shadow/execute` 和 `max_num_recv_seqs=128/256` 做 A/B，区分根因与放大因素。
