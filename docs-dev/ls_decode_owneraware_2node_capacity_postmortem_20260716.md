# LS-Decode owner-aware 两机复跑与容量鲁棒性复盘

日期：2026-07-16

## 1. 结论摘要

本轮两次 16 卡复跑确认了两件彼此独立的事情：

1. commit `83890c8` 的 owner-aware planner 修复有效。两次运行均为 0 planner failure、0 preemption、0 recovery，旧的 receiver/source-greedy 假阴性和立即恢复活锁没有再出现。
2. 整个系统仍未通过两机验收。`141 GiB` 运行在 drain 后期执行 KV consolidation 时因 NCCL/CUDA 运行期分配失败而 fatal；`140 GiB` 运行没有触发该错误，但产生了严重的 admission capacity cliff 和每步全 pending 队列重算，最终由人工停止。

把 `gpu_memory_limit_gb` 从 141 调到 140 不是修复。它使 scheduler 可见的 KV 容量从每 rank 14,044 blocks 变为 14,030 blocks，只减少 14 blocks，约 0.10%；但 6 分钟完成数从约 5,216 降到 3,165，pending 峰值从 71 batches / 185 requests 增长到 982 / 3,643，发送期 scheduler overhead p95 从 5.49 ms 增长到 233.38 ms。这个结果暴露的是容量敏感的离散可行性判断、无退避重试和 consolidation gating 共同造成的非线性放大，而不是一个应该靠固定少分配 1 GiB 解决的问题。

`140 GiB` 没有出现 NCCL OOM 也不能证明显存预留已经修好：该运行有 pending batch 时 consolidation 被无条件关闭，因此实际 consolidation 次数为 0，根本没有覆盖失败路径。

两份日志都没有产生完整 `Benchmark Results` 或最终 JSONL，不能作为最终性能结果。

## 2. 测试范围和唯一配置差异

共同配置：

- 2 nodes / 16 H200 GPUs；Attention `DP2 × SP8 × TP1`，FFN `EP16`；
- Issue 1% CSV，7,200 requests，`20 req/s`，360 秒发送窗口；最后一个采样 arrival 为 354.6172 秒；
- `max_num_seqs=256`，`max_num_recv_seqs=128`；
- `ls_decode_batch_per_master=128`，即本轮已经使用 T=128；
- 自动 initial KV DoP；
- consolidation `execute`，`stable/cooldown/check=2/2/1`，source block budget 128，migration chunk 64 tokens；
- full CUDA Graph；开启 verbose scheduler 日志和 actual-forward CUDA event 统计。

唯一有意义的配置差异如下：

| 运行 | `gpu_memory_limit_gb` | KV blocks / rank | 与 141 的差值 |
|---|---:|---:|---:|
| owner-aware 主跑 | 141 | 14,044 | 0 |
| memory retry | 140 | 14,030 | -14（-0.10%） |

两次运行使用相同请求序列和参数。分布式执行时序仍可能造成 group state 分叉，因此不能把所有结果差异严格归因于 14 个 blocks；但如此小的可见容量变化被放大为巨大的吞吐和队列差异，本身已经说明当前策略缺少容量鲁棒性。

日志：

- `docs-dev/ls_style_issue001_2node_dp2sp8_r20_t128_owneraware_6min_20260716.log`
- `docs-dev/ls_style_issue001_2node_dp2sp8_r20_t128_owneraware_mem140_retry_20260716.log`

## 3. 两次运行的定量结果

`tqdm` elapsed 只有整秒精度，所以表中的 6 分钟完成数采用 `[06:00]` 整秒桶的最后观测值，而不是伪装成精确到子秒的吞吐。

| 指标 | 141 GiB | 140 GiB |
|---|---:|---:|
| `[06:00]` 最后完成数 | 5,216 / 7,200 | 3,165 / 7,200 |
| 近似完成速率 | 14.49 req/s | 8.79 req/s |
| pending 峰值（batches / requests） | 71 / 185 | 982 / 3,643 |
| 最老 pending 最大 age / attempts | 4,089 / 4,090 | 1,802 / 1,803 |
| peak KV util / min free blocks | 99.45% / 77 | 99.59% / 57 |
| 发送期 actual forward p50 / p95 | 122.76 / 177.01 ms | 125.73 / 205.11 ms |
| 发送期 scheduler overhead mean / p95 | 4.94 / 5.49 ms | 51.75 / 233.38 ms |
| planner failure / preemption / recovery | 0 / 0 / 0 | 0 / 0 / 0 |
| consolidation 成功次数 | 8 | 0 |
| 退出方式 | NCCL/CUDA fatal @ 09:37 | 人工停止 @ 08:18 |
| 最后完成数 | 7,111；剩余 89 | 3,565；未 drain |

141 GiB 正式阶段的 actual-forward 统计为 3,646 个样本：mean 133.459 ms，p50 129.688 ms，p95 189.976 ms，p99 219.520 ms，max 233.804 ms。该字段等于 16 ranks 中的 `model_forward_gpu_rank_max_ms`，是实际 `run_model` CUDA event 的 distributed critical path，不包含 scheduler、RPC prepare 和 sampling。正式阶段 planner latency 为 mean 2.100 ms，p50 2.179 ms，p95 6.103 ms，p99 7.475 ms，max 17.695 ms。

这说明新加的 actual-forward 功能已经采到数据。两次发送期 forward p50 接近，而完成数相差 39.3%；140 GiB 的主要新增损失发生在 scheduler/admission 路径，而不能用模型 forward 本身解释。

## 4. 已修复问题和仍未修复问题的边界

owner-aware planner 修复经过了真实两机压力：

- 原故障点之后没有 `plan failed`；
- 两次运行所有 `preempted_sequence_ids` 均为空；
- 没有 recovery admission；
- 141 GiB 的 planner p99 仍为 7.475 ms，没有出现 fallback 搜索失控。

因此旧的“存在合法 owner-aware assignment，但 source-greedy 误报不可行，随后反复抢占”的问题可以认为已修复。修复内容和单机验证见 `docs-dev/ls_decode_owner_aware_planner_fix_20260716.md`。

但这只修复了 decode master assignment 的假阴性，并不等于 admission、KV consolidation、物理显存生命周期和 overload control 都已修复。本轮恰好暴露了这些后续层的问题。

## 5. 141 GiB：逻辑 KV 空闲不等于物理显存空闲

141 GiB 运行在发送窗口结束后继续 drain，到 7,111 / 7,200 时 fatal。此前已经成功完成 8 次 consolidation：group 2 四次迁移共处理 27,622 tokens，group 3 四次处理 858 tokens；8 次 maintenance 总耗时 674.51 ms，均值 84.31 ms。

随后一次 consolidation 走到：

```text
execute_planned_ls_kv_scale_down
  -> copy_kv_ranges_p2p
  -> KVCacheP2PTransport.execute
  -> dist.irecv
  -> NCCL unhandled cuda error
  -> Failed to CUDA calloc async 40 bytes
```

异常前最后一条 compact state 是：

```text
total_batch_size=103
max_sp_batch_size=61
min_free_blocks=5382
max_kv_util_pct=61.68
pending=0
```

这里的 5,382 free blocks 只是预分配 KV tensor 内部的逻辑 free-list，不是未分配的 GPU 显存。`nanodeploy/worker/cache.py:174-183` 在启动阶段已经一次性分配全部 KV blocks；scheduler 的 block manager 只管理这些槽位，释放逻辑 block 不会缩小该 tensor。因此 61.68% KV utilization 不能推出还有 38.32% raw VRAM 可供 NCCL 使用。

当前内存初始化顺序也会遗漏运行期常驻开销：

1. `CacheContext` 根据当时的 `cudaMemGetInfo`、Torch peak/current 和 migration scratch 估算 block 数；
2. 先分配固定 scratch 和整个 KV tensor；
3. 随后才 capture full CUDA Graph；
4. `KVCacheP2PTransport` 构造时不执行通信，没有对所有可能 SP peer pair 做 P2P/NCCL warmup；真正的 `isend/irecv` 只在 consolidation 运行期发生。

migration scratch 已经在 sizing 公式中扣除并提前分配，所以不能简单归因于 scratch 漏算。日志和代码共同支持的是：CUDA Graph、NCCL P2P lazy state、allocator segment/metadata 等 post-sizing 运行期内存没有被可靠纳入最终 headroom。具体是哪一个新 peer/channel 触发额外分配，现有日志没有 NCCL debug 和 per-rank memory snapshot，不能进一步当作已证实事实。

报错中的 40 bytes 也不表示“只差 40 bytes”。CUDA async allocator 可能需要获取新的 page/segment 或内部 metadata；在大 KV tensor和 graph-pinned allocation 已占据地址空间时，一个很小的请求也可能失败。

当前 engine 在 copy RPC 异常后把自身标记为 fatal，是因为各 worker 的 copy 完成状态可能不一致，不能盲目 rollback 后继续服务。这个 fail-closed 行为是安全的，但也说明 consolidation 还不具备运行时故障隔离能力。

## 6. 140 GiB：admission capacity cliff 和调度开销正反馈

140 GiB 运行没有 planner failure，也没有 runtime fatal，但从约 69 秒开始出现 pending。到 6 分钟：

```text
pending_batch_count=982
pending_request_count=3643
oldest_pending_batch_age_steps=1802
max_pending_batch_attempts=1803
atomic_admission_no_fit_count=982
total_batch_size=405
max_sp_batch_size=128
min_free_blocks=4510
max_kv_util_pct=67.85
scheduler_overhead=132.55ms
```

即当步 982 个 pending batch 被全部重新检查，并且 982 个全部 no-fit。全程 2,428 条 pending telemetry 中，2,427 条的 `atomic_admission_no_fit_count` 都等于当步 pending batch 数；累计约 775,507 次 no-fit 检查。最极端的一次 scheduler overhead 为 269.74 ms。

`rate=20` 本身高于两次运行的长期 completion rate，因此 offered-load window 内出现排队是预期的。140 GiB 在 6 分钟时日志中的 `waiting_total_blocks_sum=761694` 还把同一 centralized queue 按 DP2 重复求和；去重后约 380,847 blocks，仍超过集群 224,480 个逻辑 KV blocks。问题不是“为什么不能把所有 pending 同时 admit”，而是 overload 时 scheduler 为这些暂时不可放置的 batch 每步重复做无界工作，并且关闭了资源整理路径。

代码路径与日志完全一致：`_schedule_ls_decode_admission()` 每一步 snapshot 全部 pending IDs；每个 batch 无条件增加 attempts，先尝试每个 DP 的 standalone placement，失败后再尝试每个 DP 的每个 group merge placement。它没有 capacity epoch、失败 fingerprint、backoff 或单步 probe budget。若没有 batch 能成功 admission，就会在每个 decode step 做 O(pending × DP × groups) 的重复可行性计算。

这不是严格 FIFO head-of-line blocking：一个 batch no-fit 后代码会继续看后面的 batch。但代价是 overload 时每步遍历整个队列。形成的恶化环是：

```text
容量/形状约束导致更多 pending
  -> 每步 admission 计算变长
  -> 单位时间 decode step 和 completion 减少
  -> 可释放的 block/rank 更少，arrival 继续进入 pending
  -> 下一步扫描更大的 pending 队列
```

这解释了为什么只有 0.10% 的 KV block 差异，在不同的动态 placement 轨迹中会被放大为明显的性能悬崖。它不证明 14 blocks 是唯一初始原因，但证明系统没有把较小容量转化为平滑、有界的并发下降。

同时，`atomic no-fit` 把多个原因折叠在一个布尔结果里：可能是 per-rank KV/headroom、可用 rank pool、receiver metadata、master metadata 或 group ownership。日志没有 required/free per-rank deficit，因此不能把 140 GiB 的所有 no-fit 简单解释为“KV cache 已满”。反例是 6 分钟时最紧 rank 仍有 4,510 个逻辑 blocks，KV util 只有 67.85%，但全部 982 个 batch 仍 no-fit；一个两请求 batch 最终在第 1,804 次尝试才 merge 成功，其目标 ranks 在前一刻也有大量逻辑空闲。receiver/group shape 是强候选，但现有 telemetry 不足以最终定因。

原子 batch 只保证在 empty-system capacity 下可放置；进入 pending 后不会按当前容量拆分或重组。standalone 又只能使用 unallocated ranks，merge 只能使用目标 group 已有 ranks 加 unallocated ranks，不能任意借用其他 group 的空闲资源。这些离散约束进一步放大了小容量变化。

## 7. Pending 与 consolidation 的错误耦合

`Scheduler::_maybe_plan_ls_kv_consolidation()` 在 pending queue 非空时立即返回，并重置所有 group 的 stable counter，日志 reason 为 `pending_admission_not_proven`。代码实际上没有做 pending-aware what-if proof，只是无条件禁止 consolidation。

本轮证据：

- 141 GiB：pending 清空后才开始 consolidation，最终成功 8 次并覆盖到失败路径；
- 140 GiB：`pending_admission_not_proven` 1,599 次，`no_candidate` 658 次，consolidation candidate/action 都为 0。

这造成了错误耦合：最需要释放 group-owned rank、整理 placement 的 pending 场景，反而不允许 consolidation。consolidation 不一定总能帮助 admission，因此正确做法不是无条件执行；但应该证明“某个 consolidation 能使一个具体的 aged pending batch 可放置”后再执行，而不是 pending 非空就一票否决。

也正因为 140 GiB 从未执行 consolidation，本轮不能用它验证“多留 1 GiB 是否解决 NCCL runtime allocation”。

## 8. 正确的系统性修复方向

### P0：按真实生命周期自动确定物理显存预算

- 在最终 KV pool sizing 前，让所有可能的 attention-DP/SP P2P peer/direction 完成 communicator 和 allocator warmup；若 transport 依赖最终 KV pointer，则使用 provisional buffer，随后重新测量并重建。
- 把 CUDA Graph 纳入闭环：保守初配 KV、capture graph、测量 raw free/reserved；headroom 不足时按 block 自动回退并重建/recapture，而不是依赖人工选择 140 或 141。
- 启动结束对每个 rank 同时记录 `cudaMemGetInfo`、Torch allocated/reserved、model、graph pool、migration scratch、NCCL warmup 增量、最终 KV blocks 和剩余 raw headroom。
- reserve 应来自实测常驻开销、allocator granularity 和安全抖动，而不是固定减 1 GiB 的 magic number。

### P0：让 admission 成本和行为对容量有界

- 为 free blocks、rank ownership、receiver/master counters 建立 `capacity_epoch`；pending batch 保存上次失败 epoch 和结构化失败原因，资源形状未变化时不重复完整规划。
- 给每步 pending probe 设置预算，结合 aging/fairness 重试；不能让 queue 长度直接吃掉 decode 时间。
- `_plan_ls_initial_placement()` 返回每个 DP/rank 的结构化 deficit，而不是一个 `nullopt`：明确 required/free blocks、可用 ranks、receiver projected/cap 和 metadata cap。
- 对长期 no-fit 的原子 batch做容量自适应 re-batch/split；同时引入 future-KV-aware admission 和入口 backpressure。
- 去掉“pending 非空即关闭 consolidation”，改为 pending-aware benefit proof，并把 maintenance reservation 与后续 admission 做成受保护的两阶段操作。

### P0：限制 consolidation 故障域

- 对 copy transaction 增加 prepare/copy/ack phase，区分“任何传输开始前失败”和“完成状态不确定”。前者可安全 abort 并对 consolidation cooldown/disable；后者才 fail-closed 或重建 communicator/worker。
- 根本目标仍是让所有正常路径需要的 GPU runtime allocation 在启动 warmup 阶段完成，避免服务中途首次分配。

### 验收方法

下一轮不应只比较一个 140/141 GiB 点，而应做 block-capacity / memory-limit sweep。验收要求：

- 各容量下无 fatal；
- 容量下降时吞吐和并发平滑或至少可解释地退化；
- scheduler overhead 有固定上界，不随 pending 长度线性爆炸；
- 同一 pending batch 不在 capacity epoch 未变化时每步重试；
- backlog 存在时，consolidation 能在有可证 admission 收益时运行；
- no-fit 能准确区分 KV、rank ownership、receiver 和 metadata；
- actual forward、scheduler overhead 和 end-to-end ITL 分开报告。

## 9. 当前状态

- owner-aware planner 修复已提交：`83890c8 fix: make LS decode planner owner aware`；
- 本轮停止后没有继续修改 scheduler，也没有把 140 GiB 当作默认配置或修复；
- 两机 benchmark、Ray placement group 和 16 张 GPU 均已停止/释放；
- 后续工作应从上述容量自适应设计开始，而不是继续微调固定显存扣减值。
