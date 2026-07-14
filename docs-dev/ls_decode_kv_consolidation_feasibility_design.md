# NanoDeploy Decode-only LS KV Consolidation 可行性分析与实施设计

## 1. 结论

在 NanoDeploy 当前的 DeepSeek-V3 Decode-only、Dummy Prefill、`4 DP × 8 SP / EP32` 场景中，实现 KV Consolidation 是可落地的，也是解决以下状态退化的正确方向：

```text
master_dop 已随 batch 下降
kv_dop 仍停留在历史峰值 8
仍然存在多 rank KV owner 和 SP Attention 通信
group allocation 无法释放 rank 给同 DP 的其他 group
```

但不能把策略简单定义为：

```text
if KV cache usage < threshold:
    consolidate()
```

低 KV 使用率只能作为候选过滤条件，不能独立决定迁移。正确的执行条件应当同时满足：

1. 当前 KV placement 可以在考虑 block fragmentation、pending token、append headroom 和迁移 scratch 后，通过逐 source-rank evacuation 压缩到更少 ranks；
2. 压缩后至少能彻底清空一个 rank，而不只是把 KV 分布变得更均匀；
3. 目标 rank 数能够覆盖当前及短期内的 Decode master demand，避免下一轮马上重新扩到 8；
4. 迁移造成的暂停和传输成本能在 group 的剩余 Decode 生命周期内摊销，或者释放 rank 能立即解除 admission/resource pressure；
5. 数据面真实复制 KV，source 在 commit 前始终是权威副本，metadata 与物理 block 所有权在无失败 commit 中一起切换。

推荐将该能力命名为：

```text
LS-Decode KV Consolidation
```

它是现有 `LS-Decode-Core` 的扩展，不应声称等同于完整 LoongServe。

这里的“可落地”指架构上可实现，不是只改配置即可开启。原始项目没有 live Decode KV relayout、maintenance action、intra-DP KV P2P RPC 和 no-fail metadata commit；当前分支已经补上手动 source-only transaction 和 P2P 数据面，但自动 scheduler action、触发 policy、生产级 no-fail commit 和 4-DP/EP32 验收仍未完成。落地顺序仍必须先经过手动 correctness，再接 pressure policy，不能直接从 usage threshold 跳到生产 execute。

当前已落地一条可手动调用的 stop-the-world scale-down correctness 链路：`Scheduler::plan_ls_kv_scale_down()` 为 passive source 生成 token ranges、预留 destination blocks 并预构造 metadata；`execute_ls_kv_scale_down()` 向所有 workers 下发 `dist.isend/irecv` copy；全部 workers 成功后才交换 ACTIVE context、释放 source blocks 并缩小 group allocation，失败则释放预留并保留旧 placement。CPU 自动化集成测试、双进程 Gloo、双 GPU NCCL，以及 1-DP × 8-SP 的真实 NCCL 8→7 preflight 均已通过。该入口仍是 iteration boundary 上的手动 API，没有接 `LLMEngine.step()` action/policy，也不能作为生产 execute 开关。

本文后续定义的 `ScheduleAction`、`EngineStepResult`、pressure intent、engine-fatal 分类和真实 forward/4-DP 验收是下一阶段的确定实现目标，不是对当前代码状态的描述。

### 1.1 能解决什么

第一版 KV Consolidation 可以做到：

- 将一个 group 的逻辑 `kv_dop` 从 8 降到更小值；
- 减少真实 KV owner 数、remote Attention edges、Q/O/LSE payload 和 partial Attention participants；
- 清空并回收 `allocated_attention_ranks` 中的 ranks，供同一 DP 的其他 Decode group admission 或 scale-up 使用；
- 当一个 DP 内所有 real sequences 都变成 master-local KV 时，允许该 DP 的 Attention 路径完全跳过 SP all-to-all；
- 避免 group allocation 只增不减，恢复 scheduler 层面的动态性。

### 1.2 第一版不能解决什么

当前 workers、Attention SP mesh 和 EP32 mesh 在启动时固定创建。Scheduler 还会给没有 real master work 的 rank 补 dummy sequence，保证所有 workers 进入相同的 EP/SP collective cadence。

因此第一版 KV Consolidation 不会：

- 动态销毁或重建 NCCL/process group；
- 改变 `attention_sp=8` 或 `ffn_ep=32` 的物理 world size；
- 让被某个 group 释放的 GPU 完全停止参与 EP32 collective；
- 自动消除 fixed-mesh collective 的启动和同步基线成本；
- 把一个 DP 的 rank 借给另一个 DP；
- 改变 DeepSeek-V3 experts 在 EP32 ranks 上的权重分布。

所以必须区分三层结果：

| 层次 | Consolidation 后的变化 | 第一版是否支持 |
|---|---|---|
| Group scheduler | `allocated_attention_ranks` 和 `kv_dop` 可下降 | 是 |
| SP Attention data plane | KV owner、remote edges、payload、partial Attention 计算可下降 | 是 |
| 物理 SP/EP topology | 8-SP/EP32 process group 与 collective cadence 缩小 | 否 |

如果最终目标是“从物理执行上只让 4 张卡参与整个 DeepSeek-V3 layer”，还需要动态/稀疏 SP 通信以及 EP expert replication/repartition 或 Attention/FFN 解耦。这是独立于 KV Consolidation 的更大工程。

## 2. 当前实现为什么会停在 `kv_dop=8`

当前 Core 的扩缩容语义是：

1. Dummy Prefill admission 选择最小可行 `D_init`，为 prompt 建立逻辑 KV placement；
2. Decode compute/memory scale-up 可以加入新 master；
3. 新 master 只承接当前和后续 token，不迁移历史 KV；
4. batch 变小时，每轮 planner 可以降低 `master_dop`；
5. 仍持有 committed KV 的旧 master 退化为 passive participant；
6. 只有某个 rank 对该 group 的 committed KV 和 pending token 都归零时，rank 才会被自然回收。

相关实现位于：

- `csrc/nanodeploy/scheduler/scheduler.cpp::_schedule_ls_decode()`；
- `csrc/nanodeploy/scheduler/sp_state_manager.cpp::plan_iteration_masters_source_greedy()`；
- `csrc/nanodeploy/scheduler/sp_state_manager.cpp::commit_iteration_master_plan()`；
- `docs-dev/loongserve_style_scheduler_design.md` 第 3、7、16 节。

这意味着：

```text
master role 是 iteration-scoped
historical KV ownership 是 sticky state
```

只要历史上 8 个 ranks 都写入过仍存活 requests 的 KV，降低 Decode threshold 只能降低本轮 master 数，不能降低 `kv_dop`。

### 2.1 固定通信 cadence 的额外影响

`Scheduler::_schedule_ls_decode()` 会为每个没有 master work 的 SP rank 补 dummy sequence，并明确要求每个 DP 进入固定 EP/SP cadence。Attention metadata 仍以 8-rank mesh 生成，`hao_basic` 连接的是 full mesh。

KV Consolidation 以后：

- `q_mask`、`res_lse_mask` 和 offsets 可以减少真实传输；
- 没有 KV 的 rank 不再为该 group 计算 partial Attention；
- 但只要一个 DP 内仍有任意 remote KV relation，8 个 SP ranks 仍需以一致顺序调用 SP backend；
- EP32 仍由全部 workers 参与。

因此性能验收不能只观察 `kv_dop: 8 -> 4`，还必须观察：

```text
remote edge count
Q/O/LSE actual bytes
attention_compute_bs per rank
SP all-to-all calls 是否完全消失
EP dispatch/combine latency 是否不变
end-to-end ITL
```

## 3. 为什么“长期 KV usage 低于阈值”还不够

### 3.1 平均使用率低不等于能释放 rank

假设一个 group 有 8 个 participants，总 KV 使用率只有 25%。理论上总容量可放入 2 个 ranks，但实际 placement 还受到以下约束：

- block size 为 64，容量以 block 而不是连续 token 计费；
- 每个 sequence 在每个 owner rank 上都可能有一个 partial tail block；
- 每条 running sequence 有一个尚未进入 committed history 的 pending input token；
- planner 还会为下一 sampled token 和 `reserved_blocks_per_req` 留出空间；
- destination rank 可能需要满足 `max_num_seqs`、`max_num_recv_seqs` 和 CUDA Graph metadata capacity；
- evacuation 期间还要预留固定 migration scratch 和不能直接复用的 destination blocks。

所以必须运行 exact feasibility planner，而不能用：

```text
ceil(total_used_blocks / blocks_per_rank)
```

直接决定 target DoP。该公式只能产生下界。

### 3.2 KV 低但 batch 仍可能 compute-bound

例如：

```text
kv_dop=8
KV usage=20%
real batch=512
T_compute=64
```

当前 master planner 仍然需要 8 个 masters。此时把 KV 压到 2 个 ranks 后，下一轮 planner 又会启用其他 6 个 masters，并开始在它们上面写新 KV，`kv_dop` 很快重新增长。

因此目标 DoP 至少应满足：

```text
D_target >= D_compute
```

其中 `D_compute` 不是简单写死的阈值，而应来自当前 source-style master plan 和正式 EP32 profiling。

### 3.3 Consolidation 可能降低通信，却增加 Attention critical path

把 8 个 ranks 上的 KV 集中到 2 个 ranks，会产生两个相反效果：

- remote edges 和通信 payload 下降；
- retained ranks 的本地 KV token 数增加，单 rank Attention 时间上升。

在当前 full-mesh backend 上，通信固定开销未必随 payload 成比例下降。因此不能假定 `kv_dop` 越小越快，必须比较目标 placement 前后的预测 ITL。

### 3.4 使用率抖动会导致 scale-down/scale-up thrashing

请求完成会让 KV usage 突然下降，新 admission 或 batch 增长又可能立刻要求新 masters。如果只设置单一阈值，会出现：

```text
8 -> 4 consolidation
下一轮加入新 batch
4 -> 8 scale-up
稍后再次 8 -> 4 consolidation
```

因此需要 stable window、high/low watermark 和 cooldown。

### 3.5 当前 `max_kv_util_pct` 不能作为触发指标

`nanodeploy/engine/llm_engine.py` 当前日志中的 `max_kv_util_pct` 由所有 ranks 中的最小 free blocks 推导，只表示最忙 rank 的利用率。它不能回答：

- 哪个 group 的 KV 分布在多少 ranks；
- 哪些 ranks 可以完整 evacuation；
- destination 是否有足够 tail/新 block/append capacity；
- 释放 rank 后是否会被其他 group 使用；
- 需要迁移多少 bytes。

Consolidation 必须增加 group-level placement telemetry，不能复用该全局标量直接执行。

## 4. 推荐的触发策略

### 4.1 基础量

对每个 group `g`，每轮派生：

```text
B_g                 = real Decode batch size
K_tokens[g][rank]   = committed KV token count
K_blocks[g][rank]   = committed KV block count
D_kv                = count(K_tokens[g][rank] > 0)
D_master            = 当前 source-style master plan 的 master 数
C_free[rank]        = rank 的物理 free blocks
P_pending[seq]      = pending token target
```

定义当前 group 使用率作为观测值：

```text
U_group = sum(K_blocks[g][rank])
          / sum(usable_blocks[rank] for rank in current KV participants)
```

它可以用于快速过滤，但不用于最终 commit。

### 4.2 Target DoP

先计算长期目标：

```text
D_mem = exact relocation planner 找到的最小稳态 rank 数
D_compute = 当前及短期 master demand 需要的 rank 数
D_target = max(D_mem, D_compute)
```

随后为当前 maintenance event 选择一个 source rank，并验证从当前 placement 到“清空该 rank”这一跳真实可行。不能只证明最终 `D_target` 可行；逐 rank 执行可能在中间态遇到 destination capacity 瓶颈。

Exact planner 必须包含：

- committed KV blocks；
- destination 现有 block 的 committed tail、尚未提交的 tail 区间和新 block 开销；
- 当前 pending input token 的重新指派；
- 下一 sampled token reservation；
- `reserved_blocks_per_req`；
- per-rank block、sequence、receiver 和 graph capacity；
- 固定 migration scratch（在 KV block sizing 前已经扣除）；
- consolidation 后的安全 high watermark。

第一版建议只在：

```text
D_target < D_kv
```

时产生 candidate。MVP 的一次 maintenance event 必须完整清空一个 source rank，且最多释放一个 rank；chunk 只限制 scratch 和单次 send/recv 大小，不能把一个 source 的半迁移 placement 暴露给普通 Decode。

### 4.3 Pressure-first 与 opportunistic 两类触发

推荐维护 group-level 状态和 DP-level pressure state：

```text
candidate_target_dop
candidate_stable_steps
last_scale_up_step
last_consolidation_step
consolidation_inflight
placement_generation

oldest_no_fit_batch_id[dp]
oldest_no_fit_batch_generation[dp]
no_fit_required_ranks[dp]
```

其中 stable/cooldown 状态只约束 opportunistic path；`consolidation_inflight` 对两条路径都是硬约束。

策略应明确分为两条路径：

1. **Admission pressure**：`_schedule_ls_decode_admission()` 对最老 sealed batch 做过原子 admission 尝试并得到明确 `no_fit`，且 exact post-evacuation simulation 证明一个 source evacuation 或一个有限 evacuation chain 完成后，该 batch 可以 admission。该路径在 GPU correctness 完成后优先落地，可以绕过 opportunistic stable window 和普通 cooldown，但仍受 transaction-inflight、单 event 单 source、总 source blocks/bytes、总 pause 和 destination high watermark 限制；
2. **Opportunistic**：没有资源压力，只为降低 steady ITL 而迁移。只有同一个 `D_target` 连续保持若干 iterations、最近没有 scale-up 且真实 profiling 模型预测能够回本时才执行。

`no_fit` 必须来自已完成 rollback 的只读/原子 admission 尝试，不能用“queue 非空”代替。simulation 使用 shadow block counts、group allocations、receiver/graph capacity 和 pending reservation，检查的是目标 batch 的完整 admission，而不只是“全局多出一个 rank”。

LoongServe 的开源实现用“新 Prefill 收益 vs Decode KV migration cost”决定是否清空 Decode instance；NanoDeploy 没有真实 Prefill，最接近这一语义的是 admission-pressure path，而不是单独观察 group KV usage。

建议配置先以 shadow mode 收集数据，不直接给生产默认值：

```python
ls_kv_consolidation_mode = "off"        # off | shadow | execute
ls_kv_consolidation_candidate_util = 0.50
ls_kv_consolidation_target_high_watermark = 0.80
ls_kv_consolidation_stable_steps = 32
ls_kv_consolidation_cooldown_steps = 64
ls_kv_consolidation_max_released_ranks_per_event = 1
ls_kv_consolidation_max_source_blocks_per_event = 0
ls_kv_consolidation_max_pause_ms = 0.0
ls_kv_consolidation_migration_chunk_tokens = 0  # 0 关闭；手动 preflight 再显式设为 128 等正值
```

这些数值只能作为 shadow bring-up 起点，正式值必须来自 4-DP × 8-SP / EP32 trace 和 migration bandwidth profiling。

### 4.4 Cost/Benefit 条件

对 candidate plan 估算：

```text
MigrationCostMs = bytes_to_copy / measured_pack_send_scatter_bandwidth
                  + fixed_rpc_and_sync_cost
                  + metadata_commit_cost

StepSavingMs = PredictedITL(before) - PredictedITL(after)

PaybackSteps = MigrationCostMs / max(StepSavingMs, epsilon)
```

当没有 admission pressure 时，只在以下条件成立时执行：

```text
StepSavingMs > 0
PaybackSteps < ExpectedRemainingDecodeSteps * safety_factor
```

`ExpectedRemainingDecodeSteps` 可以结合：

- request 的 `max_tokens - num_completed_tokens`；
- 当前 group 中 requests 的剩余长度分布；
- capped prediction horizon，避免过度依赖不准确的 output length 预测。

当有 pending batch 且没有 unallocated rank 时，只有 exact admission simulation 明确证明“本次 evacuation，或受预算约束的完整 evacuation chain，会让指定 batch 成功 admission”，才允许 pressure override。不能因为全局 queue 非空就迁移；否则可能付出 pause 后仍无法 admission。

### 4.5 推荐的触发时机

不建议每轮无条件扫描和迁移。建议在以下事件发生后评估 candidate：

1. `postprocess()` 发现 requests finish，group KV 明显下降；
2. `master_dop` 从高值降到更低值并保持稳定；
3. 新 batch admission 因没有 unallocated rank 而等待；
4. 周期性低频检查，用于覆盖没有显式状态变化但已满足 payback 的 group。

事件只负责触发 planner。真实 copy 必须在没有 model forward/EP collective in flight 的 Decode iteration boundary 执行。

### 4.6 Scheduler 内的确定性决策顺序

自动路径不能在 `LLMEngine` 里根据日志指标另做一次策略判断；唯一决策点应在 centralized `Scheduler::schedule()`。推荐顺序固定为：

```text
1. 封存到达请求，刷新 finished/group/placement generation
2. 尝试最老 sealed batch 的原子 admission
3. admission 成功：返回 ADMISSION，不做 consolidation
4. admission 明确 no_fit：对指定 batch 运行 pressure chain simulation
5. pressure plan 可执行：PLAN + RESERVE，返回 KV_CONSOLIDATION
6. 无 pressure action：按 check_interval 评估 opportunistic candidate
7. opportunistic plan 可执行：PLAN + RESERVE，返回 KV_CONSOLIDATION
8. shadow mode 只记 candidate/reject reason，不 RESERVE
9. 最后才进入 _schedule_ls_decode() 并 commit_iteration_master_plan()
```

这条顺序保留当前“admission 优先于 decode”的语义，也保证 maintenance 返回之前没有修改本轮 master/pending plan。`_schedule_ls_decode_admission()` 的 no-fit 分支必须已经完整回滚 block/group 临时状态；否则不能继续做 pressure simulation。

每个 `schedule()` 最多返回一个 action。单个 engine 同时最多有一个 consolidation transaction，第一版也不并行迁移多个 DP。这样 4 个 DP 上所有 workers 都能维持相同的 maintenance/forward RPC 顺序。

### 4.7 多轮 `8→4→1` 不是一个大 transaction

一次 event 仍只完整清空一个 source rank。连续 scale-down 用 scheduler-owned intent 编排：

```text
KVConsolidationIntent
  intent_id
  trigger = PRESSURE | OPPORTUNISTIC
  group_id / dp_idx
  beneficiary_batch_id          # opportunistic 时为空
  target_kv_dop
  initial_placement_generation
  planned_source_order          # 仅作 deterministic hint
  committed_sources
  predicted_total_bytes / pause_ms
  state = ACTIVE | SATISFIED | CANCELLED | FAILED
```

创建 intent 前，shadow allocator 必须证明从当前 placement 到 `target_kv_dop` 的每一跳都可行，并证明总迁移 bytes、预计总 pause 和连续 maintenance step 数没有超限。执行时不长期持有后续 hops 的 destination blocks；每完成一跳就提交一个独立 transaction，并从新 ACTIVE placement 重新规划下一跳。`planned_source_order` 不能绕过新的 capacity/master/generation 校验。

Pressure intent 可以在相邻 engine steps 连续返回 maintenance action，直到 beneficiary batch 能 admission；在此期间不插入已经会改变 placement 的普通 Decode plan。若 batch 被取消、group finish/merge、master demand 上升，或任一后续 hop 不再可行，则取消 intent；已经成功 commit 的前序 scale-down 保留，不做跨 transaction 反向 rollback。为了避免“迁一半仍不能 admission”，如果完整 chain 在创建时就超过硬预算，第一跳也不执行。

Opportunistic intent 默认每次只执行一个 hop，下一 hop 重新经过 stable/cooldown/payback；测试环境可以放开该限制验证 `8→4→1`。任何路径都不得把多个 source 合成一个可部分 commit 的大 transaction。

## 5. Placement 目标不能只优化 group `kv_dop`

### 5.1 Remote edge 才是直接的 SP 通信指标

下面两个 placement 都可能有 `kv_dop=2`：

```text
Placement A:
  每条 sequence 的 KV 都 striped 到 rank 0 和 rank 1

Placement B:
  一半 sequences 的全部 KV 在 rank 0
  另一半 sequences 的全部 KV 在 rank 1
  每条 sequence 的 master 与 owner 对齐
```

Placement A 中每条 request 通常仍有 remote KV edge。Placement B 可以让所有 requests 走 local Attention，并在整个 DP 不存在其他 remote relation 时完全跳过 SP all-to-all。

因此 planner 的优化顺序建议为：

1. 最大化可释放 ranks 数量（等价于在约束内最小化目标 `kv_dop`）；
2. 在目标 ranks 内优先 whole-sequence packing；
3. 将 sequence 的 owner 与预计 master 对齐；
4. 单条 sequence 无法放入单 rank 时，才对该 sequence 做多 rank striping；
5. 在以上约束下平衡 retained ranks 的 Attention token load。

建议新增指标：

```text
kv_dop_per_group
owners_per_sequence histogram
remote_owner_edges_per_group
remote_sequences_per_group
dp_has_any_remote_kv
```

只用 `kv_dop_per_group` 作为性能代理是不够的。

### 5.2 与 master planner 的配合

Consolidation planner 应先得到下一轮 tentative master chunks，再选择 retained ranks 和 sequence placement。否则可能发生：

```text
KV 被集中到 ranks {0,1}
下一轮 master planner 选择 ranks {2,3}
立刻重新产生 remote edges 和新 KV participants
```

推荐的顺序为：

```text
snapshot committed state
  -> 只计算 tentative source-style master demand，不 commit
  -> source-only evacuation target placement
  -> validate future append/master capacity
  -> execute KV copy
  -> no-fail metadata commit
  -> 下一 engine step 基于新 placement 重新生成 master/pending-token plan
```

MVP 只 evacuation 当前 passive、非 pending-target 的 rank，并要求 retained ranks 能覆盖下一轮全部 masters。不能先调用现有 `commit_iteration_master_plan()` 再做迁移，因为当前 scheduler 在返回 `ScheduleResult` 前已经修改 pending/master state，无法为 maintenance failure 保留稳定快照。

## 6. 数据迁移与事务设计

### 6.1 现有 migration API 不能直接复用

NanoDeploy 已有的 `Sequence::migrate()`、`BlockContextSlot::MIGRATE` 和 `CacheContext.migrate()` 面向跨 engine 的 Prefill→Decode whole-sequence migration：

- `Sequence::migrate()` 将整个 ACTIVE context 移入 MIGRATE slot；
- worker migration 根据旧/新 `block_location` 做一一 zip；
- endpoint 以 remote engine id 组织；
- 完成后释放整个 MIGRATE context；
- Dummy Prefill 路径明确跳过 `executor.migrate()`。

Live Decode consolidation 需要的是同一个 engine、同一个 DP 内的 partial rank evacuation，并且 sequence 必须保持 RUNNING。因此不能直接调用现有 API。

可以复用的只有底层思想：

- scheduler 生成 block/token range copy assignments；
- workers 按统一计划搬运所有 layers 的 KV；
- copy 完成后才切换 block tables。

不建议把 live relayout 硬塞进 `BlockContextSlot::MIGRATE`，应增加独立 transaction 类型，避免破坏现有跨 engine 生命周期。

### 6.2 建议的数据结构

```text
KVTokenRangeMove
  seq_id
  src_sp_rank
  src_block_id
  src_token_offset
  dst_sp_rank
  dst_block_id
  dst_token_offset
  num_tokens

KVConsolidationPlan
  transaction_id
  state_generation
  dp_idx
  group_id
  source_rank
  retained_ranks
  old_kv_dop
  target_kv_dop
  moves
  reserved_destination_blocks
  source_blocks_to_release
  prebuilt_affected_sequence_contexts
  new_num_dispatched_tokens
  new_sp_block_tables
  new_pending_targets
  migration_chunk_tokens
  predicted_before_itl_ms
  predicted_after_itl_ms
  estimated_migration_bytes
```

`state_generation` 用于防止 plan 生成后 group finish、merge、preempt 或 admission 导致 stale commit。

### 6.3 推荐事务

```text
PLAN
  读取 ACTIVE placement
  只统计 committed KV，不把 pending token 当成历史 KV
  选择一个 passive、非 pending-target source rank
  只为 source 上的 committed ranges 生成 destination 和 copy ranges
  预构造受影响 sequences 的目标 metadata

RESERVE
  在 destination BlockManagers 预留 tail 以外所需的新 block IDs
  预留 commit/abort 所需的 metadata 和容器容量
  不修改 ACTIVE block tables
  不释放 source blocks

EXECUTE
  作为独占 maintenance step 在 Decode iteration boundary 执行
  对所有 KV components、layers 和 token ranges 执行 GPU copy
  source block 保持原样，等待所有 workers 报告成功

COMMIT
  校验 state_generation
  以预构造对象替换 num_dispatched_tokens、sp_block_table、block_location
  保持或重新安装 pending token target 和 append reservation
  释放 source_blocks_to_release
  从 group.allocated_attention_ranks 删除已清空 ranks
  rebuild_decode_role_counters()

ABORT
  ACTIVE metadata 保持不变
  释放 transaction-reserved destination blocks
  source blocks 保持有效
```

不能采用“先改 block table，再异步 copy”的顺序；任何 copy/RPC 失败都会让下一轮 Attention 读取未完成的数据。

`COMMIT` 不能再调用可能因为容量、分配或容器扩容而失败的普通 planning API。当前 `BlockManager` 的 `std::list`/`std::unordered_set` 更新和部分 context vector 更新仍可能分配内存；Phase 1 必须将所需对象全部预构造，并把 free-ID/ownership 更新改成预留容量后无分配的操作（更稳妥的是固定容量 vector stack 或 intrusive free list）。所有可预见失败必须发生在 `COMMIT` 前。

当前手动事务已经遵循上述可见性顺序：PLAN/RESERVE 不修改 ACTIVE metadata，copy 时 source 仍有效；commit 前按 group sequence IDs、allocation 和完整 ACTIVE context snapshot 拒绝 stale plan；commit 使用预构造 `BlockContext` swap，随后才回收 source。CPU fake 的 worker completion error 测试还刻意在目标物理 range 已写入、且所有 worker 调用已知结束后抛错，验证 abort 后旧 metadata、source KV 和 free-block accounting 均可继续 Decode；它不证明 Ray/NCCL 未知完成状态可恢复。

但当前实现还不等于严格的 no-fail commit：它使用 snapshot comparison 代替显式 `state_generation`，source 回收仍经过 `std::list`/`unordered_set`，极端 host allocation failure 只能按 engine-fatal 处理；同时 pending target 必须已在 retained rank，尚不支持事务内重指派。这些限制必须在自动 pressure execute 之前补齐或明确接受为 fail-stop 边界。

### 6.4 MVP 采用 LoongServe 式 source-only evacuation

MVP 不为整个 retained placement 建第二份副本，而是：

1. 每次选择一个 source rank；
2. 对 source 上每条 sequence 的 committed KV，优先追加到该 sequence 的下一轮 master/已有 owner；
3. 一个 destination 放不下时才拆分到多个 retained ranks；
4. 不复制 retained rank 之间已经正确的 KV；
5. source 全部复制成功后一次 commit 并释放该 rank。

这与 LoongServe 开源实现的基本形状一致：manager 选择 KV 使用量最小的 Decode instance，把该 instance 的请求 KV 分配到其他 instances，worker 按 `max_mig_len` 分片，完成后才把空 source 从 scale-down batch 中移除。NanoDeploy 应复用这个“清空 source”的方向，而不是复用它的非事务式 metadata 更新顺序。

完整 placement staging 不适合作为生产 MVP：它会重复复制 retained KV，迁移字节更多，且逐 rank 压缩时中间态可能出现“稳态目标放得下，但当前 placement + 完整目标副本放不下”的死角。source-only 只需要为 source 数据找空间，正好与“一次释放一个 rank”的目标一致。

### 6.5 Block tail 与 pending frontier 的 NanoDeploy 适配

LoongServe 使用 token-level allocator；NanoDeploy 使用 64-token block，并把 pending input token 计入 `num_dispatched_tokens`。因此 destination 写入位置必须按：

```text
dst_logical_start = committed_context_len(ACTIVE, dst_sp_rank)
```

而不是现有的 `num_dispatched_tokens[dst_sp_rank]`。后者可能包含尚未计算 KV 的 pending token。

具体规则：

- destination 已有 committed prefix 不原地覆盖；
- 可以从 committed tail 开始写，覆盖 pending reservation 对应但尚无有效 KV 的物理槽；
- RESERVE 必须按“原 committed + moved committed + pending + 下一 sampled token/headroom”准备 block table；
- commit 后 pending token 仍位于新的 committed frontier；
- abort 时 tail 中可能留下不可达字节，但 ACTIVE committed 长度未增加，下一次 pending/append 会覆盖它们；
- source KV 和 source block table 在 commit 前保持完整，因此 copy error 不会破坏旧逻辑 placement。

这要求 planner 能生成 block 内 token-range scatter，而不能只做 block-id 对 block-id 的 zip copy。

### 6.6 GPU copy backend：复用 `attn_sp_group` 的 NCCL P2P

目标拓扑中每个 8-SP allocation domain 共置于同一节点。MVP 直接选择：

```text
已有 per-DP attn_sp_group / ProcessGroupNCCL
source gather -> fixed scratch -> dist.isend/irecv -> destination scratch -> scatter
```

不在第一版扩展 DLSLIME/RDMA、复制 LoongServe 的 `rnccl` binding 或新建重复 NCCL communicator。当前 LS Attention payload 使用 DLSlime `hao_basic`，而 `attn_sp_group` 已具有目标 DP 内 8 个 SP ranks 的正确 membership；maintenance-only step 又保证不与普通 Decode 重叠，因此复用它更符合项目实际。engine 仍必须让全部 32 workers 进入同一个 maintenance action，只有受影响 DP 执行 P2P，其余 workers 等待，保证任何 rank 都不会提前进入下一轮 EP32 collective。

第一版实现按 `(src_sp_rank, dst_sp_rank)` 和物理 range 固定排序，每个 peer/chunk 顺序执行 `isend` 或 `irecv` 并等待 `Work` 完成。这样容易验证配对和 scratch 复用；`batch_isend_irecv` 只作为 correctness 稳定后的 launch-overhead 优化，不改变 plan/transaction 语义。

scratch buffer 在 engine 初始化时固定分配，并在计算 KV cache block 数之前从可用显存扣除。`migration_chunk_tokens` 只控制 gather/send/scatter 的峰值内存；一个 source 仍必须在一个 maintenance event 中完成。source blocks 或预计 pause 超过上限时跳过 candidate，不能留下长期半迁移状态。

backend 必须：

- 对 `kv_count × num_hidden_layers` 的所有 slices 复制；
- 支持 block 内 token range，而不只支持整 block；
- 用固定 transaction、sequence、layer、component、chunk 顺序收发，避免 NCCL P2P 配对不一致；
- 不与 EP32 collective 并发交错；
- 记录 gather、send/recv、scatter、sync 的实际 bytes 和 wall-clock latency。

### 6.7 可恢复失败与 engine-fatal 失败

事务不能承诺任意 GPU/RPC 故障后继续 Decode：

- PLAN、RESERVE 和“任何 NCCL P2P 尚未发出”的 worker preflight 失败可以 ABORT；旧 metadata/source KV 有效，可以继续 Decode；
- 理论上，COPY 已开始后只有所有 worker futures 都确定结束、所有 CUDA work 都确定完成、且 backend 明确证明 communicator 健康的结构化错误，才可能 ABORT 后继续 Decode；第一版不实现这类健康证明，任何 COPY-stage exception 都直接 engine-fatal；
- Ray timeout、actor 退出、future 完成状态未知、CUDA context 错误、NCCL async error/hang 或 communicator health 未知，都必须按 engine-fatal 处理，不能先释放 reservation 再恢复业务；
- COMMIT 开始后不允许返回业务失败。若 source block 的 post-commit 回收异常，应 retry、暂时记为 leak 或使 engine fail-stop，不能声称回到旧 placement。

因此 worker RPC 应拆成可判定的阶段：

```text
PREFLIGHT: 全 workers 校验 plan、range、scratch、peer 顺序；不发 NCCL
COPY:      发 P2P、scatter，并在返回前完成本 rank CUDA stream synchronize
RESULT:    driver 等到全部 actors 的结构化结果后再决定 COMMIT/ABORT
```

当前 `execute_ls_kv_scale_down()` 对 `copy_kv_ranges_p2p()` 的任意异常都会调用 `abort_ls_kv_scale_down()`；这对 CPU fake 和“全部 futures 已知完成”的测试成立，但对 Ray timeout/actor death/NCCL unknown-completion 不够安全。接自动 execute 前必须把异常分类和 engine fatal state 补齐；在此之前不能给 RPC 设置超时后声称可继续服务。

第一版明确选择更窄但可实现的恢复边界：只有 PLAN、RESERVE、generation recheck 和 no-NCCL PREFLIGHT 失败是 recoverable；首个 P2P 发出后出现的任意异常都停止 engine、终止/重建 actors，不再尝试发下一轮 collective。COPY 全成功且同步完成后的 metadata stale 在正常隔离下不应发生；若发生，按 scheduler invariant violation 和 engine-fatal 处理，而不是静默 ABORT 后继续。

如果以后需要 actor-failure recovery，必须另做 checkpoint/restart 或 communicator rebuild，不能把它隐含在本事务设计中。

## 7. Scheduler 接入方案

### 7.1 第一版范围

建议 MVP 只支持：

```text
同一个 engine
同一个 DP
同一个 running group 内
一次 evacuation 一个 source rank
source 优先为 passive、低 KV、非 pending-target rank
不跨 DP
不跨 group
不在 consolidation 中自动 merge groups
```

这个范围已经能解决“一个 group 历史扩到 8 后不能下降”的核心问题，并显著减少事务复杂度。

如果另一个 group 的 ranks 才有足够 destination capacity，MVP 直接报告 `no_group_local_capacity`。跨 group merge + consolidation 放到后续阶段。

### 7.2 Source 与 destination 选择

Source ranks 使用稳定排序：

```text
(is_current_master,
 is_pending_target,
 used_committed_blocks,
 sp_rank)
```

即优先选择：

- 非 master；
- 没有 pending token；
- committed blocks 最少；
- rank id 更小/更大按固定规则打破平局。

Retained/destination ranks 应综合：

```text
下一轮是否为 master
tail + free block capacity
当前 group KV ownership
预计 Attention critical-path load
NUMA/NVLink locality（当前 8 ranks 同节点时作为次级 key）
```

不能简单选择“当前 KV 最满”的 rank，因为 full-mesh 下过度集中可能增加 Attention latency。

### 7.3 Engine step 语义

MVP 将 consolidation 作为独占的内部 maintenance action，不与同一个 engine step 的普通 Decode 拼接。当前 `ScheduleResult` 只有 `is_prefill`，而 `False` 同时会被 `LLMEngine.step()`、worker serialization 和 metrics 理解为 Decode，因此不能用空 `dp_seqs + is_prefill=False` 伪装 maintenance。需要增加 C++ enum 并通过 pybind 暴露：

```text
ScheduleAction = ADMISSION | DECODE | KV_CONSOLIDATION

ScheduleResult
  action
  is_prefill                   # 兼容字段，严格等于 action == ADMISSION
  kv_consolidation_plan        # 仅 maintenance 非空，engine-local opaque shared_ptr
  dp_seqs/dp_sp_seqs/...       # maintenance 时为空，不生成 fake decode batch
```

`kv_consolidation_plan` 同时携带 worker 可见 moves 和 scheduler commit/abort 所需的 reservation handle。它只在 driver 进程内存活，不经 Ray 序列化；worker 只接收 immutable physical move DTO。现有手动 API 应拆成：

```text
execute_planned_ls_kv_scale_down(scheduler, executor, plan)
execute_ls_kv_scale_down(group_id, source_rank) = manual plan + 上述公共执行函数
```

自动 action 必须执行 scheduler 已经 RESERVE 的 plan，不能再按 `group_id/source_rank` 重新 plan 一次。

`KV_CONSOLIDATION` step 的顺序为：

```text
scheduler PLAN + RESERVE（不 commit iteration master plan）
  -> LLMEngine/Executor 向全部 workers 下发 maintenance RPC
  -> scheduler COMMIT 或 ABORT
  -> 本 step 不生成 token
  -> 下一 step 从新 ACTIVE placement 重新 schedule Decode/admission
```

当前 `_schedule_ls_decode()` 在返回前会调用 `commit_iteration_master_plan()`，因此 consolidation branch 必须在该 commit 之前返回，不能复用“先生成正常 Decode plan，再中途插 maintenance”的路径。

`LLMEngine.step()` 在 `scheduler.schedule()` 后必须先按 `action` 分支。maintenance 分支不能构造 `dp_sp_tp_seqs`、读取普通 Decode 通信矩阵、调用 `executor.run()` 或 `scheduler.postprocess()`；它只调用 planned coordinator，记录结果，然后返回零 token 的 maintenance result。

### 7.4 Python 返回值与 metrics 兼容

当前 `step()` 返回五元组，`generate()` 用 `num_tokens > 0` 表示 Prefill、否则表示 Decode。maintenance 返回 `0` 会被误记成一次零吞吐 Decode，而且当前逐 step 计时会把 migration pause 从下一 token 的 ITL 分母中漏掉。建议引入：

```python
@dataclass
class EngineStepResult:
    outputs: list
    num_tokens: int
    batch_size: int
    schedule_latency_ms: float
    post_schedule_latency_ms: float
    action: ScheduleAction
    maintenance_latency_ms: float = 0.0
    maintenance_transaction_id: int | None = None

    # 过渡期 __iter__ 只 yield 原五个字段，保留 examples 的五元解包
```

内部 `generate()` 和 serving loop 必须读取 `result.action`：maintenance 不增加 Prefill/Decode step count 和 token throughput sample，但把 wall time 累加到 `pending_maintenance_stall_ms`，并计入下一个真实 Decode token 的 observed ITL/E2E。单独输出：

```text
kv_consolidation_plan_ms
kv_consolidation_copy_ms
kv_consolidation_commit_ms
kv_consolidation_stall_ms
kv_consolidation_bytes / chunks / released_ranks
kv_consolidation_trigger / reject_reason / beneficiary_batch_id
```

第一版 stop-the-world。用户看到的 token gap 必须包含 maintenance pause；内部拆分指标只用于归因，不能从 benchmark 的端到端结果中扣除。

### 7.5 事务状态机与 scheduler 隔离

```text
IDLE
  -> PLANNED_RESERVED
  -> EXECUTING
  -> COMMITTED -> IDLE
  -> ABORTED   -> IDLE          # 仅明确可恢复失败
  -> ENGINE_FATAL              # 完成状态或 communicator health 未知
```

`PLANNED_RESERVED/EXECUTING` 期间，`schedule()`、`postprocess()`、admit、preempt、merge、finish/free 以及第二个 consolidation 都必须拒绝修改相关 scheduler state。正常 engine 是单线程调用链，但仍应通过 transaction state/generation 做硬检查，不能只依赖调用约定。

worker 侧的全局 ordering 依赖以下事实：前一轮 `executor.run()` 已经 `ray.get` 全部 actors；maintenance RPC 下发给全部 32 actors；参与 copy 的 actors 在返回前完成 P2P、scatter 和 CUDA stream synchronize；driver 等到所有 actor 返回后才 commit 并发起下一轮 `run()`。非目标 DP 不执行 P2P，但必须进入并返回同一个 RPC。这样不需要额外创建全局 NCCL barrier，也不会让部分 rank 提前进入下一次 EP32 collective。

### 7.6 自动路径的伪代码

```python
def step() -> EngineStepResult:
    sch = scheduler.schedule()

    if sch.action == KV_CONSOLIDATION:
        try:
            result = execute_planned_ls_kv_scale_down(
                scheduler, executor, sch.kv_consolidation_plan
            )
        except RecoverableKVConsolidationError:
            # coordinator 已在可证明安全的边界完成 ABORT
            return maintenance_result(action=KV_CONSOLIDATION, failed=True)
        except BaseException:
            engine.mark_fatal()
            raise
        return maintenance_result(result)

    if sch.action == ADMISSION:
        return run_existing_admission_path(sch)

    assert sch.action == DECODE
    return run_existing_decode_and_postprocess_path(sch)
```

recoverable maintenance failure 本 step 不生成 token，下一 step 可以重新 admission/decode；fatal failure 后 `is_finished()`、`step()` 和 `add_request()` 都不得继续推进该 engine。

## 8. 与 LoongServe 的差异

LoongServe 的 scale-down 有两个主要时机：

1. 真实 Prefill 过程中利用已有 SP KV circulation，主动只在较小目标 group 中保留 KV；
2. 新 Prefill 需要资源时，收益感知地迁移 Decode KV，清空低使用 instance。

NanoDeploy 当前场景跳过真实 Prefill，所以无法获得第一个零额外通信的 proactive retention 机会。我们只能在 Decode 运行期间做 reactive consolidation，成本更高。

其开源 Decode scale-down 数据路径值得采用：

- router 从 Decode batch 中选择 KV 使用量最低的 source instance；
- 比较新 Prefill 的预计收益与迁移 source KV 的成本；
- 把 source 上请求的 KV 分配给其他 instances，source 清空后缩小 batch；
- worker 以 `max_mig_len` 为上限分片；model 将各层 KV gather 到连续 send buffer，经 NCCL P2P 传输后由 receiver scatter。

但它的故障语义不能直接照搬：router 会在等待 worker migration 完成前修改 request/instance metadata，worker 也会在发送完成前修改请求长度和 source allocator 状态，代码没有完整 rollback。NanoDeploy 必须保留 source 和 ACTIVE metadata 直到所有 copy 成功，再走 no-fail commit。

此外：

- LoongServe 开源实现是 dense Llama ESP/TP；
- NanoDeploy 目标是 DeepSeek-V3，Attention `4DP × 8SP` 与 FFN `EP32` 复用相同 32 workers；
- LoongServe 的 logical peer shrink 不能直接解决 NanoDeploy 的 EP32 fixed collective。

因此最终选择是：

| 问题 | LoongServe 做法 | NanoDeploy 选择 |
|---|---|---|
| 搬运粒度 | 清空低使用 source instance | 清空一个 passive SP rank，source-only |
| copy 分片 | `max_mig_len` token chunk | 固定 scratch + token chunk |
| transport | 自定义 `rnccl` 直接调用 NCCL P2P | 复用 `attn_sp_group`，PyTorch `dist.isend/irecv` |
| metadata 顺序 | 迁移前已有局部更新 | copy 全成功后 no-fail commit |
| scale-down 触发 | 新 Prefill 收益驱动 | 先 admission pressure，后 calibrated ITL/payback |
| 物理 topology | elastic instances | EP32/SP mesh 保持固定 |

NanoDeploy 的 opportunistic policy 必须以真实 EP32 ITL 收益而不是只以释放 instance 数量作为验收；pressure path 则以是否真正解除 admission 阻塞为验收。

调研依据：

- LoongServe 论文机制：`/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/paper-tex-src/sections/design.tex` 中的 Elastic Scale-down、Elastic Instance Allocation 和 Elastic Scaling Plan Generation；
- LoongServe 开源调度：`/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py::_minimize_decoding_occupied_instances()`；
- source rank 清空后的 group shrink：同文件 `_scale_down_batch()`；
- LoongServe worker chunk 和请求状态更新：`/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/model_infer/model_rpc.py::exposed_migrate_batch()`；
- LoongServe gather/send/recv/scatter：`/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/models/llama/longserve_model.py::decoding_stage_migration()`；
- NanoDeploy 当前固定 cadence：`csrc/nanodeploy/scheduler/scheduler.cpp::_schedule_ls_decode()`；
- NanoDeploy 当前跨 engine KV copy：`nanodeploy/worker/cache.py::CacheContext.migrate()`。

## 9. 分阶段实施计划

### Phase 0：Telemetry 与 Shadow Planner

不改数据面，不搬 KV。

增加：

- group/sequence owner distribution；
- exact `D_mem` 和 `D_target`；
- 每一步 candidate source/retained ranks；
- destination tail/new-block capacity 是否可行；
- estimated migration blocks/bytes/time；
- predicted before/after ITL；
- payback steps；
- candidate stable steps；
- 单 hop 或有限 chain 后指定 pending batch 是否能 admission；
- candidate 被拒绝的原因。

建议拒绝原因枚举：

```text
no_releasable_rank
compute_dop_too_high
insufficient_destination_capacity
pending_target_conflict
receiver_capacity
source_event_limit
predicted_itl_regression
payback_too_long
cooldown
transaction_inflight
```

Phase 0 的 go/no-go 数据：

- serving 时间中 `D_target < D_kv` 的比例；
- 每个 candidate 可释放 rank 数；
- migration bytes 分布；
- payback steps 分布；
- 有 pending admission pressure 时 candidate 命中率；
- consolidation 后可让整个 DP `use_sp_a2a=False` 的比例。

如果绝大多数 candidate 的 predicted saving 小于 full-mesh 固定成本，或 payback 超过 requests 剩余长度，应停止 opportunistic execute；但仍可继续评估能否通过 pressure-driven evacuation 降低 admission wait/preemption。

### Phase 1：CPU Planner 与 Block Transaction

- 已增加 `LSKVConsolidationPlan`、逐 source-rank target placement 和 physical range generation；
- 已实现 destination tail/new-block reserve、copy error abort、预构造 metadata swap、source release 和 group allocation shrink；
- 已保持 retained-rank pending frontier，并拒绝 active/pending source；事务内 pending 重指派仍待实现；
- 已增加 synthetic cache/block table 的完整 transaction 测试，并在 commit 后继续生成下一轮 Decode；
- 显式 `state_generation`、完全无 host allocation 的 no-fail commit 和 shadow/execute policy 仍待实现。

### Phase 2：真实 Intra-DP GPU Copy

- 增加 executor/worker physical range-copy RPC；（第一版已完成）
- 复用 `attn_sp_group`，在 KV block sizing 前预留固定 scratch；（第一版已完成）
- 使用 `dist.isend/irecv` 做确定序 chunk copy；（CPU mock、双进程 Gloo、双 GPU NCCL 和 1-DP × 8-SP NCCL 8→7 preflight 已完成）
- 搬运全部 layers 和 KV components；（第一版 physical range copy 已完成）
- stop-the-world 手动 transaction；（已完成，scheduler 在 reservation 存续期间拒绝普通 schedule）
- 先提供手动触发，不接自动 policy；（已完成）
- 验证 CPU fake 中“全部 workers 已知结束后 completion error”的 abort；（已完成；不能外推到 Ray timeout/NCCL failure）
- 拆分 no-NCCL preflight、copy 和 structured result，并加入 recoverable/fatal 分类；（待完成）
- actor/CUDA/NCCL fatal error 的 fail-stop 注入；（待完成）
- 1-DP × 8-SP / EP8 GPU preflight；（8→7 correctness/ordering 已完成，连续 8→4→1 待完成）
- 比较迁移前后 synthetic KV 内容并验证 source 不变；（已完成）
- 验证无 block leak、hang、stale block table；（CPU transaction 和 8→7 preflight 已完成，long-run 待完成）

即使 Dummy Prefill 的 prompt KV 数值不具有语言语义，性能实验也必须真实复制对应 bytes，不能只修改 metadata。metadata-only relayout 可以作为 planner test mode，但不能作为 KV Consolidation 性能结果。

### Phase 3：Pressure-driven Execute 与 4-DP / EP32 验证

- 增加 `ScheduleAction`、planned coordinator、`EngineStepResult` 和 engine-fatal state；
- 只在 pending batch 原子 admission 明确 no-fit，且 exact simulation 证明单 hop 或完整有限 chain 后可 admission 时执行；
- 一次 event 最多释放一个 rank，多 hop 由 intent 跨 maintenance steps 编排；
- 启用 source blocks/bytes、pause、destination high watermark 硬限制；
- 记录 migration pause 到真实 ITL/E2E；
- 完成 1-DP × 8-SP 连续 `8→4→1`、intent 取消/重规划和真实 forward correctness；
- 完成 4-DP × 8-SP / EP32 的全 worker ordering 与目标 DP 轮换测试；
- 运行持续请求到达、batch 上升再下降的 long-run；
- 验证 `master_dop`、`kv_dop`、remote edges 和 allocation 都按预期变化；
- 对比 `off`、`shadow`、`execute`。

### Phase 4：Opportunistic Execute

- 基于实测 gather/send/scatter 带宽校准 migration cost；
- 基于 4-DP / EP32 trace 校准 before/after ITL predictor；
- 启用 stable window、cooldown、payback 和 scale-up hysteresis；
- 只有含 pause 的 E2E/ITL 获益后才考虑默认开放。

### Phase 5：可选扩展

- admission-pressure 驱动的跨 group merge + consolidation；
- 并行迁移互不相交的多个 DP（第一版仍全 engine 单 transaction）；
- selective SP peer communication，减少 full-mesh 固定成本；
- KV import 时直接使用 compact target placement，减少后续 migration；
- real Prefill/P-D 场景下的 proactive KV placement。

## 10. 建议配置

第一版建议新增：

```python
ls_kv_consolidation_mode: Literal["off", "shadow", "execute"] = "off"
ls_kv_consolidation_execute_policy: Literal["pressure_only", "pressure_and_opportunistic"] = "pressure_only"
ls_kv_consolidation_candidate_util: float = 0.50
ls_kv_consolidation_target_high_watermark: float = 0.80
ls_kv_consolidation_stable_steps: int = 32
ls_kv_consolidation_cooldown_steps: int = 64
ls_kv_consolidation_check_interval_steps: int = 8
ls_kv_consolidation_max_released_ranks_per_event: int = 1
ls_kv_consolidation_max_released_ranks_per_intent: int = 1
ls_kv_consolidation_max_consecutive_maintenance_steps: int = 1
ls_kv_consolidation_max_source_blocks_per_event: int = 0
ls_kv_consolidation_max_migration_bytes_per_event: int = 0
ls_kv_consolidation_max_pause_ms: float = 0.0
ls_kv_consolidation_max_migration_bytes_per_intent: int = 0
ls_kv_consolidation_max_pause_ms_per_intent: float = 0.0
ls_kv_consolidation_migration_chunk_tokens: int = 0  # 0 不预留 scratch/关闭 physical P2P API
ls_kv_consolidation_rpc_timeout_s: float = 0.0  # 0 表示未校准；触发时进入 fatal，不表示可回滚
ls_kv_consolidation_payback_safety_factor: float = 0.5
```

配置约束：

- 只允许与 `enable_ls_decode_core_scheduler=True` 一起使用；
- 第一版仍要求 Decode-only、Dummy Prefill、centralized、`loop_count=1`；
- execute mode 只对白名单拓扑开放；
- `max_released_ranks_per_event` 第一版必须等于 1；intent 可以包含多 hop，但每 hop 都是独立原子 transaction；
- `max_source_blocks_per_event`、event/intent migration bytes、event/intent pause 和 `max_consecutive_maintenance_steps` 在 execute mode 必须由 profiling 给出正值；`0` 表示尚未校准、禁止自动执行；
- `migration_chunk_tokens=0` 不分配 scratch，physical P2P API 会明确拒绝执行；设为正值后必须在 KV block sizing 前扣除对应显存；
- `rpc_timeout_s` 只负责把 hang 转为 engine-fatal 告警/重启流程；timeout 后绝不能走普通 ABORT 并继续 Decode；
- `candidate_util` 只做快速过滤，不能绕过 exact feasibility/admission simulation；
- `pressure_and_opportunistic` 只有 Phase 4 的 ITL predictor 校准并验收后才能开放；
- shadow mode 不得修改 block、group、master 或 pending state；
- mode 为 `off` 时保持 Core 行为完全不变。

## 11. 不变量

1. Consolidation 只移动 committed KV；pending input token 不得被当成已计算 KV 复制。
2. 对每条 sequence，迁移前后 committed token 总数完全一致。
3. 对每条 sequence、每层、每个 KV component，目标 cache 内容与源逻辑 KV 内容一致。
4. 只复制 source rank 上的 committed ranges；retained ranks 已有 committed KV 不因 consolidation 被重写。
5. 所有 workers 报告 EXECUTE 成功前，ACTIVE metadata 和 source blocks 不变化；destination 未提交 tail 中的字节不可被 Attention 读取。
6. PLAN、RESERVE、generation recheck 或首个 P2P 前的 preflight 失败后，旧 ACTIVE placement 仍可继续 Decode；第一版 COPY-stage exception 不在可恢复集合内。
7. actor/CUDA/NCCL fatal failure 必须 fail-stop，不得声称已 rollback 并继续服务。
8. abort 后所有 transaction-reserved blocks 都被回收；未提交 tail 字节保持逻辑不可达。
9. source blocks 只在 metadata commit 成功后释放。
10. 一个 transaction 期间 group 不得 finish、merge、preempt、admit 或被另一个 transaction 修改。
11. commit 前必须校验 group/state generation；commit 路径不得分配内存、调用可失败 capacity API 或返回业务失败。
12. consolidation 后每个 pending token 有且只有一个合法 target 和 reservation，且 pending 不被计作已迁移 KV。
13. 下一轮 masters 必须属于最终 allocation，并有 append capacity。
14. 被释放 rank 对该 group 的 committed KV、pending token 和 block table 全部为零。
15. 同一 DP 内未合并 groups 的 allocations 继续两两不交。
16. 一个 maintenance event 要么完整清空一个 source rank，要么不改变 ACTIVE placement。
17. migration latency 和 bytes 必须进入日志和端到端性能统计。
18. fixed EP collective ordering 不因 maintenance transaction 发生分叉或死锁。
19. 一个 pressure intent 开始前必须证明完整 chain 可行且在总预算内；每个 hop commit 后必须从新 ACTIVE placement 重新验证。
20. maintenance step 不得计作 Decode token step，但其 wall time 必须进入下一个真实 token 的 observed ITL 和请求 E2E。
21. timeout、actor death 或 NCCL/CUDA completion unknown 后，engine 不得再接受请求或调用普通 Decode。

## 12. 测试与验收

### 12.1 CPU 单元测试

- 低平均 usage 但因 block tails 无法释放 rank 时拒绝；
- `D_mem=2`、`D_compute=4` 时选择 `D_target=4`；
- target high watermark 不满足时拒绝；
- opportunistic stable steps/cooldown 正确，真实 pressure path 可绕过它们但不能绕过硬限制；
- source 优先选择 passive、低使用 rank；
- pending target conflict 正确规避或重指派；
- destination 已有 partial tail 和 pending reservation 时，从 committed frontier 规划写入并把 pending frontier 正确后移；
- source-only 可行但完整 placement staging 不可行时仍能产生 plan；
- 最终 `D_target` 可行但当前 source evacuation 这一步不可行时拒绝；
- 只改 source 涉及的 sequence/rank，retained committed tables 保持不变；
- destination reserve 部分失败完整 abort；
- abort 后 ACTIVE metadata 不变，未提交 tail 不可达且后续 append 可覆盖；
- state generation 变化时拒绝 stale commit；
- commit 之前完成全部 allocation，commit 路径不触发动态分配/容量失败；
- pressure candidate 只有在单 hop 或完整有限 chain 后 pending batch 能实际 admission 时通过；
- 需要多个 ranks 才能 admission、但完整 chain 超出总 bytes/pause/连续 steps 预算时，第一 hop 也不执行；
- `ScheduleResult.action` 三条路径互斥，maintenance 在 `_schedule_ls_decode()`/`commit_iteration_master_plan()` 之前返回；
- 自动路径消费 `ScheduleResult` 中已 RESERVE 的 plan，不发生二次 planning；
- maintenance 不调用普通 `executor.run()`、`postprocess()`，也不生成 dummy Decode sequences；
- `8→4→1` 每 hop 都完整 commit、重新规划，任一中间 hop 不可行时 intent 取消且已 commit placement 仍有效；
- beneficiary batch 取消、placement generation 变化和 master demand 上升时 intent 正确取消/重规划；
- group allocation 只在 source 真正清空后缩小；
- maintenance step 不污染 Decode throughput sample，stall 被累计到下一个 token ITL；
- shadow mode 状态零变化；
- feature 关闭时现有 LS tests 完全不变。

### 12.2 GPU 正确性测试

当前 `tests/test_ls_kv_scale_down.py` 已覆盖：scheduler 生成真实 plan、RESERVE 期间 ACTIVE/source 不变、同一个 `KVCacheP2PTransport` 执行分 chunk copy、成功后 metadata/source-block/group allocation commit、下一轮 Decode 不再给 source 分配真实任务，以及 CPU fake 中“物理 copy 已完成且所有 worker 调用已知结束，但 completion 报错”时的 abort。另已通过双 GPU NCCL 完整 transaction 和 1-DP × 8-SP NCCL 8→7 preflight；后者可用 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tests/ls_kv_scale_down_nccl_preflight.py --sp-size 8` 复现。以下未勾选场景仍是进入自动 execute 前的完整验收集合。

- 用确定性 pattern 填充每层 KV block；
- 迁移 full block、partial head/tail 和多个 sequences；
- 覆盖 migration bytes 超过 scratch、需要多个 chunks 的 source；
- 覆盖 destination 带 pending token 的 tail overwrite/shift；
- 检查目标 token ranges byte-for-byte 或 dtype 精确一致；
- worker preflight 失败发生在任何 NCCL P2P 之前，并能安全 ABORT 后继续 Decode；
- copy 完成后执行至少三个真实 Decode iterations；
- 验证无 invalid page、CUDA illegal access、block leak 和 collective hang；
- 注入 no-NCCL preflight error，验证 abort 后旧 placement 可继续运行；
- 注入 COPY-stage worker error，即使 source/ACTIVE metadata 尚未修改也验证 engine fail-stop；
- 注入 actor exit、CUDA fatal 或 NCCL timeout，验证 engine fail-stop，不错误恢复 Decode；
- 验证未受影响 workers 等待 maintenance 完成后再以一致顺序进入 EP32。

### 12.3 1-DP × 8-SP / EP8 preflight

至少覆盖：

```text
kv_dop: 8 -> 7 -> 4 -> 1
master_dop: 8 -> 4 -> 1
有 remote KV -> master-local KV
use_sp_a2a: True -> False
```

注意：EP8 只验证 correctness、ordering 和 transaction，不用于得出正式收益。

当前已完成单次 `kv_dop: 8 -> 7`，并确认 1 个 source、1 个 destination 和 6 个 idle workers 都能完成同一 maintenance step 后继续调度。连续 `8 -> 7 -> 4 -> 1`、master_dop 变化和真实 model forward 仍待补测。

连续测试必须由 action/intent 驱动，不能在测试脚本中预先持有 7 个 plans。每个 hop 后检查：source blocks 已释放、retained KV byte-exact、allocation 缩小、下一 source 从最新 placement 重选；到 `kv_dop=1` 还必须满足 `D_compute=1` 且唯一 retained rank 能承接全部 masters。若 `D_compute>1` 或没有 passive source，正确结果是停在对应下界，而不是强行到 1。

### 12.4 真实模型 forward correctness

synthetic pattern 只能证明 range copy，不能证明迁移后的 block table、Attention metadata 和模型 forward 联合正确。验收采用同一模型、同一请求集、固定 seed、greedy decoding 的两次独立运行：

```text
A: consolidation=off，完成真实 Prefill/KV 初始化后 Decode N steps
B: 相同初始条件，Decode 到同一 boundary，执行一个或多个 maintenance hops，再 Decode N steps
```

Dummy Prefill 当前不会生成有语言语义的 prompt KV，因此不能把 dummy prompt + 随机 cache 称为“真实模型 correctness”。候选方式有：

1. 使用项目非 dummy Prefill/P-D 路径真实填充所有 committed KV slots；
2. 增加仅测试使用的、由真实 model runner 执行 prompt forward 并安装合法 block table 的 fixture。

第一版选择第 2 种：LS Decode 目标配置仍是 Dummy Prefill，直接把完整 P/D migration 引入该测试会同时验证另一套生命周期，难以定位 consolidation 问题。fixture 必须复用生产 `BlockManager`、block table 构造和 model runner，只替代请求进入方式；禁止手写与生产不一致的 metadata。完整非 dummy P/D E2E 作为后续独立验收。

不能仅用 debug RPC 写 pattern 后比较 token 输出。至少校验：迁移前 boundary logits/top-k、迁移后连续 3 个以上 step 的 logits（按 dtype/collective 数值误差设 tolerance）、greedy token 序列、每层 KV logical content、finished 状态和无 invalid page/collective hang。token 完全一致但 logits 超 tolerance 仍算失败；因 collective reduction 顺序导致的允许误差必须事先固定阈值，不能事后放宽。

### 12.5 4-DP × 8-SP / EP32 correctness 与 ordering

先做 correctness，再做性能。至少依次让 DP0、DP1、DP2、DP3 成为目标 DP；每次 maintenance RPC 都下发给全部 32 actors，目标 DP 的 8 个 ranks 做 P2P，其余 24 个 actors 进入 no-op 分支并等待 driver fence。参与 copy 的 worker 返回前必须完成 CUDA stream synchronize，driver 收齐 32 个结果后才能发起下一次全 32-rank model forward。

每种目标 DP 验证：

- 前一 Decode/EP32 step 已全部完成，maintenance 与 dispatch/combine 不重叠；
- 非目标 DP 的 block/sequence/group state byte-for-byte/structurally 不变；
- maintenance 后第一个真实 forward 的 32-rank EP dispatch/combine 顺序一致；
- 连续多 hop、随后 scale-up、另一个 DP 再 scale-down 时不 hang；
- target actor、idle actor、driver 三类 timeout/exit 注入都进入 engine-fatal，不恢复 Decode；
- 任一时刻全 engine 只有一个 active transaction。

### 12.6 4-DP × 8-SP / EP32 性能验收

必须报告：

- migration latency 和有效带宽；
- migration bytes / released rank；
- `kv_dop`、owners-per-sequence、remote edges；
- SP Q/O/LSE bytes 和 calls；
- retained rank Attention load；
- EP dispatch/combine 与 expert compute latency；
- consolidation 前后 steady ITL；
- 把 migration pause 计入后的 E2E/ITL；
- admission wait、preemption 和 throughput；
- scale-down 后再次 scale-up 的频率；
- payback prediction 与实际 payback 的误差。

opportunistic 路径只有在正式 workload 的含迁移 ITL/E2E 改善后才能开放。pressure 路径则必须证明 admission wait/preemption/throughput 的收益大于 pause 代价，两者不能混成一个验收结论。

## 13. 主要风险

### 13.1 收益被固定 EP32 掩盖

这是最大风险。SP KV owner 减少不等于 EP32 参与者减少。Phase 0 必须先判断 remote Attention 工作在总 ITL 中占比是否足够高。

### 13.2 KV 体积很大，payback 过长

DeepSeek MLA 每 token KV bytes 约为：

```text
num_hidden_layers
× (kv_lora_rank + qk_rope_head_dim)
× dtype_bytes
```

长 context 的单 rank evacuation 可能需要复制数 GB。不能因为 utilization 低就忽略绝对迁移体积。

### 13.3 Destination capacity 与固定 scratch

source-only evacuation 不需要复制整个 retained placement，但 destination 仍需容纳 source committed tokens、pending/append headroom 和 fixed scratch。scratch 必须在 KV block sizing 前扣除；本次 source 放不下时跳过 candidate，不能退回到先释放 source 或暴露半迁移 metadata 的做法。

### 13.4 Metadata 与物理 cache 不一致

当前 `num_dispatched_tokens`、`sp_block_table`、`block_location`、BlockManager ownership、pending frontier、worker serialization 和 CUDA metadata 都依赖同一个 placement。任何非事务式局部更新都会导致错误 block 访问或静默错误。

### 13.5 Collective ordering

迁移 RPC 不能与某些 ranks 的 Decode/EP collective 交错。第一版应全局同步并牺牲 overlap，先保证 ordering。

### 13.6 故障边界比普通 metadata transaction 更窄

NCCL/CUDA fatal failure 可能同时破坏 communicator 或进程，保留旧 source 并不等于 engine 可继续。实现和测试必须区分可恢复 abort 与 engine-fatal，避免形成“任意 RPC 失败都可回滚”的错误运维预期。

## 14. 最终建议

建议实施，但按以下顺序推进：

```text
先做 telemetry + shadow exact planner
    ↓
实现 source-only CPU reservation / abort / no-fail commit
    ↓
复用 attn_sp_group，实现固定 scratch + dist.isend/irecv，先手动触发
    ↓
补齐 preflight/COPY 故障分类、ScheduleAction 和 planned coordinator
    ↓
以单 source transaction + scheduler intent 扩展到连续 8→4→1
    ↓
使用真实 Prefill/KV fixture 完成真实 Decode forward correctness
    ↓
先在 4DP/EP32 启用 exact admission-pressure path
    ↓
校准 ITL/payback 后再决定是否启用 opportunistic path
```

不建议直接实现“usage 低于 X% 就迁移”。推荐把用户提出的长期低使用率改写为：

> 每个 maintenance event 只清空一个 passive、非 pending-target source rank，并只复制该 source 的 committed KV；当指定 pending batch 因 rank 不足无法 admission、且 exact simulation 证明单 hop 或预算内完整 chain 能解除阻塞时优先执行。多 hop 由 scheduler intent 编排，每 hop 独立 commit 并从最新 placement 重规划。没有 admission pressure 时，只有 target placement 持续稳定、满足短期 master/append capacity，且校准后的收益能覆盖真实 pack/send/scatter pause，才做 opportunistic evacuation。

这一定义保留了“长期低 usage”想解决的问题，同时与 LoongServe 的 source evacuation 数据路径一致，并避开完整 staging、中间态 capacity、过强 rollback 承诺、compute demand、fixed EP32 成本和反复扩缩容带来的错误设计。
