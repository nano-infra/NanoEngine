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

1. 当前 KV placement 可以在考虑 block fragmentation、pending token 和 append headroom 后，完整压缩到更少 ranks；
2. 压缩后至少能彻底清空一个 rank，而不只是把 KV 分布变得更均匀；
3. 目标 rank 数能够覆盖当前及短期内的 Decode master demand，避免下一轮马上重新扩到 8；
4. 迁移造成的暂停和传输成本能在 group 的剩余 Decode 生命周期内摊销，或者释放 rank 能立即解除 admission/resource pressure；
5. 数据面真实复制 KV，metadata 与物理 block 所有权以事务方式一起切换。

推荐将该能力命名为：

```text
LS-Decode KV Consolidation
```

它是现有 `LS-Decode-Core` 的扩展，不应声称等同于完整 LoongServe。

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
- destination rank 可能需要满足 `max_num_seqs`、`max_num_recv_seqs` 和 CUDA Graph metadata capacity。

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
- destination 是否有足够 staging/append capacity；
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

先计算：

```text
D_mem = exact relocation planner 找到的最小可行 rank 数
D_compute = 当前及短期 master demand 需要的 rank 数
D_target = max(D_mem, D_compute)
```

Exact planner 必须包含：

- committed KV blocks；
- destination tail repacking 或 staging block 开销；
- 当前 pending input token 的重新指派；
- 下一 sampled token reservation；
- `reserved_blocks_per_req`；
- per-rank block、sequence、receiver 和 graph capacity；
- consolidation 后的安全 high watermark。

第一版建议只在：

```text
D_target < D_kv
```

时产生 candidate，并一次最多释放一个 rank。逐 rank evacuation 更容易控制 pause 时间、回滚和抖动。

### 4.3 稳定性条件

推荐维护 group-level 状态：

```text
candidate_target_dop
candidate_stable_steps
last_scale_up_step
last_consolidation_step
consolidation_inflight
```

只有同一个 `D_target` 连续保持若干 Decode iterations，且最近没有 scale-up，才允许执行。

建议配置先以 shadow mode 收集数据，不直接给生产默认值：

```python
ls_kv_consolidation_mode = "off"        # off | shadow | execute
ls_kv_consolidation_candidate_util = 0.50
ls_kv_consolidation_target_high_watermark = 0.80
ls_kv_consolidation_stable_steps = 32
ls_kv_consolidation_cooldown_steps = 64
ls_kv_consolidation_max_released_ranks_per_event = 1
ls_kv_consolidation_max_blocks_per_event = 0  # 0 表示仅由 cost policy 控制
```

这些数值只能作为 shadow bring-up 起点，正式值必须来自 4-DP × 8-SP / EP32 trace 和 migration bandwidth profiling。

### 4.4 Cost/Benefit 条件

对 candidate plan 估算：

```text
MigrationCostMs = bytes_to_copy / measured_effective_bandwidth
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

当有 pending batch 且没有 unallocated rank 时，可以引入 resource-pressure override：如果释放一个 rank 能避免 admission delay、group merge 或 preemption，可以允许 group 自身 ITL 略有回退，但仍必须限制迁移 bytes 和目标 rank high watermark。

### 4.5 推荐的触发时机

不建议每轮无条件扫描和迁移。建议在以下事件发生后评估 candidate：

1. `postprocess()` 发现 requests finish，group KV 明显下降；
2. `master_dop` 从高值降到更低值并保持稳定；
3. 新 batch admission 因没有 unallocated rank 而等待；
4. 周期性低频检查，用于覆盖没有显式状态变化但已满足 payback 的 group。

事件只负责触发 planner。真实 copy 必须在没有 model forward/EP collective in flight 的 Decode iteration boundary 执行。

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

1. 最小化可释放 ranks 数量；
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
  -> tentative source-style master plan
  -> consolidation target placement
  -> validate future append/master capacity
  -> execute KV copy
  -> atomic metadata commit
  -> install next iteration master/pending-token plan
```

第一版也可以更保守：只 evacuation 当前 passive、非 pending-target 的 rank，并要求 retained ranks 已包含下一轮全部 masters。

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
  source_ranks
  retained_ranks
  old_kv_dop
  target_kv_dop
  moves
  reserved_destination_blocks
  new_num_dispatched_tokens
  new_sp_block_tables
  new_pending_targets
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
  选择 source/retained ranks
  生成目标 block tables 和 copy ranges

RESERVE
  在 destination BlockManagers 预留 staging block IDs
  不修改 ACTIVE block tables
  不释放 source blocks

EXECUTE
  在 Decode iteration boundary 暂停该 transaction 涉及的执行
  对所有 KV components、layers 和 token ranges 执行 GPU copy
  等待所有 copy 完成

COMMIT
  校验 state_generation
  原子替换 num_dispatched_tokens、sp_block_table、block_location
  重新安装 pending token target 和 append reservation
  释放旧 source/staging 不再引用的 blocks
  从 group.allocated_attention_ranks 删除已清空 ranks
  rebuild_decode_role_counters()

ABORT
  ACTIVE metadata 保持不变
  释放 destination staging blocks
  source blocks 保持有效
```

不能采用“先改 block table，再异步 copy”的顺序；任何 copy/RPC 失败都会让下一轮 Attention 读取未完成的数据。

### 6.4 第一版建议使用完整 staging placement

最省流量的 evacuation 需要处理 destination partial tail：填充旧 tail、搬 full blocks、再搬 source tail，事务复杂。

考虑到 proposed trigger 本来就针对长期低 usage，第一版建议使用更保守的完整 staging placement：

1. 为受影响 sequences 在 retained ranks 上建立 compact 新 block tables；
2. 在旧 ACTIVE blocks 仍有效时，把 committed KV 复制到新 blocks；
3. copy 全部成功后一次性切换；
4. 再释放旧 blocks。

优点：

- 事务和 rollback 简单；
- 不存在 in-place overwrite dependency；
- block tables 天然紧凑；
- 容易做 synthetic byte-for-byte correctness test。

代价：

- transaction 期间需要额外 staging capacity；
- retained ranks 的 KV 也可能被重新复制；
- migration bytes 高于最优 source-only evacuation。

因此第一版 exact planner 必须把 temporary double-buffer capacity 作为硬约束。无法 staging 的 candidate 直接跳过，不退化为非事务式搬运。后续再实现增量 tail repack。

### 6.5 GPU copy backend

目标拓扑中每个 8-SP allocation domain 共置于同一节点。可选实现有：

1. 在 `attention_sp_group` 上使用确定顺序的 NCCL P2P/batched send-recv；
2. 为同 engine KV Cache 注册 DLSLIME/RDMA endpoints，扩展现有 assignment copy；
3. 如果 peer access 条件稳定，增加同节点 CUDA peer copy path。

第一版更重视正确性和可控 ordering，不应先追求 overlap。无论选择哪种 backend，都必须：

- 对 `kv_count × num_hidden_layers` 的所有 slices 复制；
- 支持 block 内 token range，而不只支持整 block；
- 所有 workers 以一致的 transaction ordering 进入和退出迁移；
- 不与 EP32 collective 并发交错；
- 记录实际 bytes 和 wall-clock migration latency。

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
staging free capacity
当前 group KV ownership
预计 Attention critical-path load
NUMA/NVLink locality（当前 8 ranks 同节点时作为次级 key）
```

不能简单选择“当前 KV 最满”的 rank，因为 full-mesh 下过度集中可能增加 Attention latency。

### 7.3 Engine step 语义

推荐将 consolidation 作为内部 maintenance transaction，而不是伪装成 Prefill 或普通 Decode：

```text
schedule maintenance candidate
  -> executor.consolidate_kv(plan)
  -> scheduler.commit/abort
  -> 继续生成本轮正常 Decode plan
```

第一版可以 stop-the-world，并把 migration latency 计入当前 request latency。不能把 maintenance 时间从 benchmark ITL 中扣除，否则会高估收益。

如果 engine 当前一步只能执行一个 action，也可以先引入显式：

```text
ScheduleResult.maintenance_kind = KV_CONSOLIDATION
```

该 step 不生成 token，完成后下一 step 再 Decode。无论采用哪种接口，client-visible E2E/ITL 必须包含这段暂停。

## 8. 与 LoongServe 的差异

LoongServe 的 scale-down 有两个主要时机：

1. 真实 Prefill 过程中利用已有 SP KV circulation，主动只在较小目标 group 中保留 KV；
2. 新 Prefill 需要资源时，收益感知地迁移 Decode KV，清空低使用 instance。

NanoDeploy 当前场景跳过真实 Prefill，所以无法获得第一个零额外通信的 proactive retention 机会。我们只能在 Decode 运行期间做 reactive consolidation，成本更高。

此外：

- LoongServe 开源实现是 dense Llama ESP/TP；
- NanoDeploy 目标是 DeepSeek-V3，Attention `4DP × 8SP` 与 FFN `EP32` 复用相同 32 workers；
- LoongServe 的 logical peer shrink 不能直接解决 NanoDeploy 的 EP32 fixed collective。

因此 NanoDeploy 的 policy 必须比 LoongServe 更保守，并以真实 EP32 ITL 收益而不是只以释放 instance 数量作为验收。

调研依据：

- LoongServe 论文机制：`/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/paper-tex-src/sections/design.tex` 中的 Elastic Scale-down、Elastic Instance Allocation 和 Elastic Scaling Plan Generation；
- LoongServe 开源调度：`/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py::_minimize_decoding_occupied_instances()`；
- source rank 清空后的 group shrink：同文件 `_scale_down_batch()`；
- NanoDeploy 当前固定 cadence：`csrc/nanodeploy/scheduler/scheduler.cpp::_schedule_ls_decode()`；
- NanoDeploy 当前跨 engine KV copy：`nanodeploy/worker/cache.py::CacheContext.migrate()`。

## 9. 分阶段实施计划

### Phase 0：Telemetry 与 Shadow Planner

不改数据面，不搬 KV。

增加：

- group/sequence owner distribution；
- exact `D_mem` 和 `D_target`；
- candidate source/retained ranks；
- staging capacity 是否可行；
- estimated migration blocks/bytes/time；
- predicted before/after ITL；
- payback steps；
- candidate stable steps；
- candidate 被拒绝的原因。

建议拒绝原因枚举：

```text
no_releasable_rank
compute_dop_too_high
insufficient_staging_capacity
pending_target_conflict
receiver_capacity
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

如果绝大多数 candidate 的 predicted saving 小于 full-mesh 固定成本，或 payback 超过 requests 剩余长度，应停止 execute 实现，只保留 admission-time placement 优化。

### Phase 1：CPU Planner 与 Block Transaction

- 增加 `KVConsolidationPlan` 和 transaction generation；
- 实现 exact target placement；
- 实现 staging block reserve/abort/commit；
- 实现 pending token 的事务式重新指派；
- 增加 group allocation shrink；
- 使用 synthetic cache/block tables 做 CPU transaction tests；
- feature 默认 `shadow` 或 `off`。

### Phase 2：真实 Intra-DP GPU Copy

- 增加 executor/worker consolidation RPC；
- 搬运全部 layers 和 KV components；
- stop-the-world 同步；
- copy failure 注入和 rollback；
- 1-DP × 8-SP / EP8 GPU preflight；
- 比较迁移前后 synthetic KV 内容；
- 验证无 block leak、hang、stale block table。

即使 Dummy Prefill 的 prompt KV 数值不具有语言语义，性能实验也必须真实复制对应 bytes，不能只修改 metadata。metadata-only relayout 可以作为 planner test mode，但不能作为 KV Consolidation 性能结果。

### Phase 3：Policy Execute 与 4-DP / EP32 验证

- 启用 stable/cooldown/cost policy；
- 一次最多释放一个 rank；
- 记录 migration pause 到真实 ITL/E2E；
- 运行持续请求到达、batch 上升再下降的 long-run；
- 验证 `master_dop`、`kv_dop`、remote edges 和 allocation 都按预期变化；
- 对比 `off`、`shadow`、`execute`。

### Phase 4：可选扩展

- 增量 source-only evacuation 和 tail repack；
- admission-pressure 驱动的跨 group merge + consolidation；
- migration budget/分片，多轮渐进清空一个 rank；
- selective SP peer communication，减少 full-mesh 固定成本；
- KV import 时直接使用 compact target placement，减少后续 migration；
- real Prefill/P-D 场景下的 proactive KV placement。

## 10. 建议配置

第一版建议新增：

```python
ls_kv_consolidation_mode: Literal["off", "shadow", "execute"] = "off"
ls_kv_consolidation_candidate_util: float = 0.50
ls_kv_consolidation_target_high_watermark: float = 0.80
ls_kv_consolidation_stable_steps: int = 32
ls_kv_consolidation_cooldown_steps: int = 64
ls_kv_consolidation_check_interval_steps: int = 8
ls_kv_consolidation_max_released_ranks_per_event: int = 1
ls_kv_consolidation_max_blocks_per_event: int = 0
ls_kv_consolidation_payback_safety_factor: float = 0.5
ls_kv_consolidation_allow_admission_pressure_override: bool = False
```

配置约束：

- 只允许与 `enable_ls_decode_core_scheduler=True` 一起使用；
- 第一版仍要求 Decode-only、Dummy Prefill、centralized、`loop_count=1`；
- execute mode 只对白名单拓扑开放；
- shadow mode 不得修改 block、group、master 或 pending state；
- mode 为 `off` 时保持 Core 行为完全不变。

## 11. 不变量

1. Consolidation 只移动 committed KV；pending input token 不得被当成已计算 KV 复制。
2. 对每条 sequence，迁移前后 committed token 总数完全一致。
3. 对每条 sequence、每层、每个 KV component，目标 cache 内容与源逻辑 KV 内容一致。
4. ACTIVE block tables 在 EXECUTE 成功前不变化。
5. EXECUTE 或 COMMIT 失败后，旧 ACTIVE placement 仍可继续 Decode。
6. destination staging blocks 在 abort 后全部释放。
7. source blocks 只在 metadata commit 成功后释放。
8. 一个 transaction 期间 group 不得 finish、merge、preempt 或被另一个 transaction 修改。
9. commit 必须校验 group/state generation。
10. consolidation 后每个 pending token 有且只有一个合法 target 和 reservation。
11. 下一轮 masters 必须属于最终 allocation，并有 append capacity。
12. 被释放 rank 对该 group 的 committed KV、pending token 和 block table 全部为零。
13. 同一 DP 内未合并 groups 的 allocations 继续两两不交。
14. migration latency 和 bytes 必须进入日志和端到端性能统计。
15. fixed EP collective ordering 不因 maintenance transaction 发生分叉或死锁。

## 12. 测试与验收

### 12.1 CPU 单元测试

- 低平均 usage 但因 block tails 无法释放 rank 时拒绝；
- `D_mem=2`、`D_compute=4` 时选择 `D_target=4`；
- target high watermark 不满足时拒绝；
- stable steps 和 cooldown 正确；
- source 优先选择 passive、低使用 rank；
- pending target conflict 正确规避或重指派；
- staging reserve 部分失败完整 rollback；
- state generation 变化时拒绝 stale commit；
- group allocation 只在 source 真正清空后缩小；
- shadow mode 状态零变化；
- feature 关闭时现有 LS tests 完全不变。

### 12.2 GPU 正确性测试

- 用确定性 pattern 填充每层 KV block；
- 迁移 full block、partial head/tail 和多个 sequences；
- 检查目标 token ranges byte-for-byte 或 dtype 精确一致；
- copy 完成后执行至少一个真实 Decode iteration；
- 验证无 invalid page、CUDA illegal access、block leak 和 collective hang；
- 注入 source/destination RPC failure，验证旧 placement 仍可继续运行。

### 12.3 1-DP × 8-SP / EP8 preflight

至少覆盖：

```text
kv_dop: 8 -> 7 -> 4 -> 1
master_dop: 8 -> 4 -> 1
有 remote KV -> master-local KV
use_sp_a2a: True -> False
```

注意：EP8 只验证 correctness、ordering 和 transaction，不用于得出正式收益。

### 12.4 4-DP × 8-SP / EP32 性能验收

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

功能通过但正式 workload 的含迁移 ITL/E2E 没有改善时，execute mode 不应默认开启。

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

### 13.3 Temporary staging capacity

完整 staging placement 需要额外显存。第一版宁可跳过 candidate，也不能在显存不足时做不可回滚的 in-place rewrite。

### 13.4 Metadata 与物理 cache 不一致

当前 `num_dispatched_tokens`、`sp_block_table`、`block_location`、BlockManager ownership、pending frontier、worker serialization 和 CUDA metadata 都依赖同一个 placement。任何非事务式局部更新都会导致错误 block 访问或静默错误。

### 13.5 Collective ordering

迁移 RPC 不能与某些 ranks 的 Decode/EP collective 交错。第一版应全局同步并牺牲 overlap，先保证 ordering。

## 14. 最终建议

建议实施，但按以下顺序推进：

```text
先做 telemetry + shadow exact planner
    ↓
确认经常存在可释放 ranks，且实际 payback 合理
    ↓
实现 group-local、一次一 rank、完整 staging 的事务式 GPU copy
    ↓
在 1DP/EP8 做 correctness preflight
    ↓
在 4DP/EP32 以含 migration pause 的 ITL/E2E 决定是否启用
```

不建议直接实现“usage 低于 X% 就迁移”。推荐把用户提出的长期低使用率改写为：

> 当同一个 group 的 exact target KV DoP 连续多个 Decode iterations 低于当前 `kv_dop`，目标 placement 能满足短期 master/append capacity，并且预测收益可以覆盖真实 KV migration 成本时，逐 rank 做 KV evacuation；有 admission pressure 时允许受限 override。

这一定义既保留了“长期低 usage”想解决的问题，又避免了 capacity、compute demand、full-mesh 固定成本和反复扩缩容带来的错误决策。
