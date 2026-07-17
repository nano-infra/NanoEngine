# LS-Decode-Core：Decode Logical Batch 原子 Admission 设计

## 1. 文档状态

本文已进入实施状态。CPU 侧 scheduler、SP state transaction、Python 只读观测和单元测试已经按本文语义落地；1-DP × 8-SP GPU 短时 preflight 已通过，4-DP × 8-SP long-run 仍待执行。

目标是修复 LS-Decode-Core 当前 admission 在容量或 receiver metadata 受限时逐步减小候选 `batch_size`、只接纳可行前缀的行为，使 scheduler 已经形成的 logical Decode batch 具有稳定 identity，并以 all-or-nothing 方式进入一个 Decode group。

本文只讨论：

```text
mode="decode"
dummy_prefill=true
enable_ls_decode_core_scheduler=true
scheduler_mode="centralized"
loop_count=1
```

本文中的 admission step 只是 Decode-only 请求进入 running state 前的控制步骤。虽然当前 `ScheduleResult.is_prefill` 和部分函数名仍使用 `prefill`，本文不设计真实 Prefill 计算、Prefill batching、P/D 迁移或 Prefill/Decode 资源竞争。

实现中的关键差异和验证状态见第 19 节。

## 2. 核心结论

第一版采用以下语义：

1. 外部 `LLMEngine.add_request()` 调用边界不是需要永久保持的 batch 边界。
2. scheduler 在 admission step 开始时，从尚未封存的 Decode arrival queue 中按稳定 FIFO 顺序形成最多 `attention_dp` 个 DP-local logical batches。
3. logical batch 一旦形成并取得 `batch_id`，其 sequence 集合和顺序不可再改变。
4. admission planning 必须对 logical batch 的全部 sequences 成功，才能 commit。
5. logical batch 可以完整地创建新 group，也可以完整地并入一个已有 group；不能只接纳前缀。
6. 当前所有 DP 都无法接纳该 batch 时，batch 保持完整并继续等待，不回退到更小 `batch_size`。
7. running group 之间的 merge 继续以完整 group 为单位，不拆 initial batch records。
8. 单 request preemption/readmission 作为 recovery batch 处理，不回溯并暂停其原 admission batch 中仍在运行的其他 requests。
9. seal 在分配 `batch_id` 前执行空系统静态可行性检查；候选若静态不可行可继续缩小，但已经获得 `batch_id` 的 batch 永不拆分。

该语义把“外部请求聚合”“scheduler logical batching”和“runtime group merge”分成三个层次：

```text
外部到达 requests
  -> scheduler 封存 DP-local logical batches
  -> logical batch 原子 admission 到一个 group
  -> running groups 可继续整组 merge
  -> 每轮在 group 内生成 multi-master mini-batches
```

其中只有最后一层 mini-batch 是 iteration-scoped 执行切分，不改变 logical batch 或 group membership。

## 3. 为什么不直接保留 `add_request(list)` 边界

当前正式实验拓扑为：

```text
attention_dp = 4
attention_sp = 8
```

一个 Decode group 只能属于一个 DP，不能跨 DP 使用 SP ranks。外部 benchmark 可能一次把所有 requests 作为一个 Python list 传给 `LLMEngine.add_request()`。如果把该 list 永久视为一个不可拆 batch，就会把所有 requests 强制放到同一个 DP，导致：

- 其他三个 DP 空闲；
- 单 DP metadata/capacity 更容易成为瓶颈；
- 破坏当前 admission 在多个 DP 之间均衡新请求的行为；
- 把调用者为了减少 Python 开销形成的 list 误认为调度语义。

另一方面，在线 long-run benchmark 通常逐条调用 `add_request(seq)`。如果直接保留调用边界，所有 logical batches 都会退化为 singleton，也不能表达 scheduler-visible batching。

因此，本文将 batch identity 放在 scheduler 的 seal 阶段：先根据当前等待请求和 DP 数量形成 logical batches，再保证这些 logical batches 在 placement、等待和 merge 过程中不被拆散。

## 4. 术语与不变量

### 4.1 Unsealed arrival

已经到达 scheduler、但尚未被纳入 logical batch 的 Decode request。

当前 `waiting_migration` 可以继续承载这部分请求，但在 LS-Decode-Core 开启时，它只表示 unsealed arrivals，不再同时表示已经形成 logical batch 的等待请求。

### 4.2 Pending logical batch

scheduler 已封存、已经获得稳定 `batch_id`、尚未成功 admission 的 Decode batch。

建议的数据结构：

```cpp
struct PendingDecodeBatch {
    uint64_t batch_id;
    std::vector<std::shared_ptr<Sequence>> sequences;
    uint64_t enqueue_order;
    uint32_t admission_attempts;
    bool is_recovery_batch;
    std::optional<uint64_t> parent_batch_id;
};
```

其中：

- `sequences` 按 arrival 顺序稳定保存；
- `batch_id` 在 seal 时分配，而不是 admission commit 时分配；
- `enqueue_order` 用于确定等待和日志顺序；
- `admission_attempts` 只用于观测，不能触发自动拆 batch；
- `is_recovery_batch` 区分新 arrival 和 preempted request；
- `parent_batch_id` 仅用于追踪 lineage，不恢复原 batch 的原子运行关系。

### 4.3 Initial batch placement

logical batch 成功 admission 后生成的不可变 placement record：

```text
InitialBatchPlacement
  batch_id
  admission_order
  group_id
  sequence_ids
  initial_kv_dop
  initial_kv_ranks
  prompt_kv_tokens
  provisional_pending_targets
  is_recovery_batch
  parent_batch_id
```

现有 `InitialBatchPlacement` 可以继续扩展：

- `batch_id` 改为继承 pending logical batch 的稳定 ID，不再在 commit 时重新生成；
- `admission_order` 在成功 commit 时单调分配，用于在 bounded bypass 后仍稳定恢复 group execution order；
- recovery record 保存 `is_recovery_batch` 和 `parent_batch_id`，用于区分同一 sequence 的 initial admission 和重新接纳历史。

### 4.4 Group

一个 DP 内可共同执行和共享 allocation 的完整 sequence 集合。一个 group 可以包含多个 initial logical batches。

### 4.5 Mini-batch

每个 Decode iteration 中，source-greedy planner 分给某个 master 的 request slice。mini-batch 不是持久 batch，不需要保持跨 iteration identity。

### 4.6 必须始终成立的不变量

```text
I1. 一个 pending logical batch 的 sequence_ids 创建后不再增删或重排。
I2. 一个 pending logical batch 在任意时刻只能处于 QUEUED、COMMITTING、ADMITTED 之一。
I3. admission 成功时，该 batch 的全部 sequences 在同一个 DP、同一个 group 中。
I4. admission 失败时，该 batch 的所有 sequences 仍为 WAITING，且没有 ACTIVE KV blocks、running entry 或 group ownership。
I5. 不存在同一 batch 一部分 ADMITTED、另一部分 QUEUED 的状态。
I6. batch merge 不重写任何已有 InitialBatchPlacement。
I7. iteration mini-batch 划分不能修改 batch/group membership。
I8. preemption 不造成同一 live sequence 在 pending batch 和 running group 中重复出现。
```

## 5. 当前行为与目标行为

### 5.1 当前行为

当前 `_schedule_ls_decode_admission()` 对每个 DP：

1. 根据剩余 waiting requests 计算 `balanced_batch`；
2. 从 `max_batch` 开始构造队首前缀；
3. placement 失败时执行 `batch_size--`；
4. 第一个可行前缀立即 commit；
5. 剩余 requests 留在扁平队列中，之后可能形成其他 batch/group。

因此一个 scheduler 原本尝试共同接纳的请求集合可能被持久拆散。

### 5.2 目标行为

目标流程为：

```text
unsealed arrivals
  -> seal 为 B0, B1, ...
  -> 对整个 Bi 生成 placement plan
       -> standalone group 可行：整个 Bi commit
       -> merge candidate 可行：整个 Bi commit 到目标 group
       -> 全部不可行：Bi 原样等待
```

不再存在：

```cpp
for (int batch_size = max_batch; batch_size >= 1; --batch_size)
```

或任何等价的 admission prefix fallback。

## 6. Logical Batch Seal 算法

### 6.1 Seal 时机

在每次 `_schedule_ls_decode_admission()` 开始、任何 placement planning 之前执行：

```cpp
_seal_ls_decode_arrivals();
```

seal 只读取当前已经到达的 unsealed arrivals。seal 开始之后新到达的请求留到下一个 admission step，避免一个 logical batch 在 planning 期间继续增长。

### 6.2 每轮 seal 上限

第一版保持当前“一个 admission step 每个 DP 最多接纳一个新 batch”的执行规模：

```text
seal_limit = attention_dp * max_num_seqs
n = min(unsealed_arrivals.size, seal_limit)
num_batches = min(attention_dp, n)
```

若 `n == 0`，不产生新 batch。

将前 `n` 个 requests 按稳定 FIFO 顺序均衡生成 `num_batches` 个候选：

```text
base = floor(n / num_batches)
remainder = n mod num_batches

size(B_i) = base + 1, i < remainder
            base,     otherwise
```

由 `seal_limit` 可知：

```text
1 <= size(B_i) <= max_num_seqs
```

这里的切分是 logical batch 的创建动作，不是对一个已经存在的 batch 进行拆分。每个候选在分配 `batch_id` 前还要验证它能否在“没有真实 running requests、只保留 scheduler-owned dummy sequence 常驻 block”的 DP 上完成 initial placement。实现会在 scheduler 初始化后记录每个 rank 的真实 baseline free blocks，不能直接使用配置的 block 总数，否则会高估每 rank 一个永久不可用的 dummy block。若候选静态不可行，则缩小尚未封存的 FIFO 候选，直到静态可行；尾部 requests 仍是 unsealed arrivals。若连 singleton 都静态不可行，scheduler 立即抛出明确错误，不创建一个永远无法完成的 pending batch。

该检查只读取配置上限、prompt 长度、receiver/master metadata 上限和 reservation，不读取实时 free blocks，因此 transient pressure 不会改变 batch 边界。它与旧 admission prefix fallback 的区别是：缩小发生在 batch identity 创建前；一旦 `batch_id` 分配，后续 no-fit 只等待，不再缩小。

### 6.3 稳定性

- batches 按最早 sequence arrival 顺序排列；
- batch 内 sequences 保持 FIFO；
- `batch_id` 单调递增；
- 同一组 arrivals 和相同配置必须得到相同切分；
- 不按 prompt length 重排，因为本文不设计 Prefill batching；
- 不因实时 free blocks 改变 batch 边界，capacity 只影响 admission 是否成功。

### 6.4 已有 pending batches 与新 arrivals

已有 pending batches 不与新 arrivals 重新组合。新 arrivals 只形成新的 batch IDs：

```text
pending: B0, B1
new arrivals: r8, r9, r10

下一 step seal 后：
pending: B0, B1, B2={r8,...}
```

禁止把 `B0` 的尾部与新 arrival 合并成一个新 batch。

## 7. Whole-Batch Admission Planning

### 7.1 Read-only planning

对每个 pending batch，planner 必须始终传入完整 `batch.sequences`。

`_plan_ls_initial_placement()` 继续负责：

- 选择 `D_init`；
- 选择 initial ranks；
- 生成 batch-uniform logical prompt KV placement；
- 校验 block capacity；
- 校验 receiver metadata；
- 校验首轮 master metadata/headroom。

planner 不能修改：

- sequence status/context；
- block manager；
- running queue；
- group state；
- pending batch state。

### 7.2 Candidate 类型

一个 pending batch 可以尝试两类候选：

#### Standalone candidate

在某个 DP 的全局未分配 ranks 上创建新 group。

#### Merge candidate

完整并入该 DP 的一个已有 compatible group。planning pool 为：

```text
target_group.allocated_attention_ranks
  union
该 DP 当前全局未分配 ranks
```

planning 时必须把 target group 的全部 running sequences 作为 `existing_sequences`，把其 allocation 作为 `base_allocation`，确保 merge 后第一轮 Decode metadata 和 append capacity 可行。

一个 candidate 只能包含一个目标 group。第一版不在 admission transaction 中同时 merge 多个已有 groups；runtime Decode planner 后续仍可执行 group-to-group merge。

### 7.3 Candidate 搜索顺序

为减少策略变化并保持确定性，第一版采用：

1. 对本 step 尚未接纳新 batch 的所有 DPs，生成 standalone candidates；
2. 若至少一个 standalone candidate 可行，从中选择：
   - projected live sequence 数更少的 DP；
   - 若相同，选择新增 allocation ranks 更少的 candidate；
   - 若仍相同，选择较小 `dp_idx`；
3. 如果没有 standalone candidate，再生成所有 merge candidates；
4. merge candidates 按以下 key 选择：
   - 新增 allocation ranks 更少；
   - merge 后 group live sequence 数更少；
   - `group_id` 更小；
   - `dp_idx` 更小。

推荐使用显式 tuple 比较实现，禁止依赖 `unordered_map` 遍历顺序。

该顺序的目的不是求解全局最优 placement，而是在不引入新的 profiler/cost model 的前提下：

- 优先保持 group 隔离；
- 独立放不下时完整 merge；
- 避免当前实现只检查最老 group、错过其他可行 group；
- 保证测试可重复。

### 7.4 No-fit 行为

如果一个已经通过空系统静态可行性检查的 pending batch 因当前资源压力，在所有未占用 DP admission slots 上都没有可行 candidate：

- `admission_attempts += 1`；
- batch 保留在 pending queue；
- sequence 状态保持 `WAITING`；
- 不修改任何 ACTIVE block context；
- 不接纳 prefix；
- 不改变 batch ID 或 sequence order；
- 不执行 preemption；
- 不执行历史 KV migration。

no-fit 不应阻止 scheduler 在本 step 尝试后面的 pending batches。第一版采用 bounded bypass：每个 admission step 对 step 开始时存在的每个 pending batch 至多尝试一次，允许后面的较小 batch 使用其他可行 DP。

这避免严格 FIFO 带来的全局 head-of-line blocking，同时保持每个 batch 自身完整。饥饿问题通过 age/attempt metrics 暴露，不通过自动拆 batch 解决。

## 8. Atomic Commit 与 Rollback

### 8.1 Commit 前置条件

只有完整 batch 的 plan 已通过以下检查后才能进入 commit：

- batch 中 sequence 数和 plan placement 数相同；
- sequence IDs 唯一；
- 所有 sequences 仍为 `WAITING`；
- 所有 sequences 仍属于该 pending batch；
- target DP/group 状态自 planning 后未改变；
- block capacity 和 metadata validation 仍成立。

当前 scheduler 是单线程 planning/commit，但仍应保留显式 validation，避免后续异步化时破坏原子性。

### 8.2 Commit 顺序

推荐顺序：

1. 创建 commit guard，记录 target group 的原始状态；
2. 若为 standalone candidate，创建尚未发布的新 group；
3. 为 batch 内全部 sequences 安装 planned placement；
4. 为全部 sequences 分配 logical blocks；
5. 全部分配成功后，再统一：
   - 设置 `RUNNING`；
   - 加入 worker running queue；
   - 加入 group sequence list；
   - 写入 `ls_seq_to_group_`；
   - 写入当前 active `seq_id -> batch_id` ownership；
   - 创建 `InitialBatchPlacement`；
   - 更新 group allocation；
6. 从 pending queue 移除该 batch；
7. 更新 metrics；
8. commit guard 标记成功。

pending batch 不能在 block allocation 完成前从队列删除。

本事务边界截止于 scheduler/SP state 的 initial KV allocation 和状态发布。`LLMEngine.step()` 中随后进行的 dummy pending-token append 不属于该 C++ transaction；initial planner 会为它预留容量，若该容量在单线程 scheduler 返回后仍丢失，engine 将作为 invariant violation 抛错，而不是尝试跨 Python/C++ 回滚 admission。

### 8.3 Rollback

虽然 read-only planner 后的 allocation failure 理论上不应发生，但第一版必须覆盖异常回滚：

- 释放本 transaction 已分配的 blocks；
- 从 running queue 删除已插入 sequences；
- 恢复 sequence status 为 `WAITING`；
- reset 本次安装的 ACTIVE placement；
- 删除本次写入的 `ls_seq_to_group_`；
- 恢复本次写入的 active batch ownership；
- 恢复 existing group 的 sequence/allocation/placement-record sizes；
- standalone group 不对外发布，或完整删除；
- pending batch 保持原位置和原内容。

rollback 后必须满足不变量 I4 和 I5。

禁止 catch 异常后只打印日志并保留部分 admission 状态。

## 9. Admission 后的 Batch/Group Merge

### 9.1 新 batch 并入已有 group

成功 commit 后：

- 新 batch 的所有 sequences 一次性追加到 target group；
- 新 batch 保留自己的 `InitialBatchPlacement`；
- target group 的旧 placement records 不变；
- group allocation 只加入 plan 实际使用的新 ranks；
- 不改变旧 sequences 的历史 KV placement；
- 不执行历史 KV migration。

### 9.2 Running group merge

现有 `_merge_ls_groups()` 继续：

- 合并双方完整 sequence 集合；
- 合并 allocation 并集；
- 按 `(admission_order, batch_id)` 稳定保存所有 initial placement records；
- 更新所有 sequences 的 group owner；
- 不拆 initial batches；
- 不迁移历史 KV。

需要新增 assertion：merge 前后每个 `batch_id` 对应的 live sequence 集合不能被分布到多个 groups。

不能继续只按 `batch_id` 排序 records。bounded bypass 后，较大的 `batch_id` 可能先 admission；若随后按 seal-time `batch_id` 重排，会改变已经运行 requests 的相对执行顺序。`admission_order` 是 group 内持久顺序的权威字段。

### 9.3 Iteration mini-batch

`plan_iteration_masters_source_greedy()` 仍对 group 的完整 live sequence 列表生成本轮 master assignments。

例如：

```text
logical batches: B0={r0,r1}, B1={r2,r3,r4}
merged group:    G0={r0,r1,r2,r3,r4}
iteration mini-batches: M0={r0,r1}, M1={r2,r3}, M2={r4}
```

上述 mini-batch 划分合法，因为它只控制本轮 master work；`B0` 和 `B1` 的持久 identity 仍不变。

## 10. Preemption 与 Readmission

### 10.1 为什么不能恢复原 batch 原子性

一个 initial batch 中可能只有一个 request 因 KV capacity 被 preempt，其他 requests 已经继续生成多个 tokens。如果为了维持原 batch 完整而同时暂停其余 requests，会扩大一次局部容量失败的影响，也不符合当前 request-level preemption 语义。

因此，“batch 完整”约束只覆盖 initial admission 前的 pending logical batch，以及 group merge 不拆 batch record；不要求 initial batch 的所有 requests 在整个生命周期中同步 preempt/finish。

### 10.2 Recovery batch

preempted request：

1. 从原 group 删除；
2. 释放 ACTIVE KV；
3. 重置为 prompt-only waiting state；
4. 创建 singleton `PendingDecodeBatch`；
5. `is_recovery_batch=true`；
6. `parent_batch_id` 指向原 initial batch；
7. 分配新的 recovery `batch_id`；
8. 按当前优先策略放到 pending queue 前部。

使用新的 batch ID 可以避免同一个 immutable initial batch record 被部分重建，并确保 `_merge_ls_groups()` 的稳定去重逻辑没有歧义。

原 `InitialBatchPlacement` 是历史 placement record，不是当前 live ownership 的权威来源。因此，同一个 `seq_id` 可以同时出现在原 initial record 和后来的 recovery record 中；它在任意时刻只能有一个 active batch owner。当前 owner 由 scheduler 的 `seq_id -> batch_id` map 表示，group owner 仍由 `ls_seq_to_group_` 表示。

### 10.3 重复所有权检查

seal、admission、preempt 和 reconcile 后都应能在 debug/test build 校验：

```text
每个 live seq_id 恰好属于以下一个位置：
  unsealed arrival
  xor pending logical batch
  xor running group
  xor finished/to_be_migrated terminal state
```

## 11. Queue、接口与兼容性修改

### 11.1 Scheduler 成员

建议新增：

```cpp
std::deque<PendingDecodeBatch> ls_pending_decode_batches_;
std::unordered_map<uint64_t, uint64_t> ls_seq_to_batch_;
```

`ls_seq_to_batch_` 表示 sequence 当前 active logical batch lineage：seal 时写入，initial admission 后继续保留，preemption 创建 recovery batch 时改写，finish 时删除。历史 `InitialBatchPlacement` 不参与 live ownership 判断。

已有 `waiting_migration` 在 LS-Decode-Core 路径中仅保存 unsealed arrivals。非 LS scheduler 行为保持不变。

不维护“一个 flat queue 同时作为 pending batch queue 的镜像”，避免两个可变真相源。

### 11.2 `Scheduler::add()`

LS-Decode-Core 下继续接收单个 sequence，并将其加入 unsealed arrival queue。batch boundary 不在 `add()` 中产生。

legacy decode path 不变。

### 11.3 `get_total_waiting_migration_size()`

LS-Decode-Core 下返回：

```text
unsealed arrival sequence 数
  +
所有 pending logical batches 的 sequence 数之和
```

这样 server metrics 仍表示等待进入 Decode 的 request 总数。

### 11.4 `is_finished()`

LS-Decode-Core 下必须同时检查：

- unsealed arrivals 为空；
- pending logical batches 为空；
- running queues 为空；
- 没有未完成 migration/recovery state。

### 11.5 Python binding/debug API

不需要向普通用户暴露可变的 pending batch 容器。为测试和日志提供只读 snapshot：

```text
get_ls_pending_batch_ids()
get_ls_pending_batch_sequence_ids()
get_ls_pending_batch_attempts()
```

现有直接读取 `scheduler.waiting_migration` 的 LS 测试和 debug 日志应改用总数或上述 snapshot。legacy 测试仍可访问旧 queue。

### 11.6 Feature flag

不新增 `ls_decode_preserve_batch_atomicity` 配置。

理由：

- LS-Decode-Core 本身默认关闭，已经是实验特性边界；
- batch atomicity 是 group/initial placement 语义的一部分，不应成为长期可选分支；
- 如需对比旧行为，可使用实施前 commit，而不是长期维护双 admission 状态机。

## 12. ScheduleResult 与日志

现有 admission 输出继续保留：

- `ls_initial_batch_ids`；
- `ls_initial_group_ids`；
- `ls_initial_sequence_ids`；
- `ls_initial_kv_dops`；
- `ls_initial_kv_ranks`；
- `ls_initial_prompt_kv_tokens`。

新增等待侧指标：

```text
ls_pending_batch_count
ls_pending_request_count
ls_oldest_pending_batch_age_steps
ls_max_pending_batch_attempts
ls_atomic_admission_no_fit_count
ls_atomic_admission_merge_count
ls_atomic_admission_rollback_count
```

每个成功 admission log 至少包含：

```json
{
  "batch_id": 7,
  "sequence_ids": [101, 102, 103],
  "batch_size": 3,
  "admission_order": 5,
  "dp_idx": 1,
  "group_id": 4,
  "admission_kind": "standalone | merge | recovery",
  "admission_attempts": 2
}
```

验证原子性时必须满足：同一个 `batch_id` 只出现一次成功 admission event，且 event 中的 `sequence_ids` 与 seal event 完全相同。

建议增加 seal log：

```json
{
  "mode": "ls_decode_batch_seal",
  "batch_id": 7,
  "sequence_ids": [101, 102, 103],
  "batch_size": 3
}
```

## 13. 测试设计

### 13.1 Unit tests：batch seal

1. `attention_dp=4`、等待 10 个 requests，形成稳定大小 `[3,3,2,2]`。
2. request 顺序在 batch 内和 batch 间保持 FIFO。
3. 超过 `attention_dp * max_num_seqs` 的尾部仍为 unsealed arrivals，下一 step 再 seal。
4. 已有 pending batch 不与新 arrivals 重组。
5. 相同输入重复运行得到相同 batch IDs 之外的相同切分结构。

### 13.2 Unit tests：atomic admission

1. 完整 batch 可独立 placement 时一次性全部 RUNNING。
2. receiver pressure 下完整 batch 不可行时，零 requests 被接纳。
3. block pressure 下完整 batch 不可行时，零 blocks 泄漏、零 running entries、零 group ownership。
4. 不能创建新 group、但已有 group 可容纳时，完整 batch 并入该 group。
5. 最老 group 不可行、后续 group 可行时，选择后续可行 group，而不是拆 prefix。
6. 所有 candidates 不可行时 batch ID、sequence IDs 和顺序跨 step 不变。
7. 后续容量释放后，等待 batch 一次性全部 admission。

当前 `test_receiver_pressure_admits_only_a_stable_feasible_prefix` 应被替换为 atomic 语义测试，不能继续把 prefix admission 当作正确行为。

### 13.3 Unit tests：bounded bypass

1. 队首大 batch 暂时不可行时，后面的 batch 可在其他 DP admission。
2. 被 bypass 的 batch 不发生 sequence 丢失、重排或拆分。
3. 同一 step 每个 pending batch 最多 planning 一次。
4. attempts/age metrics 正确增长。

### 13.4 Unit tests：merge 与 ordering

1. admission merge 后，旧/new initial placement records 均保留。
2. running group merge 后，每个 batch 的 sequence 集合仍完整存在于 survivor。
3. group sequence order 按 `admission_order` 稳定，bounded bypass 后不因 `batch_id` 排序发生跳变。
4. mini-batch planner 可以跨 initial batch 边界划分本轮 masters，但不修改 records。
5. merge 前后没有重复 `seq_id`。

### 13.5 Unit tests：preemption/readmission

1. preempted sequence 从原 group 删除并形成 singleton recovery batch。
2. 原 batch 的其他 live requests 继续运行。
3. recovery batch 使用新 batch ID，并记录 parent batch ID。
4. recovery admission + 后续 group merge 后，每个 live sequence 恰好出现一次。

### 13.6 Failure injection

failure injection 分为两层：在 SP batch allocation 的第 `k` 个 sequence 后抛异常，用于验证物理 block rollback；在 scheduler 发布第 `k` 个 sequence 后抛异常，用于验证 counters、running queue、group、record、ownership 和 context 的外层 rollback。两者都要求 batch 整体仍 pending。

至少覆盖：

- standalone group rollback；
- merge existing group rollback；
- metrics 不在 rollback 路径重复计数。

### 13.7 Integration tests

单机 1-DP × 8-SP 预检：

- 多个 logical batches 连续到达；
- 一个 batch 首次 no-fit，其他 batch 继续；
- request finish 后释放容量，原 batch 整体 admission；
- 连续至少两个真实 Decode iterations；
- 无 preemption、block leak、重复 sequence 或 collective hang。

4-DP × 8-SP 正式拓扑：

- seal 后 batch sizes 跨 DP 基本均衡；
- 一个 logical batch 永不跨 DP；
- 同一 `batch_id` 不出现在多个 groups；
- group merge 后仍保持 batch records；
- throughput/ITL 与修改前进行回归对比。

所有 GPU 测试按仓库要求先申请提权。

## 14. 实施文件与阶段

### Phase 1：状态模型与只读观测（已完成）

预计修改：

- `csrc/nanodeploy/scheduler/scheduler.h`
  - `PendingDecodeBatch`；
  - pending queue 和 ownership map；
  - seal/snapshot helper declarations。
- `csrc/nanodeploy/scheduler/scheduler.cpp`
  - waiting count、is_finished、preemption queue 适配；
  - seal helper；
  - debug consistency checker。
- `csrc/python/scheduler_binding.cpp`
  - pending batch 只读 snapshot。
- `nanodeploy/engine/llm_engine.py`
  - LS waiting/debug 日志改用只读 API。

Phase 1 不改变 admission policy，但需要保证新状态可以被单测独立验证。

### Phase 2：Whole-batch planner 与 atomic commit（已完成）

预计修改：

- 重写 `_schedule_ls_decode_admission()`，删除 `batch_size--` fallback；
- 增加 standalone/merge candidate 枚举和稳定排序；
- 增加 commit guard/rollback；
- `InitialBatchPlacement.batch_id` 继承 pending batch ID；
- 保证每 DP 每 step 最多一个新 batch admission。

### Phase 3：preemption、metrics 和 tests（已完成 CPU 核心覆盖）

- preemption 创建 singleton recovery batch；
- 增加 pending/atomic admission metrics；
- 替换 prefix-admission 单测；
- 增加 merge、bypass、rollback、lineage 测试；
- 更新设计主文档中的 admission 章节。

### Phase 4：CPU 与 GPU 验证（CPU 和 1-DP preflight 已完成，4-DP long-run 待执行）

1. 编辑 C++ 后执行：

   ```bash
   pip install -v -e .
   ```

2. 运行 LS scheduler/planner/metadata/config CPU tests。
3. 申请 GPU 提权后运行 1-DP × 8-SP preflight。
4. preflight 通过后再运行 4-DP × 8-SP long-run。
5. 分阶段提交，每个 phase 使用独立 commit。

## 15. 风险与权衡

### 15.1 Admission latency

prefix admission 会让部分 requests 尽早开始；atomic admission 可能让整个 batch 多等待若干 steps。因此必须记录 batch wait age 和 no-fit count。

### 15.2 DP 利用率

本文在 seal 阶段先形成多个 DP-local batches，避免把一次外部 list 强制到单 DP。但 seal 后不再重切，个别 DP 的 transient pressure 仍可能使一个 batch 等待，而其他 DP 容量不能通过拆 batch 被利用。这是保持 batch identity 的预期代价。

### 15.3 Bounded bypass 与公平性

允许后续 batch bypass 可以改善吞吐，但大 batch 可能等待更久。第一版只观测 starvation，不引入 aging boost、batch split 或请求级抢占。若实验出现显著 starvation，再单独设计 fairness policy。

### 15.4 Commit rollback 复杂度

当前 block allocation 是逐 sequence 修改状态。缺少 rollback 会把一次意外分配失败变成部分 admission，直接破坏本文核心不变量。因此 commit guard 不是可选优化，而是验收要求。

### 15.5 与 LoongServe 的边界

本文复现的是“logical batch 形成后不持久拆分”的语义，不声称复现 LoongServe 的 Prefill DP、prefill latency model 或全局 scheduling DP。

## 16. 明确不做

本次实施不包含：

- 真实 Prefill batching 或 Prefill model execution；
- 按 prompt length 排序或长短请求聚类；
- `has_wait_tokens` / `max_wait_tokens`；
- Prefill profiler 或 `max_prefill_time`；
- KV compaction、历史 KV migration 或 rank-releasing scale-down；
- 跨 DP group；
- 动态 NCCL/process group；
- GPU/Ray worker 启停；
- runtime mini-batch identity；
- 已封存 pending batch 的自动拆分；
- 新的全局最优 batching/placement DP。

## 17. 验收标准

实现完成必须同时满足：

1. scheduler seal 后的 batch sequence 集合不可变。
2. admission 不再存在 prefix fallback。
3. 每个成功 admission event 包含 sealed batch 的全部 sequence IDs。
4. no-fit 后 sequence/group/block 状态零变化。
5. logical batch 只能完整创建 group 或完整并入一个 group。
6. 同一 batch 不能跨 DP 或跨 groups。
7. running group merge 保留全部 initial batch records。
8. iteration mini-batch 可以变化，但不改变 batch membership。
9. preempted request 形成 singleton recovery batch，不复制 live sequence。
10. failure injection 证明 standalone/merge commit 均可完整 rollback。
11. LS CPU 单测全部通过。
12. 1-DP × 8-SP GPU preflight 无 hang、block leak 或重复 sequence。
13. 4-DP × 8-SP long-run 中 batch atomicity metrics 与日志一致。
14. feature flag 关闭时 legacy scheduler 行为不变。

## 18. Review 重点

开始实现前需要重点确认以下设计选择：

1. **Batch identity 边界**：采用 scheduler seal 后的 DP-local logical batch，而不是 `add_request(list)` 边界。
2. **Seal 策略**：每 step 最多封存 `attention_dp * max_num_seqs` 个 arrivals，并均衡形成最多 `attention_dp` 个 batches。
3. **No-fit 行为**：batch 原样等待，永不自动拆分。
4. **Queue 策略**：允许 bounded bypass，而不是严格 FIFO 全局阻塞。
5. **Candidate 策略**：先 standalone，全部不可行后搜索所有 compatible groups。
6. **Preemption**：单 request 使用新的 singleton recovery batch，不暂停原 batch 其他 requests。
7. **配置**：不增加 atomicity 开关，LS-Decode-Core 开启时直接采用新语义。
8. **事务要求**：必须实现 admission rollback，不接受“planner 理论上不会失败”作为省略 rollback 的理由。
9. **Merge 范围**：第一版 admission transaction 只并入一个已有 group，不同时合并多个已有 groups；若单 group + unallocated ranks 仍不可行，则保持整批等待。

## 19. 实施状态与最小文件范围

截至当前实现：

- `scheduler.h/.cpp`：sealed pending queue、active batch ownership、静态 seal 可行性、全候选搜索、bounded bypass、recovery lineage、稳定 merge ordering、等待指标与只读 snapshot 已实现。
- `sp_state_manager.h/.cpp`：整批 initial allocation 使用聚合预检、物理分配和统一 counters commit；异常时扫描并释放本 transaction 的全部 tables。
- `block_manager.cpp`：uncached allocation 在修改 free list 前预留容器，并把可能分配内存的 used-set 插入提前，降低中途异常造成 block 丢失的风险。
- `scheduler_binding.cpp`、`sp_state_manager_binding.cpp`：pending/group/active ownership snapshot、ScheduleResult 指标、counter 观测和两层 failure injection 已绑定。
- `llm_engine.py`：seal、admission lineage 和 pending/no-fit 指标日志已接入。
- `tests/test_ls_decode_scheduler.py`：覆盖 seal/FIFO、稳定 no-fit、容量释放、bounded bypass、搜索非最老 group、standalone/merge rollback、recovery lineage、抢占 DP 所有权和 feature-flag-off。

验证结果：LS scheduler/planner/metadata/config 相关 CPU 回归共 58 项全部通过。1-DP × 8-SP GPU 短时 preflight 使用 3 个 warmup requests 和 9 个正式 requests，全部完成并 drain；正式阶段包含 5 次 admission step、8 次 Decode step，未出现 preemption、hang、重复 sequence 或未完成请求。4-DP × 8-SP long-run 及修改前后吞吐/ITL 对比不在本次最小实现提交内，仍需单独执行。

从“只改必要内容”的角度，`examples/bench_serving.py` 不参与核心语义，也没有改变逐 request `add_request()` 的方式。本次核心实施不修改它；若用该脚本跑 LS 性能实验，只需另行增加现有 LS flag 与 `max_num_recv_seqs` 的 CLI 透传。正式 GPU 验证优先使用专用 LS long-run 脚本，并在运行前按仓库规则申请 GPU 提权。
