# NanoDeploy LoongServe-style baseline 源码对齐方案（会议讨论稿）

日期：2026-07-18

状态：Proposal，待会议决策

NanoDeploy 基线：`64a199154b375aba8cd469ca75542f4103ccbf64`，其中 batching 排序改动为 `908933a`

LoongServe 源码基线：`fb87896d87b170afd4afe591e29da1aa5f6d4e16`

## 0. 会议结论先行

本方案不以兼容 NanoDeploy 之前的 LS 调度设计为约束。目标是把 LoongServe 的调度策略迁移到 NanoDeploy，形成可以用于论文实验的 LoongServe-style baseline；旧的 persistent logical batch、均匀切 batch、Nano 自定义 consolidation 等逻辑，如果与 LoongServe 冲突，可以直接替换。

建议采用以下原则：

1. **LoongServe 的控制面语义优先。** 请求选择、有限 OOE、admission cadence、实例分配、batching/DoP DP、KV placement、Decode merge/scale-up 以 LoongServe 源码为准。
2. **NanoDeploy 约束只作为数据面 safety gate。** block rounding、receiver metadata、dummy pending token、事务 rollback 等必须保留，但不能反过来定义新的 batching policy。
3. **Dummy Prefill 只改变执行，不改变调度决策接口。** 当前实验不运行 Prefill model，但 LoongServe 的 profiler cost、batching DP 和 macro scheduling 仍应保留为 policy；否则只能称为近似实现。
4. **不引入 LoongServe 源码中不存在的 gap clustering、固定比例分组或 age boost。** 如果需要工程优化，作为单独增强项做消融，不混入 baseline。
5. **新 arrival 在真正获得本轮执行计划前保持 request-level waiting。** 删除“先 seal 成长期原子 batch、之后数千次重试”的核心语义；batch 原子性只覆盖一次已经生成的 placement commit。

目标控制流如下：

```text
waiting requests
      |
      | FCFS scan + bounded OOE
      | current/future KV + request/token limits + profiler
      v
本轮可运行请求集合 Rp
      |
      | LoongServe-style elastic instance allocation
      v
每个 Nano DP domain 的可用实例集合 Ep
      |
      | prompt length 降序 + instance used-token 升序
      | 原版二维 DP：连续 request range + 连续 instance range/DoP
      v
ephemeral batch plans
      |
      | LoongServe packed interval placement
      | Nano exact safety validation + atomic commit
      v
running Decode groups
      |
      | 原版 memory-deficit merge / idle-rank scale-up
      | compute-threshold multi-master scale-up
      v
下一 Decode iteration
```

完成本方案前，当前实现的准确称呼应是 **Nano LS-Decode-Core / LoongServe-inspired**，不建议直接写成“LoongServe baseline”。

## 1. Baseline 的定义与边界

### 1.1 要复现的内容

本 baseline 复现 LoongServe elastic-SP 路径的调度决策：

1. waiting queue dispatch；
2. elastic instance allocation；
3. batching 与初始 DoP 联合优化；
4. KV placement；
5. Prefill/Decode macro iteration 的资源关系；
6. Decode memory/compute scale-up；
7. pause/offload 后的 request-level readmission。

LoongServe 论文将其描述为 scalable four-step scheduling algorithm，源码主入口分别位于：

- request dispatch：`req_queue.py:80-227`；
- Prefill/Decode 联合资源分配：`manager.py:516-684`；
- batching/DoP DP：`manager.py:686-750`、`longserve_c_scheduler/src/main.cpp:33-84`；
- Decode merge/scale-up：`manager.py:844-970`。

### 1.2 不要求复现的内容

以下属于执行框架差异，不要求逐行相同：

- Ray/RPC 接口；
- NanoDeploy 的 block table 和 receiver metadata 表示；
- dummy weight；
- CUDA Graph 和固定 collective cadence；
- C++/Python 的具体模块边界；
- batch ID 的具体类型；
- allocation/commit/rollback 的异常安全实现。

但这些差异只能影响“如何安全执行 LoongServe plan”，不能悄悄改变“LoongServe plan 是什么”。

### 1.3 Dummy Prefill 的解释

当前实验是 Decode-only、Dummy Prefill：新请求的 prompt KV 由 scheduler 直接建立 placement，Python 端追加一个 dummy token，不执行真实 Prefill forward。

这意味着：

- 不能用 Nano 实测 Prefill latency评价 LoongServe profiler 的准确性；
- 仍然可以使用固定的 LoongServe analytical model 产生 batching/DoP 决策；
- 这类实验复现的是 **LoongServe scheduling policy 对 Decode 阶段资源状态的影响**，不是完整 LoongServe 端到端 Prefill 性能。

实验报告必须记录 profiler 参数文件及 hash。没有 profiler 参数而使用自定义 heuristic 的运行，不能与 source-aligned baseline 混为一组。

## 2. 当前实现为什么还不是 LoongServe baseline

| 调度环节 | Nano 当前实现 | LoongServe 源码 | 结论 |
|---|---|---|---|
| Admission cadence | 每个 `schedule()` 都 admission-first；成功则本轮不 Decode | idle 或累计 `max_wait_tokens` 个 Decode iteration 后 admission | 需要替换 |
| Request dispatch | 固定截取 `attention_dp * max_num_seqs` 个请求 | FCFS 扫描，结合 current/future capacity、token/request limit、profiler，有限 OOE | 需要替换 |
| Batch identity | placement 前先创建长期 `PendingDecodeBatch` | 本轮 request set 生成 plan 后立即运行，没有长期冻结 batch | 需要替换 |
| Batching | 排序后按请求数均分成至多 `attention_dp` 批 | 二维 DP 决定任意连续 request ranges 与对应 DoP | 需要替换 |
| Initial DoP | 每批独立选择第一个可行 `d` | batching DP 联合优化所有 batch 的 DoP | 需要替换 |
| Instance pool | DP domain 内局部选择，跨 DP 不能共享 | 单一全局 elastic instance pool | 必须做明确的拓扑适配 |
| Prompt KV placement | 每个 request 在所选 ranks 上均匀 striping | 按实例剩余容量顺序填充连续 token intervals | 需要替换 |
| Future-KV | batch/group-local envelope | running + paused + 本轮已选请求的全系统 envelope | 需要替换作用域 |
| Scale-down | utilization/stability/cooldown 驱动 consolidation | admission 时按 Prefill speedup、Decode slowdown、migration cost 决定 | baseline 中需要替换 |
| Decode scale-up | capacity/compute/planner failure 都可能触发自定义 group merge | memory deficit merge 与 compute idle-instance scale-up 分开 | 需要收敛到源码语义 |
| Preemption | 丢弃 generated tokens，回到 prompt，创建 recovery batch | 保留 output progress，offload/KVKEEP 后回 waiting | 需要替换 |

当前关键代码位置：

- 固定 window/均匀分组：`csrc/nanodeploy/scheduler/scheduler.cpp:1705-1810`；
- persistent pending admission：`scheduler.cpp:2085-2365`；
- admission-first/Decode 互斥：`scheduler.cpp:924-985`；
- group-local future-KV：`scheduler.cpp:1348-1485`；
- 最小可行 initial DoP 和均匀 KV striping：`scheduler.cpp:1488-1635`；
- 自定义 consolidation：`scheduler.cpp:641-900`；
- 自定义 Decode merge/scale-up：`scheduler.cpp:2368-2598`；
- restart-from-prompt preemption：`scheduler.cpp:2822-2900`。

## 3. 现有实验给出的直接证据

### 3.1 排序加均分没有解除长短绑定

Issue 1% 轨迹中的一个 arrival window 为：

```text
FIFO:   [276, 199, 923230, 229, 197, 213, 227]
排序后: [923230, 276, 229, 227, 213, 199, 197]
DP2 均分:
  B0 = [923230, 276, 229, 227]
  B1 = [213, 199, 197]
```

因此 `908933a` 虽然正确加入了长度降序，但超长请求仍然绑定三个短请求。对已观察到的六个混合窗口静态重放后，仍有九个短请求与 840K--972K prompt 处于同一长期 batch，只是成员身份发生变化。

### 3.2 Persistent pending 带来控制面放大

141 GiB 运行记录了：

- 七个超长 prompt 连带九个短请求等待；
- 单 batch 最多 admission 4,406 次；
- no-fit batch 在资源没有变得更有利时仍每个 Decode step 重算。

最新 140 GiB / memory fraction 0.85 运行虽然完整完成 7,200 请求，但仍出现：

- pending 峰值 `106 batches / 300 requests`；
- oldest pending 等待 4,255 scheduler steps；
- 单 batch 最多尝试 4,256 次；
- scheduler overhead P95 `115.38 ms`；
- queueing P99 `140.84 s`。

这说明当前主要问题已经不是排序缺失，而是 request 在 current-fit 之前被永久 batch 化，以及与 LoongServe 不同的 admission 重试状态机。

## 4. 目标状态模型

### 4.1 删除 fresh-arrival persistent pending batch

Fresh request 只存在两种稳定状态：

```text
WAITING_REQUEST  ->  RUNNING_REQUEST
```

中间计划全部是单次 `schedule()` 内的 ephemeral 对象：

```cpp
struct LoongDispatchCandidate {
    std::shared_ptr<Sequence> sequence;
    uint64_t fifo_order;
    std::vector<int> feasible_dp_domains;
};

struct LoongPlannedBatch {
    int dp_idx;
    std::vector<std::shared_ptr<Sequence>> sequences;
    std::vector<int> instance_ranks;
    std::vector<std::vector<TokenInterval>> kv_intervals;
    double predicted_cost;
};
```

`LoongPlannedBatch` 只有在以下条件全部满足后才取得 batch ID：

1. DP 回溯已经确定 request range 和 DoP；
2. Nano exact placement validation 成功；
3. physical allocation 成功；
4. scheduler 准备发布 running/group ownership。

如果 planning no-fit：

- request 保持在 `waiting_migration`；
- 不创建 batch ID；
- 不创建 `seq_id -> batch_id` ownership；
- 不进入长期 pending deque；
- 后续短请求是否可以执行由 LoongServe bounded OOE 决定。

### 4.2 保留 atomic commit，但改变原子性的边界

需要保留 Nano 当前的 allocation rollback，不过原子性定义改为：

> 一份已经生成的 `LoongPlannedBatch` 要么全部成功 commit，要么完全不产生物理和逻辑状态。

不再定义：

> 一组在过去某个 step 被 seal 的 request 必须在未来一直作为不可拆 batch 等待。

因此，旧文档 `ls_decode_atomic_batch_admission_design.md` 关于 fresh-arrival persistent batch 的章节仅作为历史记录，不再是新 baseline 的约束。其 allocation、publication、rollback 不变量继续复用。

### 4.3 Pause/Recovery 回到 request-level

LoongServe pause/offload 保留 `output_ids`。Nano baseline 应改为：

- 释放或保留 KV 由 pause mode 决定；
- `token_ids` 不截回 prompt；
- readmission 所需的 first-router tokens 为 `prompt + generated` 或 0（KVKEEP）；
- paused request 回到 request-level waiting scan；
- 不再创建优先级高于普通请求的 singleton recovery batch。

## 5. 第一步：按 LoongServe 实现 request dispatch

### 5.1 Admission opportunity

新增 source-equivalent 状态：

```text
decode_iterations_since_last_admission
```

只有以下情况进入 dispatch：

```text
running 为空
or decode_iterations_since_last_admission >= max_wait_tokens
```

LoongServe API default 与官方 artifact launcher 均使用：

```text
max_wait_tokens = 10
```

这对应 LoongServe `manager.py:351-373`。如果 admission 没选出请求且系统仍有 running batch，本轮继续 Decode；达到 threshold 后，后续 step 仍可继续检查，行为与源码一致。

### 5.2 FCFS scan 与 bounded OOE

扫描逻辑按 LoongServe `req_queue.py:135-223` 实现：

```text
for request in waiting FIFO order:
    检查 running request limit
    检查 batch_max_tokens
    检查 current prompt/KV capacity
    检查 running + 本轮候选的 future-KV envelope
    检查 profiler prefill time / slowdown 条件

    if 可运行:
        加入本轮候选
    else if allow_ooe:
        保留在 waiting，继续扫描后项
    else:
        停止扫描
```

维护源码同名状态：

```text
num_ooe
max_num_ooe
```

语义与源码一致：限制的是“连续有后项越过前项并获得服务的调度轮数”，不是单轮最多跳过多少 request。`api_server.py` 默认 `max_num_ooe=10`，但官方实验启动脚本会按 dataset 设为 1、4、8、64 或 -1；因此 10 是 source API default，不是所有 LoongServe 实验的固定值。

等长或同时可运行请求继续保持 FIFO。只有 membership 确定后，才按 prompt length 稳定降序。

### 5.3 不能省略的 undecided-prefix 状态机

`req_queue.py` 并不是简单的 first-fit scan。移植时必须保留以下三个集合/游标：

```text
can_run_list
undecided_req_list
can_append_run_idx
```

当 request 通过全局 future/request/token 检查，但由于 isolated-idle capacity 或 Prefill/Decode trade-off 不能立即进入 `can_run_list` 时，源码会将一段 FIFO prefix 放入 `undecided_req_list`，并累计：

```text
undecided_prefill_sum
undecided_prefill_square_sum
sum(1 / first_router_need_tokens)
```

随后用源码条件更新可以整体接纳的 prefix 长度：

```text
prefill_time(all selected + undecided prefix)
    * sum_running(1 / generated_tokens)
<=
estimated_decode_waiting_time
    * sum_undecided(1 / first_router_need_tokens)
```

扫描结束后，只把 `undecided_req_list[:can_append_run_idx]` 加入本轮执行，其余 undecided requests 放回 waiting 前部。`need_break`、首 batch 可突破 `max_prefill_time`、后续 batch 必须满足 time limit 等分支也按源码保留。

因此 Phase 1 不能把 Loong dispatch 实现成“遇到 no-fit 就 bounded skip”的简化循环；CPU differential fixture 应直接比较最终 selected IDs、waiting IDs 和 `num_ooe`。

### 5.4 Dispatch 条件

需要对齐的 source 条件：

- `running_max_req_size`，默认 1000；
- `batch_max_tokens`，默认：

  ```text
  max(max_req_total_len, total_kv_tokens / 6)
  ```

- current idle token slots；
- scale-up disabled 时，prompt 必须放入当前真正 idle instances；
- future-KV high-water mark；
- `max_prefill_time`，默认 500 ms；
- predicted Prefill slowdown 与 estimated Decode waiting cost。

当前 workload 使用 `ignore_eos=True`，Nano 已实现的 `(held_tokens, remaining_iterations)` 公式可以复用。

future tuple 仍应逐状态对齐 `io_struct.py:116-135`。在 `ignore_eos=True` 时：

```text
RUNNING:           (input + output,     max_output - output - 1)
WAIT_IN_QUEUE:     (input + 1,          max_output - 2)
PAUSED_AND_OFFLOAD:(input + output + 1, max_output - output - 2)
PAUSED_AND_KVKEEP: (input + output,     max_output - output - 1)
```

所有 remaining 项取 `max(0, ...)`。Nano 的 block rounding 和 dummy pending-token 只能在此 token-level envelope 之后做 exact safety gate，不能提前把公式替换成另一套 admission policy。

### 5.5 不复刻源码 bug

LoongServe `req_queue.py:178-180` 中 `undecided_cache_len_list` 为空时的 `[:-0]` 会清空 cache list，疑似实现 bug。baseline 应复现论文和函数设计意图，不复制明显 bug。所有 intentional deviation 必须在文档和测试中列明。

## 6. Nano DP topology 的必要适配

LoongServe 使用单一全局 elastic SP instance pool。Nano 当前 topology 为：

```text
Attention DP{1,2,4} x SP8
```

Decode group 不能跨 `attention_dp` domain 合并。这不是 scheduler.cpp 局部修改能消除的限制。

建议的 source-shaped 映射为：

1. 保留一个全局 FIFO request queue；
2. 全局 dispatcher 对每个 request 计算可行 DP domains；
3. 使用以下稳定 score 选择 domain：

   ```text
   projected_future_kv_utilization
   -> current_used_kv_tokens
   -> running_request_count
   -> dp_idx
   ```

4. 每个 DP domain 内部视为一个独立 LoongServe elastic instance pool；
5. batching/DoP DP 在每个 domain 的 SP8 ranks 上独立运行；
6. bounded OOE 仍由全局 queue 维护，避免变成多个不一致的本地 FIFO。

这不是 LoongServe 原始全局池的逐字复刻，但它是 Nano 数据并行拓扑下最小且明确的适配。实验命名应为：

```text
LoongServe scheduling policy on Nano DP-domain execution backend
```

如果会议要求严格统一实例池，则需要另立项目修改 Attention/FFN parallel topology，超出本调度改造范围。

## 7. 第二步：对齐 future-KV 的检查域

### 7.1 Domain-level envelope

对每个 DP domain，future-KV 输入必须包含：

```text
该 domain 所有 running groups 的 requests
+ 本轮已经 tentative dispatch 到该 domain 的 requests
+ 本轮正在检查的新 request
+ paused-and-KVKEEP requests 的已占用 KV
```

容量为该 domain 全部可用 KV blocks 转换出的 token slots，扣除永久 dummy/system blocks 和 paused 占用。

这替换当前：

```text
candidate + one target group
```

的局部 envelope。

### 7.2 消除 future capacity 重复承诺

当前 memory scale-up 开启时，initial placement 会用整个 `ordered_pool` 通过 future check，但 commit 只把首个可行 `d` 个 ranks 登记给 group。未登记 ranks 没有 reservation，后续 standalone group 仍可再次把它们计入 future capacity。

Domain-level envelope 通过把所有 running 和 tentative candidates 放入同一次 peak 计算，消除这种跨 group 重复承诺，不需要为每个 request 预留固定未来 rank。

### 7.3 Nano exact safety gate

LoongServe envelope 是 token-level aggregate check。Nano 仍需在其后检查：

- 64-token block rounding；
- dummy pending-token tail block；
- receiver metadata；
- master metadata；
- `reserved_blocks_per_req`；
- 当前 group ownership 和可迁移性。

这些检查只能拒绝一个 LoongServe plan，不能静默改成另一个 heuristic plan。被拒绝的 DP transition 应进入显式 replan，并记录 `nano_adapter_reject_reason`。

## 8. 第三步：移植 LoongServe elastic instance allocation

LoongServe 在 batching DP 之前先确定本轮可用于新请求的实例集合。需要新增独立函数：

```cpp
LoongInstanceAllocationPlan plan_loong_instance_allocation(
    const DispatchResult& requests,
    const RunningBatchSnapshot& running,
    const KVCapacitySnapshot& capacity,
    const LoongProfiler& profiler);
```

该步骤复现 `manager.py:516-684`：

1. 统计所有 running batches 的 used tokens；
2. 识别完全 idle instances；
3. 必要时计算最小 busy instance 数；
4. 比较 Prefill speedup 与 KV migration cost，决定是否压缩 Decode group；
5. 比较 Prefill speedup 与 Decode slowdown，决定哪些 Decode batches 本 macro iteration 继续运行；
6. 不足时释放/暂停 Decode batches，为 Prefill batch 提供实例；
7. 输出每个 DP domain 的 `available_instances`。

### 8.1 Macro 内的 Decode batch 分类

LoongServe 在一次新请求调度中把已有 Decode batches 分成两类：

- `running_decode_batch_list`：与本轮 Prefill 在不同实例上并行继续 Decode；
- `pending_decode_batch_list`：本 macro iteration 暂停执行并让出实例。

这里的 `pending_decode_batch_list` 只是一次 `_schedule_new_req_list_with_decoding()` 调用内的临时分类，不得重新实现成 Nano 当前跨 step 持久化的 fresh-arrival pending queue。

### 8.2 Prefill 后的 group 重组

源码在 Prefill 完成后，会按实例集合交集处理临时 pending Decode batches：

1. pending Decode batch 与某个 new Prefill batch 的 `occupied_instances` 相交时，执行 `_merge_batch()` 并合入该 new batch；
2. 与所有 new batches 均不相交的 pending Decode batch继续独立存在；
3. 本轮继续 Decode 的 batches由独立 coroutine并发推进。

Dummy Prefill backend 也要保留这段 group ownership 变化，只是触发点从 Prefill forward 完成变成 dummy placement/commit barrier 完成。否则 instance allocation 的结果没有真正落实到下一轮 Decode group，仍然不是 LoongServe 控制流。

### 8.3 替换 Nano 自定义 consolidation policy

source-aligned baseline 不使用以下触发条件决定 scale-down：

- utilization candidate threshold；
- stable steps；
- cooldown；
- periodic check interval；
- pending batch exact-benefit heuristic。

Nano 现有 KV consolidation transaction、P2P chunking、reservation 和 rollback 可以作为上述 LoongServe allocation plan 的执行器继续使用。

建议实验分组：

```text
Loong-base                    # 只运行 source-aligned allocation trigger
Loong-base+Nano-consolidation # 可选增强项，单独消融
```

当前 140/141 GiB 正式运行启用了 Nano `execute` consolidation，因此不能直接作为纯 LoongServe baseline 结果。

## 9. 第四步：完整移植二维 batching/DoP DP

### 9.1 输入排序

每个 DP domain：

```text
requests: stable sort by first_router_need_tokens descending
instances: stable sort key = (used_tokens ascending, node_id ascending)
```

instance key 完全对应源码 `(total_used_tokens_list[x], x // local_sp_world_size)`；同 key 时保留 `available_instances` 的原始顺序，不额外引入 rank-ID tie-breaker。

Fresh waiting request 的 `first_router_need_tokens = prompt length`。Paused-offload request 使用 `prompt + generated`，KVKEEP request 使用 0。

### 9.2 DP 状态和转移

原版状态：

```text
f[i][k] = 前 i 个已排序 requests 使用前 k 个已排序 instances 的最小输入延迟目标
```

转移枚举最后一个 batch：

```text
last_batch_size = b
last_used_instances = d
predecessor = f[i-b][k-d]
```

可行性：

```text
sum_prompt_tokens(requests[i-b:i])
    <=
sum_free_tokens(instances[k-d:k])
```

代价必须与源码一致：

```text
predict(d, sum(L), sum(L^2)) * sum(1/L)

predict(d, sum(L), sum(L^2))
    = A[d] + B[d] * sum(L) + C[d] * sum(L^2)
```

不能把 cost 替换为：

- 请求个数方差；
- 最大相邻 prompt gap；
- 固定 long/short threshold；
- 最少 batch 数；
- 最小 initial DoP。

### 9.3 Profiler 参数是 baseline 的硬依赖

新增 LoongServe-compatible profiler CSV loader，并在运行 manifest 中记录：

- CSV path；
- SHA256；
- model/hardware 标识；
- `A[d], B[d], C[d]` 完整参数；
- SP world size；
- 参数来源。

会议需要确定参数来源：

1. 复用 LoongServe 原实验 profiler；
2. 对 Nano 对应模型/H200 运行单独 Prefill microbenchmark 后拟合；
3. 使用论文给出的固定 analytical model 参数。

推荐顺序为 2 > 1 > 3。若当前 Nano 没有可用 Prefill kernel，则先使用 LoongServe 原 profile 完成 policy reproduction，但必须在实验中声明 cost model 与 Nano dummy execution 解耦。

没有 profiler 文件时，source-aligned mode 应启动失败，而不是退化到均匀分组。

### 9.4 回溯和 batch 创建

从 `min_k f[n][k]` 回溯得到：

```text
request range [i-b, i)
instance range [k-d, k)
DoP = d
```

每个回溯结果先生成 ephemeral `LoongPlannedBatch`。所有 batch 使用互不重叠的 instance ranges。

### 9.5 Nano safety rejection 后的行为

如果 aggregate token capacity 可行，但 Nano exact gate 因 block/receiver/headroom 拒绝：

1. 记录被拒绝的 DP transition `(req_begin, req_end, inst_begin, inst_end)`；
2. 禁止该 transition；
3. 重新运行 DP；
4. 如果所有 plans 都不可行，将受影响 requests 保留在 waiting；
5. bounded OOE 决定后续 request 是否仍可执行。

禁止静默回退到“从 `d=1` 开始找第一个可行值”，否则 batching 和 DoP 又被 Nano heuristic 改写。

## 10. 第五步：对齐 prompt KV placement

LoongServe `_get_batch_prefill_migration_plan()` 的语义是：

1. batch instances 按 used tokens 从高到低排列；
2. 先填更满实例的剩余空间；
3. request prompt 以连续 token interval 顺序填入；
4. 一个 request 只有跨越实例容量边界时才分布到多个 instances。

Nano 当前对 batch 中每个 request 都在 `d` 个 ranks 上平均 striping。这会增加：

- 每个短 request 的 KV owner 数；
- Decode receiver metadata；
- remote attention flow；
- block rounding；
- initial placement 的实际容量需求。

baseline 应新增 packed interval placement，并让 `InitialBatchPlacement` 记录每个 request/rank 的 token interval，而不只是 token count。Nano block allocator按 interval 分配 blocks，receiver/master exact gate继续验证。

如果执行层暂时只能接收 token counts，可以先从 interval 生成 counts；但排序填充语义必须与 LoongServe 一致。

## 11. 第六步：支持 Prefill/Decode macro iteration

### 11.1 当前问题

Nano `ScheduleResult` 只能是：

```text
ADMISSION or DECODE
```

只要 admission 成功，本轮所有旧 running requests 都不 Decode。LoongServe 会在同一个 macro iteration 中并发执行：

- 本轮新请求的 Prefill batches；
- 被保留运行的 Decode batches。

### 11.2 目标接口

增加 mixed result：

```cpp
struct ScheduleResult {
    ScheduleAction action; // DECODE, ADMISSION_ONLY, ADMISSION_AND_DECODE, MAINTENANCE
    admission_dp_seqs;
    decode_dp_sp_seqs;
    initial_batch_records;
};
```

Dummy Prefill 下：

1. 新 admitted requests 只执行 KV placement 和 dummy first-token bookkeeping；
2. 本轮之前已 running 的 requests 正常进行一次 Decode forward；
3. 新 requests 不在同一 macro iteration 再额外 Decode 一次；
4. postprocess 分别处理两个 lane；
5. `decode_iterations_since_last_admission` 只按真实 Decode iteration 更新。

这样既不制造 admission-only GPU bubble，也不会让新 request 在一个 macro iteration 多生成一个 token。

## 12. 第七步：对齐运行中 Decode scale-up/merge

### 12.1 Memory deficit

按 LoongServe `manager.py:844-906`：

1. 对每个 batch 计算：

   ```text
   idle_tokens = capacity - used_tokens - num_running_requests
   ```

2. 将 batch 分为 can-decode 和 cannot-decode；
3. cannot-decode batch 优先吸收 idle token 最大的 can-decode batch；
4. 合并后仍不足，按精确 deficit 增加 idle instances；
5. 执行 group merge。

Nano 可以使用 block-level exact append slack 作为安全验证，但 donor 顺序和触发条件必须对应上述 token-level policy。

### 12.2 Compute bound

按 LoongServe `manager.py:910-969`：

```text
while remaining_requests / remaining_instances
      > min_comp_bound_decoding_batch_size:
    add one idle instance
```

LoongServe API 默认：

```text
min_comp_bound_decoding_batch_size = 100
```

但官方 artifact 启动脚本 `test/longserve/5-start-api-server.py:143` 显式使用 128。因此 100 与 128 都有 source provenance：前者是 API default，后者是 LoongServe artifact evaluation profile。当前正式实验的 128 不能仅凭数值判定为不对齐；真正不对齐的是现有 `ls_decode_batch_per_master` 还同时承担 Nano 自定义 DoP/merge heuristic，而不是只实现源码 compute-bound 分支。当前 config 默认 64 和部分 benchmark 默认 8 则没有对应的 Loong reference profile。

### 12.3 禁止额外 compute-pressure merge

LoongServe compute-bound path只消费 idle instances，不会仅因为所需 compute DoP 被另一个健康 group 占有，就强制合并那个健康 group。

Nano 当前 compute-pressure merge、planner-failure arbitrary merge 应从 baseline path 移除。若 source policy最终无法生成安全 plan，再进入 LoongServe pause/offload，而不是引入新的 merge heuristic。

### 12.4 Scale-up disable 语义

LoongServe `disable_scale_up` 会切换到无 scale-up Decode scheduler。Nano 当前 memory flag 不完全禁止 compute scale-up。baseline 应使用单一 source-equivalent flag，保证：

- initial instance allocation；
- memory scale-up；
- compute scale-up；
- merge 所需实例扩展；

都遵循同一个开关。

## 13. 配置建议

建议新增或重命名为 source-equivalent 配置：

```text
ls_max_wait_tokens = 10
ls_max_num_ooe = 10
ls_running_max_req_size = 1000
ls_batch_max_tokens = auto
ls_max_prefill_time_ms = 500
ls_avg_decoding_time_ms = 30
ls_min_comp_bound_decoding_batch_size = 100
ls_prefill_profiler_path = <required>
ls_disable_scale_up = false
ls_use_fixed_sp = false
```

必须把“算法实现”和“参数 profile”分开。仓库中至少固定两个只读 profile：

| Profile | `max_num_ooe` | `max_prefill_time_ms` | Decode threshold | `batch_max_tokens` | 用途 |
|---|---:|---:|---:|---:|---|
| `loong_source_default` | 10 | 500 | 100 | `max(max_req_total_len, total_kv/6)` | 检查 API 默认语义 |
| `loong_artifact_derived` | dataset-specific | 5000 | 128 | 原 artifact 为 500000 | 复现实验脚本语义 |

Nano Issue 1% 含接近 1M-token prompt，不能照抄 artifact 的 500000，因为 LoongServe 本身要求 `batch_max_tokens >= max_req_total_len`。会议应为该 workload 冻结一个 `loong_nano_issue001` profile：保留 source 算法，容量相关参数按 Nano hardware/workload 合法取值，并在 manifest 中逐项标出 provenance 和 adaptation reason。

参数变化不等于算法不对齐，但禁止一次实验中隐式使用 config、benchmark script 和 CLI 三套默认值。正式 baseline 必须在启动日志打印完整 resolved profile。

baseline 固定条件：

```text
enable_ls_decode_core_scheduler = true
dummy_prefill = true
loop_count = 1
ls_decode_enable_future_kv_admission = true
ls_nano_background_consolidation = false
```

这里关闭的是 Nano 原先由 utilization/stability/cooldown 触发的后台 consolidation，不是关闭 LoongServe instance-allocation 阶段要求的 scale-down/migration。旧 `ls_kv_consolidation_mode` 应拆分或废弃，不能让一个 `off` 同时误伤 source-required reallocation executor。

旧参数 `ls_decode_initial_kv_dop` 在 source-aligned mode 中删除或禁止非零值，因为 initial DoP 必须由 DP 输出。旧参数 `ls_decode_batch_per_master` 由 source-equivalent decode threshold 替代。

不建议长期保留 `balanced` 和 `loong_dp` 两套 policy。A/B 使用 Git commit 和 manifest 回溯；开发期间如需临时开关，baseline 完成后删除。

## 14. 代码改造范围

### 14.1 Scheduler 核心

`csrc/nanodeploy/scheduler/scheduler.h/.cpp`

- 删除 fresh-arrival `_seal_ls_decode_arrivals()` 及 persistent pending 路径；
- 引入 request-level dispatch snapshot；
- 实现 cadence、bounded OOE、domain-level future envelope；
- 实现 Loong instance allocation plan；
- 接入二维 DP；
- commit 时才创建 batch ID/ownership；
- 支持 mixed admission/decode result；
- 收敛 Decode merge/scale-up；
- pause 时保留 generated progress。

### 14.2 DP 实现

建议新增：

```text
csrc/nanodeploy/scheduler/loong_batching_dp.h
csrc/nanodeploy/scheduler/loong_batching_dp.cpp
```

不直接链接或修改外部 LoongServe 代码。根据其 `main.cpp` 在 NanoDeploy 内重新实现，并增加 brute-force differential tests。

### 14.3 Placement/data plane

`csrc/nanodeploy/scheduler/sp_state_manager.*`

- packed token interval placement；
- pinned instance-range exact validation；
- adapter rejection reason；
- mixed macro iteration master plan；
- source-equivalent pause/readmission。

### 14.4 Python/config

- `nanodeploy/config.py`：source 参数及约束；
- `nanodeploy/engine/scheduler.py`：构造参数；
- `nanodeploy/engine/llm_engine.py`：mixed admission/decode 两条 lane；
- benchmark scripts：固定 source baseline 参数并写入 manifest。

## 15. 建议实施阶段与提交边界

### Phase 0：锁定 reference 和 profiler

产出：

- 固定 LoongServe commit `fb87896d...`；
- 决定 profiler 参数来源；
- 保存 profiler CSV/hash；
- 建立 source snapshot fixtures。

验收：同一 synthetic snapshot 可以输出 LoongServe 的 selected request IDs、batch ranges、DoPs 和 instance ranges。

### Phase 1：Request-level dispatch 和状态机替换

改动：

- 删除 fresh persistent pending batch；
- cadence；
- FCFS scan；
- bounded OOE；
- domain future-KV；
- commit 后才从 waiting 删除实际成员；
- 保留现有 exact allocation rollback。

该阶段可以暂时一次每 domain 只提交一个 source-selected batch，但不能产生长期 batch identity。此阶段结果仍标为 intermediate，不用于最终 baseline 性能结论。

### Phase 2：Instance allocation + 二维 DP + packed placement

改动：

- admission-triggered instance reallocation；
- exact Loong cost DP；
- backtracking；
- 多 batch/不同 DoP；
- packed intervals；
- Nano adapter rejection/replan。

完成此阶段后，Prefill dispatch/batching 决策才达到 source-aligned。

### Phase 3：Mixed macro iteration

改动：

- `ADMISSION_AND_DECODE`；
- 新请求 dummy Prefill lane；
- 旧请求 Decode lane；
- 独立 postprocess/metrics。

### Phase 4：Decode elasticity 和 pause 对齐

改动：

- token-deficit donor merge；
- precise idle-instance scale-up；
- compute threshold 100；
- 删除 baseline 中额外 compute merge；
- source-equivalent disable-scale-up；
- preserve-progress pause/readmission。

### Phase 5：清理和正式验收

- 删除临时 policy flags；
- 更新旧设计文档为 historical/superseded；
- 固化 benchmark manifest；
- 8/16 GPU 正式 A/B。

每个 Phase 单独提交，避免一个大 commit 同时重写控制面和数据面。

## 16. 必须保持的不变量

1. waiting queue membership 只在 admission commit 成功后改变。
2. OOE 只能由 source counter允许，不能由 pending bypass 隐式产生。
3. 同一 request 在 waiting、planned、running、paused 中只能有一个权威状态。
4. Domain-level future envelope必须包含该 domain 全部 running和本轮 tentative requests。
5. DP 输出的 request ranges 和 instance ranges 都是连续区间。
6. 同一 macro plan 中 batch instance ranges 不重叠，除非 source instance-allocation 阶段明确安排与 Decode batch共享并随后 merge。
7. Nano exact gate 只能 reject/replan，不能静默改变 Loong DoP。
8. physical allocation 失败必须完整 rollback。
9. paused request 保留 generated tokens和采样进度。
10. baseline 不执行未在 LoongServe 源码出现的 gap、age、utilization heuristic。

## 17. Telemetry

新增结构化事件：

### `ls_dispatch_scan`

- FIFO snapshot IDs；
- scanned/selected/deferred IDs；
- 每个 deferred reason；
- OOE before/after；
- running request count；
- batch token sum；
- per-domain future peak/capacity。

### `ls_instance_allocation_plan`

- running batch/rank usage；
- idle instances；
- migrated token count；
- paused/continued Decode batches；
- profiler speedup/slowdown/migration cost。

### `ls_batching_dp_plan`

- sorted prompt lengths；
- sorted instance used/free tokens；
- selected request ranges；
- instance ranges；
- DoP；
- predicted batch/total cost；
- adapter-rejected transitions。

### `ls_macro_iteration`

- admission-only/decode-only/mixed；
- new batch IDs；
- concurrent Decode group IDs；
- merge/scale-up/scale-down/pause decisions。

日志默认做 summary，不把完整 DP table写入长跑日志；测试/debug模式允许展开。

## 18. 测试计划

### 18.1 CPU source-conformance tests

1. **Dispatch differential**：固定 queue/running/capacity/profiler snapshot，对比 LoongServe reference selected IDs。
2. **Bounded OOE**：连续越序十轮后，前部 blocker成为 frontier；恢复可行后 counter reset。
3. **Global future-KV**：两个 groups 单独均可、合计不可时，第二组 request不得 admission。
4. **Cadence**：idle immediate；running 时十个 Decode iterations 后 admission。
5. **DP differential**：小规模随机 `n,m` 与 brute-force枚举比较最优 cost、ranges、DoPs。
6. **Variable batching**：DP1×SP8 也能产生多个 batch，证明 batch 数不再绑定 attention DP。
7. **Packed placement**：interval无重叠/缺口，token总数精确，容量不超限。
8. **Adapter rejection**：receiver/block gate拒绝 transition 后重新 DP，不回退到最小 d heuristic。
9. **Atomic rollback**：allocation/publication任意注入点失败，waiting、blocks、ownership完全恢复。
10. **Pause progress**：pause/readmission 后保留 prompt+generated，不从 prompt重新开始。
11. **Post-Prefill regroup**：临时 pending Decode batch按 instance intersection合入 new batch，disjoint batch保持独立。
12. **Feature-off**：非 LS scheduler不受影响。

### 18.2 Issue 1% trace fixtures

至少固定以下窗口：

```text
[276, 199, 923230, 229, 197, 213, 227]
[170, 960909, 199]
[971548, 198, 215, 277]
[228, 921913, 217, 286]
[941787, 199, 198, 214]
[205, 130, 214, 198, 840341, 201]
```

断言：

- 当前系统 future no-fit 的 long request保持 waiting；
- bounded OOE允许后续 short requests按源码规则执行；
- short request不因历史 persistent batch identity陪等；
- 没有 request丢失、重复或跨状态 ownership。

### 18.3 GPU 验收

8 GPU：

- DP1×SP8；
- mixed short/medium/900K prompts；
- 验证多 batch、多 DoP、packed placement、mixed macro iteration；
- 全部输出长度正确，无 planner failure/preemption/rollback。

16 GPU：

- DP2×SP8 / EP16；
- Issue 1%，rate 20，seed 0，7,200 requests；
- 先 141 GiB，再 140 GiB；
- baseline consolidation off；增强项单独开启。

## 19. Baseline 完成标准

以下全部满足后，才能在实验中称为 LoongServe-style baseline：

1. synthetic source snapshots 的 request membership 与 LoongServe一致；
2. batching DP 的 ranges/DoPs/instance ranges 与 reference或 brute-force一致；
3. profiler参数已固定并记录 hash；
4. resolved parameter profile已完整打印，所有非 source-default值有 provenance/adaptation reason；
5. fresh requests不存在长期 persistent pending batch；
6. bounded OOE 和 `max_wait_tokens` cadence生效；
7. future-KV是 DP-domain全局 envelope，无跨 group重复承诺；
8. prompt KV使用 Loong packed placement；
9. macro iteration支持 admission+Decode并行语义；
10. Prefill 后的 Decode group重组与源码一致；
11. Decode memory/compute scale-up触发与源码对齐；
12. Nano 自定义 consolidation、gap heuristic、compute-pressure merge不在 base path；
13. pause/readmission保留生成进度；
14. 8/16 GPU correctness验收通过。

性能指标不作为“是否对齐”的替代品。吞吐或延迟改善不能证明调度决策与 LoongServe一致，必须同时保存 source-conformance telemetry。

## 20. 会议需要拍板的问题

| 决策项 | 选项 | 建议 |
|---|---|---|
| Baseline 定义 | 完整 Loong policy / 仅 Decode scale-up | 完整 policy；否则名称改为 LS-Decode-Core |
| Profiler 来源 | Loong 原 profile / Nano H200 profile / synthetic | 优先 Nano H200 profile；不可用时固定 Loong profile并声明 |
| Nano DP 映射 | 每 domain 独立 pool / 修改执行拓扑形成全局 pool | 每 domain 独立 pool + 全局 FIFO dispatcher |
| Persistent batch | 保留 / 删除 | fresh arrival删除，commit事务原子性保留 |
| Macro iteration | admission-only / mixed admission+Decode | mixed |
| Custom consolidation | 进入 base / 单独增强 | 单独增强，不进入 base |
| 参数 profile | API defaults / artifact-derived / Nano workload adaptation | 三者都保留 provenance；正式 Issue 1% 冻结单一 resolved profile |
| Source bug | bitwise复刻 / 设计意图 | 设计意图，已知 bug显式修正 |
| 旧 policy 开关 | 长期双路径 / commit回溯 | 最终只保留 source-aligned路径，A/B使用 commit |

## 21. 推荐会议决议

建议会议批准以下实施口径：

1. 旧 persistent atomic pending batch设计不再约束 baseline；
2. 按 Phase 0--5 完成 source-first重构；
3. 先锁定 profiler和 topology mapping，再动 batching代码；
4. fresh admission改为 request-level waiting + bounded OOE；
5. 完整移植二维 DP和 packed placement，不使用 gap heuristic；
6. Nano exact placement/rollback作为 adapter保留；
7. baseline关闭 Nano custom consolidation，并单独报告增强版本；
8. 最终报告名称统一为：

   ```text
   LoongServe scheduling policy on NanoDeploy decode-only backend
   ```

这样做的改动会明显大于 `908933a`，但它能建立清晰、可审计的 baseline 边界。继续在现有 seal/pending状态机上叠加局部 heuristic，短期代码量较小，却无法回答实验评审中最关键的问题：运行的究竟是不是 LoongServe 调度策略。

## 22. 源码与本地证据索引

LoongServe：

- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/req_queue.py:46`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/req_queue.py:135`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/io_struct.py:116`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:351`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:516`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:686`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:764`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:844`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/api_server.py:375`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/longserve_c_scheduler/src/main.cpp:33`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/test/longserve/5-start-api-server.py:123`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/paper-tex-src/sections/design.tex:68`

NanoDeploy：

- `csrc/nanodeploy/scheduler/scheduler.cpp:924`
- `csrc/nanodeploy/scheduler/scheduler.cpp:1348`
- `csrc/nanodeploy/scheduler/scheduler.cpp:1488`
- `csrc/nanodeploy/scheduler/scheduler.cpp:1705`
- `csrc/nanodeploy/scheduler/scheduler.cpp:2085`
- `csrc/nanodeploy/scheduler/scheduler.cpp:2368`
- `csrc/nanodeploy/scheduler/scheduler.cpp:2822`
- `csrc/nanodeploy/scheduler/sp_state_manager.cpp:488`

实验记录：

- `docs-dev/2026-07-17/ls_decode_future_kv_2node_r20_141gb_result_20260717.md`
- `docs-dev/2026-07-17/ls_decode_loongserve_capacity_alignment_20260717.md`
- `docs-dev/2026-07-18/ls_style_capacity_reorg_mem085_2node_result_20260718.md`
