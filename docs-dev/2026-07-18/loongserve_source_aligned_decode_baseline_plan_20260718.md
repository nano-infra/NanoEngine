# NanoDeploy LoongServe-style Decode-only baseline 改造方案（会议讨论稿）

日期：2026-07-18

最近修订：2026-07-19

状态：Proposal，已按 Decode-only 实验范围收敛

NanoDeploy 代码基线：`64a199154b375aba8cd469ca75542f4103ccbf64`

LoongServe 源码基线：`fb87896d87b170afd4afe591e29da1aa5f6d4e16`

## 0. 结论先行

本实验只研究 Decode。新请求进入系统时，scheduler 直接建立 prompt KV placement，并通过 dummy bootstrap 补齐进入 Decode 所需的状态；之后所有 GPU iteration 都是 Decode。

固定运行条件：

```text
mode = decode
dummy_prefill = true
loop_count = 1
```

`dummy_prefill` 只负责初始化 prompt KV、追加 pending token 和记录首 token 状态，不参与 batching cost，也不引入固定 admission 延迟。

因此，本方案不设计任何其他执行阶段，不引入与其他阶段有关的 cost model、资源竞争、并发执行接口或阶段切换逻辑。

baseline 的准确名称应为：

```text
LoongServe-style Decode-only scheduler on NanoDeploy
```

它对齐 LoongServe 中与当前实验直接相关的设计：

1. request round-robin 分配到独立 DP pool；
2. 每个 pool 内 waiting queue 的 FIFO 顺序和有限越序；
3. request-level current/future KV admission；
4. 选中后按 prompt length 降序；
5. 连续 request range 形成 batch；
6. instance 按已用 token 排序；
7. capacity-aware initial DoP 和 KV placement；
8. 低 KV 利用率时的 pool-local consolidation；
9. 运行中 Decode 的 memory-deficit merge；
10. 运行中 Decode 的 compute-bound scale-up；
11. pause/readmission 保留生成进度。

第一版不迁移 LoongServe 原二维 DP 的 cost 部分。当前环境没有与原目标函数对应的运行时 cost，强行加载一组无关参数反而会让 baseline 难以解释。第一版采用：

```text
request round-robin 固定到 DP pool
        -> pool-local FIFO 有限越序选择
        -> current/future KV 检查
        -> prompt length 稳定降序后做 Nano exact planning
        -> no-fit 时按 FIFO selection order 回退并重新 planning
        -> 每 pool 一个连续候选 batch
        -> LoongServe-style packed interval placement
        -> Nano block/metadata exact adapter + allocation/rollback transaction
        -> source-shaped Decode merge/scale-up
```

这比当前实现更贴近 LoongServe，同时将改动限制在 Decode scheduler。若会议要求二维 DP，再增加一个明确命名的 Decode-cost 版本，不能把它静默混入基础组。

## 1. Baseline 边界

### 1.1 实验状态模型

Fresh request 的稳定状态只有：

```text
WAITING_REQUEST(assigned_dp)
    -> RUNNING_DECODE_REQUEST
    -> FINISHED
```

必要的临时状态只存在于一次 `schedule()` 调用内：

```text
DispatchCandidate
DecodeBatchPlan
AdmissionTransaction
```

其中：

- `DispatchCandidate` 没有 batch ID；
- `DecodeBatchPlan` 没有持久 ownership；
- request 进入 waiting 前已经取得不可变的 `assigned_dp`；
- 只有 admission transaction commit 后才创建 batch/group ownership；
- planning 或 allocation 失败时，request 仍留在所属 pool 的 waiting queue。

### 1.2 对齐范围

需要对齐的 LoongServe 源码位置：

- request future-capacity check：`req_queue.py:46-76`；
- waiting queue scan 与有限越序：`req_queue.py:80-227`；
- request/instance 排序与连续 batch range：`manager.py:686-750`；
- 二维 DP 的状态和回溯形状：`longserve_c_scheduler/src/main.cpp:33-84`；
- packed token interval placement：`manager.py:764-800`；
- 新请求到达时压紧低占用 instances：`manager.py:516-680`；
- Decode memory/compute elasticity：`manager.py:844-970`。

### 1.3 明确不做的内容

本 baseline 不包含：

- 非 Decode kernel 或执行路径；
- 与非 Decode 工作有关的 latency model；
- 跨执行阶段的资源比较和抢占；
- 两条计算 lane 的联合调度；
- 与当前 Nano topology 无关的全局通信重构；
- Nano 自定义 gap、age 或 planner-failure group merge heuristic。

这部分不是“以后补齐 baseline”的隐含任务，而是当前实验定义之外的内容。

### 1.4 名称约束

在论文、图例和日志中统一使用：

```text
LoongServe-style Decode-only
```

不要使用：

```text
LoongServe
Full LoongServe
LoongServe source-identical
```

原因是 Nano 的 DP topology、block allocator 和执行接口不同，而且第一版不使用原二维 DP cost。

## 2. 当前 Nano 与目标 baseline 的主要差异

| 环节 | Nano 当前实现 | Decode-only baseline 目标 | 处理方式 |
|---|---|---|---|
| DP assignment | admission 时按当前负载选择可行 DP | request 到达时 round-robin 固定 DP | 修改 |
| Admission opportunity | 每次 `schedule()` 都先尝试 admission，成功则不 Decode | 每个 scheduler step 都可 scan，但成功 admission 不抑制已有请求 Decode | 修改返回结果 |
| Queue scan | 一个全局 queue，固定截取 prefix | 每个 pool 独立 FIFO + bounded OOE + current/future KV | 修改 |
| Fresh batch identity | current-fit 前创建长期 `PendingDecodeBatch` | exact plan 成功前保持 request-level waiting | 修改 |
| Request ordering | 选中窗口内已按长度降序 | 保留 | 已完成 |
| Batch partition | 全局窗口按 `attention_dp` 均分，no-fit 时缩短 | 每个 pool 每轮一个 ephemeral continuous batch | 修改 |
| Batch count/DoP | batch 数绑定 `attention_dp`，DoP 取第一个可行值 | 每 pool 每轮最多一个 batch；DoP 取本 pool 第一个可行值 | 明确近似 |
| Future-KV | candidate + 单个 target group | 所属 SP8 pool 全部 running + tentative envelope | 修改 |
| Prompt KV placement | request 在 ranks 上均匀 striping | packed intervals，再转换为 Nano blocks/metadata | 与 batching 同步修改 |
| Admission/Decode | admission 成功会让已有请求少跑一次 Decode | bootstrap 不应吞掉已有请求的 Decode iteration | 修改 |
| Scale-down | utilization 或 pending benefit 都可能触发 consolidation | 只由低 KV 利用率产生 candidate，stable/cooldown 仅防抖 | 简化触发条件 |
| Memory scale-up | 多种 planner failure 都可能触发 merge | 只按 Decode token deficit merge/加 rank | 修改 |
| Compute scale-up | threshold 与 arbitrary group merge 混合 | 只消费 idle ranks，不强并健康 group | 修改 |
| Preemption | 丢弃 generated tokens 后重启 | 保留 output progress 再 admission | 修改 |

当前关键代码：

- admission-first：`csrc/nanodeploy/scheduler/scheduler.cpp:924-985`；
- group-local future-KV：`scheduler.cpp:1348-1485`；
- initial DoP/均匀 placement：`scheduler.cpp:1488-1635`；
- 排序和 persistent seal：`scheduler.cpp:1705-1810`；
- pending admission：`scheduler.cpp:2085-2365`；
- Decode merge/scale-up：`scheduler.cpp:2368-2598`；
- restart-from-prompt：`scheduler.cpp:2822-2900`。

## 3. 为什么要先删除 persistent fresh batch

### 3.1 当前问题不是缺少排序

Issue 1% 轨迹中的窗口：

```text
FIFO:   [276, 199, 923230, 229, 197, 213, 227]
排序后: [923230, 276, 229, 227, 213, 199, 197]
当前切分:
  B0 = [923230, 276, 229, 227]
  B1 = [213, 199, 197]
```

排序改变了成员顺序，但一旦 `B0` 被 seal，三个短请求仍会和 923K request 长期绑定。后续资源即使适合短请求，也不能重新组合。

### 3.2 已观测到的状态机放大

141 GiB 运行出现：

- 七个超长 request 连带九个短 request 等待；
- 单 batch admission 尝试最多 4,406 次。

140 GiB / memory fraction 0.85 运行完成全部 7,200 requests，但仍出现：

- pending 峰值 `106 batches / 300 requests`；
- oldest pending 等待 4,255 scheduler steps；
- 单 batch 尝试最多 4,256 次；
- scheduler overhead P95 `115.38 ms`；
- queueing P99 `140.84 s`。

因此必须改变 batch identity 的创建时机，而不是继续给 persistent queue 增加 bypass heuristic。

## 4. Request dispatch

### 4.1 Arrival 时固定所属 pool

Fresh request 到达 scheduler 时立即 round-robin 分配：

```text
assigned_dp = next_dp_rr
next_dp_rr = (next_dp_rr + 1) % attention_dp
waiting_by_dp[assigned_dp].push_back(request)
```

以 `4DP×8SP` 为例，请求分配顺序为：

```text
R0 -> DP0
R1 -> DP1
R2 -> DP2
R3 -> DP3
R4 -> DP0
...
```

assignment 完成后，request 不因其他 pool 更空闲而重新路由。pause/readmission 也回到原 `assigned_dp`。这对应 LoongServe artifact 使用外部 round-robin proxy 将请求固定到独立 worker，而不是 Nano 自定义的 admission-time load balancing。

### 4.2 每个 scheduler step 都允许 admission

Decode-only baseline 不使用固定 step 间隔限制 waiting scan。

规则是：

```text
for dp in [0, attention_dp):
    if waiting_by_dp[dp] 非空:
        本 scheduler step 执行该 pool 的 request-level admission scan

if admission 成功且已有 running requests:
    commit 新 requests
    同时返回已有 requests 的本轮 Decode plan

if admission 成功且系统原本 idle:
    commit 新 requests
    下一 scheduler step 开始 Decode
```

dummy bootstrap 只有 scheduler/KV 状态更新，没有需要用固定间隔摊销的 GPU 工作。照搬 10-step gate 会人为增加 queueing latency，并改变 arrival rate 实验的负载形状。

第一版直接每 step scan，先保证语义简单。若 telemetry 证明 scan 开销仍高，可以增加 correctness-preserving event cache：只有 new arrival，或者 request finish、pause/readmission、rank ownership 等可能让可行性改善的资源事件发生时，才重新执行 expensive exact planning。该优化不能延迟一个本来已经可 admission 的 request。

### 4.3 Pool-local FIFO scan 和 bounded OOE

第一版扫描：

```text
for request in waiting_by_dp[dp] FIFO order:
    检查 running request 数量
    检查本轮 admission token 上限
    检查 current KV capacity
    检查 pool-wide future-KV

    if feasible:
        selected.push(request)
    else if allow_ooe:
        deferred.push(request)
        继续扫描
    else:
        frontier = request
        停止扫描
```

使用源码同名语义：

```text
num_ooe
max_num_ooe
```

每个 pool 独立维护 `num_ooe[dp]`。它表示该 pool 连续多少个 admission round 真正让后项越过了前项，不是单轮跳过 request 的数量。

规则：

1. 本轮没有越序并成功接纳 FIFO frontier 时，`num_ooe=0`；
2. 本轮有后项越过 blocker 并成功接纳时，`num_ooe++`；
3. `num_ooe >= max_num_ooe` 时禁止继续越序；
4. no-fit 且没有 request 被接纳时，不伪造 reset；
5. aborted request 从 queue 安全移除，不计 OOE。

LoongServe API 默认 `max_num_ooe=10`，artifact 会按 dataset 使用其他值。Nano 正式 workload 必须在 manifest 记录每个 pool 共用的 resolved limit，以及各 pool 的实时 counter。

### 4.4 第一版不保留 cost-driven undecided list

LoongServe queue scan 中还有一段由运行时间比较控制的 undecided-prefix 逻辑。当前实验不使用对应 cost，第一版不照搬该分支，也不伪造替代参数。

Decode-only 规则只有：

```text
capacity feasible -> selected
capacity infeasible -> bounded defer 或停止
```

这是一项 intentional adaptation，必须出现在 baseline manifest 和论文方法说明中。

### 4.5 Pool-local candidate window 上限

仍保留有限 planning window，避免单次 scheduler 扫描无界增长：

```text
max_selected_requests_per_pool = max_num_seqs
max_selected_tokens_per_pool   = ls_admission_max_tokens_per_pool
```

建议：

```text
ls_admission_max_tokens_per_pool =
    max(max_req_total_len, total_pool_kv_tokens / 6)
```

window 只限制本轮候选，不创建长期 batch membership。

## 5. Future-KV admission

### 5.1 Request envelope

对 `ignore_eos=True` workload，按 request 状态构造：

```text
RUNNING:            held = prompt + generated
                    remaining = max_output - generated - 1

WAITING:            held = prompt + 1
                    remaining = max_output - 2

PAUSED_OFFLOAD:     held = prompt + generated + 1
                    remaining = max_output - generated - 2

PAUSED_KVKEEP:      held = prompt + generated
                    remaining = max_output - generated - 1
```

所有 `remaining` 取 `max(0, remaining)`。

对一组 requests，继续使用 LoongServe-style high-water mark：

```text
按 remaining 降序
peak = max_i(prefix_held(i) + i * remaining_i)
```

### 5.2 检查域是所属 SP8 pool 全局

每个 Nano DP domain 映射成一个固定 SP8 elastic pool。该 pool 的 envelope 包含：

```text
该 pool 全部 running requests
+ 该 pool 的 paused-and-KVKEEP requests
+ 本轮该 pool 已 selected/tentative requests
+ 当前检查的 request
```

容量为该 pool 的 8 个 SP ranks 可用于该 workload 的 KV token slots，扣除固定占用和不可迁移 reservation。

以 `4DP×8SP` 为例，独立计算四份 envelope：

```text
DP0 pool = {DP0-SP0 ... DP0-SP7}
DP1 pool = {DP1-SP0 ... DP1-SP7}
DP2 pool = {DP2-SP0 ... DP2-SP7}
DP3 pool = {DP3-SP0 ... DP3-SP7}
```

DP0 request 只进入 DP0 envelope。即使 DP1 有大量空闲 KV，DP0 request 也不会在 admission 或 scale-up 时使用 DP1 capacity。

当前 `candidate + one target group` 的局部检查会让多个 group 重复承诺同一批未来 rank capacity，必须替换。

这里的“pool 全局”是指同一 SP8 pool 内所有 groups 联合计算，不是把 4 个 DP pool 合成一个 SP32 pool。

### 5.3 Nano exact safety gate

future envelope 是 token-level policy check。之后继续使用 Nano exact gate：

- 64-token block rounding；
- dummy pending-token tail block；
- receiver metadata；
- master metadata；
- `reserved_blocks_per_req`；
- 当前 ownership 和 rank 可达性。

exact gate 只能：

```text
accept
or reject and replan
```

如果需要缩小 membership，只能撤销 FIFO scan 中最后加入的 request，再重新排序和 exact planning；不能删除长度排序后的尾部 request。这样 Nano block/metadata 约束不会把“短 request 优先被移除”变成新的 batching policy。

## 6. Candidate 排序与连续 batching

### 6.1 LoongServe 原逻辑

一个独立 SP pool 内，LoongServe：

1. 按 waiting FIFO、容量和 bounded OOE 确定 selected membership；
2. selected requests 按长度稳定降序；
3. available instances 按已用 token 数升序；
4. 二维 DP 同时枚举最后一个 batch 的连续 request 数量和 instance 数量；
5. 用 aggregate free-token capacity 判断 transition 是否可行；
6. 回溯得到一个或多个 batches，以及每个 batch 的 DoP。

所以 LoongServe 不会按 DP 数量平均切 requests，也不会从长度排序后的尾部逐个删除 request。最终 selected set 由 DP plan 完整覆盖。

### 6.2 Nano 当前代码

当前 `_seal_ls_decode_arrivals()`：

1. 从全局 `waiting_migration` 固定截取 `attention_dp * max_num_seqs` 个 requests；
2. 按 prompt length 稳定降序；
3. 固定切成最多 `attention_dp` 个等数量 ranges；
4. 某个 range no-fit 时，从排序后的 range 尾部缩短；
5. empty-system fit 后立即创建长期 `PendingDecodeBatch`；
6. 后续 admission 再选择 DP 和第一个 exact feasible DoP。

这里有两个主要问题：batch 数和 DP 数绑定；排序后的短 requests 可能因为 shrink 被移出本轮 membership。后者意味着长度排序反过来改变了 FIFO 服务资格。

### 6.3 第一版改法

对每个独立 SP8 pool：

1. pool-local FIFO/OOE scan 产生 `selected_fifo`；
2. scan 过程中检查 current/future KV，避免选择明显不可运行的 request；
3. 对 `selected_fifo` 的拷贝做 prompt length 稳定降序；
4. 调用 empty-system 和 current-system Nano exact planning；
5. exact no-fit 时撤销 `selected_fifo` 中最后加入的 request，然后重新排序、重新 planning；
6. exact fit 后，排序结果整体形成一个 ephemeral batch；
7. placement/allocation 成功后才创建 batch/group ID 并修改 waiting ownership。

伪代码：

```text
selected_fifo = pool_local_fifo_ooe_scan(dp)

while not selected_fifo.empty():
    candidate = stable_sort_copy(selected_fifo, prompt_length_desc)

    if empty_system_fit(candidate)
       and current_system_exact_plan(candidate):
        commit(candidate)
        break

    selected_fifo.pop_back()  # 按 scan order 回退，不是 candidate.pop_back()
```

未提交 requests 始终保持所属 pool 的原 waiting order，不创建长期 pending batch。`4DP×8SP` 同一 scheduler step 最多由四个 pools 各提交一个相互独立的 batch。

### 6.4 第一版与 LoongServe 的已知差异

第一版每 pool 每轮只有一个 fresh batch，initial DoP 仍取 Nano 的第一个 exact feasible 值；LoongServe 可以通过二维 DP 在一个 pool 内生成多个 batches，并联合选择 batch boundaries 和 DoPs。

第一版不使用 gap、固定长度阈值、方差或其他替代 heuristic，也不声称复现完整 batching DP。若后续需要这部分，完整 DP 作为独立 variant 实现和消融，不在本节展开。

## 7. Instance ordering、initial DoP 和 placement

### 7.1 Instance ordering

每个 DP domain 的可用 ranks 按以下 stable key 排序：

```text
(used_kv_tokens ascending, node_id ascending)
```

同 key 时保持原 rank-pool 顺序。

当前代码按 free blocks 降序，本质上接近 used tokens 升序；应统一 telemetry 命名，避免一个地方记录 free、另一个地方记录 used 后无法做 differential check。

### 7.2 Initial DoP

第一版继续使用 Nano 现有的最小可行 DoP：

```text
for d = 1 .. attention_sp:
    取排序后前 d 个 ranks
    检查 future-KV
    生成 packed token intervals
    转换并检查 block/metadata/headroom
    第一个 exact feasible d 即为 initial DoP
```

这是 Decode-only adaptation，不等同于 LoongServe 二维 DP 联合选择的 DoP。必须记录：

```text
initial_dop_policy = min_exact_feasible
```

禁止通过非零 `ls_decode_initial_kv_dop` 在正式 baseline 中强制固定 DoP。

### 7.3 Placement 一次切换

batching 和 initial placement 在同一个 admission pipeline 中一次切换，不保留“新 batching + 旧均匀 striping”的中间 baseline。

这里要区分两种排序：

- **选择 ranks**：available ranks 按 used tokens 升序，initial DoP 取该顺序的前 `d` 个；
- **在 selected ranks 内写入**：selected ranks 按 used tokens 降序，优先填更满 rank 的剩余空间。

第二个顺序与 LoongServe `manager.py:764-800` 的 packed interval 逻辑一致。对每个候选 `d`，placement 流程为：

1. request prompt 按 batch 中已经确定的 request 次序依次写入；
2. 先填当前 rank 的全部剩余 token capacity；
3. 只有 request 跨越 capacity boundary 时，才把它分布到下一个 rank；
4. 得到每个 `(request, rank)` 的连续 token interval；
5. 将 intervals 转换为 Nano block counts，计入 block rounding、pending-token headroom、receiver/master metadata；
6. exact adapter 通过后，使用现有 block allocator 和 allocation/rollback transaction 原子提交。

因此，一次切换修改的是 placement **policy**，不是抛弃 Nano 的 allocation 数据面。现有 allocator、metadata 结构和 rollback transaction 继续作为 LoongServe-style intervals 的执行适配层。任一 `d` 的 adapter 校验失败就尝试下一个 `d`；全部失败时按第 6 节的 FIFO selection order 回退 request membership 并完整重算。

packed placement 会减少短 request 的 KV owner 数、receiver 数和 block rounding，是 Decode 路径中应与 batching 同时落地的 LoongServe 设计。

### 7.4 `4DP×8SP` 映射为四个独立 elastic pools

LoongServe 的一个 RouterManager 管理一个固定 `sp_world_size` pool；batch scale-up 扩大的是 `occupied_instances`，不会扩大该 pool 的 `sp_world_size`。LoongServe artifact 使用 DP 时，会启动多个独立 API workers，由外部 proxy round-robin 分流。

Nano 当前拓扑为：

```text
Attention DP{1,2,4} x SP8
```

baseline 固定映射：

```text
Nano DP0 -> Loong worker/pool 0 -> SP0...SP7
Nano DP1 -> Loong worker/pool 1 -> SP0...SP7
Nano DP2 -> Loong worker/pool 2 -> SP0...SP7
Nano DP3 -> Loong worker/pool 3 -> SP0...SP7
```

规则：

1. request arrival 时 round-robin 固定 `assigned_dp`；
2. 每个 pool 有独立 waiting queue、OOE counter 和 future-KV envelope；
3. 每个 pool 独立执行 batching、placement、merge 和 scale-up；
4. group 可以在本 pool 内从 DoP=1 扩到 DoP=8；
5. group 不能跨 DP merge，不能扩到 DoP=9...32；
6. pause/readmission 回原 pool；
7. baseline 禁止 admission-time load-aware rerouting 和跨 pool work stealing。

这意味着某个 pool 可能因长 request 阻塞，而另一个 pool 同时存在 idle ranks。基础组保留这一结果；若增加跨 pool rerouting，必须命名为单独的 Nano load-balancing enhancement。

如果未来要复现一个统一 32-instance pool，应使用 `DP1×SP32`，或者修改 Nano collective、KV ownership 和 group merge 以支持跨现有 DP domains。该工作不属于当前 baseline。

## 8. Admission transaction

### 8.1 Batch ID 创建边界

batch/group ID 只在以下条件全部满足后创建：

1. ephemeral continuous range 已确定；
2. 已在 `assigned_dp` pool 内确定 initial ranks；
3. exact placement validation 成功；
4. physical allocation 成功；
5. scheduler 准备发布 ownership。

禁止在 current-fit 之前写入：

```text
ls_pending_decode_batches_
ls_seq_to_batch_
next_ls_batch_id_
```

### 8.2 Atomic commit

保留现有安全不变量：

```text
一份已生成的 admission plan
要么全部 commit
要么 waiting/blocks/ownership 完全恢复
```

原子性不再表示：

```text
过去某个 scheduler step 形成的 request 集合
必须永久作为一个 batch 一起等待
```

### 8.3 Admission 不吞掉已有 Decode iteration

当前 dummy bootstrap 不运行 GPU model，因此成功 admission 不应让已有 running requests 少执行一次 Decode。

目标 `ScheduleResult` 可以同时包含：

```text
admitted_sequence_records
existing_running_decode_plan
```

执行顺序：

1. commit 新 request 的 KV 和 dummy bootstrap 状态；
2. 已有 running requests 正常执行本轮 Decode；
3. 新 admitted requests 从下一轮开始参加 Decode；
4. 系统原本 idle 时，本轮只做 admission，下一轮开始 Decode。

这里没有第二条 GPU 计算 lane，只是把 CPU/KV admission side effect 与已有 Decode plan 放在同一个 scheduler result 中。

## 9. 运行中 Decode elasticity

### 9.1 Memory deficit

对每个 running group 计算：

```text
idle_tokens = group_capacity
            - group_used_tokens
            - num_running_requests
```

然后按 LoongServe `manager.py:844-906`：

1. 分成 can-decode 和 cannot-decode groups；
2. cannot-decode group 优先吸收 idle token 最大的 can-decode group；
3. 合并后仍不足时，按精确 token deficit 增加 idle ranks；
4. 执行 group merge；
5. Nano block-level append slack 做最终 safety validation。

只有 capacity deficit 才触发这类 merge。

### 9.2 Compute bound

按 LoongServe `manager.py:910-969`：

```text
while remaining_requests / remaining_instances
      > min_comp_bound_decoding_batch_size
      and idle_ranks 非空:
    add one idle rank
```

LoongServe API default 为 100，artifact launcher 使用 128。两者都有 source provenance。

正式 Issue 1% baseline 建议冻结一个值，不在运行中自适应。当前正式实验已使用 128，可以继续作为 artifact-derived profile；64 和 8 作为独立 sensitivity，不混入 base。

### 9.3 低 KV 利用率 consolidation

保留一个简单的 pool-local scale-down policy：group 的 KV 利用率低于固定阈值时，尝试把最低占用 rank 上的 KV 搬到该 group 的其他 ranks，释放一个完整 rank。

```text
group_kv_util = sum(used_kv_blocks on participating ranks)
              / sum(usable_kv_blocks on participating ranks)

if group_kv_util < ls_kv_consolidation_candidate_util:
    source_rank = argmin(used_kv_blocks, used_kv_tokens, rank_id)
    exact-plan migrate(source_rank -> retained_ranks)
```

约束：

1. group 至少保留一个 participating rank；每次只尝试释放一个 rank，并且不能跨 SP8 pool；
2. source rank 不能承载无法安全迁移的 pending token/master 状态；
3. destination 必须通过 block capacity、metadata 和 headroom exact validation；
4. migration 和 group allocation 更新必须原子 commit/rollback；
5. exact no-fit 时不 consolidation，不改变 group ownership；
6. 释放后的 rank 才算 truly idle，之后可被 admission 或 compute scale-up 使用。

这保留了 LoongServe 优先压紧低占用 instances 的方向，但把源码中的收益比较简化成 KV utilization trigger，是本 Decode-only baseline 的显式 adaptation。

baseline 不再使用 fresh persistent pending batch 是否受益、waiting age 或 length gap 产生 consolidation candidate。stable window、cooldown 和 check interval 可以保留为固定的防抖/限频保护，但它们不能绕过低利用率条件，也不参与收益比较。

`ls_kv_consolidation_candidate_util` 在正式实验中固定，建议先沿用 Nano 当前默认值 `0.50`；防抖和 transport safety 参数也全部写入 manifest。

这里的 consolidation 是同一 group 内迁移 KV 并释放 rank，不是合并两个健康 groups。以下 group merge 仍然禁止：

- compute-pressure merge healthy group；
- planner-failure arbitrary merge；
- gap/age-driven group merge。

compute-bound path 只使用真正 idle ranks。所需 rank 被其他健康 group 占用时，不强制合并那个 group。

### 9.4 Scale-up disable

使用一个 source-shaped 开关：

```text
ls_disable_scale_up
```

它必须一致控制：

- capacity deficit 加 rank；
- compute-bound 加 rank；
- merge 后所需实例扩展；
- initial placement 可否承诺未来 idle ranks。

关闭后 capacity 不足应走 preserve-progress pause/readmission，不允许 compute path 绕过开关。

### 9.5 小 DoP group 是否主动合并

LoongServe 不会因为两个 group 都小就无条件合并成大 DoP。

Decode-only baseline 中只保留两种扩展触发：

```text
KV capacity deficit
or compute-bound 且存在 idle ranks
```

因此，仅完成 capacity-aware batching 不会自动得到“小 DoP group 合成大 DoP”的行为；也不应该额外补一个主动 merge heuristic。

## 10. Pause 和 readmission

当前 restart-from-prompt 会丢弃已经生成的 token，并改变服务量和排队状态，不适合作为 baseline。

目标：

```text
PAUSED_KVKEEP:
    保留 KV 和 generated progress
    first_router_need_tokens = 0

PAUSED_OFFLOAD:
    保留 generated progress
    readmission tokens = prompt + generated
```

paused request 回到原 `assigned_dp` 的 request-level waiting queue，接受该 pool 的 FIFO/OOE/future-KV scan。

不再创建优先级高于普通 request 的长期 singleton recovery batch。

## 11. 配置建议

建议 source-shaped 参数：

```text
ls_max_num_ooe = 10
ls_running_max_req_size = 1000
ls_admission_max_tokens_per_pool = auto
ls_min_comp_bound_decoding_batch_size = 128
ls_disable_scale_up = false
ls_decode_enable_future_kv_admission = true
ls_decode_initial_kv_dop = 0
ls_kv_consolidation_mode = execute
ls_kv_consolidation_candidate_util = 0.50
ls_kv_consolidation_target_high_watermark = 0.80
ls_kv_consolidation_stable_steps = 2
ls_kv_consolidation_cooldown_steps = 2
ls_kv_consolidation_check_interval_steps = 1
ls_kv_consolidation_max_source_blocks_per_event = 128
ls_kv_consolidation_migration_chunk_tokens = 64
ls_dp_assignment_policy = round_robin
ls_cross_dp_scale_up = false
```

其中：

- `ls_decode_initial_kv_dop=0` 表示最小 exact feasible；正式 baseline 禁止强制值；
- consolidation candidate 只由 `group_kv_util < 0.50` 产生，不能由 fresh waiting/pending benefit 绕过；
- `stable_steps/cooldown/check_interval` 只用于防抖和限频，`2/2/1` 是当前已测试的起始 profile，不是 LoongServe source parameter；
- destination high-watermark、source-block budget 和 migration chunk 属于 transport safety 参数；`0.80/128/64` 是当前已测试的起始 profile，正式实验必须显式冻结；
- `ls_dp_assignment_policy=round_robin` 在 request arrival 时固定 pool；
- `ls_cross_dp_scale_up=false` 是当前执行拓扑硬约束，正式 baseline 禁止覆盖；
- memory/compute scale-up 使用独立开关；`ls_kv_consolidation_mode` 只控制低利用率 scale-down；
- 所有 resolved values 写入运行 manifest。

建议固定三个配置 profile：

| Profile | `max_num_ooe` | Decode threshold | 用途 |
|---|---:|---:|---|
| `loong_decode_source_default` | 10 | 100 | API default conformance |
| `loong_decode_artifact_derived` | workload-specific | 128 | 对齐 artifact 参数形状 |
| `loong_decode_issue001` | 会议冻结 | 会议冻结 | Nano 正式 Issue 1% 实验 |

参数 profile 变化不代表算法变化，但一次运行不能隐式混用 config、benchmark 和 CLI 三套默认值。

## 12. 代码改造范围

### 12.1 `scheduler.h/.cpp`

- 删除 fresh request 的 persistent seal/pending ownership；
- arrival-time round-robin assignment 和 `waiting_by_dp`；
- 每个 scheduler step 扫描各 pool，并维护 pool-local OOE counters；
- request-level current/future scan；
- selected 后 stable length sort；
- ephemeral continuous partition；
- available-rank ordering、initial DoP search 和 packed token interval planning；
- current no-fit 时保持 waiting；
- commit 时才创建 batch/group ID；
- SP8 pool-wide future envelope；
- admission result 可携带已有 running Decode plan；
- low-KV-util consolidation candidate 和 pool-local transaction；
- Decode memory/compute scale-up 收敛到 source-shaped 规则；
- pause 保留 generated progress。

### 12.2 `sp_state_manager.*`

- 接收 packed token intervals；
- intervals 到 Nano block counts/receiver metadata 的 exact adapter；
- pending-token headroom 和 pinned rank-range validation；
- consolidation KV migration plan、exact destination validation 和 rollback；
- 复用现有 allocator 与 transaction/rollback；
- 增加 adapter rejection reason；
- 支持 admission side effect 与已有 Decode plan 共存。

### 12.3 Python/config

- `nanodeploy/config.py`：source-shaped Decode 参数和 profile；
- `nanodeploy/engine/scheduler.py`：构造参数；
- `nanodeploy/engine/llm_engine.py`：先处理 admission records，再运行已有 Decode plan；
- benchmark script：输出完整 resolved manifest。

### 12.4 不修改的内容

- model forward kernel；
- collective 实现；
- Ray topology；
- block size；
- batch ID 类型；
- 非 LS scheduler path；
- 外部 LoongServe 仓库。

## 13. 分阶段实施

### Phase 0：Reference fixtures

产出：

- 固定 LoongServe commit；
- 固定 Nano code snapshot；
- per-pool waiting/running/capacity synthetic snapshots；
- Issue 1% 六个长短混合窗口；
- resolved config manifest schema。

### Phase 1：Fresh-request admission pipeline 一次切换

改动：

- 删除 fresh persistent pending batch；
- arrival-time round-robin assignment；
- pool-local FIFO/OOE scan；
- current/future KV 先筛选 selected membership；
- stable length sort；
- exact no-fit 时按 FIFO scan 顺序回退 membership 并重新 planning；
- 每 pool 每轮一个 ephemeral continuous range；
- 每个 SP8 pool 独立的 future envelope；
- available-rank ordering 和 initial DoP search；
- packed token intervals；
- interval 到 blocks/metadata/headroom 的 exact adapter；
- 使用现有 allocator transaction 原子 commit/rollback；
- rollback 后恢复 waiting order；
- 消除跨 group future capacity 重复承诺；
- admission 不吞掉已有 Decode iteration；
- 完整 telemetry。

Phase 1 完成前不启用正式 baseline，也不保留“新 batching + 旧均匀 striping”的实验配置。实现过程可以拆成可回溯的小提交，但语义上只做一次切换。

### Phase 2：Decode elasticity

改动：

- memory-deficit donor merge；
- exact idle-rank scale-up；
- compute threshold；
- low-KV-util exact consolidation；
- 删除 baseline 中额外 merge；
- preserve-progress pause/readmission；
- source-shaped disable switch。

### Phase 3：验收和清理

- 删除临时双 policy；
- 旧 persistent-batch 文档标记 historical；
- 固定 8/16 GPU scripts；
- 完成 source-shaped CPU differential tests；
- 正式 Issue 1% A/B。

每个逻辑单元及时提交，方便回溯；只有 Phase 1 的完整 admission pipeline 通过 CPU fixtures 后，才整体启用 baseline path。

## 14. 不变量

1. request arrival 时 round-robin 固定 `assigned_dp`，之后不做 load-aware rerouting。
2. 每个 pool 的 selected membership 由本地 FIFO/OOE 决定，长度排序不能改变本轮服务资格。
3. request 只有 admission commit 成功后才能从所属 waiting queue 移除。
4. planning no-fit 不创建 batch/group ID。
5. 同一 request 只能处于 waiting、planned、running、paused、finished 之一。
6. OOE 只能由所属 pool 的 counter 允许，不能由 pending bypass 隐式产生。
7. future envelope 包含所属 pool 全部 running 和 tentative requests。
8. group DoP 不得超过 8，不能跨 DP pool merge/scale-up。
9. Nano exact gate 只能 reject/replan，不能静默改 DoP。
10. physical allocation 失败必须完整 rollback。
11. admission side effect 不减少已有 requests 的 Decode iteration 数。
12. memory merge 只由 capacity deficit 触发。
13. compute scale-up 只消费所属 pool 的 idle ranks。
14. pause/readmission 保留 generated tokens、采样进度和 `assigned_dp`。
15. consolidation candidate 只能由本 pool 的低 KV 利用率产生；fresh pending benefit、gap、age 和 arbitrary planner failure 不能触发 consolidation/group merge。
16. 非 LS scheduler 行为不变。

## 15. Telemetry

### `ls_decode_dp_assignment`

- request ID；
- arrival order；
- assigned DP/pool；
- round-robin counter before/after。

### `ls_decode_dispatch_scan`

- DP/pool ID；
- FIFO snapshot IDs；
- scanned/selected/deferred/frontier IDs；
- per-request reject reason；
- OOE before/after；
- selected prompt token sum；
- scan trigger 和是否命中 event cache。

### `ls_decode_batch_plan`

- FIFO-selected request IDs；
- membership rollback request IDs 及顺序；
- 最终 sorted request IDs/lengths；
- empty-system fit result；
- current-system fit result；
- committed members；
- uncommitted members。

### `ls_decode_future_kv`

- DP/pool ID；
- running/tentative request IDs；
- peak tokens；
- token capacity；
- exact block requirement；
- adapter reject reason。

### `ls_decode_initial_placement`

- available ranks 及 used-token 升序；
- 每个候选 `d` 的 selected ranks；
- selected ranks 的 used-token 降序 packing order；
- per-request token intervals；
- converted block counts 和 pending-token headroom；
- receiver/master metadata counts；
- exact reject reason 或 committed initial DoP。

### `ls_decode_consolidation`

- DP/pool 和 group ID；
- group KV utilization、threshold 和防抖状态；
- source/retained ranks 及 per-rank used blocks/tokens；
- migrated token/block counts；
- exact reject reason；
- transaction commit/rollback result；
- 最终释放的 truly idle rank。

### `ls_decode_iteration`

- group IDs；
- KV DoPs；
- master DoPs；
- used/free tokens；
- capacity-deficit groups；
- donor groups；
- added idle ranks；
- compute threshold decisions；
- pause/readmission IDs。

## 16. 测试计划

### 16.1 CPU tests

1. Round-robin assignment：连续 requests 映射为 DP0、DP1、DP2、DP3、DP0。
2. Assignment stability：no-fit、pause/readmission 不改变 `assigned_dp`。
3. Pool-local FIFO：可运行请求保持所属 queue 顺序。
4. Pool-local OOE：一个 pool 的 blocker/counter 不影响其他 pool。
5. OOE reset：本 pool frontier 成功运行后 counter 清零。
6. Every-step opportunity：new arrival 在下一个 scheduler step 即进入所属 pool scan。
7. Membership-before-sort：排序不改变 selected IDs。
8. Stable length order：等长 request 保持本 pool FIFO。
9. One batch per pool：一个 scheduler step 每 pool 最多提交一个 fresh batch。
10. Continuous range：每个 batch 都是本 pool 排序数组的连续 range。
11. FIFO-order rollback：exact no-fit 时撤销 FIFO scan 中最后加入的 request，而不是删除长度排序后的尾部 request；未提交 request 保持所属 waiting 原顺序。
12. No persistent identity：current no-fit 后不存在 batch ID/ownership。
13. Pool-wide future-KV：同 pool 两个 groups 单独可行、合计不可行时拒绝第二份承诺。
14. Pool isolation：DP0 full 不会借用 DP1 capacity，group DoP 最大为 8。
15. Packed ordering：rank selection 按 used tokens 升序；selected ranks 内 packing 按 used tokens 降序。
16. Packed intervals：request 只在 capacity boundary 上跨 rank，interval 无重叠、无缺口且总 token 数守恒。
17. Exact adapter：interval 转换后的 block/metadata/headroom reject 会触发 replan，不改 heuristic。
18. Atomic rollback：任意注入点失败后本 pool queue/blocks/ownership 完全恢复。
19. Admission continuity：已有 requests 不因新 admission 少一次 Decode。
20. Memory deficit merge：只选择本 pool donor，新增 rank 数符合 source-shaped 规则。
21. Compute scale-up：只使用本 pool idle ranks，不 merge healthy group。
22. Scale-up off：memory/compute 两条路径都不能绕过开关。
23. Low-util candidate：只有 `group_kv_util` 低于固定阈值才产生 consolidation candidate，并选择最低占用 source rank。
24. No pending bypass：高利用率 group 不因 fresh waiting request no-fit 而绕过 utilization threshold。
25. Consolidation exactness：destination block/metadata/headroom no-fit 时不迁移 KV，不改变 allocation。
26. Consolidation atomicity：注入 migration/commit failure 后 KV、blocks 和 group ranks 完整恢复。
27. Pause progress：readmission 后保留 prompt+generated 和 `assigned_dp`。
28. Feature-off：非 LS scheduler 不受影响。

### 16.2 Issue 1% fixtures

固定：

```text
[276, 199, 923230, 229, 197, 213, 227]
[170, 960909, 199]
[971548, 198, 215, 277]
[228, 921913, 217, 286]
[941787, 199, 198, 214]
[205, 130, 214, 198, 840341, 201]
```

这些 length 列表必须与原始 `arrival_index/request_id` 一起保存。测试先按全局 arrival order 执行 round-robin assignment，再分别构造四个 pool-local FIFO snapshots；不能直接把整段 length window 当成单个 pool 的 queue。

断言：

- arrival order 按 DP0、DP1、DP2、DP3 round-robin 分配；
- long current no-fit 时保持 request-level waiting；
- short request 只有在所属 pool 的 OOE 允许时越过；
- short request 不因历史 batch identity 陪等；
- request 不因其他 pool 空闲而改变 `assigned_dp`；
- 没有 request 丢失、重复或跨状态 ownership；
- 某 pool 的 `num_ooe` 达到上限后，本 pool FIFO blocker 成为 frontier。

### 16.3 GPU 验收

8 GPU：

- DP1×SP8；
- mixed 200/4K/100K/900K prompt；
- placement、append、memory/compute scale-up correctness；
- 输出长度正确，无 planner failure loop。

16 GPU：

- DP2×SP8 / EP16；
- Issue 1%，rate 20，seed 0，7,200 requests；
- 141 GiB 和 140 GiB；
- low-KV-util consolidation 开启，并保存 candidate/commit/rollback telemetry；
- source-default 和 artifact-derived threshold 分组运行。

32 GPU（正式使用 `4DP×8SP` 时）：

- 验证四个独立 SP8 pools；
- request assignment 严格 round-robin；
- 每个 group 的 KV/master DoP 均不超过 8；
- DP0 capacity pressure 不触发跨 DP migration/merge/scale-up；
- 四个 pool 的 queue、OOE 和 future-KV telemetry 可独立核对。

GPU 测试必须按仓库规定申请提权。

## 17. Baseline 完成标准

以下全部满足后，才能称为 LoongServe-style Decode-only baseline：

1. 运行中不存在 fresh persistent pending batch。
2. request arrival 按 round-robin 固定 `assigned_dp`。
3. 四个 pools 的 FIFO/OOE/every-step admission 状态相互独立。
4. request membership 在本 pool 长度排序之前确定。
5. 每 pool 每轮最多一个 batch，且是排序数组的连续 range。
6. no-fit 时 request 保持在所属 waiting queue，没有 batch ownership。
7. future-KV 使用所属 SP8 pool-wide envelope。
8. group 只能在本 pool 内 scale-up，DoP 不超过 8。
9. admission commit 保持 exact allocation/rollback。
10. 已有 requests 不因 admission 丢失 Decode iteration。
11. packed placement 与 batching 同步启用，并通过 interval/block/metadata correctness。
12. memory-deficit merge 与 compute idle-rank scale-up通过 differential fixtures。
13. low-KV-util exact consolidation 在 base path，fresh pending benefit 和 arbitrary merge 不在 base path。
14. pause/readmission 保留生成进度和 `assigned_dp`。
15. resolved config 和 intentional adaptations 写入 manifest。
16. 对应实验规模的 8/16/32 GPU correctness 完成。

性能改善不能替代调度一致性检查。正式实验必须同时保存 dispatch、batch、future-KV 和 Decode elasticity telemetry。

## 18. 会议需要拍板的问题

| 决策项 | 选项 | 建议 |
|---|---|---|
| Baseline 名称 | LoongServe / LoongServe-style Decode-only | 后者 |
| Persistent fresh batch | 保留 / 删除 | 删除 |
| Batching v1 | FIFO membership 回退 + stable sort + 每 pool 一个 batch / 新 length heuristic | 前者 |
| 二维 DP | 进入第一版 / Decode-cost 独立 variant | 独立 variant |
| Placement | 与 batching 同步切 packed / 保留均匀 striping 过渡阶段 | 已决定：同步切 packed，不保留过渡 baseline |
| Admission continuity | 保留 admission-only / 同 step 保持已有 Decode | 保持已有 Decode |
| Future-KV | group-local / SP8 pool-wide | SP8 pool-wide |
| Decode threshold | 100 / 128 | Issue 1% 主实验建议 128，100 做 source-default sensitivity |
| Consolidation trigger | low KV utilization / fresh pending benefit / 全部关闭 | 已决定：只用 low KV utilization |
| Pause | restart / preserve progress | preserve progress |
| DP topology | 统一 SP32 / 4 个独立 SP8 pools | 已决定：4 个独立 pools |
| DP assignment | admission-time load-aware / arrival-time round-robin | 建议 artifact-style round-robin；load-aware 只能作为增强项 |
| Cross-DP scale-up | 允许 / 禁止 | 已决定：禁止，单 group DoP≤8 |

## 19. 推荐会议决议

建议会议批准：

1. baseline 范围严格限定为 Decode-only；
2. 旧 persistent fresh-batch 设计不再约束实现；
3. Phase 1 一次完成 arrival-time round-robin、pool-local selection、FIFO-order membership 回退、stable sort、ephemeral batching、pool-wide future-KV、initial DoP 和 packed placement；
4. packed intervals 通过 Nano block/metadata adapter 接入现有 allocator transaction，不保留均匀 striping 的过渡 baseline；
5. Phase 1 同时保证 admission 不吞掉已有 Decode iteration；
6. Phase 2 对齐 Decode memory/compute elasticity，并保留 low-KV-util exact consolidation；
7. 二维 DP 不进入第一版，单列 Decode-cost variant；
8. `4DP×8SP` 固定为四个独立 pools，禁止 load-aware rerouting 和 cross-DP scale-up；
9. 所有 Nano topology adapter 和 source deviation 写入 manifest；
10. 正式实验统一命名为 `LoongServe-style Decode-only`。

这样可以避免把当前实验不执行的内容带入设计，同时保留 LoongServe 对 Decode 资源管理最关键的思想，也能控制每个阶段的改动面和验证成本。

## 20. 源码与本地证据索引

LoongServe：

- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/req_queue.py:46`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/req_queue.py:135`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:40`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:686`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:764`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:516`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:613`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:844`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:975`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/longserve_c_scheduler/src/main.cpp:33`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/paper-tex-src/sections/design.tex:68`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/test/longserve/5-start-api-server.py:261`

NanoDeploy：

- `csrc/nanodeploy/scheduler/scheduler.cpp:924`
- `csrc/nanodeploy/scheduler/scheduler.cpp:1348`
- `csrc/nanodeploy/scheduler/scheduler.cpp:1488`
- `csrc/nanodeploy/scheduler/scheduler.cpp:1705`
- `csrc/nanodeploy/scheduler/scheduler.cpp:641`
- `csrc/nanodeploy/scheduler/scheduler.cpp:1826`
- `csrc/nanodeploy/scheduler/scheduler.cpp:2085`
- `csrc/nanodeploy/scheduler/scheduler.cpp:2368`
- `csrc/nanodeploy/scheduler/scheduler.cpp:2822`
- `csrc/nanodeploy/scheduler/sp_state_manager.cpp:488`

实验记录：

- `docs-dev/2026-07-17/ls_decode_future_kv_2node_r20_141gb_result_20260717.md`
- `docs-dev/2026-07-17/ls_decode_loongserve_capacity_alignment_20260717.md`
- `docs-dev/2026-07-18/ls_style_capacity_reorg_mem085_2node_result_20260718.md`
