# NanoDeploy LoongServe-style Decode-only baseline 改造方案（会议讨论稿）

日期：2026-07-18

状态：Proposal，已按 Decode-only 实验范围收敛

NanoDeploy 代码基线：`64a199154b375aba8cd469ca75542f4103ccbf64`

LoongServe 源码基线：`fb87896d87b170afd4afe591e29da1aa5f6d4e16`

## 0. 结论先行

本实验只研究 Decode。新请求进入系统时，scheduler 直接建立 prompt KV placement，并通过 dummy bootstrap 补齐进入 Decode 所需的状态；之后所有 GPU iteration 都是 Decode。

因此，本方案不设计任何其他执行阶段，不引入与其他阶段有关的 cost model、资源竞争、并发执行接口或阶段切换逻辑。

baseline 的准确名称应为：

```text
LoongServe-style Decode-only scheduler on NanoDeploy
```

它对齐 LoongServe 中与当前实验直接相关的设计：

1. waiting queue 的 FIFO 顺序和有限越序；
2. request-level current/future KV admission；
3. 选中后按 prompt length 降序；
4. 连续 request range 形成 batch；
5. instance 按已用 token 排序；
6. capacity-aware initial DoP 和 KV placement；
7. 运行中 Decode 的 memory-deficit merge；
8. 运行中 Decode 的 compute-bound scale-up；
9. pause/readmission 保留生成进度。

第一版不迁移 LoongServe 原二维 DP 的 cost 部分。当前环境没有与原目标函数对应的运行时 cost，强行加载一组无关参数反而会让 baseline 难以解释。第一版采用：

```text
FIFO 有限越序选择
        -> prompt length 稳定降序
        -> 连续等数量初始切分
        -> exact capacity 检查并缩短
        -> 现有 placement/admission transaction
        -> source-shaped Decode merge/scale-up
```

这比当前实现更贴近 LoongServe，同时将改动限制在 Decode scheduler。若会议要求二维 DP，再增加一个明确命名的 Decode-cost 版本，不能把它静默混入基础组。

## 1. Baseline 边界

### 1.1 实验状态模型

Fresh request 的稳定状态只有：

```text
WAITING_REQUEST
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
- 只有 admission transaction commit 后才创建 batch/group ownership；
- planning 或 allocation 失败时，request 仍留在 waiting queue。

### 1.2 对齐范围

需要对齐的 LoongServe 源码位置：

- request future-capacity check：`req_queue.py:46-76`；
- waiting queue scan 与有限越序：`req_queue.py:80-227`；
- admission cadence：`manager.py:351-373`；
- request/instance 排序与连续 batch range：`manager.py:686-750`；
- 二维 DP 的状态和回溯形状：`longserve_c_scheduler/src/main.cpp:33-84`；
- packed token interval placement：`manager.py:764-800`；
- Decode memory/compute elasticity：`manager.py:844-970`。

### 1.3 明确不做的内容

本 baseline 不包含：

- 非 Decode kernel 或执行路径；
- 与非 Decode 工作有关的 latency model；
- 跨执行阶段的资源比较和抢占；
- 两条计算 lane 的联合调度；
- 与当前 Nano topology 无关的全局通信重构；
- Nano 自定义 gap、age、utilization 或 planner-failure merge heuristic。

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
| Admission cadence | 每次 `schedule()` 都先尝试 admission | idle 时立即；busy 时按 Decode step 周期检查 | 修改 |
| Queue scan | 固定截取 queue prefix | FIFO scan + bounded OOE + current/future KV | 修改 |
| Fresh batch identity | current-fit 前创建长期 `PendingDecodeBatch` | exact plan 成功前保持 request-level waiting | 修改 |
| Request ordering | 选中窗口内已按长度降序 | 保留 | 已完成 |
| Batch partition | 按请求数均分，no-fit 时缩短 | 第一版保留连续均分形状，但在 ephemeral plan 内完成 | 移动时机 |
| Batch count/DoP | batch 数绑定 `attention_dp`，DoP 取第一个可行值 | 第一版作为明确近似；二维 DP 单列扩展 | 记录偏差 |
| Future-KV | candidate + 单个 target group | DP-domain 全局 running + tentative envelope | 修改 |
| Prompt KV placement | request 在 ranks 上均匀 striping | 最终目标为 packed intervals | 分阶段修改 |
| Admission/Decode | admission 成功会让已有请求少跑一次 Decode | bootstrap 不应吞掉已有请求的 Decode iteration | 修改 |
| Scale-down | Nano utilization/stability/cooldown consolidation | base path 关闭 | 修改配置/入口 |
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

### 4.1 Admission opportunity

新增：

```text
decode_steps_since_last_admission
```

只有以下情况执行 waiting scan：

```text
running 为空
or decode_steps_since_last_admission >= ls_max_wait_tokens
```

建议 source-default：

```text
ls_max_wait_tokens = 10
```

counter 只在成功完成真实 Decode iteration 后递增；成功 admission 后清零。idle 系统不等待 counter。

### 4.2 FIFO scan 和 bounded OOE

第一版扫描：

```text
for request in waiting FIFO order:
    检查 running request 数量
    检查本轮 admission token 上限
    检查 current KV capacity
    检查 domain-level future-KV

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

`num_ooe` 表示连续多少个 admission round 真正让后项越过了前项，不是单轮跳过 request 的数量。

规则：

1. 本轮没有越序并成功接纳 FIFO frontier 时，`num_ooe=0`；
2. 本轮有后项越过 blocker 并成功接纳时，`num_ooe++`；
3. `num_ooe >= max_num_ooe` 时禁止继续越序；
4. no-fit 且没有 request 被接纳时，不伪造 reset；
5. aborted request 从 queue 安全移除，不计 OOE。

LoongServe API 默认 `max_num_ooe=10`，artifact 会按 dataset 使用其他值。Nano 正式 workload 必须在 manifest 记录 resolved value。

### 4.3 第一版不保留 cost-driven undecided list

LoongServe queue scan 中还有一段由运行时间比较控制的 undecided-prefix 逻辑。当前实验不使用对应 cost，第一版不照搬该分支，也不伪造替代参数。

Decode-only 规则只有：

```text
capacity feasible -> selected
capacity infeasible -> bounded defer 或停止
```

这是一项 intentional adaptation，必须出现在 baseline manifest 和论文方法说明中。

### 4.4 Candidate window 上限

仍保留有限 planning window，避免单次 scheduler 扫描无界增长：

```text
max_selected_requests = attention_dp * max_num_seqs
max_selected_tokens   = ls_admission_max_tokens
```

建议：

```text
ls_admission_max_tokens = max(max_req_total_len, total_domain_kv_tokens / 6)
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

### 5.2 检查域必须是 DP-domain 全局

每个 DP domain 的 envelope 包含：

```text
该 domain 全部 running requests
+ paused-and-KVKEEP requests
+ 本轮已 tentative 分配到该 domain 的 requests
+ 当前检查的 request
```

容量为该 domain 全部可用于该 workload 的 KV token slots，扣除固定占用和不可迁移 reservation。

当前 `candidate + one target group` 的局部检查会让多个 group 重复承诺同一批未来 rank capacity，必须替换。

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

不能在 reject 后静默换成另一种 batch/DoP heuristic。

## 6. Candidate 排序与连续 batching

### 6.1 顺序

必须先决定 selected membership，再排序：

```text
waiting FIFO scan
    -> selected request IDs
    -> stable sort by num_prompt_tokens descending
```

不能先对整个 waiting queue 排序，否则会破坏全局公平性。

等长 requests 保持原 FIFO 次序。

### 6.2 第一版切分算法

第一版保留最小改动的连续切分：

```text
n = selected.size
num_batches = min(attention_dp, n)
target sizes = quotient/remainder balanced split
```

对排序后的每个连续 target range：

1. 调用 `_ls_batch_fits_empty_system(candidate)`；
2. no-fit 时从 range 尾部逐个缩短；
3. 获得 empty-system-feasible prefix 后，进入 current-system exact planning；
4. current no-fit 时不 seal，不创建 batch ID；
5. 未提交成员全部回到原 waiting order；
6. 后续 request 能否越过由 bounded OOE 决定。

这里“缩短 range”只发生在本次 ephemeral plan 中，不能生成一个长期等待的缩小 batch。

### 6.3 为什么第一版仍使用等数量切分

原因是当前没有可解释的 LoongServe batching cost。下面几种做法都不是源码逻辑：

- 最大长度 gap；
- 固定 long/short threshold；
- batch 内长度比例阈值；
- age boost；
- 人工最小化方差。

在没有 cost 的情况下，保留现有 deterministic partition 比引入新的 length heuristic 更适合作为 baseline。

它的局限也要明确：长度排序加等数量切分不能保证超长 request 一定单独成 batch。因此第一版名称只能是 LoongServe-style Decode-only。

### 6.4 二维 DP 的处理

LoongServe 原 DP 的结构仍作为 source-conformance reference：

```text
f[i][k] = 前 i 个排序后 requests 使用前 k 个排序后 instances 的最优值
```

转移仍是：

```text
最后一个 batch 使用 b 个连续 requests
最后一个 batch 使用 d 个连续 instances
```

容量仍是：

```text
sum(request tokens) <= sum(instance free tokens)
```

但第一版不实现 cost 和回溯选择。若会议要求增加 DP，只允许新增显式 variant：

```text
LoongServe-style Decode-only + Decode-cost DP
```

该 variant 的 cost 必须来自真实 Decode 数据，并单独做消融；不能覆盖基础组的结果。

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
    检查 block/metadata/headroom
    第一个 exact feasible d 即为 initial DoP
```

这是 Decode-only adaptation，不等同于 LoongServe 二维 DP 联合选择的 DoP。必须记录：

```text
initial_dop_policy = min_exact_feasible
```

禁止通过非零 `ls_decode_initial_kv_dop` 在正式 baseline 中强制固定 DoP。

### 7.3 Placement 分两步

第一阶段不修改现有 placement 数据面，只改变 batching 发生的时机：

- 继续使用现有 block allocator；
- 继续生成现有 receiver metadata；
- 继续保留 pending-token headroom；
- 继续使用现有 allocation/rollback transaction。

第二阶段再切换为 LoongServe-style packed intervals：

1. selected ranks 按 used tokens 从高到低填充；
2. request prompt 按排序后的 request 次序写入；
3. 先填更满 rank 的剩余空间；
4. 只有跨越容量边界时才把一个 request 分布到多个 ranks；
5. interval 最终转换成 Nano block counts 和 metadata。

packed placement 会减少短 request 的 KV owner 数、receiver 数和 block rounding，是 Decode 路径中值得保留的 LoongServe 设计。

### 7.4 DP-domain topology adaptation

LoongServe 是一个统一 elastic instance pool；Nano 当前是：

```text
Attention DP{1,2,4} x SP8
```

group 不能跨 DP domain 合并。建议：

1. 只有一个全局 FIFO waiting queue；
2. dispatcher 为 candidate 计算 feasible domains；
3. 按以下 stable score 选择 domain：

   ```text
   projected_future_kv_utilization
   -> current_used_kv_tokens
   -> running_request_count
   -> dp_idx
   ```

4. 每个 domain 内独立执行 batching、placement 和 Decode elasticity；
5. OOE counter 仍由全局 queue 维护。

这是必要 topology adapter，不能宣传成 LoongServe 原统一实例池。

## 8. Admission transaction

### 8.1 Batch ID 创建边界

batch/group ID 只在以下条件全部满足后创建：

1. ephemeral continuous range 已确定；
2. domain 和 initial ranks 已确定；
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

### 9.3 禁止额外 merge heuristic

baseline path 删除或关闭：

- compute-pressure merge healthy group；
- planner-failure arbitrary merge；
- pending batch benefit heuristic；
- utilization/stability/cooldown consolidation；
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

paused request 回到全局 request-level waiting queue，接受同一套 FIFO/OOE/future-KV scan。

不再创建优先级高于普通 request 的长期 singleton recovery batch。

## 11. 配置建议

建议 source-shaped 参数：

```text
ls_max_wait_tokens = 10
ls_max_num_ooe = 10
ls_running_max_req_size = 1000
ls_admission_max_tokens = auto
ls_min_comp_bound_decoding_batch_size = 128
ls_disable_scale_up = false
ls_decode_enable_future_kv_admission = true
ls_decode_initial_kv_dop = 0
ls_nano_background_consolidation = false
```

其中：

- `ls_decode_initial_kv_dop=0` 表示最小 exact feasible；正式 baseline 禁止强制值；
- `ls_nano_background_consolidation=false` 只关闭 Nano 自定义后台策略；
- source-shaped memory/compute elasticity 不受旧 consolidation mode 控制；
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
- 新增 admission cadence 和 OOE counter；
- request-level current/future scan；
- selected 后 stable length sort；
- ephemeral continuous partition；
- current no-fit 时保持 waiting；
- commit 时才创建 batch/group ID；
- DP-domain global future envelope；
- admission result 可携带已有 running Decode plan；
- Decode memory/compute scale-up 收敛到 source-shaped 规则；
- pause 保留 generated progress。

### 12.2 `sp_state_manager.*`

第一阶段：

- 复用现有 exact placement；
- 复用 transaction/rollback；
- 增加 adapter rejection reason；
- 支持 admission side effect 与已有 Decode plan 共存。

第二阶段：

- packed token intervals；
- interval 到 blocks/metadata 的转换；
- pinned rank-range validation。

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
- waiting/running/capacity synthetic snapshots；
- Issue 1% 六个长短混合窗口；
- resolved config manifest schema。

### Phase 1：Request-level batching

改动：

- 删除 fresh persistent pending batch；
- FIFO cadence/OOE scan；
- stable length sort；
- ephemeral balanced continuous ranges；
- empty-system 检查和 current-system exact commit；
- rollback 后恢复 waiting order。

该阶段不改 placement 表示和 Decode planner。

### Phase 2：Global future-KV 和 admission continuity

改动：

- DP-domain global future envelope；
- 消除跨 group future capacity 重复承诺；
- admission 不吞掉已有 Decode iteration；
- 完整 telemetry。

### Phase 3：Packed placement

改动：

- instance ordering；
- packed intervals；
- block/metadata adapter；
- exact rejection/replan。

### Phase 4：Decode elasticity

改动：

- memory-deficit donor merge；
- exact idle-rank scale-up；
- compute threshold；
- 删除 baseline 中额外 merge；
- preserve-progress pause/readmission；
- source-shaped disable switch。

### Phase 5：验收和清理

- 删除临时双 policy；
- 旧 persistent-batch 文档标记 historical；
- 固定 8/16 GPU scripts；
- 完成 source-shaped CPU differential tests；
- 正式 Issue 1% A/B。

每个 Phase 单独提交，避免一次改动同时重写 queue、placement 和 Decode group ownership。

## 14. 不变量

1. selected membership 由 FIFO/OOE 决定，长度排序不能改变本轮服务资格。
2. request 只有 admission commit 成功后才能从 waiting 移除。
3. planning no-fit 不创建 batch/group ID。
4. 同一 request 只能处于 waiting、planned、running、paused、finished 之一。
5. OOE 只能由 source-shaped counter 允许，不能由 pending bypass 隐式产生。
6. future envelope 包含 domain 全部 running 和 tentative requests。
7. Nano exact gate 只能 reject/replan，不能静默改 DoP。
8. physical allocation 失败必须完整 rollback。
9. admission side effect 不减少已有 requests 的 Decode iteration 数。
10. memory merge 只由 capacity deficit 触发。
11. compute scale-up 只消费 idle ranks。
12. pause/readmission 保留 generated tokens 和采样进度。
13. baseline 不执行 gap、age、utilization 或 arbitrary merge heuristic。
14. 非 LS scheduler 行为不变。

## 15. Telemetry

### `ls_decode_dispatch_scan`

- FIFO snapshot IDs；
- scanned/selected/deferred/frontier IDs；
- per-request reject reason；
- OOE before/after；
- selected prompt token sum；
- cadence counter。

### `ls_decode_batch_plan`

- sorted request IDs/lengths；
- target ranges；
- shrunk ranges；
- empty-system fit result；
- current-system fit result；
- committed members；
- uncommitted members。

### `ls_decode_future_kv`

- domain ID；
- running/tentative request IDs；
- peak tokens；
- token capacity；
- exact block requirement；
- adapter reject reason。

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

1. FIFO selection：可运行请求保持 queue 顺序。
2. Bounded OOE：blocker 最多被越过配置轮数。
3. OOE reset：frontier 成功运行后 counter 清零。
4. Cadence：idle 立即；busy 按成功 Decode iteration 计数。
5. Membership-before-sort：排序不改变 selected IDs。
6. Stable length order：等长 request 保持 FIFO。
7. Continuous split：每个 batch 都是排序数组连续 range。
8. Shrink-on-no-fit：只缩短 ephemeral range，剩余 request 返回 waiting。
9. No persistent identity：current no-fit 后不存在 batch ID/ownership。
10. Global future-KV：两个 group 单独可行、合计不可行时拒绝第二份承诺。
11. Exact adapter：block/metadata reject 后 replan，不改 heuristic。
12. Atomic rollback：任意注入点失败后 queue/blocks/ownership 完全恢复。
13. Admission continuity：已有 requests 不因新 admission 少一次 Decode。
14. Memory deficit merge：donor 顺序和新增 rank 数符合 source-shaped 规则。
15. Compute scale-up：只使用 idle ranks，不 merge healthy group。
16. Scale-up off：memory/compute 两条路径都不能绕过开关。
17. Pause progress：readmission 后保留 prompt+generated。
18. Feature-off：非 LS scheduler 不受影响。

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

断言：

- long current no-fit 时保持 request-level waiting；
- short request 只有在 OOE 允许时越过；
- short request 不因历史 batch identity 陪等；
- 没有 request 丢失、重复或跨状态 ownership；
- `num_ooe` 达到上限后 FIFO blocker 成为 frontier。

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
- Nano background consolidation 关闭；
- source-default 和 artifact-derived threshold 分组运行。

GPU 测试必须按仓库规定申请提权。

## 17. Baseline 完成标准

以下全部满足后，才能称为 LoongServe-style Decode-only baseline：

1. 运行中不存在 fresh persistent pending batch。
2. FIFO/bounded OOE/cadence 行为通过 CPU fixtures。
3. request membership 在长度排序之前确定。
4. batch 是排序数组的连续 ranges。
5. no-fit 时 request 保持 waiting，没有 batch ownership。
6. future-KV 使用 DP-domain global envelope。
7. admission commit 保持 exact allocation/rollback。
8. 已有 requests 不因 admission 丢失 Decode iteration。
9. packed placement 阶段完成并通过 block/metadata correctness。
10. memory-deficit merge 与 compute idle-rank scale-up通过 differential fixtures。
11. Nano background consolidation 和 arbitrary merge 不在 base path。
12. pause/readmission 保留生成进度。
13. resolved config 和 intentional adaptations 写入 manifest。
14. 8/16 GPU correctness 完成。

性能改善不能替代调度一致性检查。正式实验必须同时保存 dispatch、batch、future-KV 和 Decode elasticity telemetry。

## 18. 会议需要拍板的问题

| 决策项 | 选项 | 建议 |
|---|---|---|
| Baseline 名称 | LoongServe / LoongServe-style Decode-only | 后者 |
| Persistent fresh batch | 保留 / 删除 | 删除 |
| Batching v1 | 连续等数量 + capacity shrink / 新 length heuristic | 前者 |
| 二维 DP | 进入第一版 / Decode-cost 独立 variant | 独立 variant |
| Placement | 第一阶段就改 packed / 第二阶段改 | 第二阶段，先隔离 queue 变化 |
| Admission continuity | 保留 admission-only / 同 step 保持已有 Decode | 保持已有 Decode |
| Future-KV | group-local / DP-domain global | DP-domain global |
| Decode threshold | 100 / 128 | Issue 1% 主实验建议 128，100 做 source-default sensitivity |
| Custom consolidation | 进入 base / 关闭 | 关闭 |
| Pause | restart / preserve progress | preserve progress |
| DP topology | 声称统一 pool / 显式 domain adapter | 显式 adapter |

## 19. 推荐会议决议

建议会议批准：

1. baseline 范围严格限定为 Decode-only；
2. 旧 persistent fresh-batch 设计不再约束实现；
3. Phase 1 只改 request-level selection、排序和 ephemeral continuous batching；
4. placement/admission transaction 第一阶段保持不变；
5. Phase 2 修复 global future-KV 和 admission continuity；
6. Phase 3 再切 packed placement；
7. Phase 4 对齐 Decode memory/compute elasticity；
8. 二维 DP 不进入第一版，单列 Decode-cost variant；
9. 所有 Nano topology adapter 和 source deviation 写入 manifest；
10. 正式实验统一命名为 `LoongServe-style Decode-only`。

这样可以避免把当前实验不执行的内容带入设计，同时保留 LoongServe 对 Decode 资源管理最关键的思想，也能控制每个阶段的改动面和验证成本。

## 20. 源码与本地证据索引

LoongServe：

- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/req_queue.py:46`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/req_queue.py:135`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:351`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:686`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:764`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:844`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/longserve_c_scheduler/src/main.cpp:33`
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
