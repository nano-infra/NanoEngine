# NanoDeploy LoongServe-style Decode-only baseline 实现设计

日期：2026-07-18

最近修订：2026-07-19

状态：Implementation-ready；基础语义已按 Decode-only 实验范围冻结

NanoDeploy 代码基线：`64a199154b375aba8cd469ca75542f4103ccbf64`

LoongServe 源码基线：`fb87896d87b170afd4afe591e29da1aa5f6d4e16`

## 0. 结论先行

本实验只研究 Decode。新请求进入系统时，scheduler 直接建立 prompt KV placement，并通过 dummy bootstrap 补齐进入 Decode 所需的状态；之后所有 GPU iteration 都是 Decode。

固定运行条件：

```text
mode = decode
dummy_prefill = true
loop_count = 1
ignore_eos = true
scheduler_mode = centralized
kvcache_block_size = 64
attention_sp = 8
attention_tp = 1
sp_backend = hao_basic
use_dlslime_rpc = true
fixed_sp_size = 0
enable_dynamic_sp_size = false
dummy_bootstrap_token_id = 0
```

支持的 attention/FFN topology 只包括当前代码已经校验的 `DP1×SP8/EP8`、`DP2×SP8/EP16` 和目标 `DP4×SP8/EP32`。请求沿用 LoongServe 的 `max_tokens >= 1` 合法范围；dummy bootstrap 占用第一个 output slot，fresh/readmission 都走同一个 post-bootstrap finish check，`max_tokens=1` 会直接产生 `bootstrap_finished`，不进入 Decode。

`dummy_prefill` 只负责初始化 prompt/readmission KV、追加固定 token ID 0 作为 pending/output token 并记录首 token 状态，不参与 batching cost，也不引入固定 admission 延迟。`max_tokens` 包含该 dummy token；每次 OFFLOAD readmission 也按 LoongServe rerun tuple 追加一个 token。

因此，本方案不设计任何其他执行阶段，不引入与其他阶段有关的 cost model、资源竞争、并发执行接口或阶段切换逻辑。

baseline 的准确名称应为：

```text
LoongServe-style Decode-only scheduler on NanoDeploy
```

它对齐 LoongServe 中与当前实验直接相关的设计：

1. request round-robin 分配到独立 DP pool；
2. 每个 pool 内 waiting queue 的 FIFO 顺序和有限越序；
3. request-level current/future KV admission；
4. 选中后按 admission need tokens 降序；
5. 连续 request range 形成 batch；
6. instance 按已用 token 排序；
7. capacity-aware initial DoP 和 KV placement；
8. 运行中 Decode 的 memory-deficit merge；
9. 运行中 Decode 的 compute-bound idle-rank scale-up；
10. 低 KV 利用率时把最空 rank 的 KV 压紧到 retained ranks 并 scale-down；
11. `PAUSED_OFFLOAD`/readmission 保留生成进度。

第一版不迁移 LoongServe 原二维 DP 的 cost 部分。当前环境没有与原目标函数对应的运行时 cost，强行加载一组无关参数反而会让 baseline 难以解释。第一版采用：

```text
request round-robin 固定到 DP pool
        -> pool-local FIFO 有限越序选择
        -> current/future KV 检查
        -> admission need tokens 稳定降序后做 Nano exact planning
        -> no-fit 时按 FIFO selection order 回退并重新 planning
        -> 每 pool 一个连续候选 batch
        -> LoongServe-style packed interval placement
        -> Nano block/metadata exact adapter + allocation/rollback transaction
        -> source-shaped Decode merge/idle-rank scale-up
        -> low-KV exact consolidation/OFFLOAD
```

这比当前实现更贴近 LoongServe，同时将改动限制在 Decode scheduler。若会议要求二维 DP，再增加一个明确命名的 Decode-cost 版本，不能把它静默混入基础组。

### 0.1 最小偏离原则

基础组只允许五类 unavoidable adaptation：

1. LoongServe artifact 的多个独立 worker 映射为 Nano 的多个固定 SP8 attention pools；
2. 原 Prefill cost DP 删除后，每 pool 每轮只形成一个 batch，并选择最小 exact-feasible DoP；
3. dummy bootstrap 是 CPU/KV side effect，因此每个非 OFFLOAD step 都可 scan、且 admission 不抑制本轮已有 requests 的 Decode；admission token 数超过剩余 truly-idle rank 总容量时，才按 LoongServe 原 capacity 顺序 append/merge existing groups；
4. LoongServe token interval 通过 Nano 的 block/metadata transaction 落地；
5. LoongServe 的 scale-down 原本由真实 Prefill pressure/收益触发；本实验没有真实 Prefill cost，因此只用 low-KV utilization 触发同 group 内 exact rank evacuation，补齐 scale-up 后可回落的闭环。

以下内容不进入 baseline：

- Prefill gain/cost 驱动的 admission-time reclaim，以及 unordered/ad-hoc admission merge；
- waiting/pending benefit、gap、age 或 planner failure 绕过 low-KV threshold 的 consolidation；
- `PAUSED_KVKEEP`；
- block-rounded future predictor；
- event-driven planning cache；
- cross-pool routing/work stealing；
- 二维 Decode-cost DP 或其他 length/gap/age heuristic。

这些内容如需实验，必须使用单独的 feature flag、配置和名称，不能改变本设计所称的 `LoongServe-style Decode-only`。

## 1. Baseline 边界

### 1.1 实验状态模型

baseline 只实现 LoongServe 默认 victim path 实际使用的 OFFLOAD 状态：

```text
WAITING_FRESH(assigned_dp)
    -> RUNNING_DECODE
    -> FINISHED

WAITING_FRESH
    -> FINISHED  # fresh bootstrap 生成唯一 output token

RUNNING_DECODE
    -> PAUSED_OFFLOAD(assigned_dp, progress preserved, KV cleared)
    -> RUNNING_DECODE

PAUSED_OFFLOAD
    -> FINISHED  # readmission bootstrap 恰好生成最后一个 token
```

`PAUSED_KVKEEP`、`RERUNNING_KVKEEP` 和 singleton recovery batch 不进入第一版。无法在空 pool 中满足固定 future-token policy 或 current exact gate 的 fresh request，在 ingress 返回 `UNSCHEDULABLE_REQUEST`，从不进入 `SequenceStatus` 或 waiting queue。OFFLOAD victim 在清 KV 前必须通过第 10 节的 empty-pool exact readmission precheck。

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

每个 Sequence 只新增一个持久 scheduler 字段：

```text
assigned_dp = -1  # 尚未进入 LS ingress
```

`assigned_dp` 是不可变 canonical source，必须进入 Sequence raw serialization、pickle 和 binding；`BlockContext.dp_idx_` 只是当前 placement 的派生副本，context reset/rebuild 后必须从 `Sequence.assigned_dp` 恢复，admission commit 校验两者一致。LS `Scheduler::add()` 只接受 `assigned_dp == -1` 且未被 scheduler-owned maps 持有的 Sequence；首次 add attempt 写入最终 DP，之后对同一对象/ID 的重复 add 返回 typed error，不能重新分配。Python `seq_id` setter 在 `assigned_dp != -1` 时拒绝修改，非 LS Sequence（始终为 `-1`）维持原接口；scheduler 每次 LS boundary 也校验 ID 与 ingress 记录一致。LS scheduler、engine 和 workers 必须 lockstep 升级 serialization schema；本实验不新增 mixed-binary/旧 payload compatibility branch。

queue 自身保存 FIFO 顺序。由于 Nano `seq_id` 在对象构造时生成且 Python 可写，不能把它当作 arrival order；scheduler 只额外维护一份 live `arrival_order_by_seq_id`，在 accepted ingress commit 时写入、finish/abort cleanup 时删除。OFFLOAD FCFS victim 使用 `(arrival_order descending, seq_id descending)`，不在 Sequence 再加 `enqueue_order`。另有 engine-lifetime `seen_ls_seq_ids` registry：首次结构合法的 add attempt（无论随后 accepted 或 singleton-rejected）即登记，直到 engine teardown 都不删除，防止 finished/rejected ID 被复用并覆盖 metric。`admission_kind` 由 `SequenceStatus::WAITING` 或 `PAUSED_OFFLOAD` 推导，也不增加持久字段。

统一定义：

```text
admission_need_tokens(FRESH)           = prompt_tokens
admission_need_tokens(OFFLOAD_READMIT) = prompt_tokens + generated_tokens
```

FIFO membership 由 pool-local queue 决定；membership 确定后的稳定降序、batch token limit 和 packed placement 都使用 `admission_need_tokens`，不能对 readmission 退化为只看 prompt length。OFFLOAD 保留 `output_ids`、generated count、sampling parameters、metric timestamps 和 `assigned_dp`，清空该 request 的 ACTIVE KV 与旧 group membership，并按 LoongServe `back_to_wait_list()` 语义插回原 pool 队首；随后按本节统一 cleanup 规则释放已无 live KV/pending/master/reservation 的 ranks。

### 1.2 对齐范围

需要对齐的 LoongServe 源码位置：

- request future-capacity check：`req_queue.py:46-76`；
- waiting queue scan 与有限越序：`req_queue.py:80-227`；
- request/instance 排序与连续 batch range：`manager.py:686-750`；
- 二维 DP 的状态和回溯形状：`longserve_c_scheduler/src/main.cpp:33-84`；
- packed token interval placement：`manager.py:764-800`；
- 新 Prefill 到达时按低占用 source 压紧 instances 的 scale-down 方向：`manager.py:516-680`；
- Decode memory/compute elasticity：`manager.py:844-970`；
- request 状态、admission token 和 future tuple：`io_struct.py:14-24,110-147`；
- `max_new_tokens>=1`、`ignore_eos` 字段与 length/EOS finish：`sampling_params.py:9-56`、`io_struct.py:188-200`；
- `batch_max_tokens` 的 1/6 auto default：`api_server.py:388-395`；
- finish filtering、new-batch overlap merge 和 pause 后结束本轮：`manager.py:539-591,1138-1146`；
- 实际 OFFLOAD victim path：`pause_strategy.py:35-49`。

### 1.3 明确不做的内容

本 baseline 不包含：

- 非 Decode kernel 或执行路径；
- 与非 Decode 工作有关的 latency model；
- 跨执行阶段的资源比较和抢占；
- 两条计算 lane 的联合调度；
- 与当前 Nano topology 无关的全局通信重构；
- Nano 自定义 gap、age 或 planner-failure group merge heuristic；
- fresh waiting/pending benefit、gap、age 或 planner failure 触发的 consolidation；
- KVKEEP 及其 paused-KV reservation accounting；
- 为 dummy prefill 伪造 Prefill gain、migration-payback 或 profiler cost。

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
| Admission opportunity | 每次 `schedule()` 都先尝试 admission，成功则不 Decode | 每个非 OFFLOAD scheduler step 都可 scan，成功 admission 不抑制已有请求 Decode | 修改返回结果 |
| Queue scan | 一个全局 queue，固定截取 prefix | 每个 pool 独立 FIFO + bounded OOE + current/future KV | 修改 |
| Fresh batch identity | current-fit 前创建长期 `PendingDecodeBatch` | exact plan 成功前保持 request-level waiting | 修改 |
| Request ordering | 选中窗口内已按 prompt length 降序 | 按 `admission_need_tokens` 稳定降序 | 修正 readmission 语义 |
| Batch partition | 全局窗口按 `attention_dp` 均分，no-fit 时缩短 | 每个 pool 每轮一个 ephemeral continuous batch | 修改 |
| Batch count/DoP | batch 数绑定 `attention_dp`，DoP 取第一个可行值 | 每 pool 每轮最多一个 batch；DoP 取本 pool 第一个可行值 | 明确近似 |
| Future-KV | candidate + 单个 target group | 所属 SP8 pool 全部 running + tentative envelope | 修改 |
| Prompt KV placement | request 在 ranks 上均匀 striping | packed intervals，再转换为 Nano blocks/metadata | 与 batching 同步修改 |
| Admission target | standalone no-fit 后按 Nano planner failure 选择 existing group | idle token capacity 足够时只建 standalone；不足时按 LoongServe capacity 顺序 append/merge | 收窄 Nano fallback |
| Admission/Decode | admission 成功会让已有请求少跑一次 Decode | bootstrap 不应吞掉已有请求的 Decode iteration | 修改 |
| Proactive scale-down | utilization 或 pending benefit 可触发 consolidation | 只由 low-KV utilization 产生 candidate；stable/cooldown 只防抖 | 收窄触发并保留 scale-down |
| Memory scale-up | 多种 planner failure 都可能触发 merge | 只按 Decode token deficit merge/加 rank | 修改 |
| Compute scale-up | threshold 与 arbitrary group merge 混合 | 只消费 idle ranks，不强并健康 group | 修改 |
| Preemption | 丢弃 generated tokens 后重启 | 只做 preserve-progress OFFLOAD/readmission | 修改 |

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
require assigned_dp == -1 and seq_id not in seen_ls_seq_ids
candidate_dp = next_dp_rr
validation = validate_singleton_read_only(request, candidate_dp)
prepare_no_throw_nodes_if_accepted(validation)
no_throw_commit:
    seen_ls_seq_ids.insert_prepared(seq_id)
    assigned_dp = candidate_dp
    next_dp_rr = (next_dp_rr + 1) % attention_dp
    if validation rejected: return typed error with assigned_dp
    waiting_by_dp[assigned_dp].splice_prepared(request)
    arrival_order_by_seq_id.insert_prepared(seq_id, next_arrival_order++)
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

assignment 完成后，request 不因其他 pool 更空闲而重新路由。无效或永久不适配的 request 也已经消费本次 RR slot，避免错误流量改变后续合法请求的映射。pause/readmission 回到原 `assigned_dp`，不再次推进 RR。

这对应 LoongServe artifact 使用外部 round-robin proxy 将 TCP connection 固定到独立 worker，而不是 Nano 自定义的 admission-time load balancing。正式 benchmark 必须保持一 request 一 connection；若客户端复用一条 connection 承载多个 requests，则 artifact 的 connection-level RR 不等价于这里的 request-level RR，不能称为同一 dispatch 条件。

### 4.2 每个非 OFFLOAD scheduler step 都允许 admission

Decode-only baseline 不使用固定 step 间隔限制 waiting scan；第 10 节 mandatory OFFLOAD 是唯一在 scan 前立即返回的 source-shaped exception。第 9.3 节 low-KV candidate 只在各 pool 已完成本轮 scan、且全局没有可提交 admission plan时检查，命中后才以独占 `KV_CONSOLIDATION` 取代本轮 Decode。

语义规则是：

```text
先完成 mandatory Decode safety planning
if 本轮 commit OFFLOAD:
    立即 scheduler-only return，不 scan waiting
else:
    for dp in [0, attention_dp):
        if waiting_by_dp[dp] 非空:
            本 scheduler step 执行该 pool 的 request-level admission scan

if 全局没有 valid admission plan，且所有 tentative pool transaction 已结束:
    检查 low-KV candidate
    exact plan 成功则 KV_CONSOLIDATION return，不执行本轮 Decode

if step 入口存在 running requests 且可形成本轮 Decode plan:
    先为入口 snapshot 保留本轮 Decode 所需 ranks
    admission token 数不超过剩余 truly-idle 总容量时只尝试 standalone
    超过时才按第 7.1 节 capacity 顺序 append/merge
    同时返回入口 snapshot 的 Decode plan

if admission 成功且系统原本 idle:
    commit 新 requests
    下一 scheduler step 开始 Decode
```

dummy bootstrap 只有 scheduler/KV 状态更新，没有需要用固定间隔摊销的 GPU 工作。照搬 10-step gate 会人为增加 queueing latency，并改变 arrival rate 实验的负载形状。

第一版固定 every-non-OFFLOAD-step scan，不叠加 event cache。LoongServe 源码只在 idle 或累计到 `max_wait_tokens` 时扫描；这里删除 cadence gate 是因为 dummy bootstrap 没有 Prefill GPU cost，是一项显式 cadence adaptation。KV consolidation 发生在 scan 后，因此不取消本轮 admission opportunity；只有没有可提交 admission 时才延后一轮 Decode。若后续 telemetry 证明 scan 开销成为瓶颈，event cache 只能作为独立性能 variant，不能静默进入基础组。

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

scan 只产生 tentative membership 和 `tentative_ooe_after`，不能提前修改 counter。exact membership shrink、allocation 和 dummy bootstrap 全部结束后，再按 **final committed IDs** 更新：

1. final commit 中确有 request 越过本轮第一个 deferred blocker：`num_ooe = ooe_before + 1`；
2. 有 final commit，但没有实际 bypass：`num_ooe = 0`；
3. 没有 commit，或 admission transaction rollback：`num_ooe = ooe_before`；
4. `num_ooe >= max_num_ooe` 时，本轮从 scan 开始就禁止越序；
5. aborted/ingress-rejected request 安全移除，不消费、不增加也不重置 OOE。

如果 exact shrink 删除了 blocker 后方的全部 selected requests，就不构成 bypass，不能增加 OOE。各 pool 的 counter 在各自 transaction 中提交，某一 pool rollback 不影响其他 pool。

LoongServe API 默认 `max_num_ooe=10`，artifact 会按 dataset 使用其他值。Nano 正式 workload 必须在 manifest 记录每个 pool 共用的 resolved limit，以及各 pool 的实时 counter。

### 4.4 第一版不保留 cost-driven undecided list

LoongServe queue scan 中还有一段由运行时间比较控制的 undecided-prefix 逻辑。当前实验不使用对应 cost，第一版不照搬该分支，也不伪造替代参数。

Decode-only 规则只有：

```text
capacity feasible -> selected
capacity infeasible -> bounded defer 或停止
```

这是一项 intentional adaptation，必须出现在 baseline manifest 和论文方法说明中。

### 4.5 Pool-local planning membership 上限

仍保留有限 planning membership，限制一次 exact planner 的输入规模：

```text
max_selected_requests_per_pool = max_num_seqs
max_selected_admission_tokens_per_pool = ls_admission_max_tokens_per_pool
```

建议：

```text
ls_admission_max_tokens_per_pool =
    max(max_model_len, total_pool_kv_tokens / 6)
```

token sum 使用 `admission_need_tokens`。window 只限制本轮 selected membership，不创建长期 batch identity；`max_selected_requests_per_pool` 同时受 pool 的 `ls_running_max_req_size` 剩余 request slots 限制。

它不新增“只看前 N 个 queue entries”的 scan truncation：FIFO/OOE scan 最坏仍可能遍历本 pool waiting queue，与 LoongServe 源码形状一致。第 16.4 节若发现 every-non-OFFLOAD-step scan 成为瓶颈，只能单独评估 event-cache/scan-budget variant，不能让隐藏 truncation 改变基础组的可服务集合。

### 4.6 Ingress 永久可行性检查

Fresh request 在 RR assignment 后、入 queue 前只做一次 singleton empty-pool validation：

```text
ignore_eos == true
max_tokens >= 1
single-request future tuple <= fixed SP8 pool token capacity
admission_need_tokens <= ls_admission_max_tokens_per_pool
full-SP8 current block/metadata/pending-headroom exact fit
```

失败返回包含具体 reason 的 `UNSCHEDULABLE_REQUEST`，不进入 scheduler-owned waiting queue，不创建 batch/group ID，也不触碰 OOE。这里是固定配置下的输入合法性检查，不是新的运行时调度 policy。通过 singleton validation 但因当前资源 no-fit 的 request 只能留在 queue，不能被误判为永久失败。

冻结 ingress ABI：

```cpp
enum class LSAddError {
    NONE,
    ALREADY_ADDED_OR_ASSIGNED,
    IGNORE_EOS_REQUIRED,
    INVALID_MAX_TOKENS,
    FUTURE_TOKEN_NO_FIT,
    CURRENT_EXACT_NO_FIT,
};

struct LSAddResult {
    bool accepted;
    int assigned_dp;
    LSAddError error;
    std::string reason;
};
```

`Scheduler::add()` 先做不消费 RR 的 duplicate/already-assigned structural check；通过后在 scheduler lock 内 peek 本次 RR DP，完成只读 singleton validation，并预创建所有可能分配的 list/map nodes。随后进入 no-throw ingress commit：登记 `seen_ls_seq_ids` → RR counter advance → 写入 `Sequence.assigned_dp`；validation accepted 时再 splice enqueue、写 arrival order，validation rejected 时直接返回 typed result。singleton-rejected attempt 因此已消费 RR 且保留写入的 `assigned_dp`，但除 engine-lifetime seen registry 外不进入任何 scheduler-owned queue/owner map；duplicate/already-assigned 调用不消费 RR。所有预期 validation failure 都必须成为 `LSAddResult`，不能 throw；prepare 阶段 unexpected exception 尚未改变 scheduler state，abort metric ticket即可，进入 no-throw commit 后不得存在 fault point。文档中的 `WAITING_FRESH` 定义为 `status == WAITING && sequence owned by waiting_by_dp[assigned_dp]`，不是新增 enum；detached rejected Sequence 即使仍是默认 `WAITING` 也不属于 scheduler 状态机。

`LSAddResult.assigned_dp` 的失败语义也冻结：重复提交同一已分配对象时返回其既有 DP；不同对象复用 `seen_ls_seq_ids` 中的 ID 时返回 `-1`；singleton validation 失败返回本次已消费 RR 后实际写入的 DP。三种失败都不额外推进 RR，调用方不能从 `reason` 字符串反推 assignment。

为避免“scheduler 已 enqueue、Python metric 创建失败”的半提交，LS ingress 使用最小两阶段 metric ticket：

1. engine 先调用只读 `Scheduler::precheck_add_identity()`；duplicate/already-assigned 在创建 ticket 前返回，且 `Scheduler::add()` 内仍重复校验；
2. `MetricsManager.prepare_sequence_metric()` 以 insert-if-absent 创建 provisional entry并立即捕获 immutable arrival/decode-arrival timestamps，但不覆盖已有 metric、不 attach Sequence、不增加 server counters；
3. 调用不触碰 metric 的 `Scheduler::add()`；accepted 后才 no-throw attach ticket metric并调用 `commit_sequence_metric()`，发布已捕获 timestamps并增加 prompt counter；
4. singleton-rejected/throw-before-enqueue 时调用 `abort_sequence_metric()` 删除本 ticket 的 provisional entry，不触碰任何既有 metric。

rejected 时抛出 pybind 映射的 `UnschedulableRequestError(error, assigned_dp, reason)` 并只增加 server-level rejection telemetry。benchmark 显式统计该错误，不能把它计为 timeout。accepted enqueue 之后的任何违反 no-throw ingress commit 的异常按第 8.6 节 fatal ABI 处理，不能作为普通 add failure 后继续运行。

## 5. Future-KV admission

### 5.1 Request envelope

对 `ignore_eos=True` workload，按 request 状态构造：

这里 `generated` 包含初次 dummy bootstrap、历次 OFFLOAD readmission bootstrap 和真实 Decode 产生的全部 output tokens。

```text
RUNNING:            base_tokens = prompt + generated
                    remaining   = max_output - generated - 1

WAITING_FRESH:       base_tokens = prompt + 1
                    remaining   = max_output - 2

PAUSED_OFFLOAD:      base_tokens = prompt + generated + 1
                    remaining   = max_output - generated - 2
```

所有 `remaining` 取 `max(0, remaining)`。`base_tokens` 是 LoongServe router envelope 的逻辑量，不等于 request 当前物理持有的 KV；尤其 OFFLOAD request 的物理 KV 已经是 0。

对一组 requests，继续使用 LoongServe-style high-water mark：

```text
按 remaining 降序
peak = max_{i=1..|E|}(prefix_base_tokens(i) + i * remaining_i)
```

### 5.2 检查域是所属 SP8 pool 全局

每个 Nano DP domain 映射成一个固定 SP8 elastic pool。检查 candidate `r` 时构造 identity-deduplicated set：

```text
E = 该 pool 全部 RUNNING requests
  + 本轮该 pool 已 selected/tentative requests
  + candidate r
```

同一个 request ID 在 `E` 中只能出现一次。容量和 request-count 条件为：

```text
peak(E) <= pool_token_capacity
|E| + |unselected_paused_offload| <= ls_running_max_req_size
```

容量单位和唯一计算式冻结为：

```text
block_size = 64 tokens
usable_blocks(dp, sp) = worker_state[dp].block_manager[sp].blocks().size()
pool_token_capacity(dp) = sum_sp(usable_blocks(dp, sp)) * block_size
total_pool_kv_tokens = pool_token_capacity
```

`blocks().size()` 已经是 engine 初始化后实际交给 sequence KV allocator 的 block 数，因此 envelope 层不再扣一份模糊的 `fixed_reservations`。`reserved_blocks_per_req`、pending headroom、metadata 和当前临时 reservation 只在第 5.3 节 current exact gate 计算，不能在 token envelope 重复扣除。四个 pool 若 capacity 不同，分别使用各自启动时冻结的值。

未选中的 OFFLOAD requests 没有物理 KV，因此不再从 token capacity 扣除，但仍占 LoongServe-style paused request slot；候选 OFFLOAD 进入 `E` 后从 `unselected_paused_offload` 去重。基础组没有 KVKEEP，也就没有“既放进 envelope 又扣 paused-held KV”的双重记账分支。

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

future envelope 保持 LoongServe 的 token-level admission policy，它不是 Nano block-rounded future-feasibility proof。当前 admission 之后继续使用 Nano exact gate：

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

exact gate 只保证 **本次 commit** 的 block/metadata/pending safety。运行中未来出现 block-rounded shortage 时，使用第 9、10 节的 source-shaped capacity merge、idle-rank scale-up 或 OFFLOAD；基础组不再增加一套 conservative future-block estimator。

## 6. Candidate 排序与连续 batching

### 6.1 LoongServe 原逻辑

一个独立 SP pool 内，LoongServe：

1. 按 waiting FIFO、容量和 bounded OOE 确定 selected membership；
2. selected requests 按 `get_first_router_need_tokens()` 稳定降序；
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
3. 对 `selected_fifo` 的拷贝做 `admission_need_tokens` 稳定降序；
4. 调用 empty-system 和 current-system Nano exact planning；
5. exact no-fit 时撤销 `selected_fifo` 中最后加入的 request，然后重新排序、重新 planning；
6. exact fit 后，排序结果整体形成一个 ephemeral batch，并生成 admission prepare result；
7. 本节不 commit、不创建持久 ID、不修改 waiting；prepare result 交给第 8 节与 prospective Decode plan 联合 validate/publication。

伪代码：

```text
selected_fifo = pool_local_fifo_ooe_scan(dp)

while not selected_fifo.empty():
    candidate = stable_sort_copy(selected_fifo, admission_need_tokens_desc)

    if empty_system_fit(candidate)
       and (plan = prepare_admission_target(candidate)):
        return plan  # stable state 仍未改变

    selected_fifo.pop_back()  # 按 scan order 回退，不是 candidate.pop_back()
```

未提交 requests 始终保持所属 pool 的原 waiting order，不创建长期 pending batch。`4DP×8SP` 同一 scheduler step 最多由四个 pools 各提交一个相互独立的 batch。

### 6.4 第一版与 LoongServe 的已知差异

第一版每 pool 每轮只有一个 fresh/readmission batch，initial DoP 取 Nano 的第一个 exact feasible 值；LoongServe 可以通过二维 DP 在一个 pool 内生成多个 batches，并联合选择 batch boundaries 和 DoPs。“continuous range”在单 batch 情况下表示最终 batch 完整覆盖稳定排序后的 selected array，不再额外切段。

第一版不使用 gap、固定长度阈值、方差或其他替代 heuristic，也不声称复现完整 batching DP。若后续需要这部分，完整 DP 作为独立 variant 实现和消融，不在本节展开。

## 7. Instance ordering、initial DoP 和 placement

### 7.1 Instance ordering

`DecodeGroupState::allocated_attention_ranks` 是 rank ownership 的唯一持久 source of truth。每次 planning 从它重建并校验只读索引：

```text
rank_owner[dp][sp] = NONE | group_id
```

不再增加第二份可独立修改的 owner map。同一个 rank 同时最多属于一个 group；step-local `decode_reserved_ranks`/`admission_reserved_ranks` 只用于当前 planner，不写回 owner。基础组定义：

```text
truly_idle(rank) :=
    rank_owner == NONE
    and live/pending KV blocks == 0
    and no migration/allocation reservation
    and not in step-local decode/admission reservation

available_ranks(dp) := assigned_dp 内全部 truly_idle ranks
```

admission 使用与 LoongServe idle-capacity 顺序一致的二选一 target policy。先计算：

```text
append_token_slack(group) =
    sum(owned rank token capacity) - group_used_tokens

idle_token_capacity = sum(token capacity of truly-idle ranks)
admission_token_sum = sum(admission_need_tokens of selected membership)
```

这里的 slack 只用于复现 LoongServe 的 capacity 排序，不预扣本轮 pending/headroom；每个 prospective target 是否真的可提交，仍由 current exact adapter 判定。

所有 group-list 算法都从 canonical `ls_group_ids_by_dp_[dp]` 的当前稳定顺序取输入，不遍历 unordered map。与 LoongServe 一样，按单一数值 key 做 stable ascending sort 后从尾部 `pop_back()`；equal-key 时因此选择原 pool order 中更靠后的 group，不再增加 group-ID tie heuristic。

canonical pool order 的 mutation 也冻结：standalone/new admission survivor append tail；capacity append 将 actually merged donors 按原顺序 stable erase 后，把 new survivor append tail；memory planning commit 后直接采用第 9.1 节最终 can-list 的 survivor 顺序；finish/OFFLOAD/whole-group delete 只 stable erase；idle-rank scale-up、zero-live cleanup和同 group low-KV consolidation都不重排 group。任何路径都不得按 unordered-map iteration 或 group ID 重新排序。

group 内 Sequence 顺序同样不得由 unordered container 重建：admission capacity survivor 先放 surviving new requests 的既定 stable admission order，再按 planned-donor selection order追加每个 actually overlapped donor 的原 Sequence order；memory survivor 先保留 constrained group 原顺序，再按 can-list pop/union order追加 donors；finish/abort 只 stable erase。这一顺序直接决定后续 continuous master ranges。

1. **standalone**：当 `admission_token_sum <= idle_token_capacity` 时，只用 truly-idle ranks，按最小 exact-feasible DoP 建新 group；若 block/metadata/headroom exact no-fit，回到第 6.3 节缩小 FIFO membership，不能因此借 existing group；
2. **capacity append/merge**：仅当 `admission_token_sum > idle_token_capacity` 时，令 `token_deficit = admission_token_sum - idle_token_capacity`。将本 pool mandatory-safety 后可安全 Decode 的 groups 按 `append_token_slack` stable ascending sort，再从尾部依次 pop，选择累计 slack 首次覆盖 deficit 的最短 source-order donor sequence。candidate ranks 是 truly-idle ranks 与该固定 donor sequence 的 owned-rank union，并按 `(used tokens, sp_rank)` 做一次 min-exact-feasible DoP 搜索。所有 candidate DoP 都 exact no-fit，或全部 eligible groups 的累计 slack 不足时，回到第 6.3 节缩小 FIFO membership；不能再追加一个 donor 来掩盖 Nano planner failure。

capacity append 只能弥补源码定义的 idle aggregate token deficit，不比较 age/gap/planner failure，也不伪造 Prefill gain/cost。它允许多个 groups 仅因为 LoongServe 原 admission 会按 idle-token capacity 取得 Decode batch instances 并在 overlap 后 merge；禁止跨 pool、禁止跳过 slack 顺序、禁止在 source prefix 之外多并 group。所有 step-entry real requests 仍保留在本轮 Decode eligible set，新 admitted 不参加本轮。

可用 ranks 按以下 stable key 排序：

```text
(used_kv_tokens ascending, sp_rank ascending)
```

同 key 时保持原 rank-pool 顺序。Nano 当前没有本设计需要单独引入的 `node_id` key，使用现有 `sp_rank` 即可确定性复现。

ownership 只在 standalone admission、capacity append/merge、memory-deficit merge、idle-rank scale-up、zero-live-KV cleanup、成功 low-KV consolidation 或整个 group 删除时改变。单个 request finish/OFFLOAD 后，对 group 的每个 rank 重新计算 live KV、pending token、iteration master 和 step-local reservation；四者都为 0/空的 rank 立即从 canonical allocation 删除并令 owner 变为 NONE，即使 group 仍非空。该 zero-live 操作不移动历史 KV；low-KV consolidation 则必须走第 9.3 节独占 exact P2P transaction。

### 7.2 Initial DoP

standalone admission 使用 Nano 现有的最小可行 DoP：

```text
for d = 1 .. len(available_ranks):
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

capacity plan 选中 donor 后，candidate instance ordering 仍按第 7.1 节 `(used tokens, sp_rank)`，因此 idle ranks 先于仍有 live KV 的 donor ranks。dummy bootstrap 后先过滤本轮已完成 requests，再按 surviving new KV 的实际 rank overlap 发布 merge：只要 surviving request 使用了某 donor 的任一 rank，最终 survivor group 就包含该 donor 的完整原 allocation；未被 surviving request overlap 的 planned donor 保持原 group 不变。若没有 surviving new request，则不发布任何 donor merge。telemetry 必须分别记录 `target_kind=standalone|capacity_append`、raw idle capacity/deficit、planned donor IDs、actually merged donor IDs 和 survivor ID。

### 7.3 Placement 一次切换

batching 和 initial placement 在同一个 admission pipeline 中一次切换，不保留“新 batching + 旧均匀 striping”的中间 baseline。

这里要区分两种排序：

- **选择 ranks**：standalone available ranks 按 used tokens 升序；capacity fallback 先按 group slack 选择 owned-rank unions，再取 truly-idle rank prefix；
- **在 placement ranks 内写入**：全部 target ranks 按 used tokens 降序，优先填更满 rank 的剩余空间；used tokens 等值时保持 stable SP-rank order。

第二个顺序与 LoongServe `manager.py:764-800` 的 packed interval 逻辑一致。对每个 candidate target，placement 流程为：

1. 每个 request 的 `admission_need_tokens` 按 batch 中已经确定的 request 次序依次写入；fresh 为 prompt，OFFLOAD readmission 为 prompt+generated；
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
3. 每个 pool 独立执行 batching、placement、capacity merge 和 scale-up；
4. group 可以在本 pool 内从 DoP=1 扩到 DoP=8；
5. group 不能跨 DP merge，不能扩到 DoP=9...32；
6. pause/readmission 回原 pool；
7. baseline 禁止 admission-time load-aware rerouting 和跨 pool work stealing。

这意味着某个 pool 可能因长 request 阻塞，而另一个 pool 同时存在 idle ranks。基础组保留这一结果；若增加跨 pool rerouting，必须命名为单独的 Nano load-balancing enhancement。

如果未来要复现一个统一 32-instance pool，应使用 `DP1×SP32`，或者修改 Nano collective、KV ownership 和 group merge 以支持跨现有 DP domains。该工作不属于当前 baseline。

这里的“四个独立 pools”只描述 request queue、attention KV ownership 和 scheduler decision domain。Nano 的 FFN 使用 EP32，同一 GPU iteration 仍要求全部 workers 按统一 cadence 进入 collective；它不等价于四个可独立推进的 LoongServe worker processes。正式实验必须把这一点记录为 execution-topology adaptation，不能从 queue/KV 隔离推导出物理执行完全独立。

## 8. Admission transaction

### 8.1 Batch ID 创建边界

batch/group ID 是 scheduler-thread 内单调 ID，允许 transaction failure 留下 gap；gap 没有调度语义。ID 可以在 exact plan 成功后为 transaction reserve，但持久 identity 只在以下条件全部满足后发布：

1. ephemeral continuous range 已确定；
2. 已在 `assigned_dp` pool 内确定 initial ranks；
3. exact placement validation 成功；
4. physical allocation 成功；
5. scheduler 准备发布 ownership。

失败 ID 不得出现在任何 owner map 或 committed telemetry 中，**不回滚 global counter**。这样不同 pool 的 partial commit 不会互相覆盖 counter。禁止在 current-fit 之前写入：

```text
ls_pending_decode_batches_
ls_seq_to_batch_
```

所有 pool 先完成 isolated prepare，再按 `dp_idx` 升序执行预先保证 no-throw 的 commit。isolated prepare 只允许推进无需回滚的 monotonic ID，以及持有第 8.5 节 transaction-owned block/list/map reservations；稳定 queue/group/Sequence 状态仍不改变。pool A 有可提交 admission、pool B admission no-fit 时，A 正常返回 committed record，B 保持原 queue/OOE并仍建立 decode-only plan；不存在“回滚 B 时恢复 A 已推进的 global counter”。

### 8.2 Atomic commit

每个 pool 独立建立 transaction；一个 pool rollback 不撤销其他 pool 已成功的 commit。prepare/validate 阶段不得修改稳定状态，commit 必须一次覆盖：

- waiting queue membership 和原顺序；
- prompt/readmission blocks 与完整 `BlockContext::ACTIVE`；
- dummy token append、pending flag/target 和 running-token accounting；
- `SequenceStatus`、group/batch maps 和 canonical `allocated_attention_ranks`；派生 owner 索引只在 publication 后重建并校验；
- pool-local OOE；
- fresh admission 的 first-scheduled、decode-scheduled、first-token/last-token/generated metric state；OFFLOAD readmission 保留 first/queue时间点并prepared更新 last-token、ITL和generated。

dummy append 后统一检查 output limit。fresh 或 OFFLOAD readmission 若由该 token 完成，则在同一 transaction 内标记 FINISHED、释放刚恢复的 KV 和该 request 的 group membership，并按 zero-live-KV cleanup 释放不再承载任何 live/pending/master/reservation 的 ranks；group 变空时删除其全部剩余状态。admission record 标记 `bootstrap_finished=true`，该 request 不会进入下一轮 Decode。

post-bootstrap filtering 发生在 persistent group publication 之前，并复现 LoongServe “先 filter finished new batch，再按 surviving overlap merge”的顺序：

- selected 全部 `bootstrap_finished`：standalone/capacity target 都只用于临时 exact validation，释放 staged allocation，不创建新 group，也不 merge planned donors；
- selected 部分完成：只让 surviving requests 形成 batch，并只 merge 与 surviving KV placement 实际重叠的 donor groups；
- selected 全部未完成：按既定 placement 发布 standalone group 或实际 overlap 的 capacity merge。

只要存在 surviving new request，admission transaction 预留的 **new group ID** 就是最终 survivor ID；所有实际 overlap 的 donor group IDs 被删除并重映射到该 new group。这与 LoongServe 将 overlapped Decode batches merge 到 new batch 的方向一致，也消除了 tie-break 歧义。若全部 bootstrap-finished，则该预留 group ID 只留下允许的 gap，不发布。

metric 规则同样固定：transaction prepare捕获一个 bootstrap commit timestamp。fresh bootstrap commit幂等记录 first-scheduled、decode-scheduled和first-token，将last-token设为该timestamp、generated加1且不产生首token ITL；OFFLOAD readmission保留first/queue timestamps，把 `(commit_timestamp - previous_last_token) * 1000` 作为一条ITL sample，更新last-token并generated加1。prepare必须预留ITL vector容量，commit不得调用可能分配的`record_token()`；任何staging/validation失败都恢复完整metric snapshot。

因此 dummy bootstrap 从 Python `llm_engine.py` 移入 C++ admission transaction。Python 不再执行 admission 后的 `may_append()`、`append_token(0)`、pending mark 或 running-token counter 更新，只消费 committed admission records 和记录外部 telemetry。这样 allocation 成功但 bootstrap 失败时仍能在同一 transaction 内回滚。

安全不变量：

```text
一份已生成的 admission plan
要么全部 commit
要么 waiting/blocks/ownership/OOE/metric 完全恢复；只允许 monotonic ID 留 gap
```

原子性不再表示：

```text
过去某个 scheduler step 形成的 request 集合
必须永久作为一个 batch 一起等待
```

### 8.3 Admission 不吞掉已有 Decode iteration

当前 dummy bootstrap 不运行 GPU model，因此成功 admission 不应让已有 running requests 少执行一次 Decode。

不新增 `ADMISSION_AND_DECODE` action。删除 parallel `ls_initial_*` arrays，冻结一个 pybind-visible typed ABI：

```cpp
enum class LSAdmissionKind { FRESH, OFFLOAD_READMIT };
enum class LSAdmissionTargetKind { STANDALONE, CAPACITY_APPEND };

struct LSAdmissionRecord {
    std::shared_ptr<Sequence> sequence;
    int dp_idx;
    uint64_t batch_id;
    std::optional<uint64_t> group_id_after_commit;
    LSAdmissionKind admission_kind;
    LSAdmissionTargetKind target_kind;
    int planned_kv_dop;
    std::vector<int> planned_kv_ranks;
    bool bootstrap_finished;
    int bootstrap_token_id;  // baseline 恒为 0
};

struct ScheduleResult {
    // ... existing fields ...
    std::vector<LSAdmissionRecord> ls_admission_records;
    std::vector<std::vector<uint64_t>> ls_real_decode_ids_by_dp;
    std::vector<std::vector<uint64_t>> ls_running_ids_by_dp_after_commit;
};
```

每个 selected sequence 恰好产生一条 record。`planned_kv_dop/planned_kv_ranks` 描述该 admission batch 的 staged placement，不代表该 sequence 在 post-bootstrap cleanup 后仍有 owner。`group_id_after_commit` 是 **per-sequence persistent owner**：`bootstrap_finished=true` 时恒为 `nullopt`；surviving records 共享 new survivor group ID。全部完成时 `STANDALONE` 和 `CAPACITY_APPEND` records 都是 `nullopt`，后者不得发布 planned donor merge。

`dp_seqs/dp_sp_seqs/filtered_dp_sp_seqs` 中的 **real requests** 只来自 mandatory-safety 后仍 eligible 的 step-entry snapshot；为保持 SP8/EP collective cadence加入的 dummy sequences 继续保留，不算 real membership。`ls_real_decode_ids_by_dp` 是 model/throughput/completion 的唯一 real-set ABI；`ls_running_ids_by_dp_after_commit` 是 server running-request gauge 的唯一 ABI，并包含本轮新 admitted survivors。

LS path 的 action 语义固定为：

| `action` | scheduler-only progress | Decode fields | engine 行为 |
|---|---|---|---|
| `DECODE` | admissions 可有，OFFLOAD record 必为空 | eligible snapshot real requests + required dummies | 只执行一轮 Decode |
| `ADMISSION` | 至少一条 committed admission，或恰一条 OFFLOAD record；二者不同时出现 | 空 | scheduler-only，不调用 model |
| `KV_CONSOLIDATION` | 恰一份 RESERVED low-KV exact evacuation plan；admission/OFFLOAD records 均为空 | 空 | 独占执行 P2P、提交 scale-down，不调用 model |

为复用现有 ABI，本文的 OFFLOAD record 就是同 index 的 `ls_preempted_sequence_ids` 与 `ls_preemption_reasons`，reason 固定含 `OFFLOAD`；OFFLOAD 继续复用 `ADMISSION`，不为它再新增第四种 action。

formal baseline 固定 `ls_kv_consolidation_mode=execute`。`KV_CONSOLIDATION` 只服务第 9.3 节同一 group 内“迁走一个低占用 source rank 并释放该 rank”的 exact P2P scale-down；capacity/admission group merge 是 scheduler metadata union，OFFLOAD 是第 10 节 scheduler-local commit，二者都不能伪装成该 action。一次 `schedule()` 全局最多返回一份 consolidation plan。

field matrix 必须严格：`DECODE/ADMISSION` 的 `kv_consolidation_plan == null`；`KV_CONSOLIDATION` 恰有一个 `RESERVED` plan，且 admission records、OFFLOAD records、real/running IDs 及全部 Decode-shaped arrays 都为空。

LS path 不再用 `is_prefill` 选择 dummy bootstrap 分支；`is_prefill` 固定为 `false`，engine 必须按 `action` 决定是否执行 model。无 real Decode、无 committed admission/OFFLOAD、也无有效 RESERVED consolidation plan却仍有 waiting 的情况违反 ingress empty-pool-fit invariant，scheduler 抛第 8.6 节 typed fatal error，不能用无进展的空 `ADMISSION` 忙等。

engine 在读取 Decode-shaped arrays 前先按 action 分流。`KV_CONSOLIDATION` 必须断言 admission/OFFLOAD/Decode fields 为空，调用冻结 plan 自带的 P2P coordinator，成功后刷新 group/maintenance telemetry并直接返回 zero-token result；不得调用 model。其余 action 才先消费 `ls_admission_records`：对 `bootstrap_finished=true` 的 sequence 恰好调用一次 `metrics_manager.complete_sequence()` 并把 `(seq_id, completion_token_ids)` 加入本 step outputs；其 ID 必须与 Decode real set 不相交。其余 records 只记 telemetry。随后先用 `ls_running_ids_by_dp_after_commit` 更新 running gauge，并刷新 waiting/paused gauges；`action=ADMISSION` 才直接返回 `(outputs, num_tokens=0, real_batch_size=0, ...)`，只跳过 Decode-shaped stats、executor 与 throughput，不得访问/flatten 空 Decode arrays。`action=DECODE` 才 forward，并从 Decode results 追加普通 finished outputs；用 step-local finished-ID set 断言没有重复 completion。

`LLMEngine.generate()` 必须先把本 step `outputs` 合并进最终结果并更新 completed progress，再处理 `num_tokens == 0` 的 throughput 分支；零 GPU token 只能跳过当步吞吐率采样，不能 `continue` 掉 admission-only completion。KV consolidation stall 单独累计并归入下一次 Decode wall time，同时单独报告 maintenance latency。running gauge 使用 `ls_running_ids_by_dp_after_commit`，batch size、Decode token usage、throughput 和普通 completion 只使用 `ls_real_decode_ids_by_dp` 过滤后的 real requests，collective dummies 永不进入用户指标。

### 8.4 单步顺序和资源优先级

`Scheduler::schedule()` 的 LS contract 前提是至少存在一个 scheduler-owned RUNNING、WAITING 或 PAUSED request。`LLMEngine.generate()` 已由 `is_finished()` 保证此前提；直接在 empty engine 调用 `LLMEngine.step()` 必须在进入 scheduler 前返回明确的 non-fatal API misuse error。所以下文的 `NO_PROGRESS_INVARIANT` 只针对仍有 outstanding work 却无法产生进展，不能把正常空 engine 误判为 fatal，也不为此新增空 action。

每个 `schedule()` 使用以下固定顺序：

```text
1. finish/abort + zero-live cleanup；冻结 step-entry RUNNING IDs
2. mandatory memory-safety planning
   capacity NO_FIT -> commit 一个 recoverable OFFLOAD victim并立即返回 scheduler-only ADMISSION
   本调用不再 scan/readmit/Decode；下一次 schedule 才重新 planning
3. pool-local FIFO/OOE admission prepare
   idle aggregate token deficit -> ordered capacity append/merge prepare
   同时建立 transaction-owned admission rank/block reservations
4. 若全局没有 valid admission plan，先确保所有 tentative pool transaction 已 abort/结束，再按
   `(dp_idx asc, canonical group order)` 检查成熟 low-KV candidate；选择
   `(used_blocks, used_tokens, sp_rank) asc` 的非 master source，exact plan成功则发布全局恰一份
   RESERVED plan并立即返回 KV_CONSOLIDATION；plan reject/abort保持稳定 graph并继续
5. optional compute scale-up 只规划 admission reservation 剩余的 idle ranks
6. 以 admission 后的 prospective group graph 为输入，为 eligible entry IDs 构造并 validate `LSDecodePlanTransaction`
7. admission/optional prepare 或 combined validate 失败 -> abort tentative mutations，OOE 不变，重新构造 stable graph 的 decode-only plan
   decode-only capacity NO_FIT -> 按步骤 2 commit 恰好一个 OFFLOAD并返回；decode-only internal prepare/validate error -> typed fatal
8. 将同一 pool 的 admission 与 Decode plan 组成 `LSPoolStepTransaction`；全部 required Decode components validate 后，按 dp_idx no-throw publication
9. 有 eligible real snapshot -> DECODE；否则有 committed admission/OFFLOAD -> ADMISSION；否则按第 8.6 节处理
```

执行效果：

1. 纯 admission 不会让 step-entry running requests 少一次 Decode；mandatory OFFLOAD 和 KV consolidation 是两种独占 maintenance，整个 snapshot 本轮不 forward；
2. 新 admitted 且未完成的 requests 已完成 dummy bootstrap，但不出现在本轮 Decode snapshot，从下一轮开始 Decode；`bootstrap_finished` readmission 直接完成；
3. 系统原本 idle 时，本轮只做 admission，下一轮开始 Decode；
4. waiting admission 的 rank reservation 先于 optional compute scale-up，compute 只能消费剩余 idle ranks，不能饿死 waiting；
5. ordinary admission no-fit、prepare rollback 或 optional scale-up failure 只丢弃 tentative work，不能让稳定 running requests 少一次 Decode；
6. 成熟 low-KV candidate 只在没有 valid admission 时让 Decode 延后一轮；一次只释放一个 rank，且 stable/cooldown/check-interval 限制其频率。它不能与 admission、OFFLOAD 或 pool transaction 共存，也不能饿死可提交 admission。

这里没有第二条 GPU 计算 lane，只是把 CPU/KV admission side effect 与已有 Decode plan 放在同一个 scheduler result 中。

### 8.5 DecodePlan transaction

`LSDecodePlanTransaction` 只包含 mandatory memory-deficit group merge、空 rank scale-up 和 iteration-master/pending-token reservation，不迁移历史 KV、不执行 P2P。admission capacity append/merge 属于同 pool 的 admission plan；两者共同生成一份 prospective group graph。prepare 保存 canonical group allocations、Sequence ACTIVE contexts、blocks/running counters，并完成所有可能分配内存的容器准备；validate 在该 prospective graph 上做 exact master/headroom 和 real-membership 检查。

`LSKVConsolidationPlan` 是独立的 stop-the-world transaction，不嵌入 `LSPoolStepTransaction`。只有全局没有 valid admission、所有 tentative admission/其他 pool transaction 都已 abort/结束且没有 active prepared/committing state 时才能创建它；RESERVED 期间 scheduler 的 add/schedule/preempt/postprocess/free 全部 fail closed，直到 coordinator commit 或明确可安全 abort。

`LSPoolStepTransaction` 是原子性 wrapper，不是新的调度 policy。它只把已由第 7、9 节决定的 admission/Decode mutations 按依赖顺序发布：staged prompt/readmission KV 与 dummy 结果 → post-bootstrap filter/zero-live cleanup → surviving-overlap group union/idle-rank ownership → iteration master/pending reservation → queue/maps/OOE/derived-owner publication。所有可能失败的 allocator/container 操作必须在 prepare 阶段完成。admission/optional component prepare 或 combined validate 失败时，abort 后必须在未改变的稳定 graph 上重建 decode-only transaction；只有 decode-only 本身发生 fatal safety/internal error 时才不 forward。通过 validate 后的 commit 必须 no-throw，因此不存在“admission 已发布但 Decode layout 仍引用旧 group”的中间状态。

各 pool admission 独立 prepare；一个 pool 的 admission no-fit/rollback 不撤销其他 pool 的 valid admission，也不要求恢复 global monotonic ID，但该 pool 的 decode-only component 仍必须 validate。所有 pool 的 required Decode components 先全部通过，之后才开始任何 publication；任一 required Decode internal failure都令整个 schedule 在 publication 前 typed-fatal。只有最终 committed admissions/Decode plans 进入 `ScheduleResult` 和 committed telemetry。

现有 `allocate_ls_initial_batch()`、`commit_iteration_master_plan()` 和 `reserve_blocks()/release_blocks()` 不能直接满足上述边界，因为它们在 mutation 中仍可能分配内存。formal LS path 因此必须增加一个 allocator adapter，而不是假设现有 API 已经 no-throw：

```cpp
class BlockManager::PreparedBlockMutation {
public:
    PreparedBlockMutation(const PreparedBlockMutation&) = delete;
    PreparedBlockMutation& operator=(const PreparedBlockMutation&) = delete;
    PreparedBlockMutation(PreparedBlockMutation&&) noexcept;
    PreparedBlockMutation& operator=(PreparedBlockMutation&&) noexcept;
    ~PreparedBlockMutation() noexcept;

    void commit_noexcept() noexcept;
    void abort_noexcept() noexcept;
};

PreparedBlockMutation prepare_allocate_uncached(/* exact block IDs/counts */);
PreparedBlockMutation prepare_release(/* owned block IDs */);
```

语义冻结如下：

1. `PreparedBlockMutation` 和持有它的 `LSPoolStepTransaction` 都是 move-only RAII owner：copy 删除、move `noexcept`；状态为 `PREPARED` 的析构自动且幂等调用 `abort_noexcept()`，`COMMITTED/ABORTED` 析构为空操作，确保后续 component prepare 抛异常时不会泄漏 reservation；
2. prepare 可将 free-list nodes splice 到 transaction-owned reserved list，并预先完成 `used_block_ids_` 插入/rehash、release-side free-list node 创建和 refcount 校验；这些 reservation 对本次 scheduler 可见，但不发布到任何 Sequence/group；
3. prepare 同时构造完整 shadow `BlockContext`，为 block tables/locations 预留容量，并预建 queue/map nodes；LS pool waiting queue 改为 scheduler-private `std::list`，admission removal 和 OFFLOAD head insertion 通过 node splice；
4. abort 只做 list splice、node erase、refcount/state 恢复和 shadow-object析构，不分配内存且 `noexcept`；
5. commit 只做 prepared block ownership attach/release、shadow context/container node swap/splice 和预计算 counter 赋值，不调用现有可能分配的 `allocate*`、`may_append()`、`trim_blocks*()`、deque insertion 或 throwing assertion；
6. `SPStateManager` 新增 `prepare_ls_initial_batch()` 与 `prepare_iteration_master_plan()` 返回上述 mutations；现有 mutating APIs 继续服务非 baseline 路径，不在 formal LS commit 中调用；
7. test-only failure injection 只允许发生在 prepare/validate 阶段；一旦进入 `commit_noexcept()` 就没有可恢复 fault point；commit 返回后运行 `validate_after_publication_noexcept() -> optional<LSFatalCode>`，检查失败时先 latch，再由 noexcept 区域外抛 typed error，不能在 `noexcept` 内 throw/terminate。

这套 API 是 Nano block/list 数据结构对原子 publication 的执行适配，不增加 admission、merge、victim 或 scale-up policy。

### 8.6 Fatal error ABI

冻结 pybind-visible fatal 类型：

```cpp
enum class LSFatalCode {
    NO_PROGRESS_INVARIANT,
    UNRECOVERABLE_CAPACITY,
    DECODE_PREPARE_OR_VALIDATE_FAILED,
    METRIC_COMMIT_FAILED,
    KV_CONSOLIDATION_FAILED,
    POST_PUBLICATION_INVARIANT,
};

class LSSchedulerFatalError : public std::runtime_error {
public:
    LSFatalCode fatal_code() const noexcept;

private:
    LSFatalCode code;
};

// idempotent；只保留 first code，不复制动态字符串，供 Python publication path 调用
void Scheduler::latch_ls_fatal(LSFatalCode code) noexcept;
```

ingress 的 `UnschedulableRequestError` 是 request-local 非 fatal 错误；以上 fatal codes 则令 scheduler 设置永久 `ls_fatal_` latch。LS `LLMEngine.step()` 必须包住 `scheduler.schedule()`、consolidation coordinator/P2P/commit和admission-record publication：捕获 `LSSchedulerFatalError` 或任何 publication-boundary unexpected exception 后，先通过 binding 调用 `latch_ls_fatal(code)`，再设置现有 Python `fatal_error`、记录 code/step/state hashes并重新抛出。consolidation在确认尚未dispatch任何worker RPC时失败，才允许`abort_noexcept()` destination reservation、记录rejected maintenance并返回zero-token；从首次dispatch开始，任何completion ambiguity或commit stale/failure都必须latch `KV_CONSOLIDATION_FAILED`。`LLMEngine.add_request()` 同样包住 `Scheduler::add()` 与 metric ticket commit：scheduler ingress prepare 异常发生在任何 seen/RR/assignment mutation 前，因此 abort ticket 后可原样抛出；accepted enqueue 后的异常调用 `latch_ls_fatal(METRIC_COMMIT_FAILED)` 并 latch engine。之后 `add_request()/step()/generate()` 都经 `_raise_if_fatal()` 拒绝继续。benchmark 将它记录为 run failure，不能重试同一 engine 掩盖重复 no-progress。

fatal code 映射不得由调用点自由选择：typed `LSSchedulerFatalError` 使用其 `fatal_code()`；required Decode prepare/validate 的 unexpected exception，在确认稳定状态未改变后映射为 `DECODE_PREPARE_OR_VALIDATE_FAILED`；KV P2P 已 dispatch 后的 worker completion ambiguity、copy/commit failure映射为 `KV_CONSOLIDATION_FAILED`，保留 RESERVED guard且不继续运行；其他 commit 返回后的 post-check failure 或 publication-boundary unexpected exception映射为 `POST_PUBLICATION_INVARIANT`；accepted add 后的 metric attach/commit failure映射为 `METRIC_COMMIT_FAILED`。`NO_PROGRESS_INVARIANT` 与 `UNRECOVERABLE_CAPACITY` 只由对应显式 scheduler invariant path 产生。

## 9. 运行中 Decode elasticity

### 9.1 Memory deficit

对每个 running group 计算：

```text
idle_tokens = group_capacity
            - group_used_tokens
            - num_running_requests
```

然后按 LoongServe `manager.py:844-906`：

1. 按 canonical pool group order 建 can-decode/cannot-decode lists；两者都只按 `idle_tokens` stable ascending sort，cannot 从头处理 deficit 最大者；
2. 对每个 constrained entry，从 can-list 尾部 `pop_back()` 取得当前 source-order donor，直到累计 slack 非负；feasible union 直接 append 到 can-list 尾部，**不重新排序**，因此下一 constrained group 会像 LoongServe 源码一样优先 pop 这个 union；
3. 合并全部可用 donor 仍不足时，按精确 token deficit ceiling 使用 truly-idle ranks，rank 顺序固定为 `sp_rank ascending`；
4. memory merge 的 survivor 恒为原 constrained group ID，donor ranks 全部重映射到它，保持 unique owner；
5. Nano block-level append slack 做最终 safety validation；
6. pool 总 capacity 仍不足时进入第 10 节 OFFLOAD，不得 arbitrary merge 一个 planner-failure group。

只有 capacity deficit 才触发这类 merge。

### 9.2 Compute bound

按 LoongServe `manager.py:910-969`：

```text
while remaining_requests // remaining_instances
      > min_comp_bound_decoding_batch_size
      and idle_ranks 非空:
    add one idle rank
```

`remaining_requests` 只统计本轮 `ls_real_decode_ids_by_dp` 中属于该 prospective group 的 eligible step-entry requests；本轮 new admitted survivors 即使已使 admission new-group ID 成为 merge survivor，也不计入 compute threshold，并从下一 step 才参与 Decode/compute planning。这里必须是与源码一致的整数 floor division `//`；临界点不能替换成实数除法、ceil 或 `ceil(batch/threshold)`。

多个 groups 按第 9.1 节 mandatory memory planning 产出的 **pre-admission can-list** 顺序从头处理，共享一份已扣除 admission logical reservations 的 idle-rank list；即使某些 groups 将在 admission publication 时并入 new survivor，compute decision 仍先按其 step-entry Decode group身份计算，再映射到最终 prospective union。每个 group 按 LoongServe `extra_idle_instances.pop()` 对应的 `sp_rank descending` 逐个增加。没有 idle rank就保持当前 DoP，不 merge 健康 group。

LoongServe API default 为 100，artifact launcher 使用 128。两者都有 source provenance。

正式 Issue 1% baseline 固定为 128，不在运行中自适应；100只用于 source-default sensitivity，64和8只作为独立 sensitivity，不混入base。

### 9.3 Low-KV exact consolidation

LoongServe `manager.py:516-680` 只在真实新 Prefill 到来时尝试压紧 Decode instances：按 used tokens 选择最低占用 source，把 KV 搬到 retained ranks，source 清空后再 scale-down。普通 Decode path 不做这类迁移；finish path 只释放已经 zero-live 的 rank。因此在没有真实 Prefill cost 的 dummy-prefill/Decode-only 实验里完全关闭 consolidation，会令临时 scale-up 到 SP8 的 group 除非某 rank 自然归零，否则没有回落路径。

formal baseline 保留一个显式的 Decode-only adaptation：**candidate 只能来自持续低 KV 利用率，执行形状继续沿用 LoongServe 的 low-source → retained destination → source release**。

```text
group_kv_util = sum(used_kv_blocks on participating ranks)
              / sum(usable_kv_blocks on participating ranks)

compute_floor = minimum d such that
                real_running_requests // d
                <= ls_min_comp_bound_decoding_batch_size

eligible iff:
    group_kv_util < ls_kv_consolidation_candidate_util
    and current_kv_dop > 1
    and current_kv_dop - 1 >= compute_floor
    and candidate stable/cooldown/check gates pass
```

`compute_floor` 复用第 9.2 节同一个 source threshold，不再引入独立的 scale-down target policy；如果撤掉一个 rank 会令下一次 source compute rule 立刻 scale-up，则本轮不 consolidation。candidate 只遍历 real RUNNING group，按 `(dp_idx ascending, ls_group_ids_by_dp_ canonical order)` 选第一个可执行 group。source ranks 排除 active/pending Decode master 和任何 transaction reservation，再按 `(used_blocks, used_tokens, sp_rank)` ascending 逐个 exact 尝试。

防抖计数语义固定：只有“全局无 valid admission、已经进入 maintenance check”的 step才更新。candidate identity 为 `(dp, group_id, ordered member IDs, current allocation, compute_floor)`；identity与上一 eligible check相同且 raw utilization仍低于 threshold时 `stable_steps++`，新 identity从1开始，utilization回升、`DoP <= compute_floor`、membership/allocation改变时清零。execute还要求 global schedule step命中 check interval，且距 `last_scale_up_step` 和 `last_consolidation_step` 都至少 cooldown steps。exact plan reject保留 stable count但不更新 cooldown/epoch；成功 commit才清零并写 `last_consolidation_step`。

每个 source plan 固定：

1. 一次只 evacuation 一个 source rank，全局每个 `schedule()` 最多一份 plan，group 至少保留一个 rank且不跨 SP8 pool；
2. retained destination 按 `(available exact capacity descending, used KV ascending, sp_rank ascending)`，sequences 按 canonical group sequence order填充；
3. shadow placement 必须通过 block capacity、destination high-watermark、receiver metadata、pending/master headroom和下一轮 Decode exact validation；
4. source-block budget 和 migration chunk 只限制一次 transport，不得改变 candidate group/source 顺序；超限或 exact no-fit 时 abort reservation，保持 canonical allocation并继续本轮正常 admission/Decode；
5. plan 成功后返回独占 `KV_CONSOLIDATION`：先 P2P copy prepared ranges，全部 worker确认后再 no-throw swap ACTIVE metadata、释放 source blocks、从 canonical allocation 删除 source rank并重建 derived owner；
6. commit 后 source 才成为 truly idle，`pool_resource_epoch[dp]` 增加一次；下一 scheduler call 才 admission/Decode或继续释放另一个 rank。

RESERVED plan publication 前的确定性 planner rejection可以 `abort_noexcept()` 并继续；P2P 已 dispatch 后若 worker completion 不确定、copy 或 metadata/source-release commit 失败，则以 `KV_CONSOLIDATION_FAILED` 永久 latch，保留 transaction guard，不能假设 source/destination 哪一侧可回滚。

candidate 唯一 policy trigger 是 low-KV threshold。`stable_steps/cooldown/check_interval` 只是防止 scale-up/scale-down 抖动和连续 maintenance，destination watermark/source-block budget/migration chunk只是 exact transport safety；fresh waiting/pending benefit、gap、age、planner failure、跨 group或 multi-source event都不能产生或加速 candidate，也不能绕过 threshold/gates。

zero-live cleanup 仍先于本节执行：finish/abort/OFFLOAD 后立即释放 `live KV == 0` 且没有 pending/master/reservation 的 rank，不做 P2P。capacity/admission merge 将 donor ownership 改到 survivor，idle-rank scale-up 将 NONE 改为 group，zero-live cleanup和成功 low-KV consolidation才将 rank改回 NONE。

### 9.4 Scale-up 固定开启

formal baseline 固定并校验：

```text
ls_disable_scale_up = false
```

LoongServe 的 `disable_scale_up=true` 会替换整个 Decode scheduler，并非只关闭 add-rank 分支。为避免额外维护第二套 Decode path，本设计不实现该 sensitivity；需要时另立 source-switch variant。

### 9.5 小 DoP group 是否主动合并

LoongServe 不会因为两个 group 都小就无条件合并成大 DoP。

running Decode 中只保留两种扩展触发：

```text
KV capacity deficit
or compute-bound 且存在 idle ranks
```

admission 可以按第 7.1 节 capacity append 规则 merge groups；除此之外，仅完成 capacity-aware batching 不会自动得到“小 DoP group 合成大 DoP”的行为，也不补主动 merge heuristic。

## 10. OFFLOAD pause 和 readmission

当前 restart-from-prompt 会丢弃已经生成的 token，并改变服务量和排队状态，不适合作为 baseline。

baseline 只实现：

```text
PAUSED_OFFLOAD:
    保留 output/generated/sampling/metric progress 和 assigned_dp
    清空该 request 的全部物理 KV 与旧 batch/group membership
    admission_need_tokens = prompt + generated
```

触发仅限：第 9 节 mandatory capacity merge 和 idle-rank扩展后，pool 仍不能形成安全 Decode plan。它不能由 admission no-fit、waiting age、gap 或 compute pressure触发。

若同一 global scheduler step 有多个 pools 同时 mandatory no-fit，只对最小 `dp_idx` 的 deficit pool commit 一个 victim并立即返回；其他 pools 下一 step 重算。这是 Nano FFN global cadence 下将多个 Loong worker pause events 串行化的 execution adapter，不改变 pool 内 victim policy。

victim 使用 LoongServe `Fcfs` pause strategy 的确定性语义：candidate domain 是发生 unresolved mandatory capacity deficit 的该 pool 中全部 real `RUNNING` requests，按 `(arrival_order descending, seq_id descending)` 检查，即优先保护更早到达的 request。candidate 在清 KV 前必须同时满足：

```text
MIGRATE/SWAP contexts 为空
full-SP8 empty-system future-token policy fit
full-SP8 exact readmission fit(prompt + generated + one dummy headroom)
```

不满足则继续检查下一个 victim。若没有 recoverable victim，抛第 8.6 节 `UNRECOVERABLE_CAPACITY` fatal error，保持全部 requests/KV 不变；不能清掉一个以后无法 readmit 的布局，也不能无限 preempt loop。一次 `schedule()` 最多 commit 一个 victim，立即返回携带 OFFLOAD record 的 scheduler-only `ADMISSION`；本调用不再 admission/readmission或 Decode，下一次调用才重新 planning。这与 LoongServe pause 后结束 `_step()` 的形状一致。

OFFLOAD 使用 prepare/no-throw-commit，而不是承诺释放 block 后还能恢复原 block IDs：

1. prepare 完成第 8.5 节 `PreparedBlockMutation`、一个可 splice 到 queue head 的 list node、map/container capacity、victim/group校验和 readmission precheck；此阶段失败不改稳定状态；
2. assert 只有 ACTIVE 持有 allocation，MIGRATE/SWAP 为空；
3. commit point 只调用 prepared ACTIVE release 的 `commit_noexcept()`，一次性释放 KV、pending reservation 和 running-token accounting；不得调用现有可能创建 free-list node 的 `deallocate()`；
4. commit point 之后只执行 no-throw move/erase：保留 `token_ids`、generated count、last token、sampling params和既有 metric timestamps，标记 `PAUSED_OFFLOAD` 并插入原 pool 队首；
5. 删除 victim membership 后执行统一 zero-live-KV cleanup：已无 live KV/pending/master/reservation 的 rank 变为 NONE；group 为空时删除其全部剩余状态，非空 group 只保留仍承载状态的 ranks；
6. commit 前失败可撤销 prepare；commit 后若违反 no-throw invariant，engine fail closed，不尝试伪造已释放 KV 的 rollback。

readmission 从下一次 `schedule()` 起接受同一 pool 的 FIFO/OOE/future-KV scan和普通 ephemeral batch transaction，重新放置 `prompt+generated` tokens，再恰好追加一个固定 ID 0 的 dummy pending token；该固定 token 不调用 sampler。它不推进 RR、不重置 first-token/queueing metrics，也不调用现有会清零进度的 `SequenceMetric::on_preemption()`；它按第8.2节 prepared metric mutation增加generated、更新last-token并把pause/readmission间隔保留为ITL sample。

不再创建优先级高于普通 request 的长期 singleton recovery batch。

## 11. 冻结配置

正式 baseline 把 LoongServe 本身已有的阈值与一份冻结的 low-KV execution-adapter profile 分开记录：

```text
ls_max_num_ooe = 10
ls_running_max_req_size = 1000
ls_admission_max_tokens_per_pool = auto
ls_min_comp_bound_decoding_batch_size = 128
```

以下是 baseline 常量，只写入 validation/manifest，不新增可切换 policy branch：

```text
ls_decode_enable_future_kv_admission = true
ls_decode_initial_kv_dop = 0
ls_disable_scale_up = false
ls_kv_consolidation_mode = execute
ls_kv_consolidation_candidate_util = 0.50
ls_kv_consolidation_target_high_watermark = 0.80
ls_kv_consolidation_stable_steps = 2
ls_kv_consolidation_cooldown_steps = 2
ls_kv_consolidation_check_interval_steps = 1
ls_kv_consolidation_max_source_blocks_per_event = 128
ls_kv_consolidation_migration_chunk_tokens = 64
pause_mode = offload
dp_assignment = arrival_round_robin
cross_dp_scale_up = false
```

其中：

- `ls_decode_initial_kv_dop=0` 表示最小 exact feasible；正式 baseline 禁止强制值；
- future-KV、OFFLOAD-only、arrival RR、min-exact DoP、low-KV consolidation execute 和 cross-DP off 均由 baseline mode 固定；`off/shadow` 只用于开发验证，不能作为 formal baseline；
- consolidation candidate 只由 `group_kv_util < 0.50` 产生；formal profile固定使用已跑过压力实验的防抖/限频值`2/2/1`，它们不是LoongServe source parameter；
- `0.80/128/64` 分别是 destination safety watermark、单 event source-block budget和预留 transport chunk；它们只允许 reject/限频，不能改变 group/source选择顺序；
- global `routing_strategy` 在 LS path 不参与 DP 选择，正式脚本仍固定为 `RoundRobin` 以避免 manifest 歧义；
- `ls_admission_max_tokens_per_pool=auto` 在启动时一次解析为 `max(max_model_len, total_pool_kv_tokens / 6)`，运行中不自适应；
- 所有 resolved values、固定运行条件、topology 和 `max_tokens>=1` ingress contract 写入运行 manifest。

配置只允许以下三个显式 profile：

| Profile | `max_num_ooe` | Decode threshold | 用途 |
|---|---:|---:|---|
| `loong_decode_source_default` | 10 | 100 | API default conformance |
| `loong_decode_artifact_derived` | 必须由 workload manifest 显式给值 | 128 | 对齐 artifact 参数形状 |
| `loong_decode_issue001` | 必须由正式 manifest 显式给值 | 128 | Nano 正式 Issue 1% 实验 |

参数 profile 变化不代表算法变化，但一次运行不能隐式混用 config、benchmark 和 CLI 三套默认值。

初始化只发生在 engine construction：`next_dp_rr=0`、`next_arrival_order=0`，每个 pool 的 `num_ooe=0`、`pool_resource_epoch=0`，所有 waiting/group/arrival/seen registries为空，consolidation stable/cooldown state与 active plan为空，`ls_fatal_` 未设置。drain 到空、profile 切换或 benchmark phase 切换都不得隐式重置；需要新序列时必须新建 engine并生成新 manifest。

## 12. 代码改造范围

### 12.1 `scheduler.h/.cpp`

- 删除 fresh request 的 persistent seal/pending ownership；
- arrival-time round-robin assignment、scheduler-private list-based `waiting_by_dp` 和 `arrival_order_by_seq_id` lifecycle；
- 每个非 OFFLOAD scheduler step 扫描各 pool，并维护 pool-local OOE counters；
- request-level current/future scan；
- selected 后 stable `admission_need_tokens` sort；
- ephemeral continuous partition；
- canonical group allocation + derived unique `rank_owner`；
- idle-capacity branch：无 raw deficit 时 standalone，有 deficit 时 ordered capacity append/merge；
- available-rank ordering、initial DoP search 和 packed token interval planning；
- current no-fit 时保持 waiting；
- commit 时才创建 batch/group ID；
- SP8 pool-wide future envelope；
- dummy bootstrap 纳入 pool-local admission transaction；
- typed `LSAddResult`/`LSAdmissionRecord` 与 eligible Decode plan 正交返回；
- `LSPoolStepTransaction`/`LSDecodePlanTransaction` prepare/validate、decode-only fallback、no-throw commit/abort；
- Decode memory/compute scale-up 收敛到 source-shaped 规则；
- low-KV-only candidate、stable/cooldown/check gates、canonical group/least-KV source顺序和全局单 plan publication；删除 pending/no-fit pressure及 admission-benefit bypass；
- 只实现每 step 至多一次的 OFFLOAD pause/readmission，并保留 generated progress；
- typed `LSSchedulerFatalError` 与 permanent scheduler fatal latch；
- `KV_CONSOLIDATION` 与 admission/OFFLOAD/pool transaction互斥，RESERVED期间所有 mutation API fail closed。

### 12.2 `block_manager.*` 与 `sp_state_manager.*`

- `BlockManager::PreparedBlockMutation` 的 prepare allocate/release、noexcept commit/abort 和 transaction-owned free-list nodes；
- 接收 packed token intervals；
- intervals 到 Nano block counts/receiver metadata 的 exact adapter；
- pending-token headroom 和 pinned rank-range validation；
- prospective group-union/empty-rank/master plan validation；capacity merge 不迁移历史 KV；
- `prepare_ls_initial_batch()`/`prepare_iteration_master_plan()` 构造 shadow contexts 和 prepared block mutations；formal LS commit 不调用现有 mutating allocator API；
- `LSKVConsolidationPlan` 的 exact destination reservation、canonical sequence/rank move ranges、P2P 后 no-throw ACTIVE metadata/source-release publication及安全 abort；
- 增加 adapter rejection reason；
- 支持 admission side effect 与已有 Decode plan 共存。

### 12.3 Sequence、binding 和 metrics

- `csrc/nanodeploy/sequence/sequence.h/.cpp`、`serialization.*`：增加默认 `-1` 的持久 `assigned_dp`；`PAUSED_OFFLOAD` 紧邻 `_COUNT` 前追加，保留所有既有 enum numeric ordinals；raw/pickle schema 与 workers lockstep升级，不承诺 mixed-version/legacy-payload compatibility；
- `csrc/python/sequence_binding.cpp`、`csrc/python/scheduler_binding.cpp`：同步 enum、typed add/admission records 和 result binding；
- `csrc/nanodeploy/metrics/sequence_metric.*`、`nanodeploy/metrics.py`：preserve-progress pause path和 provisional metric ticket prepare/commit/abort；不调用 reset-style `on_preemption()`；
- `BlockContext.dp_idx_` 从 `Sequence.assigned_dp` 派生，reset/readmission 后恢复并校验一致。

### 12.4 Python/config

- `nanodeploy/config.py`：固定 Decode-only topology、low-KV execute adapter参数和 profile；
- `nanodeploy/engine/scheduler.py`：构造参数；
- `nanodeploy/engine/llm_engine.py`：删除 Python dummy append；按 metric ticket → `Scheduler::add()` → commit/abort 顺序 ingress；在 Decode arrays 前处理 `KV_CONSOLIDATION` coordinator与 admission records/ADMISSION early return；在 zero-token throughput 分支前消费 outputs；捕获并 latch LS fatal；
- `nanodeploy/engine/kv_consolidation.py`、`ray_executor.py` 和 worker P2P：只执行 scheduler 返回的冻结 plan，不按 group/source二次 planning；copy completion不确定时 fail closed；
- request ingress：校验 `ignore_eos=true`、`max_tokens>=1` 和 singleton empty-pool fit；
- benchmark script：输出完整 resolved manifest。

### 12.5 不修改的内容

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
- typed `LSAddResult`/`LSAdmissionRecord` binding fixtures；
- Issue 1% 六个长短混合窗口；
- resolved config manifest schema。

### Phase 1：Fresh-request admission pipeline 一次切换

改动：

- 删除 fresh persistent pending batch；
- arrival-time round-robin assignment；
- pool-local FIFO/OOE scan；
- current/future KV 先筛选 selected membership；
- stable `admission_need_tokens` sort；
- exact no-fit 时按 FIFO scan 顺序回退 membership 并重新 planning；
- 每 pool 每轮一个 ephemeral continuous range；
- 每个 SP8 pool 独立的 future envelope；
- available-rank ordering 和 initial DoP search；
- canonical group allocation/derived owner；按 raw idle-token deficit 二选一 standalone 或 ordered capacity append；
- packed token intervals；
- interval 到 blocks/metadata/headroom 的 exact adapter；
- `PreparedBlockMutation`、list/map/container staging 与 no-throw commit/abort；
- admission + stable Decode 的 `LSPoolStepTransaction`/`LSDecodePlanTransaction` core，以及 admission failure 后 decode-only fallback；
- prompt/readmission allocation 与 dummy bootstrap 同 transaction 原子 commit/rollback；
- rollback 后恢复 waiting order、OOE、metrics 和 group allocation；monotonic ID 可留 gap；
- 消除跨 group future capacity 重复承诺；
- ingress permanent no-fit rejection；
- typed add/admission records、per-DP real/running ID ABI 与 eligible step-entry real Decode + required dummies 同时返回；
- engine ADMISSION early return、zero-token output顺序和 typed fatal latch；
- 完整 telemetry。

Phase 1 完成前不启用任何 LS 新路径，也不保留“新 batching + 旧均匀 striping”的实验配置。Phase 1 通过后只允许 dev/CPU 验证；formal baseline 至少要等 Phase 2 完成且第 16.1 节 CPU fixtures 全部通过。实现过程可以拆成可回溯的小提交，但语义上只做一次切换。

### Phase 2：Decode elasticity

改动：

- memory-deficit donor merge；
- exact idle-rank scale-up；
- integer-floor compute threshold；
- low-KV-only exact consolidation、单 source-rank P2P transaction和防抖/transport safety gates；
- 删除 pending/no-fit pressure、admission-benefit proof及其对 threshold/stable/cooldown的 bypass；
- 删除 baseline 中额外 merge；
- OFFLOAD-only preserve-progress pause/readmission；
- 将 Phase 1 transaction core 的 mutation set 扩展到 memory merge/idle scale-up/OFFLOAD，并加入 recoverable-victim precheck；consolidation保持独立 stop-the-world transaction。

### Phase 3：验收和清理

- 删除临时双 policy；
- 旧 persistent-batch 文档标记 historical；
- 固定 8/16 GPU scripts；
- 完成 source-shaped CPU differential tests；
- 正式 Issue 1% A/B。

每个逻辑单元及时提交，方便回溯；Phase 1 admission pipeline 只启用 dev path，Phase 2 transaction mutations 与全部 CPU fixtures 完成后才整体启用 formal baseline。

## 14. 不变量

1. request arrival 时 round-robin 固定 `assigned_dp`，之后不做 load-aware rerouting。
2. 每个 pool 的 selected membership 由本地 FIFO/OOE 决定，`admission_need_tokens` 排序不能改变本轮服务资格。
3. request 只有 admission commit 成功后才能从所属 waiting queue 移除。
4. planning no-fit 不发布 batch/group identity 或 ownership；reserved monotonic ID 可留 gap。
5. 同一 request 的稳定状态只能是 waiting、running、paused-offload、finished 之一；planned 只是栈内对象。
6. OOE 只能按 final committed membership 更新；no-commit 和 rollback 不改变 counter。
7. future envelope 按 request ID 去重，包含所属 pool 全部 running、tentative 和 candidate；未选中 OFFLOAD 只计 request slot，不扣 token capacity。
8. `allocated_attention_ranks` 是 ownership 唯一真源，派生 `rank_owner[dp][sp]` 必须唯一；无 raw idle-token deficit 时 admission 只走 standalone，有 deficit 时只走 ordered capacity append/merge。
9. group DoP 不得超过 8，不能跨 DP pool merge/scale-up。
10. Nano exact gate 只能 reject/replan，不能静默改变 selected membership；membership shrink 只能按 FIFO selection order。
11. admission/optional mutation 在 publication 前失败必须 `abort_noexcept()` 并恢复 queue、blocks、ownership、OOE 和 metric；monotonic ID 不要求无 gap，validated publication 不再设置 fault point。
12. 纯 admission 不减少 eligible step-entry real requests 的 Decode iteration 数；new admitted 当步不参加 Decode，collective dummies 仍完整。OFFLOAD 和 low-KV consolidation 是显式独占 maintenance，不能与本不变量混淆。
13. memory merge 只由 capacity deficit 触发。
14. compute scale-up 使用整数 floor threshold，只消费 admission logical reservation 后剩余的 truly-idle ranks。
15. OFFLOAD/readmission 保留 generated tokens、sampling params、metric progress、arrival order 和 canonical `assigned_dp`；只有通过 empty-pool exact recoverability precheck 才能清 ACTIVE KV；每次 schedule 最多一个 victim且同调用不 readmit。
16. consolidation candidate 只能由持续 low-KV utilization 产生；pending/no-fit pressure、admission benefit、gap、age、planner failure和跨 group状态都不能产生或加速 candidate。每个 action最多释放一个 source rank且不能低于 source compute floor。
17. formal baseline 的 `ScheduleAction` 可返回 `ADMISSION`、`DECODE` 或独占 `KV_CONSOLIDATION`；三者 field matrix互斥，consolidation全局每 step最多一份 RESERVED plan。
18. user metrics 只使用 typed real/running IDs；collective dummies 永不计入 running、batch、token、throughput 或 completion。
19. 任一 fatal code 永久 latch scheduler/engine；不得在同一进程继续 schedule。
20. 非 LS scheduler 行为不变。

## 15. Telemetry

### `ls_decode_dp_assignment`

- request ID；
- arrival order；
- assigned DP/pool；
- round-robin counter before/after；
- singleton validation result/reason。

### `ls_decode_dispatch_scan`

- DP/pool ID；
- FIFO snapshot IDs；
- scanned/selected/deferred/frontier IDs；
- per-request reject reason；
- OOE before/tentative/final committed after；
- first blocker 和 final bypass IDs；
- selected admission token sum；
- scan trigger（baseline 为 `every_non_offload_step`；OFFLOAD scan-before-return 和 scan 后 consolidation decision明确记录，无 event-cache 字段）。

### `ls_decode_batch_plan`

- FIFO-selected request IDs；
- membership rollback request IDs 及顺序；
- 最终 sorted request IDs/admission need tokens；
- empty-system fit result；
- current-system fit result；
- committed members；
- uncommitted members。

### `ls_decode_future_kv`

- DP/pool ID；
- identity-deduplicated running/tentative/candidate request IDs；
- per-request `(base_tokens, remaining)`；
- unselected OFFLOAD IDs 和 request-slot count；
- peak tokens；
- per-rank `blocks().size()`、resolved pool token capacity 和 request-count limit；
- token-policy result；
- current exact adapter reject reason。

### `ls_decode_initial_placement`

- canonical group allocations、derived owner、truly-idle ranks；
- `target_kind`、raw idle capacity/deficit、group slack order、planned donor IDs；
- post-bootstrap surviving IDs、actual-overlap merged donor IDs/survivor；
- 每个 standalone `d` 或 capacity target 的 placement ranks；
- placement ranks 的 used-token 降序 packing order；
- per-request token intervals；
- converted block counts 和 pending-token headroom；
- receiver/master metadata counts；
- exact reject reason 或 committed initial DoP。

### `ls_decode_admission_transaction`

- DP/pool、transaction ID 和 final member IDs；
- target kind/ranks、merged group IDs、batch/group ID before/after；
- allocation/bootstrap/publication stage；
- injected/real failure reason；
- commit/rollback result；
- queue/OOE/canonical-group state hashes before/after；
- reserved ID gap（若有）和 `bootstrap_finished` output IDs。

### `ls_decode_plan_transaction`

- prospective capacity merges/empty-rank additions；
- eligible real snapshot IDs 和 required dummy IDs；
- pool resource epoch before/after 与 prospective reservation fingerprint；
- validate/commit/rollback stage 与 failure reason；
- group/context/block/running-counter state hashes before/after。

### `ls_decode_consolidation`

- candidate DP/group、canonical order index、group utilization/threshold和 compute floor；
- stable/cooldown/check state及唯一 decision reason；
- source candidates 的 `(used_blocks, used_tokens, sp_rank)`、排除原因和最终 source；
- retained/destination ranks、per-rank exact capacity/high-watermark和 canonical sequence order；
- reserved destination blocks、P2P token ranges、source-block count和 migration chunk；
- transaction/resource epoch、plan state、exact reject/abort/commit/fatal reason；
- P2P/maintenance wall time、最终 canonical allocation和释放的 truly-idle rank。

### `ls_decode_schedule_result`

- step-entry running IDs；
- per-sequence committed admission records（kind/target/planned placement/bootstrap-finished/per-sequence owner）；
- eligible real Decode IDs 和 collective dummy IDs；
- `ls_running_ids_by_dp_after_commit`；
- paired `ls_preempted_sequence_ids/ls_preemption_reasons` OFFLOAD records；
- nullable consolidation plan ID/state；仅 `KV_CONSOLIDATION` 时为恰一 RESERVED plan；
- `ScheduleAction`；
- bootstrap-finished output IDs。

### `ls_decode_iteration`

- group IDs；
- KV DoPs；
- master DoPs；
- used/free tokens；
- capacity-deficit groups；
- donor groups；
- added idle ranks；
- low-KV candidate/source、released rank和 consolidation stall；
- compute threshold decisions；
- pause victim/readmission IDs；
- per-victim empty-pool recoverability check；
- preserved generated count、released ACTIVE KV、queue reinsert position和 fail-closed reason。

每步另记录 scheduler wall time、exact-plan count、oldest waiting age 和 waiting request count，用于第 16.4 节非功能验收。

`pool_resource_epoch[dp]` 只在 committed block ownership、canonical group allocation、pending/master、admission/finish/OFFLOAD status mutation或成功 consolidation publication 后增加一次；consolidation reservation、planner attempt、NO_FIT、rollback、monotonic ID reserve 和纯 telemetry 不递增。单次调用内 exact-attempt identity 冻结为：

```text
LSExactPlanKey = (
    dp,
    ordered membership IDs,
    ordered candidate ranks,
    pool_resource_epoch at prepare entry,
    prospective transaction-reservation fingerprint
)
```

这里不建立跨 step event cache；key 只用于断言/telemetry，防止同一控制流无状态变化地重复求解。
`prospective transaction-reservation fingerprint` 是以下规范化内容的确定性 hash：按 `(sp_rank, block_id)` 排序的 prepared block IDs/counts、shadow group/rank unions、per-sequence pending/master targets；不得包含 transaction/attempt/monotonic ID、对象地址、时间戳或随机盐。

## 16. 测试计划

### 16.1 CPU tests

1. Initialization/RR：只在 engine construction 将 `next_dp_rr/next_arrival_order`、各 pool OOE/epoch 置 0，queues/seen/fatal 为空；drain/profile 切换不重置。连续 requests 映射 DP0、DP1、DP2、DP3、DP0。
2. Add identity：singleton-rejected request 消费 RR，duplicate/already-assigned 不消费；同一已分配对象 duplicate result 返回既有 DP，不同对象复用 seen ID 返回 `-1`，singleton reject 返回实际消费的 DP；accepted、rejected、finished ID 都进入 engine-lifetime seen registry且不能复用；`seq_id` 在 LS assignment 后不可由 Python 修改。
3. Metric ticket：identity precheck 在 ticket 前；accepted 恰好 commit 一份 metric及 arrival/decode-arrival/prompt counters，fresh bootstrap幂等写入 first-scheduled/decode-scheduled/first-token，OFFLOAD readmit保留旧时间点；singleton reject abort provisional entry且不覆盖旧 metric；enqueue 后 injected metric-commit failure进入 permanent fatal。
4. Ingress boundary：`ignore_eos=false`、`max_tokens<1`、future/current singleton no-fit 分别返回 typed `UnschedulableRequestError`；不进入 waiting/owner/arrival map、不触碰 OOE。
5. Output boundary：fresh `max_tokens=1` 直接 `bootstrap_finished` 且不调用 model；fresh `max_tokens=2` 只再 Decode 一次。`ADMISSION` early return 不访问 Decode arrays，`generate(use_tqdm=true)` 在 zero-token 分支前仍消费 completion output。
6. Assignment/serialization：`assigned_dp` 默认 `-1`；raw/pickle 当前 schema round-trip `assigned_dp`/`PAUSED_OFFLOAD`，legacy payload 明确不支持；context reset/OFFLOAD/readmission 后派生 `dp_idx` 一致且不推进 RR。
7. Pool-local FIFO：可运行 requests 保持所属 queue 的服务顺序。
8. Pool-local OOE：一个 pool 的 blocker/counter 不影响其他 pool。
9. OOE commit semantics：真实 bypass commit 增加；exact shrink 删除全部后项不增加；正常 prefix commit reset；no-commit/任意 rollback 保持 before。
10. Every-step opportunity：new arrival 在下一个 non-OFFLOAD planning step 即进入所属 pool scan；若先发生独占 OFFLOAD，则在其后的首个 non-OFFLOAD step 扫描。
11. Membership-before-sort：排序不改变 selected IDs；fresh 使用 prompt，OFFLOAD 使用 prompt+generated。
12. Stable admission order：相同 `admission_need_tokens` 保持本 pool FIFO。
13. One batch per pool：一个 step 每 pool 最多提交一个 fresh/readmission batch，并完整覆盖排序数组的一个连续 range。
14. FIFO-order shrink：exact no-fit 撤销 FIFO scan 中最后加入的 request，不删除排序后尾部；未提交 requests 保持 waiting 顺序。
15. No persistent identity：current no-fit 后无 batch/group ownership、map entry或 committed telemetry；预留 ID 可留无语义 gap。
16. Future identity/capacity：running/tentative/candidate 同 ID 只计一次，未选中 OFFLOAD 只占 request slot；pool-wide peak 使用实际 blocks×64且不重复扣 fixed/headroom。
17. Pool isolation：DP0 full 不借 DP1 capacity，group KV/master DoP 最大为 8。
18. Canonical owner/cleanup：standalone、capacity/memory merge、scale-up、zero-live cleanup、low-KV consolidation、whole-group delete 后 canonical allocation与派生 owner一致；finish/OFFLOAD 释放 zero-live且无 pending/master/reservation的 ranks，不迁移 live KV；consolidation stable-erase source且不重排 group。
19. Admission target：idle raw capacity 足够时只 standalone，exact no-fit 走 FIFO shrink；仅 raw deficit 时按 canonical pool order stable-sort slack ascending再 pop-back到首次覆盖 deficit的 donor sequence，并在固定 donor set做 min-exact search；exact failure不追加 donor。
20. Admission survivor/order：有 surviving new request 时 new group ID 恒为 survivor；new survivors 在前、actual donors 按 selection order及各自原 order追加。全部 `bootstrap_finished` 时 standalone/capacity都不留group/merge，部分完成时 finished record owner为 null、survivors共享 new group。
21. Packed intervals/exact adapter：rank/packing stable，token 守恒，interval无重叠缺口；block/metadata/headroom reject只触发 FIFO shrink/replan。
22. Prepared allocator：mutation/pool transaction均为 move-only RAII，PREPARED析构自动幂等 abort；allocate/release、shadow context、list/map node各 prepare fault均恢复 exact state；commit路径不分配、不调用旧 mutating APIs且无 fault point。
23. Admission failure fallback：step-entry running A + tentative B，B policy/prepare/combined failure后 OOE不变并在 stable graph重建 decode-only，A 当步仍 forward；只有 decode-only safety/internal error走 OFFLOAD/fatal。
24. Cross-pool admission：各 pool先 prepare required Decode；某 pool admission rollback不撤销其他 valid admission，且失败 pool仍 Decode；global IDs唯一可有 gap，任一 required Decode internal failure发生在全局 publication前。
25. Simultaneous ABI：`admitted={B}`、real `decode={A}`，collective dummies完整但 B 不是 real Decode member；`ls_running_ids_by_dp_after_commit` 同时包含 A/B，B下一 step才 Decode。
26. Real/dummy metrics：ADMISSION early return前用 after-commit IDs更新 running gauge并刷新 waiting/paused gauges；batch size、token usage、throughput、普通 completion只用 real Decode IDs，所有 collective dummies计数为0。
27. Idle/no-progress action：empty engine直接 `step()` 在 scheduler前返回 non-fatal API misuse error；idle且有 admission返回有 record 的 `ADMISSION`；有 outstanding work却无 real Decode、admission/OFFLOAD或有效 consolidation plan时抛 `NO_PROGRESS_INVARIANT`，不产生空 action/busy loop。
28. Memory safety source order：cannot/can lists按 pool order stable-sort idle ascending；donor从 can尾部 pop，feasible union append tail且不 re-sort，constrained ID/原 sequence order在前、donors按 pop/union order追加；不足时按 deficit ceiling取 `sp_rank asc` idle ranks。
29. Decode transaction failure classes：admission/optional component fault可 abort并 decode-only；required decode prepare/validate fault typed-fatal且全局无 publication；validated commit无可恢复 fault point。
30. Compute scale-up：groups按 prospective can-list顺序争用 idle ranks，每组使用 eligible step-entry real count做 floor threshold，排除 new admissions，并按 `sp_rank desc` 取 rank；admission reservations优先。
31. Low-KV trigger/gates：只有 RUNNING group的 utilization严格低于冻结 threshold且连续满足 stable/cooldown/check gates才成为 candidate；pending/no-fit/admission benefit、gap、age和planner failure不能产生或加速 candidate，formal manifest固定 execute参数。
32. Consolidation order/floor：按 `dp asc`、canonical group order选 group，按 `(used blocks, used tokens, rank)`选非 master/reserved source，destination按 exact available capacity优先；每次只释放一个 rank、group至少留一个rank，且不得低于第9.2节 compute floor。
33. Consolidation priority/action：任一 pool有 valid admission时不 consolidation；无 valid admission且exact plan成功时全局只返回一个独占 `KV_CONSOLIDATION`。field matrix中只有一个 RESERVED plan，其余 admission/OFFLOAD/real/running/Decode fields全空；plan reject/超budget保持状态并正常 Decode。
34. Consolidation transaction/fatal：prepared destination blocks和move ranges在P2P前不改ACTIVE metadata；全部copy成功后no-throw publication、stable-erase source、epoch恰加1，下一Decode不再使用source。pre-dispatch abort完整恢复；dispatch后completion ambiguity/copy/commit failure永久 latch `KV_CONSOLIDATION_FAILED`且RESERVED guard禁止继续。maintenance返回zero token/output/completion，stall计入下一Decode wall time并单独记latency。
35. OFFLOAD recoverability：candidate domain为 deficit pool全部 RUNNING，按 `(arrival_order desc, seq_id desc)` 跳过 MIGRATE/SWAP 或 full-SP8 future/current exact不可恢复者；无 candidate抛 `UNRECOVERABLE_CAPACITY`且状态不变。
36. OFFLOAD commit/action：一次 schedule 全局最多一个 victim；多 pool deficit时选最小 dp_idx。prepared ACTIVE release只执行一次、做zero-live cleanup、队首 splice，立即以含 OFFLOAD record 的 `ADMISSION` 返回；同调用不 readmit也不 Decode。
37. OFFLOAD progress/readmission：保留 output/generated/sampling params/first/queue metrics/arrival order/assigned DP；下一 schedule 按普通 FIFO/OOE放置 prompt+generated并追加恰好一个 dummy token，prepared更新generated/last-token并追加pause间隔ITL，达到上限走 `bootstrap_finished`。
38. Fatal latch：typed exception、required Decode unexpected、consolidation failure、post-publication unexpected和accepted metric failure分别映射到冻结 code；全部 scheduler fatal codes都记录code/state hash、令后续add/step/generate拒绝；ingress `UnschedulableRequestError`不latch。
39. Exact-key/epoch：epoch只在冻结的 committed resource mutations（含 consolidation publication）递增；attempt/reservation/reject/rollback/ID reserve不递增；fingerprint只含规范化内容，同一 schedule相同 `LSExactPlanKey`只求解一次。
40. ScheduleAction：`DECODE`可同时携带admissions但OFFLOAD/plan必为空；`ADMISSION`至少有admissions或恰一OFFLOAD、二者不同时出现且plan为空；`KV_CONSOLIDATION`严格符合独占field matrix且不forward。
41. Feature-off：非 LS scheduler 的 action、state、seq-id mutability、metrics 和结果不受影响。

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
- placement、append、memory/compute scale-up correctness；强制 group 到 SP8 后降低 live KV/batch，至少成功 consolidation 一个 source rank并在后续 admission复用；
- P2P token/bytes守恒，publication后下一轮 Decode不再引用 released rank；
- 输出长度正确，无 planner failure loop。

16 GPU：

- DP2×SP8 / EP16；
- Issue 1%，rate 20，seed 0，7,200 requests；
- 141 GiB 和 140 GiB；
- `ls_kv_consolidation_mode=execute`，committed consolidation plan数必须大于0；
- 保存 admission transaction、combined result、capacity merge/scale/consolidation/OFFLOAD telemetry，结束时无 RESERVED plan或fatal；
- source-default 和 artifact-derived threshold 分组运行。

32 GPU（正式使用 `4DP×8SP` 时）：

- 验证四个独立 SP8 pools；
- request assignment 严格 round-robin；
- 每个 group 的 KV/master DoP 均不超过 8；
- DP0 capacity pressure 不触发跨 DP migration/merge/scale-up；
- consolidation P2P source/destination严格留在 candidate所属SP8 pool，四pool fixture至少各成功一次scale-down；
- 四个 pool 的 queue、OOE 和 future-KV telemetry 可独立核对。

GPU 测试必须按仓库规定申请提权。

### 16.4 正式 A/B 通过门槛

同一硬件、模型、request trace、seed、memory limit 和 resolved manifest 下，至少重复三次并报告 median；reference 固定为本文 Nano code baseline 上现有 LS-style path，不能跨配置挑最好结果。

硬 correctness gates：

- accepted requests 最终全部完成，只有 ingress 明确拒绝的 requests 可以不完成；
- lost、duplicate、cross-state owner、cross-pool owner 和 allocator invariant violation 均为 0；
- 输出 token 数与 trace 要求一致；`max_tokens=1` 只产生 bootstrap output，`max_tokens=2` 只执行一次 Decode；
- 7,200-request drain run 无 planner-failure loop、永久 frontier、未释放 admission/consolidation transaction或fatal；
- controlled elasticity fixture 从 SP8 scale-up后在低KV/低batch状态至少释放一个非空source rank，迁移token守恒，最终DoP低于8且released rank可复用。

实施效率 gate：

| 指标 | 通过条件 |
|---|---|
| scheduler overhead P95 | 不高于 reference 的 1.05 倍，且低于已观测旧路径 `115.38 ms` |
| exact planner 去重 | 单次 `schedule()` 对相同 `LSExactPlanKey` 最多求解一次；epoch 不得因 attempt/reject/rollback/ID reserve 递增 |

以下是必须逐 pool 报告的 A/B outcome，而不是改变 baseline 语义的通过条件：360 s 内 completions、arrivals 结束后的 drain time、oldest waiting age、waiting request count、每轮 exact-plan/replan count。arrival-time RR 和 every-non-OFFLOAD-step scan 本来就与当前 Nano reference 不同，不能用“必须优于 reference”倒逼 load-aware rerouting、cadence gate 或额外 heuristic 混入 baseline。

任一 correctness 或实施效率 gate 失败即不合格。失败时先定位 every-non-OFFLOAD-step scan、重复 exact adapter 调用或实现 bug；event cache、load-aware rerouting等只能形成单独 variant，不能反向改写 baseline 语义。

## 17. Baseline 完成标准

以下全部满足后，才能称为 LoongServe-style Decode-only baseline：

1. 运行中不存在 fresh persistent pending batch。
2. request arrival 按 round-robin 固定 `assigned_dp`；seen ID engine-lifetime 不复用，accepted arrival order 独立于可写的构造时 seq ID。
3. 四个 pools 的 FIFO/OOE/every-non-OFFLOAD-step admission 状态相互独立。
4. request membership 在本 pool `admission_need_tokens` 排序之前确定。
5. 每 pool 每轮最多一个 batch，且是排序数组的连续 range。
6. no-fit 时 request 保持在所属 waiting queue，没有 batch ownership。
7. future-KV 使用 identity-deduplicated SP8 pool-wide token envelope，并明确不声称 future block exactness。
8. canonical group allocation、pool group order与派生 rank owner一致；admission按 raw idle-token deficit二选一 standalone 或 source-list capacity append，不使用额外 target heuristic。
9. group 只能在本 pool 内 scale-up，DoP 不超过 8。
10. block/list/map prepared mutations、admission allocation、dummy bootstrap、Decode reservation 和 OOE 在同一 pool transaction no-throw commit/abort。
11. 已有 requests 不因纯 admission 丢失 Decode iteration，新 admitted 下一 step 才 Decode；OFFLOAD和low-KV consolidation都是全局每 step至多一个的独占maintenance。
12. packed placement 与 batching 同步启用，并通过 interval/block/metadata correctness。
13. memory-deficit source-list merge、integer-threshold compute和固定 rank/group遍历顺序通过 differential fixtures。
14. low-KV-only exact consolidation在formal base固定execute，并能让scale-up后的group回落；pending/no-fit benefit、gap、age、planner failure、cross-group和multi-source trigger全部禁止。
15. OFFLOAD/readmission 保留生成、采样、metric/arrival进度及 `assigned_dp`，prepared KV release 后回队首且同调用不 readmit。
16. ingress 永久不适配 request 明确返回错误，不占 waiting frontier 或 OOE。
17. resolved config、逻辑/物理 topology 差异和 intentional adaptations 写入 manifest。
18. typed real/running membership、ADMISSION-only output、`KV_CONSOLIDATION` field matrix/zero-token maintenance和dummy-excluded metrics通过engine tests。
19. admission/consolidation transaction、fatal ABI/latch、resource epoch/exact key和对应故障注入通过。
20. 对应实验规模的 8/16/32 GPU correctness 与第 16.4 节 A/B gates 完成。

性能改善不能替代调度一致性检查。正式实验必须同时保存 dispatch、batch、future-KV 和 Decode elasticity telemetry。

## 18. 已冻结的设计决策

| 决策项 | 冻结值 | 说明 |
|---|---|---|
| Baseline 名称 | `LoongServe-style Decode-only` | 不声称 source-identical |
| Persistent fresh batch | 删除 | exact commit 前保持 request-level waiting |
| Batching v1 | FIFO/OOE membership + stable admission-token sort + 每 pool 一个 batch | 不增加新 length heuristic |
| 二维 DP | 不进入 base | Decode-cost 独立 variant |
| Placement | packed + Nano exact adapter | 不保留均匀 striping 过渡 baseline |
| Admission target | raw idle capacity 足够则 standalone；不足则 ordered capacity append/merge | 只保留 LoongServe capacity 触发和顺序，不按 exact failure/age/gap 选 target |
| Admission continuity | 纯 admission 同 step保持 step-entry Decode；OFFLOAD/consolidation独占 | OFFLOAD复用`ADMISSION`；consolidation复用现有独占action，不新增组合action |
| Future-KV | SP8 pool-wide token policy | 不新增 future-block predictor |
| Decode threshold | Issue 1% 为 128 | 100 做 source-default sensitivity |
| Proactive consolidation | formal base固定low-KV-only execute | 补齐scale-down闭环；stable/cooldown仅防抖，禁止pending/gap/age等旁路trigger |
| Pause | 每 step 至多一个 OFFLOAD、preserve progress、同调用不 readmit | KVKEEP 不进入 base |
| Transaction adapter | prepared block/list/map mutation + pool-step atomic publication | 是 Nano allocator 适配，不是新 policy |
| DP topology | 4 个逻辑 SP8 pools | 物理 FFN EP32 仍同步 |
| DP assignment | arrival-time round-robin | benchmark 一 request 一 connection |
| Cross-DP scale-up | 禁止 | 单 group DoP≤8 |

## 19. 实施结论

按以下边界实施：

1. baseline 范围严格限定为 Decode-only；
2. 旧 persistent fresh-batch 设计不再约束实现；
3. Phase 1 一次完成 arrival-time round-robin、pool-local FIFO/OOE、admission-token stable sort、ephemeral batching、pool-wide future-KV、raw-idle-capacity 驱动的 standalone/capacity-append initial DoP 和 packed placement；
4. packed intervals 与 dummy bootstrap 通过 Nano block/metadata transaction 原子落地；
5. Phase 1 同时返回 committed admissions 和 step-entry Decode snapshot，不新增第二 GPU lane或组合 action；
6. Phase 2 完成 Decode memory-deficit merge、idle-rank compute scale-up、low-KV-only exact consolidation和OFFLOAD，形成scale-up/scale-down闭环；
7. 移到单独variant的仅是pending/no-fit benefit、gap/age/planner-failure/cross-group/multi-source consolidation、KVKEEP、event cache、Prefill gain/cost reclaim、unordered admission merge和二维DP；
8. `4DP×8SP` 固定为四个逻辑 attention/KV pools，禁止 load-aware rerouting 和 cross-DP scale-up，同时承认 FFN EP32 的物理同步；
9. 所有 Nano allocator/topology adapter 和 source deviation 写入 manifest；
10. 正式实验统一命名为 `LoongServe-style Decode-only`。

这样可以避免把当前实验不执行的内容带入设计，同时保留 LoongServe 对 Decode 资源管理最关键的思想，也能控制每个阶段的改动面和验证成本。

## 20. 源码与本地证据索引

LoongServe：

- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/req_queue.py:46`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/req_queue.py:80`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/req_queue.py:135`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/io_struct.py:110`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/io_struct.py:140`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/sampling_params.py:9`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/sampling_params.py:41`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/io_struct.py:188`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/pause_strategy.py:13`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/pause_strategy.py:35`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:351`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:686`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:764`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:516`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:539`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:587`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:591`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:613`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:617`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:637`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:678`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:844`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:975`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:1138`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:1146`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/api_server.py:388`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/longserve_c_scheduler/src/main.cpp:33`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/paper-tex-src/sections/design.tex:14`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/paper-tex-src/sections/design.tex:107`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/paper-tex-src/sections/design.tex:110`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/paper-tex-src/sections/design.tex:149`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/test/longserve/5-start-api-server.py:209`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/test/longserve/5-start-api-server.py:242`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/test/longserve/5-start-api-server.py:274`

NanoDeploy：

- `csrc/nanodeploy/scheduler/scheduler.h:27`
- `csrc/nanodeploy/scheduler/scheduler.cpp:924`
- `csrc/nanodeploy/scheduler/scheduler.cpp:1348`
- `csrc/nanodeploy/scheduler/scheduler.cpp:1488`
- `csrc/nanodeploy/scheduler/scheduler.cpp:1705`
- `csrc/nanodeploy/scheduler/scheduler.cpp:641`
- `csrc/nanodeploy/scheduler/scheduler.cpp:449`
- `csrc/nanodeploy/scheduler/scheduler.cpp:971`
- `csrc/nanodeploy/scheduler/scheduler.cpp:1826`
- `csrc/nanodeploy/scheduler/scheduler.cpp:2085`
- `csrc/nanodeploy/scheduler/scheduler.cpp:2368`
- `csrc/nanodeploy/scheduler/scheduler.cpp:2822`
- `csrc/nanodeploy/scheduler/sp_state_manager.cpp:488`
- `csrc/nanodeploy/scheduler/sp_state_manager.cpp:1527`
- `csrc/nanodeploy/scheduler/sp_state_manager.cpp:1815`
- `csrc/nanodeploy/sequence/sequence.h:18`
- `csrc/nanodeploy/sequence/sequence.cpp:20`
- `csrc/nanodeploy/sequence/serialization.cpp:168`
- `csrc/nanodeploy/metrics/sequence_metric.cpp:89`
- `csrc/python/sequence_binding.cpp:66`
- `nanodeploy/config.py:295`
- `nanodeploy/engine/llm_engine.py:214`
- `nanodeploy/engine/llm_engine.py:548`
- `nanodeploy/engine/llm_engine.py:104`
- `nanodeploy/engine/kv_consolidation.py:57`
- `nanodeploy/engine/ray_executor.py:258`

实验记录：

- `docs-dev/2026-07-17/ls_decode_future_kv_2node_r20_141gb_result_20260717.md`
- `docs-dev/2026-07-17/ls_decode_loongserve_capacity_alignment_20260717.md`
- `docs-dev/2026-07-18/ls_style_capacity_reorg_mem085_2node_result_20260718.md`
- `docs-dev/2026-07-18/ls_style_capacity_reorg_908933a_2node_dp2sp8_r20_140gb_mem085_6min_20260718.manifest.json`
