# NanoDeploy 全新分层式 Decode 调度模式 Proposal

状态：设计提案，已完成开发计划与 execution contract review，限定
`mode="decode", dummy_prefill=True, loop_count=16, ignore_eos=True`

日期：2026-07-23
Review：2026-07-24

调研基线：

- NanoDeploy：`8c9d518`
- 本地 vLLM：`a5d19cbb9`

## 1. Review 结论

### 1.1 新模式与旧分支彻底切割

本方案实现一个全新的 `scheduler_arch="hierarchical"` 模式，不复用、不继承，也不把旧的
`scheduler_mode="decentralized"` 重命名后继续使用。

旧分支只是一个进程内的分队列实验实现，不具备本方案要求的进程边界、状态 ownership 和
EP 执行协调，而且它的全局 `is_prefill` 会混淆不同 DP lane 的 admission 与 decode。
继续在该分支上修改会同时背负错误语义和兼容成本。

因此建议在开发开始时删除旧分支，包括：

- `Config.scheduler_mode` 中的 `"decentralized"` 取值；
- C++ `SchedulerMode::DECENTRALIZED` 及其条件分支；
- `_schedule_decentralized()`、`_schedule_prefill_for_worker()`、
  `_schedule_decode_for_worker()` 等仅供旧分支使用的代码；
- `SPStateManager` 中仅供旧分支使用的 per-worker waiting queue/helper；
- benchmark 脚本中的 `--scheduler-mode decentralized`；
- `LLMEngine` 中针对旧分支的日志兼容判断；
- 只验证旧 routing enum/config、但没有验证真实分布式行为的测试。

删除前应使用引用搜索确认 helper 没有被 centralized 路径共用。现有
`scheduler_arch="legacy_global"` 路径保留为基线和回滚入口。

### 1.2 增加单机 `DP2 SP4 EP8`

首个单机阶段必须支持并测试以下三种 8 卡拓扑：

- `DP8 SP1 EP8`
- `DP2 SP4 EP8`
- `DP1 SP8 EP8`

`DP2 SP4 EP8` 很重要：它在一台机器上同时覆盖“多个独立 LocalEngineCore/LocalScheduler”
和“每个 DP group 内存在 SP 通信”，比只测两个端点 `SP1`、`SP8` 更容易暴露 DP engine
隔离、rank slice、router 和 EP coordination 问题。

### 1.3 固定 `LoopCount=16`

本方案始终按 `loop_count=16` 设计和验收。每次 LocalScheduler 生成一个 decode quantum，
所有 EP rank 都固定执行 16 次 inner forward；不支持 DP engine 间不同值、运行时修改或按负载
自适应 loop count。

请求在最后一个 quantum 中即使只剩不足 16 个 completion token，也仍执行满 16 次
forward。LocalScheduler 只提交请求预算内的有效前缀，其余是 overrun work，不进入 completion
或 useful token throughput。为保证这 16 次执行安全，dispatch 前必须保证完整 quantum 的 KV
capacity，且请求长度校验按 16 向上取整。

### 1.4 dummy 场景不验证数值准确性

本方案的请求先经过 dummy prefill，decode 使用未由真实 prefill 生成的 KV。因此 token
值、logits、模型质量以及与 legacy 路径的数值等价都没有意义。

验收只关注：

- admission 不触发 prefill GPU forward；
- 所有 EP rank 以相同顺序执行相同次数的 decode forward；
- real/dummy batch 的 tensor shape、rank mapping、索引和 block table 结构合法；
- 请求状态能推进并完成；
- allocated request blocks 和 permanent control-dummy blocks 分账闭合；
- 不发生越界、collective hang 或资源泄漏；
- 调度和执行开销满足性能目标。

不做 token output 对齐、logits 对齐、metadata bitwise equivalence 或固定 golden output。

### 1.5 对原计划的收缩

| 原计划内容 | Review 决定 | 原因 |
|---|---|---|
| 保留或重命名旧 decentralized 分支 | 删除 | 新模式不应背负旧的 phase bit 和分队列语义 |
| 先做完整 `AdmissionPlan/commit/rollback` | 不做事务式协议 | single-writer loop 已足够；KV 不足时在 batch freeze 前完成本地 preempt |
| 新增 `RankBatchDescriptor`、修改或跨 RPC 传输 `DecodeMetadata` | 不做 | worker 从现有 Sequence skeleton 生成 rank-local metadata |
| ZMQ exactly-once、boot epoch、消息重放和自动 reroute | 不做 | 当前是受控 benchmark，不做服务级容灾 |
| 多通道复杂协议和全量 per-rank telemetry | 收缩 | MVP 只需要请求/完成事件、最小 wave 控制和负载报告 |
| DecodeCoordinator 延后到多机阶段或仅作为可选 helper | 不做 | 单机 DP8/DP2 的不均衡 EP 执行已经需要 wave coordination |
| 沿用现有 decode preemption | 保留 | 每个 quantum 前检查 `can_append(16)`；容量不足时由当前 LocalScheduler 抢占 running 尾部 victim |
| 首版同时支持 RoundRobin/LeastBatch/LeastCache | 先做 RoundRobin | 先验证架构；LeastBatch 按数据追加，LeastCache 后置 |
| metadata bitwise/semantic equivalence | 不做 | dummy 场景只要求结构合法和执行安全 |
| replica HA、rolling restart、多 LB | 不做 | 与当前单机/多机开发验证无关 |
| 预先设定 3%/5% 服务 SLA | 改为工程回归门槛 | 同配置 legacy median 对比：fast path 5%，其他平衡 workload 10% |

### 1.6 对齐 vLLM 的进程粒度

本方案采用 vLLM 的职责粒度，而不是按物理机器切 scheduler：

- 每个 attention DP instance/rank 一个 `LocalEngineCore` 和一个 `LocalScheduler`；
- 每个 LocalEngineCore 只拥有一个完整 DP group 的请求、KV 和 SP 状态；
- 一台机器可以放置 1 个或多个 LocalEngineCore，数量由 DP/SP 拓扑决定；
- 整个 EP deployment 在 `attention_dp>1` 时只有一个轻量 `DecodeCoordinator` 进程，不按
  机器或 DP instance 复制；`attention_dp=1` 不启动它。

因此，vLLM 的 `DP2 TP4 DCP4` 对应 2 个 EngineCore/scheduler、每个管理 4 个 GPU worker，
再加 1 个 deployment-wide DPCoordinator。DCP4 复用 TP4 的 GPU，不增加 EngineCore 或
DPCoordinator 数量；逻辑上每个 DP replica 内各有一个 4-rank DCP group，共 2 个 DCP
communication groups。

这里的“1 个 DPCoordinator”指 online internal/hybrid DP，以及本 proposal 对应的 MoE/EP
online 场景。vLLM offline SPMD 不启动独立 DPCoordinator；dense model 使用 external LB
时也可以没有它。本 proposal 采用 online MoE/EP 的粒度：一个独立 coordinator 维护
wave/running 状态并负责 idle wakeup，但不保存请求/KV，也不参与 GPU forward；scheduler/
EngineCore 的数量仍等于 DP size。

本地 vLLM 基线参考：

- `docs/design/arch_overview.md`
- `docs/serving/data_parallel_deployment.md`
- `vllm/v1/engine/coordinator.py`
- `vllm/v1/engine/core.py::DPEngineCoreProc`
- `vllm/v1/engine/utils.py::launch_core_engines`

## 2. 场景定义

唯一支持的配置是：

```text
mode = "decode"
dummy_prefill = true
attention_tp = 1
ffn_tp = 1
ffn_dp = 1
ffn_ep = attention_dp * attention_sp
loop_count = 16
ignore_eos = true
```

`loop_count=16` 是本 proposal 的固定场景，不是一个待调优参数。一个 decode quantum 固定
包含 16 次 forward；hierarchical 模式首版应拒绝其他值，不能让不同 DP engine 覆盖或动态
修改。

请求生命周期为：

```text
新请求
  -> Router validation / ADD accepted
  -> WAITING_ADMISSION
  -> 选择 master/participants
  -> 按 prompt length 分配逻辑 KV blocks / block table
  -> 追加 bootstrap token，并把它计入内部 prompt/context
  -> RUNNING_DECODE
  -> quantum 开始前保证 can_append(16)，必要时 preempt 本地 victim
  -> 执行固定 16-forward decode quantum
  -> 只提交 max_tokens 范围内的有效 token 前缀
  -> postprocess / finish / free blocks
```

dummy admission 不调用 `ModelRunner`，不执行 prefill forward，也不产生有效 prefix KV。
hierarchical 模式在 deployment READY 前对 KV cache 做一次 zero initialization；它属于启动
初始化，不属于 admission，也不计入 benchmark steady-state。

这里有两种“dummy”：

1. 对真实 benchmark 请求，decode 读取的是 dummy-prefill 建立的逻辑 KV 状态，输出数值没有
   模型语义。逻辑 prompt KV 对应 zero-initialized cache pages。
2. 某个 DP engine/SP rank 没有真实 master request 时，为了保持非空执行并参与 EP
   collective，LocalScheduler 使用 `SPStateManager` 中预先建立的 control dummy Sequence。

control dummy 使用固定 token 和合法的 reserved sentinel block，不使用 worker 临时创建的
随机 Sequence。每个 SP rank 保留足够覆盖 `bootstrap + 16` 的 control blocks；这些 blocks
从可服务物理容量中扣除，但不进入真实 request KV、postprocess、output 或 useful throughput
计数。每个 quantum 都从同一份 canonical dummy skeleton 开始，worker 对其副本的 inner-loop
更新不回写。

bootstrap token 属于内部 prompt/context，不属于 completion，不消耗 `max_tokens`，也不进入
有效 token 指标。请求必须满足：

```text
original_prompt_len + 1 + round_up(max_tokens, 16) <= max_model_len
```

## 3. 目标与非目标

### 3.1 目标

- 新增独立的 `hierarchical` 架构，不修改旧分支来伪装成新模式。
- 每个 attention DP instance 由一个 `LocalEngineCore` 独占该 DP group 的请求队列、
  scheduler 和 KV 逻辑状态。
- 全局 router 只选择 LocalEngineCore，不保存 `Sequence`、block table 或执行 metadata。
- EP 范围内统一决定本轮是否执行 decode，确保所有 rank 的 forward 次数一致。
- `attention_dp>1` 时使用一个 deployment-wide wave coordinator；全局 idle 后暂停所有
  LocalEngineCore，收到首个新请求后统一唤醒。
- 每个 decode quantum 固定执行 16 次 forward。
- dispatch 前为每个真实请求保证完整 16-step KV capacity；容量不足时在当前
  LocalScheduler 内 preempt running 尾部 victim，释放其 blocks 后重试。
- control dummy 是确定性、结构合法且有独立 reserved blocks 的控制对象。
- 单机支持 `DP8 SP1 EP8`、`DP2 SP4 EP8`、`DP1 SP8 EP8`。
- 多机阶段支持 `DP16/32 SP1 EP16/32` 和 `DP2/4 SP8 EP16/32`。
- 复用 centralized 路径已使用的 `SPStateManager`、`BlockManager`、SP 策略和 worker
  kernel；不重写已工作的 KV/SP 调度算法。
- 以 liveness、结构合法性、资源账和性能作为验收标准。

### 3.2 非目标

- 真实 prefill、hybrid serving、P/D disaggregation、KV migration、prefix cache。
- token、logits 或模型质量正确性。
- 任意 attention SP 组合的通用拓扑支持；首版只支持验证矩阵中明确列出的配置。
- 跨 LocalEngineCore 的请求迁移。
- 保留已生成 token 的 resume、KV swap 或跨 LocalEngineCore preemption。
- DP engine crash 后的请求恢复、消息重放、exactly-once 或 replica HA。
- 修改、移动或跨 RPC 传输现有 `DecodeMetadata`，或新增另一套 rank metadata 结构。
- 重写 CUDA Graph 或 DeepEP 协议。
- 多 LB、rolling restart、复杂 telemetry 和自适应 weighted routing。

## 4. 最小架构

```text
request
  |
  v
+----------------------+       +---------------------+
| RequestRouter        |------>| DecodeCoordinator   |
| - engine registry    | FIRST | - wave/running      |
| - minimal load view  | REQ   | - START_WAVE        |
| - request ownership  |       +----------+----------+
+----------+-----------+                  |
           | ADD / ABORT / FINISH / LOAD  | START_WAVE
   +-------+----------------+--------------+--+
   |                        |                 |
   v                        v                 v
+----------------+  +----------------+  +----------------+
| LocalEngine 0  |  | LocalEngine 1  |  | LocalEngine N  |
| LocalScheduler |  | LocalScheduler |  | LocalScheduler |
| LocalExecutor  |  | LocalExecutor  |  | LocalExecutor  |
| local KV state |  | local KV state |  | local KV state |
+-------+--------+  +-------+--------+  +-------+--------+
        |                   |                   |
        +---------- Gloo leader group ----------+
                  (wave/quantum/unfinished sync)
        |                   |                   |
        v                   v                   v
   owned GPU workers; all global GPU ranks share the FFN EP group
```

### 4.1 RequestRouter

MVP 只负责：

- 在全部署 READY barrier 通过后维护 LocalEngineCore 列表；
- 按 RoundRobin 选择一个 DP engine；
- 保存 `request_id -> PENDING_OWNER | OWNED(engine_id)`；
- 转发 ADD/ABORT 和完成事件；
- engines paused 时发送 FIRST_REQ 通知，触发 DecodeCoordinator 广播 START_WAVE；
- 读取最小负载报告，供日志和后续 LeastBatch 使用。

它不拥有可变 `Sequence`、block table、DP/SP placement 或 decode metadata。

RequestRouter 就是本 proposal 的 LoadBalancer。它只在 ADD/ABORT 等请求事件上工作，不参与
每个 decode quantum 的 barrier 或 collective。RoundRobin 选中的 engine 若仅因为 command
queue 已满而拒绝，router 可以按 READY 顺序把其他 engine 各尝试一次；请求一旦 accepted，
sticky owner 固定，不再 reroute。

### 4.2 LocalEngineCore

每个完整 attention DP group 一个独立 Ray actor/OS process，内部包含：

- 一个 single-writer event loop；
- 一个 LocalScheduler，只管理一个 `SPStateManager`；
- 一个 LocalExecutor，只驱动该 DP group 的 `attention_sp * attention_tp` 个 worker；
- 请求 command queue 和完成事件 queue；
- 最小 LoadSnapshot。

物理机器只是 placement 边界。`DP8 SP1 EP8` 在一台 8 卡机器上有 8 个 LocalEngineCore；
`DP2 SP4 EP8` 有 2 个；`DP1 SP8 EP8` 有 1 个。

所有 scheduler mutable state 只能由 event loop 修改。event loop 在一次 16-forward GPU
quantum 内阻塞，ADD/ABORT/LOAD 最坏延迟为一个 quantum；command queue 可以异步收消息，
但只能在 quantum 边界应用。第一版不增加跨线程锁或事务式 plan/commit/rollback。

LocalEngineCore 沿用现有 decode preemption：每个 quantum 在 batch freeze 前检查
`can_append(16)`；容量不足时从本地 running 队列尾部选择 victim，释放其 KV blocks，
退回 prompt 状态并放回 `WAITING_ADMISSION` 队首。preempt 不跨 LocalEngineCore，也不发生在
16 次 inner forward 中间。

### 4.3 DecodeCoordinator（不是 LoadBalancer）

`attention_dp>1` 时启动一个轻量、deployment-wide DecodeCoordinator 进程。
DecodeCoordinator 不选择请求、不读取 Sequence/KV，也不参加 GPU forward。它只维护：

- 当前 `wave_id`；
- engines 全局处于 running 还是 paused；
- 所有 LocalEngineCore 的订阅/READY 状态；
- paused 状态收到 FIRST_REQ 后向所有 engines 广播 `START_WAVE(wave_id)`；
- global DP rank 0 上报 WAVE_COMPLETE 后推进 wave。

真正的每-quantum同步由所有 LocalEngineCore leaders 组成的 Gloo group 完成。Gloo world size
等于 `attention_dp`，Gloo rank 等于 `global_dp_idx`；GPU workers 不参加这个控制 collective。
每个 running wave 内的 quantum contract 为：

```text
drain commands / apply pending aborts
admit()
plan_decode() and freeze LocalDecodeBatch

local_unfinished = waiting_admission or running_decode
assert all engines have the same (wave_id, quantum_id)
global_unfinished = all_reduce_max(local_unfinished)

if global_unfinished:
    每个 global GPU rank 固定执行 16 次 decode forward
    LocalEngine 有真实 batch -> canonical real + per-SP control dummy skeleton
    LocalEngine 没有真实 batch -> all-control-dummy skeleton
    wait local workers
    drain in-flight ABORT notifications
    postprocess and commit only valid real-token prefixes
    quantum_id += 1
else:
    不执行 GPU forward
    所有 LocalEngineCore 进入 paused
    global DP rank 0 通知 WAVE_COMPLETE
    等待下一次 START_WAVE
```

全局 step identity 是 `(wave_id, quantum_id)`；quantum counter 在新 wave 从 0 开始。
Gloo sync 同时校验所有 engines 的 wave/quantum min/max 相等。新请求与 pause consensus
并发时，请求携带 router 观察到的 `wave_id`；paused engine 收到 stale-wave 请求后再次通知
coordinator，避免请求已经入队但其他 engines 没被唤醒。

`attention_dp=1` 走 fast path：不启动 coordinator/Gloo group，本地 command queue 直接
唤醒唯一 LocalEngineCore。无论 DP engines 是否同机，`attention_dp>1` 都使用上述同一份
wave contract。RequestRouter 只触发 wakeup，不参与每 quantum 同步。

### 4.4 DeploymentManager、placement 与 fail-fast

DeploymentManager 负责整个 Ray deployment 的资源和生命周期：

- 每个 attention DP rank 创建一个 placement group；
- 每个 placement group 包含 `engine_world_size` 个 1-GPU bundle 和一个 CPU control bundle；
- LocalEngineCore actor 放在 control bundle，所属 ModelRunner actors 放在 GPU bundles；
- LocalExecutor 接收已经确定的 placement group、global rank slice 和 worker handles，不再
  自行扫描集群或固定申请每节点 8 个 GPU；
- manager 监控所有 LocalEngineCore run refs；任一 engine/worker timeout 或退出时，kill
  全部 actors、删除全部 placement groups、终止 coordinator，并让 benchmark 非零退出。

一个 LocalEngineCore 只有在 model/worker、KV allocate+zero-init、attention/SP/EP group、
Gloo leader group、control dummy、DLSlime endpoint、CUDA Graph/warmup 和 config fingerprint
全部完成后才上报 READY。Router 只在全部 whitelist engines 通过 READY barrier 后接受请求。

## 5. 拓扑与 ownership

global rank 保持当前 attention mesh 的 `[dp, sp, tp]` row-major 布局：

```text
global_rank = (global_dp_idx * attention_sp + sp_idx) * attention_tp + tp_idx
```

首版 whitelist：

| GPU 数 | 配置 | LocalEngineCore 数 | 每个 LocalEngineCore 的 ownership |
|---:|---|---:|---|
| 8 | DP8 SP1 EP8 | 8 | 1 个 SP1 DP group，1 个 worker |
| 8 | DP2 SP4 EP8 | 2 | 1 个 SP4 DP group，4 个 workers |
| 8 | DP1 SP8 EP8 | 1 | 1 个完整 SP8 DP group |
| 16 | DP16 SP1 EP16 | 16 | 1 个 SP1 DP group，1 个 worker |
| 16 | DP2 SP8 EP16 | 2 | 1 个 SP8 DP group，8 个 workers |
| 32 | DP32 SP1 EP32 | 32 | 1 个 SP1 DP group，1 个 worker |
| 32 | DP4 SP8 EP32 | 4 | 1 个 SP8 DP group，8 个 workers |

对单机 `DP2 SP4 EP8`：

```text
global DP 0 -> global ranks 0, 1, 2, 3
global DP 1 -> global ranks 4, 5, 6, 7
```

启动两个 LocalEngineCore：engine 0 持有 DP0 的一个 `SPStateManager` 和 rank 0-3，
engine 1 持有 DP1 的一个 `SPStateManager` 和 rank 4-7。两个 engine 不共享 request queue、
block manager 或 SP collective。

对 whitelist 中的配置：

```text
num_engine_cores = attention_dp
engine_id = global_dp_idx
engine_world_size = attention_sp * attention_tp
engine_local_rank = sp_idx * attention_tp + tp_idx

ffn_ep = attention_dp * attention_sp * attention_tp
ffn_tp = 1
ffn_dp = 1
ffn_ep_rank = global_rank
```

必须校验：

- 配置完整命中 whitelist，且 `attention_tp=1`；
- `ffn_ep == attention_dp * attention_sp`、`ffn_tp=1`、`ffn_dp=1`，整个 deployment
  恰好形成一个 FFN EP group；
- 一个 attention DP group 完整落在同一个 LocalEngineCore；
- 一个 LocalEngineCore 不能拥有多个 attention DP groups；
- `engine_world_size == attention_sp * attention_tp`；
- 每个 attention DP group 对应一个独立 placement group；SP8 group 必须完整 strict-pack
  在同一节点；
- engine/worker 显式保存 `global_dp_idx`、`global_rank` 和 `engine_local_rank`；
- attention/SP/EP process group、block location 和日志使用 global rank；
- engine-local DLSlime endpoint 使用 `engine_local_rank` 索引，但 binding 显式携带
  global rank；不能通过重编号 global rank 适配局部 endpoint；
- 所有 engines 的 collective-sensitive config fingerprint 完全一致；
- 不接受仅满足 attention/FFN world-size product 相等但 FFN group 分解不同的配置。

## 6. LocalScheduler contract

新模式不调用旧 `_schedule_decentralized()`，也不再返回一个全局 `is_prefill`。API 拆为：

```text
add(request)
admit()           -> admitted requests
plan_decode()     -> LocalDecodeBatch
postprocess(batch, worker results)
abort(request_id)
preempt(victim)   -> release blocks and return to WAITING_ADMISSION
```

一次 LocalEngine quantum 的 scheduler 流程为：

1. `admit()` 从 `WAITING_ADMISSION` 读取请求；
2. 在该 engine 固定的 global DP group 内，使用精确 KV/SP 状态选择 master SP 和
   participants；
3. 验证请求在独占该 placement 可服务容量时，prompt、bootstrap 和整个
   `round_up(max_tokens, 16)` decode 生命周期能够容纳；该检查只判断请求是否永久可行，
   不为未来 decode blocks 建立 reservation；
4. 分配 prompt 逻辑 blocks并建立 block table；
5. 在 master SP 上为 deterministic bootstrap token 执行 `may_append(1)`，追加 token，并把
   它计入内部 prompt/context；
6. 请求进入 `RUNNING_DECODE`；
7. `plan_decode()` 遍历 running 请求；若当前请求不能 `can_append(16)`，从 running 队列
   尾部选择 victim，调用 `preempt(victim)` 释放 blocks、退回 prompt 状态并放回
   `WAITING_ADMISSION` 队首，然后重试；
8. 当前请求能够 `can_append(16)` 后执行 `may_append(16)`；
9. 为没有 real master request 的 SP rank 插入 persistent control dummy；
10. 固定一份 canonical full Sequence order，所有 SP/TP workers 收到相同顺序，仅裁剪
   rank-local heavy fields；
11. batch 交给 LocalExecutor 后，本 quantum 内不再修改这些请求的 placement/block table；
12. worker 从收到的现有 Sequence skeleton 调用 `prepare_decode_cpp()`，生成自己的
   rank-local `DecodeMetadata`，然后固定执行 16 次 forward；
13. worker 返回后，event loop 先收集 in-flight ABORT，再只对未 abort 的真实请求
    postprocess。

`plan_decode()` 沿用现有 decode preemption，最小流程为：

```text
遍历 running requests
if !can_append(request, 16):
    从 running 尾部选择 victim
    preempt(victim):
        释放 victim blocks
        放回 WAITING_ADMISSION 队首
    重试 can_append
may_append(request, 16)
freeze batch
```

preempt 只发生在 quantum 开始、`LocalDecodeBatch` freeze 之前，不允许发生在 16 次 inner
forward 中间。victim、KV 和 waiting queue 都属于当前 LocalScheduler，不跨
LocalEngineCore。MVP 直接复用现有 reset/re-admission 语义：victim 丢弃已经生成的 token，
退回原始 prompt 状态；重新 admission 时重新建立 dummy KV、追加 bootstrap 并从头 decode，
不支持保留已生成 token 后 resume。

`LocalDecodeBatch` 是一个 step-local ownership/lifetime 容器，不是新的 metadata
结构体或 wire schema。MVP 只需包含：

```text
wave_id
quantum_id
engine_has_real
per-rank existing Sequence skeleton
real request_id -> master_global_rank / frozen order mapping
control dummy ids
```

这里需要区分两类 metadata ownership：

- LocalScheduler 生成并独占 scheduler metadata：DP/SP placement、master/participants、
  block location、block table、dispatched tokens 和 canonical Sequence
  order。
- worker 生成 rank-local runner metadata：现有 `DecodeMetadata` 以及由它派生的 device
  tensor/context。

`prepare_decode_cpp()` 是从已冻结的 Sequence/BlockContext 到某个 rank 执行输入的纯投影，
不修改 LocalScheduler 的 canonical state，因此留在 worker 不会形成 metadata 双写。
8 个 worker 可以并行生成各自 metadata，也避免 LocalScheduler 串行生成 8 份结果。

固定 `loop_count=16` 时，worker 在 16 次 inner forward 之间还需要现有 Sequence skeleton
执行 `update_decode()`、`prepare_sample()` 和 `update_seqs_inner_loop()`。即使
`DecodeMetadata` 改由 LocalScheduler 生成，也不能省掉 skeleton；反而会增加 8 份 metadata
生成、序列化和传输。

worker skeleton 是序列化后的 step-local 副本。它可以无条件推进 16 次，但它的
`num_tokens`、`num_dispatched_tokens` 和 terminal 判断都不回写 canonical state。每个 worker
返回：

```text
wave_id
quantum_id
global_rank
forward_count = 16
mastered request ids in frozen order
16 sampled token ids per mastered request
```

LocalScheduler 只接受每个真实请求的 master SP rank 结果，检查无重复、无缺失、顺序正确。
hierarchical 模式强制 `ignore_eos=True`；对还剩 `remaining < 16` 的最后一个 quantum，
只 commit 前 `remaining` 个 token，丢弃 overrun token，然后完成请求并释放实际 blocks。

因此 MVP 保持当前边界：

```text
LocalScheduler.plan_decode()
  -> 固定 canonical full Sequence skeleton + real/dummy mapping
  -> use_dlslime_rpc=True 的 engine-local DLSlime endpoint 发送 skeleton
  -> 每个 worker 调用现有 prepare_decode_cpp()
  -> tensor materialization + set_context()
  -> 16 次 inner forward
  -> LocalExecutor 按 global rank 收集 worker results
  -> LocalScheduler 在 canonical state 上 commit
```

`use_dlslime_rpc=True` 及其 `send_seqs()/recv_seqs()` 已经实现并继续复用。这里“不新增”
仅指不把 `DecodeMetadata` 作为新的 DLSlime payload；当前 payload 是 Sequence skeleton。
每个 LocalEngineCore 持有一个局部 endpoint，binding 由 `engine_local_rank` 选择，并显式关联
global rank。只有 profiling 明确证明 worker 侧 `prepare_decode_cpp()` 是瓶颈时，才重新评估
生成位置。

single-writer 前提下，admission 继续使用 `can_allocate -> allocate`，容量判断只计算实际
allocated blocks 和永久 control-dummy blocks，不预留请求未来整个 decode 生命周期的
blocks。请求即使独占合法 placement 也无法容纳 prompt、bootstrap 和 padded decode 长度时
立即 rejected；当前暂时容量不足时留在 `WAITING_ADMISSION`。running 请求在 quantum 开始前
无法 `can_append(16)` 时，按上述本地流程 preempt；不引入跨线程
`plan/commit/rollback`。

ABORT 语义固定在 quantum boundary：

- `WAITING_ADMISSION` 请求在 event loop 处理时立即 ABORTED；
- in-flight 请求先进入 `ABORT_PENDING`；
- worker 返回后、token commit 前再次 drain abort command；
- commit 前观察到 ABORT 的请求丢弃本 quantum token，释放 blocks，并以 ABORTED
  结束；
- 同一 quantum 内 ABORT 优先于自然 FINISHED；terminal 之后的 ABORT 返回
  `already_terminal`。

## 7. 最小请求与负载协议

MVP 消息只需要：

```text
ADD(request_id, prompt_token_ids, max_tokens, temperature, ignore_eos=true, wave_id)
ADD_RESULT(request_id, accepted | rejected, reason)
ABORT(request_id)
ABORT_RESULT(request_id, aborted | abort_pending | already_terminal | not_found)
FINISH(request_id, generated_count, status)
LOAD(engine_id, ready, waiting, running, free_blocks_min, wave_id, quantum_id)

FIRST_REQ(target_engine_id, wave_id)
START_WAVE(wave_id)
WAVE_COMPLETE(wave_id)
```

约束：

- `request_id` 直接复用现有 `Sequence.seq_id:uint64`，由 frontend 生成并在一次运行中唯一；
- ADD 携带完整 prompt token IDs；length-only compact payload 不进入 MVP；
- Router 在发 ADD 前建立 `PENDING_OWNER`，accepted 后变为 sticky `OWNED`，rejected 后删除；
- 同一 request/engine command channel 保序，因此 ADD 必须先于随后到达的 ABORT；
- `accepted` 表示请求已通过永久合法性检查并进入 engine command/waiting queue，不表示已
  完成 KV admission；
- 选中的 engine 仅因 queue full 拒绝时，router 可以把其他 READY engines 各尝试一次；
- DP engine crash 直接让 benchmark run 失败，不做自动 reroute；
- 不设计 ACK timeout 重放、boot epoch、message sequence、trace envelope 或持久化；
- command/waiting queue 以 unique request 数设明确上限，满时返回 rejected；
- hierarchical 模式要求 `ignore_eos=True`、`max_tokens>=1`，并验证 token ID 范围和
  `prompt_len + 1 + round_up(max_tokens, 16) <= max_model_len`；
- FINISH 不需要向 router 传回非语义 token 值；LocalScheduler 内部仍使用 worker token
  buffers 推进下一 quantum 的 last token。

请求状态固定为：

```text
PENDING_ADD
  -> WAITING_ADMISSION
  -> RUNNING_DECODE
  -> FINISHED

WAITING_ADMISSION -> ABORTED
RUNNING_DECODE -> WAITING_ADMISSION  (preempt，释放 blocks 并退回 prompt 状态)
RUNNING_DECODE -> ABORT_PENDING -> ABORTED
PENDING_ADD -> REJECTED
```

accepted 请求恰好产生一个 terminal FINISH，status 只能是 `FINISHED` 或 `ABORTED`。
`REJECTED` 只通过 ADD_RESULT 返回。Engine/worker failure 直接终止整次 benchmark，不模拟
逐请求恢复或继续服务。

hierarchical frontend 是后台执行模型：保留 `add_request()`、`is_finished()` 和
`generate()`，新增 `poll()` 获取完成/指标事件。兼容 `step()` 时，它只能作为 `poll()` 的
别名，不能再承诺“一次 frontend step 对应一个 GPU quantum”；arrival benchmark 必须按
wall clock 投递并轮询结果。`legacy_global` 的同步 step 语义保持不变。

LoadSnapshot 初版只保留 routing 和排障真正需要的字段：

- DP engine ready 状态；
- waiting/running 的 unique request 数；
- free blocks 的 engine-local min；
- 最近完成的 `(wave_id, quantum_id)`；
- useful real batch size、control dummy count 和 all-dummy engine quantum count。

Phase 1 即使只有一台机器，也可能有多个 DP engines；router 从第一阶段就用 RoundRobin
在这些 engines 之间选择。观察到明显负载偏斜后，再基于 waiting/running 增加 LeastBatch。

LeastCache、per-SP 压力、乐观 pending、staleness penalty 和 weighted policy 均不进入 MVP。

## 8. 模块改造

| 模块 | MVP 改造 |
|---|---|
| `nanodeploy/config.py` | 删除旧 `scheduler_mode`；新增 `scheduler_arch = legacy_global \| hierarchical` 和 whitelist 校验 |
| `nanodeploy/engine/llm_engine.py` | 保留 legacy 分支；hierarchical 时变成 router/frontend proxy，提供 `poll()`，不再由 frontend `step()` 驱动 GPU |
| 新 `nanodeploy/engine/deployment_manager.py` | 创建 per-DP placement groups、启动/监控 actors、READY handshake 和全局 fail-fast cleanup |
| 新 `nanodeploy/engine/local_engine.py` | 每 attention DP instance 一个 LocalEngineCore Ray actor、single-writer event loop、command queue、LoadSnapshot |
| `nanodeploy/engine/scheduler.py` | 新 LocalScheduler adapter；只管理一个 DP group，维护 canonical state、本地 decode preemption 和 frozen batch mapping |
| C++ `Scheduler` | 删除旧 decentralized 分队列代码；拆分 admission/decode API；保留现有 `can_append(loop_count)`、running-tail victim 和 recompute preemption |
| `SPStateManager` / `BlockManager` | 每个 LocalScheduler 持有一个 SPStateManager；复用 `can_append/may_append/deallocate` 并增加 deterministic control dummy reserved blocks |
| `nanodeploy/engine/ray_executor.py` | LocalRayExecutor 接收既定 placement group/global rank slice，只驱动所属 DP group workers |
| `nanodeploy/endpoint/rpc_endpoint.py` | 每个 LocalEngineCore 一个 endpoint；继续只传 Sequence skeleton；binding 显式关联 global rank 和 engine-local rank |
| `csrc/nanodeploy/worker/model_runner_utils.*` | 保留现有 `DecodeMetadata` 和 `prepare_decode_cpp()` 计算逻辑，不改字段 |
| `nanodeploy/worker/model_runner.py` | 删除无合法 block table 的随机 dummy fallback；继续生成 rank-local metadata，执行 16 次并返回显式 global-rank result/trace |
| 新 `nanodeploy/router/request_router.py` | READY registry、RoundRobin、PENDING/OWNED sticky ownership、FIRST_REQ wakeup |
| 新 `nanodeploy/engine/decode_coordinator.py` | `attention_dp>1` 时启动一个轻量进程，维护 wave/running 并广播 START_WAVE；不参与每-quantum Gloo |

新配置只保留首版需要的项：

```text
scheduler_arch = legacy_global | hierarchical
router_policy = round_robin
load_report_interval_ms = 100
loop_count = 16
hierarchical_queue_capacity = <bounded request count>
startup_timeout_s = 600
quantum_timeout_s = <configured or warmup-derived>
```

hierarchical 模式启动时强制校验 `loop_count == 16` 和完整 topology whitelist；每个 ADD
请求单独强制校验 `ignore_eos=True`。collective-sensitive config fingerprint 至少覆盖
model/dtype、attention/FFN
topology、SP/EP backend、CUDA Graph/eager 和 buckets、batch/recv limits、KV block size、
max model length、EPLB/perfect EPLB、dummy schema version 及影响通信的环境配置。
router policy、queue capacity 和 metrics interval 不进入 collective hash。

## 9. 分阶段开发计划

### Phase 0：删除旧分支并固化最小 contract

- 删除第 1.1 节列出的旧 decentralized 代码和配置。
- 新增 `scheduler_arch`，确保 `legacy_global` 行为不变。
- 定义 whitelist topology mapping 和 `LocalDecodeBatch`。
- 将 dummy admission 与 GPU decode 从 `is_prefill` 控制流中拆开。
- 固定并校验 `loop_count=16`、`ignore_eos=True`、bootstrap 和 final-quantum overrun 语义。
- 保留现有 decode preemption：每个 quantum 在 batch freeze 前检查 `can_append(16)`，不足时
  抢占当前 LocalScheduler running 尾部 victim，释放 blocks 后退回
  `WAITING_ADMISSION` 队首。
- 将 SPStateManager control dummy 改为 deterministic、拥有合法 reserved blocks 的控制对象，
  删除 worker 随机 dummy fallback。
- 明确 LocalScheduler canonical state、worker skeleton copy 和 result commit ownership，不修改
  或移动 `DecodeMetadata`。
- 定义 `(wave_id, quantum_id)`、request 状态机、quantum-boundary abort 和 config
  fingerprint。
- 增加不依赖 GPU 的 topology/ownership 单元测试。

完成标准：

- 仓库中不再暴露 `scheduler_mode="decentralized"`；
- legacy centralized 单元测试仍通过；
- `DP2 SP4 EP8` 能稳定映射为两个独立 LocalEngineCore，每个拥有一个 SP4 DP group；
- admission 和 decode 是两个独立调用，不存在全局 phase bit；
- hierarchical 模式不能用非 16 的 `loop_count` 或非完整 EP whitelist 配置启动，且
  `ignore_eos=False` 的请求会被 ADD validation 拒绝；
- bootstrap 不消耗 completion budget；padded length 永久可行性检查和
  `can_append(16)` preemption 可单元测试；
- control dummy 和真实请求在 request/KV/output 账上严格分离。

### Phase 1：单节点 8 卡新模式

- DeploymentManager 为每个 DP rank 创建一个精确大小的 placement group，启动一个
  RequestRouter、`attention_dp` 个 LocalEngineCore 和总计 8 个 worker。
- `attention_dp>1` 时从本阶段开始启动一个 DecodeCoordinator 进程和 LocalEngine leader
  Gloo group，打通 FIRST_REQ/START_WAVE/WAVE_COMPLETE。
- 完成全部署 READY barrier 和 collective-sensitive config hash 检查后才接受请求。
- 每个 LocalScheduler 独占一个 DP group 的请求队列、一个 SPStateManager 和 block state。
- 每个 LocalExecutor 只驱动所属 DP group 的 `attention_sp * attention_tp` 个 rank。
- 打通 ADD/ADD_RESULT、ABORT、FINISH、最小 LoadSnapshot 和 frontend `poll()`。
- 每个 LocalScheduler 为所属 DP group 的 ranks 生成相同 canonical full Sequence order，
  插入必要的 per-SP control dummy。
- worker 继续使用现有 `prepare_decode_cpp()`、`DecodeMetadata` 和 worker kernel。
- 每个 LocalEngineCore 使用自己的 DLSlime endpoint；不新增 `DecodeMetadata` payload。
- 在 READY 前 zero-init KV cache，并验证 final quantum、abort pending 和
  preempt/re-admission block release/reallocate。
- 依次验证：
  - `DP8 SP1 EP8`
  - `DP2 SP4 EP8`
  - `DP1 SP8 EP8`

`DP2 SP4 EP8` 启动两个 LocalEngineCore。必须包含不均衡 workload：只向一个 engine
分配请求、两个 engine 都有请求、请求完成后重新分配，验证两个 SP4 DP groups 不串状态，
同时验证只有一个 engine active 时另一个 engine 能执行 all-control-dummy decode，以及全局
idle 后停止 forward、收到新请求后开始新 wave。

完成标准：

- dummy admission 不调用 GPU；
- DecodeCoordinator/Gloo contract 已在单机 DP8/DP2 生效；
- 每个 decode quantum 固定执行 16 次 forward；
- 三种拓扑都能完成请求且无 hang；
- 实际 request blocks 和 control dummy capacity 分账闭合；
- DP2SP4 的 rank、master SP 和 block table 均落在正确 group；
- final quantum 只 commit 有效 completion 前缀，bootstrap/dummy/overrun 不进入 useful token；
- in-flight abort 在 quantum boundary 生效且只释放一次；
- 不要求 token 或 metadata 与 legacy 数值一致。

### Phase 2：多节点 placement、liveness 与 fail-fast

- 复用 Phase 1 已实现的唯一 DecodeCoordinator、wave protocol 和 per-quantum Gloo contract，
  不按节点创建 coordinator。
- 将 per-DP placement groups 扩展到多节点；SP8 group 必须完整 strict-pack 在同一节点。
- 每个 LocalRayExecutor 仍只驱动所属 DP group 的 SP/TP ranks，所有 workers 保留 global rank。
- 验证跨机 START_WAVE、stale-wave race、all-control-dummy engine 和全局 idle pause。
- 为 quantum RPC/Gloo wait 增加 timeout；任一 actor 失败由 DeploymentManager kill 全部署并
  非零退出。
- router 先使用 RoundRobin 和 sticky ownership。
- 验证：
  - `DP16 SP1 EP16`
  - `DP2 SP8 EP16`
  - `DP32 SP1 EP32`
  - `DP4 SP8 EP32`

完成标准：

- 只有一个 DP engine 有真实 batch 时，所有 EP rank 仍固定完成 16 次 forward；
- 所有 DP engines idle 时不执行 forward，收到新请求后能恢复；
- 不均衡流量下没有 collective hang；
- DP engine/worker crash 或 quantum timeout 能让整次 run fail-fast 并清理 actors/placement
  groups，不要求请求恢复。

### Phase 3：按 profiling 结果优化

只有 Phase 1/2 数据证明存在瓶颈时才做：

- 若 RoundRobin 导致明显偏斜，增加最小 LeastBatch；
- 若 100 ms report 影响不明显，不增加更复杂 telemetry；
- 若每 step coordination 成为瓶颈，再合并多个 forward 或降低控制频率；
- 若 block capacity 成为路由瓶颈，再评估 LeastCache。

本阶段不预先承诺 HA、exactly-once、跨 DP engine 迁移或多 LB。

## 10. 验证计划

### 10.1 测试层次

1. CPU 单元测试：
   - attention/FFN 完整 topology whitelist；
   - global DP、engine、global rank、engine-local rank 和 endpoint binding mapping；
   - `DP2 SP4 EP8` 的两个 LocalEngineCore/LocalScheduler ownership；
   - 每个 LocalScheduler 恰好拥有一个 SPStateManager；
   - admission/decode API 不共享 phase bit；
   - hierarchical 配置只接受 `loop_count=16` 和完整 FFN EP group，ADD 只接受
     `ignore_eos=True`；
   - bootstrap 计入内部 prompt 但不消耗 completion budget；
   - padded length 永久可行性检查、`can_append(16)` 和 running-tail preemption；
   - deterministic control dummy blocks 不进入 request accounting；
   - request/PENDING_OWNER/preempt/abort 状态转换；
   - `(wave_id, quantum_id)` 递进和 stale-wave wakeup；
   - `DecodeMetadata` 仍使用现有字段并由 worker 生成；
   - worker result 的 master-rank、frozen-order、duplicate/missing 校验；
   - allocated/free/control-dummy block 分账闭合。
2. 单机 8 卡集成测试：
   - 三种 whitelist 拓扑各一个最小 smoke case；
   - `DP2 SP4 EP8` 作为主要开发回归拓扑；
   - idle、单 DP engine active、双 DP engine active、finish 后重用；
   - 全局 idle pause、FIRST_REQ/START_WAVE wakeup、stale-wave race；
   - KV pressure 下的 running-tail preempt、重新 dummy admission 和 block reuse；
   - final quantum overrun、in-flight abort、control dummy reset；
   - eager 和一个代表性 CUDA Graph 配置。
3. 多机 liveness 测试：
   - 只有一个 DP engine active；
   - 多 DP engines active 且负载不均；
   - 全局 idle 后 wakeup；
   - worker/engine crash、Gloo/quantum timeout、全部署 cleanup。
4. 性能测试：
   - 同 topology、workload、arrival 和 execution mode 对比 `legacy_global`；
   - 每项至少运行 3 次并比较 median；
   - 不把拓扑、arrival、SP policy、eager/graph 做全排列。

`DP2 SP4 EP8` 首版使用已工作的固定/legacy SP 策略即可，不要求为了该拓扑同步支持全部
SP8 专用 bucket、long-short 和 dynamic policy。

### 10.2 必须验证的 invariant

- dummy admission 的 ModelRunner 调用次数为 0。
- READY 前 KV cache 完成一次 zero initialization，steady-state admission 不触发 GPU。
- 每个执行 `(wave_id, quantum_id)` 的所有 EP rank forward 次数都等于 16。
- 所有 LocalEngine leaders 对 wave/quantum ID 的 min/max 一致。
- 所有 global GPU rank 以相同 quantum/inner-loop 顺序进入 model/EP collective，不发生 hang。
- 全局 idle 时所有 engines paused 且 forward count 不增加；FIRST_REQ 后全部进入同一新 wave。
- `DP2 SP4` 中 DP0 只使用 rank 0-3，DP1 只使用 rank 4-7。
- 同一个 LocalEngine 内所有 SP workers 接收相同 canonical full Sequence order。
- per-rank batch 的 shape、长度、索引和 block ID 均在合法范围内。
- control dummy 拥有合法 reserved blocks，但不增加真实 request、request KV、postprocess、
  output 或 useful throughput 计数。
- bootstrap 不消耗 completion budget；final quantum 只 commit 剩余有效 token，overrun 不进入
  completion。
- request 只有一个 LocalEngineCore owner。
- preempt 只发生在 quantum 开始、batch freeze 之前；victim 属于当前 LocalScheduler，
  释放 blocks、退回 prompt 状态并进入本地 `WAITING_ADMISSION` 队首。
- finish/abort/preempt 后 allocated request blocks 均正确释放，control dummy blocks 保持常驻。
- in-flight ABORT 只在 quantum boundary 生效，terminal event 和资源释放都恰好一次。
- accepted request 能从 waiting 推进到 running；被 preempt 后可以重新 admission，最终进入
  FINISHED 或 ABORTED。

明确不验证：

- token ID、logits 或采样分布；
- 与 legacy 的 output equivalence；
- metadata bitwise equivalence；
- dummy KV 对应的 attention 数值；
- TTFT 的模型服务语义。

### 10.3 最小观测指标

- waiting/running unique requests；
- command queue delay、admission、schedule、execute、postprocess latency；
- 每个 DP engine/group 的 real batch size；
- `(wave_id, quantum_id)` 和各 global rank forward count；
- decode-coordination wait time；
- `useful_decode_tokens`：canonical 实际 commit 的 completion token；
- `raw_token_slots`：真实、control dummy 和 final overrun 的全部执行槽位；
- `dummy_rank_forward_ratio`：all-dummy rank-forwards / all rank-forwards；
- `dummy_slot_ratio`：control dummy slots / all executed slots；
- 每个 LocalScheduler 的 preemption count；
- free/used/control-dummy blocks；
- request throughput、useful token throughput 和 raw decode forward throughput。

bootstrap、control dummy 和 final-quantum overrun 均不计入 useful token。任何沿用
`token throughput` 名称的报告必须明确它表示 useful completion 还是 raw benchmark work。

### 10.4 Execution trace

集成测试模式下，每个 worker 每 quantum 至少记录：

```text
wave_id
quantum_id
global_rank
inner_loop_idx
batch_kind = real_or_mixed | all_control_dummy
real_batch_size
control_dummy_count
use_sp_a2a
forward_begin / forward_end
```

测试结束后校验所有 global ranks 具有相同的 wave/quantum 集合和顺序，每个执行 quantum
恰好包含 inner loop 0-15，同一 SP group 的 SP branch 一致，且 terminal request 后不再
commit。该 trace 只用于集成测试和排障，不进入 steady-state 默认日志。

### 10.5 性能门槛

功能通过不等于性能通过。MVP 的暂定工程门槛为：

- `DP1 SP8` fast path 相对同配置 `legacy_global` 的 median throughput regression 不超过 5%；
- 其他平衡、饱和 workload 的 median throughput regression 不超过 10%；
- coordination/control overhead 的 median 不超过 quantum wall time 的 5%，p99 不超过 10%。

若 baseline 自身方差已经超过门槛，先修复 benchmark 稳定性，再根据数据调整门槛；不能以
“已采集性能数据”为由接受数量级 regression。

## 11. 风险与必要防线

| 风险 | 必要对策 |
|---|---|
| 新模式误调用旧 decentralized 分支 | 删除旧分支；配置和类名完全分离 |
| `DP2 SP4` 的两个 engines 串 rank 或 block state | 每 DP group 一个 scheduler；显式 global DP/rank mapping |
| DP engine 自行决定是否 forward 导致 EP hang | Phase 1 起使用 wave coordinator 和 LocalEngine leader Gloo contract |
| 全局 idle 后无法统一唤醒 | FIRST_REQ/START_WAVE/WAVE_COMPLETE，并处理 stale-wave 请求 |
| DP engines 使用不同 loop count/step | collective config hash；每 quantum 断言 wave/quantum 和 forward count |
| idle rank 没有进入 collective | LocalScheduler 插入拥有合法 blocks 的 deterministic control dummy |
| admission 与 in-flight decode 同时修改 Sequence | single-writer；batch dispatch 后到 postprocess 前不修改 |
| worker skeleton 更新污染 canonical state | skeleton 只作为序列化副本；LocalScheduler 按 frozen mapping 单点 commit |
| bootstrap/dummy/overrun 被当成有效 token | 分离 useful tokens 和 raw work，terminal commit 只取有效前缀 |
| 下一 quantum 缺 KV block 导致停滞 | batch freeze 前检查 `can_append(16)`；不足时 preempt 本地 running 尾部 victim，释放 blocks 后重试 |
| control dummy 被当成真实请求 | reserved capacity 单独记账；request、postprocess、output 和 useful 指标均排除 |
| 未初始化 dummy prompt KV 导致 NaN/不稳定 | hierarchical deployment READY 前 zero-init KV cache |
| in-flight ABORT 双重释放或提交多余 token | quantum boundary abort；commit 前 drain，terminal/resource release 恰好一次 |
| per-engine Ray executor 重复申请整节点 GPU | DeploymentManager 创建每 DP rank 一个精确大小 placement group |
| block table/shape 非法导致越界 | 做结构和范围断言；不以“不关心数值”为由跳过内存安全检查 |
| worker/engine crash 令 collective 永久等待 | quantum/Gloo timeout；manager kill 全部 actors、移除 PG 并非零退出 |

数值准确性不是目标，但 collective 顺序、tensor shape、索引范围和内存 ownership 仍是安全
底线，不能作为“过度防御”删除。

## 12. MVP 完成定义

MVP 完成需要同时满足：

- 旧 `scheduler_mode="decentralized"` 已删除，不能作为新模式入口；
- 新 `scheduler_arch="hierarchical"` 拥有独立代码路径；
- 单机 `DP8 SP1 EP8`、`DP2 SP4 EP8`、`DP1 SP8 EP8` 全部通过；
- `DP2 SP4 EP8` 启动两个独立 LocalEngineCore/LocalScheduler，各管理一个 SP4 DP group；
- 每个 attention DP instance 恰好对应一个 LocalEngineCore/LocalScheduler；
- `attention_dp>1` 时整个 EP deployment 恰好启动一个轻量 DecodeCoordinator 进程，并从
  Phase 1 起使用 wave protocol；`attention_dp=1` 走无 coordinator fast path；
- 每个 attention DP rank 恰好对应一个精确大小的 Ray placement group；
- 多节点任一 DP engine 有工作时，所有 EP rank 都固定执行 16 次 decode forward；
- 每个 decode quantum 固定执行 16 次 forward；
- 全局 idle 时不执行 forward，FIRST_REQ 能统一开始新 wave；
- dummy admission 不执行 prefill forward；KV cache 在 READY 前 zero-init；
- bootstrap 属于内部 prompt，不消耗 completion budget；final quantum overrun 不进入 useful
  completion；
- 每个 accepted 请求通过 padded 长度永久可行性检查；每个 quantum 在 batch freeze 前保证
  `can_append(16)`，不足时沿用本地 running-tail decode preemption；
- idle SP rank/engine 使用 deterministic control dummy 和合法 reserved blocks；
- `DecodeMetadata` 结构和生成位置保持不变，由 worker 从 LocalScheduler 发来的 skeleton
  生成；
- canonical Sequence 只由 LocalScheduler 在 worker result 校验后推进；
- request ownership、quantum-boundary abort、terminal exactly-once 和
  allocated/control-dummy block 分账通过；
- 没有 collective hang、越界或资源泄漏；
- fail-fast 能在 engine/worker crash 或 timeout 后清理全部 actors 和 placement groups；
- 性能达到第 10.5 节工程门槛；不要求数值输出或 metadata 与 legacy 等价。

后续是否增加 LeastBatch、LeastCache 或更完整的服务可靠性，全部由 MVP
profiling 和真实使用需求驱动，不进入本次开发的默认范围。
