# NanoDeploy Hierarchical Serving 异步请求入口与 Rate-Control Benchmark 设计

状态：Phase A-D 已实施并通过 CPU 回归；Phase E 已完成首个两机验收，完整对照矩阵待执行
日期：2026-07-27
适用范围：`scheduler_arch="hierarchical"`，首个验收配置为两机
`DP2 SP8 TP1 EP16`、`loop_count=16`

## 1. 结论

本设计明确以下职责边界：

1. 请求到达过程由上游 Bench Load Generator 产生。
2. RequestRouter 是 NanoDeploy 的 Load Balancer，只负责接收、校验、路由、
   backpressure 和请求 ownership。
3. LocalEngineCore 提供非阻塞 ingress queue，负责把跨进程请求安全地交给
   single-writer event loop。
4. LocalScheduler 不管理 request rate，也不保存尚未到达的未来请求；它只维护
   `WAITING_ADMISSION`/running 队列并执行 admission、调度和资源管理。
5. DecodeCoordinator 只负责全局 wave 的 idle/wakeup，不参与请求路由或 rate
   control。
6. Load Balancer 收到请求后不能同步等待一个 decode quantum 才返回。请求成功进入
   ingress queue 和请求被 LocalScheduler 接受必须是两个独立阶段。
7. Bench 的发送、ADD 结果消费、完成事件消费和诊断采样必须互不阻塞。

本方案不把 7,200 个未来请求及其到达时间预先放进 LocalScheduler，也不让
LocalScheduler 使用 timer 判断请求是否到达。

## 2. 背景与根因

### 2.1 Legacy serving benchmark 的工作方式

原始 rate-controlled benchmark 会先生成相对到达时间：

```python
arrival_times = cumulative_intervals(request_rate, burstiness)
```

运行期间，每次循环把当前所有已经到达的请求加入 engine，然后执行或轮询下一步：

```python
while requests_remain or not engine.is_finished():
    now = time.perf_counter() - start_time

    while next_idx < total and arrival_times[next_idx] <= now:
        engine.add_request(requests[next_idx])
        next_idx += 1

    engine.step()
```

在 `legacy_global` 路径中，`engine.add_request()` 只在当前进程中调用
`Scheduler::add()`，把请求追加到 C++ waiting queue，调用可以快速返回。因此一次
`engine.step()` 即使执行了较长时间，返回后 benchmark 仍能快速补交这段时间内的全部
到达请求。

### 2.2 Hierarchical 路径改变了 ADD 的阻塞语义

当前 hierarchical ADD 路径为：

```text
Bench
  -> LLMEngine.add_request()
  -> RequestRouter.add()
  -> RayEngineTransport.add()
  -> actor.submit_add.remote()
  -> ray.get()
  -> LocalEngineCore._submit()
  -> wait(command.completed)
  -> event loop 在 quantum 边界处理 ADD
  -> AddResult
```

`LocalEngineCore` 为保护 LocalScheduler 的 single-writer 状态，只在 decode quantum
边界处理普通 command。这个约束是正确的；错误在于 RequestRouter/Bench 同步等待该
处理完成。

两机诊断复现中：

- 一个 engine quantum 平均约 483.76 ms；
- 慢 ADD 平均约 494.28 ms；
- 目标 rate-20 的请求间隔只有 50 ms；
- 81.59 秒只提交成功 98 个请求，即 1.20 requests/s；
- 无 request rejection、scheduler waiting backlog、preemption、CUDA error 或 OOM；
- engine 持续执行，但 bench 的 catch-up loop 无法退出，因此也无法及时 poll 完成
  事件。

问题不是 LocalScheduler 需要理解 arrival time，而是请求入口错误地受 decode quantum
反向阻塞。

## 3. 目标与非目标

### 3.1 目标

- Bench 能准确产生固定或随机 arrival process，例如 20 requests/s、持续 360 秒。
- 每次进入 Load Balancer 的请求调用快速返回，不等待 decode quantum。
- RequestRouter 保持 RoundRobin、sticky ownership 和 queue-full fallback 语义。
- LocalScheduler 继续保持 single-writer，不引入跨线程直接修改 scheduler state。
- quantum 执行期间到达的请求在下一个安全边界统一转入 LocalScheduler waiting
  queue。
- ADD 结果和 finish 结果可异步消费，不阻塞后续请求到达。
- 正确区分计划到达、实际 dispatch、ingress enqueue、scheduler accept、首次执行和
  完成时间。
- 正式 benchmark 能证明实际 offered load 达到目标 rate，并输出可信的吞吐和延迟
  数据。
- `legacy_global` 行为保持不变。

### 3.2 非目标

- LocalScheduler 不实现 request rate limiter。
- LocalScheduler 不保存未来请求，不按 arrival timestamp sleep。
- 不在 16-forward quantum 中途修改 frozen batch。
- 不通过降低 `loop_count` 掩盖入口阻塞。
- 不使用无界线程池并发调用当前非线程安全的 RequestRouter。
- 不把所有请求在 benchmark 开始前一次性加入 waiting queue。
- 不在本次方案中实现生产级多 LB、HA、消息持久化或 exactly-once recovery。

## 4. 正式架构

```text
+------------------------------+
| Bench Load Generator         |
| - arrival_times              |
| - target request rate        |
| - scheduled/dispatch metrics |
+---------------+--------------+
                |
                | submit due requests, non-blocking
                v
+------------------------------+
| RequestRouter / LoadBalancer |
| - validation                 |
| - RoundRobin                 |
| - ownership                  |
| - per-engine backpressure    |
| - pending submission refs    |
+---------------+--------------+
                |
                | async ENQUEUE / ENQUEUE_BATCH
                v
+------------------------------+       +--------------------------+
| LocalEngineCore ingress      |<----->| DecodeCoordinator        |
| - thread-safe ingress queue  | wake  | - wave/idle state        |
| - immediate ingress ACK      |       | - START_WAVE broadcast   |
| - single-writer event loop   |       +--------------------------+
+---------------+--------------+
                |
                | drain all pending ADDs at a safe boundary
                v
+------------------------------+
| LocalScheduler               |
| - WAITING_ADMISSION queue    |
| - admission / KV accounting  |
| - plan_decode                |
| - preemption / postprocess   |
+---------------+--------------+
                |
                v
          LocalExecutor / GPUs
```

Rate control 位于 Load Balancer 的上游入口。若 Bench 与 RequestRouter 在同一进程，
到达时间检查可以直接写在 bench serving 主循环中，但它仍然属于 Load Generator，不属于
LocalScheduler。

生产环境的 Load Balancer 可以另行提供 rate limiting/admission policy；benchmark 的
目标 rate 不能依赖生产 LB 主动限速，否则测到的是被 LB 重塑后的流量，而不是目标
offered load。

## 5. 组件职责

### 5.1 Bench Load Generator

Bench 负责：

- 在正式计时前准备请求内容，避免 CSV、token list 构造或随机数据生成污染 arrival；
- 生成每个请求相对 benchmark `t0` 的 `scheduled_arrival_ns`；
- 按计划时间向 RequestRouter 提交请求；
- 每次循环提交所有 `scheduled_arrival <= now` 的请求；
- 记录 dispatch lag；
- 持续消费 ADD 和 finish 事件；
- 维护 progress、诊断和最终性能结果；
- 在所有 arrival 已提交后继续 drain，或者按显式测试配置停止。

Bench 不等待某个请求被 LocalScheduler admission 后才发送下一个请求。

### 5.2 RequestRouter / Load Balancer

RequestRouter 负责：

- 在入口做与 engine 状态无关的参数校验；
- 按 RoundRobin 选择目标 LocalEngine；
- 保存 `request_id -> owner state`；
- 把请求异步提交给目标 LocalEngine ingress；
- 对 ingress queue full 做一次其他 READY engine fallback；
- 处理异步 scheduler ADD 结果；
- 请求 accepted 后固定 sticky owner；
- 聚合完成事件和缓存负载快照。

RequestRouter 不保存 Sequence 的 KV/SP 执行状态，也不参加每个 decode quantum。

### 5.3 LocalEngineCore ingress

LocalEngineCore ingress 是跨 Ray actor/concurrent actor method 与 single-writer scheduler
之间的边界，负责：

- 接收单个或批量请求；
- 在锁保护下检查 ingress capacity 和 duplicate reservation；
- 将不可变 `AddCommand` 放入 FIFO/MPSC queue；
- 唤醒空闲 event loop；
- 快速返回 `IngressAck`；
- 不在 Ray actor 请求处理线程中修改 LocalScheduler。

LocalEngineCore event loop 在安全边界 drain ingress queue，并逐请求调用
`LocalScheduler.add()`。一个请求的错误不能使整个 batch 失败。

### 5.4 LocalScheduler

LocalScheduler 只负责：

- 完整 ADD 合法性和 exclusive-lifetime capacity 校验；
- 将合法请求加入 `WAITING_ADMISSION`；
- admission 和 bootstrap allocation；
- KV/SP placement、preemption、decode planning 和 postprocess；
- 产生 scheduler accept/reject、first-token 和 finish 数据。

LocalScheduler 不知道请求的目标 request rate。对它而言，到达入口的请求都是普通新请求。

### 5.5 DecodeCoordinator

DecodeCoordinator 只负责：

- READY barrier；
- wave id；
- 全局 running/paused 状态；
- 首个 ingress 请求触发 START_WAVE；
- 所有 engine idle 后结束 wave。

它不维护 arrival schedule，也不路由请求。

## 6. 请求状态机

Load Balancer 和 LocalScheduler 使用分层状态，不再把 ingress enqueue 和 scheduler
accepted 混成一个状态：

```text
CREATED
  |
  | scheduled arrival reached
  v
ARRIVED
  |
  | RequestRouter selects engine
  v
PENDING_INGRESS
  | \
  |  \ ingress full / invalid envelope
  |   -> REJECTED
  |
  | IngressAck(enqueued=True)
  v
PENDING_ADD
  | \
  |  \ LocalScheduler legality rejection
  |   -> REJECTED
  |
  | AddResult(accepted=True)
  v
WAITING_ADMISSION
  |
  v
RUNNING_DECODE
  | \
  |  -> ABORTED
  v
FINISHED
```

RequestRouter 的 ownership 状态建议扩展为：

```text
PENDING_INGRESS(engine_id)
PENDING_ADD(engine_id)
OWNED(engine_id)
TERMINAL
```

请求进入 `OWNED` 后不能 reroute。

## 7. 控制协议

### 7.1 RequestEnvelope

Bench/Router 侧请求封装：

```python
@dataclass(frozen=True, slots=True)
class RequestEnvelope:
    request_id: int
    request_index: int
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    temperature: float
    ignore_eos: bool
```

`scheduled_arrival_ns` 是 benchmark 时钟数据，保存在 Bench/LLMEngine 前端，不要求
LocalScheduler 使用或比较该时间。

### 7.2 IngressAck

```python
@dataclass(frozen=True, slots=True)
class IngressAck:
    request_id: int
    engine_id: int
    enqueued: bool
    reason: str | None = None
```

`enqueued=True` 仅表示请求已经进入 LocalEngine ingress，并为 queue capacity 预留了一个
slot，不表示 scheduler 已完成 ADD。

### 7.3 AddResultEvent

```python
@dataclass(frozen=True, slots=True)
class AddResultEvent:
    request_id: int
    engine_id: int
    accepted: bool
    reason: str | None = None
```

该事件由 event loop 在 `LocalScheduler.add()` 后发布。RequestRouter 收到 accepted 事件
后把 owner 从 `PENDING_ADD` 转为 `OWNED`。

### 7.4 FirstTokenEvent

Hierarchical dummy decode 每个 quantum 一次提交最多 16 个有效 token。首次成功提交有效
token 时发布：

```python
@dataclass(frozen=True, slots=True)
class FirstTokenEvent:
    request_id: int
    engine_id: int
    generated_count: int
```

Bench 在收到事件时记录 end-to-end observed TTFT。若后续增加 token streaming，可由更细
粒度事件替换。

### 7.5 FinishEvent

保留现有 finish 语义，并保证它在同一请求的 `AddResultEvent(accepted=True)` 和可选
`FirstTokenEvent` 之后发布。

## 8. 数据结构与并发模型

### 8.1 LocalEngine ingress queue

建议新增：

```python
self._ingress_adds: queue.Queue[_IngressAdd]
self._ingress_lock: threading.Lock
self._reserved_request_ids: set[int]
self._reserved_slots: int
```

Ray actor method 可以并发进入，但只允许修改上述 ingress 元数据。LocalScheduler、
SPStateManager、BlockManager 和 request record 仍只由 event loop 线程修改。

### 8.2 Capacity reservation

入口容量定义为：

```text
reserved_slots =
    ingress pending
  + scheduler waiting
  + scheduler running
  + abort pending
```

`enqueue` 在 lock 内先预留 slot。发生以下事件时释放：

- ingress/scheduler rejection；
- FINISHED；
- ABORTED；
- 明确 cleanup。

这样 queue-full 可以在 ingress 快速返回，不需要等待 quantum。完整 KV placement
合法性仍由 LocalScheduler 校验。

### 8.3 FIFO 与确定性

- 单个 engine 内按 ingress queue 顺序处理；
- RequestRouter 在一个 due batch 内按 `request_index` 排序后执行全局 RoundRobin；
- batch 分组不能改变同一 engine 内的相对顺序；
- retry 只允许发生在 ingress queue-full 且请求尚未进入 `OWNED` 前。

### 8.4 Batch 上限

正常 rate-20、约 500 ms quantum 时，全局每个 quantum 约到达 10 个请求，每个 DP engine
约 5 个。为处理异常 burst，建议配置：

```text
max_ingress_batch_requests = 256
max_ingress_drain_ms = 10
```

event loop 优先 drain abort，再 drain ingress ADD。超过本轮 budget 的请求保留在 ingress
queue，下一 quantum 继续处理，并记录 ingress backlog。

## 9. 正式时序

### 9.1 Active wave

```text
Bench           Router/LB         LocalEngine ingress      Event loop/Scheduler
  | arrival due    |                       |                         |
  | submit(req) -->|                       |                         |
  |                | enqueue.remote(req) ->|                         |
  |                |<-- IngressAck --------|                         |
  | return quickly |                       |                         |
  |                |                       | quantum executing       |
  | submit next -->| enqueue next -------> | queue                   |
  |                |                       |                         |
  |                |                       |<-- boundary ------------|
  |                |                       | drain all pending ADDs  |
  |                |<------ AddResultEvent | LocalScheduler.add()    |
  |                |                       | admit/plan/execute      |
  |<--------- finish/first-token events ---|                         |
```

请求在 quantum 中途到达时，不进入当前 frozen batch；它会在下一安全边界转入 waiting
queue。这一量化延迟属于 hierarchical architecture 的真实 scheduling latency。

### 9.2 Global idle

当所有 engines paused：

1. 请求到达 Router；
2. 请求先成功放入目标 engine ingress；
3. Router/LocalEngine 触发 FIRST_REQ；
4. DecodeCoordinator 广播 START_WAVE；
5. event loop drain ingress；
6. LocalScheduler admission；
7. 所有 engine 进入同步 quantum。

必须保证请求已经进入 ingress 后再触发 wakeup，避免 wave 先启动而目标请求尚未可见。

## 10. Bench Serving 主循环

推荐主循环：

```python
next_idx = 0
scheduled = 0
dispatched = 0
ingress_enqueued = 0
accepted = 0
rejected = 0
completed = 0

start_ns = time.perf_counter_ns()

while True:
    now_ns = time.perf_counter_ns()
    elapsed_ns = now_ns - start_ns

    due_begin = next_idx
    while (
        next_idx < total
        and arrival_offsets_ns[next_idx] <= elapsed_ns
    ):
        next_idx += 1

    if next_idx > due_begin:
        due = requests[due_begin:next_idx]
        handles = engine.submit_requests_async(due)
        dispatched += len(handles)

    for ingress_ack in engine.poll_ingress_acks():
        handle_ingress_ack(ingress_ack)

    for add_result in engine.poll_add_results():
        handle_add_result(add_result)

    for event in engine.poll_events(refresh_load=False):
        handle_event(event)

    emit_cached_diagnostics_if_due()

    if (
        next_idx == total
        and engine.num_pending_ingress == 0
        and engine.num_pending_adds == 0
        and engine.is_finished()
    ):
        break

    wait_for_next_arrival_or_short_poll_deadline()
```

重要约束：

- `submit_requests_async()` 不能执行 `ray.get()`；
- 每轮必须重新计算 current time；
- 一次提交全部 due request；
- finish poll 不依赖发送循环是否追上；
- 正式性能路径不执行同步 load refresh；
- idle 时 wait 到下一个 arrival 或短 poll deadline，不 busy spin。

## 11. 非阻塞诊断与 LoadSnapshot

当前 `get_load()` 通过普通 command queue 获取一致 snapshot，因此也可能等待一个 quantum。
正式 benchmark 不能在主循环里同步刷新该 snapshot。

建议 LocalEngine event loop 在以下位置更新缓存：

- drain commands 后；
- admission 后；
- quantum postprocess 后；
- wave pause 时。

缓存结构在 lock 下整体替换：

```python
self._cached_load_snapshot: LoadSnapshot
```

Ray `get_cached_load()` 直接返回不可变 snapshot，不进入 scheduler command queue。

Bench 的一秒诊断只读取 cached snapshot。高开销 execution trace 继续保持 opt-in。

## 12. 时间与性能指标

### 12.1 Bench 时钟

以下时间全部使用 Bench 进程的 `time.perf_counter_ns()`：

- `scheduled_arrival_ns`
- `actual_dispatch_ns`
- `ingress_ack_observed_ns`
- `add_result_observed_ns`
- `first_token_observed_ns`
- `completion_observed_ns`

这避免跨节点直接比较不同 monotonic clock。

### 12.2 必须输出的入口指标

```text
scheduled_requests
dispatched_requests
ingress_enqueued
ingress_rejected
scheduler_accepted
scheduler_rejected
completed_requests
pending_ingress
pending_add
active_requests
achieved_dispatch_rate
```

延迟：

```text
dispatch_lag =
    actual_dispatch - scheduled_arrival

ingress_ack_latency =
    ingress_ack_observed - actual_dispatch

add_accept_latency =
    add_result_observed - actual_dispatch

observed_ttft =
    first_token_observed - scheduled_arrival

observed_e2e =
    completion_observed - scheduled_arrival
```

分别输出 avg/p50/p90/p95/p99/max。

`add_accept_latency` 预计包含最多一个 quantum，不应用它阻塞 request producer。

### 12.3 Actor 内部诊断指标

LocalEngine 使用自己的 monotonic clock计算内部 duration，不与 Bench 时间戳直接相减：

```text
ingress_queue_delay_ms
scheduler_add_ms
waiting_admission_ms
first_quantum_delay_ms
quantum_execute_ms
postprocess_ms
```

这些指标用于根因诊断，不替代 client-observed TTFT/E2E。

## 13. Backpressure 与错误语义

### 13.1 Ingress full

- 目标 engine ingress full：Router 尝试其他 READY engine；
- 所有 engine full：立即返回 `overloaded`；
- Bench 记录 rejection，不做无界重试；
- 是否重试由明确 benchmark 参数控制，默认不重试，以保持 offered-load 语义。

### 13.2 Scheduler rejection

完整长度、KV placement 或其他 scheduler legality rejection 通过
`AddResultEvent(accepted=False)` 返回。永久非法请求不 reroute，因为同构 engine 上结果不会
改变。

### 13.3 Ray/engine failure

- ingress ObjectRef 异常：请求标记 failed；
- LocalEngine health failure：终止本次 benchmark；
- 不静默丢请求；
- 最终摘要必须满足每个 dispatched request 恰好进入 accepted、rejected 或 failed 之一。

### 13.4 Abort

- `PENDING_INGRESS`：从 Router pending 中取消；若 remote 已 enqueue，则发送 abort；
- `PENDING_ADD`：LocalEngine 将请求标记 abort pending，event loop drain 时不加入 scheduler；
- `OWNED`：沿用当前 sticky-owner abort；
- 每个请求只发布一个 terminal event。

## 14. Legacy 兼容

`legacy_global` 继续使用现有同步本地 ADD：

```python
engine.add_request(seq)
```

Bench 可以通过统一接口调用：

```python
engine.submit_requests_async(due)
```

其 legacy adapter 可以立即完成 handle，不创建线程或 Ray ObjectRef。这样 bench 主循环统一，
但 legacy engine/scheduler 行为不变。

现有同步 `add_request()` 可以保留作为兼容接口：

- legacy：保持当前行为；
- hierarchical：内部调用异步 submit，并显式等待结果；
- 正式 rate-controlled benchmark 禁止调用 hierarchical 同步 wrapper。

## 15. 代码改动范围

### 15.1 `nanodeploy/engine/hierarchical_contract.py`

- 新增 `IngressAck`；
- 新增 `AddResultEvent`；
- 新增 `FirstTokenEvent`；
- 必要时给事件增加单调 event sequence，保证单 engine 内顺序。

### 15.2 `nanodeploy/engine/local_engine.py`

- 新增 ingress queue、capacity reservation 和 duplicate reservation；
- 新增快速 `enqueue_add`/`enqueue_add_batch`；
- event loop 在安全边界 drain ingress；
- 发布 AddResult/FirstToken/Finish 事件；
- 缓存 LoadSnapshot；
- shutdown/failure 时完成或失败所有 pending ingress。

### 15.3 `nanodeploy/engine/local_scheduler.py`

- 不新增 rate-control 或 future-arrival queue；
- 保留单请求 ADD，必要时增加只做机械循环的 `add_many()`；
- 确保逐请求错误隔离；
- 暴露准确的 waiting/admission/first-token duration。

### 15.4 `nanodeploy/engine/deployment_manager.py`

- `RayEngineTransport` 新增 async enqueue；
- 新增批量 enqueue；
- poll ingress/add/finish events；
- load snapshot 改为缓存读取。

### 15.5 `nanodeploy/router/request_router.py`

- 支持 pending ingress/pending add ownership；
- 批量 RoundRobin 分组；
- queue-full fallback；
- 异步 ACK/result 状态推进；
- terminal accounting。

### 15.6 `nanodeploy/engine/llm_engine.py`

- 新增 `submit_requests_async()`；
- 新增 pending 计数；
- poll ingress/add/first-token/finish；
- metric 在 scheduled arrival/dispatch 时创建，而不是 ADD ACK 后才创建；
- 不在 ADD ACK 时错误记录 `first_scheduled`。

### 15.7 `scripts/sp_ablation/bench_serving_overhead.py`

- 使用统一异步 submit/poll 主循环；
- 继续预构造全部 Sequence；
- 记录 dispatch lag 和各阶段计数；
- 正式路径只读取 cached diagnostics；
- 中断时输出所有 pending 状态；
- 最终结果校验 accounting 闭合。

## 16. 测试方案

### 16.1 CPU 单元测试

RequestRouter：

- RoundRobin 顺序；
- 同 arrival batch 的确定性；
- ingress full fallback；
- pending owner 到 owned；
- rejection/failed/terminal accounting；
- duplicate request ID；
- finish owner 校验。

LocalEngine ingress：

- enqueue 快速返回且不修改 scheduler；
- event loop 一次 drain 多个请求；
- ingress capacity reservation；
- reject/finish/abort 后 slot 释放；
- shutdown 时 pending 请求失败；
- AddResult 在 FinishEvent 前；
- cached LoadSnapshot 不进入 normal command queue。

Bench：

- fixed-rate arrival；
- burst arrival；
- producer 落后时一次提交全部 due；
- submit pending 时仍持续 poll completion；
- termination condition；
- Ctrl-C final diagnostic。

### 16.2 合成时钟测试

使用 fake clock 和 fake async engine：

- rate=20、duration=10s 时生成并 dispatch 200 个请求；
- engine ADD ACK 固定延迟 500 ms 时，dispatch rate 仍保持 20/s；
- completion 在 ADD pending 时返回也能被消费；
- p99 dispatch lag 不随 ADD ACK 延迟增长；
- 不提前 dispatch 未来请求。

这是本次修复最关键的无 GPU 回归。

### 16.3 Ray 控制面测试

使用 fake LocalScheduler/Executor：

- 两个 LocalEngine actors；
- active quantum 人为阻塞 500 ms；
- 20 requests/s 连续提交；
- 每个 boundary drain 约 10 个全局请求；
- pending actor calls 不超过配置上限；
- idle arrival 正确触发统一 wakeup；
- 结束后 0 pending ingress/add。

### 16.4 GPU 验证

顺序：

1. 单机 DP2SP4、短 30 秒 rate-20；
2. 两机 DP2SP8、短 60 秒 rate-20；
3. 两机 DP2SP8、正式 360 秒 arrival window；
4. drain 或按明确截止策略结束；
5. 检查 Ray 资源完全释放。

正式测试保持：

```text
scheduler_arch = hierarchical
sp_backend = hao_basic
cuda_graph_mode = full
enforce_eager = false
loop_count = 16
```

## 17. 验收标准

功能：

- 7,200 个请求都有唯一 scheduled arrival；
- 360 秒 arrival window 内全部 dispatch；
- 无容量限制时全部进入 ingress 并得到 scheduler AddResult；
- 每个 dispatched 请求最终为 completed/rejected/failed 之一；
- accounting 无丢失、无重复 terminal；
- completion poll 不因 ADD pending 停止；
- LocalScheduler 不包含 future arrival/rate-control 逻辑。

入口性能：

- achieved dispatch rate 在目标 20 requests/s 的 ±1% 内；
- p99 dispatch lag 小于一个 50 ms inter-arrival interval；
- request producer 不出现约 500 ms 的同步 ADD stall；
- 一秒结构化日志持续输出 dispatch/ingress/accepted/completed；
- cached diagnostic 不显著改变 dispatch rate。

调度：

- active wave 中请求 ingress-to-scheduler delay 不超过一个 quantum 加控制面 budget；
- LocalEngine 每个边界处理当前 ingress 中全部允许处理的请求；
- 两个 DP engines 的 RoundRobin 分配和 sticky ownership 正确；
- 无额外 preemption、queue-full 或 dummy work 异常。

清理：

- benchmark 结束后 pending ingress/add 为 0；
- Ray 0/16 GPU in use；
- 无 pending placement demand；
- 两个节点健康。

## 18. 实施阶段

### Phase A：协议和合成回归

- 增加 ingress/add event contract；
- 增加 fake-clock benchmark test；
- 增加 Router pending ownership；
- 不运行 GPU。

### Phase B：LocalEngine 非阻塞 ingress

- 增加 enqueue API、reservation、event-loop drain；
- cached LoadSnapshot；
- Ray fake actor 测试。

### Phase C：Bench 主循环

- 替换同步 `add_request()`；
- 加入完整 accounting、dispatch lag、非阻塞 poll；
- legacy adapter 回归。

### Phase D：指标修正

- FirstTokenEvent；
- observed TTFT/E2E；
- actor 内部阶段 duration；
- 最终 JSON/JSONL 结果。

### Phase E：GPU 验收

- 单机短测；
- 两机 60 秒；
- 两机正式 rate-20；
- 性能报告和资源清理。

## 19. 明确拒绝的替代方案

### 19.1 LocalScheduler 保存未来 arrival time

拒绝。它混淆负载生成和服务调度，会使 scheduler benchmark 与真实 serving ingress 不一致，
还会引入跨节点时钟和 idle timer 问题。

### 19.2 保持同步 ADD，只降低 loop count

拒绝。它只缩短 stall，不解除 producer 与 decode quantum 的耦合。

### 19.3 Bench 使用 ThreadPoolExecutor 包装当前 `add_request`

拒绝。当前 RequestRouter 的 owner map 和 RoundRobin cursor 不是面向并发调用设计；加锁后又会
恢复串行，绕过 Router 直发 actor 则破坏 ownership。

### 19.4 所有请求提前加入 LocalScheduler waiting queue

拒绝。它不再是 rate-controlled serving workload，会提前消耗 queue/KV admission 资源并
污染 TTFT/E2E。

### 19.5 每个 quantum 只处理一个 ADD

拒绝。其入口上限直接受 quantum duration 限制，与目标 request rate 无关。

## 20. 最终口径

本次问题的修复不是给 LocalScheduler 增加 arrival scheduler，而是恢复正确的在线 serving
边界：

```text
Bench 按时产生请求
  -> Load Balancer 快速接收并路由
  -> LocalEngine ingress 非阻塞排队
  -> LocalScheduler 在安全边界把已收到请求加入 waiting queue
```

LocalScheduler 继续只做本地 waiting/admission/decode。目标 request rate、arrival process
和 dispatch lag 全部由 Bench Load Generator 定义和验证。

## 21. 实施记录

2026-07-27 已完成：

- `IngressAck`、`AddResultEvent`、`FirstTokenEvent` 分层协议；
- RequestRouter 的 `PENDING_INGRESS`、`PENDING_ADD`、`OWNED` 状态推进，
  RoundRobin 和 queue-full fallback；
- LocalEngine 生命周期容量 reservation、非阻塞 ingress、边界批量 drain、
  pending-ingress abort 和缓存 `LoadSnapshot`；
- LLMEngine 的 `submit_requests_async()` 及 ingress/add/first-token/finish poll；
- rate-controlled benchmark 的非阻塞发送、完整 accounting 和入口延迟统计；
- 500 ms ACK 延迟、20 requests/s、200 请求的合成时钟回归；
- hierarchical 及相邻 CPU 测试共 49 项通过。

Phase E 已完成一个两机 DP2/SP8、20 requests/s、7,200 请求的 360 秒到达窗口
验收；7,200 请求全部成功。完整性能结论仍需补齐第 16.4 节定义的单机与多次
对照重复实验。
