# LoongServe-style scheduler 分阶段计时

## 开启方式

计时默认关闭。必须在创建 Scheduler 的 NanoDeploy 进程启动前设置：

```bash
export NANODEPLOY_LS_SCHEDULER_PHASE_TIMING=1
```

随后照常运行 serving benchmark。Python 日志会为每次 LS scheduler 调用额外输出一条结构化记录：

```text
{'mode': 'ls_scheduler_phase_timing',
 'action': '...',
 'planning_latency_ms': ...,
 'phase_ms': {...},
 'counts': {...}}
```

该诊断记录在显式开启后使用 WARNING 日志级别，因此正式 benchmark 默认关闭 INFO 日志时也能被 `tee` 捕获，不需要额外传 `--verbose-nanodeploy-logs`。

关闭时不输出该记录，也不执行 `steady_clock::now()` 的内部埋点调用；`ScheduleResult` 中两个统计 map 为空。

## 统计口径

`phase_ms` 中以下顶层阶段互斥，可以相加后和 `total` 比较：

- `snapshot_copy`：每次 prepare attempt 的完整调度状态快照。
- `mandatory_safety`：mandatory graph safety、group 合并或扩容检查。
- `admission`：pool admission，包括候选扫描、exact plan 和 reservation prepare。
- `kv_consolidation`：低 KV consolidation candidate 检查和 prepare。
- `decode_plan_prepare`：每个 DP/group 的动态 Decode plan、校验和 reservation prepare。
- `publication`：最终 composition 校验、commit 及 scheduler state publication。
- `unattributed`：总时间减去以上阶段，主要包括 attempt 间对象析构、错误/重试处理和框架胶水。

以下 `nested.*` 是内部热点的 inclusive 时间，会和顶层阶段重叠，也可能彼此嵌套，不能与顶层阶段相加计算百分比：

- `nested.rollback_shadow_copy`
- `nested.admission_scan`
- `nested.future_kv_pool`
- `nested.future_kv_empty_system`
- `nested.empty_system_fit`
- `nested.initial_placement`
- `nested.admission_plan`

`counts` 同时记录 prepare attempt、完整/rollback snapshot、尝试 admission 的 pool、扫描的 waiting candidate、各类 future-KV/exact planner 调用以及 Decode pool plan attempt 次数。分析时优先同时看总时间和次数，例如：

```text
nested.future_kv_pool / future_kv_pool_calls
nested.admission_plan / admission_plan_calls
nested.admission_scan / waiting_candidates_scanned
```

这样可以区分“单次 planner 很慢”和“单次不慢但调用次数被 waiting/OOE 放大”。
