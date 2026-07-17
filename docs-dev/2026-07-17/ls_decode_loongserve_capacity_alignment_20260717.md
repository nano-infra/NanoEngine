# LS Decode 向 LoongServe 容量重组策略对齐（2026-07-17）

## 结论

本次补齐了两条此前缺失的调度闭环：

1. Decode group 因 KV append 容量不足时，不再盲目与最老 group 合并，而是优先吸收一个自身可 Decode、剩余 KV 容量最大的 group；没有合适 donor 时，原有 planner 继续尝试吸收全局空闲 rank。
2. pending logical batch 在本轮确实发生 no-fit 时，不再无条件禁止 KV consolidation。调度器会对每个候选 consolidation 做一次精确的 post-commit admission 影子规划，只有它能让完整 pending batch 在下一轮 standalone admission 或 merge admission 成功时才执行搬运。

这对应 LoongServe Decode 调度中的“先合并有余量 batch，再按缺口加入 idle instance”主干策略，同时保留 NanoDeploy 已有的事务化 KV 搬运和原子 logical-batch admission。

## 与 LoongServe 的对应关系

| LoongServe | NanoDeploy 本次实现 |
| --- | --- |
| `manager.py:844-905` 按 `idle_tokens` 将不能 Decode 的 batch 与有余量 batch 合并，不足部分再 scale up | `scheduler.cpp:1875-2019, 2381-2400` 计算 block 级 append slack，按可行性和余量选择 donor，随后由原有 planner 使用未分配 rank |
| `idle_tokens = capacity - used - decode_need` | `append_slack = free blocks + 可回收尾块 - 本轮最小 append blocks - reserved headroom` |
| donor 优先取 idle token 较大的可 Decode batch | donor 必须自身 `append_slack >= 0`；优先选合并后 exact planner 成功者，再比较 append slack、free blocks 和稳定 group id |
| Admission/Decode 可以通过资源重组继续推进 | pending no-fit 会触发候选 consolidation，但只有 exact post-commit admission proof 成功才执行 |
| 空闲实例补足剩余 KV 缺口 | `plan_iteration_masters_source_greedy()` 继续把全局未分配 rank 作为 `extras`，可报告 `scale_reason=memory` |

LoongServe 参考实现位置：

- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/manager.py:844`
- `/mnt/nvme1n1/ml_research/linbinbin1/LoongServe/loongserve/longserve_server/router/req_queue.py:54`

## 1. Decode 容量驱动的 group 重组

### 容量判定

`Scheduler::_ls_group_append_slack()` 使用 block 作为统一单位：

```text
available_blocks
  = group ranks 当前 free blocks
  + 已分配 block table 中超过 committed context 的可回收尾块

minimum_append_blocks
  = 每个 running request 在最佳现有 rank 上容纳 pending token + next token 的最小新增 block 数
  + reserved_blocks_per_req 对应的最小安全余量

append_slack = available_blocks - minimum_append_blocks
```

只有 `append_slack < 0` 才把 group 视为确定的 KV capacity deficit。这个判定是低成本下界：它用于决定是否进入重组，不替代后面的 exact iteration planner。

### donor 选择

`Scheduler::_select_ls_capacity_merge_target()` 的顺序为：

1. 排除自身也无法 Decode 的 donor，避免把两个独立缺口简单叠在一起；
2. 优先选择合并 allocation 后 exact Decode planner 可以直接成功的 donor；
3. 再比较 donor 的 append slack；
4. 再比较当前 free blocks；
5. 完全相同时选择较小 group id，保证结果稳定。

KV 压力预检查发生在消费全局 idle rank 之前；如果没有 donor，原有 iteration planner 仍会使用 `extras` 自动 memory scale-up。原有 compute-threshold merge 和 planner-failure merge 也改为优先使用同一个容量 donor 选择器，不再固定选最老 group。

## 2. pending-aware KV consolidation

### 触发条件

只在同一个 `schedule()` admission pass 中记录到真实 no-fit batch 时，pending 才能触发压力型 consolidation。仅仅 queue 中存在 pending 请求，不足以触发搬运。

压力型 consolidation 会跳过面向后台整理的：

- utilization candidate threshold；
- stable window；
- cooldown；
- check interval。

执行模式仍严格遵守：

- `ls_kv_consolidation_max_source_blocks_per_event`；
- `ls_kv_consolidation_target_high_watermark`；
- 单事务 reservation/commit/abort 约束。

### 精确收益证明

`plan_ls_kv_scale_down()` 成功后，destination blocks 已经被 reservation 占用，但 ACTIVE context 尚未切换。`_ls_consolidation_enables_pending_batch()` 构造 post-commit 影子状态：

1. 将 commit 后会释放的 source blocks 加回 shadow free counters；
2. 对参与搬运的 sequence 使用 `staged_context`，而不是旧 ACTIVE context；
3. 将被释放的 source rank 加入 post-consolidation unallocated pool；
4. 使用与正常 admission 相同的 placement、receiver quota、headroom 和 future-KV high-water 规则；
5. 先尝试 standalone admission，再依次尝试把完整 logical batch merge 到现有 group。

只有至少一个完整 admission 方案成功时，execute 模式才返回 `KV_CONSOLIDATION`，telemetry reason 为 `execute_pressure`。否则立即 abort reservation，保持 block counters 和 ACTIVE metadata 不变，reason 为 `pending_no_benefit` 或具体安全限制。

shadow 模式也会执行同样的精确证明，但随后 abort reservation，只报告 `shadow_pressure_candidate`；execute-only 的 source transfer budget 不会屏蔽 shadow 观测。

## 为什么这能降低 cliff 风险

旧行为存在两个直接耦合：

- 有 pending batch 时完全禁止 KV consolidation，即使释放一个 rank 就能让它入场；
- Decode 失败时固定与最老 group 合并，不看那个 group 是否真的有 KV 余量。

新行为把决策改为容量驱动，并要求搬运对 admission 有可证明的单步收益。因此可以减少“资源分布方式造成 no-fit，但总容量仍够”的排队停滞，也避免为了 pending 做无效 P2P 搬运。

## 测试覆盖

新增回归覆盖：

- KV 压力 group 会选择高余量 donor，而不是最老 donor；
- pending no-fit 可以绕过后台整理窗口，执行一次能使下一轮 admission 成功的 consolidation；
- future-KV admission 开启时，收益证明使用 staged context 和 post-commit free blocks；
- 搬运无法带来 admission 收益时，reservation 被完整回滚；
- shadow pressure proof 不保留 block reservation，也不受 execute-only transfer budget 影响。

验证命令：

```bash
pytest -q \
  tests/test_ls_decode_scheduler.py \
  tests/test_ls_decode_planner.py \
  tests/test_ls_kv_scale_down.py \
  tests/test_ls_decode_config.py \
  tests/test_llm_engine_kv_maintenance.py
```

结果：`97 passed`。

C++ 扩展已按仓库要求通过 `pip install -v -e .` 重新构建。

## 仍未完全对齐的部分

本次是容量重组闭环，不是 LoongServe 调度器的完整复刻。后续优先级建议如下：

1. **pending logical batch 可拆分/重组**：NanoDeploy 仍把 seal 后的 logical batch 作为原子 admission 单元；LoongServe waiting queue 可以逐请求选择和跳过。当前 exact proof 也只证明一个完整 batch。
2. **Prefill 实测性能模型**：LoongServe 用 profiler 预测不同实例数的 Prefill 时间；NanoDeploy 当前仍主要按内存可行性和固定阈值确定初始 DoP。
3. **Decode compute-bound DoP**：LoongServe 使用 `min_comp_bound_decoding_batch_size` 动态判断是否增加实例；NanoDeploy 仍使用 `ls_decode_batch_per_master` 固定阈值。
4. **多跳资源重组**：当前 pending proof 验证一次 consolidation 后能否直接 admission，不搜索“连续两次 consolidation”或“先合并多个 group 再 admission”的组合空间。
5. **pause/offload 层级**：NanoDeploy 有 structured preemption，但尚未完整复刻 LoongServe 的 KV-keep/KV-offload pause 策略。

下一步若以减少 140/141 GiB 附近的 cliff 为目标，应先做第 1 项（自适应拆分 pending batch），然后用两机 rate=20、6 分钟配置比较：no-fit 次数、pending queue age、capacity merge 次数、memory scale-up 次数、consolidation proof/execute/no-benefit 次数以及端到端吞吐和延迟。
