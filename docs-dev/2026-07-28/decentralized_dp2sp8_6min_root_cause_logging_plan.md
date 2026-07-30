# 去中心化 DP2SP8 6min 根因定位：日志完备性与补点计划

## 当前判断

如果直接按现有配置再跑一次 rate=40、360 秒注入的去中心化 DP2SP8，
当前日志足以：

- 判断两路 DP 的 request count、KV blocks、SP rank 负载是否均衡；
- 将 1 秒窗口内的 KV 差距与 decode ITL、排队、preemption 做相关分析；
- 粗分 admission、schedule、coordination、execute、postprocess 等阶段；
- 用全程累计值排除明显的 Ray 返回或 Python result rebuild 开销。

但当前粒度仍不足以把全部性能差距闭环到某个 quantum 和具体阶段。
它能证明“负载不均衡与慢同时出现”，不能严格区分：

1. KV 接近满载造成的资源压力；
2. 长上下文带来的 attention/GPU 计算量增加；
3. 一侧 DP 变慢后，另一侧在 EP16/DP consensus 上等待；
4. worker、DLSlime/Ray、driver result handling 中的具体尾延迟。

因此，正式消耗一次完整 6min 测试前，应先增加轻量
per-engine/per-quantum 结构化日志。

## 当前已有粒度

`[BENCH_DIAG]` 默认每 1 秒采样一次，包含：

- client unsent/outstanding/completed；
- ingress pending、add pending、waiting/running；
- 每个 LocalEngine 的 KV free blocks；
- 每个 SP rank 的 active/master request、batch、free blocks；
- 累计 admission、schedule、coordination、execute、postprocess；
- 累计 `ray.get`、result index/validate/pack/rebuild；
- useful token、dummy slot、preemption、decode quantum count。

这些字段可以做 1 秒累计差分，但一次采样可能跨越多个 quantum，且采样
时间没有与 `wave_id/quantum_id` 边界对齐。

目前 per-quantum ITL 样本在内存中包含
`engine_id/wave_id/quantum_id/itl_ms/token_count`，最终 summary 只保留
聚合分位数，没有持久化原始样本。

executor boundary recorder 当前也只保留
`sample_count/total/mean/min/max`，并在测试结束时写入 summary，无法将
最慢 quantum 与当时的 KV、batch 和 coordination wait 对齐。

## 已经排除的路径

120 秒同配置试跑已经表明：

- worker 完成到 `ray.get` 返回约 `1.6–1.8 ms/engine-quantum`；
- post-`ray.get` result rebuild 约 `0.11 ms/engine-quantum`；
- 同一 SP8 group 内 worker finish skew 约 `0.08 ms/quantum`。

这些量级不能解释旧 6min 测试约 5–7% 的性能差距。后续不应继续把重点
放在 request-id dict/set 或 token tuple 重建上。

## 正式 6min 前需要增加的轻量记录

每个 LocalEngine 每个 quantum 写一条 compact JSONL：

### 标识与负载

- `engine_id`
- `wave_id`
- `quantum_id`
- `elapsed_s`
- `waiting`
- `running`
- `useful_real_batch_size`
- 每个 SP rank 的 `active_master_requests`
- 每个 SP rank 的 `master_batch_size`
- 每个 SP rank 的 `free_blocks/total_blocks`
- `useful_decode_tokens`
- `control_dummy_slots`
- `preemption_count`

### LocalEngine 阶段时间

- `admission_ms`
- `schedule_ms`
- `consensus_wait_ms`
- `execute_ms`
- `postprocess_ms`
- `quantum_total_ms`

### LocalExecutor/driver 边界

- `actor_submit_ms`
- `send_seqs_ms`
- `ray_get_wait_ms`
- `worker_finish_to_ray_get_ms`
- `result_index_ms`
- `result_validate_ms`
- `result_pack_ms`
- `result_rebuild_ms`

### worker 本地时间

worker 使用自身的 monotonic clock 返回 duration，不依赖跨节点 wall
clock：

- `recv_seqs_ms`
- `prepare_or_update_decode_ms`
- `forward_loop_ms`
- `token_materialize_ms`
- `worker_total_ms`
- 上述指标在 8 个 rank 中的 min/max 及 critical rank。

如需要精确 GPU 时间，可在整个 16-loop forward 外围放一对 CUDA event，
利用现有 token `.tolist()` 已发生的同步读取 elapsed time；不要在每个
inner loop 显式 `cuda.synchronize()`。

## 不采用完整 execution trace

现有 hierarchical execution trace 会记录每个 rank 的 16 次 inner
forward，包含大量 Python 对象、Ray 返回序列化和 driver 校验，官方配置
说明也注明存在非平凡诊断开销。性能归因运行应保持该选项关闭，改用上述
每 quantum 一条的 compact timing。

旧 6min 运行约有一千个 engine-quantum 样本，因此 compact 日志规模很
小，预期开销也远低于完整 trace。

## 结果判定

| 观测 | 判定方向 |
| --- | --- |
| 重侧 KV/上下文上升，`worker_total_ms`/`forward_loop_ms` 同步上升 | attention work 路由不均衡 |
| 重侧 worker 变慢，轻侧 `consensus_wait_ms` 上升 | DP straggler 经 EP16/consensus 扩散 |
| worker 本地时间稳定，`ray_get_wait_ms` 或返回尾延迟上升 | DLSlime/Ray/序列化路径 |
| worker 与 Ray 稳定，rebuild/postprocess 上升 | Python 控制路径 |
| 两侧负载已均衡但 worker 仍整体慢于中心化 | 去中心化 worker/collective 执行路径 |

## 推荐执行顺序

1. 实现 compact per-quantum 日志，并用 1–2 分钟 smoke 验证字段闭合及
   埋点开销；
2. 使用完全相同的请求序列运行 360 秒去中心化 DP2SP8；
3. 如需严格解释“相对中心化”的差距，再运行一次同请求、同机器状态的
   中心化 DP2SP8 对照；单条去中心化运行只能定位内部瓶颈，不能消除跨轮
   系统波动；
4. 若负载不均衡复现，再用相同请求流做 `least_batch` 与
   `least_cache`/token-aware routing 因果 A/B。

只有当路由干预同时消除 KV/attention-work 差距、
`worker/consensus` 差距和 ITL/TPOT 差距时，才能把负载不均衡确定为
最终根因。

## 相关记录

- `docs-dev/2026-07-28/worker_boundary_central_vs_decentral_validation.md`
- `docs-dev/2026-07-28/hierarchical_result_rebuild_validation.md`
- 旧 6min 去中心化日志：
  `bench_logs/decentralized_dp2sp8_r40_6min_no_ingress_timecap_20260728_0830/decentralized_dp2sp8/console.log`

## 实现状态（2026-07-28）

上述 compact per-quantum 诊断已经实现：

- `Config.hierarchical_quantum_diagnostics` 默认关闭，不影响普通运行；
- worker 返回自身 monotonic duration，以及整个 decode loop 的 CUDA
  event duration；
- LocalExecutor 记录 actor submit、DLSlime send、`ray.get`、result
  rebuild 和每 rank worker timing；
- LocalEngine 将阶段时间与执行前后的逐 rank KV load 合并为
  `engine_id/wave_id/quantum_id` 记录；
- benchmark 周期性 drain 并写入
  `<timestamp>.hier_quantum.jsonl`，warmup quantum 会在正式请求前清空；
- `scripts/run_2node_rate30_6min_matrix.sh` 默认仅为去中心化 stage 打开
  lightweight quantum diagnostics，中心化 stage 不增加该开销；
- 可设置 `HIERARCHICAL_QUANTUM_DIAGNOSTICS=0` 关闭。

启用后的去中心化运行目录同时包含：

- `<timestamp>.jsonl`：逐请求指标；
- `<timestamp>.summary.json`：聚合指标；
- `<timestamp>.hier_quantum.jsonl`：逐 DP/quantum 根因诊断；
- `console.log`：1 秒 `[BENCH_DIAG]` 状态快照。

非 GPU 验证结果：

- hierarchical contract/control-plane/benchmark tests：`60 passed`；
- Python compile、shell syntax、`git diff --check`：通过；
- rate=40、360 秒 central/decentral 命令 dry-run：通过，诊断 flag 只出现
  在 decentralized DP2SP8 命令中。

真实 GPU smoke 和 6min 正式运行尚未在本次代码修改后启动。
