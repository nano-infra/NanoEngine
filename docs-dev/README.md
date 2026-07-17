# NanoDeploy 开发进展索引

文档与实验产物按日期归档。日期优先采用文档中明确记录的日期；未注明日期时，采用 Git 首次纳入日期或实验产物生成日期。

## 最新进展

当前最新报告是 2026-07-17 的 [Original NanoDeploy loop1 enforce eager 两机 A/B](2026-07-17/nanodeploy_original_loop1_enforce_eager_cuda_launch_blocking_20260717.md)。在相同 worker-level `CUDA_LAUNCH_BLOCKING=1` 配置上启用 `--enforce-eager` 后，运行稳定推进到用户手动中止的 616 秒 / 5,224 请求，期间 KV 一度达到 100% 并发生正常 preemption，但未出现 CUDA illegal。对照 full graph 在 154 秒 / 2,157 请求同步失败于 `graph.replay()`，结果高置信指向 CUDA Graph 路径。

## 2026-07-17

以下按进展时间从新到旧排列：

1. [Original NanoDeploy loop1 enforce eager 两机 A/B](2026-07-17/nanodeploy_original_loop1_enforce_eager_cuda_launch_blocking_20260717.md)
2. [Original NanoDeploy loop1 CUDA_LAUNCH_BLOCKING 两机诊断](2026-07-17/nanodeploy_original_loop1_cuda_launch_blocking_diagnosis_20260717.md)
3. [Original NanoDeploy loop_count=1 两机复跑失败与三组局部对照](2026-07-17/nanodeploy_original_loop1_2node_failure_and_partial_comparison_20260717.md)
4. [Original NanoDeploy 同配置复跑与 LS future-KV 性能对比](2026-07-17/nanodeploy_original_rerun_vs_ls_futurekv_2node_r20_20260717.md)
5. [LS Decode future-KV 两机 rate=20 六分钟验收](2026-07-17/ls_decode_future_kv_2node_r20_141gb_result_20260717.md)
6. [LS Decode future-KV admission implementation](2026-07-17/ls_decode_future_kv_admission_implementation.md)

同目录保留 Original loop1 失败运行、CUDA_LAUNCH_BLOCKING 诊断的原始日志、结构化失败摘要，以及已完成两机运行的 7,200 条 JSONL 指标。

## 2026-07-16

以下按进展时间从新到旧排列：

1. [LS-Decode owner-aware 两机复跑与容量鲁棒性复盘](2026-07-16/ls_decode_owneraware_2node_capacity_postmortem_20260716.md)
2. [LS-Decode owner-aware planner 修复与单机验证](2026-07-16/ls_decode_owner_aware_planner_fix_20260716.md)
3. [LS-Decode T=128 16 卡测试与调度假阴性修复建议](2026-07-16/ls_decode_t128_6min_result_and_planner_fix_20260716.md)
4. [LS-Decode-Core T=128、16 卡 rate=20 六分钟测试结果](2026-07-16/ls_decode_t128_r20_6min_result_20260716.md)
5. [LS-Decode-Core rate 20 频繁抢占复盘](2026-07-16/ls_decode_r20_preemption_postmortem_20260716.md)
6. [两节点 LoongServe-style Issue 1% Bench Serve](2026-07-16/bench_ls_decode_serving_2node.md)
7. [LS-Decode-Core 两机 16-GPU Consolidation 5 分钟压力测试](2026-07-16/ls_decode_longrun_16gpu_pressure_5min.md)

同目录还保留上述工作的 JSON、JSONL 与原始日志。

## 2026-07-15

- [LS Decode 动态负载 8-GPU E2E](2026-07-15/ls_decode_elastic_e2e_8gpu.md)
- [LS KV Consolidation 单机 8-GPU Smoke](2026-07-15/ls_kv_consolidation_8gpu_smoke.md)
- 同目录包含 8-GPU long-run、pressure、forward timing、interleaved scale-down 及 consolidation 实验产物。

## 2026-07-14

- [LS-Decode-Core Decode Logical Batch 原子 Admission 设计](2026-07-14/ls_decode_atomic_batch_admission_design.md)
- [LS-Decode-Core 单机 8 卡预检](2026-07-14/ls_decode_core_8gpu_preflight.md)
- [Decode-only LS KV Consolidation 可行性分析与实施设计](2026-07-14/ls_decode_kv_consolidation_feasibility_design.md)
- 同目录包含 batch-level smoke 结果。

## 2026-07-13

- [LoongServe-Style Multi-Master Core 调度设计](2026-07-13/loongserve_style_scheduler_design.md)
- 同目录包含 ITL 对比、8-GPU long-run 和 CUDA Graph smoke 实验产物。

## 维护约定

- 新文档和实验产物放入 `docs-dev/YYYY-MM-DD/`。
- 当天有多份进展报告时，在本索引中按最新到最旧排列。
- JSON 中的 `output_json`、`completion_jsonl` 等字段记录实验运行时使用的原始路径，因此归档时不改写。
