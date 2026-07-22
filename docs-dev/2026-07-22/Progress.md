# Progress — 2026-07-22

## 2026-07-22 13:41 UTC

- 任务：为 LoongServe-style `_schedule_ls_combined_pool_step()` 增加默认关闭、由环境变量控制的分阶段计时，用实验数据严谨区分状态复制、waiting scan、future-KV、exact admission planner 与 Decode planning 开销。
- 开关：`NANODEPLOY_LS_SCHEDULER_PHASE_TIMING=1`（Scheduler 构造时读取）。
- 统计口径：顶层阶段互斥、可相加；内部热点记录 inclusive 时间与调用次数，不可直接相加计算占比。
- 当前进度：已加入环境变量解析、RAII 累加计时器以及 `ScheduleResult`/Scheduler 内部统计字段；尚需在调度各路径埋点、暴露 pybind、输出 Python 结构化日志、补测试并执行 `pip install -v -e .`。
- 工作区有用户原有修改与大量实验产物；提交时只包含本任务相关文件。
- 上一个相关提交：`54e425a fix: keep LoongServe decode loops fixed`。

## 2026-07-22 13:54 UTC

- 已完成 `_schedule_ls_combined_pool_step()` 分阶段计时、nested hotspot inclusive 计时及调用次数统计。
- 已通过 pybind 暴露三个字段，并在 `LLMEngine` 中输出 `mode=ls_scheduler_phase_timing` 结构化日志。
- 诊断记录使用显式 opt-in 的 WARNING 级别，可穿过正式 benchmark 默认的 INFO 过滤，无需开启全部 verbose 日志。
- 已补充默认关闭和开启后 admission/decode 统计测试；执行 `pip install -v -e .` 成功。
- 回归结果：`test_ls_decode_scheduler.py + test_ls_kv_scale_down.py` 57 passed；`test_llm_engine_kv_maintenance.py + test_sp_state_manager_prepared_iteration_cpu.py` 28 passed，共 85 passed。
- 使用说明见 `docs-dev/2026-07-22/ls_scheduler_phase_timing.md`。
