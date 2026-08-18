# 旧 decentralized 与 long-short 调度功能清理计划

状态：待 review，尚未修改运行时代码  
日期：2026-08-18  
调研基线：`gemm-update@f5ac869`

## 1. 目标与结论摘要

本次清理针对两类不再需要的功能：

1. 最初由单个中心化 `Scheduler` 加多个 per-worker waiting queue 模拟的
   `scheduler_mode="decentralized"`；
2. 以 prompt 长度阈值在 SP1 与指定 SP size 之间二选一的
   `dynamic_sp_size_strategy="long_short_sp8"`。

调研后的核心结论如下。

- 旧 decentralized **运行时本体已经在 2026-07-24 删除**。提交
  `00114bd support decentralized scheduler` 引入了该功能；提交
  `2f94869 Implement hierarchical scheduler phase 0 contracts` 删除了
  `SchedulerMode::DECENTRALIZED`、`_schedule_decentralized()`、
  `_schedule_prefill_for_worker()`、`_schedule_decode_for_worker()` 以及
  `SPStateManager` 的 per-worker waiting queue。当前源码中没有这些运行时残留。
- 现在的 `scheduler_arch="hierarchical"` 是新的分层调度架构，不是旧
  decentralized 分支的改名。它的 `RequestRouter + LocalEngine + LocalScheduler`
  及 ingress/local queue 都是当前在用的功能，不能作为旧代码删除。
- `scheduler_arch="legacy_global"` 是仍在使用的中心化基线与回滚入口，也不是
  本次所说的旧 decentralized。建议明确保留。
- 旧 decentralized 仍有三类外围残留：旧参数的 fail-fast 护栏与测试、仍传递
  `--scheduler-mode` 的过期脚本、以及把现行 hierarchical 实验命名为
  `decentralized_*` 的标签。脚本残留需要清理；参数护栏建议保留，理由见第 5 节。
- long-short 仍是一条完整的生产调用链，并非只有脚本残留。它同时存在于
  `Config`、Python/C++ 构造接口、C++ placement、hierarchical admission mirror、
  benchmark CLI 和实验脚本中，必须按同一个提交原子删除。

## 2. 范围定义

### 2.1 本次删除范围

- 旧 `scheduler_mode`/`--scheduler-mode` 的活动入口、过期脚本参数和误导性
  `decentralized` 实验标签；
- `long_short_sp8` 策略值及其两个专属配置字段：
  `dynamic_sp_long_request_threshold`、`dynamic_sp_long_request_size`；
- long-short 对应的 C++ enum、构造参数、成员、placement 分支与 pybind 参数；
- hierarchical frontend admission mirror 中同语义的阈值分支；
- long-short 专属 benchmark 参数、启动脚本、绘图工具和实验编排。

### 2.2 明确不删除

- `scheduler_arch="legacy_global"`；
- `scheduler_arch="hierarchical"`，以及 `RequestRouter`、`LocalEngine`、
  `LocalScheduler`、`DecodeCoordinator`；
- hierarchical 架构当前需要的 global ingress queue、local command queue、
  local scheduler waiting queue 和相关 queue latency metric；
- `routing_strategy`、`router_policy`、`sp_master_selector`；
- surviving SP placement：`legacy`、`bucket`、`fixed_sp_size` 和新的 decode batch
  planner；
- `RequestRouter._poll_centralized_admission()` 等“centralized admission”命名。
  这里的 centralized 指当前 LB 统一提交 admission plan，不是旧
  `scheduler_mode="decentralized"` 的反面分支；
- `docs-dev/` 中的 dated proposal、进展和性能结论。它们是历史记录，不做全仓
  关键词抹除；Git 历史同样保留。

## 3. 源码调研

### 3.1 旧 decentralized 的历史调用链

`00114bd` 当时增加了以下结构：

```text
Config.scheduler_mode
  -> Python Scheduler adapter
  -> pybind Scheduler(..., scheduler_mode)
  -> C++ SchedulerMode::DECENTRALIZED
  -> Scheduler::add() 提前选择 DP
  -> SPStateManager[dp].waiting / waiting_migration
  -> _schedule_decentralized()
       -> _schedule_prefill_for_worker(dp)
       -> _schedule_decode_for_worker(dp)
```

这个实现仍在单一中心进程内完成路由和所有调度，只是把一个 global waiting queue
拆成多个 `SPStateManager` queue；不同 DP lane 最后仍共用一个全局 `is_prefill`。
它不具备当前 hierarchical 架构的进程 ownership、独立 LocalScheduler 和 EP wave
协调语义。

`2f94869` 已经删除上述整条调用链。当前确认结果：

- `csrc/` 与 `nanodeploy/` 中不存在 `SchedulerMode`；
- `SPStateManager` 只有 `running`，没有 `waiting`/`waiting_migration`；
- 不存在 `_schedule_decentralized()` 或两个 `*_for_worker()` helper；
- `Config` 不再有 `scheduler_mode` 字段；
- 当前 C++ `Scheduler.waiting` 与 `waiting_migration` 是 legacy global scheduler 的
  两个全局 phase queue，同时也被 hierarchical 的单-DP `LocalScheduler` wrapper
  复用，不能按“多队列残留”删除。

当前真正残留为：

1. `nanodeploy/engine/llm_engine.py` 对 `scheduler_mode` 的显式 TypeError；
2. `tests/test_routing_config.py` 对已删除字段的回归检查；
3. 下列活动脚本仍传递已经不存在的 `--scheduler-mode`：
   - `scripts/sp_ablation/sweep_4node_variants.sh`
   - `scripts/run_kimi_conversation_16gpu.sh`
   - `scripts/run_issue003_deepseek_2node_dp16_dp2sp8_chain.sh`
   - `scripts/issue003/run_issue003_deepseek_rate5_sweep.sh`
   - `scripts/issue003/run_issue003_deepseek_rate5_sweep_0409_linbinbin.sh`
   - `scripts/issue003/run_issue003_longshort_sp8_r20_40_60.sh`
4. `scripts/run_2node_rate30_6min_matrix.sh` 用 `decentralized_dp*` 表示现在的
   `scheduler_arch=hierarchical`，容易把两代实现混为一谈。

其中 `scripts/issue003/start_bench.sh` 和 `scripts/sp_ablation/start_bench.sh` 只接受
`--scheduler-arch`。它们会把未知参数当作 rate，因此旧 `--scheduler-mode` 不只是
死文本，还可能把 `--scheduler-mode` 和 `centralized` 错当成 benchmark rate。

### 3.2 long-short 的当前调用链

```text
Config.dynamic_sp_size_strategy == "long_short_sp8"
Config.dynamic_sp_long_request_threshold / _size
  |
  +-> nanodeploy.engine.scheduler.Scheduler
  |     -> pybind Scheduler constructor
  |     -> C++ Scheduler constructor
  |     -> SPStateManager constructor
  |     -> DynamicSPSizeStrategy::LongShortSP8
  |     -> SPStateManager::can_allocate()
  |          short: target SP = 1
  |          long : target SP = configured size (historically attention_sp)
  |
  +-> AdmissionPlannerConfig.from_config()
        -> AdmissionPlanner._plan_legacy()
        -> 生成与 C++ 相同的 frontend AdmissionReservation
```

关键文件：

- 配置与校验：`nanodeploy/config.py`；
- Python 构造适配：`nanodeploy/engine/scheduler.py`；
- hierarchical mirror：`nanodeploy/router/admission_planner.py`；
- C++ 接口与转发：
  `csrc/nanodeploy/scheduler/scheduler.{h,cpp}`；
- C++ 策略实现：
  `csrc/nanodeploy/scheduler/sp_state_manager.{h,cpp}`；
- Python ABI：
  `csrc/python/scheduler_binding.cpp`、
  `csrc/python/sp_state_manager_binding.cpp`。

必须同时删除 C++ placement 与 Python mirror。只删一边会导致 hierarchical router
认为某请求可按一种 SP placement 接收，而 LocalScheduler/C++ 实际使用另一种
placement，进而触发 reservation mismatch、错误的 KV credit 或不必要的
`queue_full`。

当前没有发现专门覆盖 long/short 阈值边界以及 C++/frontend 等价性的命名测试；
现有 `test_frontend_admission_mirror_matches_local_cpp_placements` 主要提供一般性 mirror
覆盖。因此清理后的测试重点应放在 surviving `legacy`/`bucket` placement，而不是
简单删除断言后结束。

### 3.3 long-short 的 CLI 与脚本影响面

直接暴露 long-short 参数的 CLI/adapter：

- `examples/bench_serving_overhead.py`
- `scripts/issue003/bench_serving_overhead.py`
- `scripts/sp_ablation/bench_serving_overhead.py`
- `examples/dummy_prefill.py`
- `scripts/issue003/start_bench.sh`
- `scripts/sp_ablation/start_bench.sh`

专门围绕 long-short 或 threshold sweep 编排的脚本/工具，应从活动源码树删除而不是
静默换成另一策略：

- `scripts/decent-e2e/` 整个目录；
- `scripts/plot_nano_longshort_matrix.py`；
- `scripts/issue003/run_issue003_longshort_sp8_r20_40_60.sh`；
- `scripts/issue003/run_kimi_issue001_4node_dp4sp8_longshort_10min_rates.sh`；
- `scripts/issue003/run_dpsk_gemini_kimi_issue_rates_ordered.sh`（依赖上一脚本）；
- `scripts/run_issue001_issue003_longshort_0409_linbinbin.sh`；
- `scripts/run_issue001_issue003_longshort_then_legacy_rate5_chain.sh`；
- `scripts/run_issue005_4node_rate40_dp32_dp4sp8_original_setting.sh`；
- `scripts/sp_ablation/run_4node_issue001_rate40_ablation.sh`；
- `scripts/run_kimi_conversation_16gpu.sh`；
- `scripts/run_kimi_32x8k_16gpu_profile.sh`。

最后两个文件名不含 long-short，但当前主要实验矩阵分别是 threshold 对比和
`DP2SP8 long_short` 对 `DP16 legacy` 对比。建议删除，避免悄悄改变历史实验含义；
如这些 workload 仍需保留，应在后续用 surviving policy 新建明确命名的脚本。

其余混合/通用脚本保留并移除 long-short 参数，见实施阶段 3。

## 4. 计划中的目标状态

清理完成后，外部可见的 SP 策略为：

- `dynamic_sp_size_strategy="legacy"`；
- `dynamic_sp_size_strategy="bucket"`；
- 与上述 dynamic strategy 互斥的 `fixed_sp_size > 0` baseline。

调度架构只使用：

- `scheduler_arch="legacy_global"`；
- `scheduler_arch="hierarchical"`。

活动源码、示例和脚本中不再把 hierarchical 称为 decentralized，也不再传递
`--scheduler-mode`。只有明确的 removed-option fail-fast 护栏、相应负向测试和
dated historical notes 可以保留旧名称。

## 5. 实施计划

### 阶段 0：冻结范围并保护现有工作区

1. 实施前重新记录 `git status --short`。
2. 当前调研时已有用户改动：
   - `nanodeploy/config.py`
   - `nanodeploy/engine/local_executor.py`
   - `nanodeploy/engine/ray_executor.py`
   - `nanodeploy/worker/model_runner.py`
3. `config.py` 与本次清理有重叠，实施时只做目标字段/hunk 的精确修改，不覆盖用户
   现有配置改动；其余三个文件不应被本任务触碰。
4. 先运行现有 CPU focused tests，记录任何基线失败，避免把环境或用户改动造成的
   失败误归因于清理。

### 阶段 1：清理旧 decentralized 的外围残留

1. 保留 `LLMEngine` 对 `scheduler_mode` 的 fail-fast 语义，但把它与 long-short
   removed options 合并为一个小型、集中式的 removed-option 检查。
2. 保留一个行为级负向测试。不能直接删掉护栏：`LLMEngine.__init__` 当前会过滤掉
   不在 `Config` dataclass 中的 kwargs；若只删护栏，旧
   `scheduler_mode="decentralized"` 会被静默忽略并默认运行 `legacy_global`，比显式
   报错更危险。
3. 对仍有价值的通用脚本，把变量和参数统一迁移为：
   - `SCHEDULER_ARCH=legacy_global|hierarchical`
   - `--scheduler-arch "$SCHEDULER_ARCH"`
4. `scripts/run_2node_rate30_6min_matrix.sh` 的 stage 名改为
   `legacy_global_dp*`/`hierarchical_dp*`，不保留 `decentralized_*` alias，避免继续
   传播两代实现混淆。
5. 删除的 long-short 专属脚本无需先修旧 flag；它们在阶段 3 直接移除。

### 阶段 2：原子删除 long-short 生产调用链

1. `nanodeploy/config.py`
   - `dynamic_sp_size_strategy` 的合法值收缩为 `legacy|bucket`；
   - 删除 `dynamic_sp_long_request_threshold`、
     `dynamic_sp_long_request_size`；
   - 删除 size=0 归一化及范围校验；
   - 更新 fixed/bucket 冲突报错和注释；
   - 保留 fingerprint 中的 strategy/bucket/fixed 信息。
2. `nanodeploy/engine/llm_engine.py`
   - 在集中式 removed-option 检查中加入两个已删除 long-short 字段；
   - 对 `dynamic_sp_size_strategy="long_short_sp8"` 保持明确 ValueError；
   - 不在本任务中顺带改成“拒绝所有未知 kwargs”，以免扩大公共 API 兼容面。
3. `nanodeploy/engine/scheduler.py`
   - 从 C++ `Scheduler` 构造调用中删除两个 long-short 参数。
4. `nanodeploy/router/admission_planner.py`
   - 从 `AdmissionPlannerConfig` 及 `from_config()` 删除两个字段；
   - 删除 `_plan_legacy()` 的 long-short forced-rank 分支；
   - 保留 fixed/bucket 的 `recompute_segments` 行为。
5. `csrc/nanodeploy/scheduler/scheduler.{h,cpp}`
   - 删除构造参数、成员、转发和日志字段；
   - 不改 `waiting`/`waiting_migration`、`worker_state` 或 surviving scheduler path。
6. `csrc/nanodeploy/scheduler/sp_state_manager.{h,cpp}`
   - 删除 `DynamicSPSizeStrategy::LongShortSP8`；
   - 删除 threshold/size 构造参数与成员；
   - 删除字符串解析、name/logging 和 `can_allocate()` 阈值分支；
   - 保留 `Legacy`/`Bucket`，并让未知字符串继续 fail fast。
7. 两个 pybind 文件同步删除参数、默认值和构造转发。该步骤会改变 extension 构造
   ABI，必须与 C++/Python adapter 同一提交完成，并在测试前重新 editable install。

### 阶段 3：清理 CLI、实验脚本和文档入口

1. 三份 `bench_serving_overhead.py`：
   - 删除 `--long-request-sp-threshold`、`--long-request-sp-size`；
   - strategy choices 改为 `legacy|bucket`；
   - 将 `--dynamic-sp-bucket-preset` 作为正式参数透传，保证 surviving bucket 无需
     monkeypatch 才能运行。
2. 两份 `start_bench.sh`：
   - 删除 long-request 默认变量、help、parser、日志、tag 和 Python 参数转发；
   - 增加/保留 bucket preset 的正式转发；
   - 对未知 `--*` 参数直接报错，不再把拼错的 option 当作 rate。这也能防止旧
     `--scheduler-mode` 再次静默污染 rate 列表。
3. `examples/dummy_prefill.py`：删除仅用于二选一 long-short 的
   `--sp-size-policy` 与 threshold 参数；默认走 surviving legacy placement。
4. 保留并清理以下通用脚本：
   - `scripts/sp_ablation/sweep_4node_variants.sh`
   - `scripts/issue003/run_issue003_deepseek_rate5_sweep.sh`
   - `scripts/issue003/run_issue003_deepseek_rate5_sweep_0409_linbinbin.sh`
   - `scripts/run_issue003_deepseek_2node_dp16_dp2sp8_chain.sh`：删除 threshold stage，
     保留 DP16 与 DP2SP8 legacy 对比，并改成明确的 legacy 命名；
   - `scripts/run_issue001_deepseek_v3_issue001_bucket.sh`：删除 long-request no-op
     参数，并改用正式 bucket preset CLI，删除 argparse/LLM monkeypatch；
   - `scripts/run_kimi_dp16_ep16_gemm_debug.sh`：删除 legacy 模式下无效的 threshold
     参数。
5. 删除第 3.3 节列出的 long-short 专属脚本/工具。
6. 更新 `scripts/README.md`，删除已移除绘图工具与实验脚本的活动说明；不修改
   dated `docs-dev/` 历史记录。

### 阶段 4：测试与验证

1. 增补/调整 CPU tests：
   - `long_short_sp8` 被配置层明确拒绝；
   - 两个 removed long-short kwargs 不会被 `LLMEngine` 静默忽略；
   - 旧 `scheduler_mode` 仍明确报错；
   - `legacy` 与 `bucket` 的 frontend admission mirror 和 C++ placement 保持一致；
   - fixed SP 与 bucket/legacy 的互斥校验不回归。
2. C++ 修改后按仓库要求重装：

   ```bash
   export http_proxy=http://127.0.0.1:15409
   export https_proxy=http://127.0.0.1:15409
   export HTTP_PROXY=http://127.0.0.1:15409
   export HTTPS_PROXY=http://127.0.0.1:15409
   python -m pip install -v -e .
   ```

3. 先运行 focused CPU suites：

   ```bash
   python -m pytest tests/test_routing_config.py
   python -m pytest tests/test_hierarchical_contract.py
   python -m pytest tests/test_hierarchical_control_plane.py
   python -m pytest tests/test_hierarchical_serving_ingress.py
   ```

4. 运行 shell 静态检查：

   ```bash
   bash -n <每个保留且修改过的脚本>
   ```

   并对两个 `start_bench.sh` 做不启动 benchmark 的参数解析测试，覆盖
   `legacy_global`、`hierarchical+bucket` 和未知 option 拒绝。
5. 运行残留扫描。活动源码/脚本应无以下符号，removed-option 护栏与负向测试除外：
   - `SchedulerMode`
   - `_schedule_decentralized`
   - `--scheduler-mode`
   - `long_short` / `longshort` / `LongShortSP8`
   - `dynamic_sp_long_request_*`
   - `--long-request-sp-*`
6. CPU 与构建通过后再考虑 GPU smoke。GPU 不是本次纯策略删除的第一验收门槛；如做，
   必须先获得 elevated permission，并在 driver 环境设置 `SLIME_QP_NUM=4`。建议只跑
   两个 surviving path：一个 `legacy_global+legacy`，一个
   `hierarchical+bucket`，记录 GPU 数与 topology。

## 6. 提交拆分建议

为便于 review 与回滚，建议拆成三个窄提交：

1. `fix: remove stale decentralized scheduler script options`
   - 只做 `scheduler_mode -> scheduler_arch` 脚本迁移和命名清理；
2. `refactor: remove long-short SP scheduling policy`
   - Config、Python mirror、C++、bindings、tests；
3. `chore: remove obsolete long-short experiment tooling`
   - CLI、launch/plot scripts、README。

第 2 个提交必须原子覆盖 Python mirror、C++ implementation 和 bindings；不能拆成会
产生 frontend/local placement 不一致或 extension 构造 ABI 不一致的中间提交。

## 7. 风险与控制

| 风险 | 后果 | 控制方式 |
|---|---|---|
| 把当前 hierarchical queue 当成旧 per-worker queue 删除 | 当前分层运行时不可用 | 按第 2.2 节白名单保留，只删除历史 `scheduler_mode` 语义 |
| C++ 构造参数与 pybind/Python adapter 不同步 | 编译失败或运行时构造 TypeError | 同一提交改六个 C++/binding 文件与 adapter，随后 editable install |
| 只删 C++ 或只删 admission mirror | frontend reservation 与本地 placement 失配 | 两边原子删除，并跑 mirror 等价测试 |
| 删除字段后 kwargs 被静默过滤 | 用户以为 long-short 仍生效，实际跑默认策略 | 保留集中式 removed-option fail-fast 护栏 |
| 将 long-short 脚本直接改成 bucket/legacy | 历史脚本名与实验语义不一致，结果不可比较 | 专属脚本直接删除；新策略另建明确命名脚本 |
| bucket CLI 仍依赖 monkeypatch | surviving policy 的公开入口脆弱 | 把 bucket preset 变成正式 benchmark/start_bench 参数 |
| 覆盖当前未提交的 `config.py` 改动 | 丢失用户工作 | 精确 hunk 修改，提交前逐文件 diff，绝不 restore/reset 用户文件 |

## 8. Review 决策点

请重点确认以下选择：

1. **保留 `legacy_global`（推荐）**：本计划只删除旧
   `scheduler_mode="decentralized"`，不删除当前中心化基线。
2. **保留 removed-option fail-fast 护栏（推荐）**：源码仍会出现极少量旧名称，但不会
   存在功能实现；这样可以避免 `LLMEngine` 静默忽略旧参数。
3. **专属实验脚本直接删除（推荐）**：依赖 Git 历史追溯，不在活动树内另建 archive。
4. **正式暴露 bucket preset CLI（推荐）**：删除 long-short 后，用小范围相邻清理替代
   当前 bucket benchmark 的 argparse/LLM monkeypatch。
5. **`run_kimi_conversation_16gpu.sh` 与
   `run_kimi_32x8k_16gpu_profile.sh` 删除（推荐）**：两者核心矩阵依赖 long-short；若仍有
   独立使用价值，需要 review 时指定希望保留的 surviving policy 和新脚本命名。

以上决策确认后再进入实现，避免把“删除旧 decentralized”误扩展为删除
`legacy_global` 或当前 hierarchical control plane。
