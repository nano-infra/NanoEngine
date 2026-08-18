# 开源前调度、SP placement 与 SP backend 清理计划

状态：已实施；非 GPU 验收通过，GPU 测试按要求未运行
日期：2026-08-18  
调研基线：首次 `gemm-update@f5ac869`；补充 `gemm-update@12acd29`

实施记录：

- `e5a6168 refactor: remove stale decentralized routing interfaces`
- `a4e292d refactor: remove sp_debug placement path`
- `d6d2558 refactor: remove unused dynamic SP placement paths`
- `6c79b10 chore: remove obsolete long-short experiment tooling`
- `a40aa06 refactor: remove legacy_ll SP backend`
- `9fd0282 refactor: remove nccl_compact SP backend`

验收结果：editable build 成功；routing/config、hierarchical contract、control plane、
serving ingress 与 SP backend 共 148 项 CPU 测试通过；所有保留且修改过的 shell
脚本通过 `bash -n`，修改过的 Python 入口通过 `py_compile`。未运行 CUDA、`torchrun`
或其他 GPU 测试。

## 1. 目标与结论摘要

本次清理针对七类不再需要的功能：

1. 最初由单个中心化 `Scheduler` 加多个 per-worker waiting queue 模拟的
   `scheduler_mode="decentralized"`；
2. 以 prompt 长度阈值在 SP1 与指定 SP size 之间二选一的
   `dynamic_sp_size_strategy="long_short_sp8"`；
3. 允许 legacy 单请求 SP allocator 在初始 SP rank 数分配失败后继续扩大到更多
   rank 的 `enable_dynamic_sp_size`；
4. MLA SP all-to-all 的旧 `legacy_ll` 后端，并把默认后端改为 `hao_basic`；
5. 旧 decentralized DP 路由遗留的 `routing_strategy="VLLMLoadBalance"`。
   保留现行 `RoundRobin`、`LeastBatch`、`LeastCache`，以及 hierarchical 独立使用的
   `router_policy`；
6. 实验性的 MLA SP variable-split NCCL 后端 `nccl_compact`。SP backend 最终只保留
   `hao_basic` 与静态 padding 版本 `nccl`；
7. 绕过正常 segment placement、以 KV block size 做简化切分的 `sp_debug` 分支。

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
- `VLLMLoadBalance` 不是 vLLM scheduler 集成，而是旧 decentralized
  `select_dp_worker_for_routing()` 中一个简单的 DP 打分策略：最小化
  `waiting * 4 + running`。该实现已随旧 per-worker queue 一起删除；当前只剩
  C++ enum、pybind、Config/CLI 和脚本字符串。若等待队列中真有请求，现行 C++
  `_schedule_prefill()` 没有对应分支，会落到 `Unknown routing strategy`，因此它是
  失效的公开入口，应完整删除。
- long-short 仍是一条完整的生产调用链，并非只有脚本残留。它同时存在于
  `Config`、Python/C++ 构造接口、C++ placement、hierarchical admission mirror、
  benchmark CLI 和实验脚本中，必须按同一个提交原子删除。
- `enable_dynamic_sp_size` 的主要语义确实是扩大 legacy 单请求 placement 的重试
  rank 范围，但它还与 `use_new_decode_dynamic_sp_scheduler` 一起充当新 decode batch
  planner 的入口条件。删除该布尔量时，surviving 新 planner 必须改为只由
  `use_new_decode_dynamic_sp_scheduler` 控制；C++ placement 与 hierarchical admission
  mirror 则都固定为只尝试初始 rank 数。bucket 会覆盖 start/end rank，不依赖这个
  布尔量，可以独立保留。
- `legacy_ll` 是 `nanodeploy/worker/sp_backend.py` 中一条相对独立的 lazy import、
  adapter、factory 与 dispatch 分支；可以窄范围删除。删除后默认使用 `hao_basic`。
- `nccl_compact` 由 `bfc9f23 feat: add compact nccl sp backend` 引入，只在 Q
  non-transpose+offsets 路径使用 variable split-size collective，Res/Lse 仍走静态交换。
  它不能被 full CUDA Graph capture，Config 和 microbenchmark 都限制为 eager 或
  piecewise；当前 ablation sweep 虽生成该 variant，`start_bench.sh` 校验却不接受该值。
  删除时要保留静态 `nccl` 共用的 `comm_bs`、mask/stride remap 和 graph batch 逻辑。
- `sp_debug` 由 `263a583 add sp debug utils` 引入，是一条独立于正常 legacy/bucket/fixed
  placement 的调试捷径：临时强制 RoundRobin master，以 `kvcache_block_size` 为 segment，
  给非 master 各分一个 block、其余 token 留给 master。它没有活动 CLI、专属测试或论文
  路径，却同时存在于 C++ allocator 和 hierarchical admission mirror；必须两边原子删除。
- 删除 `legacy_ll` **不等于删除 DLSlime 依赖**。`hao_basic` 自身需要 DLSlime 的
  `AllToAllBuffer`/`KernelImpl`，NanoDeploy 的 RPC 与 KV cache 路径也仍依赖 DLSlime。

## 2. 范围定义

### 2.1 本次删除范围

- 旧 `scheduler_mode`/`--scheduler-mode` 的活动入口、过期脚本参数和误导性
  `decentralized` 实验标签；
- `RoutingStrategy::VLLMLoadBalance` 的 C++ enum、pybind 导出与枚举查找、Config
  合法值、CLI choice、shell 校验/短名/tag 和脚本调用；
- `long_short_sp8` 策略值及其两个专属配置字段：
  `dynamic_sp_long_request_threshold`、`dynamic_sp_long_request_size`；
- long-short 对应的 C++ enum、构造参数、成员、placement 分支与 pybind 参数；
- hierarchical frontend admission mirror 中同语义的阈值分支；
- long-short 专属 benchmark 参数、启动脚本、绘图工具和实验编排；
- `enable_dynamic_sp_size` 的 Config/CLI、Python/C++ 构造参数、成员、pybind 参数、
  legacy rank 扩大重试分支与 hierarchical admission mirror 分支；
- `legacy_ll` 的公开配置值、默认值、CLI choice、lazy import、adapter、factory、
  factory dispatch、专属测试与 benchmark/documentation 列表项；
- `nccl_compact` 的公开配置值、eager/piecewise 特殊校验、compact adapter/factory/
  dispatch、attention/model-runner 集合判断、专属测试、benchmark 与 ablation variant；
- `sp_debug` 的 Config 字段、fixed-SP 冲突判断、Python/C++ 构造参数和成员、pybind
  参数、C++ `can_allocate()` 调试分支及 hierarchical admission mirror 分支。

### 2.2 明确不删除

- `scheduler_arch="legacy_global"`；
- `scheduler_arch="hierarchical"`，以及 `RequestRouter`、`LocalEngine`、
  `LocalScheduler`、`DecodeCoordinator`；
- hierarchical 架构当前需要的 global ingress queue、local command queue、
  local scheduler waiting queue 和相关 queue latency metric；
- `routing_strategy` 字段及其现行值 `RoundRobin`、`LeastBatch`、`LeastCache`；
- hierarchical global DP 路由使用的 `router_policy` 及其
  `round_robin|least_batch|least_batch_v2|least_cache`，以及 `sp_master_selector`；
- surviving SP placement：`legacy`、`bucket`、`fixed_sp_size` 和新的 decode batch
  planner。这里的 `dynamic_sp_size_strategy="legacy"` 是默认 segment placement，
  与本次删除的通信后端 `legacy_ll` 不是同一概念；
- surviving SP all-to-all 后端：`hao_basic`、静态 padding 版本 `nccl`，以及 `nccl`
  需要的 attention `comm_bs`、mask/stride remap、buffer layout 和 CUDA Graph batch
  shape 处理；
- `torch.distributed` 的 NCCL process group 和其他 NCCL collective。不得通过全仓删除
  `nccl` 关键词来清理 SP backend；
- `dlslime` Python/package 依赖、`hao_basic` native symbols、DLSlime RPC、RDMA KV cache；
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

### 3.2 `VLLMLoadBalance` 的历史语义与当前状态

`d7913a7 add vLLM scheduling strategy` 当时增加的调用链为：

```text
Config.routing_strategy == "VLLMLoadBalance"
  -> Python RoutingStrategy enum lookup
  -> 旧 Scheduler::add() / select_dp_worker_for_routing()
  -> 对每个 DP 计算 waiting * 4 + running
  -> 选择分数最小的 DP（同分时选择编号更小的 DP）
```

这里的 “VLLM” 只表示借用了 waiting request 权重大于 running request 的负载分数，
没有调用 vLLM、也没有复用 vLLM scheduler。它依赖旧 decentralized 的 per-worker
waiting/running 状态；`2f94869` 删除这套队列与
`select_dp_worker_for_routing()` 后，没有留下等价的运行时实现。

当前残留为：

- `csrc/nanodeploy/scheduler/sp_state_manager.h` 中的
  `RoutingStrategy::VLLMLoadBalance`；
- `csrc/python/sp_state_manager_binding.cpp` 中的 enum value、
  `__class_getitem__` 与 `__members__`；
- `nanodeploy/config.py` 的 `routing_strategy` Literal；
- examples、两套 benchmark adapter 与 shell 脚本中的 CLI choice、合法值校验、
  display short name 和运行参数。

已确认需要清理的外围文件为：

- examples：`examples/dummy_prefill.py`、`examples/bench_2seq.py`、
  `examples/bench_serving.py`、`examples/bench_serving_overhead.py`；
- benchmark adapter：`scripts/issue003/bench_serving_overhead.py`、
  `scripts/sp_ablation/bench_serving_overhead.py`；
- shell 入口：两份 `start_bench.sh`、`scripts/sp_ablation/sweep_4node_variants.sh`、
  `scripts/run_issue001_deepseek_v3_issue001_bucket.sh`、
  `scripts/decent-e2e/run_decent_e2e_longshort_dp4sp8_ep32.sh`、
  `scripts/issue003/run_issue003_deepseek_rate5_sweep.sh` 和
  `scripts/issue003/run_issue003_deepseek_rate5_sweep_0409_linbinbin.sh`。

现行 `_schedule_prefill()` 只实现 `RoundRobin`、`LeastBatch`、`LeastCache`，其他值会
抛出 `Unknown routing strategy`。它可能在空 waiting queue 时暂时不报错，但这不代表
策略有效；一旦真实请求进入需要路由/placement 的路径就会失败。因此不能只从 CLI
隐藏，应从 C++、binding、Config 与脚本一起删除，并在 Config 层对旧值明确报错。

hierarchical 架构的 global ingress 到 DP 路由由 `router_policy` 控制，策略集合是
`round_robin|least_batch|least_batch_v2|least_cache`。它没有
`VLLMLoadBalance` 选项，也不读取 `routing_strategy`；本次删除不得改动该调用链。

### 3.3 long-short 的当前调用链

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

### 3.4 long-short 的 CLI 与脚本影响面

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

### 3.5 `enable_dynamic_sp_size` 的真实作用域

legacy 单请求 placement 首先按 prompt segment 数计算 `initial_num_ranks`。当前
C++ 与 hierarchical frontend mirror 都使用同一逻辑：

```text
enable_dynamic_sp_size = false
  -> 只尝试 initial_num_ranks

enable_dynamic_sp_size = true
  -> 从 initial_num_ranks 逐级尝试到 attention_sp
```

对应位置：

- C++：`SPStateManager::can_allocate()` 中的 `start_ranks/end_ranks`；
- Python mirror：`AdmissionPlanner._plan_legacy()` 中相同的 rank range；
- 接口转发：`Config -> engine.scheduler -> pybind Scheduler -> C++ Scheduler ->
  SPStateManager`。

`fixed_sp_size`、`bucket`、`long_short_sp8` 随后都会把 `start_ranks=end_ranks`，因此
这个 flag 对 forced-rank 策略没有实际扩大重试作用。清理后：

- 默认 segment placement 只尝试 `initial_num_ranks`；
- bucket 继续直接选择实验测得的目标 SP size；
- fixed baseline 继续使用固定 SP size；
- 不再保留“初始 placement 失败后悄悄增加参与 rank”的单请求 fallback。

该 flag 还有第二个作用：C++ `Scheduler::_schedule_prefill()` 目前要求
`enable_dynamic_sp_size && use_new_decode_dynamic_sp_scheduler` 才进入 latency-aware
decode batch planner。hierarchical frontend 则直接按
`use_new_decode_dynamic_sp_scheduler` 选择 mirror planner。删除前一个 flag 后，应让
C++ 同样只检查 `use_new_decode_dynamic_sp_scheduler`，从而消除双开关和 frontend/local
入口条件不一致的可能性。除非后续单独确认论文也不需要新 batch planner，本次不顺带
删除 `use_new_decode_dynamic_sp_scheduler` 本体。

### 3.6 `legacy_ll` SP backend 的当前调用链

后端选择链为：

```text
Config.sp_backend
  -> ModelRunner.set_sp_context(..., backend=...)
  -> SPContext.backend
  -> create_sp_backend_factory()
       legacy_ll    -> LegacyIntraLLBackendFactory
                    -> LegacyIntraLLBufferAdapter
                    -> dlslime.buffer.intra.AllToAllIntraLLBuffer
       hao_basic    -> HaoBasicBackendFactory
       nccl         -> NcclStaticBackendFactory
       nccl_compact -> NcclCompactBackendFactory
```

`legacy_ll` 自身只占以下生产实现：

- `SPBackend` Literal 中的一个值；
- `_resolve_legacy_buffer_cls()` lazy import；
- `LegacyIntraLLBufferAdapter`；
- `LegacyIntraLLBackendFactory`；
- `create_sp_backend_factory()` 的一个 dispatch 分支。

因此无需重写 `SPContext` 或移除 factory abstraction。结合第 3.7 节的 compact
清理，目标是把 surviving backend 集合收缩为：

```text
hao_basic（默认） | nccl
```

外围影响面包括：

- `nanodeploy/config.py` 当前仍以 `legacy_ll` 为默认值；
- `nanodeploy/worker/sp_context.py` 的 dataclass 和 `set_sp_context()` 也仍以
  `legacy_ll` 为默认值；
- 四份 example/benchmark CLI 和两份 `start_bench.sh` 仍把它列为 choice；
- `tests/test_mla_sp_backend_correctness.py`、`tests/benchmark_mla_sp_backend.py`、
  `tests/README.md` 仍把它列为对比后端；
- `utils_analysis/plot_mla_sp_backend_csv.py` 仍默认以 `legacy_ll` 为 baseline；
- 若干 `docs/` 设计文档把它作为历史 reference implementation。

删除后仍保留跨后端测试与 benchmark：以 `hao_basic` 为 reference/baseline，继续验证
静态 `nccl`。`tests/test_sp_attention_cudagraph.py` 等文件中的
`dist.init_process_group(backend="nccl")` 是 PyTorch 分布式 transport，不是待删的
SP backend choice，必须保留。

配置层建议新增显式 whitelist 校验。Python `Literal` 不做运行时验证；若只从 factory
删掉 `legacy_ll`，旧配置会一直到 Ray worker 初始化 `SPContext` 时才失败。应在
`Config.__post_init__()` 早期拒绝 `legacy_ll` 和其他未知值，让错误发生在 driver
启动阶段。

### 3.7 `nccl_compact` SP backend 的当前调用链

`bfc9f23 feat: add compact nccl sp backend` 在静态 NCCL 后端之外增加了：

```text
Config.sp_backend == "nccl_compact"
  -> create_sp_backend_factory()
  -> NcclCompactBackendFactory
  -> NcclCompactAllToAllBufferAdapter
       Q + mask + offsets
         -> 按 mask/offsets 计算 input/output split sizes
         -> variable-size dist.all_to_all_single()
         -> 将 recv payload 回填到完整 comm_bs 输出
       Res/Lse transpose 或无 offsets
         -> equal-size static dist.all_to_all_single()
```

该实现的目标是只传递 Q 的有效 rows，避免静态 `nccl` 发送 padding 和额外 mask
all-to-all。代价是每轮把 mask/offsets 转到 CPU 生成 Python split-size list，collective
shape 随请求变化，不能被 full CUDA Graph capture。因此当前约束为：

- `Config.__post_init__()` 要求 `enforce_eager=True` 或
  `cuda_graph_mode="piecewise"`；
- correctness/benchmark microbenchmark 发现 compact 时只允许 `--mode eager`；
- end-to-end piecewise 依赖 collective 留在 graph capture 外执行。

生产实现与外围影响面为：

- `nanodeploy/worker/sp_backend.py`：`SPBackend` 值、compact adapter、factory 与
  factory dispatch；
- `nanodeploy/layers/attention.py`：`_uses_nccl_comm_bs()` 把 `nccl` 与
  `nccl_compact` 合并处理；删除后不能删 helper 语义，只需收缩为静态 `nccl` 判断；
- `nanodeploy/worker/model_runner.py`：graph master batch 选择的 backend 集合判断；
- `nanodeploy/config.py`：Literal、full graph 兼容性报错和通用 backend fingerprint；
- examples 与 `scripts/sp_ablation/bench_serving_overhead.py` 的 choices；
- `tests/test_sp_backend.py` 的两个 compact adapter 单元测试，以及 correctness/
  benchmark 的 choice 和 eager-only 护栏；
- `scripts/sp_ablation/sweep_4node_variants.sh` 的 variant 名、tag 和 piecewise 参数。

还存在一个活动脚本不一致：sweep 会生成 `nccl_compact` 并调用
`scripts/sp_ablation/start_bench.sh`，但该脚本的 backend validation 只接受
`legacy_ll|hao_basic|nccl`（help 又宣称 compact 可用）。因此 compact ablation 当前并非
稳定可运行的公开路径。删除时应同时删 sweep variant 和错误 help choice，不做迁移或
alias；保留独立静态 `nccl` variant。

### 3.8 `sp_debug` placement 的当前调用链

`263a583 add sp debug utils` 引入的 legacy/C++ 路径为：

```text
Config.sp_debug
  -> nanodeploy.engine.scheduler.Scheduler
  -> pybind Scheduler(..., sp_debug)
  -> C++ Scheduler::sp_debug_
  -> SPStateManager(..., sp_debug)
  -> SPStateManager::can_allocate() 的首个特殊分支
```

当 `attention_sp > 1 && sp_debug` 时，它完全绕过正常 legacy/bucket/fixed placement：

1. 把调试 segment 固定为 `kvcache_block_size`；
2. 目标 SP 数为 `min(ceil(num_tokens / block_size), attention_sp)`；
3. 临时把 C++ master selector 改成 RoundRobin；
4. 按 free blocks 选择 non-master，每个 non-master 最多放一个完整 block；
5. 所有剩余 token 都放到 master，再执行 receiver、reservation 和 KV allocation
   capacity 检查。

hierarchical 架构在 `1136a72 Eliminate speculative decentralized admission retries`
中复制了同语义的 frontend mirror：

```text
Config.sp_debug
  -> AdmissionPlannerConfig.from_config()
  -> AdmissionPlanner._plan_legacy() 特殊分支
```

这条 mirror 在进入分支前已经按配置的 `sp_master_selector` 选择 master，并不像 C++
实现那样无条件临时切到 RoundRobin；默认 selector 又是 `LeastBatch`。因此
hierarchical `sp_debug` 存在 frontend reservation 与 LocalScheduler master 选择不一致
的风险。仓库中没有 `sp_debug` CLI、命名测试或 benchmark 覆盖，也没有发现论文实验
脚本使用它。

删除必须覆盖 Config、Python adapter、两个 C++ 类、两个 pybind 构造接口及 admission
mirror，不能只删除 C++ if-block。`fixed_sp_size` 冲突条件里的 `sp_debug` 同步删除。
由于 `LLMEngine` 会过滤未知 kwargs，还应把 `sp_debug` 加入集中式 removed-option
fail-fast 护栏，防止旧调用静默退化为默认 legacy placement。

## 4. 计划中的目标状态

清理完成后，外部可见的 SP 策略为：

- `dynamic_sp_size_strategy="legacy"`；
- `dynamic_sp_size_strategy="bucket"`；
- 与上述 dynamic strategy 互斥的 `fixed_sp_size > 0` baseline。

`enable_dynamic_sp_size` 不再存在。默认 legacy segment placement 不做扩大 rank 数的
fallback；新 decode batch planner 若启用，只由
`use_new_decode_dynamic_sp_scheduler` 一个开关控制。

外部可见的 SP all-to-all 后端为：

- `hao_basic`（默认）；
- `nccl`（静态 padding 版本）。

`legacy_ll` 与 `nccl_compact` 不再是合法配置。活动代码中不再 import 旧 DLSlime
buffer class，也不再维护 variable split-size NCCL adapter/eager-only 兼容分支。

`sp_debug` 字段、构造 ABI 与特殊 placement 不再存在。legacy、bucket、fixed baseline
和新 decode batch planner 继续使用各自的正式 placement 语义；删除 debug 分支不改变
`sp_master_selector` 的三个现行值。

调度架构只使用：

- `scheduler_arch="legacy_global"`；
- `scheduler_arch="hierarchical"`。

legacy/local C++ scheduler 的 `routing_strategy` 只保留：

- `RoundRobin`；
- `LeastBatch`；
- `LeastCache`。

`VLLMLoadBalance` 不再是合法配置，也不再出现在 C++/pybind 枚举或 CLI 中。hierarchical
的 global DP routing 继续由独立的 `router_policy` 控制，其现行策略和值不变。

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
3. `config.py` 与 `model_runner.py` 都和本次清理有重叠，实施时只做目标字段/hunk 的
   精确修改，不覆盖用户现有 bucket policy 与 decode mask logging 改动；
   `local_executor.py`、`ray_executor.py` 不应被本任务触碰。
4. 先运行现有 CPU focused tests，记录任何基线失败，避免把环境或用户改动造成的
   失败误归因于清理。

### 阶段 1：清理旧 decentralized 的外围残留与失效路由值

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
6. 原子删除 `VLLMLoadBalance` 公开入口：
   - 从 `csrc/nanodeploy/scheduler/sp_state_manager.h` 删除 enum value；
   - 从 `csrc/python/sp_state_manager_binding.cpp` 删除 pybind value，并同步收缩
     `__class_getitem__` 与 `__members__`；
   - 从 `Config.routing_strategy` Literal 删除旧值，并增加显式 whitelist 校验，确保
     `VLLMLoadBalance` 和其他未知值在 driver 端得到清晰 ValueError，而不是稍后触发
     C++ `Unknown routing strategy`；
   - examples、benchmark adapter、`start_bench.sh`、sweep/run 脚本中的 choice、case、
     display short name、tag 和参数转发全部删除；
   - 保留 `RoundRobin|LeastBatch|LeastCache`，不改 hierarchical `router_policy`。
7. 删除值后不保留同名 alias。与已删除字段不同，`routing_strategy` 字段本身仍存在，
   Config whitelist 足以提供 fail-fast；无需在 `LLMEngine` 的 removed-option kwargs
   护栏里增加特殊分支。

### 阶段 2：原子删除 long-short、`enable_dynamic_sp_size` 与 `sp_debug`

1. `nanodeploy/config.py`
   - `dynamic_sp_size_strategy` 的合法值收缩为 `legacy|bucket`；
   - 删除 `dynamic_sp_long_request_threshold`、
     `dynamic_sp_long_request_size`；
   - 删除 `enable_dynamic_sp_size`；
   - 删除 `sp_debug`；
   - 删除 size=0 归一化及范围校验；
   - 更新 fixed/bucket 冲突报错和注释，移除 fixed 与 `sp_debug` 的冲突项；
   - 从 fingerprint 删除 `enabled` 位，保留 strategy/bucket/fixed/new planner 信息。
2. `nanodeploy/engine/llm_engine.py`
   - 在集中式 removed-option 检查中加入两个已删除 long-short 字段；
   - 将 `enable_dynamic_sp_size` 加入 removed-option 检查；
   - 将 `sp_debug` 加入 removed-option 检查，避免 LLM kwargs 静默忽略；
   - 对 `dynamic_sp_size_strategy="long_short_sp8"` 保持明确 ValueError；
   - 不在本任务中顺带改成“拒绝所有未知 kwargs”，以免扩大公共 API 兼容面。
3. `nanodeploy/engine/scheduler.py`
   - 从 C++ `Scheduler` 构造调用中删除两个 long-short 参数和
     `enable_dynamic_sp_size`、`sp_debug` 参数。
4. `nanodeploy/router/admission_planner.py`
   - 从 `AdmissionPlannerConfig` 及 `from_config()` 删除 long-short 两个字段和
     `enable_dynamic_sp_size`、`sp_debug`；
   - 删除 `_plan_legacy()` 的 `sp_debug` block-size placement 分支；
   - 删除 `_plan_legacy()` 的 long-short forced-rank 分支；
   - legacy placement 固定 `start_ranks=end_ranks=initial_ranks`；
   - 保留 fixed/bucket 的 `recompute_segments` 行为。
5. `csrc/nanodeploy/scheduler/scheduler.{h,cpp}`
   - 删除 long-short、`enable_dynamic_sp_size` 与 `sp_debug` 的构造参数、成员、转发和
     日志字段；
   - latency-aware decode planner 的入口改为只检查
     `use_new_decode_dynamic_sp_scheduler`；
   - 不改 `waiting`/`waiting_migration`、`worker_state` 或 surviving scheduler path。
6. `csrc/nanodeploy/scheduler/sp_state_manager.{h,cpp}`
   - 删除 `DynamicSPSizeStrategy::LongShortSP8`；
   - 删除 threshold/size 构造参数与成员；
   - 删除 `enable_dynamic_sp_size` 构造参数与成员；
   - 删除 `sp_debug` 构造参数、成员、fixed-SP 冲突条件及 `can_allocate()` 的完整调试
     placement/capacity 分支；
   - 删除字符串解析、name/logging 和 `can_allocate()` 阈值分支；
   - legacy `can_allocate()` 只尝试 `initial_num_ranks`；
   - 保留 `Legacy`/`Bucket`，并让未知字符串继续 fail fast。
7. 两个 pybind 文件同步删除 long-short、`enable_dynamic_sp_size`、`sp_debug` 参数、
   默认值与构造转发。该步骤会改变 extension 构造
   ABI，必须与 C++/Python adapter 同一提交完成，并在测试前重新 editable install。

### 阶段 3：清理 CLI、实验脚本和文档入口

1. Python CLI/adapter：
   - 三份 `bench_serving_overhead.py` 删除 `--long-request-sp-threshold`、
     `--long-request-sp-size`；
   - `examples/bench_2seq.py`、`examples/bench_serving.py` 和三份
     `bench_serving_overhead.py` 删除 `--enable-dynamic-sp-size` 及 LLM 参数转发；
   - `examples/dummy_prefill.py` 删除仅用于二选一 long-short 的
     `--sp-size-policy`、threshold 和隐式 `enable_dynamic_sp_size` 转发，默认走
     surviving legacy placement；
   - strategy choices 改为 `legacy|bucket`；
   - 将 `--dynamic-sp-bucket-preset` 作为正式参数透传，保证 surviving bucket 无需
     monkeypatch 才能运行。
2. 两份 `start_bench.sh`：
   - 删除 long-request 与 `enable_dynamic_sp_size` 的默认变量、help、parser、日志、
     tag 和 Python 参数转发；
   - 增加/保留 bucket preset 的正式转发；
   - 对未知 `--*` 参数直接报错，不再把拼错的 option 当作 rate。这也能防止旧
     `--scheduler-mode` 再次静默污染 rate 列表。
3. 保留并清理以下通用脚本：
   - `scripts/sp_ablation/sweep_4node_variants.sh`
   - `scripts/issue003/run_issue003_deepseek_rate5_sweep.sh`
   - `scripts/issue003/run_issue003_deepseek_rate5_sweep_0409_linbinbin.sh`
   - `scripts/run_issue003_deepseek_2node_dp16_dp2sp8_chain.sh`：删除 threshold stage，
     保留 DP16 与 DP2SP8 legacy 对比，并改成明确的 legacy 命名；
   - `scripts/run_issue001_deepseek_v3_issue001_bucket.sh`：删除 long-request no-op
     参数，并改用正式 bucket preset CLI，删除 argparse/LLM monkeypatch；
   - `scripts/run_kimi_dp16_ep16_gemm_debug.sh`：删除 legacy 模式下无效的 threshold
     参数。
4. 对所有 surviving script 做 `enable_dynamic_sp_size` 残留扫描，删除对应环境变量、
   positional function 参数、条件拼接、日志和下游转发；不能只清理两份
   `start_bench.sh`。
5. 删除第 3.4 节列出的 long-short 专属脚本/工具。
6. 更新 `scripts/README.md`，删除已移除绘图工具与实验脚本的活动说明；不修改
   dated `docs-dev/` 历史记录。

### 阶段 4：删除 `legacy_ll` 与 `nccl_compact`，保留两种 SP backend

1. `nanodeploy/config.py`
   - `sp_backend` 类型收缩为 `Literal["hao_basic", "nccl"]`；
   - 默认值从 `legacy_ll` 改为 `hao_basic`；
   - 在 `__post_init__()` 增加显式 whitelist 校验，使 `legacy_ll`、`nccl_compact` 和
     其他未知值在 driver 端 fail fast；
   - 删除 compact 专属的 eager/piecewise 兼容性校验；保留通用 backend fingerprint。
2. `nanodeploy/worker/sp_backend.py`
   - 从 `SPBackend` 删除 `legacy_ll` 与 `nccl_compact`；
   - 删除 `_resolve_legacy_buffer_cls()`、`LegacyIntraLLBufferAdapter`、
     `LegacyIntraLLBackendFactory` 及对应 dispatch；
   - 删除 `NcclCompactAllToAllBufferAdapter`、`NcclCompactBackendFactory`、split-size
     cache/pack helpers 与对应 dispatch；
   - 保留 protocol/factory abstraction、`HaoAllToAllBufferAdapter`、
     `NcclStaticAllToAllBufferAdapter`；
   - 保留 `torch.distributed as dist`，静态 `nccl` backend 仍使用
     `dist.all_to_all_single()`。
3. `nanodeploy/worker/sp_context.py`
   - `SPContext.backend` 与 `set_sp_context(..., backend=...)` 的默认值改为
     `hao_basic`；
   - 继续通过 `create_sp_backend_factory()` 初始化 q/res/lse buffer，不移除后端选择。
4. attention 与 CUDA Graph 相邻分支：
   - `nanodeploy/layers/attention.py` 的 `_uses_nccl_comm_bs()` 收缩为只判断 `nccl`；
   - 保留静态 NCCL 所需的 `comm_bs`、mask narrowing、stride remap、buffer zeroing 和
     q/res/lse shape 逻辑；
   - `nanodeploy/worker/model_runner.py` 的 graph master batch backend 集合收缩为
     `nccl`，不触碰用户现有 decode A2A mask logging 改动；
   - 保留 `_build_sp_graph_attn_bs_candidates()` 中静态 `nccl` 的 graph shape 上限。
5. examples 与 benchmark adapter：
   - `examples/bench_2seq.py`、`examples/bench_serving.py`、
     `examples/bench_serving_overhead.py`、`examples/dummy_prefill.py`；
   - `scripts/issue003/bench_serving_overhead.py`、
     `scripts/sp_ablation/bench_serving_overhead.py`；
   - choices 统一改为 `hao_basic|nccl`，默认统一为 `hao_basic`。
6. 启动与实验脚本：
   - 两份 `start_bench.sh` 的 help/validation 统一收缩为 `hao_basic|nccl`，顺便消除
     sp_ablation help 宣称 compact 可用而 validation 拒绝它的不一致；
   - 保留有价值脚本中的 `--sp-backend`，因为 hao/static-NCCL 对比仍是合法实验；
   - long-short 阶段要删除的专属脚本不单独迁移；
   - `scripts/sp_ablation/sweep_4node_variants.sh` 删除 `nccl_compact` variant、policy key、
     tag、piecewise/gpu-util override，保留 `nccl` variant。
7. 测试与性能工具：
   - `tests/test_sp_backend.py` 删除 legacy 和 compact adapter/factory 测试，保留 hao、
     static NCCL 与 factory/context 覆盖，并增加两个旧值的拒绝测试；
   - `tests/test_mla_sp_backend_correctness.py` 与
     `tests/benchmark_mla_sp_backend.py` 的 choices 收缩为 `hao_basic|nccl`，删除 compact
     eager-only 护栏，继续以 `hao_basic` 为 reference 对比静态 `nccl`；
   - `utils_analysis/plot_mla_sp_backend_csv.py` 默认 baseline 改为 `hao_basic`、
     candidate 改为 `nccl`；
   - 更新代码 docstring 与 `tests/README.md` 的 backend 列表和命令。
8. 文档策略：
   - 更新面向当前用户的 README、CLI 文档，不能继续宣称两个旧 backend 可选；
   - `docs-dev/` dated notes 不改；
   - `docs/` 下以 legacy/compact implementation 作技术对照的旧设计文档不做关键词
     硬删除，应增加“历史设计、当前 backend 已移除”的页首说明，或在 review 后迁入
     `docs-dev/`。Git 历史继续保留。
9. 不删除 `dlslime` 依赖，不修改外部 DLSlime 源码；NanoDeploy 只删除对旧
   `AllToAllIntraLLBuffer` Python class 的 import。不得删除 PyTorch NCCL process group
   或静态 `nccl` backend。

### 阶段 5：测试与验证

1. 增补/调整 CPU tests：
   - `routing_strategy` 接受 `RoundRobin|LeastBatch|LeastCache`，明确拒绝
     `VLLMLoadBalance` 与其他未知值；
   - pybind `RoutingStrategy.__members__` 只包含三个 surviving 值；
   - 三个 surviving routing strategy 的现有调度测试继续通过；
   - `long_short_sp8` 被配置层明确拒绝；
   - 两个 removed long-short kwargs、`enable_dynamic_sp_size` 与 `sp_debug` 不会被
     `LLMEngine` 静默忽略；
   - 旧 `scheduler_mode` 仍明确报错；
   - `legacy` 与 `bucket` 的 frontend admission mirror 和 C++ placement 保持一致；
   - 默认 legacy placement 不再扩大 rank 重试范围；
   - 新 decode batch planner 仅由其专属 flag 启用；
   - fixed SP 与 bucket/legacy 的互斥校验不回归；
   - 删除 `sp_debug` 后，legacy/bucket/fixed 的 frontend mirror 与 C++ placement
     等价测试仍覆盖不同 master selector、SP size 和 block capacity 边界；
   - `sp_backend` 默认是 `hao_basic`，只接受 `hao_basic|nccl`，并在 Config 层拒绝
     `legacy_ll`、`nccl_compact` 与未知值；
   - hao/static-NCCL adapter 单元测试继续通过。
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
   python -m pytest tests/test_sp_backend.py
   ```

4. 运行 shell 静态检查：

   ```bash
   bash -n <每个保留且修改过的脚本>
   ```

   并对两个 `start_bench.sh` 做不启动 benchmark 的参数解析测试，覆盖
   `legacy_global`、`hierarchical+bucket`、`hao_basic|nccl` 接受，以及
   `legacy_ll|nccl_compact` 和未知 option 拒绝。
5. 运行残留扫描。活动源码/脚本应无以下符号，removed-option 护栏与负向测试除外：
   - `SchedulerMode`
   - `_schedule_decentralized`
   - `--scheduler-mode`
   - `VLLMLoadBalance`
   - `long_short` / `longshort` / `LongShortSP8`
   - `dynamic_sp_long_request_*`
   - `--long-request-sp-*`
   - `enable_dynamic_sp_size` / `--enable-dynamic-sp-size`
   - `sp_debug`
   - `LegacyIntraLLBufferAdapter` / `LegacyIntraLLBackendFactory`
   - `NcclCompactAllToAllBufferAdapter` / `NcclCompactBackendFactory`
   - 活动源码、CLI、测试 choices 中的 `legacy_ll` / `nccl_compact`
6. CPU 与构建通过后再考虑 GPU smoke。GPU 不是本次纯策略删除的第一验收门槛；如做，
   必须先获得 elevated permission，并在 driver 环境设置 `SLIME_QP_NUM=4`。建议只跑
   调度/placement 的两个 surviving path：一个 `legacy_global+legacy`，一个
   `hierarchical+bucket`；backend 至少覆盖默认 `hao_basic`，并按保留承诺分别做
   静态 `nccl` 的匹配 full/piecewise smoke，记录 GPU 数与 topology。

## 6. 提交拆分建议

为便于 review 与回滚，建议拆成七个窄提交：

1. `fix: remove stale decentralized scheduler script options`
   - 只做 `scheduler_mode -> scheduler_arch` 脚本迁移和命名清理；
2. `refactor: remove stale VLLM load-balance route`
   - C++/pybind enum、Config 校验、CLI/脚本 choices 和 routing tests；
3. `refactor: remove sp_debug placement path`
   - Config/removed-option 护栏、Python mirror、C++、bindings、tests；
4. `refactor: remove unused dynamic SP placement paths`
   - Config、Python mirror、C++、bindings、tests；
5. `chore: remove obsolete long-short experiment tooling`
   - CLI、launch/plot scripts、README；
6. `refactor: remove legacy_ll SP backend`
   - legacy adapter/factory、Config 默认值、CLI choices、tests/docs；
7. `refactor: remove nccl_compact SP backend`
   - compact adapter/factory、共享 NCCL 条件收缩、ablation、tests/docs。

第 3、4 个提交都必须各自原子覆盖 Python mirror、C++ implementation 和 bindings；
不能产生 frontend/local placement 不一致或 extension 构造 ABI 不一致的中间提交。
第 7 个提交不能删除静态 `nccl` 共用的 attention/graph shape 逻辑。

## 7. 风险与控制

| 风险 | 后果 | 控制方式 |
|---|---|---|
| 把当前 hierarchical queue 当成旧 per-worker queue 删除 | 当前分层运行时不可用 | 按第 2.2 节白名单保留，只删除历史 `scheduler_mode` 语义 |
| 把 `VLLMLoadBalance` 当成 hierarchical router policy | 误删当前 global DP 路由或错误迁移配置 | 只删除精确 enum/配置值；明确保留整个 `router_policy` 调用链 |
| 只隐藏 `VLLMLoadBalance` CLI、保留 C++/pybind 值 | 失效 API 仍可被 Python 配置触发并在请求到达后报错 | 同一提交删除 enum、binding、Config/CLI，并增加旧值拒绝测试 |
| C++ 构造参数与 pybind/Python adapter 不同步 | 编译失败或运行时构造 TypeError | 同一提交改六个 C++/binding 文件与 adapter，随后 editable install |
| 只删 C++ 或只删 admission mirror | frontend reservation 与本地 placement 失配 | 两边原子删除，并跑 mirror 等价测试 |
| 只删 `sp_debug` 的 C++ if-block | hierarchical mirror 仍生成已不存在的 block-size placement | 同一提交删除 Config、mirror、C++、bindings，并跑等价测试 |
| 删除字段后 kwargs 被静默过滤 | 用户以为 long-short/`sp_debug` 仍生效，实际跑默认策略 | 保留集中式 removed-option fail-fast 护栏 |
| 删除 `enable_dynamic_sp_size` 后新 batch planner 永远不再进入 | surviving planner 变成死代码 | C++ 入口改为只检查 `use_new_decode_dynamic_sp_scheduler`，并增加正负测试 |
| 将 long-short 脚本直接改成 bucket/legacy | 历史脚本名与实验语义不一致，结果不可比较 | 专属脚本直接删除；新策略另建明确命名脚本 |
| bucket CLI 仍依赖 monkeypatch | surviving policy 的公开入口脆弱 | 把 bucket preset 变成正式 benchmark/start_bench 参数 |
| 把 `dynamic_sp_size_strategy="legacy"`、`legacy_global` 与 `legacy_ll` 混为一谈 | 误删仍需保留的 placement 或中心化调度 | 精确按完整符号清理，只删除通信后端 `legacy_ll` |
| 删除 compact 时顺带删掉共享 NCCL shape 逻辑 | 静态 `nccl` 的 mask、stride 或 graph batch 失配 | 把 backend 集合判断收缩为 `nccl`，保留 `comm_bs`/remap/graph 分支及现有 static tests |
| 全仓删除 `nccl` 关键词 | 破坏 surviving 静态 SP backend 或 PyTorch process group | 只精确删除 `nccl_compact` 类/choice；明确保留 `nccl` 和 NCCL transport |
| 误删 DLSlime 依赖 | `hao_basic`、RPC、KV cache 无法启动 | 只删旧 buffer class import，不改依赖声明或外部库 |
| 默认 backend 从 `legacy_ll` 改为 `hao_basic` 后旧 DLSlime build 不兼容 | worker 初始化失败 | driver 端校验配置，启动时保留 hao native-symbol fail-fast，并在支持的 DLSlime build 上做 GPU smoke |
| 只删 factory 分支、不做 Config 校验 | 两个旧值延迟到 Ray actor 初始化才失败 | 在 `Config.__post_init__()` 明确 whitelist 校验 |
| 覆盖当前未提交的 `config.py`/`model_runner.py` 改动 | 丢失用户 bucket policy 或 mask logging 工作 | 精确 hunk 修改，提交前逐文件 diff，绝不 restore/reset 用户文件 |

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
6. **删除 `enable_dynamic_sp_size`，保留新 batch planner（当前计划）**：新 planner 改为
   只由 `use_new_decode_dynamic_sp_scheduler` 控制；若论文也未使用它，应另行确认后把
   planner 本体纳入删除范围。
7. **删除 `legacy_ll`（已确认）**：默认改为 `hao_basic`，不移除 DLSlime 依赖。
8. **删除 `nccl_compact`（已确认）**：保留 `hao_basic`、静态 `nccl`、backend switch、
   两后端对比测试和 benchmark。
9. **删除 `sp_debug`（已确认）**：同步删除 C++ placement 与 hierarchical mirror；不把
   该调试语义迁移为新的 bucket/fixed 策略。
10. **删除 `VLLMLoadBalance`（已确认）**：它是旧 decentralized 的失效枚举，不是
   hierarchical `router_policy`；保留 `RoundRobin`、`LeastBatch`、`LeastCache`。
11. **旧 SP backend 设计文档的归档方式**：推荐保留技术历史但加醒目标记；若开源活动
   `docs/` 只允许现行说明，则将这些文档迁入 dated `docs-dev/`，不直接抹除内容。

以上决策确认后再进入实现，避免把“删除旧 decentralized、`VLLMLoadBalance`、
`sp_debug`、`legacy_ll`、`nccl_compact`”误扩展为删除 `legacy_global`、默认 legacy
segment placement、当前 hierarchical control plane/`router_policy`、静态 `nccl` 或
PyTorch NCCL transport。
