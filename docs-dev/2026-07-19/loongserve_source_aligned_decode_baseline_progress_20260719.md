# LoongServe source-aligned Decode baseline 任务罗盘

更新时间：2026-07-19 13:58:27 UTC

## 任务目标

按 `docs-dev/2026-07-18/loongserve_source_aligned_decode_baseline_plan_20260718.md`
实现 NanoDeploy 的 LoongServe source-aligned Decode-only baseline。先完成单机 CPU
构建与测试；任何 GPU 脚本运行前必须申请提权。只修改 NanoDeploy，不修改外部依赖。

## 当前实施状态

### 已完成并通过阶段验证

- Sequence ABI：保留旧 `SequenceStatus` ordinal 0--3，新增
  `PAUSED_OFFLOAD=4`；新增持久、write-once `assigned_dp`；raw/pickle/binding
  round-trip；Python assignment 后禁止修改 `seq_id`。
- Typed scheduler ABI：`LSAddError`/`LSAddResult`、`LSAdmissionRecord`、
  `LSFatalCode`/fatal latch，以及 action-specific real/running ID 字段已加入 C++/pybind。
- Config/engine/metrics：严格 LS profiles、metric ticket 两阶段 ingress、fatal
  propagation、ADMISSION/DECODE/KV_CONSOLIDATION 字段矩阵、bootstrap completion 和
  real-ID metrics 路径已实现。
- Block allocator foundation：move-only RAII `PreparedBlockMutation`，支持 prepared
  allocate/release、幂等 noexcept commit/abort、析构自动回滚及旧 API 穿透保护。
- Scheduler 主路径已切到 pool-local list、arrival RR、FIFO + bounded OOE、
  membership-before-stable-sort、每 pool 一个 ephemeral admission batch、pool-wide
  future-KV、packed placement、C++ dummy bootstrap、entry-snapshot real Decode IDs、
  memory donor list/compute floor/low-KV/OFFLOAD 基础路径。
- 已删除 scheduler 内旧 persistent `PendingDecodeBatch` 类型、队列和 batch-owner
  map；兼容 getter 固定返回空。LS `preempt()` 已改为 preserve-progress OFFLOAD。
- LS ingress 已预创建 seen/arrival/list/context 节点，再消费 RR 并发布，避免
  accepted enqueue 后的普通容器分配故障。
- scheduler admission 已切换到 `PreparedLSInitialBatch`：placement/blocks、完整
  group/map/list/deque/telemetry shadows 和 metric/vector capacity 都在 prepare 构造；
  bootstrap-finished reservation 在 survivor commit 前 abort；commit 后不再调用 legacy
  allocate/deallocate/may_append/add-running-token。
- admission 与 step-entry Decode 同步 telemetry 已冻结 planned request identity，修复
  capacity-append 后遍历扩大 group 导致 `sequence_master_ranks` 越界的问题。
- scheduler mandatory OFFLOAD 与显式 LS `preempt()` 已改用 `PreparedLSRelease`，保留
  token/metric/arrival/assigned-DP 进度并预建 scheduler publication shadows。
- `PreparedLSIterationMasterPlan` 已完成：支持满 rank 的 pending-only block rebalance、
  多 group/DP combined request、move-only RAII、规范化 reservation fingerprint；scheduler
  已删除逐 group legacy commit，改为所有 DP prepare/validate 后一次 no-throw publish。
- `schedule()` 已加入 publication-stage function-try boundary；admission、mandatory shadow、
  OFFLOAD、iteration/preempt commit 会标记 publication，之后任何普通异常统一 latch
  `POST_PUBLICATION_INVARIANT`。planner internal/validation failure不再泛化为 OFFLOAD。
- 已加入 engine-lifetime `pool_resource_epoch`，每 pool 每次 resource publication至多递增一次；
  按学术 demo 范围移除了 iteration fingerprint/exact-attempt/duplicate telemetry与 Python固定 ABI
  field-matrix热扫，只保留 action互斥和真正的动态 transaction关系。
- low-KV transaction 已收口：destination/source使用 prepared mutation；scheduler shadow在
  RESERVED时预建；首次 RPC 前必须 mark `DISPATCHED`；dispatch 后 copy/commit/stale失败永久
  latch `KV_CONSOLIDATION_FAILED` 并保留 guard，禁止 abort；成功才递增 epoch。destination
  顺序为 exact capacity降序、used KV升序、rank升序。
- empty-system capacity cache 已从仅 DP0 改为 `[dp][sp]`；inactive MIGRATE/SWAP 空判定与
  ACTIVE release 后 `master=-1` 后置条件已分离，避免自动 OFFLOAD 误拒全部 victim。
- feature-off compatibility 已恢复 Python `Scheduler.add()` 返回 `None`，LLMEngine empty-step
  早拒仅用于 LS；新增对应 CPU 回归测试。pybind fatal exception现在携带 `fatal_code`。

### 已跑的 CPU 验证

- 修改前基线：相关四个测试文件共 `80 passed`。
- Sequence/config/metrics/engine/allocator targeted suite：最近一次由子任务完成
  C++ 重装后 `81 passed in 6.95s`。
- Prepared allocator 独立 harness：`7 passed`，ASan/UBSan 同样通过（容器关闭
  LeakSanitizer）；SP initial/release transaction harness 与其联合为 `11 passed`。
- C++ scheduler 冒烟：`max_tokens=1` 在 ADMISSION bootstrap 当步完成且不调用
  model；`max_tokens=2` 下一步只执行一次真实 Decode。
- `tests/test_ls_decode_scheduler.py` 已重写为 13 个 source-aligned 契约场景，包括
  simultaneous admission+entry Decode、capacity append identity 与 prepare rollback。
- 最近一次 C++ 重装后的单机联合 suite：scheduler、allocator/SP transaction、Sequence
  ABI、config、metrics、engine action、planner、benchmark profile 共
  `126 passed in 15.50s`。
- prepared iteration/SP/allocator 联合 harness：`16 passed`；prepared iteration
  ASan/UBSan：`5 passed`。
- low-KV prepared transaction 联合 harness：`12 passed`；新增 low-KV ASan/UBSan 三场景
  通过；Python coordinator dispatch-failure 边界 `2 passed`；scheduler source syntax通过。
- 最新 Python validator/feature-off targeted subset：`6 passed`；Python `py_compile`通过。

## 当前正在进行

- 13:58 UTC 单机 GPU检查点：经提权确认本机8×H200空闲，已实际运行三项8-GPU测试。
  `tests/ls_kv_scale_down_nccl_preflight.py --sp-size 8 --transport-only` 使用8进程NCCL 2.27.3，
  rank7→rank0 KV range copy与逐元素内容校验通过、exit 0；`test_mla_sp_backend_correctness.py`
  在world_size=8/max_num_seqs=8/num_requests=8下分别跑eager和CUDA Graph，`hao_basic`与`nccl`
  的Q/Res/Lse全部一致、两次均exit 0。完整scheduler+consolidation preflight仍有fixture gap：旧脚本
  用单个8-token request却断言packed admission立即占满SP8，因此在spawn NCCL前失败；当前结果只
  声明transport与SP backend GPU正确性，不虚称完成完整metadata transaction E2E。GPU测试未使用Ray。

- 13:38 UTC 最终提交检查点：主实现、tests、fixtures与当日罗盘已提交为
  `b8e0b7d feat: align decode scheduler with LoongServe baseline`。提交后工作树仅有明确排除的用户
  `AGENTS.md`、`nanodeploy/engine/ray_executor.py` 与历史实验产物。学术 demo 单机 CPU目标已经完成；
  原计划的production-grade exact fingerprint/attempt/dedupe、waiting age/fatal hash、stable required
  Decode fault hooks按用户范围决定不实现，不应再恢复到热路径。未运行 GPU。

- 13:32 UTC 单机 CPU 验收完成检查点：按仓库规则设置代理、显式禁用 GPU并提权执行 editable
  reinstall，C++全量重编译/链接成功。安装后 task-scope 联合套件在 `PYTHONMALLOC=debug` 下
  `221 passed in 34.49s`；Sequence proxy 通过；scheduler单文件`17 passed`，low-KV/P2P
  `43 passed`且进程exit 0；source-built transaction harness `20 passed`。新增 KV plan 外部引用跨
  Scheduler析构case后，普通CPU `4 passed`、ASan/UBSan `4 passed`。三条只读终审均确认 transaction、
  Sequence ABI、low-KV 无剩余 correctness blocker；未运行 GPU。
- 联合回归修复了五个实质问题：`PreparedLSRelease`空 ACTIVE context 显式设master=-1；
  `schedule()`在entry guards前清per-call publication flag；typed `LSSchedulerFatalError`继承Python
  `RuntimeError`；KV consolidation plan持有source+全部retained `BlockManager`保证prepared mutation
  析构安全；容量测试补算rank-local cadence dummy block。当前仅剩工作树终审、更新罗盘、显式排除
  `AGENTS.md`/`ray_executor.py`/历史实验产物后分组提交。

- 13:15 UTC 自动压缩恢复检查点：已先完整重读本罗盘和 `Progress.md`。学术 demo 范围裁剪已
  落地：删除全部 `ExactPlanKey`/iteration reservation fingerprint/attempt/duplicate telemetry，以及
  binding、Python logger、fake/test 引用；删除跨 admission-overlay 的 prepared iteration cache，保留
  rollback 前 `pending_dp.prepared.reset()` 触发 RAII abort 和 outer owner 销毁后重建。scheduler 又
  删除 schedule 入口 stable canonical-owner 全量 census 与 publication 后重复全量复核，但保留最终
  每-DP composition validate、global publication barrier、pool-local rollback、RAII/policy/typed fatal。
  子任务对该 C++ 裁剪已通过 syntax、diff-check、source-built transaction harness `17 passed`；未重装、
  未运行 GPU。主线程已把 Python `_validate_ls_action_fields` 收缩为最小 action guard，gauge 改为直接
  统计，删除 KV postcommit epoch 全量 census及相应伪故障测试。下一步先重跑 fake-engine subset并
  统一做 rg/diff-check/py_compile/C++ syntax/source harness；然后按规则提权执行非 GPU editable install，
  再跑单机 CPU suite。

- 12:51 UTC scope收缩检查点：用户明确这是学术demo，不要求生产级审计系统。今天不再补combined
  admission+iteration fingerprint、initial/combined adapter attempt、pre-solve logical dedupe、额外
  step-entry/dummy telemetry、waiting oldest age、统一schedule/iteration事件、fatal state hash或两套
  stable Decode fault hooks。纯观测字段现已全部撤回；保留真正影响正确性的
  `CachedPreparedIteration`彻底删除、TENTATIVE rollback前
  `pending_dp.prepared.reset()`触发RAII abort/释放blocks、outer owner销毁后重建。核心验收聚焦
  LoongServe policy、pool-local transaction、共享collective publication barrier、Sequence/Ray ABI和关键
  CPU场景。

- 12:44 UTC 状态检查点：已向用户明确今天以“单机CPU通过、代码可提交”为完成线；当前功能实现
  约80%，但latest C++尚未统一重装，验收完成度约65%。用户要求移除固定ABI运行时反射后，主线程
  已将LS ingress的`precheck_add_identity`/`LSAddResult`/arrival getter、fatal code/latch、KV plan
  日志/state及epoch类型census改为直接契约；fake对象同步使用真实enum/method，engine subset重新为
  `22 passed, 2 deselected`。`kv_consolidation.py`中固定dataclass/pybind/executor/config字段反射正在
  清理，动态plan→move、worker DP/SP/TP/role/chunk/bytes守恒与state transition保留。
- Issue-1%六个authoritative窗口已新增`tests/fixtures/ls_issue001_windows.json`及独立参数化CPU测试：
  固定source CSV SHA256、完整arrival/request/prompt/output/type、DP4 snapshots，验证RR、pool FIFO、
  uniqueness/no-loss/no-reroute，并用真实Scheduler做4DP×1SP routing-only覆盖。测试没有虚称覆盖SP8
  capacity行为；long blocker、short OOE bypass/limit/frontier仍是明确runtime gap。
- stable required Decode fault audit已给出两个精确插入点：stable iteration adapter entry前，以及全部
  pool component完成后的最终global validate/publication barrier；仍待实现hook与typed fatal/global-
  zero-publication状态快照测试。exact agent仍在完成combined fingerprint、overlay-safe key/cache、
  initial/iteration真实attempt计数和pre-solve dedupe，完成后再合并上述hook以避免并发冲突。

- 12:31 UTC 自动压缩恢复检查点：主线程已先完整重读本罗盘和 `Progress.md`，未重复已完成工作。
  transaction safety 终审已完成：post-admission component 的显式 pool-local TENTATIVE 只回滚该 DP
  overlay并从post-mandatory base重建decode-only；rollback-shadow构造及其余意外异常不再借
  `include_admission` 全局降级，而是 INTERNAL/typed fatal/global-zero-publication。exact telemetry
  agent正在直接修改C++，目标是删除或隔离跨admission-overlay的prepared cache、补initial+iteration+
  canonical group/final target combined fingerprint以及真实initial/iteration adapter attempt count；不向
  Python热路径恢复任何固定ABI反射检查。Sequence agent已完成raw optimized skeleton/target trim、raw
  allocator high-water、pickle legacy/magic/version gate及`assigned_dp`与已初始化ACTIVE.dp双向校验，
  C++ syntax已通过，待主线程审查。Issue-1%只读审计确认计划中的六个exact-length/ID fixture当前均
  未覆盖，且尚缺stable required Decode prepare/validate失败的typed-fatal+global-zero-publication hook/test。
  installed extension仍过期；最近无需重装的engine fake subset为`22 passed, 2 deselected`；未运行GPU。

- 11:59 UTC 自动压缩恢复检查点：已按协议完整重读本文件与 `Progress.md`，并先写回任务目标、
  当前故障和精确恢复点。用户追问 pool-local rollback 是否仍符合 LoongServe style；最终边界为
  “post-mandatory base + per-DP admission overlay”。若 admission overlay 或随后该 DP 的 required
  Decode component planner/validate/prepare/composition 返回普通 TENTATIVE，则只 reset 该 DP 的
  `PreparedAdmission` 和 scheduler delta，再基于相同 post-mandatory base 重建该 DP decode-only；
  其他 DP admission 保留。若稳定 decode-only component 仍失败，则按 INTERNAL/CAPACITY 分类并
  保持全局 required-Decode 零发布/fatal。这是 LoongServe 独立 DP pool admission 语义与 NanoDeploy
  共享 FFN collective 原子 publication 的组合，而不是跨 pool admission 事务。
- `/root/transaction_safety_audit` 正在实现上述关键修复：保存每 DP post-mandatory shadow/admission
  owner，支持局部 overlay rollback/rebuild，并新增 2-DP post-admission component failure hook/test，
  目标断言 DP0 admission 保留、DP1 waiting/OOE/blocks 不变且两个 DP entry Decode 都存在。
- combined pool transaction 的 earlier prepare-failure isolation、schedule-global exact-key、prepared
  iteration cache、canonical owner/arrival 与 allocation-free composition validator 已完成；独立
  C++ syntax/harness `6/6` 通过。但 earlier 2-DP 测试只覆盖 admission block 内注入失败，不能替代
  当前 post-admission component failure 回归。
- Sequence ABI 已补 raw/pickle 共用 BlockContext topology/table/location/inactive-state 校验、重复
  location/非法 target SP 拒绝、raw token tail 一致性、standalone pickle get/set 校验和 malformed
  pickle 测试；`test_decode_rpc_optimization.py` orphan fixture 已规范化。
- typed `LSAdmissionRecord` 已替代并删除 public legacy `ScheduleResult.ls_initial_*` 字段、binding、
  engine logger/test 路径；保留 config `ls_initial_kv_dop` 和内部 placement history getter 所需结构。
- pool resource epoch boundary 已修复：manual low-KV commit 与 public explicit `preempt()` 强制新
  epoch；同一步内部 publication 去重。low-KV 亦已完成 frozen-plan predispatch 校验、worker
  DP/SP/TP/role/move/chunk/bytes 守恒、postdispatch fail-closed、postcommit epoch exact +1 及最小
  `Scheduler::get_ls_pool_resource_epochs()` binding；纯 Python subsets 为 `18/19/1 passed`。
- 12:21 UTC 用户纠正 runtime 检查边界：固定 `ScheduleResult`/pybind 字段存在性、enum/type、DP/SP
  shape、iteration master Counter和telemetry schema不应在每个 Decode step反射扫描。主线程已将
  `_validate_ls_action_fields` 改成固定字段直接访问，只保留 action互斥、admission canonical owner/
  running、new admission与entry real隔离、real属于scheduled/running及OFFLOAD victim移出running等
  真正跨 publication 的动态关系；exact/shape/master census回到C++ runtime与真实scheduler tests。
  删除对应 fake negative tests；保留 gauge/executor异常 post-publication fail-close tests。无需重装的
  engine subset为 `22 passed, 2 deselected`，`py_compile`/局部diff-check通过。
- Python post-publication fatal boundary已用 `step()` 外层 result-returned boundary收口：schedule成功
  返回后 admission consume、gauge、executor、metrics/logger/postprocess任一未分类异常都会 latch
  `POST_PUBLICATION_INVARIANT`；KV dispatch失败仍保留更具体 `KV_CONSOLIDATION_FAILED`。
- transaction safety生产改造已落地并通过syntax/diff：每 DP admission overlay后独立plan/prepare/
  composition，TENTATIVE仅撤该pool并重建decode-only，其他DP owner/overlay保留；新增post-admission
  hook及2-DP回归。终审正在把per-DP loop之外的unexpected exception一律分类为INTERNAL/global-zero-
  publication，避免outer `include_admission` fallback再次全局丢 admission。
- exact-plan telemetry终审已确认 MUST 缺口：current attempt count只统计成功 iteration prepare，
  fingerprint/cache key遗漏initial block reservation、admission frontier、donor/new-group overlay与group
  boundary；rollback后可能错误复用旧overlay的prepared iteration。下一步在C++ pool transaction层补
  combined admission+iteration+group-graph fingerprint、overlay-aware key和真实adapter attempt count，
  不把这些固定ABI检查塞回Python热路径。§16.2 Issue-1% 六 fixtures、required Decode
  fault injection、部分 Sequence restore/legacy rejection 也仍为测试缺口。installed extension 已过期，
  尚未重新安装；未运行任何 GPU 操作。

## 下一步顺序

1. 单机学术 demo 目标已完成并提交。保留用户 `AGENTS.md`、`ray_executor.py`、历史 JSON/JSONL、
   profile 脚本，不纳入本任务提交。
2. GPU/真实 Ray-NCCL 端到端不属于本次学术 demo CPU完成线。若后续运行，任何
   GPU操作必须单独提权，Ray操作需清除代理。

## 工作树注意事项

- `nanodeploy/engine/ray_executor.py` 是用户原有 RPC buffer 修改，必须保留但不要纳入
  本任务提交。
- `nanodeploy/engine/llm_engine.py` 在任务前已有大范围换行变化；提交时需审慎选择
  本任务语义 diff。
- 多个 `docs-dev/2026-07-14..18/*.json[l]` 是用户未跟踪实验产物，不触碰、不暂存。
- 当前 `AGENTS.md` 显示为 modified；先检查其来源，仅将其视为用户规则，不自动提交。
- 所有外部 HTTP(S) 操作先设置 `http_proxy/https_proxy/HTTP_PROXY/HTTPS_PROXY` 为
  `http://127.0.0.1:15409`；Ray 相关操作反而必须 unset。

## 压缩恢复协议

发生上下文压缩后，继续任务前先完整读取本文件和 `Progress.md`，再查看原设计相关章节、
`git status`/`git diff`、subagent状态与最新测试结果；不得从头重复已完成实现。每次压缩前
更新本文件的时间、最新提交、验证结果、当前故障和精确下一步。
