# Progress — 2026-07-23

## 2026-07-23 03:29 UTC

- 任务：运行单机 LoongServe-style 测试，验证
  `54e425a fix: keep LoongServe decode loops fixed`。
- CPU 定向回归：4 passed。
- 本机 Ray `10.102.206.14:7789`：单节点、8×H200，测试前后均为
  `0/8 GPU`。
- 权威 GPU 用例：DP1×SP8/EP8、8 requests、prompt 63、
  max_tokens 18、loop_count 16；1 admission + 2 Decode，
  scheduler K 为 `[16, 16]`，worker CUDA-event 聚合 K 也为
  `[16, 16]`，最终 8/8 完成，进程 exit 0。
- 额外 GPU 交叉验证：prompt 4096、max_tokens 34，Decode K 为
  `[16, 16, 16]`，最终 8/8 完成。
- 结论：固定尾轮修改奏效；旧 `[16, 1]` 尾轮已变为 `[16, 16]`，
  postprocess 正确丢弃无效尾 token。
- 详细命令、断言与产物见
  `loongserve_loop16_fixed_tail_single_node_validation_20260723.md`。
- 本任务没有修改代码；保留用户原有 `AGENTS.md`、
  `nanodeploy/engine/ray_executor.py` 和历史实验产物。

## 2026-07-23 03:41 UTC

- 后续目标：把同一固定尾轮场景延长为 5 分钟单机长跑。
- 原 longrun 入口硬编码 `loop_count=1`；已增加向后兼容的
  `--loop-count` 参数、resolved manifest 校验、per-step execution K 记录和
  按实际 K 折算的 step ITL。CPU profile/CLI 回归 `5 passed`，提交为
  `23db1fd feat: support chunked LS longrun tests`。
- GPU 配置：单机 DP1×SP8/EP8、8×H200、eager、prompt 63、
  max_tokens 18、loop_count 16、4 req/s、burstiness 2、inflight cap 64。
- 结果：注入 300 秒、drain 6.61 秒、总运行 306.90 秒；accepted/completed
  `1264/1264`、unfinished 0、manifest `success`、进程 exit 0。
- 90 个 Decode step 的 scheduler `execution_loop_count` 全部为 16；
  90 条 worker CUDA-event 聚合记录同样全部为 16。1,264 个 completion 的
  output length 全部精确为 18。
- 稳定段 Decode ITL：mean 208.36、p99 211.43 ms/token。输出吞吐
  74.14 token/s，请求吞吐 4.12 req/s。
- inflight cap 在到达 burst 中丢弃 35 个尚未 admission 的 arrival；这是
  有界 backpressure，不是 accepted request 失败。
- 日志没有 inconsistent K、traceback、scheduler fatal、CUDA/NCCL error
  或 drain timeout。测试后 Ray `0/8 GPU`，8 张卡均 `0 MiB / 0%`。
- 详细结果见
  `loongserve_loop16_fixed_tail_longrun_5min_20260723.md`。

## 2026-07-23 03:55 UTC

- 后续目标：提高 request rate，制造大量 waiting backlog，观察拥塞场景下的
  调度开销与整体性能。
- rate=20 req/s 的短校准只形成约 218 个 inflight，已中止；在 waiting=113
  时观察到单轮外层 scheduler 约 8.26 ms、内部 planning 约 7.03 ms。
- 当前正式运行 rate=50 req/s、inflight cap 1024、其余配置保持
  DP1×SP8/EP8、prompt 63、max_tokens 18、loop_count 16、发压 300 秒。
  已启用 `NANODEPLOY_LS_SCHEDULER_PHASE_TIMING=1`。
- 正式运行在 31 秒时达到 inflight 960；运行批次 128、waiting 约 832。
  最近一次进度（elapsed 183.25 秒）为 accepted/completed
  `4225/3265`、inflight 960、backpressure drops 4756，稳定窗口 Decode ITL
  mean 210.83 ms/token，无已知错误。
- waiting=832 时每轮扫描 896 个候选，外层 scheduler 约 11–12 ms，
  内部 planning 约 9.8–10.6 ms；其中 admission scan 约 8.0–8.4 ms。
  正式运行仍在继续，完成后需要等待 drain、做分桶分析、资源清理并提交报告。

## 2026-07-23 04:02 UTC

- rate=50 的高 backlog 正式轮已完成，进程 exit 0、manifest `success`。
  accepted/completed `6401/6401`、unfinished 0、drain 50.03 秒；
  max inflight 1024，backpressure drops 8217。
- 103 个 Decode step 中 81 个稳定在 `waiting=832`。这一平台的 scheduler
  mean/median/p99 为 `11.404/11.367/12.180 ms`，内部 planning mean
  9.954 ms；相比低负载稳定 scheduler mean 1.306 ms 放大 8.73 倍。
- planning 热点是 admission 8.499 ms，其中 inclusive admission scan
  7.832 ms；每轮扫描 896 candidates，调用 65 次 admission plan 和
  194 次 future-KV pool plan。
- waiting=832 的 Decode ITL mean 210.522 ms/token，比低负载稳定段
  208.356 ms/token 高 1.04%。稳定饱和完成吞吐约 18.93 req/s、
  340.8 output token/s。
- 性能代价主要是排队：queueing median/p99 `43.821/44.185 s`，
  E2E median/p99 `53.885/54.391 s`。
- 103 个 Decode scheduler execution K 全为 16，6401 个 completion 的
  output length 全为 18；无 traceback、scheduler fatal、K 不一致、
  CUDA/NCCL/OOM 或 drain timeout。
- 测试后 Ray `0/8 GPU`，8 张 H200 全部 `0 MiB / 0%`。
- 详细报告：
  `loongserve_loop16_backlog_r50_5min_20260723.md`。

## 2026-07-23 07:17 UTC

- 当前目标：解释相同 `loop_count=16` 下，LoongServe-style
  `ls_style_loop16_dp2sp8_r20_diag01_3` 为什么仍明显慢于最近一次
  NanoDeploy original rate=20 基线。
- 两轮真实 logical sequence-step 完全相同，均为 `272430`；因此差异不是
  loop 数量或输出工作量。Original/LS 总耗时分别为 `411.06/510.15 s`。
- 外层 scheduler 累计从 `0.474 s` 增至 `14.964 s`，增加 `14.490 s`；
  但 model-runner/executor 路径从推算的 `404.494 s` 增至实测
  `488.411 s`，增加 `83.918 s`，约解释总差距的 84.7%。7 次
  consolidation 仅累计 `0.670 s`。所以 loop=16 确实摊薄了 scheduler，
  但 scheduler 不是主要瓶颈。
- LS 的实际 KV DoP（按 sequence-step 加权）为：
  DoP1 `60.77%`、DoP2 `32.11%`、DoP3 `6.35%`、DoP4+ `0.77%`；
  original 的 DoP1 为 `99.15%`。LS 的 attention sequence-rank work
  为 `401107`，original 为 `288532`，增加约 39.0%。
- 忙时 LS/original 的最忙 rank batch 平均约 `129.4/39.0`，
  `max_rank_batch / mean_rank_batch` 平均约 `2.18/1.11`。当前 LS 的
  group ownership、master 扩缩容与历史 KV 分片，使请求虽然经常显示
  master DoP=1，仍要访问多个 KV rank；最慢 rank、SP A2A 和同步路径决定
  每 token 时间。
- LS/original 的 hottest-rank KV utilization 分别达到 `100%/56.55%`；
  LS 每 rank KV block 又因 memory utilization 0.85 而比 original 的
  0.90 少约 11.9%，加剧容量压力、KV 分片和排队的正反馈。
- 序列元数据/RPC send 累计从约 `5.585 s` 增至 `12.847 s`，可解释
  model-runner 差距中的约 `7.26 s`；其余大头仍在 worker 侧
  prepare/forward/SP communication/synchronization。当前日志未启用
  CUDA-event forward timing，不能再精确拆出纯 kernel 与通信占比。
- 这不是严格同机同版本 A/B：两轮 Ray 节点组合、代码版本、KV memory
  utilization、warmup 和 recv 配置均不同。下一步最有判别力的是在同一
  节点/同一代码下启用 forward timing，分别跑 KV-local 短 prompt 与当前
  长 prompt workload。

## 2026-07-23 07:39 UTC

- 当前目标：从
  `docs-dev/2026-07-23/ls_style_loop16_dp2sp8_r20_diag01_3.log`
  分析现有 LoongServe-style 调度器进行 DoP 扩张和缩小的实际条件。
- 分析会区分两层 DoP：每轮执行的 master/compute DoP，以及序列历史 KV
  分布对应的 KV DoP；前者可随 batch 快速变化，后者缩小需要 KV
  consolidation。
- 接下来提取本轮全部 `scale_reasons`、master/rank allocation 和
  consolidation 决策事件，再回查 `scheduler.cpp` 与
  `sp_state_manager.cpp` 确认阈值和资源约束。

## 2026-07-23 07:48 UTC

- 已完成当前 LoongServe-style DoP 扩缩容分析，详细报告写入
  `loongserve_style_dop_scaling_analysis_20260723.md`。
- 需要区分 master/compute DoP 与 KV/allocation DoP。master DoP 每轮重算，
  可以在已有 allocation 内快速扩缩；KV DoP 的非空 rank 缩小需要
  consolidation。
- 本轮 3,818 个 group-step 的 `scale_reasons` 全为 `none`，未发生 Decode
  planner 的 compute/memory/receiver 物理加 rank。531 个 admission batch
  中有 513 个 `CAPACITY_APPEND`；其中 226 个新 group 的首次 allocation
  大于 admission 自身 planned DoP，说明 KV DoP 增长主要来自 donor group
  merge。
- 当前实际 compute threshold 是
  `ls_min_comp_bound_decoding_batch_size=128`；manifest 中的
  `ls_decode_batch_per_master=64` 未被 core planner 使用。
- KV consolidation 需要 group utilization `<50%`、相同 candidate 稳定
  2 steps、scale-up/consolidation cooldown 2 steps、source migration
  `<=128` blocks，且迁移后 retained ranks utilization `<=80%`。
- 尾部执行 7 次 consolidation：group 525 `2→1`、group 531 `3→1`、
  group 532 `5→1`，总 maintenance stall 约 669.69 ms。

## 2026-07-23 08:06 UTC

- 当前目标：判定 LoongServe-style 慢主要来自 GPU 负载不均衡，还是短请求
  被不必要地扩成 CP2+。
- 已确认日志中的 `sp_size_hist_global` 是逐序列已提交 KV rank 数的精确
  统计；本轮共有 `272430` logical sequence-step、`401107`
  attention sequence-rank work。
- 已重放 admission、decode master 分配与 consolidation：以 admission
  planned ranks 初始化序列 KV ownership 的重放，在 346 个 Decode step
  中精确匹配 339 个，累计 sequence-rank work 为 `401104`，与日志精确值
  只差 3，可用于把 CP2+ 归因到具体 prompt 长度。
- 下一步：按 prompt 长度统计 CP2+ 和额外 rank-work，构造逐轮 master/KV
  rank 负载偏斜指标，并在相近 batch/KV 压力下比较 ITL，给出直接瓶颈与
  根因的区分。

## 2026-07-23 08:13 UTC

- 已完成 LoongServe-style 慢因拆分，详细报告为
  `loongserve_style_slowdown_cp_vs_imbalance_20260723.md`。
- 本轮 128,674 个可归因的额外 CP rank-work 中，prompt `<=1K` 请求贡献
  126,502（98.31%）；这些请求最终 context 最大仅 2,301 tokens。短请求
  额外 work 中 99.91% 来自 admission 初始 DoP1、后续 master 切换造成的
  CP 扩散。
- `batch>=512` 的 249 个忙轮中，最忙 attention GPU 的 master batch 平均
  是 16 卡均值的 2.16 倍；同批量范围的 original 参考约 1.11 倍。
- 控制 batch、max KV utilization 和另一个因素后，忙轮 ITL 与平均 CP 的
  partial correlation 为 0.655，与 master imbalance 为 0.335；二者均有
  独立信号，但短请求 CP 更强。
- `batch>=800` 的 2×2 分桶中，低 CP/低偏斜 ITL 90.57 ms，单独一个因素高
  约 94 ms，而高 CP/高偏斜达到 118.07 ms，说明存在明显交互。
- 代码根因是逐 group source-greedy fast path 不按单请求 owner locality
  或 context 长度决策，combined 阶段只拼接/验证而不做 DP 全局 rebalance；
  master 切换又会保留 historical KV。建议分别做 sticky-local、global
  balance 和二者同时启用的三组 A/B。

## 2026-07-23 08:24 UTC

- 当前目标：实现一个独立的静态 Decode 数据面 harness，直接运行
  T00/T10/T01/T11 四个反事实布局，用实验区分短请求 CP 扩散与
  master batch 偏斜的影响；本轮由用户明确要求写代码。
- 固定代表性 workload：DP2×SP8、每个 DP 520 条请求、context 800、
  `loop_count=16`。当前 CP 分布为全局
  DoP1/2/3=`574/347/119`，当前 master batch 为
  DP0=`[102,75,0,1,120,113,58,51]`、
  DP1=`[16,138,66,1,36,125,16,122]`；均衡布局为每 rank 65。
- 实现策略：复用现有 `LLM`、`SPStateManager.allocate_ls_initial_batch`
  和 worker `ModelRunner.run`，绕过 `engine.step()` 与生产 scheduler；
  以固定 sequence metadata/KV block table 重复执行 Decode，读取
  CUDA-event forward timing。布局生成、约束校验与 2×2 effect 计算保持
  纯 Python，并增加无需 GPU/Ray 的单元测试。
- CP2/3 默认复现短请求历史 KV 碎片形态：master 保留 16 tokens，
  dominant owner 持有其余大段，DoP3 再有一个 16-token shard；这样测试
  的是额外 KV rank-work/通信，不会把 CP 人为变成均匀上下文切分优化。
- 计划新增 `scripts/bench_ls_decode_static_layout.py` 和对应 CPU 测试；
  不修改 C++ 或生产 scheduler。完成后运行定向测试和 CLI help，并提交
  独立 commit。

## 2026-07-23 08:43 UTC

- 静态 2×2 harness 已实现为
  `scripts/bench_ls_decode_static_layout.py`。它自动清除 Ray 代理、启用
  worker CUDA-event timing，使用正式 Issue001 engine profile 和 DLSlime
  metadata 传输，但计时路径不包含 scheduler。
- T00/T10/T01/T11 使用同一 DP2×SP8、1040 requests、context 800、
  loop16 workload；CP2/3 分别采用 `[784,16]` 和 `[768,16,16]` 的历史
  碎片形态。每个 case/round 都做成批量原子 allocation，结束后释放并
  断言两个 SPStateManager 的 running sequence/token 计数归零。
- 默认做 3 个随机 case-order rounds，每 case 每轮 1 次 warmup + 10 次
  measurement；JSON 会逐 case checkpoint，并保存全部 16 rank×16 loop
  CUDA timings、wall time、master/participant/receiver batch、KV
  token/block load、总 effect、interaction 和 paired-round effect。
- `--dry-run` 不导入 Ray 或使用 GPU，可先审查精确布局。CPU 定向测试覆盖
  四组边际、碎片形态、容量失败、真实 C++ block allocation/cleanup、
  critical-path 和 2×2 effect；相关回归共 `18 passed`。`black --check`、
  `py_compile`、CLI help、dry-run 和 `git diff --check` 均通过。
- 本轮没有修改 C++ 或生产 scheduler，因此不需要 `pip install -v -e .`；
  尚未申请 GPU，也没有实际启动 16-GPU harness。
