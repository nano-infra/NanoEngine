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
