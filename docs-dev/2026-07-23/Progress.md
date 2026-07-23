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
