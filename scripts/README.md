# Scripts README

## `benchmark_least_batch_cpu.py`

用途：仅用 CPU 验证 least-batch staged admission 的控制面改动，不启动 CUDA、
Ray 集群、模型或 worker。

- 使用真实 `RequestRouter` 测批量派发到 positive receipt 的流水线，并断言所有
  请求都在 scheduler commit 数仍为 0 时进入 `PENDING_ADD`；
- 使用真实 localhost ZMQ 协议、服务端 Sequence 解码和
  `LocalEngineCore._stage_add_batch()`，验证 receipt 不等待 scheduler drain；
- `--modeled-commit-gate-ms` 模拟旧 stop-and-wait 的门控延迟。该结果只是敏感性
  模型，不是旧版本或线上 GPU 性能实测。

运行默认的五轮基准并保留 JSON：

```bash
python3 scripts/benchmark_least_batch_cpu.py \
  --requests 4096 \
  --engines 2 \
  --batch-size 64 \
  --prompt-tokens 32 \
  --modeled-commit-gate-ms 10 \
  --repeats 5 \
  --json-output docs-dev/2026-08-25/least_batch_cpu_benchmark.json
```

这个基准可以验证请求分发、receipt/commit 解耦、消息 batching、payload 单次传输
和本地暂存深度；不能替代真实 GPU quantum、跨节点 Ray/RDMA、吞吐、TTFT 或 TPOT
验收。

## `benchmark_least_batch_ray_cpu.py`

用途：从 Ray head 节点执行一条命令，把 CPU-only staged-ingress actors 硬绑定到
两个 Ray 节点，并由 head 上的真实 `RequestRouter` 通过跨机 ZMQ 完成请求分发。
Ray 只负责 actor 部署、状态读取和回收，Sequence payload 仍走 NanoDeploy 的真实
ZMQ ingress 路径。

前提：两个节点已经加入同一 Ray 集群，且各自声明至少 1 个 CPU 资源。脚本会在连接
Ray 前自动移除当前进程的 HTTP(S)/ALL proxy 变量，不会申请 GPU 资源。

```bash
python3 -m scripts.benchmark_least_batch_ray_cpu \
  --ray-address auto \
  --ray-nodes 2 \
  --engines 2 \
  --requests 4096 \
  --batch-size 64 \
  --prompt-tokens 32 \
  --repeats 5 \
  --json-output docs-dev/2026-08-25/least_batch_ray_cpu_benchmark.json
```

默认按 Ray `NodeManagerAddress` 选择前两个存活 CPU 节点。需要固定机器时，各传
一次 `--node-ip`：

```bash
python3 -m scripts.benchmark_least_batch_ray_cpu \
  --ray-address auto --ray-nodes 2 \
  --node-ip 10.0.0.1 --node-ip 10.0.0.2
```

计时包含 Router submit/planning、跨机 ZMQ send、服务端 Sequence decode、
LocalEngine stage 和 receipt；不包含 Ray actor 启动和结束后的 stats RPC。该模式
不启动 NanoDeploy GPU worker、模型或 scheduler commit，因此仍不能替代 RDMA、
GPU quantum、TTFT/TPOT 的两机验收。

## `run_2node_rate30_6min_matrix.sh`

用途：

- 串行运行两节点/16 卡的四组合对照：legacy global 与 hierarchical 各自的 DP16、
  DP2SP8
- 每组固定 rate 30、投递 6 分钟，共 10,800 个请求
- 四组统一使用 `hao_basic` SP 后端和 910,000 total-token 数据集过滤
- 支持 dry-run 和按 stage 单独补跑

检查命令：

```bash
DRY_RUN=1 scripts/run_2node_rate30_6min_matrix.sh
```

完整运行：

```bash
scripts/run_2node_rate30_6min_matrix.sh
```

详细备用记录见
[`docs-dev/2026-07-28/rate30_6min_2node_matrix/Runbook.md`](../docs-dev/2026-07-28/rate30_6min_2node_matrix/Runbook.md)。

### 默认 TPOT 口径

`scripts/sp_ablation/bench_serving_overhead.py` 的默认
`tpot_with_queue_ms` 使用统一的调度执行边界：

- legacy global：C++ `SequenceMetric` 的首次 scheduled 到本地完成时间，加调度器
  GPU 容量排队；
- hierarchical：LocalScheduler 首次进入 `executor.run` 到本地完成时间，加
  RequestRouter 全局 GPU 容量排队；
- 两者都会扣除最后一个固定 16-loop quantum 中未生成真实 token 的
  decode slot，再除以实际生成 token 数。legacy global 使用
  `record_step_tokens()` 保存的末 step ITL，hierarchical 使用末 quantum 的
  `executor.run / 16`，统计口径一致；
- GPU 容量排队完整保留；不计 benchmark dispatch、RPC/command pickup 或
  非容量的 quantum 边界等待。

每条请求同时保留原始 `first_forward_to_terminal_ms`、修正后的
`first_forward_to_terminal_real_token_ms`、末 quantum 的真实 token 数和
被扣除的 `final_quantum_unused_decode_ms`，便于复核修正幅度。

旧的 benchmark-observed `dispatch -> completion / generated tokens` 保留为
`dispatch_normalized_latency_ms`，不再作为默认 TPOT 或 goodput 的输入。

### Admission 延迟拆解

hierarchical 请求的 `T0 -> T1 -> T2 -> T3` 指标会写入 per-request JSONL 和
`summary.json`：

其中 T2 是 staged-ingress receipt，T3 是 authoritative AddResult commit。

- `dispatch_lag_ms`：T0 计划到达至 T1 实际投递；
- `ingress_ack_latency_ms`：T1 投递至 staged-ingress receipt；
- `router_pending_ms`：RequestRouter 中等待发起 admission RPC 的累计时间；
- `admission_rpc_ms`：前端观察到的 LocalEngine admission RPC 累计时间；
- `local_command_queue_ms` / `local_admission_ms`：同步 admission 的兼容口径；
- `staged_queue_ms`：positive receipt 后至最终 planned commit 尝试被拾取；
- `planned_commit_ms`：最终 planned placement 校验与 commit 耗时；
- `sequence_deserialize_ms`：服务端当前 ingress batch 的 Sequence 解码总耗时；
- `sequence_payload_bytes`：本请求的序列化 Sequence payload 字节数；
- `admission_rpc_residual_ms`：RPC 总时间扣除本地命令排队和 admission，
  用于定位 Ray actor mailbox、传输、轮询或前序 fallback；
- `frontend_ack_overhead_ms`：T1-T2 扣除 router 和 receipt RPC 后的前端观察余量。

所有跨节点字段传递的是各进程内单调时钟计算出的“时长”，不比较不同节点
的绝对时间戳。

## `plot_run_metrics.py`

用途：
- 从一个 benchmark run 目录里递归扫描 `.log`
- 提取 `TTFT`、`TPOT with queue`、`TPOT without queue`
- 按 `model / dataset / strategy / rate` 汇总并画总览图

输入要求：
- `--run-dir` 指向一个实际包含 benchmark `.log` 的目录
- 这些 `.log` 需要包含 `Benchmark Metadata` 和 `Benchmark Results`

基本用法：

```bash
MPLCONFIGDIR=/tmp/matplotlib python scripts/plot_run_metrics.py \
  --run-dir bench_logs/kimi_conversation_2node_16gpu_20260409_104224
```

指定输出目录：

```bash
MPLCONFIGDIR=/tmp/matplotlib python scripts/plot_run_metrics.py \
  --run-dir bench_logs/kimi_conversation_2node_16gpu_20260409_104224 \
  --output-dir bench_logs/kimi_conversation_2node_16gpu_20260409_104224/plots_run_metrics
```

输出内容：
- `data/metrics_summary.tsv`
- `data/metrics_duplicates_dropped.tsv`
- `plots/*__run_metrics.png`

说明：
- 这个脚本适合直接对单个 run 目录出图。
- 如果是链式 run，顶层目录只有 `chain_summary.tsv`、实际 `.log` 分散在别的子目录里，需要先把目标 `.log` 整理到一个 staging 目录，再对 staging 目录运行这个脚本。


## `plot_step_log_timeseries.py`

用途：
- 解析 `llm_engine.py` 打出的 `step - {...}` decode 结构化日志
- 画出随时间变化的：
  - `waiting_reqs`
  - `waiting_head_blocks / waiting_total_blocks`
  - `batch size / sp batch size`
  - `free blocks / used kv-cache / kv utilization`
  - `ITL / scheduler overhead`
- 同时输出按 rank 的热力图和明细 TSV

输入要求：
- 支持输入单个日志文件，或一个目录递归扫描
- 适合 `sweep.out` 这类包含 `step - {...}` 的日志
- 不适合只有最终汇总结果的 benchmark `.log`
- `run_progress.log`、`chain.progress` 这类纯进度文件通常没有 step 级结构化数据

基本用法：

```bash
MPLCONFIGDIR=/tmp/matplotlib python scripts/plot_step_log_timeseries.py \
  bench_logs/kimi_conversation_2node_16gpu_20260409_104224/dp16sp1_step10/sweep.out
```

批量处理多个日志：

```bash
MPLCONFIGDIR=/tmp/matplotlib python scripts/plot_step_log_timeseries.py \
  bench_logs/kimi_conversation_2node_16gpu_20260409_093429/dp2sp8_step10/sweep.out \
  bench_logs/kimi_conversation_2node_16gpu_20260409_104224/dp2sp8_step10/sweep.out \
  bench_logs/kimi_conversation_2node_16gpu_20260409_104224/dp16sp1_step10/sweep.out \
  --output-dir bench_logs/kimi_conversation_2node_16gpu_20260409_123614/plots_step_log_timeseries
```

跳过前面若干 step，并只看前 N 个 step：

```bash
MPLCONFIGDIR=/tmp/matplotlib python scripts/plot_step_log_timeseries.py \
  bench_logs/<run>/<stage>/sweep.out \
  --skip-steps 5 \
  --max-steps 200
```

可选参数：
- `--skip-steps`
  - 跳过前 N 个 decode step
- `--max-steps`
  - 只保留前 N 个 decode step
- `--output-dir`
  - 指定输出根目录
- `--num-kvcache-blocks`
  - 当日志里没打出 `num_cache_blocks` 时，手动指定每 rank 的总 block 数
- `--kvcache-block-size`
  - 当日志里无法解析 block size 时使用，默认 `64`

输出内容：
- `timeseries_summary.tsv`
- `sp_batch_sizes.tsv`
- `free_blocks.tsv`
- `used_kvcache_blocks.tsv`
- `metadata.tsv`
- `overview.png`
- `heatmaps.png`

说明：
- 时间轴优先使用估算的 decode 时间；拿不到时退化为 wall clock 或 step 序号。
- 多个输入文件批量跑时，输出目录名会自动带上 run tag 和 stage，避免不同 `sweep.out` 互相覆盖。


## 建议

如果 `matplotlib` 提示默认缓存目录不可写，统一用下面这种方式运行：

```bash
MPLCONFIGDIR=/tmp/matplotlib python <script> ...
```
