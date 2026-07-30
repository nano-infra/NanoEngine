# Scripts README

## `run_2node_rate30_6min_matrix.sh`

用途：

- 串行运行两节点/16 卡的四组合对照：中心化与去中心化各自的 DP16、
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

- 中心化：C++ `SequenceMetric` 的首次 scheduled 到本地完成时间，加调度器
  GPU 容量排队；
- 去中心化：LocalScheduler 首次进入 `executor.run` 到本地完成时间，加
  RequestRouter 全局 GPU 容量排队；
- 两者都除以实际生成 token 数，不计 benchmark dispatch、RPC/command pickup
  或非容量的 quantum 边界等待。

旧的 benchmark-observed `dispatch -> completion / generated tokens` 保留为
`dispatch_normalized_latency_ms`，不再作为默认 TPOT 或 goodput 的输入。

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


## `plot_nano_longshort_matrix.py`

用途：
- 针对 `issue001_issue003_issue005_mixed60k_longshort...` 这类 NanoDeploy 链式 run
- 扫描对应 stage 的 `sweep_summary.tsv`
- 从每个点位的 benchmark `.json` 逐行复算 `Normalized Latency`
- 输出 `SLO Attainment`、`Goodput`、`NormLat Avg`、`NormLat P99` 等汇总表
- 生成和论文侧 nano 图类似的 5 列数据集矩阵图

当前内置的数据集：
- `ShareGPT4o`
- `Issue1%`
- `Issue3%`
- `Issue5%`
- `Gemini Issues`

基本用法：

```bash
MPLCONFIGDIR=/tmp/matplotlib python scripts/plot_nano_longshort_matrix.py \
  --run-dir bench_logs/issue001_issue003_mixed60k_longshort_dp4dp32_bs256_20260408_165526
```

输出内容：
- `data/metrics_summary.tsv`
- `data/metrics_duplicates_dropped.tsv`
- `data/slo90_crossings.tsv`
- `data/metadata.tsv`
- `plots/nano_longshort_matrix.png`
- `plots/nano_longshort_matrix.pdf`

说明：
- 会自动合并 `issue003/issue005/gemini` 这种分段补跑 stage，并按 `dataset + strategy + rate` 去重，默认保留时间戳最新的点位。
- 当前只读取 DeepSeek v3 的 nano 相关 stage，不会把 `kimi_k2` 混进图里。


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
  bench_logs/issue001_issue003_mixed60k_longshort_dp4dp32_bs256_20260407_193009/longshort_issue003_deepseek_v3/sweep.out \
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
