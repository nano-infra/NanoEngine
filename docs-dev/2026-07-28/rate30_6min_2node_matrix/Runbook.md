# 两节点 Rate-30 / 6min 四组合 Benchmark 备用手册

记录日期：2026-07-28

入口脚本：

```text
scripts/run_2node_rate30_6min_matrix.sh
```

## 测试矩阵

| Stage | 对外名称 | `scheduler_arch` | 拓扑 | GPU 数 |
|---|---|---|---|---:|
| `centralized_dp16` | 中心化 DP16 | `legacy_global` | DP16 / SP1 / TP1 / EP16 | 16 |
| `centralized_dp2sp8` | 中心化 DP2SP8 | `legacy_global` | DP2 / SP8 / TP1 / EP16 | 16 |
| `decentralized_dp16` | 去中心化 DP16 | `hierarchical` | DP16 / SP1 / TP1 / EP16 | 16 |
| `decentralized_dp2sp8` | 去中心化 DP2SP8 | `hierarchical` | DP2 / SP8 / TP1 / EP16 | 16 |

默认按表中顺序串行执行。四组不能并行运行，因为每组都会占用全部
16 张 GPU。

## 固定基线参数

- 请求速率：`30 requests/s`
- 投递窗口：`360 s`（6 分钟）
- 请求数：`30 * 360 = 10,800`
- SP 后端：`hao_basic`，四组均显式传入
- Batch size / max num seqs：`192`
- Segment size：`65,536`
- Loop count：`16`
- CUDA Graph：`full`
- GPU memory limit：`141 GB`
- GPU memory utilization：`0.9`
- Max model length：`1,000,000`
- 数据集最大请求长度：`prompt_len + output_len <= 910,000`
- Routing：`LeastBatch`
- Hierarchical router policy：`least_batch`
- 诊断间隔：`1 s`
- Slow-add 阈值：`20 ms`
- Dynamic SP：关闭；DP2SP8 使用固定的 2 个 DP group、每组 8 个 SP rank

“6 分钟”只表示 10,800 个请求的投递窗口。实际 wall time 还包括模型
启动、CUDA Graph warmup、最后一批请求 drain，以及各 stage 之间
`start_bench.sh` 自带的等待时间。

## 默认环境

```text
Ray:     10.102.206.14:8776
Master:  10.102.206.14:29500
Model:   /mnt/nvme1n1/ml_research/chenjiefei/models/deepseek-v3
Dataset: /mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv
```

这些值都可以通过同名环境变量覆盖，SP 后端除外；这个备用矩阵固定使用
`hao_basic`。

## 运行方式

先检查四条最终命令，不启动 Ray actor：

```bash
DRY_RUN=1 scripts/run_2node_rate30_6min_matrix.sh
```

运行完整四组合：

```bash
scripts/run_2node_rate30_6min_matrix.sh
```

显式指定集群与日志标签：

```bash
RUN_TAG=rate30_6min_backup_01 \
RAY_ADDR=10.102.206.14:8776 \
MASTER_ADDR=10.102.206.14:29500 \
scripts/run_2node_rate30_6min_matrix.sh
```

只运行或补跑某个 stage：

```bash
RUN_TAG=rate30_6min_backup_01 \
scripts/run_2node_rate30_6min_matrix.sh decentralized_dp2sp8
```

也可以一次指定多个 stage：

```bash
scripts/run_2node_rate30_6min_matrix.sh \
  centralized_dp16 \
  decentralized_dp16
```

默认任一 stage 失败后停止，以免残留 actor 或 GPU allocation 污染后续
结果。确认集群已经清理后，如确实希望继续其他 stage，可设置：

```bash
CONTINUE_ON_ERROR=1 scripts/run_2node_rate30_6min_matrix.sh
```

## 结果位置

默认根目录：

```text
bench_logs/2node_rate30_6min_matrix_<UTC timestamp>/
```

其中：

- `matrix.progress`：四组合总进度及完整复现命令
- `matrix_summary.tsv`：stage、scheduler、DP/SP、请求数、状态、退出码及日志目录
- `<stage>/console.log`：该 stage 的完整外层输出
- `<stage>/run_progress.log`：`start_bench.sh` 的运行进度
- `<stage>/<model>/<dataset>/<setting>/`：benchmark `.log`、请求级 `.jsonl`
  和 `.summary.json`

复用同一个 `RUN_TAG` 补跑时，`matrix_summary.tsv` 会追加记录；以最后一条
同名 stage 记录为准。

## 运行前检查

1. Ray 集群应为两节点、共 16 张空闲 GPU。
2. 确认没有上一次失败遗留的 Ray actor 或 GPU allocation。
3. 确认模型目录和 CSV 数据集在两台机器上路径一致且可读。
4. 先执行一次 `DRY_RUN=1`，检查地址、数据集、`10,800` 请求、
   `--max-request-tokens 910000` 和 `--sp-backend hao_basic`。
5. 四组保持同一 commit、模型、数据集和硬件状态，避免横向比较失真。

## 脚本验证记录

2026-07-28 已完成 dry-run，确认生成的四条命令分别为：

- `legacy_global`, DP16 / SP1
- `legacy_global`, DP2 / SP8
- `hierarchical`, DP16 / SP1
- `hierarchical`, DP2 / SP8

四条命令均为 rate 30、360 秒对应的 10,800 请求，并显式使用
`--sp-backend hao_basic` 和 `--max-request-tokens 910000`。
