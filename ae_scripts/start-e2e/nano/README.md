# NanoDeploy 4-node E2E launcher

这里保存 NanoDeploy E2E 实验的完整启动链，不调用 NanoDeploy checkout
中的 `scripts/*.sh` 或 benchmark Python。`NanoDeploy-July` 仅作为被测代码和
已编译扩展，通过 `NANODEPLOY_WORKDIR` 加载。

| 文件 | 用途 |
|---|---|
| `run_kimi_e2e_rates.sh` | 四节点 rate sweep、环境变量、日志和失败处理 |
| `bench_serving_overhead.py` | 数据集读取、Poisson 请求发送、warmup 和指标输出 |

## 默认实验配置

默认配置与原来的 Kimi 10 分钟脚本保持一致的部分包括：4 节点 × 8 H200、
DP=4、SP=8、TP=1、`hao_basic`、full CUDA Graph、256 个 warmup 请求、
每个 rate 发送约 `rate * 600` 个请求并在发送完成后 drain。

调度策略有意改成下面这组配置：

| 配置 | 默认值 | 含义 |
|---|---|---|
| scheduler architecture | `legacy_global` | 中心化的全局调度器 |
| routing | `LeastBatch` | 在 4 个 DP group 之间路由 |
| dynamic SP strategy | `bucket` | 按序列长度 bucket 直接选择 SP/CP size |
| bucket preset | `kimi_k2` | NanoDeploy 内置的 Kimi-K2 bucket policy |

内置 bucket policy 的展开值会写进每次实验的 `config.txt`：

```text
1:1024-10240;
2:10241-22528;
3:22529-190464;
4:190465-210944;
5:210945-354304;
6:354305-624640;
7:624641-673792;
8:673793-1000000
```

DeepSeek-V3 的其他 launcher 继续使用独立的 `deepseek_v3` preset：

```text
1:1024-63488;
5:63489-210944;
6:210945-399360;
7:399361-428032;
8:428033-1048576
```

这不是 `long_short_sp8`，也不是 `legacy` 的 segment-size 搜索。启动命令虽然
仍包含 `--segment-size 65536`，但该值只保留为 KV block/segment 的内部记账
粒度；在 `bucket` 分支中，参与计算的 SP size 由上面的 bucket policy 强制
选择，而不是由 segment size 决定。

实现细节：内置 preset 的第一个显式区间从 1024 开始。小于 1024 token 的
序列会回退到初始 SP size；在这里的 65536 accounting granularity 下该值是
SP=1，与第一个 bucket 的选择一致。1024 token 及以上完全按上述区间选择。

## 数据集路径以及如何切换

数据集由 `DATASET_PATH` 指定，CSV 必须包含 runner 所需的 `prompt_len` 和
`output_len` 两列。launcher 中的站点默认路径仅用于本地实验；AE 运行时建议
显式设置数据集路径。

切换数据集只需要覆盖 `DATASET_PATH`，不需要修改脚本：

```bash
env \
  RUN_TAG="sharegpt_mixed_random" \
  DATASET_PATH="<dataset-csv>" \
  bash start-e2e/nano/run_kimi_e2e_rates.sh
```

`MAX_REQUEST_TOKENS=0` 表示不按 `prompt_len + output_len` 过滤。默认的
`MAX_INPUT_LEN=1000000` 与模型长度上限一致；如果想完全关闭 prompt-length
过滤，可显式传空值：`MAX_INPUT_LEN=`。

## 启动

在 node 0 上运行。Ray 集群需要提前启动，NanoDeploy checkout、模型、数据集
和本 AE 目录必须以相同绝对路径对四个节点可见。Python/CUDA 环境沿用调用
该脚本时已经激活的环境。默认连接总 README 中配置的 Ray 集群；只有使用
其他 Ray 集群时才设置 `RAY_ADDR`。

先做 dry-run，检查展开后的每个 rate 命令：

```bash
env \
  RUN_TAG="kimi_bucket_dryrun" \
  DRY_RUN=1 \
  bash start-e2e/nano/run_kimi_e2e_rates.sh
```

按原来的 rate 25/30/35、每个 rate 发送 600 秒运行：

```bash
RUN_TAG="kimi_issue005_rates"

env \
  RUN_TAG="$RUN_TAG" \
  NANODEPLOY_WORKDIR="<nanodeploy-checkout>" \
  MODEL_PATH="<kimi-k2-model-path>" \
  DATASET_PATH="<dataset-csv>" \
  REQUEST_RATES="25 30 35" \
  SEND_DURATION_SEC=600 \
  MAX_REQUEST_TOKENS=0 \
  SLIME_QP_NUM=4 \
  bash start-e2e/nano/run_kimi_e2e_rates.sh
```

默认日志统一保存在：

```text
bench_logs/e2e/${RUN_TAG}/
├── config.txt
├── run.progress
├── run_summary.tsv
└── rate_*/
    ├── command.txt
    ├── driver.log
    └── itl_samples.jsonl
```

另一个 tmux pane 中可以查看进度：

```bash
tail -f "bench_logs/e2e/${RUN_TAG}/run.progress"
```

完整运行日志在 `rate_*/driver.log`。`run.progress` 只记录 sweep 的开始、完成、
失败和 rate 间 cooldown。

runner 不会自动给 `RUN_TAG` 添加时间戳；未设置时固定使用
`kimi_issue005_rates`，因此默认输出目录始终是
`bench_logs/e2e/kimi_issue005_rates/`。为避免静默混合两次实验的数据，如果
目标目录已经存在，runner 会拒绝覆盖。需要保留另一组结果时再显式指定新的
`RUN_TAG` 或 `OUTPUT_DIR`。

## 可覆盖的环境参数

常用参数包括 `RAY_ADDR`、`MASTER_ADDR`、
`REQUEST_RATES`、
`SEND_DURATION_SEC`、`BATCH_SIZE`、`GPU_UTIL`、`MAX_MODEL_LEN`、
`MAX_INPUT_LEN`、`MAX_REQUEST_TOKENS`、`SP_BACKEND` 和
`ROUTING_STRATEGY`。默认的中心化架构和 `bucket` 策略在 runner 内固定，
以免误跑成 hierarchical、long/short 或 segment-based baseline。

如需不用内置 preset 的自定义 bucket，必须同时关闭 preset：

```bash
env \
  DYNAMIC_SP_BUCKET_PRESET=none \
  DYNAMIC_SP_BUCKET_POLICY='1:1-100000;4:100001-200000;8:200001-1048576' \
  bash start-e2e/nano/run_kimi_e2e_rates.sh
```

自定义区间格式和范围由 NanoDeploy 的 `Config` 再次校验。
