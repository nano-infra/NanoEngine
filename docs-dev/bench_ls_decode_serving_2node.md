# 两节点 LoongServe-style Issue 1% Bench Serve

本文记录如何启动 `2 DP × 8 SP / EP16`、共 16 GPU 的 LoongServe-style Decode-only serving benchmark。

这里测试的是 NanoDeploy 中实现的 LoongServe-style multi-master Decode scheduler，不是原版 LoongServe 的完整 serving stack。

## 测试配置

- Ray：`10.102.243.60:8776`
- distributed master：`10.102.243.60:29906`
- 节点：`10.102.243.60`、`10.102.206.14`，每节点 8 GPU
- topology：Attention `DP2 × SP8 × TP1`，FFN `EP16`
- model：`/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3`
- dataset：Issue 1% CSV，共 60,000 条，其中约 1% 为长请求
- request rate：`20 req/s`
- offered-load window：`360 s`
- request count：`20 × 360 = 7200`
- warmup：32 requests，512 input tokens，8 output tokens
- CUDA Graph：full
- routing：LeastBatch
- LS initial KV DoP：自动选择（`0`）
- KV consolidation：execute

入口脚本为 `scripts/bench_ls_decode_serving.py`。

## 启动命令

从 NanoDeploy-July 仓库根目录执行。Ray 通信不使用 HTTP 代理，因此必须先清除代理变量。

```bash
cd /mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-July

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

export NANODEPLOY_LOG_DECODE_A2A_MASKS=0
export NANODEPLOY_LOG_DECODE_STEP_DETAIL=0
export NANODEPLOY_LOG_MODEL_FORWARD_TIMING=0
export RAY_DEDUP_LOGS=1
export PYTHONUNBUFFERED=1

set -o pipefail
python scripts/bench_ls_decode_serving.py \
  --attention-dp 2 \
  --ray-address 10.102.243.60:8776 \
  --master-address 10.102.243.60:29906 \
  --model-path /mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3 \
  --csv-path /mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv \
  --request-rate 20 \
  --duration-sec 360 \
  --num-requests 7200 \
  --output-jsonl docs-dev/ls_style_issue001_2node_dp2sp8_r20_6min.jsonl \
  2>&1 | tee docs-dev/ls_style_issue001_2node_dp2sp8_r20_6min.log
```

该命令会使用 GPU。通过 Codex 启动时，必须在执行这条命令前单独申请 GPU 提权。

## 时间语义

当前 driver 按现有 NanoDeploy bench-serve 口径生成 7,200 个 Poisson 到达请求。`360 s` 是目标 offered-load window，而不是强制 wall-clock timeout；随机采样后的最后到达时间会在 360 秒附近。

最后一个请求发出后，benchmark 会继续 drain，直到全部请求完成。因此总运行时间可能显著超过 6 分钟。若中途 `Ctrl-C`，该次运行是不完整结果。

正式计时开始前还会执行以下初始化：

1. 创建两节点上的 16 个 Ray workers；
2. 捕获每个 rank 的 20 个 local graphs 和 180 个 SP graphs；
3. 运行 32-request warmup；
4. 打印 `LS_BENCH_ARRIVALS` 后进入正式 workload。

200 个 CUDA Graph 的捕获可能需要数分钟，不计入正式 workload 时间。

## 监测

实时查看日志：

```bash
tail -f docs-dev/ls_style_issue001_2node_dp2sp8_r20_6min.log
```

筛选关键阶段和错误：

```bash
tr '\r' '\n' < docs-dev/ls_style_issue001_2node_dp2sp8_r20_6min.log | \
  rg 'LS_WARMUP_SUMMARY|LS_BENCH_ARRIVALS|Processing Requests:|Benchmark Results|Traceback|RuntimeError|CUDA error|out of memory'
```

正常完成时，日志末尾应包含 `Benchmark Results`，并生成：

- `docs-dev/ls_style_issue001_2node_dp2sp8_r20_6min.log`
- `docs-dev/ls_style_issue001_2node_dp2sp8_r20_6min.jsonl`

JSONL 在全部请求完成后统一写入；中途终止时可能不存在，不能将其当作有效结果。

## 停止与清理检查

前台运行时按 `Ctrl-C`。随后确认 benchmark driver 已退出：

```bash
ps -eo pid,args | rg '[b]ench_ls_decode_serving.py' || true
```

不要为了停止单次 benchmark 直接执行 `ray stop`，因为该 Ray 集群可能由其他任务共享。

## 当前已知状态

截至本文撰写时，rate 20 和 rate 30 的试跑都由人工中止，尚无完整可用于性能比较的结果。高负载阶段观察到反复出现：

```text
LS-Decode-Core plan failed ... append capacity cannot cover remaining requests
Preemption happens ...
```

rate 20 下请求仍有完成进展，但完成速率明显低于到达速率，队列会增长。因此正式复跑时应重点观察 preemption 是否收敛，以及停止注入后能否完整 drain；中止日志只能用于诊断，不能用于报告吞吐或延迟。

超长 Issue 请求要求保留以下参数：

- `ls_decode_initial_kv_dop=0`：让 scheduler 自动选择能够容纳 prompt 的最小 KV DoP；
- `max_num_recv_seqs=128`：为多-rank KV placement 和 receiver metadata 留出容量。

脚本已将它们设为默认值。

## NanoDeploy original setting 对照组

若要在相同的两节点、Issue 1%、rate 20、6 分钟 workload 上运行用户给出的 sh 所使用的 NanoDeploy 策略，直接调用原始 CSV serving driver：

```bash
cd /mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-July

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

export NANODEPLOY_LOG_DECODE_A2A_MASKS=0
export NANODEPLOY_LOG_DECODE_STEP_DETAIL=0
export NANODEPLOY_LOG_MODEL_FORWARD_TIMING=0
export RAY_DEDUP_LOGS=1
export PYTHONUNBUFFERED=1

set -o pipefail
python scripts/issue003/bench_serving_overhead.py \
  --dataset csv \
  --csv-path /mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv \
  --num-requests 7200 \
  --request-rate 20 \
  --dp 2 \
  --sp 8 \
  --tp 1 \
  --ep 16 \
  --max-num-seqs 256 \
  --gpu-memory-limit-gb 141 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 1000000 \
  --max-input-len 1000000 \
  --dummy-prefill \
  --ray-address 10.102.243.60:8776 \
  --master-address 10.102.243.60:29906 \
  --loop-count 16 \
  --model-path /mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3 \
  --routing-strategy LeastBatch \
  --scheduler-mode centralized \
  --segment-size 65536 \
  --sp-backend hao_basic \
  --cuda-graph-mode full \
  --fixed-sp-size 0 \
  --enable-dynamic-sp-size \
  --dynamic-sp-size-strategy long_short_sp8 \
  --long-request-sp-threshold 100000 \
  --long-request-sp-size 8 \
  --itl-log-path docs-dev/nanodeploy_original_issue001_2node_dp2sp8_r20_6min.jsonl \
  2>&1 | tee docs-dev/nanodeploy_original_issue001_2node_dp2sp8_r20_6min.log
```

这条命令对应参考 sh 的 original setting：

- `batch_size/max_num_seqs=256`；
- `segment_size=65536`；
- `scheduler_mode=centralized`；
- `gpu_memory_limit_gb=141`、`gpu_memory_utilization=0.9`；
- `max_model_len=max_input_len=1000000`；
- `loop_count=16`；
- `fixed_sp_size=0`；
- `sp_backend=hao_basic`；
- full CUDA Graph，不启用 eager；
- legacy Decode dynamic-SP path；
- `long_short_sp8`，长请求阈值 100,000，长请求使用 SP8；
- `LeastBatch` routing；
- 默认启用 non-uniform KV split。

它不会传入任何 LS scheduler 或 KV consolidation 参数，因此运行的是 NanoDeploy original 策略。

参考 sh 中模型默认路径是 `/mnt/nvme1n1/ml_research/models/deepseek-v3`；当前两节点集群使用的是等价且已验证存在的 `/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3`。

原始 driver 没有 `--duration-sec` 参数。这里用 `7200 = 20 req/s × 360 s` 表达 6 分钟 offered load；最后一个 Poisson 到达会在 360 秒附近，随后继续 drain。该 driver 使用参考 sh 原有的 256-request warmup。

该命令同样会使用 16 块 GPU；通过 Codex 启动前必须单独申请 GPU 提权。建议不要直接调用参考 sh 的 sweep wrapper，因为它按 600 秒硬编码请求数，会把 rate 20 转成 12,000 请求，而不是本次需要的 7,200 请求。

实时监测：

```bash
tail -f docs-dev/nanodeploy_original_issue001_2node_dp2sp8_r20_6min.log
```

正常完成时应在日志末尾看到 `Benchmark Results`，并生成对应 JSONL。中途终止的运行仍不能用于性能对比。
