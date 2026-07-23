# LoongServe-style `loop_count=16` 单机高 backlog 5 分钟长跑

时间：2026-07-23 03:51--03:56 UTC

## 结论

把 request rate 从 4 req/s 提高到 50 req/s、把 inflight cap 从 64
提高到 1024 后，测试稳定形成了 128 running + 832 waiting 的高 backlog
平台，并持续到 300 秒发压结束。

- 103 个 Decode step 中有 81 个 step 的 `waiting_reqs=832`；
- 高 backlog 下单轮外层 scheduler 平均 11.404 ms，是低负载稳定段
  1.306 ms 的 8.73 倍；
- 其中内部 C++ planning 平均 9.954 ms，8.499 ms 花在 admission，
  `nested.admission_scan` 为 7.832 ms；
- 尽管调度开销显著放大，`waiting=832` 时 Decode ITL 平均
  210.522 ms/token，仅比低负载稳定段 208.356 ms/token 高 1.04%；
- 饱和平台的完成吞吐约 18.93 req/s、340.8 output token/s；
- 代价主要体现在排队延迟：queueing median 43.821 秒、p99 44.185 秒，
  E2E median 53.885 秒。

所以当前 waiting 较多时的主要问题不是 GPU cadence 明显退化，而是 scheduler
对 waiting candidate 的线性扫描和 admission planner 调用被放大。scheduler
绝对耗时仍只占一轮 16-token Decode wall time 的约 0.36%，但请求端已经被
约 44 秒排队时间主导。

本轮显式开启了 scheduler phase timing 诊断；它默认关闭，并会执行额外的
内部计时与日志记录。因此 11.404 ms 适合作为诊断开启时的实测上界，和未开启
诊断的低负载轮对比时需保留这一口径差异。

## 配置

- Ray：`10.102.206.14:7789`
- 单节点 8×NVIDIA H200
- topology：Attention DP1×SP8×TP1，FFN EP8
- LoongServe-style Decode-only、`hao_basic`、eager
- `dummy_prefill=true`、`dummy_weight=true`
- prompt length：63
- `max_tokens=18`
- `loop_count=16`
- request rate：50 req/s
- burstiness：1.0
- inflight cap：1024
- `max_num_seqs=64`
- `max_num_recv_seqs=128`
- 发送时间：300 秒
- drain timeout：300 秒
- scheduler 诊断：`NANODEPLOY_LS_SCHEDULER_PHASE_TIMING=1`

运行前清除了四个 HTTP(S) proxy 变量。正式命令：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  NANODEPLOY_LS_SCHEDULER_PHASE_TIMING=1 \
  PYTHONUNBUFFERED=1 \
  python scripts/bench_ls_decode_longrun_8gpu.py \
  --ray-address 10.102.206.14:7789 \
  --master-address 10.102.206.14:29906 \
  --duration-sec 300 \
  --drain-timeout-sec 300 \
  --request-rate 50 \
  --burstiness 1.0 \
  --max-inflight-requests 1024 \
  --prompt-len 63 \
  --max-tokens 18 \
  --max-num-seqs 64 \
  --max-num-recv-seqs 128 \
  --attention-dp 1 \
  --loop-count 16 \
  --steady-start-sec 60 \
  --progress-interval-sec 30 \
  --ls-max-num-ooe 8 \
  --output-json \
    docs-dev/2026-07-23/ls_loop16_backlog_r50_longrun_8gpu_5min_20260723.json \
  --completion-jsonl \
    docs-dev/2026-07-23/ls_loop16_backlog_r50_longrun_8gpu_5min_20260723.jsonl \
  --manifest-json \
    docs-dev/2026-07-23/ls_loop16_backlog_r50_longrun_8gpu_5min_20260723.manifest.json
```

此前先做了 rate=20 req/s 的短校准；它只形成约 100--150 waiting，未达到
“大量 backlog”的目标，因此中止校准并改用 rate=50。rate=20 的中止产物不作为
正式结果。

## 完成性和负载形态

```text
manifest status:               success
process exit:                        0
total elapsed:                 351.464 s
send duration:                 300.000 s
drain duration:                 50.029 s
drain completed:                   true
accepted/completed:            6401/6401
unfinished requests:                   0
max inflight observed:              1024
backpressure dropped arrivals:      8217
decode steps:                        103
maintenance steps:                    1
```

总 arrival 数为 14,618；其中 8,217 个在 inflight cap 已满时、进入 engine
之前被负载发生器丢弃，占 56.21%。所有 6,401 个 accepted request 都完成，
没有 engine 内请求失败。

在发压 31 秒时已达到：

```text
accepted:                    1345
completed:                    385
inflight:                     960
running batch:                128
waiting:                      832
```

之后 81 个 Decode step 保持 `waiting=832`。300 秒停止发新请求时仍有 960
inflight，drain 用 50.03 秒全部清空。

固定 K 和输出正确性：

```text
scheduler execution K histogram: {16: 103}
completion records:                 6401
completion output length:      {18: 6401}
```

## 每轮调度开销

### 和低负载 5 分钟轮对比

低负载对照是同一天的 rate=4、inflight cap=64 长跑；其 90 个 Decode step
全部 `waiting=0`，稳定段从 elapsed 60 秒开始。

| 指标 | 低负载稳定段 | 高负载 `waiting=832` | 变化 |
|---|---:|---:|---:|
| 样本数 | 75 | 81 | — |
| scheduler mean | 1.306 ms | 11.404 ms | 8.73× |
| scheduler median | 1.287 ms | 11.367 ms | 8.83× |
| scheduler p99 | 1.983 ms | 12.180 ms | 6.14× |
| post-scheduler mean | 0.183 ms | 0.580 ms | 3.17× |
| scheduler + post mean | 1.489 ms | 11.984 ms | 8.05× |
| 每个 K=16 token 的 scheduler | 0.082 ms | 0.713 ms | 8.73× |
| 每 token 含 post | 0.093 ms | 0.749 ms | 8.05× |
| Decode ITL mean | 208.356 ms | 210.522 ms | +1.04% |
| Decode ITL p99 | 211.430 ms | 216.054 ms | +2.19% |

高 backlog 的 scheduler + post 平均为 11.984 ms；以该组
`16 × 210.522 ms` 的一轮 Decode wall time 为分母，占约 0.36%。

### waiting 分桶

过渡区样本很少，主要用于展示随候选数量上升的趋势；稳定平台是最后一行。

| `waiting_reqs` | step 数 | scheduler mean | internal planning mean | Decode ITL mean |
|---:|---:|---:|---:|---:|
| 64 | 1 | 8.538 ms | 7.249 ms | 209.315 ms |
| 128--511 | 9 | 8.600 ms | 7.318 ms | 217.080 ms |
| 512--767 | 7 | 10.560 ms | 9.169 ms | 215.584 ms |
| ≥768 | 82 | 11.414 ms | 9.962 ms | 210.464 ms |
| 832（精确平台） | 81 | 11.404 ms | 9.954 ms | 210.522 ms |

`waiting` 和 scheduler latency 在全轮样本上的 Pearson correlation 为
0.822；`waiting` 和内部 planning latency 的 correlation 为 0.829。
过渡区 ITL 包含冷启动和 drain，不能据少量样本解释成 waiting 越大 ITL
越低。

### `waiting=832` 内部阶段

以下为 81 个稳定高 backlog step 的均值：

| 顶层 planning 阶段 | 平均耗时 |
|---|---:|
| snapshot copy | 0.252 ms |
| mandatory safety | 0.024 ms |
| admission | 8.499 ms |
| decode plan prepare | 1.146 ms |
| publication | 0.033 ms |
| internal planning total | 9.954 ms |

嵌套热点：

```text
waiting candidates scanned/step:        896
nested.admission_scan:                7.832 ms
admission_plan calls/step:                65
nested.admission_plan:                5.076 ms
future_kv_pool calls/step:               194
nested.future_kv_pool:                5.722 ms
initial_placement calls/step:             65
nested.initial_placement:             2.817 ms
```

`nested.*` 是 inclusive 计时，会彼此和顶层 admission 重叠，不能相加。
扫描摊到每个候选约 8.74 μs。结果表明每次单独 planner 调用不算很慢，主要是
waiting/OOE 把扫描候选数以及 exact/future-KV planner 调用次数放大。

日志中的 `waiting_reqs=832` 是 step 输出时的 waiting gauge，而 phase
counter 是调度快照中实际扫描的候选数，因此二者采样点不同，不能直接用
`896 - 832` 解释为丢失请求。

## 吞吐与排队延迟

正式 summary 以整个 351.46 秒（含启动平台和 drain）计算：

```text
request throughput:             18.212 req/s
output throughput:             327.823 token/s
```

从 completion JSONL 派生，elapsed 60--300 秒稳定饱和区完成 4,544 个请求：

```text
saturated completion rate:      18.933 req/s
saturated output throughput:   340.800 token/s
```

50 req/s 的输入显著高于约 19 req/s 的服务能力，所以 cap 很快打满。GPU
execution cadence 基本保持，但用户可见延迟由 waiting 决定：

| 指标 | mean | median | p99 | max |
|---|---:|---:|---:|---:|
| queueing/TTFT | 39.841 s | 43.821 s | 44.185 s | 44.192 s |
| E2E latency | 49.984 s | 53.885 s | 54.391 s | 54.406 s |
| request avg ITL | 211.185 ms | 210.031 ms | 249.209 ms | 352.556 ms |

这里使用 eager 和 dummy weight；吞吐和 GPU 时间只用于本次改动的相对回归，
不作为生产权重的绝对性能结论。

## 错误和资源回收

日志中没有 Python traceback、scheduler fatal、inconsistent loop K、CUDA
error、NCCL error、OOM 或 drain timeout。

测试结束后：

- Ray：`0/192 CPU`、`0/8 GPU`、无 pending demand；
- 8 张 H200：全部 `0 MiB / 0%`。

## 产物

- `ls_loop16_backlog_r50_longrun_8gpu_5min_20260723.json`
- `ls_loop16_backlog_r50_longrun_8gpu_5min_20260723.manifest.json`
- `ls_loop16_backlog_r50_longrun_8gpu_5min_20260723.jsonl`
- `ls_loop16_backlog_r50_longrun_8gpu_5min_20260723.log`（本地日志，git ignore）
