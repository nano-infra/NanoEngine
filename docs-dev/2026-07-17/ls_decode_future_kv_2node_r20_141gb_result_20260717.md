# LS Decode future-KV 两机 rate=20 六分钟验收

日期：2026-07-17

## 结论

future-KV admission 默认开启后，`141 GiB`、两节点 16 H200、rate 20 的 Issue 1% workload 首次自然完成全部 7,200 个请求并正常退出：

- 360 秒时完成 `5,366 / 7,200`，旧 141 GiB owner-aware 基线为 `5,216 / 7,200`；
- 总运行时间 566.04 秒，最后一个请求在 354.617 秒到达，随后用约 211.42 秒完成 drain；
- planner failure、preemption、recovery、admission rollback 均为 0；
- pending 峰值从旧基线的 `71 batches / 185 requests` 降为 `7 / 16`；
- 3 次 KV consolidation 均成功，进程 exit code 为 0，没有 NCCL/CUDA fatal；
- 最高 KV 利用率 99.65%，最紧 rank 仍余 49 blocks。

这次结果说明 future-KV guard 已把 141 GiB 轨迹中的 admission backlog 控制在很小范围，并避免了旧运行在 drain 期失败。不过它还不能证明 140 GiB cliff 和 NCCL 运行期显存预留已经普遍解决；需要继续跑同配置 140 GiB 对照。

本次还暴露出明确的下一处调度问题：7 个 no-fit 原子 batch 仍然每个 decode step 重试，单 batch 最多尝试 4,406 次；其中 7 个超长 prompt 连带拖住 9 个同 batch 的短请求。LoongServe 的未来 KV 估计已经接入，但 NanoDeploy 仍未具备 LoongServe 的逐请求 admission 粒度和 capacity-epoch/backoff。

## 配置

- 代码：`b3415a7 feat(scheduler): add future KV admission guard`；
- Ray：`10.102.243.60:8776`，节点 `10.102.243.60`、`10.102.206.14`；
- topology：Attention `DP2 × SP8 × TP1`，FFN `EP16`；
- model：DeepSeek-V3，dummy prefill / dummy weight；
- dataset：Issue 1% CSV，seed 0，Poisson rate 20；
- requests：7,200，目标发送窗口 360 秒，实际最后 arrival 354.6172 秒；
- `gpu_memory_limit_gb=141`，每 rank 14,044 KV blocks，block size 64；
- `max_num_seqs=256`，`max_num_recv_seqs=128`；
- `ls_decode_batch_per_master=128`，initial KV DoP 自动选择；
- `ls_decode_enable_future_kv_admission=True`；
- consolidation `execute`，`stable/cooldown/check=2/2/1`；
- full CUDA Graph，开启 actual-forward CUDA event timing。

完整命令：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  NANODEPLOY_LOG_DECODE_A2A_MASKS=0 \
  NANODEPLOY_LOG_DECODE_STEP_DETAIL=0 \
  NANODEPLOY_LOG_MODEL_FORWARD_TIMING=1 \
  RAY_DEDUP_LOGS=1 PYTHONUNBUFFERED=1 \
  bash -o pipefail -c '
python scripts/bench_ls_decode_serving.py \
  --attention-dp 2 --attention-sp 8 \
  --ray-address 10.102.243.60:8776 \
  --master-address 10.102.243.60:29906 \
  --model-path /mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3 \
  --csv-path /mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv \
  --request-rate 20 --duration-sec 360 --num-requests 7200 \
  --burstiness 1 --seed 0 \
  --warmup-requests 32 --warmup-prompt-len 512 --warmup-max-tokens 8 \
  --max-num-seqs 256 --max-num-recv-seqs 128 \
  --max-model-len 1000000 --max-input-len 1000000 \
  --max-num-batched-tokens 1024000 \
  --gpu-memory-limit-gb 141 --gpu-memory-utilization 0.90 \
  --segment-size 65536 --routing-strategy LeastBatch \
  --cuda-graph-mode full \
  --ls-initial-kv-dop 0 --ls-batch-per-master 128 \
  --ls-kv-consolidation-mode execute \
  --ls-kv-consolidation-candidate-util 0.50 \
  --ls-kv-consolidation-target-high-watermark 0.80 \
  --ls-kv-consolidation-stable-steps 2 \
  --ls-kv-consolidation-cooldown-steps 2 \
  --ls-kv-consolidation-check-interval-steps 1 \
  --ls-kv-consolidation-max-source-blocks-per-event 128 \
  --ls-kv-consolidation-migration-chunk-tokens 64 \
  --verbose-nanodeploy-logs \
  --output-jsonl docs-dev/2026-07-17/ls_style_issue001_2node_dp2sp8_r20_futurekv_141gb_6min_20260717.jsonl \
  2>&1 | tee docs-dev/2026-07-17/ls_style_issue001_2node_dp2sp8_r20_futurekv_141gb_6min_20260717.log'
```

## 六分钟进度

`tqdm` elapsed 只有整秒精度。下表取每个整分钟桶的最后完成值：

| elapsed | completed | 已完成请求平均延迟 |
|---:|---:|---:|
| 60 s | 130 | 38.91 s |
| 120 s | 882 | 59.74 s |
| 180 s | 1,891 | 71.86 s |
| 240 s | 3,011 | 79.17 s |
| 300 s | 4,295 | 83.39 s |
| 360 s | 5,366 | 85.57 s |

360 秒完成比例为 74.53%，按该窗口计算的 completed-request rate 为 14.91 req/s。offered rate 20 仍高于完成速率，所以发送结束后继续 drain 是正常现象。

## 与旧 141 GiB 基线对比

旧基线来自 [owner-aware 两机复跑与容量鲁棒性复盘](../2026-07-16/ls_decode_owneraware_2node_capacity_postmortem_20260716.md)。配置、请求序列和 seed 相同，但分布式执行时序仍可能导致 placement 轨迹分叉，因此不能把每个差值严格视为单变量因果。

| 指标 | 旧 141 GiB | future-KV 141 GiB |
|---|---:|---:|
| 360 秒完成数 | 5,216 | 5,366 |
| 360 秒完成速率 | 14.49 req/s | 14.91 req/s |
| pending 峰值（batch / request） | 71 / 185 | 7 / 16 |
| pending 最大 age / attempts | 4,089 / 4,090 | 4,404 / 4,405 |
| peak KV util / min free blocks | 99.45% / 77 | 99.65% / 49 |
| 发送期 actual forward p50 / p95 | 122.76 / 177.01 ms | 127.48 / 146.65 ms |
| 发送期 scheduler overhead mean / p95 | 4.94 / 5.49 ms | 4.46 / 6.41 ms |
| 发送期 planner p50 / p95 / max | 2.18 / 6.10 / 17.70 ms | 2.43 / 2.72 / 3.85 ms |
| planner failure / preemption | 0 / 0 | 0 / 0 |
| consolidation | 成功 8 次后 fatal | 成功 3 次，正常退出 |
| 最终完成 | 7,111 / 7,200 | 7,200 / 7,200 |

future-KV 版本的 360 秒完成数增加 150（约 +2.9%），pending requests 峰值减少约 91.4%。planner p95 和 max 明显收敛；Python 侧完整 scheduler overhead p95 略高，说明 future envelope 计算本身仍有成本，但没有随 pending 数量失控。

## 完整 drain 指标

benchmark 输出：

- total time：566.04 秒；
- requests：7,200 sent / 7,200 completed；
- input / output tokens：41,285,449 / 4,312,075；
- generation throughput：7,618 output tokens/s；
- full-run request throughput：12.72 req/s；
- average TTFT：509.44 ms；
- average E2E latency：91.18 s；
- TPOT（不含 queue）avg / p50 / p95 / p99：143.49 / 146.60 / 165.59 / 180.63 ms；
- ITL（含 decode queue）avg / p50 / p95 / p99：144.67 / 147.94 / 161.00 / 162.26 ms；
- TPOT-with-queue < 100 ms 的 goodput：172 / 7,200，2.39%。

运行期状态高水位：

- max total batch size：1,942；
- max KV util：99.65%，min free blocks：49；
- max pending：7 batches / 16 requests；
- full-run planner latency p50 / p95 / max：1.62 / 2.70 / 3.85 ms；
- admission rollback：0；
- `LS-Decode-Core plan failed`、`Preemption happens`、非空 preemption telemetry：均为 0；
- Traceback、RuntimeError、CUDA error、OOM、segfault：均为 0。

3 次 consolidation 分别耗时 76.34、74.02、64.86 ms，总 maintenance stall 约 215.22 ms，全部成功。

## Pending 尾部行为

只有 16 个请求的 queueing time 超过 10 ms，但它们形成非常长的尾部：

- queue p99：1.93 ms；
- queue p99.9：210.91 秒；
- max queue：393.53 秒；
- 13 个请求排队超过 60 秒；
- benchmark 报告的 average queue 509.19 ms，主要由这 16 个 outliers 拉高。

最终需要延迟 admission 的 7 个逻辑 batch 如下：

| sequences | admission attempts | initial KV DoP | 最长 queue |
|---|---:|---:|---:|
| `[6966, 6967, 6968]` | 346 | 1 | 41.88 s |
| `[1482]` | 4,361 | 2 | 386.47 s |
| `[1530, 1531, 1532, 1533]` | 4,406 | 2 | 393.53 s |
| `[5282, 5283]` | 2,026 | 2 | 210.91 s |
| `[5420, 5421]` | 2,027 | 2 | 213.76 s |
| `[5770, 5771]` | 1,803 | 2 | 197.33 s |
| `[6455, 6456]` | 1,391 | 2 | 164.72 s |

这些 batch 中共有 7 个 840k--972k token 的 Issue 长 prompt，其余 9 个只是同一原子 batch 内的短请求。future-KV guard 正确地没有在高水位时把长请求继续塞入系统，但原子 batch 语义让短请求一同等待。所有 7 个 batch 最终都在容量释放后成功 merge admission，没有进入 recovery。

## 下一步

1. **先跑 140 GiB 同配置对照。** 这是判断原 140 GiB cliff 是否真正消失的直接验收；本次 141 GiB 成功不能代替该测试。
2. **把 pending admission 细化到 request 粒度。** logical batch future-fit 失败时，至少把超长请求隔离为 child batch，让同 batch 的短请求继续参与 admission；RPC/model execution 仍可在 admission 后重新批处理。
3. **加入 capacity epoch、失败 fingerprint 和 probe budget。** 资源状态没有产生可影响可行性的变化时，不要每 step 重做同一完整规划；aging 只负责公平唤醒，不应制造 4,406 次无效尝试。
4. **补结构化 future-KV telemetry。** 记录 projected peak、available token slots、目标 DP/group 和拒绝原因，避免继续把 future envelope、rank ownership、receiver cap 等都折叠成 `atomic_admission_no_fit_count`。
5. **继续处理物理显存 headroom。** 本次 3 个 consolidation path 成功是好信号，但 peer/channel 覆盖少于旧运行，仍需 NCCL/P2P warmup 和最终 KV sizing 闭环，不能据此宣布运行期 allocation 问题根治。

## 产物

- `docs-dev/2026-07-17/ls_style_issue001_2node_dp2sp8_r20_futurekv_141gb_6min_20260717.log`（约 59 MiB）；
- `docs-dev/2026-07-17/ls_style_issue001_2node_dp2sp8_r20_futurekv_141gb_6min_20260717.jsonl`（约 63 MiB，7,200 条）；
- 本报告。
