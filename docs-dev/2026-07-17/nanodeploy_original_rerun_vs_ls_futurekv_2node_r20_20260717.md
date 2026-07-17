# Original NanoDeploy 同配置复跑与 LS future-KV 性能对比

日期：2026-07-17

## 结论

按 2026-07-16 失败实验的命令行配置复跑后，本次 Original NanoDeploy 正常完成 `7,200 / 7,200` 个请求并退出，没有重现 rank 5 illegal memory access / NCCL watchdog：

- 360 秒完成 `6,495 / 7,200`，完成窗口速率 18.04 req/s；
- 总时间 411.06 秒，output throughput 10,490.01 token/s；
- average E2E 29.42 秒，average TPOT 47.93 ms/token；
- queue average / p99 / max 为 0.63 / 10.56 / 16.54 ms，goodput 100%；
- peak KV util 56.55%，最紧 rank 仍余 6,103 blocks，全程 `waiting_reqs=0`；
- Traceback、CUDA illegal memory、NCCL fatal、OOM 均为 0；退出后 Ray 为 `0 / 16 GPU` in use。

和同日 LS future-KV 141 GiB 运行相比，Original NanoDeploy 在 360 秒多完成 1,129 个请求（+21.0%），完整 drain 快 27.4%，output throughput 高 37.7%。但这不是只切换一个 feature 的严格 A/B：两边的 loop count、SP 调度方式、receiver cap、KV ownership 和 consolidation 均不同。

本次成功也不能证明昨日故障已经消失。昨日运行在 111 秒、完成 1,648 个请求时失败；本次在同一时刻完成 1,655 个并继续运行。两条轨迹在故障点前非常接近，因此更像运行时序相关的 GPU/RDMA/NCCL 或 graph path 问题，而不是请求排队或 KV 容量 cliff。

## 复跑配置与可复现性边界

主要配置：

- dataset：Issue 1% CSV，固定 seed 0；
- requests / offered rate：7,200 / 20 req/s，Poisson arrival；
- topology：Attention `DP2 × SP8 × TP1`，FFN `EP16`；
- `gpu_memory_limit_gb=141`，allocator 配置 14,045 KV blocks/rank；
- `max_num_seqs=256`，`max_num_recv_seqs=32`；
- `loop_count=16`；
- full CUDA Graph，20 个 local graph + 60 个 SP graph；
- legacy dynamic SP：`long_short_sp8`，100k token 以上请求使用 SP8；
- `enable_ls_decode_core_scheduler=False`，不启用 LS future-KV admission 和 KV consolidation。

完整命令：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  NANODEPLOY_LOG_DECODE_A2A_MASKS=0 \
  NANODEPLOY_LOG_DECODE_STEP_DETAIL=0 \
  NANODEPLOY_LOG_MODEL_FORWARD_TIMING=0 \
  RAY_DEDUP_LOGS=1 PYTHONUNBUFFERED=1 \
  bash -o pipefail -c '
python scripts/issue003/bench_serving_overhead.py \
  --dataset csv \
  --csv-path /mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv \
  --num-requests 7200 --request-rate 20 \
  --dp 2 --sp 8 --tp 1 --ep 16 \
  --max-num-seqs 256 \
  --gpu-memory-limit-gb 141 --gpu-memory-utilization 0.9 \
  --max-model-len 1000000 --max-input-len 1000000 \
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
  --itl-log-path docs-dev/2026-07-17/nanodeploy_original_issue001_2node_dp2sp8_r20_6min_rerun_20260717.jsonl \
  2>&1 | tee docs-dev/2026-07-17/nanodeploy_original_issue001_2node_dp2sp8_r20_6min_rerun_20260717.log'
```

### 与昨日代码状态的一处差异

命令行配置相同，但复跑前工作区已有一项未提交的 Python 修改：

```diff
- RPCServerEndpoint(8 * 32_000_000, ...)
+ RPCServerEndpoint(32 * 32_000_000, ...)
```

即 server 侧每个 rank 的序列元数据 RDMA buffer 从 256,000,000 增到 1,024,000,000 bytes。该修改不是本报告提交的一部分，并且它不是 NCCL KV-cache 搬运 buffer。

从日志看，本次 `dlslime_send_seqs_BYTES` 的单次最大值只有 2,702,868 bytes，而且这是 16 个 endpoint 的总和；任一 endpoint 的有效 payload 因而必然小于 2.71 MB，远低于原 256 MB 上限。昨日失败运行的最大总 payload 也只有 2,680,480 bytes。因此现有证据不支持“旧 RPC buffer 容量不足导致本次 illegal memory”；若要获得字节级同代码复现，需要把这项独立变量纳入后续 A/B。

## 六分钟进度

`tqdm` elapsed 只有整秒精度。下表取各整分钟桶的最后完成值：

| elapsed | completed | completed rate |
|---:|---:|---:|
| 60 s | 661 | 11.02 req/s |
| 120 s | 1,865 | 15.54 req/s |
| 180 s | 3,018 | 16.77 req/s |
| 240 s | 4,205 | 17.52 req/s |
| 300 s | 5,304 | 17.68 req/s |
| 360 s | 6,495 | 18.04 req/s |

seed 0 对应的最后一个 arrival 为 355.941 秒。最后 arrival 后 drain 约 55.12 秒；相对 360 秒目标发送窗口，总运行超出 51.06 秒。

## 与 LS future-KV 141 GiB 对比

LS 数据来自 [LS Decode future-KV 两机 rate=20 六分钟验收](ls_decode_future_kv_2node_r20_141gb_result_20260717.md)。两个运行处理了相同的 41,285,449 input tokens 和 4,312,075 output tokens。

| 指标 | Original NanoDeploy | LS future-KV | Original 相对 LS |
|---|---:|---:|---:|
| 360 秒完成数 | 6,495 | 5,366 | +1,129 / +21.0% |
| 360 秒完成速率 | 18.04 req/s | 14.91 req/s | +21.0% |
| 完整 drain 时间 | 411.06 s | 566.04 s | -27.4% |
| 完整运行请求吞吐 | 17.52 req/s | 12.72 req/s | +37.7% |
| output throughput | 10,490.01 token/s | 7,618.00 token/s | +37.7% |
| average TTFT | 1.45 ms | 509.44 ms | -99.7% |
| average E2E | 29.42 s | 91.18 s | -67.7% |
| TPOT avg / p95 | 47.93 / 52.16 ms | 143.49 / 165.59 ms | avg -66.6% |
| queue avg / max | 0.63 ms / 16.54 ms | 509.19 ms / 393.53 s | 无长尾 pending |
| goodput（<100 ms） | 100.00% | 2.39% | +97.61 pp |
| max total batch size | 648 | 1,942 | -66.6% |
| peak KV util / min free | 56.55% / 6,103 | 99.65% / 49 | 显著更大余量 |
| pending / waiting peak | 0 req | 7 batches / 16 req | Original 无 admission backlog |

Original 更快的主要原因不是“没有 future-KV 计算”这一项，而是整条执行路径不同：

1. `loop_count=16` 一次控制/RPC 循环推进 16 个 token；LS 测试使用 `loop_count=1`，分布式调度和同步成本每 token 支付一次。
2. Original 的超长请求固定使用 SP8，把 KV 分散到 8 个 SP rank；LS 的 group-owned KV placement 会在少数 owner 上形成更高水位。
3. Original 的并发稳定在较低水平，max batch 648；LS 在 offered rate 高于完成速率后积累到 1,942 个 in-flight，step latency 和 E2E 随之上升。
4. LS 还启用了 multi-master planning、future-KV admission 和 consolidation。它们主要解决容量安全和可恢复性，不保证在当前实现下提高吞吐。
5. `max_num_recv_seqs` 也不同（32 vs 128），所以本表是两套系统配置的端到端比较，不是 future-KV guard 的单变量开销。

## 对昨日失败的判断

| 指标 | 2026-07-16 失败运行 | 2026-07-17 成功复跑 |
|---|---:|---:|
| 60 秒完成数 | 665 | 661 |
| 110 秒完成数 | 1,647 | 1,633 |
| 111 秒完成数 | 1,648 后失败 | 1,655，继续运行 |
| 故障前 / 全程 peak KV util | 45.08% | 56.55% |
| min free blocks | 7,713 | 6,103 |
| waiting peak | 0 | 0 |
| 结果 | rank 5 illegal memory + NCCL watchdog | 7,200/7,200，正常退出 |

因此：

- 昨日故障不是“系统没有处理好排队请求”：故障前 `waiting_reqs` 始终为 0，KV 也未接近耗尽；
- 成功复跑已经排除“该确定性请求序列在 111 秒必然失败”，但没有排除异步竞态、CUDA Graph shape/path、通信 buffer 生命周期或 NCCL/RDMA 时序问题；
- 单次成功不足以给出稳定性结论。至少还需要 3 次相同运行，并把 RPC server buffer 256 MB / 1.024 GB 做成显式 A/B；如果只在某一 graph/batch shape 失败，再针对该窗口增加 shape、通信字节数和 buffer lifetime telemetry。

## 后续建议

1. 保留 Original 作为性能上界和稳定性回归项，连续复跑 3 次；不能因为本次成功就关闭 illegal-memory 问题。
2. 做真正的 LS 开关 A/B：固定 arrival、loop count、receiver cap、dynamic-SP policy 和 consolidation，仅切换 LS core / future-KV admission。当前对比适合看端到端取舍，不适合归因单个 feature。
3. 优先把 Original 的 SP8 KV 分散能力带入 LS owner-aware placement。当前 99.65% vs 56.55% 的水位差，比 future envelope 计算本身更能解释 LS 的性能 cliff。
4. 将 RPC endpoint buffer 改为显式配置并记录有效 payload 高水位；现状用源码常量改变 16 个大内存区，既难复现，也无法证明容量需求。

## 产物

- `docs-dev/2026-07-17/nanodeploy_original_issue001_2node_dp2sp8_r20_6min_rerun_20260717.log`；
- `docs-dev/2026-07-17/nanodeploy_original_issue001_2node_dp2sp8_r20_6min_rerun_20260717.jsonl`（7,200 条）；
- 本报告。
