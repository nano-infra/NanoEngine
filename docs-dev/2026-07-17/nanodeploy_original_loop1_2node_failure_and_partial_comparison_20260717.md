# Original NanoDeploy loop_count=1 两机复跑失败与三组局部对照

日期：2026-07-17

## 结论

按已完成的 Original NanoDeploy `loop_count=16` 两机命令，只把
`--loop-count` 改为 `1` 后，连续两次正式运行都因 head 节点全局 rank 0
的 CUDA illegal memory access / NCCL watchdog 退出：

- 第一次在正式阶段 107 秒、完成 1,295 / 7,200 时退出；
- 第二次在正式阶段 173 秒、完成 2,647 / 7,200 时退出；
- 两次故障前 `waiting_reqs=0`，故障时 KV cache 也未耗尽；
- 两次都没有生成 JSONL，因为原始 driver 只在全部请求完成后写 JSONL；
- 退出后 Ray 均恢复为 `0 / 16 GPU`，两节点保持 active。

因此本轮没有得到可用于完整吞吐、延迟或 drain 对比的第三组结果，不能把
partial run 外推成 Original NanoDeploy loop1 的最终性能。不过两次早期轨迹一致，
已经能做有限的交叉校验：60 秒完成数为 409 / 402，介于 Original loop16 的
661 和 LS future-KV loop1 的 130 之间；第二次 120 秒完成 1,499，同样介于
Original loop16 的 1,865 和 LS future-KV 的 882 之间。

这里的 `LS future-KV` 是 NanoDeploy 中的 LoongServe-style Decode scheduler-policy
baseline，不是原版 LoongServe 完整 serving stack。

## 实验配置

本轮复用 2026-07-17 Original loop16 成功复跑的参数和当前工作区代码状态，只改变：

```text
--loop-count 16 -> --loop-count 1
```

关键配置：

- dataset：Issue 1% CSV，固定 seed 0；
- requests / offered rate：7,200 / 20 req/s，Poisson arrival；
- topology：Attention `DP2 × SP8 × TP1`，FFN `EP16`；
- `gpu_memory_limit_gb=141`，每 rank 14,045 KV blocks；
- `max_num_seqs=256`，`max_num_recv_seqs=32`；
- full CUDA Graph：20 个 local graph + 60 个 SP graph；
- legacy dynamic SP：`long_short_sp8`，100k token 以上请求使用 SP8；
- `scheduler_mode=centralized`，`LeastBatch` routing；
- `enable_ls_decode_core_scheduler=False`，KV consolidation off；
- RPC server endpoint 沿用工作区已有的 1.024 GB 修改。

运行前 Ray 显示两节点 active、`0 / 16 GPU` 占用。完整命令：

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
  --loop-count 1 \
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
  --itl-log-path <attempt-specific.jsonl> \
  2>&1 | tee <attempt-specific.log>'
```

## 两次运行

| 指标 | Attempt 1 | Attempt 2 |
|---|---:|---:|
| warmup | 256 req / 26.89 s | 256 req / 25.78 s |
| 60 秒完成数 | 409 | 402 |
| 120 秒完成数 | 未到达 | 1,499 |
| 最后进度 | 1,295 @ 107 s | 2,647 @ 173 s |
| exit code | 1 | 1 |
| 首个 fatal rank | head global rank 0 | head global rank 0 |
| peak batch | 905 | 954 |
| peak KV util | 57.00% | 76.57% |
| min free blocks | 6,040 | 3,291 |
| max waiting requests | 0 | 0 |
| max total RPC metadata bytes/step | 3,655,640 | 3,844,740 |

Attempt 1 最后一个完整 step 为：

```text
itl=69.57 ms, batch=891, min_free_blocks=6249,
max_kv_util=55.51%, waiting_reqs=0
```

Attempt 2 最后一个完整 step 为：

```text
itl=61.34 ms, batch=814, min_free_blocks=10621,
max_kv_util=24.38%, waiting_reqs=0
```

Attempt 2 在运行期曾达到 76.57% KV 高水位，但在故障前已经回落到 24.38%。
因此两次故障都不是 KV cache 耗尽触发。RPC metadata 的单步总字节数也低于 4 MB，
远低于修改前的 256 MB server endpoint 容量；现有证据同样不支持 metadata buffer
容量不足。

两次首个可见 fatal 都是：

```text
[PG ID 3 PG GUID 11(mesh_attn_tp) Rank 0]
Process group watchdog thread terminated with exception:
CUDA error: an illegal memory access was encountered
```

watchdog 是异步观察到错误的位置，不能据此断言 NCCL collective 是最初写坏显存的
kernel。随后其他 ranks 的 TCPStore reset、Ray worker SYSTEM_ERROR 和 driver shutdown
报错都是 rank 0 abort 后的级联结果。

## 三组数据的有效对照边界

| elapsed / 结果 | Original loop16 | Original loop1 Attempt 1 / 2 | LS future-KV loop1 |
|---|---:|---:|---:|
| 60 s completed | 661 | 409 / 402 | 130 |
| 120 s completed | 1,865 | — / 1,499 | 882 |
| 180 s completed | 3,018 | — / 已在 173 s 失败 | 1,891 |
| 360 s completed | 6,495 | 无有效值 | 5,366 |
| final completed | 7,200 | 失败 / 失败 | 7,200 |
| full drain time | 411.06 s | 无有效值 | 566.04 s |

第二次 partial run 相对两个已完成运行：

- 60 秒 402 个，比 Original loop16 少 39.2%，比 LS future-KV 多 209.2%；
- 120 秒 1,499 个，比 Original loop16 少 19.6%，比 LS future-KV 多 69.9%。

这些差值只描述故障前的累计完成轨迹，不能报告为稳态吞吐、goodput、TPOT 或最终
speedup。尤其 60 秒完成数强烈受首批请求延迟和输出长度分布影响。

局部轨迹仍支持两个判断：

1. Original 从 loop16 改为 loop1 后，每 token 支付一次控制/RPC/同步成本，早期完成
   进度确实下降；
2. 在故障前，Original loop1 仍明显快于 LS future-KV loop1，说明此前两组差距不能
   只归因于 loop count；legacy SP placement、receiver cap、KV ownership、multi-master
   planning 和 consolidation 仍是独立变量。

## 与历史 Original 故障的关系

2026-07-16 Original loop16 失败运行在 111 秒、完成 1,648 个请求时由 rank 5 报
illegal memory；次日同命令又成功完成 7,200 个请求。本轮 loop1 两次分别在 107 秒和
173 秒失败，且都由 rank 0 首报。

这说明故障不是固定请求序号或固定 wall-clock 点必现，但同一类异步 CUDA/RDMA/NCCL/
CUDA Graph 问题能够跨 loop count 和不同运行时序重现。Original loop16 的一次成功不能
视为稳定性问题已消失。

## 产物

- `nanodeploy_original_loop1_issue001_2node_dp2sp8_r20_6min_20260717.log`：Attempt 1，
  1.6 MB；
- `nanodeploy_original_loop1_issue001_2node_dp2sp8_r20_6min_rerun_20260717.log`：Attempt 2，
  2.4 MB；
- `nanodeploy_original_loop1_2node_failure_summary_20260717.json`：结构化摘要；
- 本报告。

两个 `.log` 受仓库 `*.log` ignore 规则保护，只保留在当前工作区；结构化摘要和本报告
纳入 Git。两次都没有完成，因此没有生成请求级 JSONL。
