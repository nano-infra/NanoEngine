# LS-Decode-Core T=128、16 卡 rate=20 六分钟测试结果

日期：2026-07-16

## 1. 结论

本次测试完整跑完了 360 秒 offered-load window，7,200 个请求均按计划提交，但没有完成 drain，因进入确定的 recovery 活锁，在正式运行 549 秒、完成 3,708 个请求后主动停止。

将 `ls_decode_batch_per_master` 从 8 提高到 128 有明显收益，但不是根因修复：

- 首次 `append capacity cannot cover remaining requests` 从旧 T=8 运行的约 20 秒推迟到约 250 秒；
- 首分钟完成数从 T=8 的 24 提高到 126；
- T=128 在首次失败前大部分请求保持 SP1，owner 扩散显著变慢；
- 但 T=128 同时参与 group merge 和 master chunk 决策。当一个 127-request 小 group 合入 3 个新请求后，它与已有 313-request group 合并，当前 source-greedy 在合并后的 443 个请求上产生 receiver-capacity 假阴性，随后仍进入抢占/立即恢复循环。

因此，T=128 是有效的缓解参数，不是 receiver planner 和 recovery liveness bug 的修复。

## 2. 测试配置与命令

- 2 nodes，16 H200 GPUs；
- Attention `DP2 × SP8 × TP1`，FFN `EP16`；
- Issue 1% CSV，7,200 requests，`20 req/s`，360 秒发送窗口；
- `max_num_seqs=256`，`max_num_recv_seqs=128`；
- 每 rank 14,044 个 KV blocks，block size 64；
- `ls_decode_initial_kv_dop=0`；
- `ls_decode_batch_per_master=128`；
- consolidation `execute`，`stable/cooldown/check=2/2/1`；
- actual-forward CUDA-event timing 开启；
- 关闭逐 step 的 receiver/KV 矩阵 dump，保留 LS iteration 和 compact decode telemetry。

完整启动命令：

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
  --output-jsonl docs-dev/2026-07-16/ls_style_issue001_2node_dp2sp8_r20_t128_6min_20260716.jsonl \
  2>&1 | tee docs-dev/2026-07-16/ls_style_issue001_2node_dp2sp8_r20_t128_6min_20260716.log'
```

## 3. 运行结果

时间线：

| 事件 | 时间/进度 |
|---|---|
| 200 张 CUDA Graph 捕获完成 | 08:56:50 UTC |
| warmup | 32/32，2.34 秒 |
| 正式 workload 开始 | 08:57:06 UTC |
| 1 分钟 | 126/7200，Avg Latency 39.08 秒 |
| 首次 planner failure | 09:01:17 UTC，约 04:10，约 3169/7200 |
| 360 秒发送窗口结束 | 已提交 7200；已完成 3414/7200 |
| 主动停止活锁 drain | 09:06:17 UTC，09:09 elapsed，3708/7200 |

360 秒时的完成口径：

- 完成 `3414 / 7200 = 47.42%`；
- completed-request throughput 为 `3414 / 360 = 9.48 req/s`；
- 已完成请求的 tqdm Avg Latency 为 82.72 秒；
- 3414 是 05:28 时的最后一次完成更新，下一次完成发生在 06:01，因此 06:00 的完成数仍精确为 3414。

停止时：

- 完成 3,708/7,200，Avg Latency 95.03 秒；
- planner failure/preemption 各 1,934 次，涉及 1,235 个 unique sequences；
- group 2 failure 779 次，group 3 failure 1,155 次；
- pending 峰值为 3,142 requests / 1,655 batches；
- 无 `Traceback`、`RuntimeError`、CUDA error、OOM 或 segfault；
- consolidation candidate/action 均为 0；
- 由于主动停止且未完成 drain，没有 `Benchmark Results`，目标 JSONL 也未生成。因此这不是一个可用于最终吞吐/尾延迟对比的 completed benchmark。

## 4. 首次失败的精确原因

首次失败前最后一个成功 step：

```text
total_batch_size   = 1941
max_sp_batch_size  = 128
min_free_blocks    = 415
max_kv_util_pct    = 97.05%
sp_size_hist       = {SP1: 1613, SP2: 328}
```

当时 DP1 的主要形状为：

- group 3：313 requests，masters `[0, 1, 7]`，master batches `[128, 128, 57]`；
- group 7：127 requests，master rank 2；
- 新 batch `seq_id=5164..5166` 的 KV owner 均为 rank 2，合入后 group 7 从 127 增至 130；
- T=128 的 group merge 逻辑随后把 group 7 与 group 3 合成 443 requests。

重放 admission 和历史 master assignment 后，group 3 的 313 个请求 owner-set size 分布为：

```text
{1: 44, 2: 256, 3: 13}
```

当前 source-greedy 按 used-KV 得到 candidate 顺序 `[2, 0, 1, 7]`，重放结果为：

1. rank 2 取 128 个请求，receiver 变为 `[98, 64, 0, 0, 0, 0, 0, 34]`；
2. rank 0 只能取 98 个，receiver 变为 `[98, 128, 23, 0, 0, 0, 0, 64]`；
3. rank 1 只能取 55 个，receiver 变为 `[128, 128, 38, 0, 0, 0, 0, 74]`；
4. rank 7 的连续 prefix capacity 已为 0，仍余 162 个请求，返回 `append capacity cannot cover remaining requests`。

但同一状态存在满足约束的分配：保留 group 3 原分配，再把 group 7 的 130 个请求分为 rank 2 的 128 个和 rank 7 的 2 个，可得到：

```text
master batches = [128, 128, 128, 59]
receiver counts = [101, 96, 2, 0, 0, 0, 0, 85]
```

所有 receiver 都低于 128，相关 rank 的 append/KV 余量也满足要求。这证明首次失败不是物理 receiver 总容量不足，也不是 KV blocks 无可用空间，而是 owner 碎片化下，固定 candidate/order 和连续 prefix source-greedy 没有找到已存在的合法 assignment。

`max_sp_batch_size=128` 是故障表象和约束边界，但不能据此推导“receiver 128 太小”。把 receiver 上限继续调大可能掩盖该快照，却不会修复 planner false negative。

## 5. Recovery 活锁

首次失败后，victim 被放入 singleton recovery batch，并立即从 pending 队首重新 admission。资源形状没有发生足以解除瓶颈的变化，因此相同请求反复经历：

```text
plan failed -> preempt -> recovery admission -> merge -> plan failed
```

发送停止后 KV 压力已经明显下降，日志中仍出现：

```text
max_kv_util_pct   ~= 67.7%
min_free_blocks   ~= 4530
max_sp_batch_size = 128
pending requests  ~= 2905
```

但 `seq_id=4382` 等 recovery request 仍被立即 admission 和再次抢占，完成速度降到约每 4--7 秒一个请求，tqdm ETA 达到数小时。由此可以排除 drain 只是正常的高延迟排队。

## 6. Actual-forward 与调度开销

`NANODEPLOY_LOG_MODEL_FORWARD_TIMING=1` 在正式 workload 中采到 2,251 个 CUDA-event 样本。`model_forward_gpu_ms` 是 16 ranks 中的 distributed critical path，覆盖实际 `run_model` 和模型内 collective，不包含 driver scheduler/RPC prepare/sampling。

| 区间 | 样本 | forward mean | p50 | p95 | max | step ITL mean | scheduler overhead mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| 首次失败前 | 1,874 | 114.40 ms | 121.05 ms | 147.08 ms | 242.86 ms | 125.95 ms | 3.67 ms |
| 首次失败后 | 377 | 152.67 ms | 148.63 ms | 217.95 ms | 239.45 ms | 445.42 ms | 287.44 ms |

失败后 forward mean 增加约 33%，而 step ITL 增加约 3.5 倍。主要恶化来自 planner/recovery：scheduler overhead 从 3.67 ms 增至 287.44 ms，约 78 倍。

## 7. 对 T=128 的判断

T=128 的 clean-start 路径确实显著优于 T=8：它减少早期 master 数量和 owner churn，让绝大多数请求在前 250 秒保持 SP1，因而大幅延迟 receiver 假阴性的出现。

但该参数同时控制至少三类语义：compute scale-up chunk、group merge 目标和 consolidation target DoP。它不是单纯的“scale-up 阈值”，也不等于 `max_num_recv_seqs`。本轮恰好展示了这种耦合：T=128 先减少扩散，随后又在 127+3 crossing 时触发 group merge，把已有 owner 历史的两个 group 合并，暴露 source-greedy 的缺陷。

此外，首次失败前累计完成速率约 `3170 / 250 = 12.7 req/s`，仍低于 20 req/s offered load；即使 planner 修好，rate=20 是否可长期稳定仍需重新测量，不能由本轮证明。

建议下一步优先级：

1. 把 source-greedy 改为 receiver/owner-aware assignment，至少加入跳过、回溯或 matching；
2. recovery 增加 capacity/receiver epoch 或 backoff，禁止无状态变化的立即重入；
3. 解耦 compute scale-up、group merge 和 consolidation 的 threshold；
4. 修复后再用相同 workload 扫 `T={64,100,128}`，先 consolidation off，再单独做 execute A/B。

## 8. 产物

- 原始日志：`docs-dev/2026-07-16/ls_style_issue001_2node_dp2sp8_r20_t128_6min_20260716.log`；
- 前一轮复盘：`docs-dev/2026-07-16/ls_decode_r20_preemption_postmortem_20260716.md`；
- 本轮未生成 JSONL，原因是 drain 活锁后主动停止。
