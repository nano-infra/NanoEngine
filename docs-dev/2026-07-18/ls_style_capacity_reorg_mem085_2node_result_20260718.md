# LoongServe-style 两机 mem frac 0.85、6 分钟测试结果

日期：2026-07-18

## 结论

本次两机 16×H200、Attention `DP2 × SP8` 的 LoongServe-style 测试完整成功：warmup `32 / 32`，正式 workload `7200 / 7200` 全部完成，进程 exit code 为 0，没有 OOM、NCCL/CUDA 异常、preemption 或 planner failure。测试后 Ray 显示 `0 / 16 GPU`、`0 / 384 CPU`，资源已经释放。

这里的“6 分钟”是 360 秒请求注入窗口；由于 offered load 为 20 req/s、系统未能稳态跟上，之后继续 drain，最终 wall time 为 660.44 秒。

- `gpu_memory_limit_gb=140`，`gpu_memory_utilization=0.85`；每 rank 分配 12,359 个 KV blocks；
- 360 秒时完成 5,111 个请求，窗口完成速率 14.20 req/s，为 7,200 个请求的 70.99%；
- 最终 output-token throughput 6,529.09 token/s，平均 TTFT 6,132.74 ms，平均 E2E 103.95 s；
- P99 queueing time 为 140,843.35 ms，TPOT-with-queueing `<100 ms` 的 goodput 为 `580 / 7200`（8.06%）；
- KV 峰值 99.60%，最紧 rank 仅余 50 blocks；pending 峰值为 106 batches / 300 requests；
- 成功执行 13 次 KV consolidation，共迁移 16,003 tokens、561 个物理 range，没有 consolidation failure。

因此，这次验证了 `mem frac=0.85` 下当前 LoongServe-style 路径能够稳定跑完整个请求集，并避免旧 140 GiB / 0.90 运行的 fatal NCCL allocation；但它没有吃满 20 req/s 的输入负载，排队尾延迟和 goodput 仍然较差。

## 配置与可复现状态

- 运行 commit：`908933a5c0b8d5a54f86bfa589a372b14e9f19ca`（`feat(scheduler): sort LS admission window by prompt length`）；
- Ray：head `10.102.243.60:8776`，worker `10.102.252.174`，共 16×H200；
- topology：Attention `DP2 × SP8 × TP1`，FFN `EP16`；
- workload：Issue 1% CSV，seed 0，Poisson rate 20，360 秒，7,200 requests；
- model：DeepSeek-V3 dummy prefill / dummy weight；
- `gpu_memory_limit_gb=140`、`gpu_memory_utilization=0.85`、KV block size 64；
- `max_num_seqs=256`、`max_num_recv_seqs=128`、`max_num_batched_tokens=1024000`；
- full CUDA Graph；
- LS core scheduler、future-KV admission、memory scale-up、capacity reorganization 和 execute-mode KV consolidation 开启；
- consolidation candidate/high-water=`0.5/0.8`，stable/cooldown/check=`2/2/1`，source budget 128 blocks，chunk 64 tokens；
- 运行前按仓库要求执行了 `pip install -v -e .`；Ninja 报 `no work to do`，安装成功；
- 工作区仍带有既有 RPC endpoint buffer override：`32 * 32_000_000`，以及 `llm_engine.py` 的行尾差异；二者均在 manifest 中记录，未被本次文档提交纳入。

完整命令、环境、源码/二进制 hash 和产物 hash 见同目录 manifest：

- `ls_style_capacity_reorg_908933a_2node_dp2sp8_r20_140gb_mem085_6min_20260718.manifest.json`

## 运行轨迹

按仓库既有统计口径，取 tqdm 每个整分钟 bucket 内最后一个完成值：

| elapsed | completed | 已完成请求平均延迟 |
|---:|---:|---:|
| 60 s | 190 | 38.79 s |
| 120 s | 1,095 | 54.57 s |
| 180 s | 2,097 | 63.11 s |
| 240 s | 3,217 | 69.32 s |
| 300 s | 4,259 | 74.51 s |
| 360 s | 5,111 | 77.90 s |
| 660.44 s | 7,200 | 103.95 s |

最后一次采样 arrival 为 354.6172 秒，即窗口结束前约 5.38 秒；到 360 秒时，全部 7,200 个请求都已按预定 arrival 提交。之后约 300 秒主要用于清空 backlog。

## 最终延迟

| 指标 | Avg | P50 | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| TPOT without queue (ms/token) | 155.00 | 135.71 | 274.67 | 311.30 | 367.96 |
| ITL with decode queue (ms/token) | 166.15 | 142.89 | 249.48 | 281.01 | 457.86 |
| Queueing time (ms) | 6,132.07 | 1.12 | 2.03 | 34.40 | 140,843.35 |

Queueing 的 P95 仍只有 34.40 ms，而 P99 跳到 140.84 秒，说明性能问题集中在一小批被长时间阻塞的请求；这也与 pending 最老 batch 等待 4,255 scheduler steps、最多重试 4,256 次一致。

## Scheduler / consolidation 观测

- decode iterations：4,445；max total batch size：1,978；max SP batch size：159；
- scheduler overhead：mean 16.09 ms，P50 4.95 ms，P95 115.38 ms，P99 176.70 ms，max 195.60 ms；
- planner latency：mean 1.85 ms，P50 1.72 ms，P95 6.68 ms，P99 7.41 ms，max 12.11 ms；
- model-forward GPU critical path：mean 116.60 ms，P50 117.41 ms，P95 168.09 ms，P99 182.11 ms，max 240.85 ms；
- 所有记录到的 scale reason 都是 `none`，没有 memory scale-up 事件；
- 13 次 consolidation 都发生在后期 drain，共 16,003 tokens / 561 moves，maintenance 累计 1,007.15 ms，stall 累计 1,015.66 ms；
- consolidation 未执行原因以 `pending_no_beneficial_consolidation`（2,417 次）、`no_candidate`（981 次）和 `source_block_budget`（764 次）为主。

这说明 0.85 提供的物理显存余量解决了本次稳定性问题，但调度容量仍然被 KV 可行性和大 admission batch 限制；控制面在 backlog 期间也有明显放大。

## 与旧 0.90 运行的观测对比

| 指标 | 当前 140 GiB / 0.85 | 旧 140 GiB / 0.90 |
|---|---:|---:|
| 运行 commit | `908933a` | `ab6ae47` |
| 每 rank KV blocks | 12,359 | 14,030 |
| 360 秒完成 | 5,111 | 3,563 |
| pending peak | 300 req | 2,395 req |
| peak KV util | 99.60% | 99.91% |
| consolidation | 13 成功 | 12 成功，第 13 次 fatal |
| 最终结果 | 7,200，660.44 s，成功 | 7,090，1,124 s，失败 |

这不是严格的 mem-frac 单变量 A/B：当前运行还包含 `908933a` 的 prompt-length admission sorting，而旧运行停留在 `ab6ae47`；因此不能把完成数改善单独归因于 0.85。能够确定的是，当前代码与 0.85 组合在相同 140 GiB limit、workload、seed 和 topology 下完整跑通，且没有复现旧运行的 NCCL/CUDA fatal。

## 产物

- 完整日志：`ls_style_capacity_reorg_908933a_2node_dp2sp8_r20_140gb_mem085_6min_20260718.log`（60,081,428 bytes，受 `*.log` 忽略规则保护）；
- 逐请求 ITL：`ls_style_capacity_reorg_908933a_2node_dp2sp8_r20_140gb_mem085_6min_20260718.jsonl`（65,005,352 bytes，7,200 行）；
- manifest：`ls_style_capacity_reorg_908933a_2node_dp2sp8_r20_140gb_mem085_6min_20260718.manifest.json`；
- 本报告。

日志和 JSONL 均保留在工作区；由于体积分别约 57 MiB 和 62 MiB，本次只提交 report 与 manifest。
