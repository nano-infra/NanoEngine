# LS capacity-reorg 两机 140 GiB rate=20 运行结果

日期：2026-07-17

## 结论

本次 LoongServe-style capacity reorganization 两机测试没有完成：正式 workload 在 `7090 / 7200`、elapsed `18:44` 时，第 13 次自动 KV consolidation 的 NCCL 接收操作失败，进程 exit code 为 1。

- warmup `32 / 32` 正常完成，正式 workload 从 `10:11:06 UTC` 运行至 `10:29:50 UTC`；
- 360 秒只完成 `3563 / 7200`，完成窗口速率 9.90 req/s；
- pending 峰值为 `459 batches / 2395 requests`，最老 batch 等待 2535 个 scheduler step；
- KV 峰值 99.91%，最紧 rank 仅余 12 blocks；
- 新功能不是没有运行：pending 压力期成功执行 6 次 consolidation，pending 清空后又成功执行 6 次后台 consolidation；
- 12 次成功事件共迁移 27,648 tokens、911 个物理 range，maintenance 约 996.58 ms；
- 第 13 次事务在 `dist.irecv()` 失败，NCCL 报 `Failed to CUDA calloc async 8 bytes`；
- 没有 preemption、planner failure 或当时新增的 GPU Xid；退出后 Ray 为 `0 / 16 GPU`，资源已释放；
- benchmark 没有进入最终序列化阶段，因此没有生成 output JSONL，也没有完整延迟分位数。

这次结果同时说明两件事：新加入的压力重组路径能够真实执行并帮助 pending admission，但它还不足以消除 140 GiB cliff；物理 NCCL/CUDA headroom 也仍未闭环。

## 可复现状态

- 实际运行代码 commit：`ab6ae471d1a8f949be807a22909999e295715dc8`（`feat(scheduler): align LS capacity reorganization`）；
- 启动记录 commit：`f2d3bac`；
- topology：Attention `DP2 × SP8 × TP1`，FFN `EP16`，两节点 16 H200；
- workload：Issue 1% CSV，seed 0，Poisson rate 20，360 秒，7200 requests；
- DeepSeek-V3 dummy prefill / dummy weight；
- `gpu_memory_limit_gb=140`、`gpu_memory_utilization=0.9`，每 rank 14,030 KV blocks；
- `max_num_seqs=256`、`max_num_recv_seqs=128`、`loop_count=1`；
- full CUDA Graph；
- LS core scheduler、future-KV admission、memory scale-up、capacity donor merge、pending no-fit proof 全部开启；
- consolidation execute，candidate/high-water=`0.5/0.8`，stable/cooldown/check=`2/2/1`，source budget 128 blocks，chunk 64 tokens；
- 工作区的 RPC server buffer 覆盖为 `32 * 32_000_000`，已在 manifest 中记录；它不是本次 NCCL KV migration scratch。

完整命令、环境、dirty override hash 和二进制 hash 见同目录 manifest：

- `ls_style_capacity_reorg_ab6ae47_2node_dp2sp8_r20_140gb_6min_20260717.manifest.json`

## 运行轨迹

| elapsed | completed | 已完成请求平均延迟 |
|---:|---:|---:|
| 60 s | 113 | 39.48 s |
| 120 s | 872 | 61.33 s |
| 180 s | 1,885 | 72.96 s |
| 240 s | 2,901 | 79.72 s |
| 300 s | 3,296 | 84.06 s |
| 360 s | 3,563 | 90.07 s |
| 600 s | 4,809 | 138.20 s |
| 900 s | 5,875 | 207.01 s |
| 1,124 s | 7,090 | 288.12 s |

前 3 分钟与 141 GiB future-KV 运行非常接近，约从第 4 分钟开始出现明显路径分叉。系统随后呈现“已有请求释放容量 -> 一批 pending 原子 admission -> KV 再次接近 100%”的锯齿行为，而不是平滑 admission。

### 与已有结果的观测对比

当前 140 GiB 运行包含 commit `ab6ae47` 的新重组逻辑，因此下面不是只改变 1 GiB 的严格单变量 A/B。不过请求序列、rate、DP/SP、receiver cap、loop count 和主要 LS 配置相同，可以定位 cliff 的量级。

| 指标 | 140 GiB capacity-reorg | 141 GiB future-KV | Original NanoDeploy |
|---|---:|---:|---:|
| 360 秒完成 | 3,563 | 5,366 | 6,495 |
| 360 秒完成速率 | 9.90 req/s | 14.91 req/s | 18.04 req/s |
| pending peak | 2,395 req | 16 req | 0 req |
| peak KV util | 99.91% | 99.65% | 56.55% |
| min free blocks | 12 | 49 | 6,103 |
| consolidation | 12 成功 + 第 13 次 fatal | 3 成功 | 未启用 LS consolidation |
| 最终结果 | 7,090 后失败 | 7,200，566.04 s | 7,200，411.06 s |

140 GiB 每 rank 只有 14 blocks 少于 141 GiB 运行（14,030 vs 14,044），但 future-KV/原子 admission 的可行性是离散边界。一旦某个 logical batch 在当下 group allocation 中 no-fit，它会改变后续 group ownership、DoP、KV 分布和完成顺序，最终放大为 2,395-request backlog。这里不是线性的“少 14 blocks 就少一点吞吐”，而是调度轨迹跨过了可行性边界。

## 新重组功能实际做了什么

### 成功事件

前 6 次发生在 pending/no-fit 压力窗口；后 6 次发生在 pending 清空后的后台整理阶段。

| UTC | tx | 阶段 | DP/group | source | tokens | moves | maintenance / stall ms |
|---|---:|---|---|---:|---:|---:|---:|
| 10:19:35 | 1553 | pending pressure | 1/3 | 7 | 4,950 | 134 | 90.59 / 229.58 |
| 10:22:15 | 6050 | pending pressure | 1/3 | 1 | 219 | 7 | 104.30 / 215.91 |
| 10:23:33 | 8542 | pending pressure | 0/2 | 7 | 1,408 | 61 | 81.27 / 144.71 |
| 10:23:34 | 8543 | pending pressure | 0/2 | 5 | 6,405 | 219 | 90.18 / 142.60 |
| 10:23:34 | 8547 | pending pressure | 1/3 | 0 | 691 | 22 | 63.21 / 110.24 |
| 10:23:35 | 8551 | pending pressure | 1/3 | 4 | 1,150 | 48 | 74.74 / 116.42 |
| 10:29:10 | 18423 | background | 1/3 | 4 | 241 | 8 | 66.96 / 74.51 |
| 10:29:27 | 19228 | background | 1/3 | 5 | 7,523 | 243 | 116.77 / 116.94 |
| 10:29:49 | 20174 | background | 0/2 | 1 | 342 | 10 | 74.13 / 74.29 |
| 10:29:49 | 20175 | background | 0/2 | 2 | 342 | 13 | 86.43 / 86.59 |
| 10:29:49 | 20176 | background | 0/2 | 0 | 742 | 24 | 67.72 / 67.88 |
| 10:29:49 | 20177 | background | 0/2 | 6 | 3,635 | 122 | 80.28 / 80.70 |

其中前 6 次共迁移 14,823 tokens、491 moves，stall 约 959.47 ms。它们证明 pending-aware exact post-consolidation proof 和执行链路确实走通，不是仅有 shadow decision。

### 没有执行的判定

| decision reason | 次数 | 含义 |
|---|---:|---|
| `pending_no_beneficial_consolidation` | 1,368 | 当前没有可缩减并释放 rank 的 group |
| `source_block_budget` | 1,347 | 候选 source 超过每事件 128 blocks 上限 |
| `target_high_watermark` | 803 | 搬运后目标 rank 会超过 80% 水位 |
| `pending_no_benefit` | 201 | 单次搬运完成后仍不能让完整原子 batch admission |
| `no_candidate` | 660 | 无 pending 时没有后台候选 |
| `stable_window` | 195 | 后台候选尚未满足稳定窗口 |

调度开销峰值达到 529.99 ms。pending batch 每 step 重试、逐候选构造 reservation、再做 exact future-KV placement proof，在 backlog 达到数百 batch 后会明显放大控制面开销。

## 为什么仍然出现 cliff

1. **原子 logical batch 仍然很大。** waiting queue 不能像 LoongServe 一样逐请求跳过或拆分；一个长请求可以连带同 batch 的短请求一起等待。
2. **收益证明只接受单步可达。** 当前 `_ls_consolidation_enables_pending_batch()` 只验证“释放当前 1 个 source rank 后，整个 batch 是否立即可 admission”。若需要连续释放 2 个 rank，第一步会被 `pending_no_benefit` 拒绝，无法积累多步重组。
3. **安全阈值在高压时大量拦截。** 128-block source budget 和 80% target high-water 共拒绝 2,150 个 step；它们避免一次搬运过大，但也让系统主要依赖请求自然完成来释放容量。
4. **没有 capacity epoch/backoff。** 资源状态没有实质变化时仍反复证明同一个 no-fit，导致 459 个 pending batch 下 scheduler overhead 达到数百毫秒。
5. **重组能缓解局部碎片，但不能替代 admission 粒度。** 6 次 pressure consolidation 有效，却不足以持续吸收 offered rate 20；最终仍进入反复蓄水/放水的轨迹。

## 第 13 次 consolidation 为什么失败

失败前 group 2 的 KV DoP 已由 8 连续降到 4。下一候选为 DP0/group2/source rank5：

- source 上只有 86 blocks、4,720 tokens，低于 128-block budget；
- 当时 decode 侧 max logical KV util 为 72.06%，min free blocks 为 3,920；
- 接收 actor 为 head 节点 global rank4；
- 异常发生在 `nanodeploy/worker/kv_p2p.py:155` 的 `dist.irecv()`；
- NCCL 2.27.3 最后错误为 `Failed to CUDA calloc async 8 bytes`；
- 10:29 没有对应 NVRM Xid，进程结束后显存正常释放。

这里的 `free blocks` 只是已经预分配 KV tensor 内的逻辑空闲块，不等于 CUDA runtime 仍有物理显存可申请。因此 72% 逻辑 KV 水位不能排除 NCCL 内部 allocation 失败。

### 不是哪个 buffer 变大了

KV migration 使用的是固定 scratch：

- `CacheContext.allocate_kvcache()` 初始化时一次性创建 `migration_scratch`；
- 每个 chunk 只取 `self.scratch[:token_count]` 的 view，不会每次重新申请传输 tensor；
- 当前 chunk 为 64 tokens，DeepSeek-V3 MLA 的 scratch 为 `64 × 61 × (512 + 64) × 2 = 4,497,408 bytes`，约 4.29 MiB；
- 失败的 8-byte allocation 来自 NCCL 内部 `ncclCudaCallocAsync`，不是这个 4.29 MiB scratch，也不是消息长度超过 scratch；
- dirty worktree 中 `32 × 32,000,000` 的 RPC endpoint buffer 是序列元数据的 CPU/pinned-host RDMA buffer，与 `attn_sp_group` 的 NCCL KV P2P 通道无关。

现有证据把故障范围缩小到 NCCL/CUDA runtime headroom、async allocator 或通信资源初始化/生命周期，但还不能只凭这一条日志区分三者。下一次必须同时打开 `NCCL_DEBUG=INFO` 并在每个 worker 迁移前记录 `torch.cuda.mem_get_info()`、allocator allocated/reserved 和 peer pair。

## 后续修改优先级

### P0：先让 KV P2P 的物理资源可证明安全

1. 在 full CUDA Graph 大规模 capture 前，对 `attn_sp_group` 所有可能的 source/destination pair 做最小 NCCL P2P warmup，提前建立通信资源。
2. KV block sizing 显式扣除 graph capture 后的 NCCL/runtime headroom；记录 capture 前后和 migration 前的每 rank physical free memory，不能只看 logical free blocks。
3. 先做 16-rank 全 peer 小数据迁移 smoke，再跑 6 分钟；smoke 使用 `NCCL_DEBUG=INFO`，并验证每个 pair 至少成功一次。
4. 临时性能隔离可把 `candidate_util` 设为 0，只保留 pending pressure consolidation、关闭 pending 清空后的后台整理；这能判断 cliff 性能，但不是最终稳定性修复。

### P1：让压力重组支持多步收益

1. 将单事务 proof 扩展为 bounded multi-rank evacuation plan：一次规划 1--N 个 source rank，只有完整组合可 admission 时才统一 reserve/execute/commit。
2. 更优先的低风险方案仍是拆分 pending logical batch，把极长请求隔离，短请求继续 admission；这更接近 LoongServe waiting queue 的请求级选择。
3. 加 capacity epoch、失败 fingerprint 和指数 backoff；只有 free blocks、group allocation、pending head 或完成集合变化时才重做 expensive proof。

### P2：补足结构化 telemetry

记录 capacity donor merge 次数、group merge 前后 DoP、pressure/background execute reason、failed plan 的 moves/peer pairs，以及物理显存。当前日志能数 consolidation，但不能精确统计 capacity-driven group merge 的次数。

## 产物

- `docs-dev/2026-07-17/ls_style_capacity_reorg_ab6ae47_2node_dp2sp8_r20_140gb_6min_20260717.log`；
- `docs-dev/2026-07-17/ls_style_capacity_reorg_ab6ae47_2node_dp2sp8_r20_140gb_6min_20260717.manifest.json`；
- 本报告；
- output JSONL 未生成，因为 benchmark 在结果序列化前退出。
