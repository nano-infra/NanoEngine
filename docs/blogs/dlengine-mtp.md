## DLEngine - MTP 投机解码

> 当前部署参数、支持边界和验收清单以 [GLM Recurrent MTP](../site/glm-recurrent-mtp.md) 为准。本文保留实现原理与带环境口径的历史性能快照。

### 1. 背景：为什么需要投机解码？

自回归解码（Autoregressive Decoding）是大语言模型推理的核心瓶颈。每个 decode step 只产出一个 token，而 MoE 模型的一次 forward pass 需要经历完整的 All-to-All 专家路由。对于 Qwen3.5-397B-A17B 这类千亿级 MoE 模型，单步 decode 的延迟中有相当比例被跨节点通信占据。

投机解码（Speculative Decoding）的核心思想很简单：**用一个轻量的"草稿模型"（Draft Model）快速猜测未来若干个 token，再用完整的目标模型一次性验证**。猜对的 token 直接跳过，猜错的重新采样。只要猜中率足够高，就能在不牺牲生成质量的前提下实现多倍加速。

MTP（Multi-Token Prediction）是一种特殊的投机解码方案：**草稿模型不是外挂的小模型，而是目标模型自带的 MTP 层**。Qwen3.5 和 DeepSeek V3 等最新模型在训练时就内置了 MTP head，这些 head 共享主模型的 embedding 和部分权重，能以极低的额外计算开销产出高质量的草稿 token。

### 2. 核心设计：Lazy Verify（惰性验证）

#### 2.1 为什么不用 Tree Speculation？

主流的投机解码框架（如 SpecInfer、Medusa）通常采用树状推测（Tree Speculation）：在每个 draft 位置展开多个候选分支，构造一棵推测树，然后一次性验证整棵树。

但在 EP MoE 的生产环境下，树状推测面临严重的工程障碍：

| 问题                   | 影响                                                                |
| ---------------------- | ------------------------------------------------------------------- |
| 指数级状态爆炸         | 每个分支都需要独立的 KV Cache 和 GDN 状态，显存开销随 $k^n$ 增长    |
| CUDAGraph 不友好       | 树结构的动态 shape 难以被 CUDAGraph 捕获，无法享受图优化的低延迟    |
| 线性注意力状态管理复杂 | GatedDeltaNet 等 RNN-like 层的 recurrent state 在分支上需要完整快照 |
| EP 通信放大            | 树的每个候选 token 都要参与 All-to-All 路由，通信量与分支数成正比   |

DLEngine 的选择是一种更务实的策略：**线性 Lazy Verify（惰性验证）**。每个位置只有一个 draft，不展开 tree；目标模型通过固定的 `seqlen_q=N+1` 一次验证整条线性链。

#### 2.2 seqlen_q=N+1 的核心思路

标准 decode 是 `seqlen_q=1`。开启 MTP 后，`num_speculative_tokens=N` 表示 predictor recurrent 运行 N 次，目标模型的验证宽度由系统推导为 `K=N+1`：

```
标准 decode (seqlen_q=1):
  输入: [sampled_token]
  输出: [logit_for_next]

Linear lazy verify (seqlen_q=N+1):
  输入: [base, draft_1, ..., draft_N]
  输出: [verify_1, ..., verify_N, bonus]
```

布局采用 sequence-major：`[base_0, d01, ..., d0N, base_1, d11, ...]`。如果前 `a` 个 draft 被接受，就输出这 `a` 个 draft，再输出拒绝恢复 token；如果 N 个全部接受，则追加 bonus token。

GLM 是最重要的多步场景。GLM checkpoint 只有 **1 个 MTP predictor layer**；`num_speculative_tokens=5` 的含义是同一层 recurrent 运行 5 次，保留 5 个 draft，并由 target 验证 6-token span。它不是 5 个或 6 个 predictor layer。

draft 固定使用 greedy，因而 draft distribution 是 one-hot。`temperature > 0` 时，第 i 个 draft `d` 以 target 概率 `p_i(d)` 被接受；拒绝时从 `p_i` 屏蔽 `d` 后的残差分布采样。这样输出严格保持 target distribution。completion logprob 始终取原始 target distribution，而不是残差分布。

#### 2.3 三阶段 Decode 流程

开启 MTP 后，每个 decode step 被拆分为三个阶段：

```
┌─────────────────────────────────────────────────────────────────┐
│                     一次完整的 Decode Step                       │
├───────────────────┬──────────────────┬──────────────────────────┤
│  Phase 1: 验证     │  Phase 2: 采样    │  Phase 3: 草稿生成       │
│  Lazy Verify       │  Sample           │  MTP Draft Generation   │
│                   │                  │                          │
│  拼接 seqlen_q=N+1 │  逐位 rejection   │  取接受前缀末端 hidden    │
│  input_ids =      │  sampling；全接受  │  recurrent 运行 MTP 层    │
│  [base,d1,...,dN] │  时采 bonus        │  产出 N 个 greedy draft   │
│                   │    free token!    │  保存到 _prev_drafts     │
│  Target Model     │  GDN rollback     │                          │
│  Forward          │  if rejected      │  保存/恢复 decode 上下文   │
└───────────────────┴──────────────────┴──────────────────────────┘
```

**Phase 1（输入准备 + Forward）**：按 `seq_id` 对齐上一步保存的 draft chain，拼成 sequence-major 的 `[base,d1,...,dN]`，走 `LazyVerifyGraphRunner` 做 `seqlen_q=N+1` forward。

**Phase 2（采样 + 状态回滚）**：

- greedy target 下逐位比较 argmax；stochastic target 下执行精确 one-hot rejection sampling
- 接受长度可以是 `0..N`；全接受时从最后一行 target logits 采 bonus
- target forward 虽然写入全部 K 个位置，但下一步只把 base 和接受前缀标为逻辑可见
- N=1 的 GDN 路径在拒绝时恢复快照；N>1 首先支持无 recurrent state 的 GLM DSA/MLA

**Phase 3（MTP 草稿生成）**：先用 target verify 的精确 hidden replay 接受前缀和新采样 token，刷新 predictor KV；再把 `shared_head.norm(hidden, residual)` 的输出同时用于 logits 和下一次 recurrent 输入。GLM 配置开启 `index_share_for_mtp_iteration` 时，从 draft-extend 选出每个请求的最后一行 DSA TopK，并在余下递归中复用。

### 3. 关键工程挑战与解法

#### 3.1 GDN 线性注意力的状态管理

Qwen3.5 的注意力层采用混合架构：部分层是标准的 Full Attention，部分层是 GatedDeltaNet（GDN）线性注意力。GDN 维护了类似 RNN 的循环状态（`conv_states` 和 `recurrent_states`），这些状态会随着每个 token 的处理而不可逆地更新。

这给投机解码带来了独特挑战：**如果 draft token 被拒绝，GDN 状态已经被推进了，必须回滚**。

DLEngine 的解法是 **双倍状态池 + 快照回滚**：

```
GDN State Pool Layout:
┌──────────────────┬──────────────────┬───────┐
│  Active Slots    │  Backup Slots    │ Dummy │
│  [0, max_bs)     │  [max_bs, 2*max) │ [2*N] │
│  正在使用的序列    │  快照副本          │ 图填充 │
└──────────────────┴──────────────────┴───────┘
```

- **Active Slots**：当前正在处理的序列的 GDN 状态
- **Backup Slots**：在 lazy verify forward 之前，由 GDN kernel 自动将 active 状态快照到 backup 区域
- **Dummy Slot**：CUDAGraph 需要固定的 batch size，多出来的填充位置指向 dummy slot，避免污染真实状态

当 draft 被拒绝时，回滚逻辑非常简洁：

```python
# 只对被拒绝且拥有真实 GDN slot 的序列做回滚
rollback_mask = rejected_mask & real_slot_mask
if rollback_mask.any():
    rej_backup = active_slots[rollback_mask] + backup_offset
    gdn_conv_states[:, active_slots[rollback_mask]] = gdn_conv_states[:, rej_backup]
    gdn_recurrent_states[:, active_slots[rollback_mask]] = gdn_recurrent_states[:, rej_backup]
```

> **attention_dp > 1 的陷阱**：在多路 DP 场景下，并非所有 DP rank 都拥有某个序列的真实 GDN slot。非 owner rank 的 `gdn_state_slots` 指向 dummy slot，回滚前必须用 `real_slot_mask = slots < gdn_max_active_slots` 过滤，否则会越界写入 dummy 区域，造成状态污染。

#### 3.2 CUDAGraph 捕获

为了最大化 decode 性能，DLEngine 对三种不同的 forward 模式分别捕获 CUDAGraph：

| 模式        | Runner                      | seqlen_q | 特点                                                                  |
| ----------- | --------------------------- | -------- | --------------------------------------------------------------------- |
| 标准 Decode | `DecodeGraphRunner`         | 1        | 常规自回归解码                                                        |
| 惰性验证    | `LazyVerifyGraphRunner`     | N+1      | 固定 K 的 sequence-major 输入，支持 DSA/MLA 稀疏验证                  |
| N=1 草稿    | `MTPGraphRunner`            | 1        | 兼容既有无 cache predictor 路径                                       |
| GLM 草稿链  | `CachedMTPChainGraphRunner` | 1        | draft-extend 后，将同一 predictor 的余下 N-1 次 recurrent 合并 replay |

三个 Runner 共享同一个 `torch.cuda.graphs.MemPool`——由 `DecodeGraphRunner` 首次 capture 时分配，随后传给其他 Runner。这保证了 graph replay 之间可以零拷贝地复用临时显存。

每个 Runner 预先捕获一组离散的 batch size（`[1, 2, 4, 8, 16, 32, ...]`），运行时自动向上取整到最近的已捕获 batch size。GLM N=5 的第一次 draft-extend 需要刷新 predictor KV 和 DSA TopK，仍然 eager 执行；后续 4 次复用同一个 IndexShare 状态，由一个 cached-chain graph replay 完成。可通过 `DLENGINE_MTP_CHAIN_GRAPH=0` 回退 eager 路径做 A/B。

> **关键细节**：所有 capture 方法都必须在 `@torch.inference_mode()` 下执行。这是因为 GDN 层内部调用了 flashinfer 的 cutlass DSL kernel，其中的 `from_dlpack()` 会拒绝 `requires_grad=True` 的张量，在非推理模式下会触发 `BufferError`。

#### 3.3 MTP 草稿生成的上下文切换与持久化 KV

N=1 兼容路径和 GLM 多步路径使用不同的 predictor 上下文：

- **N=1**：使用 `is_prefill=True` 的无 cache predictor forward。
- **GLM N=5**：为唯一的物理 predictor layer 分配独立 MLA/DSA cache slice；prefill 用 shifted token/target hidden 建 cache，decode 和 draft-extend 使用共享 page table 与独立 layer index 持久更新。

由于 DLEngine 使用全局 Context 单例传递模型执行上下文，predictor forward 前后必须保存并恢复 target 的 batch context、dense MLA metadata 和 sparse DSA metadata。attention-DP 的 dummy rank 没有有效 page table，但仍需执行相同次数的无 cache predictor padding，以保证 8 路 EP collective 顺序一致。完整 prefix-cache 命中会产生零长度 fresh segment；该轮不使用 `cu_seqlens_q - 1` 提取末行，而是安全退回普通 decode，并让空 rank 继续执行 5 次最小 dummy predictor forward，避免 CUDA gather 越界或 EP collective 失配。

GLM checkpoint 只有一个 predictor layer，因此额外 cache 层数是 1，而不是 5 或 6。

#### 3.4 KV Cache 预算与调度协同

MTP 每步只运行 1 次 target decode，但可能提交 1 到 N+1 个 token。prefill 需要为首批 draft 预留 N 个 lookahead token；decode 最坏情况下既要写入 N 个 verify 位置，又要为下一批 N 个 recurrent draft 保留位置，因此预留 2N：

```cpp
prefill_needed = num_tokens + N;
decode_needed = num_tokens + 2 * N;
// 例: 当前 num_tokens=100、N=5 → decode 预留到 token 110
```

Rust 调度器同时在 EOS、`max_tokens` 和 `max_model_len` 的第一个终止点截断 speculative bundle，事件回传、completion logprob 和 decode-token metrics 都只使用实际提交的前缀。

### 4. 模块化代码架构

为了保持代码的可读性和可维护性，MTP 实现采用 **组合模式（Composition）** 而非继承或 Mixin：

```
ModelRunner (编排者, ~590 行)
  ├── InputPreparer       (输入准备, ~235 行)
  │     prepare_prefill_bytes()
  │     prepare_decode_bytes()
  │
  ├── MTPRunner           (MTP 生命周期与精确采样)
  │     prepare_lazy_verify_decode()
  │     lazy_verify_sample()
  │     generate_and_store()
  │     build_output_tokens()
  │
  ├── VisionEmbedManager  (视觉 embedding, ~198 行)
  │     fetch_rdma()
  │     inject()
  │
  ├── DecodeGraphRunner    (标准 decode 图)
  ├── LazyVerifyGraphRunner(验证 decode 图)  ← 由 MTPRunner 持有
  ├── MTPGraphRunner       (N=1 草稿图)      ← 由 MTPRunner 持有
  └── CachedMTPChainGraphRunner (GLM recurrent 草稿链图)
```

`ModelRunner` 作为顶层编排者，通过持有各组件实例来协调整个 decode 流程。每个组件职责单一、高内聚低耦合，可以独立理解和修改。

### 5. 端到端执行流程

以 GLM-5.2-FP8 在 8×H100/H200（attention_dp=8, ffn_ep=8, `num_speculative_tokens=5`）上的一次完整推理为例：

```bash
dlengine serve /path/to/GLM-5.2-FP8 \
  --attention_dp 8 \
  --ffn_ep 8 \
  --ctrl_address 127.0.0.1:4479 \
  --executor_backend dlslime \
  --num_speculative_tokens 5
```

```
请求到达
  │
  ▼
Prefill Phase
  │  标准 prefill forward → 采样 token_0
  │  MTP: 同一个 predictor recurrent 5 次 → [draft_1, ..., draft_5]
  │
  ▼
Decode Step 1 (有 draft)
  │  Phase 1: 拼接 [token_0,d1,...,d5] → seqlen_q=6 → target graph replay
  │  Phase 2: 精确验证，接受 a∈[0,5]，输出 [d1,...,da,next]
  │  Phase 3: 从接受前缀末端 hidden recurrent 生成新的 5-token draft chain
  │
  ▼
Decode Step 2 (有 draft)
  │  ... 重复上述三阶段 ...
  │
  ▼
直到 EOS 或 max_tokens
```

### 6. 性能表现

GLM-5.2-FP8、8×Hopper、attention_dp=8、ffn_ep=8、dlslime，单请求生成 256 tokens。MTP 使用同一个 predictor layer recurrent 5 次，target K=6：

| Python merge-sort 编码任务 |          墙钟 |            ITL | Tokens/Step | 相对无 MTP |
| -------------------------- | ------------: | -------------: | ----------: | ---------: |
| 无 MTP                     |     约 6.88 s |       约 24 ms |        1.00 |      1.00× |
| N=5 eager recurrent        |     约 3.41 s |     约 10.5 ms |        3.94 |   约 2.02× |
| N=5 cached-chain graph     |   2.91–2.96 s | 10.22–10.43 ms |        3.94 |   约 2.35× |
| N=5 target/MTP kernel 优化 | 2.274–2.294 s |   7.77–7.85 ms |        4.00 |   约 3.01× |
| N=5 row-strided KV RMSNorm | 1.940–2.029 s |   6.45–6.82 ms |        4.74 |   约 3.49× |

cached-chain graph 相比 eager MTP 再减少约 14% 墙钟。在此基础上，inactive attention-DP rank 也执行对称 recurrent graph，small-M FP8 quant 使用 packed reduction，Q-A/KV-A 共享一次输入量化，并用单个 Triton kernel 融合 interleaved-to-half 转换和 Q/K RoPE。最后加入 row-strided compressed-KV RMSNorm 后，8 次无 profiler 编码请求平均 1.974 s、ITL 平均 6.56 ms，最终相较无 MTP 约 3.49×。不同 rank 的 FP8 MoE 数值路径可能改变 draft 内容和接受长度，因此性能比较看 target-policy 语义与统计指标，不要求生成文本逐字一致。

最后一个 target 热点来自 MLA compressed KV：576-wide projection 的前 512 维是 row-strided view，旧通用 RMSNorm 因为要求整块 contiguous，每层退回 `float → pow → mean → rsqrt → cast → mul → copy-back` 的 eager 链。现在 `rms_norm_strided_inplace` 直接在 stride=576 的 view 上原地归一化，可通过 `DLENGINE_INPLACE_MLA_KV_NORM=0` 回退旧路径。

新 profiler 中 active target verify 从 3829 降到 3205 kernels，恰好减少 78 层 × 8 kernels；rank 0 GPU span p50 从 26.80 ms 降到 23.96 ms，八卡汇总 p50 为 24.94 ms。4-step recurrent chain 从 241 降到 237 kernels，八卡 p50 为 3.37 ms；非 active attention-DP rank 的 padded verify 为 2987 kernels。完整时间线和 kernel summary 应保存在配置的持久化 profiler 目录中。

`temperature=0.7` 的 128-token Python 编码请求也完成了 exact rejection 路径；原地 KV RMSNorm 后复测 3/3 返回 128 tokens，接受长度为 4.00–4.74。16 路并发和连续两轮 8 路请求均通过。

1M 边界也在 16×H200 非对称 PD 拓扑上完成了正确性验证：PP8 prefill 对接 attention-DP8/EP8 HiSparse decode，输入 999,999 tokens，`temperature=0.7`，输出 64 tokens。两端都分配 15,626 个 cache pages，8 个 PP peer 的 RDMA 全部完成，decode 成功恢复 predictor KV 与 5-token MTP handoff；近似 TTFT 为 186 s，迁移加首轮 verify 为 48.35 s，请求以 25 个 decode steps 完成，Tokens/Step=2.52，ITL=21.44 ms，端到端墙钟 237.19 s。该压缩随机提示主要验证 1M 容量、跨页迁移和 stochastic 语义，不作为编码 workload 接受长度或稳态 decode 性能基准。

### 7. 设计取舍总结

| 设计决策                       | 取舍                  | 理由                                     |
| ------------------------------ | --------------------- | ---------------------------------------- |
| 线性 chain 而非树状推测        | 每位置只有 1 个 draft | 固定 shape，CUDAGraph 友好，无指数级膨胀 |
| 独立的三路 Graph Runner        | 额外的 buffer 显存    | 职责清晰，capture/replay 解耦，便于调试  |
| GDN 双倍状态池                 | 2× 显存开销           | 快照/回滚零重计算，无需 recompile        |
| MTP 低延迟 EP 模式             | 上下文切换开销        | 确保 draft 生成的专家路由延迟最小        |
| 按 num_speculative_tokens 预留 | 调度器多分配 block    | 提前预留 KV Cache，运行时无 OOM 风险     |
| 组合模式拆分                   | 多文件、多类          | 高内聚低耦合，单文件可读，便于独立修改   |

### 8. 当前能力边界与未来方向

- **已支持**：N=1 的既有 MTP；GLM DSA/MLA 在 Hopper 上的 N=5/K=6 线性多步路径；hybrid 与 PD 分离；greedy 与 `temperature > 0`；completion logprob；batch reorder/shrink；以及 `PP>1` prefill 对接 `PP=1` decode 的非对称 PD 拓扑。PP prefill 仅在最后 stage 运行 predictor，并通过常规 KV MR 和独立 `mtp_handoff` MR 迁移 predictor KV 与 5-token draft bundle；decode 首轮可直接 verify，失配或 stale row 安全回退 target decode。GLM decode 也可叠加 HiSparse：K=6 的 DSA top-k 在 request 内合并去重，六个 target 输出使用独立 hot slot，并在 verify 后完整写回 cold host cache。
- **HiSparse 配置**：GLM N=5/K=6、`index_topk=2048` 时，decode 端添加 `--enable_hisparse true --hisparse_device_buffer_size 12288`；容量下限为 `(num_speculative_tokens + 1) * index_topk`。prefill 端保持普通 PD cache，并通过数据面把 KV 迁移到 decode cold host tier。长上下文 PP prefill 建议使用 `--max_num_batched_tokens 8192` 作为每 stage microbatch 上限；不要把完整 100K/1M prompt 合成一次巨型 GPU forward。
- **暂不支持**：tree、非 GLM 的 HiSparse+MTP、PP decode、非 Hopper multi-step、GDN multi-step、DeepSeek/Qwen multi-step。
- **下一性能热点**：recurrent 与 active target verify 已不再是唯一主导项；下一阶段优先降低 profiler 外的 CPU 调度抖动、专家通信长尾和 cold prefill/routing 开销，而不是为了计数继续拆改 recurrent 或 verify 控制流
- **自适应投机深度**：根据运行时接受率动态调整 draft 数量，在高接受率时激进投机，低接受率时退回标准 decode
- **与 EPLB 协同**：将 MTP 的 draft token 纳入专家负载均衡（EPLB）的统计，优化 EP 场景下的热点专家调度
