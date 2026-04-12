## NanoDeploy - MTP 投机解码

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

NanoDeploy 的选择是一种更务实的策略：**Lazy Verify（惰性验证）**——每步只验证一个 draft token，通过 `seqlen_q=2` 的巧妙编码将验证融入正常 decode 流程。

#### 2.2 seqlen_q=2 的核心思路

标准 decode 是 `seqlen_q=1`：每个序列送入 1 个 token，产出 1 个 logit。Lazy verify 将其扩展为 `seqlen_q=2`：

```
标准 decode (seqlen_q=1):
  输入: [sampled_token]
  输出: [logit_for_next]

Lazy verify (seqlen_q=2):
  输入: [prev_sampled, draft_0]      ← 两个 token 交替拼接
  输出: [verify_logit, bonus_logit]   ← 分别用于验证和奖励采样
```

对于 batch 中的每个序列，输入从 1 个 token 扩展为 2 个 token（上一步采样的 token + MTP 预测的 draft token），interleave 排列为 `[prev_0, draft_0, prev_1, draft_1, ...]`。

输出 logits 也相应产生两组：

- **偶数位 `[0::2]`（verify logits）**：基于 prev_sampled 和 draft_0 的联合注意力产出，用于验证 draft_0 是否正确
- **奇数位 `[1::2]`（bonus logits）**：如果 draft_0 被接受，直接从 bonus logits 采样下一步的 token（免费多得一个 token）

#### 2.3 三阶段 Decode 流程

开启 MTP 后，每个 decode step 被拆分为三个阶段：

```
┌─────────────────────────────────────────────────────────────────┐
│                     一次完整的 Decode Step                       │
├───────────────────┬──────────────────┬──────────────────────────┤
│  Phase 1: 验证     │  Phase 2: 采样    │  Phase 3: 草稿生成       │
│  Lazy Verify       │  Sample           │  MTP Draft Generation   │
│                   │                  │                          │
│  拼接 seqlen_q=2   │  verify_logit:    │  取 last_hidden          │
│  input_ids =      │    accepted?      │  走 MTP 层 forward       │
│  [prev, draft]    │  bonus_logit:     │  产出 N 个 draft token   │
│                   │    free token!    │  保存到 _prev_drafts     │
│  Target Model     │  GDN rollback     │                          │
│  Forward          │  if rejected      │  保存/恢复 decode 上下文   │
└───────────────────┴──────────────────┴──────────────────────────┘
```

**Phase 1（输入准备 + Forward）**：将上一步的 `prev_sampled_token` 和 `prev_draft[0]` 交替拼接，走 `LazyVerifyGraphRunner` 做 `seqlen_q=2` 的 forward。

**Phase 2（采样 + 状态回滚）**：

- 从 verify logits 采样 `target_pred`，与 `prev_draft[0]` 比较
- 如果 `target_pred == prev_draft[0]`：接受！从 bonus logits 采样下一个 token（免费多获得一个 token）
- 如果不匹配：拒绝，使用 `target_pred` 作为新的采样结果，并回滚 GDN 线性注意力状态

**Phase 3（MTP 草稿生成）**：利用 target model 的 `last_hidden_states`，通过 MTP 层快速生成下一批草稿 token，储存起来供下一个 decode step 使用。

### 3. 关键工程挑战与解法

#### 3.1 GDN 线性注意力的状态管理

Qwen3.5 的注意力层采用混合架构：部分层是标准的 Full Attention，部分层是 GatedDeltaNet（GDN）线性注意力。GDN 维护了类似 RNN 的循环状态（`conv_states` 和 `recurrent_states`），这些状态会随着每个 token 的处理而不可逆地更新。

这给投机解码带来了独特挑战：**如果 draft token 被拒绝，GDN 状态已经被推进了，必须回滚**。

NanoDeploy 的解法是 **双倍状态池 + 快照回滚**：

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

#### 3.2 三路 CUDAGraph 捕获

为了最大化 decode 性能，NanoDeploy 对三种不同的 forward 模式分别捕获 CUDAGraph：

| 模式        | Runner                  | seqlen_q | 特点                                                            |
| ----------- | ----------------------- | -------- | --------------------------------------------------------------- |
| 标准 Decode | `DecodeGraphRunner`     | 1        | 常规自回归解码                                                  |
| 惰性验证    | `LazyVerifyGraphRunner` | 2        | 输入 buffer 为 `max_bs * 2`，tile scheduler 按双倍 q-heads 计算 |
| MTP 草稿    | `MTPGraphRunner`        | 1        | 以 prefill 模式运行（无 KV Cache），使用低延迟 EP 模式          |

三个 Runner 共享同一个 `torch.cuda.graphs.MemPool`——由 `DecodeGraphRunner` 首次 capture 时分配，随后传给其他 Runner。这保证了 graph replay 之间可以零拷贝地复用临时显存。

每个 Runner 预先捕获一组离散的 batch size（`[1, 2, 4, 8, 16, 32, ...]`），运行时自动向上取整到最近的已捕获 batch size。

> **关键细节**：所有 capture 方法都必须在 `@torch.inference_mode()` 下执行。这是因为 GDN 层内部调用了 flashinfer 的 cutlass DSL kernel，其中的 `from_dlpack()` 会拒绝 `requires_grad=True` 的张量，在非推理模式下会触发 `BufferError`。

#### 3.3 MTP 草稿生成的上下文切换

MTP forward 和标准 decode forward 使用完全不同的上下文配置：

- **Decode**：`is_prefill=False`，需要 KV Cache（slot_mapping, block_tables, context_lens）
- **MTP**：`is_prefill=True`（伪 prefill），不需要 KV Cache，仅需 cu_seqlens

由于 NanoDeploy 使用全局 Context 单例来传递模型执行上下文，MTP 在做 forward 前必须：

1. **保存**当前的 decode context（slot_mapping, block_tables 等）
2. **切换**到 MTP context（cu_seqlens, 无 KV Cache，低延迟 EP 模式）
3. 执行 MTP forward
4. **恢复** decode context

这个 save/restore 过程由 `MTPWorker.generate_and_store()` 负责，确保 MTP 的上下文不会泄漏到后续的 decode step。

#### 3.4 KV Cache 预算与调度协同

MTP 虽然每步只运行 1 次 decode loop，但可能产出 1 + N 个 token（1 个采样 + N 个被接受的草稿）。调度器必须提前预留足够的 KV Cache block：

```python
# config.py
self._mtp_original_loop_count = self.loop_count  # 保存原始值 (1)
self.loop_count = self.loop_count + num_speculative_tokens + 1
# 例: num_speculative_tokens=1 → loop_count 从 1 膨胀到 3
```

调度器使用膨胀后的 `loop_count` 计算 block 需求，而实际 decode 循环使用原始值（恒为 1）。这样既保证了 KV Cache 不会 OOM，又避免了不必要的多轮循环。

### 4. 模块化代码架构

为了保持代码的可读性和可维护性，MTP 实现采用 **组合模式（Composition）** 而非继承或 Mixin：

```
ModelRunner (编排者, ~590 行)
  ├── InputPreparer       (输入准备, ~235 行)
  │     prepare_prefill_bytes()
  │     prepare_decode_bytes()
  │     update_decode_inplace()
  │
  ├── MTPWorker           (MTP 生命周期, ~385 行)
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
  ├── LazyVerifyGraphRunner(验证 decode 图)  ← 由 MTPWorker 持有
  └── MTPGraphRunner       (草稿生成图)      ← 由 MTPWorker 持有
```

`ModelRunner` 作为顶层编排者，通过持有各组件实例来协调整个 decode 流程。每个组件职责单一、高内聚低耦合，可以独立理解和修改。

### 5. 端到端执行流程

以 Qwen3.5-397B-A17B-FP8 在 8×H200（attention_dp=8, ffn_ep=8）上的一次完整推理为例：

```
请求到达
  │
  ▼
Prefill Phase
  │  标准 prefill forward → 采样 token_0
  │  MTP: generate_and_store() → 产出 draft_1
  │  保存 _prev_drafts = [draft_1], _prev_sampled = token_0
  │
  ▼
Decode Step 1 (有 draft)
  │  Phase 1: 拼接 [token_0, draft_1] → seqlen_q=2 → LazyVerifyGraphRunner.replay()
  │  Phase 2: verify_logit 采样 target_pred
  │           target_pred == draft_1?
  │           ├─ Yes: 接受! 从 bonus_logit 采样 token_2, 输出 [draft_1, token_2]
  │           └─ No:  拒绝, 回滚 GDN 状态, 输出 [target_pred]
  │  Phase 3: MTP generate_and_store() → 产出新的 draft
  │
  ▼
Decode Step 2 (有 draft)
  │  ... 重复上述三阶段 ...
  │
  ▼
直到 EOS 或 max_tokens
```

### 6. 性能表现

在 Qwen3.5-397B-A17B-FP8（8×H200, attention_dp=8, ffn_ep=8, `num_speculative_tokens=1`）上的单请求测试：

| 指标                          | 数值      |
| ----------------------------- | --------- |
| ITL（Token 间延迟，不含排队） | ~17 ms    |
| Decode 吞吐                   | ~71 tok/s |
| E2E 延迟（131 tokens）        | ~3.26 s   |

CUDAGraph capture：5 decode graphs + 5 MTP graphs + 5 lazy verify graphs，所有 8 个 worker 均无报错，capture 耗时约 15 秒。

### 7. 设计取舍总结

| 设计决策                | 取舍                  | 理由                                       |
| ----------------------- | --------------------- | ------------------------------------------ |
| seqlen_q=2 而非树状推测 | 每步只验证 1 个 draft | 状态管理简洁，CUDAGraph 友好，无指数级膨胀 |
| 独立的三路 Graph Runner | 额外的 buffer 显存    | 职责清晰，capture/replay 解耦，便于调试    |
| GDN 双倍状态池          | 2× 显存开销           | 快照/回滚零重计算，无需 recompile          |
| MTP 低延迟 EP 模式      | 上下文切换开销        | 确保 draft 生成的专家路由延迟最小          |
| loop_count 膨胀         | 调度器多分配 block    | 提前预留 KV Cache，运行时无 OOM 风险       |
| 组合模式拆分            | 多文件、多类          | 高内聚低耦合，单文件可读，便于独立修改     |

### 8. 未来方向

- **多 draft 验证 (num_speculative_tokens > 1)**：当前 lazy verify 每步验证 1 个 draft，后续可探索 seqlen_q=N+1 的多 token 并行验证
- **自适应投机深度**：根据运行时接受率动态调整 draft 数量，在高接受率时激进投机，低接受率时退回标准 decode
- **与 EPLB 协同**：将 MTP 的 draft token 纳入专家负载均衡（EPLB）的统计，优化 EP 场景下的热点专家调度
