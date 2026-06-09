## NanoDeploy - NSA 原生稀疏注意力

### 1. 背景：为什么需要稀疏注意力？

DeepSeek V3.2 引入了 **NSA（Native Sparse Attention）**——一种在训练阶段就内置的稀疏注意力机制。与 MLA（Multi-Latent Attention）配合使用时，NSA 通过学习一个轻量级 Indexer 网络来判断每个 decode token 真正需要关注哪些历史 token，从而在不损失生成质量的前提下大幅减少 decode 阶段的计算量。

传统的 Full Attention decode 需要对所有历史 token 计算注意力分数。对于长上下文场景（V3.2 支持 1024K context），这意味着每个 decode step 的注意力计算量与序列长度线性增长。NSA 的核心思想是：**大部分历史 token 对当前生成的贡献很小，只需要关注最重要的那一小部分**。

V3.2 的 Indexer 在每一层独立运行，对所有 KV cache block 打分后选出 top-k 个最相关的 token 位置，然后只对这些位置执行稀疏注意力。这种"先粗筛后精算"的策略使得 decode 的注意力复杂度从 $O(n)$ 降至 $O(k)$（其中 $k = 2048$ 是固定的 top-k 常数）。

### 2. 核心设计：FP8 KV Cache + Indexer + Sparse Decode

NSA 的工程落地需要三个紧密协作的模块：**FP8 KV Cache** 提供量化存储，**Indexer** 负责块级打分和 top-k 选择，**Sparse Decode** 执行稀疏注意力计算。

```
┌──────────────────────────────────────────────────────────────────────┐
│                      一次 Decode Step (NSA)                          │
├─────────────────┬──────────────────────┬─────────────────────────────┤
│  Stage 1        │  Stage 2             │  Stage 3                    │
│  MLA Q/KV 投影   │  Indexer 打分 + TopK  │  Sparse Decode              │
│                 │                      │                             │
│  q_lora → Q     │  Q*K(FP8) → logits   │  FlashMLA(q, kv_cache,      │
│  hidden → KV    │  + gate weights      │           indices)          │
│  KV → FP8 Pack  │  TopK → 2048 slots   │  → output                  │
│  → paged cache  │  logical → physical  │                             │
└─────────────────┴──────────────────────┴─────────────────────────────┘
```

#### 2.1 FP8 KV Cache：656 字节/token 的量化存储

V3.2 的 MLA 使用 compressed KV，原始维度为 576（512 NoPE + 64 RoPE）。直接存 BF16 需要 1152 字节/token。NSA 的 FP8 方案将 NoPE 部分量化为 FP8（float8_e4m3fn），RoPE 部分保持 BF16 精度：

```
Per-token layout (656 bytes):
  [0..511]    512 bytes  float8_e4m3fn  — NoPE（4 tiles × 128，per-tile 量化）
  [512..527]   16 bytes  4 × float32    — per-tile scale factors（UE8M0 格式）
  [528..655]  128 bytes  64 × bfloat16  — RoPE（保持原始精度）
```

量化采用 **per-tile UE8M0** scale：将 512 维 NoPE 分为 4 个 128 维 tile，每个 tile 独立计算 absmax / 448.0 后向上取整为 2 的幂次（UE8M0 格式），这保证了与 deep_gemm FP8 GEMM kernel 的兼容性。

存储时通过 Triton kernel 将量化和 scatter 融合为单个 kernel call：

```python
@triton.jit
def _store_kcache_fp8_kernel(kv_ptr, cache_ptr, slot_mapping_ptr, ...):
    pid = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + pid)
    if slot == -1:
        return

    # 逐 tile 量化 NoPE → FP8 + scale
    for tile_idx in tl.static_range(NUM_TILES):
        tile_vals = tl.load(kv_ptr + pid * kv_row_stride + tile_offs)
        absmax = tl.max(tl.abs(tile_vals))
        scale = tl.math.pow2(tl.math.ceil(tl.math.log2(absmax / 448.0)))
        quantized = (tile_vals / scale).to(tl.float8e4nv)
        tl.store(cache_ptr + cache_base + tile_offs, quantized)
        tl.store(cache_ptr + scale_offset, scale, dtype=tl.float32)

    # RoPE 部分直接拷贝（BF16 → BF16）
    rope_bf16 = tl.load(kv_ptr + pid * kv_row_stride + D_NOPE + rope_offs)
    tl.store(cache_ptr + rope_out_offset + rope_offs, rope_bf16)
```

> **显存节约**：相比 BF16（1152 字节/token），FP8 方案（656 字节/token）节省约 43% 显存，这对长上下文场景至关重要。

#### 2.2 Stride Padding 的工程细节

FlashMLA kernel 在处理 FP8 cache 时对内存对齐有要求。NanoDeploy 采用 **block_size+1 padding** 策略：

```python
# 分配时多一行 padding
kv_cache_padded = torch.empty(
    1, num_layers, num_blocks,
    block_size + 1,  # 65 行（64 数据 + 1 padding）
    1, fp8_head_dim,
    dtype=torch.float8_e4m3fn,
)
# 对外暴露前 block_size 行
self.kv_cache = kv_cache_padded[:, :, :, :block_size, :, :]
```

这意味着物理 stride（每 block 的字节间距）是 `(block_size + 1) * head_dim = 65 * 656 = 42640` 字节，而非直觉上的 `64 * 656 = 41984`。这个 stride 差异在 RDMA 传输中必须被正确处理（详见第 5 节）。

#### 2.3 Indexer：块级打分与 TopK 选择

Indexer 是每层独立的轻量级网络，负责判断当前 query 应该关注哪些历史 token：

```
Forward Flow:
  ┌─────────────┐
  │ q_lora (1536)│──→ wq_b ──→ query (64 heads × 128 dim)
  └─────────────┘              │
  ┌──────────────┐             │ RoPE + Hadamard rotation
  │hidden (7168) │──→ wk ──→ key (128 dim)
  └──────────────┘    │        │
                      │        ├──→ FP8 quantize query
                      │        └──→ FP8 quantize + store key to IndexerCache
                      │
                      └──→ weights_proj ──→ gate weights (64 heads)
                                            │
                                   deep_gemm.fp8_paged_mqa_logits
                                   (q_fp8, kv_cache_fp8, weights, ...)
                                            │
                                     TopK(logits, k=2048)
                                            │
                                     topk_indices (2048 个 token 位置)
```

关键设计点：

1. **共享 block table**：Indexer cache 使用与主 KV cache 相同的 page table 和 block size（64），因此 Indexer 的 top-k 结果可以直接映射到主 KV cache 的物理地址。

2. **FP8 Paged MQA Logits**：调用 `deep_gemm.fp8_paged_mqa_logits` 执行 FP8 精度的 paged MQA（Multi-Query Attention），一次性为所有 64 个 indexer head 计算 logits。这个 kernel 直接读取 paged FP8 cache，避免了 dequantize 的额外开销。

3. **Gate 加权**：每个 query token 通过 `weights_proj` 线性层产出 64 个 gate 权重，与 FP8 MQA logits 逐元素相乘后做 TopK，这相当于一种 learned 的注意力分数偏置。

4. **Hadamard 旋转**：对 query 和 key 都施加 Hadamard 变换（`fast_hadamard_transform`），这是 V3.2 训练时的设计选择，推理时必须忠实复现。

5. **UE8M0 Graph-Safe 量化**：标准的 `deep_gemm.per_token_cast_to_fp8` 内部调用了 `.item()`，会打断 CUDA Graph 捕获。NanoDeploy 实现了纯 tensor 操作的替代版本（`_per_token_cast_to_fp8_ue8m0`），通过 `torch.exp2(torch.ceil(torch.log2(...)))` 代替 `.item()` 来计算 UE8M0 scale。

#### 2.4 IndexerCache：分层分页的 FP8 缓存

Indexer 需要自己的 KV cache 来存储历史 key（与主 KV cache 分开，因为 Indexer 的 key 维度和量化方式不同）：

```
IndexerCache Layout:
  buffer: (num_layers, num_pages, page_size * bytes_per_token) uint8

  每个 token 占 132 bytes:
    [0..127]   128 bytes  float8_e4m3fn  — FP8 key 数据
    [128..131]   4 bytes  float32        — per-token scale

  每个 page: 64 tokens × 132 bytes = 8448 bytes
  每层: num_pages × 8448 bytes
  总计: 61 layers × num_pages × 8448 bytes（约为主 KV cache 的 ~20%）
```

整个 buffer 是一块连续的 `torch.uint8` 张量，这使得 RDMA 注册只需一次 `register_memory_region` 调用。

### 3. Sparse Decode：物理地址转换与 FlashMLA

Indexer 输出的 top-k indices 是逻辑 token 位置（0 ~ context_len-1），但 FlashMLA 稀疏 decode kernel 需要物理 slot 索引（block_idx × block_size + offset）。`topk_indices_to_physical` 负责这一转换：

```python
def topk_indices_to_physical(topk_indices, block_table, block_size):
    """逻辑 token 位置 → 物理 paged cache 索引"""
    valid_mask = topk_indices >= 0
    safe_indices = topk_indices.clamp(min=0)

    logical_block = safe_indices // block_size     # 哪个 page
    offset_in_block = safe_indices % block_size    # page 内偏移

    # 查 page table: logical_block → physical_block
    physical_block = torch.gather(block_table, dim=1, index=logical_block.long())
    physical_indices = physical_block * block_size + offset_in_block

    # 无效位置保持 -1（FlashMLA 的稀疏 kernel 会跳过 -1）
    physical_indices = torch.where(valid_mask, physical_indices, -1)
    return physical_indices
```

转换后，调用 FlashMLA 的稀疏 decode 接口：

```python
o, lse = flash_mla.flash_mla_with_kvcache(
    q.reshape(bs, ntps, num_head, head_dim),
    k_cache,             # FP8 paged cache
    None,                # block_table (稀疏模式不需要)
    None,                # cache_seqlens (稀疏模式不需要)
    v_head_size,
    sparse_meta,
    None,                # num_splits
    scale,
    False,               # causal=False（稀疏模式由 indices 控制范围）
    is_fp8_kvcache=True,
    indices=indices_3d,  # (bs, ntps, topk) — 物理 slot 索引
)
```

> **关键细节**：稀疏模式下 `causal=False`，因为 token 的因果关系已经由 Indexer 的 logits masking 保证（超出 context_len 的位置在 TopK 前被设为 `-inf`）。`block_table` 和 `cache_seqlens` 传 None，所有寻址完全由 `indices` 驱动。

### 4. CUDA Graph 集成

NSA 的 Indexer 和 Sparse Decode 都需要在 CUDA Graph 中运行。这带来了几个工程挑战：

#### 4.1 Graph-Safe 的 FP8 量化

Indexer 内部的 FP8 量化不能使用标准的 `deep_gemm.per_token_cast_to_fp8`（因为它内部有 `.item()` 调用），必须使用纯 tensor 实现的 `_per_token_cast_to_fp8_ue8m0`：

```python
def _per_token_cast_to_fp8_ue8m0(x):
    """Graph-safe per-token FP8 quantization with UE8M0 scales."""
    # Pad to 128-byte alignment
    x_view = x_padded.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(min=1e-4)
    sf = x_amax / 448.0
    # UE8M0: ceil to power of 2（无 .item() 调用）
    sf = torch.exp2(torch.ceil(torch.log2(sf)))
    x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn)
    return x_fp8, sf
```

#### 4.2 IndexerCache 的 Scatter 与 Graph 兼容

`store_key_fp8` 使用 `torch.Tensor.scatter_` 将 FP8 key 写入 paged cache。对于无效的 slot（`slot_mapping == -1`），不能使用 boolean mask（动态 shape 会打断 graph），而是将 -1 clamp 到 0，让无效数据写入 slot 0（后续会被真实数据覆盖）：

```python
safe_slots = torch.where(
    slot_mapping >= 0, slot_mapping,
    torch.zeros_like(slot_mapping)
)
```

#### 4.3 Sparse Tile Scheduler Metadata

FlashMLA 的稀疏 decode kernel 需要 `sparse_tile_scheduler_metadata`。NanoDeploy 通过 Context 单例管理这个状态，首次调用时创建，后续 graph replay 复用：

```python
sparse_meta = context.sparse_tile_scheduler_metadata
if sparse_meta is None:
    sparse_meta, _ = flash_mla.get_mla_metadata()
# ... 执行 sparse decode ...
context.sparse_tile_scheduler_metadata = sparse_meta  # 保存供后续 replay
```

#### 4.4 Warmup 期间的 Dummy Indices

CUDA Graph capture 时 Indexer 尚未有真实数据产出。此时构造全 -1 的 dummy indices，让 sparse decode kernel 走完整执行路径但不读取任何有效数据：

```python
if k_cache.dtype == torch.float8_e4m3fn and sparse_indices is None:
    sparse_indices = torch.full(
        (bs * ntps, nsa_index_topk), -1,
        dtype=torch.int32, device=q.device,
    )
```

### 5. PD 分离场景下的 RDMA 传输

在 Prefill-Decode 分离（PD Disaggregation）部署中，prefill 节点完成首次推理后需要将 KV cache 迁移到 decode 节点。NSA 模型需要额外迁移 **IndexerCache**。

#### 5.1 三路 Memory Region 注册

NanoDeploy 为每种缓存独立注册 RDMA Memory Region：

| 缓存类型     | 张量形状                           | MR 注册路径                 |
| ------------ | ---------------------------------- | --------------------------- |
| 主 KV Cache  | `(1, L, N, 65, 1, 656)` FP8        | `_local_mr_handler`         |
| GDN States   | `(L_gdn, max_slots, D_conv/D_rec)` | `_local_gdn_*_mr_handler`   |
| IndexerCache | `(L, N, 64*132)` uint8             | `_local_indexer_mr_handler` |

注意主 KV Cache 的物理维度是 `block_size + 1 = 65`（含 padding 行），RDMA 传输时必须使用这个物理 stride：

```python
def block_stride(self, block_idx):
    if self.is_fp8_kvcache and self.mode == "mla":
        # 物理 stride: (block_size + 1) * head_dim = 65 * 656 = 42640
        return block_idx * (self.block_size + 1) * 1 * self._fp8_head_dim * 1
    return block_idx * self.block_size * self.num_local_kv_heads * self.head_dim * self.dtype.itemsize
```

#### 5.2 IndexerCache 的 Stride 计算

IndexerCache 是连续的 `(num_layers, num_pages, page_bytes)` uint8 buffer，stride 计算相对简单：

```python
def local_indexer_stride(self, layer_idx, block_idx):
    page_bytes = self.indexer_cache.page_size * self.indexer_cache.bytes_per_token
    return (layer_idx * self.num_local_kvcache_blocks + block_idx) * page_bytes

def remote_indexer_stride(self, layer_idx, block_idx, remote_engine_id):
    page_bytes = self.indexer_cache.page_size * self.indexer_cache.bytes_per_token
    return (layer_idx * self.num_remote_kvcache_blocks[remote_engine_id] + block_idx) * page_bytes
```

RDMA 传输时，对每个需要迁移的 block，组装 `(local_mr, remote_mr, target_offset, source_offset, length)` 五元组，通过 DLSlime 的 batch RDMA read 一次性提交。

### 6. 显存预算与调度

NSA 引入了额外的显存开销（IndexerCache），调度器在分配 KV cache block 时必须一并考虑：

```python
# 每个 block 的总字节数 = 主 KV Cache + IndexerCache
block_bytes = num_layers * block_size * 1 * fp8_head_dim * 1  # 主 KV
if index_head_dim > 0:
    indexer_bpt = index_head_dim + index_head_dim // 128 * 4  # 128 + 4 = 132
    block_bytes += num_layers * block_size * indexer_bpt       # Indexer
```

对于 V3.2（61 层，block_size=64），每个 block 的显存开销：

| 组件         | 计算公式      | 字节数                       |
| ------------ | ------------- | ---------------------------- |
| 主 KV Cache  | 61 × 64 × 656 | 2,561,024                    |
| IndexerCache | 61 × 64 × 132 | 515,328                      |
| **合计**     |               | **3,076,352（~3 MB/block）** |

`num_local_kvcache_blocks` 的计算公式为：

$$\\text{num_blocks} = \\frac{\\text{GPU_total} \\times \\text{utilization} - \\text{model_used}}{\\text{block_bytes}}$$

### 7. 配置与开关

NSA 默认对 V3.2 模型自动启用（通过检测 `index_head_dim > 0`）。用户可以通过 `--disable_nsa` 参数关闭稀疏注意力，退回标准的 BF16 dense MLA decode：

```python
# config.py
disable_nsa: bool = False  # NSA 默认开启

# model_runner.py — KV cache 类型决策
index_head_dim = getattr(hf_config, "index_head_dim", 0)
is_fp8_kvcache = (mode == "mla" and index_head_dim > 0) and not getattr(
    config, "disable_nsa", False
)
```

关闭 NSA 后：

- KV cache 回退为 BF16（1152 字节/token）
- 不分配 IndexerCache
- Decode 走 dense FlashMLA 路径（无 Indexer 打分）
- 显存占用增加但避免了 Indexer 的计算开销（短上下文场景）

### 8. 端到端执行流程

以 DeepSeek-V3.2-Exp 在 8×H200（attention_dp=8, ffn_ep=8）上的一次 decode step 为例：

```
输入: [token_id], position, context_lens, block_tables
  │
  ▼
Layer 0..60 循环:
  │
  ├─ MLA Q/KV 投影
  │    q_lora = q_a_proj(hidden) + q_a_layernorm
  │    compressed_kv = kv_a_proj(hidden) + kv_a_layernorm
  │    KV 写入 FP8 paged cache (store_kcache_fp8)
  │
  ├─ Indexer.forward(hidden, q_lora, positions, context_lens, block_tables, slot_mapping)
  │    ├─ wq_b: q_lora → query (64, 128)
  │    ├─ wk: hidden → key (128), k_norm, RoPE, Hadamard
  │    ├─ RoPE + Hadamard on query
  │    ├─ FP8 quantize query (graph-safe UE8M0)
  │    ├─ Store FP8 key to IndexerCache
  │    ├─ Gate weights: weights_proj(hidden) → (64,)
  │    ├─ deep_gemm.fp8_paged_mqa_logits (FP8 Q × FP8 IndexerCache)
  │    └─ TopK(logits, k=2048) → topk_indices (逻辑位置)
  │
  ├─ topk_indices_to_physical(topk_indices, block_tables, 64)
  │    → sparse_indices (物理 slot 索引)
  │
  ├─ Sparse Decode: flash_mla.flash_mla_with_kvcache(
  │      q, k_cache_fp8, indices=sparse_indices, is_fp8_kvcache=True)
  │    → attention output
  │
  ├─ MLP / MoE (All-to-All EP routing)
  │
  └─ → next layer hidden_states
  │
  ▼
Final: LM Head → logits → sample → output token
```

### 9. 与 MTP 的协同

当 NSA 和 MTP（投机解码）同时启用时，三路 CUDAGraph Runner 都在稀疏模式下运行：

| Runner                  | seqlen_q | 稀疏/密集 | 说明                             |
| ----------------------- | -------- | --------- | -------------------------------- |
| `DecodeGraphRunner`     | 1        | 稀疏      | 标准 decode（首步，无 draft 时） |
| `LazyVerifyGraphRunner` | 2        | 稀疏      | MTP 验证 + bonus 采样            |
| `MTPGraphRunner`        | 1        | N/A       | 草稿生成（无 KV Cache，纯 MLP）  |

LazyVerify 的 `seqlen_q=2` 模式下，Indexer 对两个 token 联合打分，产出 2 组 top-k indices，分别驱动两个位置的稀疏注意力。MTP GraphRunner 不涉及 KV Cache 操作，因此与 NSA 无交互。

### 10. 设计取舍总结

| 设计决策                    | 取舍                                  | 理由                                               |
| --------------------------- | ------------------------------------- | -------------------------------------------------- |
| FP8 量化 NoPE + BF16 RoPE   | 额外的 pack/unpack 开销               | RoPE 精度敏感，NoPE 可安全量化至 FP8               |
| Per-tile UE8M0 scale        | 4 个 float32 overhead/token           | 与 deep_gemm kernel 对齐，无需额外 dequantize 路径 |
| block_size+1 stride padding | 每 block 浪费 656 字节                | FlashMLA FP8 kernel 的对齐要求                     |
| 独立的 IndexerCache         | 额外 ~20% 显存开销                    | 与主 KV Cache 解耦，Indexer key 维度不同           |
| 共享 page table             | 耦合 Indexer 和主 cache 的 block 分配 | 简化 top-k → 物理地址转换，无需双 page table       |
| Graph-safe FP8 量化         | 纯 tensor 实现略慢于 native deep_gemm | 兼容 CUDA Graph capture，无 `.item()` 调用         |
| RDMA 三路 MR 注册           | 连接建立复杂度增加                    | 单次 batch RDMA read 迁移全部状态                  |
| `--disable_nsa` 回退        | 需要维护两条 decode 路径              | 短上下文 / 调试场景可退回 dense decode             |

### 11. 未来方向

- **动态 TopK**：根据层的位置/序列长度自适应调整 $k$ 值，浅层可能需要更少的 token，深层需要更多
- **Indexer 权重共享**：探索跨层共享 Indexer 权重以减少参数量和延迟
- **Streaming 预填充 Indexer key**：在 chunked prefill 中流式填充 IndexerCache，避免 prefill 完成后的批量写入
- **稀疏注意力 + Prefix Caching 协同**：利用 Indexer 的打分信息智能决定哪些 prefix block 值得缓存
