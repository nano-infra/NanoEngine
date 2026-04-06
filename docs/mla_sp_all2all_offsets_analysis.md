# NanoDeploy MLA(Deepseek) SP推理路径中 all2all offsets 分析

**日期**: 2026-04-05  
**通信后端**: hao_basic  
**关注路径**: CUDA Graph (非Eager)  
**分析来源**: 实际推理代码路径（非测试代码）

## 目录
1. [概述](#概述)
2. [实际推理路径追踪](#实际推理路径追踪)
3. [offsets 的作用](#offsets-的作用)
4. [offsets 的计算与设置](#offsets-的计算与设置)
5. [输入输出矩阵形状变化](#输入输出矩阵形状变化)
6. [数据布局的变化对比](#数据布局的变化对比)
7. [CUDA Kernel 实现细节](#cuda-kernel-实现细节)
8. [总结](#总结)

---

## 概述

### 推理流程概览

1. **准备阶段** (`model_runner.py::prepare_decode`)
   - C++ 计算元数据 (`prepare_decode_cpp`)
   - 生成 `q_offsets` 等张量
   - 设置到 Context

2. **图捕获阶段** (`model_runner.py::init_cuda_graphs`)
   - 预先捕获多个 CUDA Graph
   - `q_offsets` 固定为零初始化的占位符

3. **执行阶段** (`model_runner.py::run_model`)
   - 从 Context 复制实际的 `q_offsets` 到 graph_vars
   - 执行 `graph.replay()`
   - Attention 层读取并使用 `q_offsets`

## 实际推理路径追踪

### 完整调用链

```
model_runner.run()
  ↓
model_runner.prepare_decode(dp_seqs)  [每次推理调用]
  ↓
prepare_decode_cpp(...)  [C++]
  ↓ 统计每个 rank 有效请求数
  ↓ 计算 q_offsets = cumsum(sp_valid_request_counts)
  ↓ 返回 DecodeMetadata
  ↓
set_context(q_offsets=q_offsets, ...)  [设置到全局 Context]
  ↓
model_runner.run_model(input_ids, positions, is_prefill=False)
  ↓ [CUDA Graph 路径]
graph_vars["q_offsets"].copy_(context.q_offsets)  [line 613-614]
  ↓
graph.replay()  [执行预捕获的图]
  ↓ [在图内部]
model.forward(input_ids, positions)
  ↓
attention_layer.forward(q, k, v, k_cache, v_cache)
  ↓ [MLA Decode with SP]
q_buffer.all_to_all_ll(q, mask=q_mask, offsets=context.q_offsets)  [line 261-265]
  ↓
HaoAllToAllBufferAdapter.all_to_all_ll(...)  [Python 后端适配]
  ↓
AllToAllBuffer.all_to_all(...)  [DLSlime C++]
  ↓
intranode_alltoall_kernel<<<...>>>(offsets_ptr)  [CUDA Kernel]
```

### 关键代码位置

| 阶段 | 文件 | 行号 | 说明 |
|------|------|------|------|
| **元数据计算** | `csrc/nanodeploy/worker/model_runner_utils.cpp` | 217-236 | 统计 `sp_valid_request_counts` |
| | | 323-328 | 计算 `q_offsets = cumsum` |
| **准备 Decode** | `nanodeploy/worker/model_runner.py` | 350-430 | 调用 C++ 并转换为 GPU tensor |
| **设置 Context** | `nanodeploy/worker/model_runner.py` | 447-472 | `set_context(q_offsets=...)` |
| **Graph 执行** | `nanodeploy/worker/model_runner.py` | 613-615 | 复制到 graph_vars 并 replay |
| **Attention Q** | `nanodeploy/layers/attention.py` | 68-89 | FlashAttentionImpl: Q all2all |
| **MLA Q** | `nanodeploy/layers/attention.py` | 261-265 | FlashMLAImpl: Q all2all |
| **后端适配** | `nanodeploy/worker/sp_backend.py` | 173-219 | HaoAllToAllBufferAdapter |
| **DLSlime 入口** | `DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.cpp` | 168-279 | `dispatch_basic` |
| **CUDA Kernel** | `DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/all_to_all_intra_ll.cu` | 106-199 | `intranode_alltoall_kernel` |

---

## offsets 的核心概念

在 NanoDeploy 的 MLA (Multi-head Latent Attention，Deepseek V2/V3 架构) Sequence Parallel (SP) 推理路径中，`offsets` 参数用于优化 all-to-all 通信中的数据分发策略。当启用 offsets 时，系统可以处理**不均匀的批次分布**，每个 rank 发送/接收的数据量可以不同。

### 关键文件位置

- **Python 层调用**: `nanodeploy/layers/attention.py` (line 88, 264)
- **后端适配器**: `nanodeploy/worker/sp_backend.py` (line 173-214)
- **元数据计算**: `csrc/nanodeploy/worker/model_runner_utils.cpp` (line 323-328)
- **CUDA Kernel**: `DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/all_to_all_intra_ll.cu`

---

## offsets 的作用

### 核心功能

`offsets` 是一个长度为 `sp_size + 1` 的整型数组（int32），用于描述每个 SP rank 在 all-to-all 通信中**实际有效请求数量的累积和**（cumulative sum）。

**实际计算逻辑** (from `model_runner_utils.cpp` line 217-236):
```cpp
// 统计每个 SP rank 对当前 rank 有多少有效请求
std::vector<int> sp_valid_request_counts(sp_size, 0);

for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
    const auto& batch_seqs = sp_seqs[sp_idx];  // 属于 sp_idx 的序列
    int valid_count = 0;
    for (int seq_id = 0; seq_id < max_num_seqs; ++seq_id) {
        if (seq_id < (int)batch_seqs.size()) {
            Sequence* seq = batch_seqs[seq_id];
            // 关键：检查该序列在当前 sp_rank 上的 context_len
            int ctx_len = seq->context_len(BlockContextSlot::ACTIVE, sp_rank);
            if (ctx_len > 0) {
                valid_count++;  // 该序列在当前 rank 上有数据，计数
            }
        }
    }
    sp_valid_request_counts[sp_idx] = valid_count;
}

// q_offsets 就是 sp_valid_request_counts 的累积和
q_offsets[0] = 0;
for (int i = 0; i < sp_size; ++i) {
    q_offsets[i + 1] = q_offsets[i] + sp_valid_request_counts[i];
}
```

**关键理解**：`sp_valid_request_counts[sp_idx]` 表示"master 在 sp_idx 的所有序列中，有多少序列在当前 sp_rank 上有数据（ctx_len > 0）"。

```python
# 例子: sp_size = 3, 每个 rank 的有效请求数为 [2, 3, 1]
q_offsets = [0, 2, 5, 6]
#            ^  ^  ^  ^
#         start R0 R1 R2 end
```

### 设计目的

1. **负载不均衡场景优化**: 在 decode 阶段，不同 SP rank 上的有效请求数量可能不同（某些 rank 可能没有请求）
2. **避免无效通信**: 没有启用 offsets 时，即使某个 rank 没有数据，也需要按照 `max_batch_size` 分配空间
3. **紧凑数据布局**: 启用 offsets 后，输出 buffer 中的数据按照实际有效请求数量紧密排列

---

## offsets 的计算与设置

### C++ 侧计算逻辑

在 `csrc/nanodeploy/worker/model_runner_utils.cpp` 的 `prepare_decode_cpp` 函数中：

```cpp
// 1. 统计每个 SP rank 的有效请求数
std::vector<int> sp_valid_request_counts(sp_size, 0);

for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
    const auto& batch_seqs = sp_seqs[sp_idx];
    int valid_count = 0;
    for (int seq_id = 0; seq_id < max_num_seqs; ++seq_id) {
        if (seq_id < (int)batch_seqs.size()) {
            int ctx_len = batch_seqs[seq_id]->context_len(BlockContextSlot::ACTIVE, sp_rank);
            if (ctx_len > 0) {
                valid_count++;
            }
        }
    }
    sp_valid_request_counts[sp_idx] = valid_count;
}

// 2. 计算 offsets (累积和)
meta.q_offsets.resize(sp_size + 1);
meta.q_offsets[0] = 0;
for (int i = 0; i < sp_size; ++i) {
    meta.q_offsets[i + 1] = meta.q_offsets[i] + sp_valid_request_counts[i];
}
```

### 实际场景示例

假设 `sp_size = 2`, `max_num_seqs = 4`，当前在 **Rank 0** 执行：

**序列分布**：
- 3 个序列 master 在 Rank 0: seq_A, seq_B, seq_C
- 1 个序列 master 在 Rank 1: seq_D

**各序列在各 rank 上的 KV 分布**（根据 Sequence Parallel 策略）：
- seq_A: 在 Rank0 有数据（ctx_len=100），在 Rank1 有数据（ctx_len=50）
- seq_B: 仅在 Rank0 有数据（ctx_len=200），Rank1 无数据（ctx_len=0）
- seq_C: 仅在 Rank1 有数据（ctx_len=150），Rank0 无数据（ctx_len=0）
- seq_D: 在 Rank0 有数据（ctx_len=80），在 Rank1 有数据（ctx_len=120）

**在 Rank 0 上计算 `sp_valid_request_counts`**：
```python
sp_valid_request_counts[0] = 2  # Rank0's seqs: seq_A(有), seq_B(有), seq_C(无) -> 2个有效
sp_valid_request_counts[1] = 1  # Rank1's seqs: seq_D(有) -> 1个有效
```

**计算 `q_offsets`**：
```python
q_offsets = [0, 2, 3]
#            ↑  ↑  ↑
#         start R0 R1 end
```

**含义**：在 Rank 0 的视角下
- 来自 Rank 0 的 Q（seq_A, seq_B 的 Q）会被写入输出 buffer 的 `[0:2]` 行
- 来自 Rank 1 的 Q（seq_D 的 Q）会被写入输出 buffer 的 `[2:3]` 行
- seq_C 不会参与（因为它在 Rank 0 上没有 KV 数据）

---

## 输入输出矩阵形状变化

### MLA Q 路径的 all-to-all 调用流程

**位置**: `nanodeploy/layers/attention.py` line 246-265

#### 第一步：Q Copy 到 local buffer

```python
bs, num_head, head_dim = q.shape  # bs = 当前 rank master 的序列数
max_num_seqs = get_sp_context().max_num_seqs

# 准备 local buffer（每个 rank 都有自己的 IPC 共享内存）
local_q_buffer_3d = q_buffer.local_buffer.view(dtype)[
    : sp_size * max_num_seqs * num_head * head_dim
].view(sp_size * max_num_seqs, num_head, head_dim)

# 使用 copy_batch_indexed_triton 将 Q 按照索引拷贝到指定位置
copy_batch_indexed_triton(
    q.view(bs, num_head, head_dim),  # 源：当前 rank 的 Q
    local_q_buffer_3d,                # 目标：IPC 共享内存 buffer
    context.q_slice_get,              # 从 q 中取哪些行
    context.q_slice_fill,             # 写入到 buffer 的哪些位置
    context.q_copy_mask,              # 拷贝掩码
)
```

**关键**：这一步将当前 rank 生成的 Q 按照特定索引排列到 local buffer，为后续 all-to-all 做准备。

#### 第二步：All-to-All 通信

```python
q = q_buffer.all_to_all_ll(
    q.view([bs, -1]),           # 输入：[bs, num_head * head_dim]
    mask=context.q_mask,         # mask: [sp_size, max_num_seqs]
    offsets=context.q_offsets,   # offsets: [sp_size + 1]
).view([sp_size * max_num_seqs, num_head, head_dim])
```

**关键点**：
- 输入 `q` 形状是 `[bs, msg_dim]`，其中 `bs` 是当前 rank master 的实际序列数
- 输出固定是 `[sp_size, max_num_seqs, msg_dim]`，但有效数据的分布由 offsets 控制

### 形状对比表

| 场景 | 输入形状 | 输出形状 | 说明 |
|------|---------|---------|------|
| **无 offsets** | `[max_bs, msg_dim]` | `[sp_size, max_bs, msg_dim]` | 每个 rank 固定分配 max_bs 行 |
| **有 offsets** | `[actual_bs, msg_dim]` | `[sp_size, max_bs, msg_dim]` | 每个 rank 根据实际需求分配 |

**关键差异**：
- **无 offsets**: 输入必须 padding 到 `max_bs` 行（即使实际只有少量数据）
- **有 offsets**: 输入可以是实际的 `actual_bs` 行，无需 padding

### hao_basic 后端的处理

在 `nanodeploy/worker/sp_backend.py` 中：

```python
def all_to_all_ll(self, x, is_transpose=False, mask=None, offsets=None):
    if offsets is not None:
        if is_transpose:
            raise NotImplementedError(
                "hao_basic offsets only support non-transpose all-to-all."
            )
        # offsets 模式下，输入是 actual_bs，不需要 padding
        batch_size = x.size(0)  # actual_bs
    elif mask is not None and not is_transpose:
        # 非 offsets 的 mask 模式，需要 padding 到 max_bs
        backend_x = self._pad_masked_non_transpose_input(x, mask)
```

**关键逻辑**：
- **offsets 路径**: `batch_size = total_rows` (line 205, 229)
- **非 offsets 路径**: 需要检查并 padding 输入到 `max_bs` (line 199-200)

---

## 数据布局的变化对比

### 场景设置

- `sp_size = 2` (Rank 0, Rank 1)
- `max_num_seqs = 4`
- `num_heads = 2`, `head_dim = 8`
- 实际请求分布：
  - Rank 0: 2 个请求
  - Rank 1: 1 个请求

### 无 offsets 模式（NanoDeploy 不使用，仅作对比）

#### 理论输入布局
```
Rank 0 输入 (必须 padding 到 max_bs=4):
  [seq0_data]  ← 实际数据
  [seq1_data]  ← 实际数据
  [padding]    ← 填充0
  [padding]    ← 填充0
  shape: [4, 16]  (4 = max_bs, 16 = num_heads * head_dim)

Rank 1 输入 (必须 padding 到 max_bs=4):
  [seq0_data]  ← 实际数据
  [padding]    ← 填充0
  [padding]    ← 填充0
  [padding]    ← 填充0
  shape: [4, 16]
```

#### 理论输出布局
```
all_to_all_ll 输出 (每个 rank 都一样):
固定分配策略：每个源 rank 占据固定的 max_bs 行
  [rank0_slot0, rank0_slot1, rank0_slot2, rank0_slot3,  ← 来自 Rank 0 的 4 个 slot
   rank1_slot0, rank1_slot1, rank1_slot2, rank1_slot3]  ← 来自 Rank 1 的 4 个 slot
  shape: [2, 4, 16]  (sp_size=2, max_bs=4, msg_dim=16)
  
目标索引计算: dst_row_idx = local_rank * batch_size + token_i
  - Rank 0: 写入行 [0, 1, 2, 3]
  - Rank 1: 写入行 [4, 5, 6, 7]
```

**问题**：
1. 输入需要 padding 到 max_bs，浪费带宽
2. 输出 buffer 按固定间隔分配，大量空洞（例子中 5/8 = 62.5% 是 padding）
3. 后续 attention 需要知道哪些行是有效的

### 有 offsets 模式（NanoDeploy 实际使用）

#### 实际输入准备

**关键前置步骤**：在调用 all2all 前，通过 `copy_batch_indexed_triton` 将 Q 拷贝到 local buffer 的正确位置。

```python
# 假设 Rank 0 有 2 个序列 (seq_A, seq_B)，Rank 1 有 1 个序列 (seq_D)
# 在 Rank 0 上执行时：

# q: [2, num_head, head_dim]  ← Rank 0 的 2 个序列的 Q
# local_q_buffer_3d: [sp_size * max_num_seqs, num_head, head_dim]
#                    = [2 * 4, num_head, head_dim] = [8, num_head, head_dim]

# q_slice_get = [0, 1]           # 从 q 取第 0,1 行（seq_A, seq_B）
# q_slice_fill = [0, 1]          # 填充到 local_buffer 的第 0,1 位置
# q_copy_mask = [1, 1]           # 都是有效的

copy_batch_indexed_triton(q, local_q_buffer_3d, q_slice_get, q_slice_fill, q_copy_mask)

# 执行后 local_q_buffer_3d[0] = seq_A 的 Q
#        local_q_buffer_3d[1] = seq_B 的 Q
#        local_q_buffer_3d[2:] = 其他 rank 通过 IPC 访问后写入
```

#### all2all 调用

```python
# 输入：[bs, msg_dim] = [2, num_head * head_dim]
# 虽然输入是 2 行，但 all2all 会通过 IPC 共享内存读取所有 rank 的 local buffer
q_output = q_buffer.all_to_all_ll(
    q.view([2, -1]),              # 输入形状可以是实际的 bs
    mask=context.q_mask,          # [sp_size, max_num_seqs] = [2, 4]
    offsets=context.q_offsets,    # [0, 2, 3]
)
```

#### q_offsets 的实际含义

```python
q_offsets = [0, 2, 3]
#            ↑  ↑  ↑
#         start R0 R1 end

# 在当前 rank (Rank 0) 的视角：
# - offsets[0]=0: 起始位置
# - offsets[1]=2: Rank 0 的数据占据 [0:2)，共 2 行（seq_A, seq_B 的 Q）
# - offsets[2]=3: Rank 1 的数据占据 [2:3)，共 1 行（seq_D 的 Q）
```

#### 输出布局（在 Rank 0 上）

```
all_to_all_ll 输出: [sp_size, max_num_seqs, msg_dim] = [2, 4, 128]
  
实际数据排列（展平后看前 3 行，因为 q_offsets[-1]=3）:
  [0] seq_A 的 Q  ← 来自 Rank 0 local buffer[0]，由 offsets[0]=0 决定
  [1] seq_B 的 Q  ← 来自 Rank 0 local buffer[1]，由 offsets[0]+1 决定
  [2] seq_D 的 Q  ← 来自 Rank 1 local buffer[0]，由 offsets[1]=2 决定
  [3] 未定义     ← buffer 剩余空间
  [4:7] 第二个 rank 的 max_num_seqs slots（稀疏利用）
```

#### CUDA Kernel 中的索引计算

**位置**: `DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/all_to_all_intra_ll.cu` line 156-158

```cuda
// local_rank: 当前数据来自哪个 rank (0 或 1)
// token_i: 该 rank 内的第几个 token (0, 1, ...)
// dst_rank: 写入到哪个目标 rank 的 buffer (由 blockIdx 决定)

const uint64_t dst_row_idx = offsets != nullptr
    ? static_cast<uint64_t>(offsets[local_rank]) + token_i
    : static_cast<uint64_t>(local_rank) * batch_size + token_i;

// 有 offsets 时：
//   Rank 0, token_i=0: dst_row_idx = offsets[0] + 0 = 0 + 0 = 0
//   Rank 0, token_i=1: dst_row_idx = offsets[0] + 1 = 0 + 1 = 1
//   Rank 1, token_i=0: dst_row_idx = offsets[1] + 0 = 2 + 0 = 2
```

**关键优势**：
- Rank 0 的 2 个 token 写入连续的 `[0, 1]`
- Rank 1 的 1 个 token 紧接着写入 `[2]`
- 没有固定间隔的浪费，数据紧凑排列

### mask 的配合使用

**mask 在 offsets 模式下的作用** (`alltoall_buffer.cpp` line 198-206):

```cpp
if (has_offsets) {
    TORCH_CHECK(
        mask_batch_size == max_batch_size_,
        "Offset mask shape must be [world_size, max_batch_size]");
    batch_size = total_rows;  // 使用输入的实际行数
}
```

**mask 的设计**：
- 形状固定为 `[sp_size, max_batch_size]`
- 在 Rank 0 计算时，`mask[dst_rank][seq_slot]` 指示是否需要向 `dst_rank` 发送数据给第 `seq_slot` 个序列

**实际推理中的 q_mask** (`model_runner.py` line 389-394):

```python
# context_lens: [sp_size, max_num_seqs]，存储每个序列在各 rank 上的 ctx_len
# global_context_lens: [sp_size, max_num_seqs]，当前 rank master 的序列在各 rank 上的 ctx_len

# q_mask: 指示需要发送 Q 到哪些 rank
q_mask = global_context_lens.clone()
q_mask[sp_rank].fill_(0)  # 不发送给自己（自己已经有 Q 了）
q_mask[q_mask != 0] = 1    # 其他 ctx_len > 0 的位置设为 1
```

**配合 offsets 的工作流**：
1. Kernel 遍历 `token_i in [0, batch_size)` (batch_size 是输入实际行数)
2. 检查 `mask[dst_rank * mask_batch_size + token_i]` 是否为 0
3. 如果不为 0，计算 `dst_row_idx = offsets[local_rank] + token_i` 并写入
4. mask 和 offsets 协同工作，确保数据写入正确位置且不浪费带宽

---

## CUDA Kernel 实现细节

### 核心函数: `intranode_alltoall_kernel`

**位置**: `DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/all_to_all_intra_ll.cu`

### 关键代码段

#### 1. 输入指针调整 (line 141-145)
```cuda
const bool use_target_major_input = offsets == nullptr && (is_transpose || mask == nullptr);
const int4* src_base = reinterpret_cast<const int4*>(x);
if (use_target_major_input) {
    src_base += static_cast<uint64_t>(dst_rank) * batch_size * ints_per_token;
}
```

**含义**：
- **无 offsets 且 transpose**: 输入按目标 rank 组织，每个 dst_rank 的数据是连续的
- **有 offsets**: 输入是源 rank 连续的，不按目标分块

#### 2. 目标 buffer 索引计算 (line 150-158)
```cuda
const int mask_batch_size = offsets != nullptr ? max_batch_size : batch_size;
for (int token_i = start_token_idx; token_i < end_token_idx; ++token_i) {
    if (mask != nullptr && __ldg(&mask[dst_rank * mask_batch_size + token_i]) == 0) {
        continue;
    }
    
    const uint64_t dst_row_idx = offsets != nullptr
        ? static_cast<uint64_t>(offsets[local_rank]) + token_i
        : static_cast<uint64_t>(local_rank) * batch_size + token_i;
```

**关键差异**：

| 模式 | dst_row_idx 计算 | 说明 |
|------|-----------------|------|
| **无 offsets** | `local_rank * batch_size + token_i` | 固定间隔分配 |
| **有 offsets** | `offsets[local_rank] + token_i` | 紧凑连续分配 |

#### 3. mask 的索引方式 (line 150, 152)
```cuda
const int mask_batch_size = offsets != nullptr ? max_batch_size : batch_size;
if (mask != nullptr && __ldg(&mask[dst_rank * mask_batch_size + token_i]) == 0) {
    continue;
}
```

- **有 offsets**: `mask_batch_size = max_batch_size`，mask 的第二维是固定大小
- **无 offsets**: `mask_batch_size = batch_size`，mask 第二维与实际批次相同

---

## 总结

### offsets 的核心价值

1. **紧凑的输出布局**: 有效数据在输出 buffer 中紧密连续存储，避免固定间隔分配的空洞
2. **支持动态负载**: 每个 rank 可以有不同数量的有效请求，自动处理不均衡
3. **与 copy_batch_indexed 配合**: 通过预先的索引拷贝 + offsets 引导的 all2all，实现灵活的数据重排

### 实际推理中的完整流程

```
1. C++ 统计 sp_valid_request_counts（每个 rank 有多少有效请求）
   ↓
2. 计算 q_offsets = cumsum(sp_valid_request_counts)
   ↓
3. Python 接收并设置到 Context
   ↓
4. CUDA Graph replay 时从 Context 复制到 graph_vars
   ↓
5. Attention 层先用 copy_batch_indexed 排列 Q 到 local buffer
   ↓
6. 调用 all_to_all_ll(q, mask, offsets)
   ↓
7. CUDA Kernel 根据 offsets[local_rank] + token_i 计算写入位置
   ↓
8. 输出 buffer 中数据紧凑排列，前 offsets[-1] 行是有效数据
```

### 数据布局变化总结表

| 维度 | 无 offsets (固定分配) | 有 offsets (紧凑分配) |
|------|---------------------|---------------------|
| **输入要求** | 必须 padding 到 max_bs | 实际行数即可（通过 local buffer 处理） |
| **输出形状** | `[sp_size, max_bs, msg_dim]` | `[sp_size, max_bs, msg_dim]` |
| **输出布局** | 每个 rank 固定占 `max_bs` 行，间隔分配 | 紧凑排列，前 `offsets[-1]` 行 |
| **有效数据密度** | 低（rank 间有固定间隔，大量空洞） | 高（连续紧密排列） |
| **目标索引公式** | `rank * batch_size + i` | `offsets[rank] + i` |
| **适用场景** | 负载均衡，所有 rank 相同 bs | 负载不均，动态 batch |

### 关键设计理解

**为何需要 copy_batch_indexed + offsets 组合**：
1. **copy_batch_indexed**: 在 all2all 前将当前 rank 的 Q 按特定索引排列到 local buffer，解决"哪些序列的 Q 参与通信"的问题
2. **offsets**: 在 all2all 时指导 CUDA kernel 将各 rank 的数据写入紧凑连续的位置，解决"数据在输出 buffer 中如何排列"的问题
3. **mask**: 进一步控制哪些数据真正需要写入（跳过 ctx_len=0 的情况）

**与传统 all2all 的区别**：
- 传统：每个 rank 固定发送 batch_size 个 token，固定接收 batch_size 个 token
- NanoDeploy：每个 rank 发送的数量可变（sp_valid_request_counts），通过 offsets 紧凑排列，后续 attention 只处理前 `attention_compute_bs` 行

### 适用场景

**启用 offsets 的必要性**：
- Sequence Parallel 中不同序列的 KV 分布不均（某些序列只在部分 rank 有数据）
- Decode 阶段动态批处理，避免固定分配造成的带宽浪费
- 需要紧凑的 attention 输入布局，方便后续计算

**技术限制**：
- hao_basic 后端的 offsets **仅支持非 transpose** 模式 (`sp_backend.py` line 186-188)
- 需要准确的 `sp_valid_request_counts` 统计（依赖 C++ 元数据计算）
- offsets 数组本身也需要传递到 GPU（额外开销很小，只有 `sp_size+1` 个 int32）

### 完整代码路径

| 阶段 | 代码位置 | 关键操作 |
|------|---------|---------|
| **1. 统计** | `model_runner_utils.cpp:217-236` | 遍历序列，统计每个 rank 的 `sp_valid_request_counts` |
| **2. 累积和** | `model_runner_utils.cpp:323-328` | `q_offsets = cumsum(sp_valid_request_counts)` |
| **3. 转 GPU** | `model_runner.py:428-430` | `torch.tensor(...).cuda()` |
| **4. 设 Context** | `model_runner.py:469` | `set_context(q_offsets=q_offsets)` |
| **5. 图捕获** | `model_runner.py:915` | `q_offsets` 作为 graph_vars |
| **6. 图执行** | `model_runner.py:613-615` | `copy_(context.q_offsets)` + `replay()` |
| **7. Q 拷贝** | `attention.py:253-259` | `copy_batch_indexed_triton(...)` |
| **8. All2All** | `attention.py:261-265` | `all_to_all_ll(q, mask, offsets)` |
| **9. 后端转发** | `sp_backend.py:173-219` | 检查参数，转发给 DLSlime |
| **10. Kernel** | `all_to_all_intra_ll.cu:156-158` | `dst_row_idx = offsets[rank] + i` |

---

**分析完成时间**: 2026-04-05  
**通信后端**: hao_basic (DLSlime)  
**分析路径**: 实际推理 CUDA Graph 路径（非测试代码）  
**核心发现**: offsets 通过累积和引导 all2all 输出的紧凑布局，配合 copy_batch_indexed 和 mask 实现高效的动态负载处理
