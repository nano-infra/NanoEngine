# MLA SP 路径中 `optimize_decode_block_table + hao_basic offsets` 的问题分析与修复计划

日期：2026-04-06  
范围：NanoDeploy 当前主干代码、`/mnt/nvme1n1/ml_research/linbinbin1/DLSlime`、以及参考实现 `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime-a2a`  
目标：只整理分析与修复计划，不改代码

## 1. 修订说明

上一版文档里把 `q_offsets` 直接写成了 “receiver-major”，这个表述不准确。

更准确的说法是：

1. NanoDeploy 当前 decode 路径里的 **顺序语义** 是 `sender-major / source-major / master-major`。
2. 但当前 `prepare_decode_cpp(...)` 产出的 `q_offsets`，只描述了 **当前 receiver（也就是当前 `sp_rank`）这一行** 的 packed 布局。
3. 也就是说，`q_offsets[i] : q_offsets[i+1]` 表示：
   - 对当前 receiver 而言；
   - 来自 `master/source = i` 的那段 Q；
   - 在 packed buffer 里的区间。

这和你给的“之前正确能跑版本”的总结是一致的。

因此，当前真正的问题不是“顺序语义错了”，而是：

1. NanoDeploy 上层期望的是“每个 receiver 的 flatten 前缀天然已经 packed，可直接 `q[:attention_compute_bs]` 使用”；
2. 但 `hao_basic` 当前 native offsets 接口只接受 **一条 1D offsets**；
3. 一条 1D offsets 只能表达一个 destination 的 packed row；
4. 不能同时表达“同一个 sender 在多个 destination 上各自不同的 packed row”；
5. 所以在 partial-SP / mixed batch 下，当前 `hao_basic` 接口能力不够，不能无损承接 NanoDeploy 现有上层 contract。

## 2. 当前正确的语义基线

### 2.1 NanoDeploy 里的 Q packed 顺序是 sender-major

`csrc/nanodeploy/worker/model_runner_utils.cpp:221-235`：

1. 先按 `master_sp_idx_` 分组得到 `sp_seqs[sp_idx]`
2. `sp_valid_request_counts[sp_idx]` 统计的是：
   - `master/source == sp_idx`
   - 且在当前 `sp_rank` 上 `ctx_len > 0`
   - 的序列个数

`csrc/nanodeploy/worker/model_runner_utils.cpp:253-261`：

1. `context_lens_for_attn` 也是按 `sp_idx = 0..sp_size-1` 的顺序追加
2. 所以 packed attention 输入顺序本身就是 sender-major

`csrc/nanodeploy/worker/model_runner_utils.cpp:275-284`：

1. `q_slice_fill` 通过累加前面 sender 的 valid count 来计算
2. 本质上就是把 self Q 预写到 `offsets[self_sender] + local_k`

因此，当前 NanoDeploy 的 decode 上层 contract 是：

1. packed Q 的顺序是 sender-major
2. `context_lens_for_attn`
3. `block_tables`
4. `q_slice_fill`
5. `q_offsets`

这几者在当前 receiver 上是同一套 packed 语义

### 2.2 调度层的 Q 统计语义也是 sender-major

`csrc/nanodeploy/scheduler/scheduler.h:62-65`：

1. `sp_q_matrix` 的定义就是 `[master_sp_rank][participant_sp_rank]`

`csrc/nanodeploy/scheduler/scheduler.cpp:398-400`：

1. Q 的计数也是 `Master -> Participant`

这和上面的 sender-major / master-major 口径一致。

### 2.3 参考的 `DLSlime-a2a` offsets 语义

参考实现 `DLSlime-a2a` 的 kernel：

`/mnt/nvme1n1/ml_research/linbinbin1/DLSlime-a2a/csrc/dlslime/ops/intra_ll/all_to_all/all_to_all_intra_ll.cu:72-75`

```cpp
if (offsets) {
    buffer_idx = (offsets[rank] + msg_idx) * num_msg_per_warp;
}
```

这里的 `rank` 是 sender，不是 receiver。

所以旧实现的 offsets 语义是：

1. `offsets[i] : offsets[i+1]`
2. 表示 sender/source `i` 那段数据的区间

`/mnt/nvme1n1/ml_research/linbinbin1/DLSlime-a2a/tests/python/test_intra_all_to_all_offsets.py:75-98`

以及

`/mnt/nvme1n1/ml_research/linbinbin1/DLSlime-a2a/tests/python/test_intra_all_to_all_offsets.py:175-184`

都是按这个 sender-major 语义验证的。

### 2.4 一个容易混淆但必须分清的点

在这个协议里：

1. `offsets` 看的是 sender/source
2. `mask` 看的是 destination/receiver

也就是：

1. `offsets` 负责“sender 段从哪里开始”
2. `mask[dst_rank, msg_idx]` 负责“这个 sender 的第 `msg_idx` 条消息要不要发给这个 dst”

这一点和你给出的总结一致。

## 3. 当前真正的矛盾点是什么

### 3.1 上层想要的是“每个 receiver 的 packed 前缀可直接消费”

`nanodeploy/layers/attention.py:261-267` 当前 MLA Q 路径是：

```python
q = q_buffer.all_to_all_ll(
    q.view([bs, -1]),
    mask=context.q_mask,
    offsets=context.q_offsets,
).view([sp_size * max_num_seqs, num_head, head_dim])

q = q[: context.attention_compute_bs]
```

这说明当前 NanoDeploy 上层 contract 非常明确：

1. 每个 receiver 的 local buffer flatten 后；
2. 前 `attention_compute_bs` 行就必须已经是该 receiver 需要的 packed Q；
3. 不希望再额外做 receiver gather。

这也是本文后续修复方案要保留的目标。

### 3.2 但当前 `hao_basic` native offsets 只支持一条 1D offsets

当前 `DLSlime` 主干 `AllToAllBuffer` 的 API：

`/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.h:36-40`

以及 Python 绑定：

`/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/python/bind.cpp:306-312`

只有：

1. `mask`
2. `offsets`

其中 `offsets` 是单条 1D tensor。

这意味着：

1. 当前 sender 在发起一次 all2all 时；
2. 只能拿到一条 offsets；
3. 但它实际上要同时给所有 `dst_rank` 写数据；
4. 不同 `dst_rank` 所需的 packed row 通常并不相同；
5. 所以一条 1D offsets 不足以表达所有 destination 的布局要求。

### 3.3 为什么一条 1D offsets 不够

当前 `prepare_decode_cpp(...)` 生成的 `q_offsets`，实际上是：

1. 对当前 receiver 的那一行 packed offsets

也就是更精确地说：

1. `q_offsets == q_dst_offsets[current_sp_rank]`

但 sender 在一次 kernel 调用里要做的是：

1. 往 `dst = 0`
2. 往 `dst = 1`
3. ...
4. 往 `dst = sp_size - 1`

这些 receiver 的 packed row 往往不同。

因此，如果只传一条 1D `offsets`，就会出现：

1. 对某个 destination 是对的；
2. 对另外的 destination 就不是它想要的 packed row；
3. 最终某些 receiver 的 flatten 前缀不再等于自己的 `context_lens_for_attn / block_tables` 顺序。

### 3.4 这才是 current `hao_basic` 和 NanoDeploy 的核心接口缺口

所以核心矛盾不是：

1. sender-major vs receiver-major

而是：

1. NanoDeploy 想保留“每个 receiver 都已经 packed 好”的 contract
2. 当前 `hao_basic` 只给了一条 sender-major 1D offsets
3. 这条 1D offsets 无法同时满足多个 destination 的 packed 布局

## 4. `optimize_decode_block_table=True` 为什么还必须改

虽然问题主轴不再表述为“RPC 过滤把 `q_offsets` 变成 receiver-major”，但 `optimize_decode_block_table=True` 当前实现方式还是必须改。

原因很直接：

1. 如果要让 sender 在一次 all2all 中知道所有 destination 各自的 packed row；
2. 它就必须能在本地重建一整张 destination-aware 的 Q metadata；
3. 这要求每个 worker 在 decode 时都看到完整 `dp_seqs` 骨架和稳定顺序；
4. 不能再把整条 sequence 按 target rank 过滤掉。

当前 `serialization.cpp` 的 decode 优化路径：

`csrc/nanodeploy/sequence/serialization.cpp:126-147`

做的是：

1. 直接按 `ctx_len(target_sp_rank) > 0` 过滤 sequence

这会让不同 rank 拿到不同的 sequence skeleton。

而新的 destination-aware 方案需要：

1. 所有 rank 都能重建相同的 sender/source 级别骨架
2. 进而重建全部 destination 的 packed metadata

所以 decode RPC 必须改成：

1. 保留完整 sequence 集合与顺序
2. 只裁 target rank 不需要的重字段

## 5. 更新后的修复目标

修复目标现在明确成下面四条：

1. 保留 NanoDeploy 当前上层 contract：`q[:attention_compute_bs]` 继续成立
2. 不引入 receiver 侧 gather / compaction
3. 让 `hao_basic` 真正支持 destination-aware 的 packed write
4. decode RPC 不再过滤整条 sequence，只做字段裁剪

## 6. 更新后的修复方案

### 6.1 总体思路

不改变上层 MLA attention 的消费方式。

要改的是：

1. NanoDeploy 生成更完整的 Q metadata
2. `hao_basic` 从“单 1D offsets 模式”扩成“destination-aware packed write 模式”

最终效果是：

1. 每个 receiver 的 local buffer flatten 前缀依然已经是 packed Q
2. 所以 `q = q[:attention_compute_bs]` 可以继续保留

### 6.2 NanoDeploy 侧要新增什么 metadata

当前已有：

1. `q_offsets`
2. `q_slice_fill`
3. `q_mask`

但这三个量只足够表达“当前 receiver 自己这一行”的语义。

为了让 sender 能同时正确写所有 destination，建议新增两组 metadata。

#### A. `q_dst_offsets`

建议形状：

1. `[sp_size, sp_size + 1]`

含义：

1. 第 `dst_rank` 行，表示该 destination 对应的 sender-major packed offsets
2. 即：
   - `q_dst_offsets[dst_rank][src] : q_dst_offsets[dst_rank][src + 1]`
   - 是 destination=`dst_rank` 看来，来自 sender=`src` 的 packed 区间

当前已有的 `context.q_offsets` 可以保留，定义为：

1. `context.q_offsets = q_dst_offsets[current_sp_rank]`

这样上层和已有逻辑不用大改。

#### B. `q_dst_compact_pos`

建议形状：

1. `[sp_size, max_num_seqs]`

含义：

1. 对当前 sender 而言；
2. `q_dst_compact_pos[dst_rank][local_msg_idx]`
3. 表示 sender 的第 `local_msg_idx` 条本地 Q；
4. 若要发给 `dst_rank`，它应该落在该 sender 段里的第几个 compacted 位置；
5. 若不发，则写 `-1`

这个张量本质上是：

1. 对 `mask[dst_rank, :]` 的 prefix-compaction 编号

建议在 NanoDeploy 侧预先算好，而不是让 kernel 现场做 prefix scan。

理由：

1. 语义更清晰
2. 和 CUDAGraph 更兼容
3. 便于调试和单测

### 6.3 NanoDeploy 侧哪些现有量可以保持不变

以下语义建议保持：

1. `context_lens_for_attn`
2. `attention_compute_bs`
3. `q_slice_get`
4. `q_slice_fill`
5. `q_copy_mask`
6. `context.q_offsets = q_dst_offsets[current_sp_rank]`

也就是说，上层 attention 仍然保持：

1. self Q 先预写到当前 receiver 自己的 packed row
2. all2all 后直接取前缀

### 6.4 `hao_basic` 侧应如何扩展

当前 `AllToAllBuffer::all_to_all(...)` 只有：

1. `mask`
2. `offsets`

建议扩成 destination-aware 模式，至少支持：

1. `dst_offsets`
2. `dst_compact_pos`

形式上有两种可选方案。

#### 方案 A：新增显式参数

例如：

1. `all_to_all(..., mask=None, offsets=None, dst_offsets=None, dst_compact_pos=None)`

优点：

1. 语义最清晰
2. 和当前 1D offsets 路径兼容最好

#### 方案 B：在 `offsets` 上做 2D overload

例如：

1. `offsets.shape == [world_size + 1]` 时走旧路径
2. `offsets.shape == [world_size, world_size + 1]` 时走 destination-aware 路径

但即便这样，仍然需要额外的 `dst_compact_pos`，否则 sparse mask 仍会留下 hole。

因此更推荐方案 A。

### 6.5 `hao_basic` kernel 的新写入公式

在新的 destination-aware packed 模式下，kernel 不应再用：

```cpp
dst_row_idx = offsets[rank] + msg_idx
```

而应改成：

```cpp
compact_idx = dst_compact_pos[dst_rank][msg_idx]
if (compact_idx >= 0) {
    dst_row_idx = dst_offsets[dst_rank][rank] + compact_idx;
}
```

也就是：

1. 段起点仍然是 sender-major
2. 但段内位置不再直接使用原始 `msg_idx`
3. 而是使用针对该 destination 已经 compact 过的位置

这样就能保证：

1. 对每个 destination 来说
2. flatten 前缀都是 dense packed
3. 不需要 receiver 再 gather

### 6.6 为什么这条路不需要 receiver gather

因为 packed 行已经在写入远端 buffer 的时候完成了。

具体来说：

1. self Q 通过 `q_slice_fill` 预写到当前 receiver 的 packed row
2. remote sender 再根据 `dst_offsets[current_receiver] + compact_pos` 写进当前 receiver 的 packed row
3. 最终当前 receiver 的 local buffer flatten 前缀天然就是：
   - sender 0 的有效 Q
   - sender 1 的有效 Q
   - ...
   - sender N 的有效 Q
4. 而且顺序与 `context_lens_for_attn` 完全一致

所以：

1. `q = q[:attention_compute_bs]` 继续有效
2. 上层 `FlashMLAImpl` 不需要新增 receiver gather

## 7. `optimize_decode_block_table=True` 的具体修法

这里建议和 Q 修复一起做。

### 7.1 保留完整 sequence skeleton

decode RPC 优化路径中：

1. 不再按 `ctx_len(target_sp_rank) > 0` 过滤整条 sequence
2. 保留完整 sequence 集合
3. 保留原始顺序

### 7.2 只裁字段，不裁 sequence

建议第一版只裁：

1. `BlockContext.sp_block_table`

可选再裁：

1. `BlockContext.block_location`

必须完整保留：

1. `master_sp_idx_`
2. `num_dispatched_tokens`
3. decode 元数据重建所需的所有标量头字段

### 7.3 这样做的原因

因为 destination-aware 的 Q metadata 需要每个 rank 都能完整看到：

1. 所有 sender 的 sequence skeleton
2. 各 sender 在各 destination 上的有效性

过滤整条 sequence 会直接破坏这一步。

## 8. 实施落点

如果后面按这个计划改代码，主要会落在下面这些文件。

### 8.1 NanoDeploy

1. `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/csrc/nanodeploy/sequence/serialization.cpp`
2. `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/csrc/nanodeploy/worker/model_runner_utils.h`
3. `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/csrc/nanodeploy/worker/model_runner_utils.cpp`
4. `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/csrc/python/model_runner_binding.cpp`
5. `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/nanodeploy/worker/model_runner.py`
6. `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/nanodeploy/worker/context.py`
7. `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/nanodeploy/worker/sp_backend.py`

### 8.2 DLSlime / hao_basic

1. `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.h`
2. `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.cpp`
3. `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/all_to_all_intra_ll.h`
4. `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/all_to_all_intra_ll.cu`
5. `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/python/bind.cpp`

## 9. 验证计划

### 9.1 必测 case

至少覆盖下面三类 batch。

1. 单请求，全跨 SP
2. 一个 sender 上同时存在：
   - 跨 SP 请求
   - 只在本地 rank 有 KV 的请求
3. 多 sender、各 destination 有效请求数不同

### 9.2 需要直接断言的量

对每个 receiver rank，断言：

1. `q.view(-1, msg_dim)[:attention_compute_bs]`
2. `context_lens_for_attn`
3. `block_tables`

三者顺序一致。

对每个 sender rank，断言：

1. `dst_compact_pos[dst_rank][msg_idx] == -1` 时不写
2. `dst_compact_pos[dst_rank][msg_idx] >= 0` 时写入：
   - `dst_offsets[dst_rank][sender] + dst_compact_pos[dst_rank][msg_idx]`

### 9.3 RPC optimize on/off 一致性

分别验证：

1. `optimize_decode_block_table=False`
2. `optimize_decode_block_table=True`

在上述两条路径下：

1. destination-aware Q metadata 一致
2. MLA 最终输出一致
3. 优化路径的 `dlslime_send_seqs_BYTES` 仍然更小

## 10. 最终建议

最终建议可以浓缩成一句话：

**保留 NanoDeploy 当前“每个 receiver 的 packed 前缀可直接消费”的上层 contract，不做 receiver gather；为此把 `hao_basic` 从单 1D offsets 扩成 destination-aware packed write，并把 decode RPC 优化改成只裁字段、不裁 sequence。**

具体落地上：

1. `q_offsets` 的口径保留 sender-major / master-major
2. `context.q_offsets` 继续表示“当前 receiver 自己这一行”的 packed offsets
3. NanoDeploy 新增 `q_dst_offsets` 与 `q_dst_compact_pos`
4. `hao_basic` 按 destination-aware packed 方式写远端 buffer
5. `attention.py` 保持 `q = q[:attention_compute_bs]` 不变

这条路线最符合你给出的旧语义基线，也最符合“不增加 receiver 侧 gather”的约束。
