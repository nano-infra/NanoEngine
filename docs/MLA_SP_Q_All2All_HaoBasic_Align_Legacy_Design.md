# MLA SP Q All2All：修订版修复设计

日期：2026-04-06  
范围：`NanoDeploy-April`、`/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-readonly`、`/mnt/nvme1n1/ml_research/linbinbin1/DLSlime`、`/mnt/nvme1n1/ml_research/linbinbin1/DLSlime-a2a`  
目标：把当前版本的 `hao_basic` Q all2all 修到可承接 MLA SP decode 路径，同时把 `optimize_decode_block_table=True` 下真正破坏 decode 协议的问题描述准确、修法收敛、验证闭合

## 1. 结论先行

这次修复应当沿着下面两条线推进：

1. `hao_basic` 的 Q all2all 语义对齐 `DLSlime-a2a` 的老 contract：
   - 单条 `1D sender-major offsets`
   - `mask` 只表达 destination 侧是否写入
   - `offsets + mask` 允许 hole
   - 不引入 destination-aware packed write
2. `optimize_decode_block_table=True` 的 decode RPC 需要修，但根因不是“`q_offsets` 必然漂移”，而是“过滤整条 sequence 会破坏 decode metadata 中依赖 master 分组局部序号的一整组张量”

本次明确不做：

1. `q_dst_offsets`
2. `q_dst_compact_pos`
3. destination-aware packed write kernel
4. receiver 侧 gather / compaction

本次需要做的事可以收敛成一句话：

**保留单条 sender-major `q_offsets`，让 `hao_basic` 模拟 `DLSlime-a2a` 的 `1D offsets + mask` Q all2all；同时修正 decode RPC，使 worker 在优化路径下仍看到完整且顺序稳定的 sequence skeleton，只裁重字段，不裁 sequence。**

## 2. 代码复核后的事实基线

### 2.1 `DLSlime` 当前 native offsets 语义已经接近老实现

对当前 `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime` 代码复核后的结论是：

1. `offsets` 只接受单条 `1D int32`，长度为 `[world_size + 1]`
2. `offsets` 只支持 `is_transpose=False`
3. `has_offsets` 时，`batch_size = x.rows`，允许 `x.rows < max_bs`
4. kernel 写入位置仍是：

```cpp
dst_row_idx = offsets[local_rank] + token_i;
```

5. `mask[dst_rank, token_i] == 0` 时直接跳过写入，不做 compaction
6. 返回视图仍然是 `[world_size, max_bs, msg]`
7. `tests/python/test_intra_all_to_all_offsets.py` 也在验证：
   - offsets 下 sender-major packed 段
   - offsets + mask 下 hole 保留

因此，当前 `DLSlime` 的 basic native path 并没有表现出“必须扩成 destination-aware metadata 才能工作”的事实依据。

### 2.2 `NanoDeploy-April` 与 `NanoDeploy-readonly` 在两个关键点上当前是相同的

本地对照代码后，下面两点是逐行一致的：

1. `prepare_decode_cpp(...)`
2. decode RPC 优化路径下的 `serialization.cpp` / `rpc_endpoint.py`

这意味着：

1. 不能把 Step 4 表述成“单纯恢复 `NanoDeploy-readonly` 行为”
2. Step 4 更准确的性质是：
   - 修正当前 decode RPC contract
   - 让其重新满足 worker decode 元数据生成的真实前提

### 2.3 真正会被 sequence 过滤破坏的，不只 `q_offsets`

对真实 `serialize -> deserialize -> prepare_decode_cpp` 做最小 roundtrip 复核后，可以观察到：

1. 在某些 case 下，`q_offsets` 不变
2. `context_lens_for_attn` 不变
3. `block_tables_flat` 也可能不变
4. 但 `context_lens_flat` 会变
5. `res_slice_fill_to_buffer_input` 会变

一个最小例子里，`sp_rank = 1` 时：

1. full seqs 的 `context_lens_flat` 为 `[0, 2, 0, 0, 4, 2, 0, 0]`
2. roundtrip 后变成 `[2, 0, 0, 0, 4, 2, 0, 0]`
3. full seqs 的 `res_slice_fill_to_buffer_input` 为 `[1]`
4. roundtrip 后变成 `[0]`
5. 同一个例子里 `q_offsets` 两边都是 `[0, 1, 3]`

这说明：

1. “优化路径过滤 sequence 会破坏 decode 协议”这个判断是对的
2. 但最先被破坏的未必是 `q_offsets`
3. 如果验证只盯 `q_offsets / attention_compute_bs`，会漏掉实际会导致 decode 错位或 hang 的 `res_* / context_lens_*`

## 3. 正确的 contract 应该是什么

### 3.1 Q all2all 的 contract

`NanoDeploy` 上层 decode Q 路径应继续保持：

```python
q = q_buffer.all_to_all_ll(
    q.view([bs, -1]),
    mask=context.q_mask,
    offsets=context.q_offsets,
)
q = q[: context.attention_compute_bs]
```

这里的含义应明确固定为：

1. `q_offsets` 是单条 `1D sender-major prefix sum`
2. `q_offsets[i + 1] - q_offsets[i]` 表示当前 receiver 看来，来自 sender/master `i` 的有效 Q 数量
3. `q_offsets[-1] == attention_compute_bs`
4. `q_mask` 是 `[world_size, max_bs]`
5. primitive 只负责 sender-major packed 段写入与 destination mask
6. primitive 不负责 per-destination compaction

### 3.2 Decode metadata 的 contract

`prepare_decode_cpp(...)` 真正依赖的不是“只要 `q_offsets` 对就行”，而是：

1. worker 在 decode 时看到完整且顺序稳定的 `dp_seqs`
2. 基于这份稳定 skeleton，本地重建出一致的一组 metadata

这组 metadata 至少包括：

1. `context_lens_flat`
2. `global_context_lens_flat`
3. `block_tables_flat`
4. `context_lens_for_attn`
5. `q_slice_get`
6. `q_slice_fill`
7. `q_copy_mask`
8. `res_slice_get_to_buffer_output`
9. `res_slice_fill_to_buffer_output`
10. `res_to_buffer_output_mask`
11. `res_slice_get_to_buffer_input`
12. `res_slice_fill_to_buffer_input`
13. `res_to_buffer_input_mask`
14. `q_offsets`

其中最容易被忽略的一点是：

1. `res_slice_fill_to_buffer_input = sp_idx * max_num_seqs + seq_id`
2. 这里的 `seq_id` 不是全局 `seq_id`
3. 它是“该 `master_sp_idx` 分组内的局部序号”
4. 一旦 sequence 集合被过滤，这个局部序号就会左移
5. 回传路径随之错位

因此，decode RPC 如果裁的是 sequence 集合，而不是 sequence 内部的重字段，就会直接破坏这个 contract。

## 4. 不该继续走的方向

### 4.1 不要把问题升级成 destination-aware Q protocol

当前没有足够证据支持下面这条路线：

1. NanoDeploy 上层天然要求每个 destination 都有独立 packed row
2. 所以 `hao_basic` 必须扩成 destination-aware offsets protocol

与当前代码和测试更一致的结论是：

1. `hao_basic` 先对齐老 `DLSlime-a2a` 的 `1D offsets + mask`
2. decode 错误优先排查 metadata / RPC skeleton

### 4.2 不要把 decode RPC 问题只写成 `q_offsets` 问题

sequence 过滤的影响范围更大：

1. `q_offsets` 可能变
2. 也可能不变
3. 但 `context_lens_flat`
4. `res_slice_fill_to_buffer_input`
5. 以及其它依赖 master 分组局部序号的 metadata
6. 一样会被破坏

因此，修法和验证都不能只围绕 `q_offsets` 展开。

### 4.3 不要把这一步描述成“回到 readonly 当前代码”

本地代码已经证明：

1. `NanoDeploy-readonly` 和 `NanoDeploy-April` 当前都在 decode optimize path 过滤整条 sequence
2. 所以 Step 4 是“修正当前设计”
3. 不是“仅仅切回 readonly 现状”

## 5. 修订后的落地方案

### Step 1：冻结 `hao_basic` Q offsets contract

目标：

1. 只支持 `is_transpose=False + offsets != None`
2. `offsets.shape == [world_size + 1]`
3. `offsets[i] : offsets[i + 1]` 表示 sender `i` 的 packed 段
4. 当前 rank 输入 `x.shape[0] == offsets[rank + 1] - offsets[rank]`
5. `dst_row_idx = offsets[rank] + msg_idx`
6. `mask` 仍是 `[world_size, max_bs]`
7. `offsets + mask` 不做 per-destination compaction
8. 输出仍是 `[world_size, max_bs, msg]`

修改/确认文件：

1. `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.h`
2. `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.cpp`
3. `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/all_to_all_intra_ll.cu`
4. `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/python/bind.cpp`

### Step 2：保持 NanoDeploy 上层调用口径不变

目标：

1. `attention.py` 继续使用 `all_to_all_ll(..., mask=context.q_mask, offsets=context.q_offsets)`
2. 不新增 destination-aware metadata tensor
3. 不新增 receiver gather
4. 不修改 `q = q[: context.attention_compute_bs]`

修改/确认文件：

1. `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/nanodeploy/worker/sp_backend.py`
2. `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/nanodeploy/layers/attention.py`

### Step 3：修正 decode RPC 的优化方式

这是本次真正需要落地的关键修复。

目标：

1. `optimize_decode_block_table=True` 时，不再过滤整条 sequence
2. decode RPC 继续传完整 sequence 集合与顺序
3. 只在 sequence 内部做 target-aware trimming

建议保留的字段：

1. `seq_id`
2. `status`
3. `temperature`
4. `max_tokens`
5. `ignore_eos`
6. `last_token`
7. `num_tokens`
8. `num_prompt_tokens`
9. `num_checkpointed_tokens`
10. `num_cached_tokens`
11. `BlockContext.engine_id_`
12. `BlockContext.dp_idx_`
13. `BlockContext.master_sp_idx_`
14. `BlockContext.attention_sp_`
15. `BlockContext.attention_dp_`
16. `BlockContext.num_dispatched_tokens`

建议裁剪的字段：

1. `BlockContext.sp_block_table`
2. `BlockContext.block_location`

建议策略：

1. `sp_block_table` 保留完整 outer size
2. 只对 `target_sp_rank` 写真实内容
3. 其它 `sp_idx` 写空 list
4. `block_location` 只保留 target-rank 相关项，或直接写空
5. decode 阶段 `token_ids` 继续不传，保持现状

修改文件：

1. `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/csrc/nanodeploy/sequence/serialization.cpp`

### Step 4：把“协议一致性验证”升级成全量 metadata 对比

不能只断言：

1. `q_offsets[-1] == attention_compute_bs`
2. `context_lens_for_attn.numel() == attention_compute_bs`

还必须断言 optimize path roundtrip 前后，`prepare_decode_cpp(...)` 的关键输出一致。

建议基线：

1. baseline：`prepare_decode_cpp(full_seqs, sp_rank, ...)`
2. roundtrip：`prepare_decode_cpp(deserialize(serialize(full_seqs, decode optimize path)), sp_rank, ...)`

至少比对：

1. `use_sp_a2a`
2. `input_ids`
3. `positions`
4. `context_lens_flat`
5. `global_context_lens_flat`
6. `block_tables_flat`
7. `context_lens_for_attn`
8. `q_slice_get`
9. `q_slice_fill`
10. `q_copy_mask`
11. `res_slice_get_to_buffer_output`
12. `res_slice_fill_to_buffer_output`
13. `res_to_buffer_output_mask`
14. `res_slice_get_to_buffer_input`
15. `res_slice_fill_to_buffer_input`
16. `res_to_buffer_input_mask`
17. `q_offsets`

## 6. 验证要求

### 6.1 Primitive 级验证

直接复刻 `DLSlime-a2a` 的验证口径：

1. `offsets` 无 mask：flatten 后检查 `[offsets[i], offsets[i + 1])`
2. `offsets + mask`：被 mask 掉的位置保持旧值，不做 compaction
3. 输出 shape 保持 `[world_size, max_bs, msg]`

### 6.2 Metadata 级验证

新增两类断言：

1. receiver 内部一致性
   - `q_offsets[-1] == attention_compute_bs`
   - `context_lens_for_attn.numel() == attention_compute_bs`
   - `q_slice_fill` 落点与 `q_offsets` 对齐
   - `block_tables[:attention_compute_bs]` 与 `context_lens_for_attn` 顺序一致
2. optimize roundtrip 一致性
   - `prepare_decode_cpp(full_seqs)` 与 `prepare_decode_cpp(roundtrip_seqs)` 在上节列出的所有 metadata 上完全一致

### 6.3 Backend 对齐验证

还必须补一类更关键的验证：

1. 不要只用 synthetic offsets case 证明 `hao_basic(offsets)` 正确
2. 必须使用 `prepare_decode_cpp(...)` 真实生成的 decode metadata
3. 用同一组真实 metadata，直接对比：
   - reference：`legacy_ll(mask)`
   - candidate：`hao_basic(offsets)`

建议做法：

1. 从真实 `dp_seqs` 调 `prepare_decode_cpp(...)`
2. 取其真实产物：
   - `q_mask`
   - `q_offsets`
   - `q_slice_get`
   - `q_slice_fill`
   - `q_copy_mask`
   - `attention_compute_bs`
   - `context_lens_for_attn`
3. 用这组真实 metadata 驱动两条 Q 路径：
   - `legacy_ll` 走现有 masked path
   - `hao_basic` 走 offsets path
4. 直接比较：
   - all2all 返回后前缀 `q[:attention_compute_bs]` 的逐元素结果
   - `context_lens_for_attn[:attention_compute_bs]`
   - `block_tables[:attention_compute_bs]`
5. case 必须覆盖：
   - mixed batch
   - partial SP
   - 同一个 sender 在不同 receiver 上活跃集合不同
   - 活跃 token 不是 sender-local prefix 的情况

只有这组“真实 metadata 驱动”的 backend 对齐验证通过，才能说明当前方案确实向旧版可运行行为看齐，而不是只复刻了 `DLSlime-a2a` primitive 的表面接口。

### 6.4 端到端验证

至少覆盖：

1. 单请求，全跨 SP
2. mixed batch：同时存在跨 SP 请求和本地请求
3. dynamic SP size / non-uniform split
4. `optimize_decode_block_table=False`
5. `optimize_decode_block_table=True`
6. `sp_backend=legacy_ll`
7. `sp_backend=hao_basic`

目标是：

1. `legacy_ll` 与 `hao_basic` 输出一致
2. optimize on/off 输出一致
3. 不引入 destination-aware metadata 也能跑通

## 7. 最终建议

最终建议可以收敛成下面这句话：

**主修复方向是正确的：保留单条 sender-major `q_offsets`，让 `hao_basic` 对齐 `DLSlime-a2a` 的 `1D offsets + mask` Q all2all 语义；同时把 decode RPC 的优化实现从“过滤 sequence 集合”改成“保留完整 skeleton、只裁重字段”。但 Step 1 / Step 2 是否真正完成“向旧版可运行行为看齐”，必须以 6.3 的真实 metadata 驱动 backend 对齐验证为准。**

需要特别避免的偏差有两个：

1. 不要把问题重新升级成 destination-aware packed write 设计
2. 不要把 RPC 过滤问题只表述成 `q_offsets` 问题；真正要守住的是整组 decode metadata 的一致性

如果后续实现仍然出现 `q[:attention_compute_bs]` 不成立或 decode hang，优先排查顺序应为：

1. `serialize_sequences(...)` 是否仍在过滤 sequence 集合
2. optimize path roundtrip 前后 `prepare_decode_cpp(...)` 的全量 metadata 是否一致
3. `q_mask / q_slice_fill / q_offsets / context_lens_for_attn / res_*` 是否仍是同一套 sender-major 顺序
4. 最后才检查 `hao_basic` native offsets 路径是否偏离 `DLSlime-a2a`
