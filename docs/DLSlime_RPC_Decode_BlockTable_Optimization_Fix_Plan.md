# DLSlime RPC Decode BlockTable 优化修复计划

> 历史设计说明（2026-08-18）：本文提到的 `legacy_ll` 已从当前实现删除；当前支持的
> SP backend 为 `hao_basic` 和 `nccl`。

## 1. 问题定义

当前在以下条件同时满足时，会出现 decode 卡住：

1. `use_dlslime_rpc=True`
2. `optimize_decode_block_table=True`
3. batch 中存在真正的 SP 请求，即某些请求跨多个 SP rank 分发

目前已知现象是：

1. 将 `optimize_decode_block_table=False` 后，问题消失
2. 该问题不是 `legacy_ll` 或 `hao_basic` 后端自身的必要属性
3. 该问题并不只局限于 MLA；只要 decode 路径进入 `use_sp_a2a=True`，GQA 也会继承相同风险

## 2. 根因结论

根因不是“offsets 和优化目标天然冲突”，而是“当前优化实现方式破坏了 decode worker 的输入语义”。

当前 scheduler 和 worker 的契约是：

1. 每个 `(dp, sp)` worker 在 decode 阶段都应看到同一份完整 `dp_seqs`
2. worker 再基于完整 `dp_seqs` 本地重建：
   - `use_sp_a2a`
   - `context_lens`
   - `global_context_lens`
   - `context_lens_for_attn`
   - `q_slice_get`
   - `q_slice_fill`
   - `q_offsets`

当前 `optimize_decode_block_table=True` 的 RPC 序列化实现，实际做成了：

1. 按目标 `sp_rank` 过滤 sequence
2. 仅保留 `ctx_len(target_sp_rank) > 0` 的 sequence

这会导致：

1. 不同 SP rank 收到不同的 `dp_seqs`
2. 各 rank 重建出的 `q_offsets` 和 packed layout 不一致
3. 一旦进入 `q_buffer.all_to_all_ll(..., offsets=context.q_offsets)`，collective 语义被破坏
4. 最终表现为 SP decode hang

结论是：

1. 当前实现下，`optimize_decode_block_table` 和 offsets 会打架
2. 冲突的是“按 target rank 过滤 sequence 集合”这个实现方式
3. 不是“少传 block table”这个优化目标本身有问题

## 3. 设计目标

本次修复目标如下：

1. 恢复 decode correctness，不再出现 SP hang
2. 保留 `optimize_decode_block_table` 的通信量优化收益
3. 不改 scheduler 的外部语义
4. 不改 worker decode 主逻辑和 `prepare_decode_cpp(...)` 的语义
5. 同时兼容 MLA 和 GQA 的 SP-A2A decode 路径

本次不追求：

1. 一次性做到 decode RPC payload 的极限压缩
2. 重写 worker 端 decode 元数据协议
3. 引入一套与 `Sequence` 完全不同的新 RPC 载荷格式

## 4. 修复原则

修复原则只有一句话：

1. decode RPC 必须保留完整 `dp_seqs` 的 sequence 集合和顺序
2. 只裁剪每条 `Sequence` 中 target rank 不需要的大字段

也就是说：

1. 不能再按 target `sp_rank` 过滤掉整条 sequence
2. 可以继续只给 target `sp_rank` 发送它真正需要的 `sp_block_table`

## 5. 建议方案

### 5.1 总体方案

在 `decode + optimize_decode_block_table=True` 时：

1. 保留全部 `Sequence`
2. 保留全部 sequence 的顺序
3. 保留重建 decode 元数据所需的所有标量字段
4. 只裁剪 target-rank 无关的重字段

这样 worker 端看到的 `dp_seqs` 与非 RPC 路径保持一致，`prepare_decode_cpp(...)` 无需改语义即可继续工作。

### 5.2 需要完整保留的字段

以下字段必须完整保留，因为它们参与 decode 元数据重建：

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

说明：

1. decode 阶段 `token_ids` 继续不传，保持现状
2. `num_dispatched_tokens` 必须保留完整向量，因为 `prepare_decode_cpp(...)` 依赖它判断 `use_sp_a2a`

### 5.3 建议裁剪的字段

以下字段在 decode RPC 中属于可裁剪重字段：

1. `BlockContext.sp_block_table`
2. `BlockContext.block_location`

建议策略：

1. `sp_block_table` 只保留目标 `target_sp_rank` 对应的一列
2. 其它 `sp_idx` 的 `sp_block_table[i]` 直接写空
3. `block_location` 仅保留 `first == target_sp_rank` 的条目，或者直接写空

这能保留绝大多数通信优化收益，因为 decode payload 的主要体积正来自这些按 rank 展开的列表。

### 5.4 对 `BlockContext` 的具体序列化策略

建议把 `serialize_block_context(...)` 改为支持 target-aware trimming。

在 `decode + optimize_decode_block_table=True + target_sp_rank >= 0` 时：

1. 先写入所有标量头字段
2. `num_dispatched_tokens` 全量写入
3. `sp_block_table` 仍写完整 outer size
4. 只有 `sp_block_table[target_sp_rank]` 写真实内容
5. 其它 `sp_block_table[i]` 写 `inner_sz = 0`
6. `block_location` 仅写 target-rank 相关项，或写空

在其它场景：

1. 保持当前全量序列化行为不变

### 5.5 对 sequence 集合的具体策略

当前 decode 优化路径中的这一步应删除：

1. “按 `ctx_len(target_sp_rank) > 0` 过滤 `filtered_seqs`”

应改为：

1. decode 阶段始终序列化全部 `seqs`
2. 仅在每条 sequence 内部做 target-aware trimming

这是本次修复最核心的一步。

## 6. 为什么这条方案最稳

该方案的优势是：

1. 不修改 scheduler 对 worker 的输入契约
2. 不修改 `prepare_decode_cpp(...)` 的核心逻辑
3. 不修改 `FlashMLAImpl` / `FlashAttentionImpl` 的 offsets 语义
4. 可以直接复用当前的反序列化逻辑
5. 通信量仍明显低于 `optimize_decode_block_table=False`

换句话说，这是一条“恢复 correctness，同时保留主要收益”的最小修复路径。

## 7. 对 MLA 与 GQA 的影响

这次修复不应只按 MLA 理解。

对 MLA：

1. `FlashMLAImpl` 在 SP-A2A decode 时会直接消费 `q_offsets`
2. 修复后 offsets 语义恢复一致

对 GQA：

1. `FlashAttentionImpl` 在 `use_sp_a2a=True` 时也会消费 `q_offsets`
2. 只要 GQA batch 中存在真实跨 SP 请求，也会依赖完整 `dp_seqs`
3. 因此本修复也应覆盖 GQA

结论：

1. 这不是 MLA 专属修复
2. 这是 decode SP-A2A 路径的通用修复

## 8. 实施步骤

### Step 1: 修正序列化逻辑

修改文件：

1. `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/csrc/nanodeploy/sequence/serialization.cpp`

实施内容：

1. 删除 decode 优化路径对 sequence 集合的过滤
2. 为 `BlockContext` 增加 target-aware trimming 逻辑
3. 默认行为保持向后兼容

### Step 2: 保持 RPC 入口参数语义不变

修改文件：

1. `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/nanodeploy/endpoint/rpc_endpoint.py`

实施内容：

1. 保持现有 `sp_rank` / `sp_size` 传参方式
2. 不改 Python 调用层协议
3. 只让下层序列化在 decode 优化路径中改变“裁剪对象”

### Step 3: 补充回归测试

建议新增测试覆盖：

1. sequence 数量与顺序不变
2. 非目标 rank 的 `sp_block_table` 被清空
3. `prepare_decode_cpp(...)` 在 serialize/deserialize 前后产物一致
4. `use_sp_a2a` 与 `q_offsets` 在优化路径下不再漂移

## 9. 验证方案

### 9.1 单元测试

建议新增以下测试：

1. 构造一个包含跨 SP 请求的 `dp_seqs`
2. 分别走：
   - 直接 `prepare_decode_cpp(full_seqs, sp_rank, ...)`
   - `deserialize(serialize(full_seqs, decode optimize path))` 后再 `prepare_decode_cpp(...)`
3. 断言以下结果一致：
   - `use_sp_a2a`
   - `input_ids`
   - `positions`
   - `context_lens_flat`
   - `global_context_lens_flat`
   - `block_tables_flat`
   - `context_lens_for_attn`
   - `q_slice_get`
   - `q_slice_fill`
   - `q_offsets`

### 9.2 集成验证

对当前复现 case 做 A/B 验证：

1. `use_dlslime_rpc=True`
2. `optimize_decode_block_table=True`
3. `SP` 请求存在

验证项：

1. 不再 hang
2. 输出 token 与 `optimize_decode_block_table=False` 一致
3. `dlslime_send_seqs_BYTES` 小于 `optimize_decode_block_table=False`
4. `use_sp_a2a`、`q_offsets`、`attention_compute_bs` 在各 rank 上保持协议一致

## 10. 风险与取舍

### 10.1 主要风险

风险点有两个：

1. `block_location` 若被未来其它 decode 逻辑隐式依赖，过度裁剪可能引入新问题
2. 若未来 decode 路径开始依赖 `MIGRATE/SWAP` slot 的完整 block table，进一步裁剪也可能触发兼容性问题

### 10.2 当前建议取舍

为了降低首版修复风险，建议：

1. 第一版先只裁 `sp_block_table`
2. `block_location` 可以先保守保留，或仅做 target-only 裁剪但配套测试
3. 等 correctness 稳定后，再继续压缩 `block_location`

换句话说：

1. 第一优先级是恢复正确性
2. 第二优先级是在不改协议语义的前提下保留主要优化收益
3. 第三优先级才是继续压榨最后一部分 payload 大小

## 11. 第二阶段可选优化

如果第一阶段修完后，decode RPC payload 仍然偏大，可考虑第二阶段优化：

1. 引入专用 decode RPC payload，而不是沿用通用 `Sequence` 序列化
2. 将“sequence skeleton”和“target-rank block table payload”拆开发送
3. 进一步压缩 `block_location`

但这不应作为本次主修复的一部分。

原因是：

1. 第一阶段已经能解决 hang
2. 第一阶段仍能保留大部分通信量优化
3. 第二阶段需要引入新的协议与测试面，风险明显更高

## 12. 最终建议

最终建议如下：

1. 采用“保留完整 `dp_seqs`，仅裁剪 target-rank 重字段”的方案
2. 不再按 `ctx_len(target_sp_rank) > 0` 过滤整条 sequence
3. 优先保证 `prepare_decode_cpp(...)` 在 RPC 前后语义完全一致
4. 第一版先稳住 correctness，再继续做 payload 压缩

这是当前风险最低、收益最高、且最不容易破坏 MLA/GQA decode 语义的修复路径。
