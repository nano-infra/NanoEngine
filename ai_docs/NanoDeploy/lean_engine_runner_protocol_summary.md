# Lean Engine-Runner Protocol Summary

**Date**: 2026-03-04
**Feature Branch**: `feature/lean-engine-runner-protocol` (Merged to `main`)

## 1. Background & Motivation

在之前的架构中，Engine (`llm_engine.py`) 与 Ray Worker (`model_runner.py`) 之间的通信是通过 Ray 序列化完整的 Python `Sequence` 对象完成的。这带来了一系列严重的技术债务和性能问题：

1. **序列化开销巨大**：包含许多 Worker 侧根本用不到的字段，特别是 Prefill 阶段传输了大量已被 Cache 的 `token_ids`（单请求可能达数十 MB）。
2. **Ray OOM 与死锁**：庞大的序列化数据导致 Ray 对象存储容易打满，进而引发 NanoRouter 层的无限死锁 Bug。
3. **隔离性差**：Worker 层与 Engine 层的强耦合导致了 `Sequence` 状态管理混乱，阻碍了后续的架构优化（如跨进程调度、C++ Runner 等）。

## 2. Core Improvements (核心改进)

为了解决以上问题，本次重构引入了 **Lean Engine-Runner Protocol**，彻底移除了 Worker 层对 Python `Sequence` 对象的依赖，转而使用高效、精简的 FlatBuffers 二进制通信。

### 2.1 FlatBuffers 协议定义

新增了 `interface.fbs`，定义了极简的数据传输结构：

- **Run 路径 (`RunBatchInput`)**：仅传输 Worker 真正需要的 9 个核心字段。Decode 阶段不再传输任何 `token_ids`；Prefill 阶段仅传输未被 Cache 的增量 `token_ids`。
- **Migrate 路径 (`MigrateBatchInput`)**：仅提取 ACTIVE 槽位 和 MIGRATE 槽位所需的 8 个关键路由与内存块位置信息。

### 2.2 C++ 侧高效的序列化与反序列化

- **Engine 侧 (`serialization.cpp`)**：将 `Sequence` 列表快速序列化为 FlatBuffers Bytes，大幅降低了序列化耗时。
- **Worker 侧 (`model_runner_utils.cpp`)**：接收到 Bytes 后，直接在 C++ 层解析 FlatBuffers，并一次性构建出 `PrefillMetadata` 或 `DecodeMetadata` 结构体，**中间不产生任何 Python 对象**，从而完全去除了 Python 层的反序列化开销。

### 2.3 Python 层的彻底解耦

- `model_runner.py` 内**全面移除了** `Sequence` 及其相关类的 Import，将 `run` 接口改为 `run_from_bytes`，将 `migrate` 接口改为 `migrate_from_bytes`。
- 将旧版中充斥重复逻辑的 `migrate` 功能重构，提取出 `_ensure_peer_connections` 和 `_execute_rdma_reads` 私有帮助函数，减少了近 200 行代码重复。

## 3. Security & Correctness Fixes (安全与正确性修复)

由于 Worker 端如今直接接收来自网络的“不受信任”的 FlatBuffers 裸数据，本次重构也在 C++ 和 Python 层增加了严格的安全性校验：

1. **FlatBuffers Verifier**：在解析每条消息前，严格调用 `verifier.VerifyBuffer<T>` 校验内存安全性，防止非法构造的数据造成 C++ 指针越界崩溃。
2. **防整数溢出 (Integer Overflow Prevention)**：在计算 `slot_mapping` 时的 `block_id * block_size` 和 `page_id * block_size` 运算中，强制使用 `int64_t` 进行乘法运算，防止超大整数输入导致溢出产生负数内存索引。
3. **KV Cache 索引上界校验**：接收 `num_gpu_blocks` 参数，在 C++ 和 Python 侧对恶意的或非法的 `block_id`、`page_id` 等进行 `[0, num_gpu_blocks)` 的上界校验，防止 GPU 显存越界。
4. **GDN State 槽位校验**：在 Python 端补齐了 `state_slot` 对于 RNN/GDN 隐状态索引的上界合法性校验 (`0 <= s < dummy_gdn_slot`)。
5. **`block_tables` 索引 Bug 修复**：修正了 `update_decode_inplace` 中的 SP 并行索引问题，引入了 `q_offsets` 保证非 `sp_rank=0` 的节点能够正确对应其 Block Table 行偏移，避免显存写串。

## 4. Impact (影响与收益)

1. **彻底修复死锁**：极大地降低了通过 Ray 传输的 Payload 体积，从根本上消除了因 Ray Object Store 满载导致的不可预知死锁。
2. **性能飞跃**：消除了每 Step 的 Python 对象序列化/反序列化开销，TTFT 和 ITL 延迟更加平滑。
3. **架构解耦**：Engine 与 Runner 实现了清晰的网络边界，为未来支撑纯 C++ Runner (无需 Python 运行池)、跨节点 gRPC 调度奠定坚实基础。
