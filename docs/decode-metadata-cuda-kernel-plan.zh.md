# Decode Metadata Mapped Slab 与 CUDA Unpack Kernel 实施计划

## 1. 背景与目标

DLEngine 当前的 decode metadata 热路径包含多次序列化、反序列化和
host/device 中转：

```text
Python bytes
  -> Rust Vec / bincode decode
  -> PyO3 Python list
  -> pinned torch.Tensor
  -> CUDA tensor
  -> CUDA Graph persistent buffer copy
```

同一 batch 还会分别经过 `RunnerIn.aux()` 和 `RunnerIn.decode()`，导致重复解析。
每个字段独立构造 tensor、传输和复制，增加了 CPU 开销、CUDA launch 数量和两次
模型 graph replay 之间的 GPU idle gap。

本方案把 decode metadata 路径直接替换为：

```text
flat decode wire payload
  -> persistent mapped pinned host slab
  -> one CUDA unpack kernel
  -> persistent device output slab
  -> model CUDA Graph replay
```

首版目标：

- 热路径只有一个 decode metadata kernel launch；
- kernel 通过 UVA 直接读取 mapped pinned slab，不产生独立 metadata H2D memcpy；
- 不在每步创建 Python list、host tensor 或 device tensor；
- kernel 直接写模型和 CUDA Graph 使用的固定地址；
- 覆盖 eager/graph、GQA、MLA、DSv4、HiSparse 和 MTP decode；
- 保持 Pipeline Parallelism（PP）的 worker 隔离语义；
- Qwen3.5 Dense 在 RTX 4060 上端到端吞吐至少提升 5%。

Prefill metadata 不属于首版范围，继续使用当前路径。

## 2. 实施前置实验

Mapped host memory 会减少一次 memcpy launch，但 GPU 需要通过 PCIe 直接读取 host
memory。metadata 较小时 launch 开销可能占主导，但不能预设 UVA 一定更快。因此在
完整改造前必须在目标 RTX 4060 上实现最小 benchmark，对比：

1. mapped pinned slab -> CUDA kernel 直接读取；
2. pinned slab -> `cudaMemcpyAsync` -> CUDA kernel；
3. 当前逐字段 `torch.tensor(...).cuda()` 基线。

benchmark 使用与正式实现相同的字段数量、block-table 分布和 batch size。记录：

- host 侧 prepare 时间；
- CUDA memcpy/kernel 时间；
- 两次模型 graph launch 之间的 GPU idle gap；
- batch 1、8、16 的 P50/P99。

若 UVA 直读不能稳定优于一次聚合 memcpy，停止完整 UVA 集成，先更新本 RFC 和
Issue #228，再决定是否采用“一次 H2D + 一个 kernel”。不能在未验证时依靠理论
launch 数量选择生产路径。

## 3. Decode Wire 与 Slab Layout

### 3.1 协议边界

Decode 使用新的 versioned flat wire format；prefill、migration 和 result 协议保持
不变。Scheduler 和 worker 必须同步升级，旧 decode payload 不提供运行时 fallback，
而是返回明确的协议版本错误。

Header 至少包含：

- magic、protocol version、flags 和 payload byte length；
- `num_seqs`、普通 block 总数、最大 row blocks；
- compressed table descriptor 数量；
- 各 typed array 的 byte offset 和 element count。

Payload 使用 little-endian，所有 typed arrays 至少按 8 bytes 对齐，CUDA 向量化读取
字段按 16 bytes 对齐。变长二维表使用 row offsets 加连续 values：

```text
input_ids:             int64[num_seqs]
positions:             int64[num_seqs]
temperatures:          float32[num_seqs]
state_slots:           int64[num_seqs]
hisparse_slots:        int64[num_seqs]
seq_ids:               uint64[num_seqs]
slot_to_batch_row:     int32[max_num_seqs + 1]
block_row_offsets:     uint32[num_seqs + 1]
block_ids:             int32[num_blocks]
compressed_descriptors
compressed_row_offsets
compressed_block_ids
```

Decode 中恒定为 1 或 true 的 `seq_lens`、`sample_mask` 不再发送。Batch 级 flags
携带 dummy、是否全部 greedy、是否返回 completion logprobs 等控制信息。

### 3.2 Host Slab

每个 worker 只创建一个持久 host slab：

- 地址和容量在 worker 生命周期内保持不变；
- 初始化时通过 `cudaHostRegisterMapped` 注册；
- 注册后使用 `cudaHostGetDevicePointer` 获取 kernel 可访问地址；
- 初始化时检查 `canMapHostMemory`，不支持时启动失败；
- Rust staging API 借用输入 bytes，完成 header/offset/length 校验后执行一次 CPU
  memcpy，不构造 `Vec<Vec<_>>` 或 Python list；
- worker cleanup 时解除注册，保证异常初始化和重复 cleanup 安全。

### 3.3 Device Output Slab

每个 worker 创建一个固定地址的 device allocation，并从中建立 typed tensor views：

- input IDs 和 positions，容量支持 MTP 的 `2 * max_num_seqs`；
- temperatures；
- slot mapping；
- context lengths；
- SP-expanded block tables；
- GDN/DSv4 state slots；
- DSv4 compressed block tables；
- HiSparse slots 和 ring slot mapping；
- graph padding 和必要的 scalar metadata。

模型、`BatchContext` 和 graph runner 继续消费 typed tensors，不感知 raw slab layout。
所有 views 的地址从 graph capture 到 worker cleanup 保持不变。

## 4. CUDA Unpack Kernel

新增一个与模型架构无关的 CUDA JIT module，使用仓库已有的 TVM FFI/JIT 接入方式。
模块在 graph capture 前完成编译和 warmup，并在当前 PyTorch CUDA stream 上 launch。

一次 kernel launch 完成以下工作：

- 合并读取 host slab 中的 IDs、positions、temperatures 和控制字段；
- 计算 `context_lens = position + 1`；
- 根据 row offsets、position 和 block size 计算 `slot_mapping`；
- 展开普通 block tables，并安全填充 inactive rows/columns；
- clamp GDN 和 DSv4 state slots 到对应 dummy slot；
- 按 state slot scatter DSv4 compressed block tables，未使用位置填 dummy page；
- 生成 HiSparse slots 和 ring mapping；
- 根据 CPU 生成的 owner-change bitset 清理发生复用的 HiSparse residency；
- MTP lazy verify 时直接生成双 token IDs、positions、slot mapping，并更新 context
  lengths；
- 按选中的 graph bucket 初始化 inactive sequences，避免上一 batch 的残留数据。

Kernel 只能读取 CPU 已验证的 offset/count。CPU 校验必须覆盖：

- magic/version/header size；
- payload 截断、offset 溢出、alignment；
- `num_seqs`、block counts 和 compressed descriptors 的配置上限；
- row offsets 单调性和最后一个 offset 与 values 长度一致；
- state/HiSparse slot 值域；
- compressed ratio 与 worker cache plan 的一致性。

非法输入在 launch 前失败，不能依赖 device assertion 处理协议错误。

## 5. Runtime 与 CUDA Graph 集成

Metadata kernel 不进入模型 CUDA Graph。每步执行顺序为：

```text
stage payload into mapped host slab
  -> launch metadata unpack kernel on current stream
  -> FlashInfer plan / FlashMLA metadata preparation
  -> model CUDA Graph replay
```

同一 stream 的顺序保证后续 planning 和 replay 看到本 batch 的输出，无需 host
synchronize。Graph runner 改为直接使用 output slab views，删除 replay 前的
`copy_()`、`fill_()` 和临时 context-to-graph buffer 中转。

FlashInfer 的 page-plan key 仍由 CPU control metadata 提供。FlashInfer planning 必须
位于 unpack kernel 之后，因为它读取本步 context lengths 和 block tables。
FlashMLA/DSv4 自身需要捕获的 scheduling kernel 保持现有 capture 生命周期。

### 5.1 DLSLime Prepared Batch

每个 worker 只有一个持久 slot，状态机为：

```text
FREE -> PREPARED -> RUNNING -> FREE
```

- `run_batch` 在同一个调用内占用、执行并释放 slot；
- `prepare_batch` 占用 slot，`run_prepared` 执行后释放；
- 同一 worker 在 `PREPARED` 状态收到第二个 prepare 时立即报错；
- 取消、异常或断连必须释放 slot，不能永久卡在 `PREPARED`；
- 引擎不实现 slot pool，也不为 DLSLime 的多 outstanding 请求分配额外 buffer。

PP 不与此约束冲突：每个 PP stage/rank 是独立 worker 进程，各自拥有一个 slab 和
状态机。同一个 decode batch 可以同时进入多个 PP workers。Static PP prefill 不使用
本 decode slab。

## 6. 代码迁移边界

Rust/Python decode 接口由“返回包含多个 `Vec` 的 `DecodeMeta/BatchAuxData`”改为：

1. staging API 校验 flat payload 并写入 host slab；
2. 返回只包含 CPU 控制信息的轻量 `DecodeControl`；
3. Python 调用单个 unpack kernel；
4. `BatchContext` 直接引用持久 output views。

需要从 decode 热路径删除：

- 重复的 `RunnerIn.from_bytes()` 和 bincode decode；
- PyO3 `Vec -> list` getter；
- `torch.tensor(..., pin_memory=True).cuda()`；
- CPU 上按 SP 复制 block tables；
- graph runner 中对 IDs、positions、slot mapping、context lengths、block tables、
  state slots 和 HiSparse slots 的二次复制。

旧 decode 算法仅允许保留为测试 reference，不提供环境变量或生产 fallback。

## 7. 测试与正确性验收

### 7.1 Rust/Wire Tests

- flat decode round-trip；
- empty/dummy、batch 1 和 `max_num_seqs`；
- 不同长度普通及 compressed block rows；
- magic/version、截断 payload、未对齐 offset、溢出 count、非单调 row offsets；
- scheduler 生成的 descriptor 与 cache plan 一致。

### 7.2 CUDA Parity Tests

使用测试 reference 逐字段对比：

- GQA、MLA 和 DSv4；
- eager 与 CUDA Graph；
- SP block-table expansion；
- 无 block、跨 block boundary 和最大 context；
- GDN/DSv4 dummy state slots；
- compressed ratio 4/128 scatter；
- HiSparse slot reuse、owner change 和 ring mapping；
- MTP 首次 decode 与 lazy verify；
- 连续 replay 不同 batch size，确认 inactive 区域无 stale metadata。

### 7.3 Runtime Tests

- output tensor 地址在多步和不同 graph bucket 间不变；
- 热路径没有 host/device tensor allocation；
- DLSLime 第二个 outstanding prepare 返回明确错误；
- prepare/run 异常能够释放 slot；
- 多 PP workers 可以同时处理同一个 decode payload；
- malformed payload 在 CUDA launch 前失败。

Nsight Systems/Compute 必须证明：

- 每步只有一个 DLEngine decode metadata kernel；
- UVA 方案没有 metadata H2D memcpy；
- graph replay 前没有旧的逐字段 copy kernels；
- mapped host load 是合并访问，没有异常 page fault 或隐式同步。

## 8. 性能验收

首轮正式平台：

- 模型：Qwen3.5 Dense；
- GPU：RTX 4060；
- 模式：CUDA Graph enabled；
- 基线：同一 commit 的旧 decode metadata 实现；
- 主场景：batch 16、context 4K；
- 辅助场景：batch 1/8、context 1K/4K。

每个场景执行 100 steps warmup 和 1000 decode steps，独立运行 5 次。验收规则：

- 主场景 5 次 tokens/s 的中位数至少为旧实现的 105%；
- 辅助场景 tokens/s 不得回退超过 2%；
- 同时报告 decode prepare P50/P99、metadata kernel 时间、GPU idle gap、CPU 占用和
  mapped slab 实际字节数；
- benchmark 命令、配置、模型 checkpoint、CUDA/PyTorch/driver 版本和原始结果必须
  随实现 PR 提供。

GLM 5.2 是后续验证目标，不阻塞本次 Qwen3.5 Dense 交付。

## 9. 交付拆分

本 RFC 文档独立合入。后续实现至少拆为：

1. RTX 4060 mapped-UVA 与 memcpy 聚合路径 microbenchmark；
2. flat decode wire 与 mapped slab 生命周期；
3. CUDA unpack kernel 和逐字段 parity tests；
4. InputPreparer/GraphRunner/DLSLime 集成；
5. Qwen3.5 Dense 性能验收；
6. GLM 5.2 验证和必要的架构扩展。

性能实现 PR 必须引用本 RFC Issue，并为未完成的 GLM 5.2 工作建立独立 follow-up
Issue，不能只在 PR 正文留下说明。
