# DeepSeek SP 下 MLA 路径 Q `offsets` 支持与 DLSlime `hao_basic` 原生扩展设计

## 1. 目标

本文档的目标是两件事：

1. 修改 **NanoDeploy 的 MLA SP 路径**，让 MLA decode 中 **Q 的 all-to-all** 真正使用 `offsets`，从而减少 Q 传输通信量。
2. 修改 **DLSlime 的 `hao_basic`**，为其增加原生 `offsets` 支持，使 MLA 的 Q 路径可以直接走 `hao_basic`。

这里的设计边界已经明确：

1. **只有 Q 的传输需要 `offsets` 参数。**
2. `res` / `lse` 的传输不需要 `offsets`，保持现状，不在本次设计范围内。

## 2. 结论先行

### 2.1 本次要落地的能力

本次正式落地的能力是：

1. `FlashMLAImpl` 的 Q all-to-all 调用改为显式传入 `offsets=context.q_offsets`。
2. `hao_basic` 在 native path 上支持 `all_to_all(..., offsets=...)`。
3. `offsets` 在本次 patch 中的正式语义只服务于 **MLA Q non-transpose all-to-all**。
4. 该能力必须真正减少 Q 通信量，而不是只做接口透传。

### 2.2 本次不采用的方案

本次明确不采用：

1. 不采用“在 `offsets` 分支中切回旧 `all_to_all_intra_ll` kernel”的方案。
2. 不把本次改造定义成单纯的 GQA 兼容。
3. 不在 `hao_basic` adapter 层用额外重排来模拟 offsets。
4. 不在 `_compat_mode` 下支持 offsets。
5. 不给 `res` / `lse` 传输增加 offsets。

### 2.3 本次的技术边界

本次 patch 的正式范围只有一件事：

1. **MLA 的 Q all-to-all non-transpose 路径使用 offsets。**

本次 patch 明确不包含：

1. `res` / `lse` 路径的 offsets 化。
2. 任何 `is_transpose=True` 场景下的 offsets 支持。

正式口径是：

1. 本 patch 必须让 MLA 主路径中的 **Q all-to-all** 真正使用 offsets。
2. `res` / `lse` 路径保持现有实现，不传 offsets，不修改布局协议。

## 3. 当前代码事实

### 3.1 当前 MLA decode 路径没有使用 offsets

当前 `FlashMLAImpl` 在 SP decode 中：

1. 会先把本 rank 的 Q 拷入 `q_buffer.local_buffer`。
2. 然后调用 `q_buffer.all_to_all_ll(q.view([bs, -1]), mask=context.q_mask)`。
3. 这里没有传 `offsets`。

因此当前 MLA Q all-to-all 在 `hao_basic` 上无法利用 `q_offsets` 做 packed 发送。

参考：

1. [attention.py](/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/nanodeploy/layers/attention.py:247)
2. [attention.py](/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/nanodeploy/layers/attention.py:261)

### 3.2 NanoDeploy 已经有可直接复用的 `q_offsets`

当前 `prepare_decode_cpp` 已经在 decode metadata 中构造了：

1. `sp_valid_request_counts`
2. `q_offsets`

其语义是：

1. `q_offsets[i] ~ q_offsets[i+1]` 表示 source rank `i` 的有效 Q 在 packed 输出中的区间。
2. `q_offsets` 长度为 `sp_size + 1`。
3. `q_offsets` 之后会被拷到 CUDA tensor，并放入运行时 context。

这说明 MLA 路径并不缺元数据，缺的是“真正消费它的调用和底层实现”。

参考：

1. [model_runner_utils.cpp](/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/csrc/nanodeploy/worker/model_runner_utils.cpp:217)
2. [model_runner_utils.cpp](/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/csrc/nanodeploy/worker/model_runner_utils.cpp:323)
3. [model_runner.py](/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/nanodeploy/worker/model_runner.py:428)
4. [model_runner.py](/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/nanodeploy/worker/model_runner.py:613)

### 3.3 当前 `hao_basic` 不支持 offsets

当前限制包括：

1. `HaoAllToAllBufferAdapter.all_to_all_ll(...)` 在 `offsets is not None` 时直接抛 `NotImplementedError`。
2. DLSlime `AllToAllBuffer::all_to_all(...)` 没有 `offsets` 参数。
3. `AllToAllBuffer::dispatch_basic(...)` 只调用 `intranode_alltoall(...)`。
4. `intranode_alltoall(...)` 当前也没有 offsets 参数。

参考：

1. [sp_backend.py](/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/nanodeploy/worker/sp_backend.py:173)
2. [alltoall_buffer.h](/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.h:36)
3. [alltoall_buffer.cpp](/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.cpp:155)
4. [all_to_all_intra_ll.h](/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/all_to_all_intra_ll.h:21)

### 3.4 当前 `hao_basic` 的 non-transpose masked path 还带有 padding 兼容逻辑

当前 NanoDeploy adapter 在 native `hao_basic` 的 masked non-transpose path 上会：

1. 若 `x.rows < max_bs`，先 pad 到 `max_bs`。

这套逻辑是为当前 `hao_basic` 的固定 batch-size 语义服务的。

但对于 Q offsets 路径，这会直接破坏“只发有效 Q”的目标，因为：

1. offsets 模式下应发送 `counts[rank]` 个真实 Q。
2. 若先 pad 到 `max_bs`，通信量不会下降。
3. 若底层 kernel 把 pad 行也当有效输入处理，还会破坏 packed 布局。

因此本次 patch 必须显式绕过这个 padding 逻辑。

参考：

1. [sp_backend.py](/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/nanodeploy/worker/sp_backend.py:189)
2. [sp_backend.py](/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/nanodeploy/worker/sp_backend.py:252)

## 4. 正式设计

### 4.1 设计原则

1. MLA 必须真正消费 `context.q_offsets`。
2. offsets 的收益必须来自 **少发无效 Q**，而不是只改变输出布局。
3. `hao_basic` 的 offsets 支持必须是 **native kernel 能力**，不是 fallback 到旧实现。
4. 不修改 MLA attention 的上层语义，只修改其 Q 的 SP all-to-all 实现方式。
5. `_compat_mode` 不支持 offsets。
6. `offsets` 只用于 **Q 的 non-transpose all-to-all**。
7. `res` / `lse` 路径保持现状，不接入 offsets。

### 4.2 MLA 路径里 offsets 的正式使用位置

本次 MLA 路径中，正式使用 offsets 的位置只有一处：

1. `FlashMLAImpl.forward(...)` 中的 `q_buffer.all_to_all_ll(...)`。

正式改法：

1. 当前：`q_buffer.all_to_all_ll(q.view([bs, -1]), mask=context.q_mask)`
2. 改为：`q_buffer.all_to_all_ll(q.view([bs, -1]), mask=context.q_mask, offsets=context.q_offsets)`

改完后的含义是：

1. 每个 source rank 只发送自己的有效 Q 数量。
2. 接收侧仍得到 `[sp_size, max_num_seqs, ...]` 兼容视图。
3. flatten 后前 `attention_compute_bs` 行是 packed 的有效 Q。
4. `q = q[: context.attention_compute_bs]` 这段上层逻辑保持不变。
5. `res_buffer` / `lse_buffer` 调用保持原样，不传 offsets。

### 4.3 本次 offsets 语义定义

#### 4.3.1 输入输出语义

当 `is_transpose == False` 且 `offsets` 存在时：

1. `offsets.dtype == int32`
2. `offsets.shape == [world_size + 1]`
3. `offsets[0] == 0`
4. `counts[i] = offsets[i+1] - offsets[i]`
5. `counts[i]` 表示 source rank `i` 的有效 Q 数量
6. 当前 sender rank 为 `rank` 时，输入 `x.shape[0] == counts[rank]`
7. 输出视图保持 `[world_size, max_bs, msg]`
8. flatten 输出后，source rank `i` 的有效消息位于 `flat[offsets[i]:offsets[i+1]]`

这一定义与 NanoDeploy 当前 `q_offsets` 的构造语义一致，也与 MLA Q all-to-all 的需要完全一致。

#### 4.3.2 `mask` 语义保持不变

保持：

1. `mask.shape == [world_size, max_bs]`
2. `mask[target_rank, slot]` 表示给 target rank 的第 `slot` 条消息是否发送

在 Q offsets 模式下：

1. kernel 只会处理 `slot < counts[rank]` 的真实输入行
2. `slot >= counts[rank]` 的部分不会参与发送
3. 因此 offsets 带来的通信缩减来自“只遍历真实 `counts[rank]`”，而不是依赖 mask 去屏蔽 pad 行

### 4.4 DLSlime `hao_basic` 的原生扩展方案

#### 4.4.1 总体方案

不引入旧 kernel fallback，直接扩展当前 `hao_basic` 路径：

1. `AllToAllBuffer::all_to_all(...)` 增加 `offsets` 参数
2. `AllToAllBuffer::dispatch_basic(...)` 增加 `offsets` 参数
3. `intranode_alltoall(...)` 增加 `offsets` 参数
4. `intranode_alltoall_kernel` 在 non-transpose 分支中增加 offsets-aware 写入逻辑

#### 4.4.2 native offsets 模式下的 kernel 语义

当 `is_transpose == false` 且 `offsets` 存在时：

1. 本 rank 的真实 batch size 定义为 `counts[rank]`
2. kernel 不再按 `max_bs` 遍历输入 token
3. kernel 只遍历 `counts[rank]` 个真实 token
4. 写入远端 buffer 时，不再写入 `rank * max_bs + token_idx`
5. 改为写入 `offsets[rank] + token_idx`

等价地说：

1. 远端 local buffer 的 flatten 视图前 `total_messages = offsets[world_size]` 行是 packed 有效数据
2. source rank `i` 的有效 slice 位于 `offsets[i] ~ offsets[i+1]`

#### 4.4.3 为什么这能减少 MLA Q 通信量

当前没有 offsets 时：

1. `hao_basic` non-transpose path 逻辑上是按固定 batch 形态工作
2. MLA decode 中每个 rank 的真实 Q 数常常远小于 `max_bs`
3. 这会导致无效槽位仍然参与通信路径或至少参与底层处理流程

增加 offsets 后：

1. 发送侧 batch size 下降为 `counts[rank]`
2. 无效 pad 行不会进入 kernel 主循环
3. 给每个 target rank 的写入数据量也下降到真实有效 Q 数

因此 offsets 在 MLA Q path 的收益是直接、可验证的通信缩减。

#### 4.4.4 offsets 的支持范围

本次 patch 中，`hao_basic` 原生 offsets 支持的正式范围限定为：

1. `is_transpose == false`

因此在 DLSlime 里应明确：

1. 若 `offsets` 存在且 `is_transpose == true`，直接 `TORCH_CHECK(false, ...)`

这不是待补充语义，而是本次设计的正式边界，因为只有 Q 的传输需要 offsets。

#### 4.4.5 参数校验

当 `offsets` 存在时，DLSlime 侧必须新增以下校验：

1. `x.is_cuda()`
2. `x.is_contiguous()`
3. `x.dim() == 2`
4. `offsets` 必须存在于 CUDA 上
5. `offsets.dtype == int32`
6. `offsets.dim() == 1`
7. `offsets.numel() == world_size + 1`
8. `offsets[0] == 0`
9. `offsets` 单调不减
10. `counts[i] <= max_batch_size` 对所有 `i` 成立
11. `total_messages <= world_size * max_batch_size`
12. `is_transpose == false` 时，`x.size(0) == counts[rank]`
13. `msg_size * itemsize` 仍需 16-byte aligned

#### 4.4.6 Python binding 变更

需要同步修改 pybind：

1. `AllToAllBuffer.all_to_all(x, impl=..., is_transpose=..., mask=None, offsets=None)`

这样 NanoDeploy 才能把 Q 的 `offsets` 原样透传到 `hao_basic`。

### 4.5 NanoDeploy adapter 方案

修改 [sp_backend.py](/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/nanodeploy/worker/sp_backend.py) 中的 `HaoAllToAllBufferAdapter`：

1. `offsets is not None` 时，不再直接报错
2. 若 `_compat_mode`，仍报 `NotImplementedError`
3. 若 `offsets is not None and is_transpose == false and native path`：
   1. 不做 `_pad_masked_non_transpose_input(...)`
   2. 不做 compat mask transpose
   3. 直接透传 `offsets`
4. 若 `offsets is not None and is_transpose == true`：
   1. 直接报错

这保证了：

1. MLA Q path 可以走到新的 native offsets kernel
2. 现有无 offsets 路径行为不变
3. 不会因为历史兼容逻辑吃掉 offsets 的收益
4. 文档和实现边界保持一致：只有 Q 路径用 offsets

## 5. Patch Plan

### 5.1 NanoDeploy Patch Plan

#### Patch N1: 修改 MLA 路径，真正传入 offsets

修改文件：

1. `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/nanodeploy/layers/attention.py`

修改内容：

1. 在 `FlashMLAImpl.forward(...)` 的 Q all-to-all 调用中增加 `offsets=context.q_offsets`
2. 保持 `q = q[: context.attention_compute_bs]` 逻辑不变
3. `res_buffer` / `lse_buffer` 的调用不改

#### Patch N2: 修改 `HaoAllToAllBufferAdapter`

修改文件：

1. `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/nanodeploy/worker/sp_backend.py`

修改内容：

1. native `hao_basic` path 允许 non-transpose offsets
2. `_compat_mode + offsets` 继续报错
3. `is_transpose + offsets` 直接报错
4. offsets path 显式跳过 `_pad_masked_non_transpose_input(...)`

#### Patch N3: NanoDeploy 单测补齐

建议修改：

1. `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April/tests/test_sp_backend.py`

新增覆盖点：

1. native `hao_basic` + non-transpose + offsets 会透传到 `_buffer.all_to_all(...)`
2. native `hao_basic` + non-transpose + offsets 不会触发 padding
3. `_compat_mode + offsets` 报错
4. `is_transpose + offsets` 报错

### 5.2 DLSlime Patch Plan

#### Patch D1: 扩展 `AllToAllBuffer` C++ / Python ABI

修改文件：

1. `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.h`
2. `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.cpp`
3. `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/python/bind.cpp`

修改内容：

1. `all_to_all(...)` 新增 `offsets`
2. `dispatch_basic(...)` 新增 `offsets`
3. pybind 新增 `py::arg("offsets") = py::none()`

#### Patch D2: 扩展 `hao_basic` native kernel 接口

修改文件：

1. `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/all_to_all_intra_ll.h`
2. `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/all_to_all_intra_ll.cu`

修改内容：

1. `intranode_alltoall(...)` 增加 `offsets`
2. `intranode_alltoall_kernel` 增加 offsets-aware non-transpose 分支
3. 保持无 offsets 时现有行为完全不变
4. `offsets + is_transpose=True` 直接拒绝

说明：

1. 这里改的是 `hao_basic` 当前路径本身
2. 不是把 `dispatch_basic` 分流到旧 `all_to_all_intra_ll(...)`

#### Patch D3: 在 `dispatch_basic(...)` 中接入 offsets 路径

修改文件：

1. `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.cpp`

修改内容：

1. offsets 为空：保持现有 `intranode_alltoall(...)`
2. offsets 非空：仍走 `intranode_alltoall(...)`，但传入 offsets
3. 在 host 侧做所有必要的参数校验

#### Patch D4: DLSlime direct tests

建议新增：

1. `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime/tests/python/test_alltoall_buffer_offsets.py`

新增覆盖点：

1. `AllToAllBuffer.all_to_all(..., is_transpose=False, offsets=...)`
2. non-transpose + offsets + mask
3. 非法 offsets shape / dtype / monotonicity
4. `x.rows != counts[rank]`
5. `offsets + is_transpose=True` 必须报错

## 6. 验收标准

### 6.1 功能验收

必须同时满足：

1. MLA Q all-to-all 在 NanoDeploy 中显式传入 `context.q_offsets`
2. `hao_basic` native path 能接受并执行 non-transpose offsets
3. offsets 路径不会触发 native `hao_basic` 的 padding 兼容逻辑
4. `res` / `lse` 路径保持现状，不传 offsets
5. 无 offsets 路径行为不变

### 6.2 正确性验收

必须覆盖：

1. MLA Q path 在 `legacy_ll` 与 `hao_basic` 下结果一致
2. `hao_basic` non-transpose offsets 的 packed 输出布局正确
3. 非法 offsets 参数能被显式拒绝
4. `_compat_mode + offsets` 被显式拒绝
5. `is_transpose + offsets` 被显式拒绝

### 6.3 性能验收

必须体现：

1. MLA Q path 的实际发送行数从 `max_bs` 降为 `counts[rank]`
2. offsets path 不引入额外 pad/copy
3. graph 路径中 `q_offsets` 继续作为 CUDA tensor 使用，不引入隐藏 CPU->CUDA copy

## 7. 实施顺序

建议顺序：

1. 先做 NanoDeploy `FlashMLAImpl` 调用点改造
2. 再做 `HaoAllToAllBufferAdapter` offsets 分支
3. 再做 DLSlime `AllToAllBuffer` / pybind ABI 扩展
4. 再做 `hao_basic` native kernel offsets 支持
5. 再补 DLSlime direct tests
6. 最后回归 NanoDeploy MLA correctness 测试

原因：

1. 这样可以先把上层意图固定为“MLA 的 Q 确实要消费 offsets”
2. 再让 adapter 和底层按这个目标对齐
3. 便于把“接口问题”和“kernel 语义问题”拆开调试

## 8. 风险与注意事项

### 风险 1: 只改 DLSlime，不改 MLA 调用点

后果：

1. `hao_basic` 虽然新增了 offsets 支持
2. 但 MLA 路径仍不会用上它
3. 实际通信量不会下降

规避：

1. `FlashMLAImpl` 的调用点改造必须是本 patch 的第一优先级

### 风险 2: offsets 路径误走 padding 兼容逻辑

后果：

1. 通信量没有下降
2. packed 布局可能被破坏

规避：

1. adapter 中 offsets 分支必须显式绕过 padding

### 风险 3: 文档边界不清导致实现扩散到 `res` / `lse`

后果：

1. 需求会从“MLA Q path 用 offsets”膨胀成“重构整个 MLA 通信协议”
2. 实现复杂度和调试复杂度显著上升

规避：

1. 当前文档明确写死：只有 Q 的传输需要 offsets
2. `res` / `lse` 路径保持不变

## 9. 最终落地口径

本次 patch 完成后的正式口径是：

1. MLA SP 路径已经开始真实使用 offsets
2. 这个 offsets 使用点仅位于 MLA 的 Q all-to-all 路径
3. 该能力用于减少 MLA decode 中 Q 的通信量
4. DLSlime `hao_basic` 已具备原生 non-transpose offsets 支持
5. `res` / `lse` 传输不使用 offsets，并保持现状
6. 本次不通过旧 kernel fallback 提供 offsets
