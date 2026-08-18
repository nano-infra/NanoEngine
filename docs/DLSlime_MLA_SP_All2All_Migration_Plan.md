# DLSlime MLA SP All2All 迁移与双后端切换计划

> 历史设计说明（2026-08-18）：本文记录迁移过程，其中的 `legacy_ll` 与
> `nccl_compact` 已从当前实现删除；当前支持的 SP backend 为 `hao_basic` 和 `nccl`。

## 仓库路径

- 正在使用的DLSlime 仓库：`/mnt/nvme1n1/ml_research/linbinbin1/DLSlime`

- NanoDeploy 路径: `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April`

- 灏哥分支路径: `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime_hao_0403`

之前我使用的安装方式

```
cd /mnt/nvme1n1/ml_research/linbinbin1/DLSlime
BUILD_INTRA_OPS=ON pip install -v -e  .  --no-build-isolation
```

## 1. 范围

本计划只覆盖 MLA 模型路径，即当前 dpsk/DeepSeek V2 在 NanoDeploy 中实际使用的 SP decode 路径。

并且这里的“双后端切换”定义为：

- **启动时选后端**
- 不做单个 worker 进程内的运行中热切换
- 不要求在已经完成 CUDAGraph capture 的进程里动态换 backend

- 关注调用点：
  - `nanodeploy/layers/attention.py:261`
  - `nanodeploy/layers/attention.py:373`
  - `nanodeploy/layers/attention.py:378`
- 关注初始化点：
  - `nanodeploy/worker/sp_context.py:5`
  - `nanodeploy/worker/sp_context.py:37`
  - `nanodeploy/worker/sp_context.py:44`
  - `nanodeploy/worker/sp_context.py:52`
  - `nanodeploy/worker/sp_context.py:60`
- 当前实际模型路径：
  - `nanodeploy/models/deepseek_v2.py:577`
  - `nanodeploy/models/deepseek_v2.py:584`

不在本次主范围内：

- GQA/`FlashAttentionImpl` 路径
- `offsets` 驱动的老 All2All 兼容改造
- 非 `KernelImpl.Basic` 的新 kernel 版本
- 任何业务代码修改
- `DLSlime` 的发版、wheel 发布、版本 pin 与自动安装流程

## 2. 现状结论

### 2.1 NanoDeploy 对 SP Buffer 的真实依赖

NanoDeploy 的 MLA SP 路径并不直接依赖某个具体 kernel 名，而是依赖一组 **Buffer 语义**：

- `local_buffer`：可写 staging 区，用于先把本 rank 的 Q / Res / LSE 写入本地 buffer
- `connect_full_mesh(...)`：完成 IPC/NVLink 指针互联
- `all_to_all_ll(x, is_transpose=False, mask=None)`：执行实际通信并返回 `[world_size, max_bs, msg]` 视图

对 MLA 路径而言：

- `q_buffer` 调用是 `is_transpose=False`，且只传 `mask`
- `res_buffer` / `lse_buffer` 调用是 `is_transpose=True`，且只传 `mask`
- MLA 路径 **不依赖 `offsets`**

### 2.2 旧版 DLSlime 能力

旧版 `DLSlime` 中，NanoDeploy 当前用的是：

- Python 包装：`dlslime/buffer/intra/all_to_all_intra_ll_buffer.py`
- C++ 类：`AllToAllIntraLLBuffer`
- Kernel：`all_to_all_intra_ll_kernel`

旧接口是 NanoDeploy 现在的直接依赖：

- `get_buffer_size_hint(...)`
- `get_local_buffer()`
- `connect_full_mesh(...)`
- `all_to_all_ll(x, is_transpose, mask, offsets)`

### 2.3 灏哥分支的新实现现状

`DLSlime_hao_0403` 里新增的是：

- C++ 类：`AllToAllBuffer`
- 本计划接入目标：`KernelImpl.Basic`
- 绑定入口：`csrc/python/bind.cpp`
- 编译接入：`csrc/ops/CMakeLists.txt` 中新增 `alltoall_buffer.cpp`
- 基准测试：
  - `bench/python/intra_alltoall_test.py`
  - `microbench-linbinbin/dlslime_intranode_alltoall_bench.py`

新类的 Python/C++ 接口是：

- `get_ipc_handle_info()`
- `connect_full_mesh(all_handles)`
- `reset_semaphore()`
- `all_to_all(x, impl=KernelImpl.Basic, is_transpose=True, mask=None)`

补充说明：

- 灏哥分支里虽然还保留了 `AllToAllIntraLLBuffer` 类名，但该分支的 `allToAllLL2D(...)` 当前没有连到实际 intranode kernel，不能拿它作为“旧语义”的判断依据
- 本文关于 `mask` 语义的结论，只以：
  - 当前生产仓库 `DLSlime` 的 `AllToAllIntraLLBuffer`
  - 灏哥分支的 `AllToAllBuffer::dispatch_basic`
  为准

### 2.4 已核实的 All2All Mask 定义

先说 NanoDeploy 当前调用侧的语义：

- `q_mask` 的 shape 是 `[sp_size, max_num_seqs]`
- `q_mask[target_rank, seq_id] = 1` 表示：**当前 master rank 的第 `seq_id` 条请求，需要把该条 Q 发送给 `target_rank`**
- `q_mask` 由 `global_context_lens` 派生，并且会把 `q_mask[sp_rank, :]` 整行清零

- `res_lse_mask` 的 shape 也是 `[sp_size, max_num_seqs]`
- `res_lse_mask[target_rank, seq_id] = 1` 表示：**当前 rank 需要把该条 partial `res/lse` 发回 `target_rank`**
- `res_lse_mask` 由 `context_lens` 派生，并且同样会把 `res_lse_mask[sp_rank, :]` 整行清零

- 因为 self 行会被清零，`attention.py` 会先把本 rank 自己的 Q / Res / LSE 写进 `local_buffer`
- 所以返回视图里的 self slice 不是通过通信得到的，而是本地预填得到的

旧版 `DLSlime` 的 `AllToAllIntraLLBuffer`：

- kernel 按 `mask + sm_id * max_bs + msg_idx` 读取 mask，其中 `sm_id == dst_rank`
- 这等价于按 `mask[dst_rank, slot]` 读取
- 因此旧接口的 `mask` 定义是 **`[world_size, max_bs] = [target_rank, slot]`**
- `mask[target_rank, slot] = 1` 的含义是：**当前 sender rank 的第 `slot` 行消息要发给 `target_rank`**
- `is_transpose` 只改变输入 `x` 的解释方式，不改变 `mask` 的索引方式
- 返回布局固定是 `[world_size, max_bs, msg]`，第一维表示 source rank

灏哥分支 `AllToAllBuffer::dispatch_basic`：

- 当 `mask` 存在时，C++ 显式校验 `mask.shape == [batch_size, world_size]`
- 此时 `batch_size = x.size(0)`，而不是 `x.size(0) / world_size`
- Basic kernel 按 `mask[token_idx, dst_rank]` 读取，索引公式是 `mask[token_i * world_size + dst_rank]`
- 因此新 Basic 的 masked path 定义是 **`[batch_size, world_size] = [slot, target_rank]`**
- `mask[slot, target_rank] = 1` 的含义是：**当前 rank 的第 `slot` 行 `x` 要发给 `target_rank`**
- 当 `mask is None` 时，输入 `x` 才按 `[world_size * batch_size, msg]` 的 dst-major 布局解释
- `dispatch_basic(...)` 当前没有使用 `is_transpose` 参数；也就是说，bench 里传 `is_transpose=True` 只能说明“无 mask 的 dst-major 路径能跑”，不能说明“masked transpose 语义已经实现”

这部分结论可以收敛为一句话：

- **旧接口的 `mask` 是 `[target_rank, slot]`，新 Basic masked path 的 `mask` 是 `[slot, target_rank]`**
- **两者不是“可能不同”，而是已经由代码确认的不同约定**
- **`AllToAllBuffer::KernelImpl.Basic` 当前也不是“可能不支持 transpose”，而是 masked path 下根本没有消费 `is_transpose`**

## 3. 兼容性判断

## 3.1 结论

### 结论 A

**可以把新实现迁移进旧 DLSlime，并与旧实现共存。**

理由：

- 灏哥分支已经证明同一个 `_slime_c` 模块里可以同时暴露：
  - `AllToAllIntraLLBuffer`
  - `AllToAllBuffer`
  - `KernelImpl`
- 编译层面只需要在 `BUILD_INTRA_OPS` 下补充新文件和绑定，不需要移除旧实现

### 结论 B

**可以在 NanoDeploy 中同时接入两套实现并做切换，但不能直接把 `AllToAllBuffer` 当成 `AllToAllIntraLLBuffer` 替换。**

理由：

- 方法名不同：`all_to_all_ll(...)` vs `all_to_all(...)`
- 句柄交换接口不同：`buffer_info` vs `get_ipc_handle_info`
- `AllToAllBuffer` 当前没有暴露 `local_buffer`
- `mask` 布局已确认不一致：旧接口是 `[target_rank, slot]`，新 Basic masked path 是 `[slot, target_rank]`
- `KernelImpl.Basic` 当前 masked path 不消费 `is_transpose`；因此不能直接覆盖旧接口的 `is_transpose=True` 语义
- 新基准主要覆盖的是 **无 mask 的** `[world_size * bs, msg]` 路径，不能证明 MLA 所需的 masked 路径已经对齐

### 结论 C

**如果只做 MLA/dpsk，问题比“全量兼容旧接口”简单，但仍然需要一个兼容层。**

原因：

- MLA 路径不需要 `offsets`
- 但仍然需要：
  - `q_buffer` 的非 transpose 路径
  - `res/lse` 的 transpose 路径
  - 对 `local_buffer` 的直接写访问

### 结论 D

**本次迁移计划应把“让新版 `KernelImpl.Basic` 真正消费 `is_transpose`”作为 Step 1 的明确改造项，而不是把它留给 NanoDeploy adapter 侧兜底。**

理由：

- `q` 路径天然对应新版 masked `x=[batch, msg]` 的非 transpose 语义
- 真正缺失的是 `res/lse` 需要的 transpose 输入解释
- 如果把 transpose 语义补在 `DLSlime` 新后端里，NanoDeploy adapter 只需要负责：
  - 方法名适配
  - 句柄交换适配
  - bring-up 阶段的临时 `mask` 转换
- 这样可以避免在 `NanoDeploy` 侧额外做 `res/lse` 重排，迁移边界更清晰

## 4. 关键差异清单

| 维度 | 旧版 `AllToAllIntraLLBuffer` | 新版 `AllToAllBuffer` | 对 MLA 接入的影响 |
| --- | --- | --- | --- |
| Python 调用名 | `all_to_all_ll` | `all_to_all` | 需要适配层 |
| 本地 buffer 访问 | 有 `local_buffer` | 当前未暴露 | 必须补 getter 或兼容包装 |
| 句柄交换 | `buffer_info` + `connect_full_mesh` | `get_ipc_handle_info` + `connect_full_mesh` | 需要适配层 |
| kernel 选择 | 单一老 kernel | 本计划只接 `Basic` | 切换点可下沉到 backend 配置 |
| 返回 view | `[world_size, max_bs, msg]` | `[world_size, max_batch_size, msg]` | 输出布局基本一致，可继续沿用上层 combine 逻辑 |
| `mask` 形状 | `[world_size, max_bs] = [target_rank, slot]` | masked path 当前为 `[batch_size, world_size] = [slot, target_rank]` | bring-up 可由 adapter 转换，服务态应让新版直接支持旧布局 |
| `is_transpose` | kernel 真正消费该参数并切换输入布局 | `Basic` 当前不消费该参数；是否 masked 反而决定输入约定 | `q` 和 `res/lse` 不能共用同一套“只传 flag”的接法 |
| `offsets` | 支持 | 新主实现未提供 | MLA 不受影响，GQA 不在本次范围 |
| 典型验证路径 | 老单测 + 偏兼容性 | benchmark 为主 | 需要补 MLA 数值对比 |

## 5. 推荐方案

## 5.1 总体策略

推荐采用“两层共存”：

1. 在 `DLSlime` 中保留旧 `AllToAllIntraLLBuffer`
2. 新增 `AllToAllBuffer + KernelImpl`
3. 在 `NanoDeploy` 中增加一个 **SP Buffer 抽象层**
4. 通过配置切换：
   - `legacy_ll`
   - `hao_basic`

这样可以做到：

- 默认路径不变
- 新实现灰度验证
- 同一套 `attention.py` 调用点保持稳定

这里的“切换”进一步收敛为：

- 通过 `NanoDeploy` 启动配置选择 backend
- 一个 `ModelRunner` 进程在生命周期内只绑定一种 SP backend
- 如果切换 backend，需要重启 worker，并重新完成 `SPContext` 初始化与 CUDAGraph capture

并且推荐在 `DLSlime` 新后端内部直接补齐 transpose 语义：

- `is_transpose=False`：
  - 输入 `x` 解释为 `[batch, msg]`
  - 适用于 MLA 的 `q_buffer`
- `is_transpose=True`：
  - 输入 `x` 解释为 `[world_size * batch, msg]`，即按 target-rank 分块后的展平视图
  - 适用于 MLA 的 `res_buffer` / `lse_buffer`
- `mask` 的服务态目标应继续沿用 NanoDeploy 当前语义的 `[world_size, batch] = [target_rank, slot]`
  - bring-up 阶段允许 adapter 临时转成新版当前使用的 `[batch, world_size] = [slot, target_rank]`
  - 但这不应成为 NanoDeploy 推理服务的长期热路径设计

## 5.2 NanoDeploy 侧的最佳切换点

切换点应放在：

- `nanodeploy/worker/sp_context.py`

不要放在：

- `nanodeploy/layers/attention.py`

原因：

- `attention.py` 当前已经只依赖统一的 buffer 语义
- 真正与具体后端强绑定的是 buffer 的构造与互联
- 把切换放在 `sp_context.py` 可以让上层调用点不变

## 5.3 推荐的兼容抽象

建议定义一个 MLA 专用的 buffer 协议，最少包含：

```python
class MLAAllToAllBufferProtocol:
    @property
    def local_buffer(self) -> torch.Tensor: ...

    def connect_full_mesh(self, group) -> None: ...

    def all_to_all_ll(
        self,
        x: torch.Tensor,
        is_transpose: bool = False,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
```

仅有实例协议还不够，`SPContext` 构造层还需要一个 backend factory 来负责：

- `buffer_size_bytes` 的计算
- `q_buffer` / `res_buffer` / `lse_buffer` 的实例化
- 新旧后端各自的句柄交换入口差异

原因是当前 `sp_context.py` 在构造期就依赖 `get_buffer_size_hint(...)` 来申请 buffer，而新版 `AllToAllBuffer` 的构造参数已经变成 `buffer_size_bytes`，两者不能只靠运行时实例协议抹平。

然后实现两个 adapter：

- `LegacyIntraLLBufferAdapter`
  - 直接包装 `AllToAllIntraLLBuffer`
- `HaoAllToAllBufferAdapter`
  - 内部包装 `AllToAllBuffer`
  - 对外模拟旧接口语义
  - dev bring-up 阶段允许临时把旧接口的 `mask[target_rank, slot]` 转成 `mask[slot, target_rank]`
  - 服务态目标是直接把旧 `mask[target_rank, slot]` 透传给新版 `KernelImpl.Basic`
  - 直接把 `is_transpose` 透传给新版 `KernelImpl.Basic`

并建议在 `NanoDeploy` 侧再加一个 factory 抽象，例如：

```python
class MLAAllToAllBackendFactory:
    @staticmethod
    def get_buffer_size_hint(
        max_dispatch_per_msg: int,
        max_bs: int,
        max_msg_size: int,
        itemsize: int,
    ) -> int: ...

    def create_q_buffer(...) -> MLAAllToAllBufferProtocol: ...
    def create_res_buffer(...) -> MLAAllToAllBufferProtocol: ...
    def create_lse_buffer(...) -> MLAAllToAllBufferProtocol: ...
```

## 5.4 推荐的新版 Basic 改造方式

建议把这项改造直接放进 `DLSlime` Step 1，而不是留到 NanoDeploy 再绕过去。

目标语义：

1. `mask is not None` 且 `is_transpose=False`
   - 保持现有行为
   - `x.shape == [batch_size, msg]`
   - `mask.shape == [world_size, batch_size]`
   - `mask[target_rank, slot] = 1` 表示把当前第 `slot` 行消息发给 `target_rank`
   - kernel 按 `x[slot] -> dst_rank` 发送
2. `mask is not None` 且 `is_transpose=True`
   - 新增并补齐的行为
   - `x.shape == [world_size * batch_size, msg]`
   - `mask.shape == [world_size, batch_size]`
   - kernel 对每个 `dst_rank` 读取 `x[dst_rank, slot]` 对应的那一段输入，再按 `mask[dst_rank, slot]` 判定是否发送
3. `mask is None`
   - 保持当前 benchmark 已覆盖的 `[world_size * batch_size, msg]` 路径

建议修改点：

- `DLSlime_hao_0403/csrc/ops/intra_ll/all_to_all/alltoall_buffer.h`
  - 保持 `all_to_all(..., mask=...)` 入口不变
  - 明确 masked path 的目标语义直接收敛为 `mask[target_rank, slot]`
- `DLSlime_hao_0403/csrc/ops/intra_ll/all_to_all/alltoall_buffer.cpp`
  - 在 `dispatch_basic(...)` 中让 `batch_size` 与 `x` 的 shape 校验真正受 `is_transpose` 影响
  - masked path 下直接校验 `mask.shape == [world_size, batch_size]`
  - 区分：
    - 非 transpose masked：`total_rows == batch_size`
    - transpose masked：`total_rows == world_size * batch_size`
- `DLSlime_hao_0403/csrc/ops/intra_ll/all_to_all/all_to_all_intra_ll.cu`
  - 给 `intranode_alltoall(...)` 和 `intranode_alltoall_kernel(...)` 增加 `is_transpose` 形参
  - 在 masked path 下也像旧版一样切换 source base 的解释方式，而不是固定读取同一块 `src_base`
  - 读取 mask 时直接按 `mask[dst_rank, slot]` 解释
- `DLSlime_hao_0403/csrc/python/bind.cpp`
  - Python 层接口保持不变，只更新 masked path 的语义与说明
- `bench/python/intra_alltoall_test.py`
  - 增加两组 correctness case：
    - masked + non-transpose + `mask[target_rank, slot]`
    - masked + transpose + `mask[target_rank, slot]`
- `microbench-linbinbin/dlslime_intranode_alltoall_bench.py`
  - 单列 `q` 与 `res/lse` 两类输入布局，不再只跑一个 `is_transpose=True` 的无 mask benchmark
  - 额外对比：
    - adapter 内 `transpose().contiguous()` 的临时兼容方案
    - 原生 `mask[target_rank, slot]` 直通 API

推荐验收口径：

- 新 Basic 在 API 层面具备和旧版一致的“双输入语义”：
  - `q` 走 non-transpose
  - `res/lse` 走 transpose
- 新 Basic 的 masked path 直接支持：
  - `mask[target_rank, slot]`
- NanoDeploy 服务态 adapter 不需要再为 `mask` 做 `transpose().contiguous()`
- `attention.py` 现有 `all_to_all_ll(..., is_transpose=...)` 调用语义保持不变

## 5.5 NanoDeploy 中的 CUDAGraph 设计

### 设计结论

`hao_basic` 在 NanoDeploy 中应采用：

- **启动时选后端**
- **启动时完成 buffer 构造与 full-mesh 互联**
- **启动后按既有流程 capture CUDAGraph**

而不是：

- 在图内做句柄交换
- 在图内切换 backend
- 在 replay 前后重建 `SPContext`

### 现有 Graph 路径的真实行为

当前 `NanoDeploy` 的 graph capture 发生在：

- `nanodeploy/worker/model_runner.py:capture_cudagraph`

它的关键特征是：

1. 先在 worker 启动阶段创建好 `SPContext`
2. 再分配 graph placeholder tensor
3. 通过 `set_context(...)` 把 placeholder 绑定到 attention 路径
4. 直接 capture `self.model(...)`

因此对新 backend 的要求不是“支持热切换”，而是：

- `get_sp_context().q_buffer/res_buffer/lse_buffer` 在 capture 前就已经稳定存在
- 这些 buffer 的本地显存地址和 IPC 互联关系在整个 worker 生命周期内保持不变
- `all_to_all_ll(...)` 在 capture/replay 期间只做 CUDA graph-safe 的事情

### 对 `hao_basic` 的约束

`hao_basic` 若要跑进 NanoDeploy 的 CUDAGraph，adapter 的热路径必须满足：

1. **不能在 `all_to_all_ll(...)` 内做任何分布式对象通信**
   - `dist.all_gather_object(...)`
   - `connect_full_mesh(...)`
   - 任意 Python 侧句柄交换
   - 这些都必须在 `SPContext` 初始化阶段完成
2. **不能在 `all_to_all_ll(...)` 内重新分配或重建 backend 对象**
   - `AllToAllBuffer` 必须在 `SPContext` 里一次性构造
   - graph capture/replay 期间只能复用同一实例
3. **必须复用稳定的 `local_buffer` 地址**
   - `attention.py` 直接把 `local_buffer` 当作 staging 区使用
   - graph replay 依赖这些地址稳定
4. **不能把 backend 选择放进 layer 内分支**
   - graph capture 后，执行路径应该是固定的
   - backend 选择只能发生在 worker 初始化时

### 是否需要在 NanoDeploy 图里显式调用 `reset_semaphore()`

经代码核对，当前答案是：

- **不需要作为 NanoDeploy 主方案的一部分**

依据：

- `DLSlime_hao_0403/csrc/ops/intra_ll/all_to_all/all_to_all_intra_ll.cu`
  - `intranode_alltoall(...)` 内部已经执行：
    - `cudaMemsetAsync(device_semaphore_ptr, 0, world_size * sizeof(int), stream);`
- 也就是说，`AllToAllBuffer.all_to_all(...)` 本身就已经把 semaphore reset 放在 launch 路径里

这意味着：

- NanoDeploy adapter 在 graph 路径里只需要调用 `all_to_all(...)`
- 不需要额外在 `attention.py` 前后插入 `reset_semaphore()`
- `reset_semaphore()` 保留为 standalone microbench / debug 能力即可

补充说明：

- 灏哥分支的 benchmark 里虽然显式 capture 了 `reset_semaphore() + all_to_all()`，但对 NanoDeploy 而言不是必须前提
- 如果后续 `DLSlime` 在 `all_to_all(...)` 内移除了这段 `cudaMemsetAsync`，那时再把 `reset_semaphore()` 升级为 adapter 必需接口

### Adapter 内做 mask 转换会不会成为服务瓶颈

结论：

- **它大概率不是第一优先级的带宽瓶颈**
- **但它是推理服务场景里不该长期保留的固定热路径开销**

原因：

1. 当前 `NanoDeploy` 的 `q_mask` / `res_lse_mask` 都是按 `mask[target_rank, slot]` 生产的
   - shape 来自 `context_lens/global_context_lens`
   - 在 graph 路径里会先拷进固定 placeholder，再进入各层 attention
2. 如果 adapter 在 `all_to_all_ll(...)` 里做 `mask.transpose(0, 1).contiguous()`
   - `q_buffer` 会做 1 次
   - `res_buffer` 会做 1 次
   - `lse_buffer` 还会对同一张 `res_lse_mask` 再做 1 次
   - 也就是每层 decode 至少多 3 次 mask materialization
3. 以当前 graph 常见上限 `world_size=8`、`max_bs=512`、`int32 mask` 估算
   - 单张 mask 大小约 `8 * 512 * 4 = 16KB`
   - 每层额外复制约 `3 * 16KB = 48KB`
   - 对 60+ 层模型而言，每 token 会多出数 MB 级别的纯 mask copy
4. 更关键的是它是**固定成本**
   - 即使真实活跃 batch 很小，graph placeholder 仍是 max-shaped tensor
   - 小 batch decode 时，这种固定开销更容易体现在首 token / 单 token latency 上
5. 与 `q/res` 本体的 A2A 数据量相比，mask copy 的字节量不算大
   - 因此它未必压过真正的数据搬运
   - 但它会额外增加 graph node / kernel launch，并且 `res` / `lse` 还会重复做同一件事

因此建议把它定性为：

- **bring-up 阶段可接受**
- **服务态需要消除**

### 服务态建议：直接让新版 DLSlime 支持旧 mask 布局

推荐把长期方案放在 `DLSlime` Step 1，而不是继续把 `mask` 转换留在 NanoDeploy adapter。

具体设计：

1. 新版 `AllToAllBuffer.all_to_all(...)` 保持接口不变，但 masked path 直接接受 `mask[target_rank, slot]`
2. `dispatch_basic(...)` 直接按 `mask[target_rank, slot]` 解释路由语义
3. kernel 端直接按 `mask[dst_rank, slot]` 索引
   - 不再要求 NanoDeploy 先物化一份转置后的 contiguous mask
4. `is_transpose` 只负责输入 `x` 的布局解释
   - `q` 路径：`is_transpose=False`
   - `res/lse` 路径：`is_transpose=True`
5. `HaoAllToAllBufferAdapter` 最终只负责接口名兼容
   - 不再承担每次调用的 mask 重排

迁移策略建议收敛为两阶段：

1. **本地 bring-up**
   - adapter 内允许临时 `transpose().contiguous()`
   - 仅用于尽快验证 eager + CUDAGraph 功能正确
2. **服务态收口**
   - 在 `DLSlime` 中补齐 masked path 对 `mask[target_rank, slot]` 的原生支持
   - NanoDeploy 去掉 adapter 内的 mask materialization
   - graph trace 中不再出现额外的 mask transpose/copy 节点

### NanoDeploy 侧推荐实现

推荐把 graph 相关设计收敛成下面几条：

1. 在 `nanodeploy/config.py` 新增：
   - `sp_backend: Literal["legacy_ll", "hao_basic"] = "legacy_ll"`
2. 在 `ModelRunner.__init__` 中读取 `config.sp_backend`
3. 在 `set_sp_context(...)` 时把 backend 选择一并传入
4. `SPContext.__post_init__` 内：
   - 根据 backend 选择 factory
   - 构造 `q_buffer` / `res_buffer` / `lse_buffer`
   - 完成 full-mesh 互联
5. `capture_cudagraph()` 保持现有结构不变
   - graph capture 前不再做任何 backend 特判
6. `attention.py` 保持现状
   - 仍然只调用 `local_buffer`
   - 仍然只调用 `all_to_all_ll(..., mask=..., is_transpose=...)`

### 与当前代码直接相关的一个修正项

当前 `ModelRunner.__init__` 调 `set_sp_context(...)` 时，`rank` / `sp_size` 的位置参数顺序是反的。

当前代码实际没有立即出错，是因为 `SPContext.__post_init__` 又从 `get_dist_context()` 里重新拿了一次 rank/world_size；但这不应继续依赖。

建议在 Step 2 一并修正为：

- 改成关键字参数调用
- 或把 `set_sp_context(...)` 改成 keyword-only API

推荐做法：

```python
set_sp_context(
    max_num_seqs=config.max_num_seqs,
    head_size=max_head_dim,
    num_attention_heads=hf_config.num_attention_heads,
    dtype=torch.get_default_dtype(),
    rank=sp_rank,
    sp_size=sp_size,
    backend=config.sp_backend,
)
```

### CUDAGraph 验收口径

对 NanoDeploy 中的 `hao_basic`，graph 侧至少要验：

1. eager correctness 先通过
2. 同一 backend 下 capture 与 replay 的结果一致
3. `legacy_ll` 与 `hao_basic` 在 graph replay 下结果一致
4. replay 多轮不 hang、不 silent corruption
5. graph replay 性能对比单独记录

建议单独写新的测试脚本，而不是继续堆在现有 exploratory 脚本上：

- `tests/test_mla_sp_backend_correctness.py`
- `tests/test_mla_sp_backend_cudagraph.py`

## 6. 两步迁移计划

迁移主线按两个仓库串行推进：

1. 先修改 `DLSlime` 仓库 `/mnt/nvme1n1/ml_research/linbinbin1/DLSlime`
2. 再修改 `NanoDeploy` 仓库 `/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April`

### 准备项

这部分不算正式迁移步骤，只用于冻结基线，避免后续对比失真。

动作：

1. 记录当前 MLA SP 路径的输入输出形状与关键张量约束
2. 明确三类通信张量：
   - `q`: `num_heads * head_dim`
   - `res`: `num_heads * v_head_dim`
   - `lse`: `num_heads * 1`
3. 固化老后端下的数值基线和时延基线

交付物：

- 一组 MLA 真实 batch 的老后端基线数据

开发期约束：

- `DLSlime` 的安装与切换采用手工方式
- 本文不展开 wheel / 发布 / 自动部署
- 本文只要求本地开发环境里能显式安装并验证新旧后端

## Step 1: 修改 DLSlime 仓库

目标：

- 在 `DLSlime` 仓库内完成新 `AllToAllBuffer` 的迁移，并保证只接 `KernelImpl.Basic`
- 保持旧 `AllToAllIntraLLBuffer` 完整可用，为 NanoDeploy 的后续切换提供稳定依赖

修改范围：

1. 从 `DLSlime_hao_0403/csrc/ops/intra_ll/all_to_all` 迁入：
   - `alltoall_buffer.cpp`
   - 对应头文件 `alltoall_buffer.h`
2. 更新旧仓库中的编译与绑定：
   - `DLSlime/csrc/dlslime/ops/CMakeLists.txt`
   - `DLSlime/csrc/python/bind.cpp`
   - `DLSlime/dlslime/__init__.py`
3. 保持以下旧实现原样可用：
   - `DLSlime/dlslime/buffer/intra/all_to_all_intra_ll_buffer.py`
   - `DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/all_to_all_intra_ll_buffer.*`

Step 1 内必须解决的兼容问题：

1. 为新 `AllToAllBuffer` 暴露 `local_buffer`
   - NanoDeploy 的 `q_buffer` / `res_buffer` / `lse_buffer` 都依赖这个能力
2. 直接修改新版 `KernelImpl.Basic`，让 masked path 真正消费 `is_transpose`
   - `q_buffer` 对应 `is_transpose=False`，输入 `x=[batch, msg]`
   - `res/lse` 对应 `is_transpose=True`，输入 `x=[world_size * batch, msg]`
   - 目标是让新版具备和旧版一致的两种输入解释，而不是把 `res/lse` 重排逻辑堆到 NanoDeploy
3. 固化 `mask` 适配策略
   - 旧接口：`[target_rank, slot]`
   - 新 Basic masked path 直接收敛到这一种布局
   - adapter 内 `transpose().contiguous()` 只允许作为 bring-up 临时方案
4. 明确本次迁移不引入其他新 kernel 分支

设计原则：

- 新增，不替换
- 先把新实现迁入 `DLSlime`
- 不覆盖老 wrapper
- 不改变旧类名和旧行为

Step 1 验收标准：

1. `DLSlime` 编译后可同时 import：
   - `AllToAllIntraLLBuffer`
   - `AllToAllBuffer`
   - `KernelImpl`
2. `AllToAllBuffer` 的 MLA 所需最小能力已经齐备：
   - `local_buffer`
   - `connect_full_mesh(...)`
   - `all_to_all(..., impl=KernelImpl.Basic, ...)`
   - masked path 下 `is_transpose=False/True` 都已生效
   - masked path 默认直接接受 `mask[target_rank, slot]`
3. 老接口回归不受影响
4. 新接口至少完成 MLA 相关 shape 的正确性验证：
   - `q`: `128 * 576`
   - `res`: `128 * 512`
   - `lse`: `128 * 1`
5. 新增专门的 transpose correctness 验证：
   - masked non-transpose 对齐 `q` 路径
   - masked transpose 对齐 `res/lse` 路径
6. 新增当前 mask 布局正确性验证：
   - `mask[target_rank, slot]` 在 `q` / `res` / `lse` 路径上都正确生效
   - NanoDeploy 路径不需要额外 mask transpose

## Step 2: 修改 NanoDeploy 仓库

目标：

- 在 `NanoDeploy` 仓库中接入 `DLSlime` 的新 basic 后端
- 保持 `attention.py` 的调用语义不变，只在 buffer 构造层完成切换
- 后端选择限定为**启动时配置选择**，不做运行中热切换

修改范围：

1. 在 `NanoDeploy` 中增加 MLA 专用 buffer 抽象、adapter 与 backend factory
2. 在 `nanodeploy/config.py` 中增加启动时 backend 选择项：
   - `sp_backend="legacy_ll" | "hao_basic"`
3. 在 `sp_context.py` 中根据配置选择：
   - `legacy_ll`
   - `hao_basic`
4. 让 `q_buffer` / `res_buffer` / `lse_buffer` 都走统一 protocol
5. 修正 `set_sp_context(...)` 的 `rank/sp_size` 参数传递方式
6. 默认仍保留 `legacy_ll`

建议做法：

1. `sp_context.py` 从“直接 new 旧 buffer”改成“根据配置构造 factory + adapter”
2. 切换逻辑只放在 `ModelRunner.__init__` 与 `sp_context.py`
3. `capture_cudagraph()` 不单独区分 backend，只复用初始化好的 `SPContext`
4. `attention.py` 保持现有调用方式不变
5. kernel 选择不要散落到各个 layer
6. dev 第一阶段允许 adapter 内做 `mask.transpose(...).contiguous()`，先保证功能和 graph 跑通
7. 服务态以 `DLSlime` 原生支持 `mask[target_rank, slot]` 为收口目标，不把 `mask` 预计算前移到 `ModelRunner` 作为默认方案

Step 2 验收标准：

1. NanoDeploy 可以在**启动时**选择：
   - `legacy_ll`
   - `hao_basic`
2. 同一批 MLA 请求下完成 eager 结果对比：
   - `q_buffer` 输出
   - `res/lse` combine 前输出
   - MLA 最终输出
   - 最终 logits
3. 在 CUDAGraph 场景下重复验证：
   - capture 成功
   - replay 多轮稳定
   - `legacy_ll` 与 `hao_basic` 结果一致
4. 完成性能对比：
   - 单独 A2A microbenchmark
   - 模型端 decode eager 延迟
   - 模型端 decode graph replay 延迟
   - 不同 batch size 下 `legacy_ll` 与 `hao_basic` 的收益曲线
5. 新增独立测试脚本：
   - `tests/test_mla_sp_backend_correctness.py`
   - `tests/test_mla_sp_backend_cudagraph.py`

最终交付物：

- 一份 `DLSlime` 迁移完成的 basic 后端
- 一份 `NanoDeploy` 集成完成的后端切换能力
- 一份 correctness + latency 对比结论

## 7. 决策门

## Gate 1: 是否允许在 DLSlime 内共存

判定：

- **允许**

原因：

- 编译和绑定层都可以增量接入
- 新旧类名不同，没有天然符号冲突风险
- 旧接口可以原样保留

## Gate 2: 是否能在 NanoDeploy 中无侵入切换

判定：

- **可以，但这里的“切换”仅限启动时选择 backend，前提是先补一个兼容层**

关键前提：

- 新后端能提供 `local_buffer`
- `q` 路径能映射到新 Basic 的 masked `x=[batch, msg]` 语义
- 新 Basic 已经在 kernel 内真正支持 `is_transpose=True` 的 masked path
- `mask` 的 `[target_rank, slot] <-> [slot, target_rank]` 差异已在 adapter 内收敛
- backend 选择发生在 worker 初始化阶段，而不是 graph capture/replay 期间

## Gate 3: 是否建议直接替换旧实现

判定：

- **不建议**

原因：

- 风险集中在接口语义不对齐，不是 build 问题
- 直接替换会把验证、回退、灰度都变难

## 8. 主要风险

1. `AllToAllBuffer` 当前未直接暴露 `local_buffer`
2. 如果 Step 1 没有把新版 `KernelImpl.Basic` 的 transpose 语义补齐，`res/lse` 路径就只能依赖上层重排，后续维护成本会更高
3. `mask` 不是“可能不一致”，而是新版当前约定与 NanoDeploy 当前 `mask[target_rank, slot]` 约定确定不同；若适配点搞错会直接造成跨 rank 数据错位
4. 如果把灏哥分支里的旧 wrapper 整体覆盖到旧仓库，可能会误伤原有接口语义
5. 当前 `ModelRunner -> set_sp_context(...)` 的 `rank/sp_size` 位置参数顺序是反的；若 Step 2 不顺手修正，后续 adapter/factory 落地时很容易把这个隐患变成真实 bug
6. dev 第一阶段若在 adapter 内做 `mask.transpose(...).contiguous()`，功能上可行，但会把固定 mask materialization 留在每层热路径；服务态应通过 `DLSlime` 原生支持 `mask[target_rank, slot]` 来消除这段开销

## 9. 建议的实施顺序

1. 先冻结 MLA 老后端基线
2. 再修改 `DLSlime` 仓库，把 `AllToAllBuffer` basic 版迁入，并先补齐 masked path 的 transpose 语义
3. 确认 `DLSlime` 新旧接口可共存后，再修改 `NanoDeploy` 仓库
4. 在 `NanoDeploy/sp_context.py` 接入 `hao_basic` 切换
5. 最后做 MLA correctness 和 latency 对比

## 10. 最终建议

如果目标是“让 NanoDeploy MLA 模型同时接两套 SP All2All 并可切换”，最稳妥路线不是替换旧算子，而是：

- 在 `DLSlime` 里 **共存**
- 在 `NanoDeploy` 里 **抽象**
- 在 `SPContext` 里 **切换**

这条路线的优点是：

- 回退简单
- 验证边界清晰
- 可以先 correctness，后 performance
- 不会把 MLA 接入问题扩散到整个 attention 层
- 迁移边界明确，先 `DLSlime` 后 `NanoDeploy`

---

当前判断：

- `DLSlime` 侧共存：**可行**
- `NanoDeploy` 侧双接入并切换：**可行**
- 直接拿 `AllToAllBuffer` 替换旧 buffer：**不可取**
- 对 MLA/dpsk 只做 `hao_basic`：**推荐**
