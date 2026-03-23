# DLSlime 实现文档

> C++ RDMA 引擎内部架构、对象模型、内存管理、数据面详解

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│  Python Layer                                                   │
│  dlslime/__init__.py  →  _slime_c (pybind11)  →  _slime_torch  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  RDMAEndpoint (统一端点)                                        │
│  ┌──────────────┐ ┌─────────────┐ ┌───────────────────────┐    │
│  │ IO 数据面    │ │ MSG 数据面   │ │ 内存管理              │    │
│  │ io_data_ch   │ │ meta_ch     │ │ local_pool_ (user MR) │    │
│  │ rw_ctx_pool  │ │ msg_data_ch │ │ meta_pool_  (sys MR)  │    │
│  │ imm_recv_pool│ │ send/recv   │ │ remote_pool_ (peer)   │    │
│  └──────┬───────┘ │ ctx_pool    │ └───────────────────────┘    │
│         │         └──────┬──────┘                               │
│         └───────┬────────┘                                      │
│                 ▼                                                │
│         RDMAChannel (QP 抽象)                                   │
│         post_send/recv/oneside_batch                            │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│  RDMAContext (per-device ibverbs context)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐                    │
│  │ ib_ctx_  │ │ CQ       │ │ Future Thread│                    │
│  │ pd_      │ │ (shared) │ │ (CQ 轮询)    │                    │
│  └──────────┘ └──────────┘ └──────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
                  │
                  ▼
            libibverbs / RDMA HCA
```

## 2. 核心对象模型

### 2.1 RDMAContext (`rdma_context.h/cpp`)

每个 IB 设备一个实例，由 `GlobalContextManager` (Meyer's Singleton) 缓存。

| 成员             | 说明                               |
| ---------------- | ---------------------------------- |
| `ib_ctx_`        | `ibv_context*`，ibverbs 设备上下文 |
| `cq_`            | `ibv_cq*`，共享完成队列            |
| `device_name_`   | 设备名 (e.g. `"mlx5_0"`)           |
| `future_thread_` | 后台线程，轮询 CQ 并触发回调       |

```cpp
auto ctx = GlobalContextManager::instance().get("mlx5_0", 1, "RoCE");
ctx->launch_future();  // 启动 CQ 轮询线程
```

### 2.2 RDMAMemoryPool (`memory_pool.h/cpp`)

管理本地 MR 注册，持有 Protection Domain (PD)。

**两级存储**：

| 存储              | 数据结构                                                                         | 用途                                                 |
| ----------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Handle-based (新) | `vector<ibv_mr*> id_to_mr_` + `unordered_map<uintptr_t, int32_t> ptr_to_handle_` | 快速路径 `get_mr_fast(handle)` — O(1) 数组访问，无锁 |
| Name-based        | `unordered_map<string, int32_t> name_to_id_`                                     | Python 层按名字查找 handle                           |
| Legacy (兼容)     | `unordered_map<uintptr_t, ibv_mr*> mrs_`                                         | `get_mr(ptr)` 回退路径                               |

**PD 共享机制**：

```cpp
// 主 pool：拥有 PD
local_pool_ = make_shared<RDMAMemoryPool>(ctx);          // owns_pd_ = true

// Meta pool：借用 PD（同一 PD 下的 MR 可用于同一 QP）
meta_pool_ = make_shared<RDMAMemoryPool>(local_pool_);   // owns_pd_ = false
```

**MR 注册与长度校验**：

```cpp
int32_t registerMemoryRegion(uintptr_t data_ptr, uint64_t length, optional<string> name)
{
    lock(name_mutex_);

    if (ptr_to_handle_.count(data_ptr)) {
        int32_t handle = ptr_to_handle_[data_ptr];
        ibv_mr* existing = id_to_mr_[handle];

        if (existing->length >= length)
            return handle;  // 快速路径：已有 MR 足够大

        // 地址被 allocator 复用且新 buffer 更大 → 重注册
        ibv_dereg_mr(existing);
        ibv_mr* mr = ibv_reg_mr(pd_, (void*)data_ptr, length, access);
        id_to_mr_[handle] = mr;  // 同一 handle，新 MR
        return handle;
    }

    // 全新注册
    ibv_mr* mr = ibv_reg_mr(pd_, (void*)data_ptr, length, access);
    int32_t handle = id_to_mr_.size();
    id_to_mr_.push_back(mr);
    ptr_to_handle_[data_ptr] = handle;
    return handle;
}
```

> **重要设计决策**：`send()` / `recv()` 每次都调用 `registerMemoryRegion` 而非先
> `get_mr_handle` + 条件跳过。这确保 PyTorch allocator 地址复用时 MR 长度始终正确。

### 2.3 RDMARemoteMemoryPool (`remote_memory_pool.h`)

存储远端 MR 元数据（addr, length, rkey），纯内存结构，不涉及 ibverbs 调用。

| 接口                                          | 说明                       |
| --------------------------------------------- | -------------------------- |
| `registerRemoteMemoryRegion(name, json)`      | 按名注册，幂等             |
| `registerRemoteMemoryRegion(addr, len, rkey)` | 匿名注册（sendProcess 用） |
| `get_remote_mr_fast(handle)`                  | O(1) vector 访问           |

### 2.4 RDMAChannel (`rdma_channel.h/cpp`)

对 QP 的封装，一个 Channel 管理 N 个 QP。

```
RDMAEndpoint 拥有 3 个 Channel：
├── io_data_channel_   (num_qp 个 QP) — IO read/write
├── meta_channel_      (1 个 QP)      — MSG 元数据交换
└── msg_data_channel_  (num_qp 个 QP) — MSG 数据传输
```

**核心 posting 方法**：

```cpp
// One-sided (RDMA READ/WRITE) — 仅 handle 快速路径
int64_t post_rc_oneside_batch(int qpi, RDMAAssign* assign,
                               shared_ptr<RDMAMemoryPool> local_pool)
{
    for (auto& sub : assign->batch_) {
        ibv_mr*     mr  = local_pool->get_mr_fast(sub.mr_key);        // handle → MR
        remote_mr_t rmr = remote_pool_->get_remote_mr_fast(sub.remote_mr_key);

        sge.addr   = mr->addr + sub.source_offset;
        sge.lkey   = mr->lkey;
        wr.rdma.remote_addr = rmr.addr + sub.target_offset;
        wr.rdma.rkey        = rmr.rkey;
    }
    ibv_post_send(qp_[qpi], wr, &bad_wr);
}

// Two-sided (SEND/RECV) — handle 或 ptr 兼容
int64_t post_recv_batch(int qpi, RDMAAssign* assign,
                         shared_ptr<RDMAMemoryPool> local_pool)
{
    for (auto& sub : assign->batch_) {
        ibv_mr* mr;
        if (sub.mr_key < 1000000)  // handle（小整数）
            mr = local_pool->get_mr_fast((int32_t)sub.mr_key);
        else                       // 原始指针（兼容旧路径）
            mr = local_pool->get_mr(sub.mr_key);
        // ...
    }
    ibv_post_recv(qp_[qpi], wr, &bad_wr);
}
```

> `local_pool` 由调用方注入：系统 buffer 传 `meta_pool_`，用户数据传 `local_pool_`。

### 2.5 RDMAAssign (`rdma_assignment.h/cpp`)

Work Request 的高层封装，包含：

```cpp
struct Assignment {
    uintptr_t mr_key;          // 本地 MR handle (int32_t 范围)
    uintptr_t remote_mr_key;   // 远端 MR handle (int32_t 范围)
    uint64_t  target_offset;   // 远端偏移
    uint64_t  source_offset;   // 本地偏移
    uint64_t  length;          // 传输长度
};

class RDMAAssign {
    OpCode                    opcode_;
    vector<Assignment>        batch_;    // 批量 WR
    function<void(int, int)>  callback_; // CQ 完成回调
    int32_t                   imm_data_;
    bool                      is_inline_;
};
```

### 2.6 RDMAEndpoint (`rdma_endpoint.h/cpp`)

统一端点，合并了原版的 `RDMAIOEndpoint` + `RDMAMsgEndpoint`。

**构造方式（3 种）**：

```cpp
// 1. 从 MemoryPool（共享 PD）
RDMAEndpoint(shared_ptr<RDMAMemoryPool> pool, size_t num_qp, ...);

// 2. 从 Context（自动创建 pool）
RDMAEndpoint(shared_ptr<RDMAContext> ctx, size_t num_qp, ...);

// 3. 从设备名（自动创建 context + pool）
RDMAEndpoint(string dev_name, int32_t ib_port, string link_type, size_t num_qp, ...);
```

**内存布局**：

```
meta_pool_ 注册的系统 MR：
├── io_dummy_     (8 bytes)  — IO recv 占位
├── msg_dummy_    (8 bytes)  — MSG recv 占位
└── send_ctx_pool_ + recv_ctx_pool_ (连续分配)
    — 对端通过 RDMA WRITE 写入 SendContext.remote_meta_info_

local_pool_ 注册的用户 MR：
└── 用户 tensor buffer（send/recv/read/write 时按需注册）
```

**Context 池**：

| 池                     | 类型               | 深度                       | 用途        |
| ---------------------- | ------------------ | -------------------------- | ----------- |
| `read_write_ctx_pool_` | `ReadWriteContext` | `SLIME_MAX_IO_FIFO_DEPTH`  | IO 读写     |
| `imm_recv_ctx_pool_`   | `ImmRecvContext`   | `SLIME_MAX_IO_FIFO_DEPTH`  | IO imm 接收 |
| `send_ctx_pool_`       | `SendContext`      | `SLIME_MAX_MSG_FIFO_DEPTH` | MSG 发送    |
| `recv_ctx_pool_`       | `RecvContext`      | `SLIME_MAX_MSG_FIFO_DEPTH` | MSG 接收    |

所有 Context 通过 `posix_memalign` 64 字节对齐分配，避免 False Sharing。

### 2.7 RDMAWorker (`rdma_worker.h/cpp`)

后台轮询线程，NUMA-aware CPU 绑定。

```cpp
void RDMAWorker::run() {
    while (running_) {
        for (auto& ep : endpoints_) {
            ep->process();  // 驱动所有 Endpoint 的状态机
        }
    }
}
```

`GlobalWorkerManager`：每个 NUMA 节点一个默认 Worker。

## 3. IO 数据面实现

### 3.1 dispatchTask (生产者)

用户线程调用 `write()` / `read()` / `writeWithImm()`：

```
1. 从 rw_slot_id_ 原子递增获取 slot
2. 填充 ReadWriteContext:
   - 按 (mr_key, remote_mr_key, target_off, source_off, length) 拆分到 num_qp 个 AssignBatch
   - 大请求按 SLIME_MAX_WR_SIZE 分 chunk，最后一个 chunk 才带 WRITE_WITH_IMM
3. 绑定 DeviceSignal + CUDA stream
4. enqueue 到 read_write_buffer_ring_ (lock-free jring)
```

### 3.2 readWriteProcess (消费者 — Worker 线程)

```
1. dequeue burst 从 ring
2. 检查 GPU ready (signal)
3. 检查 token_bucket_[qpi] 有配额
4. 调用 io_data_channel_->post_rc_oneside_batch()
5. CQ 回调中 token_bucket_ 回收配额，设置 signal
```

### 3.3 immRecvProcess (pre-post 窗口)

滑动窗口预投递 RECV：

```
posted_recv_cnt_ < user_req_cnt + PRE_POST_WINDOW (32)
→ 为每个 slot 创建 ImmRecvContext
→ post_recv_batch 到 io_data_channel_
→ CQ 完成时设置 signal + imm_data
```

## 4. MSG 数据面实现

### 4.1 Send 路径

**用户线程** (`send`):

```
1. registerMemoryRegion(data_ptr, length)  // 确保 MR 足够大
2. 填充 SendContext { view_, local_meta_info_ }
3. enqueue 到 send_buffer_ring_
```

**Worker 线程** (`sendProcess`):

```
状态机:
WAIT_GPU_READY → 等 GPU signal
WAIT_META      → 等 meta_arrived_flag_
                  (被 recvProcess 的 RDMA WRITE 触发)
                  收到 remote_meta_info_ (rkey + view)
POST_DATA_SEND → registerRemoteMemoryRegion
                  get_mr_handle(local data_ptr)
                  按 num_qp 拆 chunk
                  post_rc_oneside_batch (WRITE_WITH_IMM)
                  → 完成回调: signal→set_comm_done(qpi)
```

### 4.2 Recv 路径

**用户线程** (`recv`):

```
1. registerMemoryRegion(data_ptr, length)
2. 获取 MR 的 rkey → RecvContext.local_meta_info_
3. enqueue 到 recv_buffer_ring_
```

**Worker 线程** (`recvProcess`):

```
状态机:
WAIT_GPU_BUF    → 等 GPU signal
INIT_SEND_META  → pre-post RECV (data channel, per QP)
                   计算 meta 偏移
                   RDMA WRITE meta_info_t 到 sender 的 SendContext
                   (meta_channel_, inline, 使用 meta_pool_)
                   → sender 的 meta_arrived_flag_ 被置位
```

### 4.3 Meta 交换细节

```
Receiver 的 RecvContext.local_meta_info_ 包含:
├── r_key_: 本地 recv buffer 的 rkey
└── view_:  { data_ptr, offset, length }

通过 RDMA WRITE 写入 Sender 的 SendContext.remote_meta_info_:
├── 目标地址: send_ctx_pool_[slot].remote_meta_info_
├── 本地 MR:  send_ctx_handle_ (meta_pool_ 中)
├── 远端 MR:  remote_meta_key_ (connect 时注册的 handle)
└── 偏移计算: slot * sizeof(SendContext) + send_ctx_meta_offset_
```

## 5. 线程模型与同步

```
┌──────────────┐     jring (lock-free)     ┌───────────────┐
│  User Thread │ ─── enqueue ───────────→  │  Worker Thread │
│  send/recv   │                           │  process()     │
│  read/write  │                           │  状态机推进     │
└──────────────┘                           └───────┬───────┘
                                                   │
                                            ibv_post_send/recv
                                                   │
                                                   ▼
                                           ┌───────────────┐
                                           │ CQ Poll Thread │
                                           │ (RDMAContext)   │
                                           │ future_thread_  │
                                           │ 回调 → signal   │
                                           └───────────────┘
```

**三类线程**：

| 线程           | 角色                                       | 绑定             |
| -------------- | ------------------------------------------ | ---------------- |
| User Thread    | 调用 send/recv/write/read，enqueue 到 ring | 用户控制         |
| Worker Thread  | `process()` 状态机，ibv_post_send/recv     | NUMA-aware CPU   |
| CQ Poll Thread | ibv_poll_cq，触发 RDMAAssign 回调          | RDMAContext 内部 |

**同步原语**：

| 原语                                | 用途                                   |
| ----------------------------------- | -------------------------------------- |
| `jring_t*`                          | User → Worker 的 lock-free SPSC ring   |
| `atomic<bool> meta_arrived_flag_`   | Receiver meta WRITE 通知 Sender        |
| `atomic<uint32_t> finished_qp_mask` | 多 QP 完成聚合                         |
| `DeviceSignal`                      | GPU-CPU 同步（CUDA event 或 CPU flag） |
| `atomic<int32_t> token_bucket_[]`   | 每 QP 发送 WR 配额控制                 |

## 6. 析构顺序

```cpp
~RDMAEndpoint() {
    // 1. 停止 Worker 轮询
    connected_ = false;

    // 2. 先销毁 Channel（销毁 QP，flush 未完成 WR）
    //    CQ thread 的回调仍可安全访问 context pool
    io_data_channel_.reset();
    meta_channel_.reset();
    msg_data_channel_.reset();

    // 3. 再释放 context pool 和 ring
    free(read_write_ctx_pool_);
    free(imm_recv_ctx_pool_);
    free(send_ctx_pool_);  // recv_ctx_pool_ 是同一块内存的后半段
    freeRing(...);
}
```

> 原版 DLSlime 的析构顺序相反（先 free pool 再隐式销毁 channel），在某些时序下
> 可能导致 CQ 回调访问已释放内存。NanoInfra 版本修正了此问题。

## 7. 与原版 DLSlime 的关键差异

| 维度          | 原版 DLSlime                                                      | NanoInfra 版本                                                  |
| ------------- | ----------------------------------------------------------------- | --------------------------------------------------------------- |
| Endpoint 结构 | `RDMAEndpoint` 持有 `io_endpoint_` + `msg_endpoint_` (shared_ptr) | 统一 `RDMAEndpoint`，直接内联所有逻辑                           |
| MR 查找       | `get_mr(ptr)` — mutex + hash map                                  | `get_mr_fast(handle)` — 无锁 vector 下标                        |
| MR 存储       | 单一 `mrs_` map (ptr → ibv_mr\*)                                  | 双路径: `id_to_mr_` (handle) + `ptr_to_handle_` + `name_to_id_` |
| Memory Pool   | 单一 `memory_pool_` (local + remote)                              | 三池分离: `local_pool_` + `meta_pool_` + `remote_pool_`         |
| Channel       | 不接受外部 pool 参数                                              | `post_*_batch` 接受 `local_pool` 参数，调用方决定用哪个池       |
| 性能          | ~34 GB/s 峰值 (send/recv)                                         | ~48 GB/s 峰值 (send/recv)，约 42% 提升                          |
| 析构          | 先 free pool 再隐式销毁 QP                                        | 先销毁 QP 再 free pool                                          |

## 8. pybind11 绑定 (`bind.cpp`)

### 8.1 模块结构

```cpp
PYBIND11_MODULE(_slime_c, m) {
    // Build flags
    EXPOSE_BUILD_FLAG(m, BUILD_RDMA);
    EXPOSE_BUILD_FLAG(m, BUILD_NVLINK);

    // 基础类型
    py::enum_<OpCode>(m, "OpCode");
    py::class_<Assignment>(m, "Assignment");
    py::class_<DeviceSignal>(m, "DeviceSignal");

    #ifdef BUILD_RDMA
    // RDMA 类型
    py::class_<RDMAContext>(m, "RDMAContext");
    py::class_<SendFuture>(m, "SlimeSendFuture");      // .wait()
    py::class_<RecvFuture>(m, "SlimeRecvFuture");       // .wait()
    py::class_<ReadWriteFuture>(m, "SlimeReadWriteFuture"); // .wait()
    py::class_<ImmRecvFuture>(m, "SlimeImmRecvFuture");     // .wait(), .imm_data()
    py::class_<RDMAMemoryPool>(m, "RDMAMemoryPool");
    py::class_<RDMAEndpoint>(m, "RDMAEndpoint");        // 3 种构造方式
    py::class_<RDMAWorker>(m, "RDMAWorker");
    m.def("available_nic", ...);
    m.def("socket_id", ...);
    #endif
}
```

### 8.2 GIL 管理

所有阻塞 / 长耗时操作使用 `py::call_guard<py::gil_scoped_release>()`：

- `connect`, `send`, `recv`, `read`, `write`, `write_with_imm`, `imm_recv`
- `register_memory_region`, `register_remote_memory_region`
- `future.wait()`

### 8.3 JSON 互操作

使用 `nanocommon/pybind_json` 实现 `nlohmann::json` ↔ Python `dict` 自动转换：

```python
info = ep.endpoint_info()   # C++ json → Python dict
ep.connect(remote_info)     # Python dict → C++ json
```

## 9. CMake 构建层次

```
DLSlime/CMakeLists.txt
├── cmake/utils.cmake                  # 选项定义
├── NanoCommon (external)              # JSON, pybind_json
├── third_party/hiredis                # Redis C client
└── dlslime/csrc/CMakeLists.txt
    ├── device/CMakeLists.txt          → _slime_device  (CUDA/Host)
    ├── engine/CMakeLists.txt          → _slime_engine  (Assignment)
    │   └── rdma/CMakeLists.txt        → _slime_rdma    (8 cpp, links ibverbs/numa/hiredis)
    ├── python/CMakeLists.txt          → _slime_c       (pybind11 module)
    └── torch/CMakeLists.txt           → _slime_torch   (PyTorch backend)

INTERFACE target "dlslime" 聚合: _slime_engine + _slime_device + _slime_rdma
```

### 构建选项

| CMake 选项            | 默认 | 说明              |
| --------------------- | ---- | ----------------- |
| `BUILD_RDMA`          | ON   | RDMA 后端         |
| `USE_CUDA`            | OFF  | CUDA 设备支持     |
| `BUILD_NVLINK`        | OFF  | NVLink P2P        |
| `BUILD_ASCEND_DIRECT` | OFF  | 华为昇腾          |
| `BUILD_PYTHON`        | OFF  | pybind11 模块     |
| `BUILD_TORCH_PLUGIN`  | OFF  | PyTorch c10d 后端 |

## 10. 文件索引 (engine/rdma/)

| 文件                    | 行数 | 职责                                                                     |
| ----------------------- | ---- | ------------------------------------------------------------------------ |
| `rdma_endpoint.h`       | 345  | Endpoint 定义：Context 结构体、状态机枚举、MetaInfo、RDMAEndpoint 类声明 |
| `rdma_endpoint.cpp`     | 924  | Endpoint 实现：构造/析构、connect、send/recv/read/write、process 状态机  |
| `rdma_channel.h`        | 80   | Channel 声明：QP 管理、post 接口                                         |
| `rdma_channel.cpp`      | ~350 | Channel 实现：init QP、connect、post_send/recv/oneside_batch             |
| `memory_pool.h`         | 130  | MemoryPool 声明：PD、MR 存储、get_mr_fast/get_mr                         |
| `memory_pool.cpp`       | 132  | MemoryPool 实现：registerMemoryRegion（含长度校验和重注册）              |
| `remote_memory_pool.h`  | ~90  | RemoteMemoryPool：远端 MR 元数据管理                                     |
| `rdma_context.h/cpp`    | ~200 | Context：ibverbs 初始化、CQ 创建、future 线程                            |
| `rdma_context_pool.h`   | ~50  | GlobalContextManager singleton                                           |
| `rdma_worker.h/cpp`     | ~150 | Worker 线程：NUMA 绑定、endpoint 轮询                                    |
| `rdma_worker_pool.h`    | ~50  | GlobalWorkerManager singleton                                            |
| `rdma_assignment.h/cpp` | ~100 | RDMAAssign：OpCode → ibv_wr_opcode、回调                                 |
| `rdma_config.h`         | ~80  | rdma_info_t：QP/GID/LID/PSN/MTU JSON 序列化                              |
| `rdma_common.h`         | ~40  | remote_mr_t、storage_view_t POD                                          |
| `rdma_env.h`            | ~60  | 环境变量配置                                                             |
| `rdma_utils.h`          | ~50  | NIC 发现、NUMA 工具                                                      |
| `rdma_future.h/cpp`     | ~100 | Future 封装：wait()、imm_data()                                          |
| `ibv_helper.h/cpp`      | ~200 | ibverbs 底层工具                                                         |
