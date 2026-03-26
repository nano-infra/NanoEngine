# DLSlime Walkthrough

> 从代码结构到完整运行：RDMA 通信库的使用流程、调试方法和关键路径

## 1. 项目概览

DLSlime 是一个高性能 RDMA 通信库，提供：

- **One-sided IO**：`read` / `write` / `write_with_imm` / `imm_recv`（RDMA READ/WRITE）
- **Two-sided Messaging**：`send` / `recv`（带元数据协商的 RDMA WRITE）
- **多后端**：RDMA (RoCE/IB)、NVLink P2P、Ascend Direct

Python 通过 pybind11 扩展模块 `_slime_c` 直接访问 C++ 核心。

## 2. 目录结构

```
DLSlime/
├── CMakeLists.txt              # 根构建：选项开关、子目录
├── pyproject.toml              # scikit-build-core wheel 打包
├── cmake/                      # utils.cmake, torch.cmake, Config.cmake.in
├── dlslime/
│   ├── __init__.py             # re-export _slime_c; optional PeerAgent
│   ├── peer_agent.py           # 控制面：Redis Streams + 连接引导
│   ├── _slime_c.pyi            # 类型 stub
│   └── csrc/                   # C++ 核心
│       ├── python/bind.cpp     # pybind11 主绑定
│       ├── torch/              # PyTorch c10d 后端 (_slime_torch)
│       ├── device/             # DeviceSignal 抽象 (CUDA / Host)
│       ├── engine/
│           ├── assignment.h/cpp    # Assignment 地址切片
│           ├── rdma/               # ← 核心 RDMA 实现 (25 files)
│           ├── nvlink/             # NVLink P2P 后端
│           └── ascend_direct/      # 昇腾直通后端
├── bench/python/               # 性能基准测试
├── examples/                   # 使用示例
├── tests/                      # 单元测试
└── docs/                       # 设计文档、迁移指南
```

## 3. 快速上手：IO Benchmark

### 3.1 构建

```bash
cd DLSlime
pip install -e . -v   # scikit-build 自动 -DBUILD_PYTHON=ON -DBUILD_RDMA=ON
```

### 3.2 One-Sided IO (write_with_imm)

```bash
cd bench/python
python endpoint_io_bench.py
```

**代码流程** (`endpoint_io_bench.py`)：

```python
from dlslime._slime_c import RDMAEndpoint, available_nic

# 1. 发现网卡
dev = available_nic()[0]   # e.g. "mlx5_0"

# 2. 创建 Endpoint（自动分配 Context, MemoryPool, Worker）
ep1 = RDMAEndpoint(dev, 1, "RoCE", num_qp=1)
ep2 = RDMAEndpoint(dev, 1, "RoCE", num_qp=1)

# 3. 注册本地 MR（返回 int32 handle）
local_handle = ep1.register_memory_region("buf_a", data_ptr, size)

# 4. 注册远端 MR（从对端的 endpoint_info 中获取）
ep2_info = ep2.endpoint_info()
remote_handle = ep1.register_remote_memory_region("buf_b", ep2_info["mr_info"]["buf_b"])

# 5. 连接 QP
ep1.connect(ep2.endpoint_info())
ep2.connect(ep1.endpoint_info())

# 6. 发起 RDMA WRITE_WITH_IMM
future = ep1.write_with_imm(
    [(local_handle, remote_handle, src_offset, dst_offset, length)],
    imm_data=42
)

# 7. 对端 imm_recv 等待通知
recv_future = ep2.imm_recv()
recv_future.wait()
print(recv_future.imm_data())  # 42

future.wait()
```

### 3.3 Two-Sided Send/Recv

```bash
python endpoint_sendrecv_bench.py --device cpu --qp 1 --iters 128
```

**代码流程** (`endpoint_sendrecv_bench.py`)：

```python
# 1-2. 同上，创建两个 Endpoint 并 connect

# 3. Send: 直接传 (data_ptr, offset, length) 三元组
send_future = send_ep.send(
    (tensor.data_ptr(), tensor.storage_offset(), tensor.numel()),
    stream_handle   # CUDA stream or None
)

# 4. Recv: 对端同样传接收 buffer 的三元组
recv_future = recv_ep.recv(
    (recv_buf.data_ptr(), recv_buf.storage_offset(), recv_buf.numel()),
    stream_handle
)

# 5. 等待完成
recv_future.wait()
send_future.wait()
```

> **关键区别**：IO 操作需要手动注册 MR 并获得 handle；Send/Recv 内部自动注册 MR。

## 4. 连接建立流程 (Walkthrough)

```
   Endpoint A                              Endpoint B
   ──────────                              ──────────
   1. RDMAEndpoint(dev, port, "RoCE", nqp)
      → init():
        • 创建 local_pool_ (PD)
        • 创建 meta_pool_ (借用 PD)
        • 创建 remote_pool_
        • 创建 io_data_channel_ (nqp 个 QP)
        • 创建 meta_channel_ (1 QP)
        • 创建 msg_data_channel_ (nqp 个 QP)
        • 分配 context 池 (Send/Recv/RW/ImmRecv)
        • 注册系统 MR (dummy, ctx pool) 到 meta_pool_

   2. A.endpoint_info() ─── JSON 交换 ──→ B.connect(info_A)
      B.endpoint_info() ←── JSON 交换 ─── A.connect(info_B)

      connect() 内部：
        • io_data_channel_→connect(QP info)       ← RC QP 握手
        • meta_channel_→connect(QP info)
        • msg_data_channel_→connect(QP info)
        • registerRemoteMemoryRegion(meta base)   ← 远端 SendCtxPool 地址
        • pre-post RECV (meta + data channels)    ← 预投递接收请求
        • worker_→addEndpoint(this)               ← 注册到 Worker 轮询

   3. connected_ = true → Worker 线程开始调用 process()
```

## 5. Send/Recv 数据流 (Walkthrough)

```
   Sender                          Worker Thread                    Receiver
   ──────                          ─────────────                    ────────
   send(ptr, off, len)
   → registerMemoryRegion
   → 填充 SendContext
   → enqueue(send_ring)
                                   sendProcess():
                                   dequeue(send_ring)
                                   等待 GPU ready
                                   等待 meta_arrived_flag ──────── recv(ptr, off, len)
                                                                   → registerMemoryRegion
                                                                   → 填充 RecvContext (含 rkey)
                                                                   → enqueue(recv_ring)

                                                                   recvProcess():
                                                                   dequeue(recv_ring)
                                                                   等待 GPU ready
                                                                   RDMA WRITE meta_info_t
                                                                   到 sender 的 SendContext
                                                                   ─────────────────────────→

                                   meta_arrived!
                                   registerRemoteMemoryRegion
                                   按 QP 数拆分 chunk
                                   RDMA WRITE_WITH_IMM 数据
                                   ─────────────────────────────→ data_recv callback
                                                                  signal→set_comm_done(qpi)

                                   send callback:
                                   signal→set_comm_done(qpi)

   send_future.wait()                                              recv_future.wait()
```

## 6. Worker 线程模型

```python
# Worker 线程伪代码
while running:
    for ep in registered_endpoints:
        if ep.connected:
            work = ep.readWriteProcess()   # IO 读写进度
            work += ep.immRecvProcess()     # IO 接收进度（pre-post window）
            work += ep.sendProcess()        # MSG 发送状态机
            work += ep.recvProcess()        # MSG 接收状态机
```

- 每个 NUMA 节点一个默认 Worker（`GlobalWorkerManager`）
- Worker 绑定 CPU 核心（NUMA-aware）
- `connect()` 时自动注册到 Worker

## 7. 调试技巧

### 7.1 环境变量

| 变量                         | 说明                             |
| ---------------------------- | -------------------------------- |
| `SLIME_LOG_LEVEL`            | 日志级别 (DEBUG/INFO/WARN/ERROR) |
| `SLIME_QP_NUM`               | QP 数量 (\< 64)                  |
| `SLIME_MAX_SEND_WR`          | 每 QP 最大发送 WR 深度           |
| `SLIME_GID_INDEX`            | RoCE GID 索引                    |
| `SLIME_BYPASS_DEVICE_SIGNAL` | 跳过 GPU signal 同步             |

### 7.2 常见问题排查

**"remote access error, Vendor Err: 136"**

- 原因：MR 长度不足。PyTorch allocator 复用地址时，旧 MR 长度覆盖不了新 buffer。
- 检查：`send()` / `recv()` 是否正确调用 `registerMemoryRegion`（内部有长度校验和自动重注册）。

**"Data Recv Failed during pre-post"**

- 通常是 teardown 时 QP flush 导致的正常现象（`IBV_WC_WR_FLUSH_ERR`），非数据错误。
- 已降级为 DEBUG 日志。

**segmentation fault in sendProcess/recvProcess**

- 检查 Assignment 中是否误用了原始指针代替 handle。
- `post_rc_oneside_batch` 使用 `get_mr_fast(int32_t handle)`，传入原始指针会越界。

### 7.3 Loopback 测试

所有 benchmark 支持单机回环测试（两个 Endpoint 在同一进程），适合开发调试：

```bash
python endpoint_io_bench.py         # IO 回环
python endpoint_sendrecv_bench.py   # Send/Recv 回环
```

## 8. Python API 速查

```python
from dlslime._slime_c import (
    available_nic,       # -> list[str]        发现可用网卡
    socket_id,           # (dev) -> int        网卡所在 NUMA node
    RDMAContext,         # ibverbs context 封装
    RDMAMemoryPool,      # PD + MR 管理
    RDMAEndpoint,        # 统一通信端点
    RDMAWorker,          # 后台 Worker 线程
)

# Endpoint 构造（3 种方式）
ep = RDMAEndpoint("mlx5_0", 1, "RoCE", num_qp=1)       # 从设备名
ep = RDMAEndpoint(context, num_qp=1)                     # 从 RDMAContext
ep = RDMAEndpoint(pool, num_qp=1)                        # 从 RDMAMemoryPool

# MR 注册
handle = ep.register_memory_region("name", data_ptr, length)      # 本地 MR
handle = ep.register_remote_memory_region("name", mr_info_json)   # 远端 MR

# 连接
info = ep.endpoint_info()              # -> dict (JSON)
ep.connect(remote_info)                # RC QP 建连

# 通信
future = ep.write_with_imm([(lh, rh, soff, doff, len)], imm=0)   # IO
future = ep.send((ptr, offset, length), stream)                    # MSG
future.wait()
```
