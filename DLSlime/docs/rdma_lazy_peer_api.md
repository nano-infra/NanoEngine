# RDMALazyPeer API 文档

每进程一个 Broker 的自组织 P2P RDMA 封装，支持惰性建链、按 id 注册 buffer、RC Write/Read。

## 架构

- **每进程一个 Broker**：每个 peer 既是 server（自己的 broker）也是 client（连到对方 broker）。
- **惰性建链**：`connect(remote_id, remote_broker_addr)` 时按需建立 RDMA 连接，无需事先约定谁 bind、谁 connect。
- **Buffer 注册**：按 `(endpoint_id, buffer_id)` 在 broker 中存储 `mr_info`，支持跨 peer 的 `read` / `write`。

## 前置条件

- 编译时启用 `BUILD_RDMA_RENDEZVOUS_ZMQ=ON`
- 系统有 RDMA 设备（如 `rxe0`）

______________________________________________________________________

## Python API

### start_broker / Broker

```python
from dlslime import start_broker

broker = start_broker(addr="0.0.0.0:50051")
peer = broker.peer(my_id="A", device_name="", ib_port=1, link_type="RoCE")
# 使用完毕后
broker.stop()
```

启动本进程的 Broker（无 RDMA endpoint，仅做路由）。返回 `Broker`（C++ 实现；与 RDMA 无关的通用部分在 `csrc/dlslime/rendezvous`）。

- **broker.peer(my_id, ...)**：创建挂在该 broker 下的 `RDMALazyPeer`，`my_broker_addr` 自动为本 broker 的 client 地址（如 `0.0.0.0:50051` → `127.0.0.1:50051`）。
- **broker.client_addr**：本机连本 broker 用的地址，用于传给对端做 `connect(remote_id, broker.client_addr)`。
- **broker.stop()**：关闭 broker。

| 参数 | 类型 | 说明                           |
| ---- | ---- | ------------------------------ |
| addr | str  | 绑定地址，如 `"0.0.0.0:50051"` |

______________________________________________________________________

### RDMALazyPeer（也可直接构造）

```python
from dlslime import RDMALazyPeer, start_broker

# 推荐：通过 broker 创建
broker = start_broker("0.0.0.0:50051")
peer = broker.peer("A", device_name="", ib_port=1, link_type="RoCE")

# 或直接构造（需自己传 my_broker_addr）
peer = RDMALazyPeer(my_broker_addr="127.0.0.1:50051", my_id="A", device_name="", ib_port=1, link_type="RoCE")
```

| 参数           | 类型 | 说明                                                         |
| -------------- | ---- | ------------------------------------------------------------ |
| my_broker_addr | str  | 本进程 broker 的地址（客户端连入用，如 `"127.0.0.1:50051"`） |
| my_id          | str  | 本 peer 的 id，用于 handshake 和 buffer 注册                 |
| device_name    | str  | RDMA 设备名，空则用首块设备                                  |
| ib_port        | int  | IB 端口，默认 1                                              |
| link_type      | str  | 链路类型，默认 `"RoCE"`                                      |

______________________________________________________________________

### connect

```python
peer.connect(remote_id="B", remote_broker_addr="127.0.0.1:50052")
```

与 `remote_id` 建立 RDMA 连接。`remote_broker_addr` 为对方 broker 的地址。

- **对等建链**：不区分 initiator 与 target；两端逻辑相同：向对方 broker 登记自己的 endpoint_info，在本 broker 上等待对方登记，然后完成 RDMA connect。
- **幂等**：若已与该 remote 建链，再次调用 `connect` 会直接返回；谁先发起、发起几次，最终状态一致。
- **对端未建 peer 会失败**：若对端尚未创建对应的 peer 并调用 `connect`，本端会在等待对端登记时超时。

**注意**：`connect` 会释放 GIL，适合多线程并发建链。

______________________________________________________________________

### register_buffer

```python
peer.register_buffer(buffer_id="buf_a", ptr=ptr, size=16)
```

注册已有内存为 buffer，并向 broker 上报 `mr_info`。

| 参数      | 类型 | 说明                  |
| --------- | ---- | --------------------- |
| buffer_id | str  | buffer 标识           |
| ptr       | int  | 内存地址（uintptr_t） |
| size      | int  | 大小（字节）          |

______________________________________________________________________

### alloc_and_register_buffer

```python
ptr, size = peer.alloc_and_register_buffer(buffer_id="buf_a", size=16)
```

页对齐分配内存并注册。返回 `(ptr, aligned_size)`。

| 参数      | 类型 | 说明                         |
| --------- | ---- | ---------------------------- |
| buffer_id | str  | buffer 标识                  |
| size      | int  | 期望大小（字节），会按页对齐 |

______________________________________________________________________

### get_local_mr_key

```python
local_mr = peer.get_local_mr_key(remote_id="B", buffer_id="buf_a")
```

获取本地 buffer 在指定 remote 的 endpoint 上的 mr_key，用于构建 `assign`。

| 参数      | 类型 | 说明           |
| --------- | ---- | -------------- |
| remote_id | str  | 远端 peer id   |
| buffer_id | str  | 本地 buffer id |

______________________________________________________________________

### get_remote_mr_key

```python
remote_mr = peer.get_remote_mr_key(remote_id="B", buffer_id="buf_t")
```

获取远端 buffer 的 mr_key（可能阻塞直到远端注册），用于构建 `assign`。

| 参数      | 类型 | 说明           |
| --------- | ---- | -------------- |
| remote_id | str  | 远端 peer id   |
| buffer_id | str  | 远端 buffer id |

______________________________________________________________________

### read / write（与 RDMAEndpoint 一致）

```python
# assign: [(local_mr_key, remote_mr_key, target_offset, source_offset, length)]
local_mr = peer.get_local_mr_key("B", "buf_a")
remote_mr = peer.get_remote_mr_key("B", "buf_t")
future = peer.write("B", [(local_mr, remote_mr, 0, 0, 8)], None)
future.wait()
```

`read` / `write` 接口与 `RDMAEndpoint` 一致：接受 `assign` 列表，返回 `SlimeReadWriteFuture`。

| 参数      | 类型   | 说明                                                                         |
| --------- | ------ | ---------------------------------------------------------------------------- |
| remote_id | str    | 远端 peer id                                                                 |
| assign    | list   | `[(local_mr_key, remote_mr_key, target_offset, source_offset, length), ...]` |
| stream    | object | 可选，默认 None                                                              |

- **write**：`target_offset`=远端偏移，`source_offset`=本地偏移
- **read**：`target_offset`=本地偏移，`source_offset`=远端偏移

______________________________________________________________________

### close

```python
peer.close()
```

关闭所有 broker stub 连接。

______________________________________________________________________

## 完整示例

```python
from dlslime import start_broker
from dlslime import available_nic

# 每进程一个 broker，用 broker.peer() 创建 peer
broker_a = start_broker("0.0.0.0:50051")
broker_b = start_broker("0.0.0.0:50052")

channel_a = broker_a.peer("A", device_name=available_nic()[0])
channel_b = broker_b.peer("B", device_name=available_nic()[-1])

# 惰性建链（可并行），对端地址用 broker.client_addr
channel_a.connect("B", broker_b.client_addr)
channel_b.connect("A", broker_a.client_addr)

# 分配并注册 buffer
ptr_a, _ = channel_a.alloc_and_register_buffer("buf_a", 16)
ptr_b, _ = channel_b.alloc_and_register_buffer("buf_t", 16)

# RDMA Write（与 RDMAEndpoint 一致）
local_mr = channel_a.get_local_mr_key("B", "buf_a")
remote_mr = channel_a.get_remote_mr_key("B", "buf_t")
channel_a.write("B", [(local_mr, remote_mr, 0, 0, 8)], None).wait()

# 清理
channel_a.close()
channel_b.close()
broker_a.stop()
broker_b.stop()
```

详见 `example/python/p2p_rdma_rc_write_encapsulated.py`。

______________________________________________________________________

## 常量

| 常量           | 值   | 说明                         |
| -------------- | ---- | ---------------------------- |
| LAZY_TIMEOUT   | 60.0 | handshake 超时（秒）         |
| BUFFER_TIMEOUT | 30.0 | get_remote_buffer 超时（秒） |

______________________________________________________________________

## 线程安全

### RDMALazyPeer：**线程安全**（同一 broker 下）

- **同一实例**：支持多线程并发调用。内部使用互斥锁保护共享状态；ZMQ stub 按线程惰性创建（每个线程使用自己的 socket，满足 ZMQ 线程亲和性）。
- **Connect**：对同一 `remote_id` 的并发 `connect` 会串行化（先到的完成 handshake 后，后到的发现已连接即返回，幂等）。
- **推荐用法**：同一进程内可共用一个 `RDMALazyPeer` 实例，多线程可并发 `connect` 不同 remote、`register_buffer`、`read`、`write`、`get_local_mr_key`、`get_remote_mr_key`、`close`。

### Broker / start_broker：**线程安全**

- Broker 在独立线程中运行，`stop()` 可从任意线程调用。
- 多个 broker 可同时运行，各自独立。`broker.peer(...)` 返回的 RDMALazyPeer 线程安全（见上）。

### 建链时的 GIL

- `connect` 会释放 GIL，因此建链阶段其他 Python 线程可正常运行。
- 若 `connect` 不释放 GIL，B 线程在 `zmq_recv` 阻塞时会持有 GIL，主线程无法启动 A 线程。

______________________________________________________________________

## 多线程建链建议

- 两端都需调用 `connect`（谁先谁后、是否并行均可；建链是对等且幂等的）。
- `connect` 内部会释放 GIL，主线程可正常启动其他线程。
- 建链完成后，任意线程均可调用 `get_local_mr_key`、`get_remote_mr_key`、`read`、`write` 等（同一 `RDMALazyPeer` 实例线程安全）。

______________________________________________________________________

## 相关文档

- [lazy_handshake.md](lazy_handshake.md) - 惰性建链协议说明
