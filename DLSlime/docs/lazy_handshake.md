# 惰性建链（Lazy Handshake）

## 目标

当「本机 IP + 对端 IP」尚未建立 RDMA 连接时，可以按需建链：

1. **发起方**：先创建自己的 endpoint，把「我要连对端」的请求和自己的 endpoint 信息交给一个**中心 Broker**；
2. **对端**：从 Broker 拉取待处理请求，为每个请求创建自己的 endpoint，**先连到发起方的 endpoint**，再把对端的 endpoint 信息回写给 Broker；
3. **发起方**：从 Broker 取回对端的 endpoint 信息，**再连到对端**，完成双向建链。

这样不需要事先约定谁 bind、谁 connect，也不需要为每对 (A,B) 提前起一个 ZMQ server；只要有一个中心 Broker（一个 ZMQ 服务），所有建链都可以按需、惰性完成。

## 协议

- **Broker**：一个 `ZmqRendezvousServer`，backend 使用 `RdmaRendezvousBackend(None)`（无本地 endpoint，仅做路由）。
- **LazyHandshakeRequest(initiator_id, peer_id, my_endpoint_info)**
  发起方调用：向 Broker 登记「initiator_id 想连 peer_id，这是我的 endpoint_info」。
- **GetPendingLazyHandshakes(peer_id)**
  对端调用：拉取「谁想连我」的列表，每项为 `{initiator_id, endpoint_info}`，**消费式**（取走后从 Broker 删除）。
- **LazyHandshakeResponse(initiator_id, peer_id, my_endpoint_info)**
  对端在连上发起方后调用：把「对端的 endpoint_info」写回 Broker，供发起方取用。
- **GetLazyHandshakeResponse(initiator_id, peer_id, timeout_sec)**
  发起方调用：阻塞直到对端调用了 LazyHandshakeResponse，或超时；返回对端的 endpoint_info（超时返回空 dict）。

## 建链顺序（RDMA 语义）

- 当前 RDMA 建链是对称的：两边都 `connect(对方_info)`。
- 惰性流程里：**对端先连发起方**（对端 `connect(发起方_info)`），**发起方再连对端**（发起方 `connect(对端_info)`）。
  这样保证「先有 endpoint 的一方」先被连，再完成反向连接。

## 使用方式

1. 启动一个中心 Broker（例如一台机器或一个已知地址）：
   - `backend = RdmaRendezvousBackend(None)`
   - `server = ZmqRendezvousServer(backend, "0.0.0.0:50051")`
   - `server.start()`
2. 发起方 A：创建 `RDMAEndpoint`，`stub = ZmqRendezvousStub(broker_addr)`，
   `stub.request_lazy_handshake("A", "B", my_ep.endpoint_info())`，
   然后 `peer_info = stub.get_lazy_handshake_response("A", "B", 30.0)`，
   最后 `my_ep.connect(peer_info)`。
3. 对端 B：轮询或一次性 `pending = stub.get_pending_lazy_handshakes("B")`，
   对每个 `item`：创建 `RDMAEndpoint`，`ep.connect(item["endpoint_info"])`，
   `stub.respond_lazy_handshake(item["initiator_id"], "B", ep.endpoint_info())`。

详见示例脚本：`example/python/rdma_lazy_handshake_usage.py`。

______________________________________________________________________

## 每进程一个 Broker（P2P 自组织）

在「每进程一个 Broker」模式下，每个 peer 既是 server 也是 client，无需中心 Broker：

- **RDMALazyPeer**：`connect(remote_id, remote_broker_addr)` 时需传入对方 broker 地址。
- **对等建链**：不区分 initiator 与 target；两端都向对方 broker 登记自己的 endpoint_info，再在本 broker 上等待对方登记，然后完成 RDMA connect。谁先发起、发起几次，最终状态一致（幂等）。
- 详见 [rdma_lazy_peer_api.md](rdma_lazy_peer_api.md)。
