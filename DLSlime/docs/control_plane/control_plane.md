# 控制平面

## 背景

我们在 LMDeploy/NanoDeploy PD 和 EPD 的开发中明显发现，当出现大的 DP Instances 对大的 DP Instances 的时候，DLSlime 的建链流程和建链效率有待优化。

1. DP 32 对 DP 32 将建立 144 条链路。必须采用多线程。DLSlime 0.0.2 对线程安全的检查更严格，同时要支持 send/recv 建链，导致信令的载荷和传输效率均有所降低。
2. 当前建链采用 Handshake 的方式，交互周期是比较长的，且需要在应用层（NanoDeploy，LMDeploy）管理，需要下沉到 DLSlime 中简化 PD 分离的上层逻辑。否则，应用层的节点发现，容错逻辑都会变得比较复杂，难以做横向扩展。
3. AFD，EPD，PD 每一个都需要在应用层去写控制逻辑，使用极度不方便。

## DLSlime New API proposal

### 数据库 Schema 设计

我们需要新加入数据结构来完成 rust-server 和 cpp-client 的交互从而完成建链。我们使用 redis / etcd 来管理数据（当前采用 redis）。

| Logical Table  | Redis Key Pattern   | Redis Type | Fields (Key)       | Value Format   | 备注            |
| :------------- | :------------------ | :--------- | :----------------- | :------------- | :-------------- |
| **PeerAgent**  | `agent:{name}`      | Hash       | device             | String         |                 |
|                |                     |            | ib_port            | Int String     |                 |
|                |                     |            | addr               | String         | IP地址          |
| **Connection** | `conn:{low}:{high}` | Hash       | status             | Int String     | 原子锁的关键    |
|                |                     |            | qp_num             | Int String     |                 |
|                |                     |            | low_endpoint_info  | **JSON/Proto** | 存 EndpointInfo |
|                |                     |            | high_endpoint_info | **JSON/Proto** | 存 EndpointInfo |
| **Mailbox**    | `inbox:{name}`      | List       | (Element)          | JSON           | 命令队列        |

### 客户端设计

客户端维护对称的五个 API 和一个对称的 Redis 订阅事件 handler。

#### API 示意

```python
peer_agent_0 = start_peer_agent(
    alias="A",
    address="127.0.0.1:6379",
    device="mlx5_0"，
    link_type="RoCE",
    ib_port=1
)

peer_agent_1 = start_peer_agent(
    alias="B",
    backend="redis",
    address="127.0.0.1:6379",
    device="mlx5_1",
    link_type="RoCE",
    ib_port=1
)

print(peer_agent_0.query)

# {"A": {"device": "mlx5_0"}, "B": {"device": "mlx5_1"}}

# 当使用 peer_agent_0 对 peer_agent_1 进行连接的时候，在服务端水平触发两个连接。
peer_agent_0.init("B", qp_num=1)

"""
client: 这个时候其实是 cpp client 给 rust server 发送了一条 /init 请求。
Server: 这个时候客户端首先去 query 是否有 "A,B" (字典序) 的 peer 数据结构。如果有并大于等于 initialized，那就返回链路已经创建就好了。如果有但小于 initialized，这个时候会过滤掉这条消息并恢复已经在创建中了。 如果没有，Server 将会发布一条从 <server>:<low> 和 <server>:<high> 分别发布一条消息，这个时候 server 端会 await future，future 等待 A,B:A_DONE 和 A,B:B_DONE 的 peer 的 init 完成事件。 server 端会维护一个接收 ACK 的 loop handle，从而让 future 响应返回给用户。
client: 这个时候 client 的 event loop 订阅到了这条消息，就触发 Endpoint 的创建，创建完成后向发送 Server 端发送 ACK。
"""

# 这个时候因为已经完成了建链，由于是幂等的，所以这条其实可以省略。
peer_agent_1.init("A", qp_num=1)

"""
和 init 类似的方式 Connect。
"""
peer_agent_1.connect("A")
peer_agent_0.connect("B")

"""
这个时候创建内存并通知 rust server 端更新数据库
"""
A_tensor = torch.tensor(...)
peer_agent_0.register_memory_region("KVCACHE", A_tensor.data_ptr(), A_tensor.storage_offset(), A_tensor.numel() * A_tensor.itemsize)

B_tensor = torch.tensor(...)
peer_agent_1.register_memory_region("KVCACHE", B_tensor.data_ptr(), B_tensor.storage_offset(), B_tensor.numel() * B_tensor.itemsize)

"""
首先 Query 远端内存的句柄
"""
remote_mr_info = peer_agent_0.get_mr_info("B", "KV_CACHE")

peer_agent_0.register_remote_memory_region("B", "KV_CACHE", remote_mr_info)

"""
远程读写 (发生在本地的 P2P)
"""
peer_agent_0.write([("KVCACHE", "KVCACHE", target_off, src_off, length)]).wait()
```

```cpp
// Event handler
// 事件响应
// 响应建链
// 响应连接
```

### 服务端设计

服务端维护五个 API 和一个 Redis 订阅事件 handler。

````rust
async fn query(State(state): State<AppState>, Json(body): <QueryBody>) -> Impl IntoResponse {
    // 查询有哪些 peer agent
}

async fn init(State(state): State<AppState>, Json(body): <InitBody>) -> Impl IntoResponse {
    let (node1, node2) = sort(req.src, req.dst);
    let _guard = state.locks.lock(format!("{}:{}", node1, node2)).await;
    // Step 1. 去重锁。
    // Step 2. 接收请求，发布 init 请求。生成 event future
    // Step 3. await event future。
}

async fn connect(State(state): State<AppState>, Json(body): <ConnectBody>) -> Impl IntoResponse {
    // Step 1. 去重锁。
    // Step 2. 接收请求，发布 connect 请求。生成 event future
    // Step 3. await event future。
}

async fn release_memory_region(State(state): State<AppState>, Json(body): <ReleaseMemoryRegionBody>) -> Impl IntoResponse {
    // 上报 mr 信息
}

async fn get_mr_info(State(state): State<AppState>, Json(body): <GetMrInfoBody>) -> Impl IntoResponse {
    // 获取 mr_info 的信息

## 和 DLSlime 的交互与改进

### 改进点分析
[share_memory.md](share_memory.md)

### API 交互

使用到的 DLSlime 中的 API。详见 rdma_endpoint.h。

``` cpp
class RDMAEndpoint: public std::enable_shared_from_this<RDMAEndpoint> {
    friend class RDMAWorker;

public:
    RDMAEndpoint(std::shared_ptr<RDMAContext>    ctx,
                 size_t                          num_qp,
                 std::shared_ptr<RDMAWorker>     worker      = nullptr);

    RDMAEndpoint(std::string                     dev_name    = "",
                 int32_t                         ib_port     = 1,
                 std::string                     link_type   = "RoCE",
                 size_t                          num_qp      = 1,
                 std::shared_ptr<RDMAWorker>     worker      = nullptr);

    ~RDMAEndpoint();

    void connect(const json& remote_endpoint_info);

    json endpointInfo() const;
    void shutdown();

    int32_t registerOrAccessMemoryRegion(uintptr_t mr_key, uintptr_t ptr, uintptr_t, size_t length);
    int32_t registerOrAccessRemoteMemoryRegion(uintptr_t ptr, json mr_info);

    // TwoSide Primitive
    std::shared_ptr<SendFuture> send(const chunk_tuple_t& chunk, void* stream_handler);
    std::shared_ptr<RecvFuture> recv(const chunk_tuple_t& chunk, void* stream_handler);

    // OneSide Primitive
    std::shared_ptr<ReadWriteFuture> read(const std::vector<assign_tuple_t>& assign, void* stream);
    std::shared_ptr<ReadWriteFuture> write(const std::vector<assign_tuple_t>& assign, void* stream);
    std::shared_ptr<ReadWriteFuture>
    writeWithImm(const std::vector<assign_tuple_t>& assign, int32_t imm_data, void* stream);

    std::shared_ptr<ImmRecvFuture> immRecv(void* stream = nullptr);

    int32_t process();

    void setId(int64_t id)
    {
        id_.store(id, std::memory_order_relaxed);
    }
    int64_t getId() const
    {
        return id_.load(std::memory_order_relaxed);
    }

    void cancelAll();
    ...
}
````

## 第三方 Library 选型

1. cpp-httplib:  用于 HTTP 请求
2. hiredis: 用于 Redis 交互。

## 预期效果

- DLSlime 将支持控制面的端点管理，DLSlime 将通过 broker 模式，简化建链流程，增加建链的效率，并将在 NanoDeploy 中投入使用。
