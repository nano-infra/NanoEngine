# 3. 内存池复用

在 DLSlime 控制面中，不但要处理点对点的问题，还要处理一个点，与多个点的通信。这就涉及到了多个端点 (Endpoint) 同时复用同一块内存池的逻辑。

## 改动点分析

### 1. String Hash 到 Int Hash

原来的 MemoryRegion 是 Int Hash 的，控制面更像是一个数据库，因此需要采用 string hash。为了保证不要在数据面的 write/read 热路径上做 String Hash，要把开销转移到 Setup（控制面建链） 阶段。

- Setup 阶段 (Slow Path): 当控制面（Rust Server/Redis）告诉 C++ 有一块内存叫 "KV_CACHE" 时，C++ 内部维护一个 Registry，分配一个唯一的 int32_t mr_handle。
- Data Path (Fast Path): 用户在 write 时，不传字符串，传 mr_handle。

#### 1.1 流程设计：

```cpp
// 在 RDMAMemoryPool 中维护映射

class RDMAMemoryPool {// 双向映射std::unordered_map<std::string, int> name_to_id_;
    std::vector<ibv_mr*> id_to_mr_; // O(1) 访问，极快public:

    // Setup 阶段调用，返回一个 handle
    int get_mr_handle(const std::string& name) {
        if (name_to_id_.find(name) == name_to_id_.end()) {
             // 没找到？这是个新名字，或者报错
             return -1;
        }

        return name_to_id_[name];
    }

    // Data Path 调用，极速
    ibv_mr* get_mr_fast(int handle) {
        return id_to_mr_[handle];
    }
};
```

```cpp
// 用户代码
int kv_handle = endpoint->get_pool()->get_mr_handle("KV_CACHE"); // 慢，只做一次
// ... 循环很多次 ...
endpoint->write(kv_handle, ...); // 快，无 String Hash 开销
```

### 2. remote_mr 冲突

remote_mr 可能来自于多个不同的 Peer，需要对不同 peer 的 remote 做作用域隔离。解决的方法是在首先将 MemoryPool 进行解耦成 LocalMemoryPool 和 RemoteMemoryTable。这两个都可以各自去在 Endpoint 中创建或者注入。

```cpp
// before
class RDMAMemoryPool {
  std::map<int, ibv_mr*> local;
  std::map<int, RemoteMR> remote;
}

// After

class RDMALocalMemoryPool {
public:
  std::map<int, ibv_mr*> local;
}

class RDMARemoteTable {
public:
  std::map<int, RemoteMR> remote;
}

class RDMAMemoryManager {
public:
  std::shared_ptr<RDMALocalMemoryPool>;
  std::shared_ptr<RDMARemoteTable>;
}

```

3. 存储共享
   当前 Endpoint 的初始化方式仅包含根据 Context，根据设备信息，现在加入一种根据内存池注册 Endpoint 的方式，是的保有相同 MemoryRegion 的 Endpoint 可以实现内存的共享。控制面给 Endpoint 传入的是一个共享的 MemoryPool 和一个独立的 RemoteMemoryTable。

```cpp
// old construction
RDMAEndpoint(std::shared_ptr<RDMAContext>    ctx,
             size_t                          num_qp,
             std::shared_ptr<RDMAWorker>     worker      = nullptr);

RDMAEndpoint(std::string                     dev_name    = "",
             int32_t                         ib_port     = 1,
             std::string                     link_type   = "RoCE",
             size_t                          num_qp      = 1,
             std::shared_ptr<RDMAWorker>     worker      = nullptr);

// append a new construction
RDMAEndpoint(std::shared_ptr<RDMAMemoryPool> ctx,
             size_t                          num_qp,
             std::shared_ptr<RDMAWorker>     worker      = nullptr);
```

## 4. PeerAgent 实现 (Python)

PeerAgent 在创建时**预先分配**共享 MemoryPool，所有 Endpoint 创建时传入该 pool：

```python
# __init__: 预先分配共享 MemoryPool
self._rdma_context = RDMAContext()
self._rdma_context.init(self.device, self.ib_port, self.link_type)
self._memory_pool = RDMAMemoryPool(self._rdma_context)

# 创建 Endpoint 时传入共享 pool
endpoint = RDMAEndpoint(pool=self._memory_pool, num_qp=qp_num)

# register_memory_region 直接使用共享 pool
handler = self._memory_pool.register_memory_region(ptr, length, mr_name)
mr_info = self._memory_pool.mr_info()[mr_name]
```

## 5. 双池设计 (meta_pool + user_pool)

- **meta_pool** (per-endpoint): sys 缓冲区 (`io_dummy`, `msg_dummy`, `send_ctx`)，不可共享。从 user_pool 借用 PD。
- **user_pool** (shared): 用户 MR，由 `register_memory_region` 注册。
- **Same PD**: meta_pool 通过 `RDMAMemoryPool(parent_pool)` 构造，复用 user_pool 的 PD。
- **RDMAChannel post\_**\*: 调用时直接注入 pool (meta_pool 或 user_pool)，不做双池查找。

## 6. connect 语义

- connect 只交换 meta 信息 (io_info, msg_info)，**不**自动注册 remote MR。
- 用户 MR 需手动: `get_mr_info()` (control plane) + `register_remote_memory_region()`。
- 推荐流程: init -> connect -> register_memory_region -> get_mr_info -> register_remote_memory_region -> read/write
