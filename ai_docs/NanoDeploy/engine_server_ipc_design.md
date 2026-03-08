# 基于多进程 IPC 的 Engine Server 重构方案

为了彻底解决计算与 I/O 的互相争抢问题，提供最纯粹的网络吞吐量和隔离性，我们将 `EngineServer` 拆分为两个独立的操作系统进程：**Frontend (网络处理)** 和 **Backend (执行引擎)**。

## 架构设计

1. **Frontend 进程 (`EngineServer` 主进程)**：
   - 包含当前的 ZMQ 循环 (`recv_loop`, `send_loop`, `p2p_recv_loop`)。
   - 负责与 NanoCtrl 注册、维持心跳。
   - 这部分完全非阻塞，只做网络包的搬运工。
2. **Backend 进程 (`EngineProcess`)**：
   - 包含底层的 `LLMEngine` 实体及其伴随的 `Scheduler` 和 `RayExecutor`。
   - 内部包含一个死循环或独立的 asyncio 事件循环，持续执行 `self.engine.step()`。
3. **IPC 通信桥梁**：
   - 使用 `multiprocessing.Queue` (或更高性能的基于共享内存/zmq `ipc://` 的队列) 连接前后端。我们需要一对队列：
     - `requests_queue`: Frontend 收到 **Action 1 (Add)** 或 **Action 3 (P2P Free)** 时，将反序列化好的内部 Python 数据结构 (序列/请求) 入队，Backend 不断从中 pop 出来调用 `add_request`。
     - `results_queue`: Backend 每次 `step()` 结束后，产生 `output` 或 `is_to_be_migrated` 的结果序列，入队送回给 Frontend。Frontend 一旦收到，就立即组装 FlatBuffers 字节流 (Action 0 StepOut / Action 1 Migration) 发给对应的 ZMQ sockets。

## 具体修改模块梳理

### 1. `nanodeploy/server/engine_server.py`

重构入口 `EngineServer`。

- 将原本耦合在一起的 `EngineService` 的逻辑拆分。ZMQ 接收部分仅仅将包解码后推入 `requests_queue`。
- 启动 `multiprocessing.Process(target=run_engine_backend, args=(config, requests_queue, results_queue))`。
- 在主进程的 `asyncio` 事件循环中，新增一个 `async def results_loop()`，专门用 `asyncio.get_event_loop().run_in_executor` 监听 `results_queue` 并发送 ZMQ 包。

### 2. 新增或在内部定义 `run_engine_backend` 函数

- 在这个新进程里，实例化真正的 `LLMComponent` / `LLMEngine`。
- 实现一个死循环：
  1. 尽可能榨干 `requests_queue` 中积累的新请求并加入到 `engine.scheduler`。
  2. 如果队列为空且引擎也没有事情做 (`scheduler.is_finished()`)，短暂 `time.sleep` / `asyncio.sleep` 以降低 CPU 空转。
  3. 执行一次 `outputs = engine.step()`。
  4. 遍历 `outputs` 生成结果，放入 `results_queue` 发给主进程。

### 3. 数据结构的清理

因为 `Sequence` 类可能包含一些 C++ 对象引用或锁（比如 `metrics`，或者 block allocation），直接通过 multiprocessing 队列传递大型对象可能会触发 `pickle` 问题。

- 解决思路：如果可能遇到 pickle 问题，前端可以将收到的原始 bytes (Payload) 直接塞给后端解析；后端产生结果时，也**由底层 Engine 进程直接做 FlatBuffers 序列化为 payload bytes**，通过队列传给前端再发出。这样队列里跑的就全是纯字节 `bytes`，没有任何序列化瓶颈。

## 验证计划 (Verification Plan)

用户将自行验证：

1. **启动测试**: `nanodeploy/server/engine_server.py --config ...` 跑起来，确保两个进程能成功拉起，Ray backend 正常初始化且不报错。
2. **基本功能测试**: 提交一个测试请求，观察日志确保包经过 `Frontend -> Queue -> Backend (Step) -> Queue -> Frontend ZMQ Send` 流程正确流转，没有数据损坏或停滞。
3. **并发测试**: 确保即使后端在长时间（如 50ms）处理 `step()`，前端的 P2P 探活 / NanoCtrl 响应不被阻塞。
