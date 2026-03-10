# EP 分离架构设计文档

> Encoder-Prefill Separation Architecture for Vision-Language Inference

## 1. 设计动机

VLM（Vision-Language Model）推理中，Vision Encoder（ViT）和 LLM 的计算特征截然不同：

| 特征     | Vision Encoder                | LLM                       |
| -------- | ----------------------------- | ------------------------- |
| 计算模式 | 纯 prefill（无自回归）        | prefill + decode 交替     |
| 显存占用 | 固定（模型参数 + 图像 batch） | 动态（KV cache 持续增长） |
| 计算密度 | 高（大矩阵乘法）              | 低（decode 阶段逐 token） |
| 并发需求 | 低（图像到达率受限）          | 高（持续 serving）        |

传统 co-located 方案将 Vision Encoder 和 LLM 放在同一进程、同一 GPU，问题：

1. **显存争抢**：ViT 激活与 KV cache 共享 GPU 显存，限制 batch size
2. **调度耦合**：encode 和 prefill 串行，encoder 空闲时 GPU 利用率低
3. **扩展不灵活**：encoder 和 LLM 必须 1:1 绑定，无法独立扩缩

**EP 分离架构**解决以上问题：Vision Encoder 运行在独立进程（可以是独立 GPU），通过 RDMA 零拷贝传输 vision embedding 到 LLM worker，实现：

- Encoder 与 LLM 独立扩缩（N:M 部署）
- GPU 显存各自专用，互不干扰
- Encoder 可持续流水线编码，不阻塞 LLM decode

## 2. 整体架构

```
                    NanoCtrl (Redis + HTTP)
                   ┌──────────────────────┐
                   │  Engine Registry      │
                   │  - encoder engines    │
                   │  - llm engines        │
                   │  Peer Discovery       │
                   │  Heartbeat Monitor    │
                   └──────┬───────┬───────┘
                          │       │
              register    │       │   register
              + heartbeat │       │   + heartbeat
                          │       │
         ┌────────────────▼─┐   ┌─▼────────────────────┐
         │  EncoderEngine   │   │  LLM Engine (Ray)     │
         │  (独立进程)       │   │  (TP/EP workers)      │
         │                  │   │                       │
         │  ┌────────────┐  │   │  ┌─────────────────┐  │
         │  │ VisionEncoder│ │   │  │ ModelRunner      │  │
         │  │ (ViT)       │  │   │  │ RDMA fetch +    │  │
         │  └──────┬─────┘  │   │  │ inject embeds   │  │
         │         │        │   │  └────────┬────────┘  │
         │         ▼        │   │           │           │
         │  ┌────────────┐  │   │  ┌────────▼────────┐  │
         │  │EmbeddingPool│──RDMA──►│ recv_buf (temp) │  │
         │  │ (GPU + MR) │  │   │  │ (per-step alloc)│  │
         │  └──────▲─────┘  │   │  └─────────────────┘  │
         │         │        │   │                       │
         │    free slots    │   │                       │
         │         │        │   │    ┌──────────────┐   │
         │  ┌──────┴─────┐  │   │    │ LlmComponent  │   │
         │  │ P2P ZMQ    │◄─────────│ send_free_    │   │
         │  │ ROUTER     │  │   │    │ vision_slots()│   │
         │  └────────────┘  │   │    └──────────────┘   │
         └──────────────────┘   └───────────────────────┘
```

## 3. 核心组件设计

### 3.1 EmbeddingPool — 固定槽位显存池

**设计理念**：预分配固定大小的 GPU 显存 buffer，通过 slot 机制管理并发 encode 结果。

```python
@dataclass
class EmbeddingPool:
    num_slots: int              # 并发 slot 数量（默认 16）
    max_tokens_per_slot: int    # 每个 slot 最大 token 数（默认 4096）
    hidden_size: int            # embedding 维度
    device: str                 # CUDA 设备
    dtype: torch.dtype          # 数据类型（bfloat16）
```

**显存布局**：

```
buffer: torch.zeros(num_slots, max_tokens_per_slot, hidden_size, dtype=dtype, device=device)

┌─────────────────────────────────────────────────────────────┐
│ Slot 0: [max_tokens_per_slot × hidden_size]                 │
├─────────────────────────────────────────────────────────────┤
│ Slot 1: [max_tokens_per_slot × hidden_size]                 │
├─────────────────────────────────────────────────────────────┤
│ ...                                                         │
├─────────────────────────────────────────────────────────────┤
│ Slot N-1: [max_tokens_per_slot × hidden_size]               │
└─────────────────────────────────────────────────────────────┘

slot_byte_offset(i) = i × max_tokens_per_slot × hidden_size × itemsize
slot_num_bytes(i)   = actual_tokens[i] × hidden_size × itemsize
```

**为什么用固定 slot 而不是动态分配？**

1. RDMA Memory Region 注册要求 buffer 地址和大小在注册时确定
2. 固定 slot 使得 offset 计算确定性，无需动态 MR 注册/注销
3. 简化并发管理——slot 分配/释放是 O(1) 操作

**Slot 生命周期**：

```
free → allocate(num_tokens) → write_slot(data) → [RDMA read by LLM] → free(slot_idx) → free
```

### 3.2 RDMA 传输层

**RDMA MR 注册（Encoder 端）**：

```python
# EncoderEngine._start_peer_agent()
peer_agent = dlslime.start_peer_agent(
    alias=f"{engine_id}:0",
    server_url=nanoctrl_address,  # Redis for peer discovery
    device=nic,
    ib_port=1,
    link_type="RoCE",
    scope=nanoctrl_scope
)
mr_handler = pool.register_mr(peer_agent)
# → peer_agent.register_memory_region("vision_embed", buffer.data_ptr(), buffer_size_bytes)
```

**RDMA Read（LLM 端 — `_fetch_vision_embeds_rdma`）**：

```python
# Step 1: 分配本地接收 buffer
recv_buf = torch.zeros(total_tokens, hidden_size, dtype=model_dtype, device=device)

# Step 2: 注册为临时 MR
peer_agent.register_memory_region("vision_recv", recv_buf.data_ptr(), recv_buf.nbytes)

# Step 3: 建立连接（按需）
peer_agent.set_desired_topology([encoder_alias])
peer_agent.wait_for_peers(timeout=30)

# Step 4: 获取远端 MR 信息
remote_mr = peer_agent.get_mr_info(encoder_alias, "vision_embed")
peer_agent.register_remote_memory_region(encoder_alias, remote_mr)
endpoint = peer_agent.get_endpoint(encoder_alias)

# Step 5: 构造批量 RDMA read ops
slot_stride = max_tokens_per_slot * hidden_size * itemsize
for slot in slots:
    remote_offset = slot.slot_idx * slot_stride
    read_len = slot.num_tokens * hidden_size * itemsize
    ops.append(RdmaOp(remote_offset, local_offset, read_len))
    local_offset += read_len

# Step 6: 批量执行
endpoint.read(ops)  # 硬件级零拷贝 GPU→GPU
```

**关键设计点**：

- **Dtype 一致性**：recv_buf 使用 `model.embed_tokens.weight.dtype`（而非 `torch.get_default_dtype()`），确保与 encoder 端 EmbeddingPool 的 dtype 匹配
- **Slot stride vs actual bytes**：remote_offset 按 `max_tokens_per_slot` 计算（定位 slot 起始），read_len 按实际 `num_tokens` 计算（只读有效数据）
- **临时 MR**：LLM 端不持有固定 EmbeddingPool，每次 step 动态分配 recv_buf 并注册为临时 MR，step 结束后释放
- **按需连接**：LLM worker 通过 `set_desired_topology` + `wait_for_peers` 按需与 encoder 建立 RDMA 连接，复用跨 step

### 3.3 Vision Embedding 注入

```python
def _inject_vision_embeds(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
    """在 prefill 阶段用 vision embedding 替换占位符 token"""
    if self._vision_embeds is None:
        return None  # 纯文本请求

    # 1. 先做正常 text embedding
    inputs_embeds = self.model.model.embed_tokens(input_ids)

    # 2. 找到 image placeholder token 位置
    image_mask = (input_ids == self.image_token_id)

    # 3. masked_scatter：在 placeholder 位置填入 vision embedding
    inputs_embeds[image_mask] = vision_embeds.to(inputs_embeds.dtype)

    return inputs_embeds
```

**注入时机**：`run_model()` 调用 `_inject_vision_embeds()` 获取 `inputs_embeds`，若非 None 则传给 `model.forward(input_ids, positions, inputs_embeds=inputs_embeds)`，模型跳过内部 `embed_tokens` 直接使用注入的 embedding。

**对非 VL 模型的兼容**：`Qwen3MoeForCausalLM.forward()` 不接受 `inputs_embeds` 参数。因此 `run_model()` 只在 `inputs_embeds is not None` 时传递该关键字参数。

### 3.4 P2P Slot 回收

**为什么需要 P2P 回收？**

Encoder 端 EmbeddingPool slot 是有限资源（默认 16 个）。LLM prefill 完成后 vision embedding 已被消费（注入到 text embedding 中），encoder 端 slot 必须及时释放以供后续请求使用。

**消息格式**：

```
ZMQ DEALER → ZMQ ROUTER
├── Action = 4 (FreeVisionSlots)
└── FlatBuffer payload:
    ├── encoder_engine_id: string
    ├── source_engine_id: string
    └── slot_indices: [int32]
```

**触发时机**（`engine_server.py` 中的 engine backend loop）：

```python
# 每个 step 完成后
for seq in scheduled_sequences:
    for vision_slot in seq.vision_slots:
        vision_free_by_encoder[vision_slot.encoder_engine_id].append(vision_slot.slot_idx)
    seq.clear_vision_slots()

for encoder_id, slot_indices in vision_free_by_encoder.items():
    engine.send_free_vision_slots(encoder_id, slot_indices)
```

**Encoder 端处理**：

```python
# EncoderEngine._handle_p2p_message
packet = decode_packet(raw)
if packet.action == 4:  # FreeVisionSlots
    fb = FreeVisionSlots.GetRootAs(packet.payload)
    slot_indices = [fb.SlotIndices(i) for i in range(fb.SlotIndicesLength())]
    self.pool.free_many(slot_indices)
```

### 3.5 NanoCtrl 集成

**Encoder 注册**：

```python
POST /register_engine
{
    "engine_id": "uuid",
    "role": "encoder",
    "world_size": 1,
    "peer_addrs": ["10.0.0.1:port"],
    "p2p_host": "10.0.0.1",
    "p2p_port": 12345,
    "scope": "my_cluster"
}
```

**LLM 查询 Encoder 信息**：

```python
# LlmComponent._fetch_peer_info_from_nanoctrl()
# → GET /engine_info?engine_id=xxx&scope=yyy
# 返回 p2p_host, p2p_port, peer_addrs 用于 ZMQ 和 RDMA 连接
```

**心跳机制**：Encoder 每 15s 发送 `POST /heartbeat_engine`，NanoCtrl 通过 Redis TTL 检测超时。

**幂等 unregister**：Encoder shutdown 时 `POST /unregister_engine`，即使引擎已被 TTL 清除也不报错（返回 WARN + 成功）。

## 4. 配置设计

### 4.1 EncoderConfig

```python
class EncoderConfig(BaseModel):
    model: str                              # HF 模型目录
    vision_device: str = "cuda:0"           # ViT GPU
    vision_dtype: str = "bfloat16"          # ViT dtype
    num_slots: int = 16                     # EmbeddingPool slot 数
    max_tokens_per_slot: int = 4096         # 每 slot 最大 token 数
    nanoctrl_address: Optional[str] = None  # NanoCtrl HTTP 地址
    nanoctrl_scope: Optional[str] = None    # 多租户 scope
    host: str = "0.0.0.0"                   # P2P ZMQ 绑定地址
    p2p_port: int = 0                       # P2P 端口（0=自动分配）
```

### 4.2 LLM Config（NanoDeploy Config）

EP 相关字段：

- `nanoctrl_address`：NanoCtrl 地址，**有值即启用** NanoCtrl 集成
- `nanoctrl_scope`：多租户隔离 scope

**设计原则**：无 `enable_nanoctrl` 开关，纯 address-based 启用，减少配置歧义。

### 4.3 参数传递原则

**显式注入，拒绝环境变量**：

所有 `nanoctrl_scope` / `nanoctrl_address` 通过构造函数参数传递，不从 `os.getenv()` 读取。

原因：

1. Ray actor 跨进程场景下子进程不一定继承环境变量
2. 环境变量隐式覆盖导致调试困难
3. 显式参数更容易追踪数据流

唯一例外：NanoOps（编排器）设置环境变量供子进程使用，这是编排层的职责。

## 5. 数据流序列图

```
User Request (image + prompt)
        │
        ▼
   ┌─────────┐      ┌──────────────┐      ┌────────────┐
   │ Scheduler│─────►│ EncoderEngine│      │ LLM Worker │
   │(NanoCtrl)│      │              │      │ (Ray)      │
   └─────────┘      │  1. encode() │      │            │
                     │  pixel_values│      │            │
                     │      │       │      │            │
                     │      ▼       │      │            │
                     │  VisionEncoder      │            │
                     │  (ViT forward)      │            │
                     │      │       │      │            │
                     │      ▼       │      │            │
                     │  pool.write_slot()  │            │
                     │  → GPU buffer│      │            │
                     │      │       │      │            │
                     │      ▼       │      │            │
                     │  VisionSlotMeta     │            │
                     │  (slot_id,   │      │            │
                     │   remote_addr,      │            │
                     │   rkey,      │      │            │
                     │   num_tokens)│      │            │
                     └──────┬───────┘      │            │
                            │              │            │
                     FBS VisionSlot        │            │
                     in Sequence           │            │
                            │              │            │
                            └──────────────┤            │
                                           │            │
                                    2. run_from_bytes() │
                                    extract VisionSlots │
                                           │            │
                                    3. _fetch_vision_   │
                                       embeds_rdma()    │
                                    RDMA read ◄─────────┤
                                    (GPU→GPU)  ─────────┤
                                           │            │
                                    4. _inject_vision_  │
                                       embeds()         │
                                    masked_scatter      │
                                           │            │
                                    5. model.forward()  │
                                    (prefill)           │
                                           │            │
                                    6. decode loop      │
                                    (autoregressive)    │
                                           │            │
                     ┌──────────────┐      │            │
                     │ EncoderEngine│◄─────┤            │
                     │              │      │            │
                     │ 7. P2P free  │  FreeVisionSlots  │
                     │ pool.free()  │  (Action=4)       │
                     └──────────────┘      └────────────┘
```

## 6. 关键约束与设计权衡

### 6.1 为什么 LLM 端不持有 EmbeddingPool？

LLM worker 使用临时 `recv_buf`（每 step 分配/释放）而非固定 EmbeddingPool，原因：

1. LLM worker 显存主要用于 KV cache，固定 EmbeddingPool 会挤占 KV cache 空间
2. Vision embedding 只在 prefill 阶段使用一次，之后立即释放
3. 动态分配的 recv_buf 大小精确等于实际 token 数，无浪费

### 6.2 FlatBuffers 选择

使用 FlatBuffers 而非 Protobuf/JSON 传递 VisionSlot 信息：

- 零反序列化开销（直接从 buffer 读字段）
- 与 NanoSequence 的 Sequence 序列化方案一致
- 固定 schema 保证版本兼容

### 6.3 ZMQ P2P vs 通过 NanoCtrl 中转

Slot 释放消息走 ZMQ P2P 直连（DEALER→ROUTER），不经 NanoCtrl：

- 低延迟：P2P 直连 \< 1ms，中转需 2 次网络 RTT
- 减少 NanoCtrl 压力：slot 释放频率高（每个 prefill step 后）
- 容错简单：ZMQ DEALER 发送失败只影响本次释放，不影响 NanoCtrl 状态

### 6.4 Scope 隔离

通过 `nanoctrl_scope` 实现多租户隔离：

- Redis key 前缀：`{scope}:engine:{id}`
- RDMA PeerAgent scope：限制 peer discovery 范围
- 同一物理集群可运行多个独立的 EP 推理集群
