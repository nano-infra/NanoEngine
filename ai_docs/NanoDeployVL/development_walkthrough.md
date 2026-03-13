# EP 分离 VL 推理 — 开发 Walkthrough

> 从零到端到端运行：代码路径、调试技巧、常见陷阱

## 1. 前置条件

### 1.1 基础设施

| 组件          | 说明                              |
| ------------- | --------------------------------- |
| **Redis**     | NanoCtrl 后端存储，peer discovery |
| **NanoCtrl**  | 引擎注册、心跳、peer 信息查询     |
| **Ray**       | LLM worker 分布式执行             |
| **dlslime**   | RDMA 通信库（PeerAgent, MR 管理） |
| **RoCE 网卡** | RDMA over Converged Ethernet      |

### 1.2 模型要求

HuggingFace 格式模型，需包含 `vision_config`（如 Qwen3.5-VL-35B-A3B）。

确认模型有 `vision_config`：

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("/path/to/model")
assert hasattr(cfg, "vision_config"), "模型无 vision_config，不支持 EP 模式"
print(f"hidden_size={cfg.vision_config.out_hidden_size}")
```

## 2. 启动流程 Walkthrough

### Step 1: 启动 NanoCtrl

```bash
cd NanoCtrl
cargo run -- --config config.toml
```

确认 Redis 可达，NanoCtrl 日志显示 `Listening on ...`。

### Step 2: 启动 EncoderEngine

```python
from nanodeployvl import EncoderConfig, EncoderEngine

encoder_config = EncoderConfig(
    model="/path/to/qwen3.5-vl-35b",
    vision_device="cuda:7",         # 独立 GPU
    vision_dtype="bfloat16",
    num_slots=16,
    max_tokens_per_slot=4096,
    nanoctrl_address="http://10.0.0.1:8080",
    nanoctrl_scope="my_cluster",
    p2p_port=0                      # 自动分配
)
encoder_engine = EncoderEngine(encoder_config)
```

**初始化序列**：

```
EncoderEngine.__init__
├── 1. 生成 engine_id (UUID)
├── 2. 加载 VisionEncoder (ViT)
│   └── vision_config → VisionModel → GPU
├── 3. 分配 EmbeddingPool
│   └── torch.zeros(16, 4096, 2048) on cuda:7 ≈ 250MB
├── 4. 启动 RDMA PeerAgent
│   ├── dlslime.available_nic() → 选网卡
│   ├── dlslime.start_peer_agent(alias, server_url, RoCE)
│   └── pool.register_mr(peer_agent) → "vision_embed" MR
├── 5. 启动 P2P ZMQ listener
│   └── ZMQ ROUTER on tcp://0.0.0.0:{auto_port}
├── 6. 注册到 NanoCtrl
│   ├── POST /register_engine (role=encoder)
│   └── 启动心跳线程 (15s interval)
└── 7. atexit.register(shutdown)
```

**验证**：

```python
print(f"Engine ID: {encoder_engine.engine_id}")
print(f"P2P port: {encoder_engine._p2p_port}")
# NanoCtrl 日志应显示 register_engine 成功
```

### Step 3: 编码图像

```python
from nanodeployvl import ImageProcessor

processor = ImageProcessor(encoder_config.model)
pixel_values, image_grid_thw = processor.process_images(
    ["/path/to/image.jpg"],
    device=encoder_config.vision_device,
    dtype=torch.bfloat16
)

slot_metas = encoder_engine.encode(pixel_values, image_grid_thw)
# slot_metas: list[VisionSlotMeta]
# 每个 VisionSlotMeta 包含:
#   encoder_engine_id, slot_idx, num_tokens, hidden_size, max_tokens_per_slot
```

**编码内部流程**：

```
encode(pixel_values, image_grid_thw)
├── VisionEncoder.encode() → list[Tensor]
│   └── 每张图一个 Tensor: [num_tokens, hidden_size]
├── for each embedding:
│   ├── pool.allocate(num_tokens) → slot_idx
│   ├── pool.write_slot(slot_idx, embedding)
│   └── 构造 VisionSlotMeta
└── return slot_metas
```

### Step 4: 启动 LLM Engine 并推理

```python
from nanodeploy import Config, LLMComponent

config = Config(
    model="/path/to/qwen3.5-vl-35b",
    nanoctrl_address="http://10.0.0.1:8080",
    nanoctrl_scope="my_cluster",
    tp=2,
    # ... 其他参数
)
llm = LLMComponent(config)

# 构造 Sequence，附加 vision_slot 信息
from nanosequence import Sequence
seq = Sequence(request_id="req1", input_ids=token_ids)
for meta in slot_metas:
    seq.add_vision_slot(
        encoder_engine_id=meta.encoder_engine_id,
        slot_idx=meta.slot_idx,
        num_tokens=meta.num_tokens,
        hidden_size=meta.hidden_size,
        max_tokens_per_slot=meta.max_tokens_per_slot,
    )

# 推理
output = llm.generate(seq)
```

**LLM 端推理流程**：

```
ModelRunner.run_from_bytes(serialized_batch)
├── 反序列化 FlatBuffers → 提取 VisionSlot 信息
├── _fetch_vision_embeds_rdma(vision_slots)
│   ├── 按 encoder_engine_id 分组
│   ├── 分配 recv_buf: torch.zeros(total_tokens, hidden_size)
│   ├── 注册临时 MR "vision_recv"
│   ├── per encoder:
│   │   ├── 建立 RDMA 连接 (set_desired_topology + wait_for_peers)
│   │   ├── 获取远端 MR "vision_embed"
│   │   └── 批量 RDMA read (endpoint.read(ops))
│   └── self._vision_embeds = {"image": recv_buf}
├── run_model(input_ids, positions)
│   ├── _inject_vision_embeds(input_ids)
│   │   ├── inputs_embeds = embed_tokens(input_ids)
│   │   ├── mask = (input_ids == image_token_id)
│   │   └── inputs_embeds[mask] = vision_embeds
│   └── model.forward(input_ids, positions, inputs_embeds=inputs_embeds)
└── [prefill 完成]

EngineServer backend loop:
├── step 完成
├── 收集所有 seq 的 vision_slots
├── seq.clear_vision_slots()
└── send_free_vision_slots(encoder_id, slot_indices)
    ├── 查询 NanoCtrl 获取 encoder p2p 地址
    ├── ZMQ DEALER connect
    └── 发送 FlatBuffer FreeVisionSlots (Action=4)

EncoderEngine P2P listener:
├── recv ZMQ message
├── decode_packet → action=4
├── FreeVisionSlots.GetRootAs → slot_indices
└── pool.free_many(slot_indices)
```

## 3. 端到端测试脚本

使用 `NanoDeploy/examples/test_ep_full.py`：

```bash
python NanoDeploy/examples/test_ep_full.py \
    --model /path/to/qwen3.5-vl-35b \
    --image_path /path/to/test.jpg \
    --nanoctrl_scope my_cluster \
    --nanoctrl_address http://10.0.0.1:8080 \
    --encoder_device cuda:7 \
    --tp 2 \
    --max_tokens 256 \
    --enforce_eager   # 调试时禁用 CUDA graph
```

测试脚本执行两阶段：

1. **step_encoder**：创建 EncoderEngine → 处理图像 → encode → 返回 VisionSlotMeta
2. **step_llm**：创建 LLM Config → 构造 Sequence（附 vision slots）→ generate → 打印输出 → P2P free slots

## 4. 调试指南

### 4.1 RDMA 连接问题

**症状**：`_fetch_vision_embeds_rdma()` 超时或报 `wait_for_peers timeout`

**排查**：

```python
# 1. 确认 encoder 端 PeerAgent 已注册
# NanoCtrl 日志应有 register_engine 记录

# 2. 确认 RoCE 网卡可达
dlslime.available_nic()  # 应返回可用网卡列表

# 3. 确认 scope 一致
# encoder scope 和 LLM scope 必须相同，否则 peer discovery 找不到对方

# 4. 确认 MR 注册成功
# encoder 日志应有 register_memory_region("vision_embed", ...) 成功
```

### 4.2 Dtype 不匹配

**症状**：模型输出乱码（garbage text）

**原因**：RDMA 按原始字节拷贝，如果 recv_buf dtype 与 encoder 端 EmbeddingPool dtype 不一致，字节解读方式不同，embedding 值全错。

**检查**：

```python
# Encoder 端
print(encoder_engine.pool.dtype)  # 应为 torch.bfloat16

# LLM 端 (model_runner.py)
print(self.model.model.embed_tokens.weight.dtype)  # 应同为 bfloat16
```

### 4.3 Scope 不匹配

**症状**：LLM 查询 NanoCtrl 找不到 encoder，P2P free 失败

**原因**：Redis key 前缀不同。encoder 注册在 `{scope}:engine:{id}`，LLM 查 `engine:{id}`。

**检查**：确保 `EncoderConfig.nanoctrl_scope` == `Config.nanoctrl_scope`。

### 4.4 inputs_embeds TypeError

**症状**：非 VL 模型（如 Qwen3Moe）报 `TypeError: forward() got an unexpected keyword argument 'inputs_embeds'`

**原因**：`run_model()` 始终传 `inputs_embeds=None`，但某些模型的 forward 不接受该参数。

**已修复**：只在 `inputs_embeds is not None` 时传递。

### 4.5 环境变量问题

**规则**：NanoDeploy / DLSlime / NanoDeployVL 中 **不使用** `os.getenv("NANOCTRL_SCOPE")` 或 `os.getenv("NANOCTRL_ADDRESS")`。所有配置通过构造函数参数显式传递。

如果遇到 `NameError: name 'NANOCTRL_SCOPE' is not defined` 或 scope 为 None，检查参数传递链：

```
CLI args → Config(nanoctrl_scope=...)
         → set_cache_context(nanoctrl_scope=...)
         → CacheContext.nanoctrl_scope
         → get_engine_info_batch(scope=self.nanoctrl_scope)
```

## 5. 文件修改速查表

开发 EP 功能时最常修改的文件：

| 修改目标             | 文件                                                  |
| -------------------- | ----------------------------------------------------- |
| Encoder 生命周期     | `NanoDeployVL/nanodeployvl/encoder/encoder_engine.py` |
| Encoder 配置         | `NanoDeployVL/nanodeployvl/encoder/encoder_config.py` |
| ViT 实现             | `NanoDeployVL/nanodeployvl/vision/encoder.py`         |
| 图像预处理           | `NanoDeployVL/nanodeployvl/vision/processor.py`       |
| EmbeddingPool        | `NanoDeploy/nanodeploy/context/embedding_pool.py`     |
| RDMA fetch + inject  | `NanoDeploy/nanodeploy/worker/model_runner.py`        |
| P2P slot 释放        | `NanoDeploy/nanodeploy/llm_component.py`              |
| 释放触发             | `NanoDeploy/nanodeploy/server/engine_server.py`       |
| LLM 配置             | `NanoDeploy/nanodeploy/config.py`                     |
| KV cache + PeerAgent | `NanoDeploy/nanodeploy/context/cache.py`              |
| 端到端测试           | `NanoDeploy/examples/test_ep_full.py`                 |

## 6. 常见陷阱

| 陷阱                         | 说明                                           | 解决                                                                    |
| ---------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------- |
| RDMA dtype 不匹配            | recv_buf 用 float32，pool 用 bfloat16          | 统一使用 `model.embed_tokens.weight.dtype`                              |
| scope 不一致                 | encoder/LLM scope 不同导致 peer discovery 失败 | 统一传入 `nanoctrl_scope`                                               |
| 用环境变量传 scope           | Ray actor 子进程不继承                         | 全部走构造函数参数                                                      |
| hybrid 模式启动 PeerAgent    | 无 NanoCtrl 地址时初始化失败                   | `mode=="hybrid"` 跳过                                                   |
| `num_kv_layers` NameError    | 参数名写错                                     | 使用 `num_hidden_layers`                                                |
| 向非 VL 模型传 inputs_embeds | TypeError                                      | 条件传递                                                                |
| unregister 已过期引擎        | NanoCtrl 404 ERROR 日志                        | 幂等化（WARN + 返回成功）                                               |
| slot_stride 计算错误         | 用 actual_tokens 而非 max_tokens_per_slot      | remote_offset = slot_idx × max_tokens_per_slot × hidden_size × itemsize |
