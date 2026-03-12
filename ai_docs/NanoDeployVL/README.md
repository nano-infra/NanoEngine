# NanoDeployVL — 视觉-语言推理引擎

> 分支: `feature/nanodeploy_vl`

## 概述

**NanoDeployVL** 为 NanoInfra 提供视觉-语言（VL）多模态推理能力。当前支持 **Qwen3.5-MoE** 系列 VLM 模型（35B-A3B / 397B-A17B-FP8）。

核心设计思路：**EP（Encoder-Prefill）分离架构** — 独立的 `EncoderEngine` 进程运行 Vision Encoder，通过 RDMA 将视觉 embedding 传输到 LLM workers 的 `EmbeddingPool`，由 ModelRunner 在 prefill 阶段注入。

## 架构

```
                        NanoCtrl (Redis + HTTP)
                       ┌──────────────────────┐
                       │  Engine Registry      │
                       │  Peer Discovery       │
                       │  Heartbeat Monitor    │
                       └──────┬───────┬───────┘
                              │       │
                  register    │       │  register
                              │       │
    Client ──HTTP──► NanoRoute (HTTP + ZMQ Router)
                       │                │
                ZMQ Action=5/6    ZMQ Sequence
                       │          (FlatBuffer)
                       ▼                ▼
┌───────────────────────┐          RDMA          ┌──────────────────────────┐
│    EncoderEngine      │  ─────────────────────► │   LLM Worker (Ray)       │
│                       │  (EmbeddingPool slots)  │                          │
│  ┌─────────────────┐  │                         │  ┌──────────────────┐    │
│  │  ImageProcessor │  │                         │  │  ModelRunner      │    │
│  │  (HF Processor) │  │                         │  │  _fetch_vision_   │    │
│  └─────────────────┘  │                         │  │  embeds_rdma()    │    │
│  ┌─────────────────┐  │                         │  │  _inject_vision_  │    │
│  │  VisionEncoder  │  │                         │  │  embeds()         │    │
│  │  (ViT on GPU)   │  │                         │  └──────────────────┘    │
│  └─────────────────┘  │                         │  ┌──────────────────┐    │
│  ┌─────────────────┐  │   P2P FreeVisionSlots   │  │  EmbeddingPool    │    │
│  │  EmbeddingPool  │ ◄─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │  │  (RDMA MR)       │    │
│  │  (RDMA MR)      │  │                         │  └──────────────────┘    │
│  └─────────────────┘  │                         └──────────────────────────┘
└───────────────────────┘
```

**数据流**:

1. 客户端发送 HTTP `/v1/chat/completions` 请求（含 `image_url` 多模态消息）到 **NanoRoute**
2. NanoRoute 检测到图片内容，通过 ZMQ Action=5 将 encode 请求发送给 `EncoderEngine`
3. `EncoderEngine` 内部：`ImageProcessor` 预处理 → `VisionEncoder` ViT 编码 → 写入 `EmbeddingPool` slot
4. `EncoderEngine` 返回 ZMQ Action=6 响应（`input_ids` + `vision_slots` 元数据）给 NanoRoute
5. NanoRoute 构建 FlatBuffer `Sequence`（含 `VisionSlot`），通过 ZMQ 发送给 LLM Engine
6. LLM worker 的 `ModelRunner` 从 FlatBuffer 提取 `VisionSlot`，通过 `_fetch_vision_embeds_rdma()` RDMA 读取 encoder 的 EmbeddingPool slot
7. `_inject_vision_embeds()` 在 prefill 阶段用 vision embeddings 替换占位符 token 的 text embedding
8. NanoRoute 流式返回 LLM 生成的 token 给客户端（SSE / JSON）
9. Prefill 完成后，LLM 端通过 P2P `FreeVisionSlots`（Action=4）通知 encoder 释放 slot

## 文件结构

### NanoDeployVL/

| 文件                                      | 说明                                                                                  |
| ----------------------------------------- | ------------------------------------------------------------------------------------- |
| `nanodeployvl/__init__.py`                | 包入口，导出 VLConfig、EncoderConfig、EncoderEngine、VisionEncoder、ImageProcessor    |
| `nanodeployvl/config.py`                  | `VLConfig` — 继承 NanoDeploy `Config`，提取 vision_config 和特殊 token ID             |
| `nanodeployvl/encoder/encoder_config.py`  | `EncoderConfig` — 独立 encoder 进程配置                                               |
| `nanodeployvl/encoder/encoder_engine.py`  | `EncoderEngine` — 独立 vision encoder 进程，连接 NanoCtrl，管理 EmbeddingPool 及 RDMA |
| `nanodeployvl/server/vl_engine_server.py` | `VLEngineServer` — Encoder-only 服务入口（health check），客户端请求由 NanoRoute 路由 |
| `nanodeployvl/vision/encoder.py`          | `VisionEncoder` + `VisionModel` — Qwen3VL ViT 实现                                    |
| `nanodeployvl/vision/processor.py`        | `ImageProcessor` — 封装 HF `Qwen3VLProcessor`                                         |

### NanoDeploy/ （EP 模式相关）

| 文件                                   | 说明                                                                             |
| -------------------------------------- | -------------------------------------------------------------------------------- |
| `nanodeploy/context/embedding_pool.py` | `EmbeddingPool` — GPU 显存 slot 管理 + RDMA Memory Region 注册                   |
| `nanodeploy/worker/model_runner.py`    | `_fetch_vision_embeds_rdma()` RDMA 拉取 + `_inject_vision_embeds()` prefill 注入 |
| `nanodeploy/llm_component.py`          | `send_free_vision_slots()` — P2P 释放 encoder 端 slot                            |
| `nanodeploy/server/engine_server.py`   | Step 完成后触发 vision slot 释放                                                 |

## 关键设计决策

1. **EP 分离**: Vision Encoder 和 LLM 运行在独立进程，通过 RDMA 传输 embedding，解耦 encoder 和 prefill 阶段。
2. **EmbeddingPool + RDMA MR**: Encoder 端固定 GPU 显存 slot，注册为 RDMA Memory Region，LLM worker 直接 RDMA read。
3. **FlatBuffers VisionSlot**: Encode 结果通过 FBS `VisionSlot`（含 slot_id、remote_addr、rkey、num_tokens、max_tokens_per_slot）传递给 LLM 调度器。
4. **P2P Slot 回收**: LLM prefill 完成后通过 ZMQ P2P 发送 `FreeVisionSlots`（Action=4）消息释放 encoder 端 slot。
5. **HF Processor 复用**: 使用 HuggingFace `Qwen3VLProcessor` 处理图文分词和图像预处理。

- **流式输出**: `generate_vl()` 当前同步，可扩展为 streaming token generation
- **视频支持**: 框架已预留 `encode_video` 接口，但未做端到端测试

______________________________________________________________________

## Bugfix 记录

### 2026-03-10: NanoCtrl scope 不匹配

**问题**: `Config` 只在 `enable_nanoctrl=True`（非 hybrid 模式）时读取 `NANOCTRL_SCOPE` 环境变量。hybrid 模式下即使显式传入 `nanoctrl_address`，scope 仍为 None，导致 LLMComponent 查 NanoCtrl 时 Redis key 前缀不匹配（encoder 注册在 `JimyMa:engine:xxx`，LLM 查 `engine:xxx`），P2P free 失败。

**修复**: 改为只要有 `nanoctrl_address` 就读取 scope。后续进一步重构：**删除 `enable_nanoctrl` 字段**，有 `nanoctrl_address` 就启用，没有就禁用。

### 2026-03-10: RDMA dtype 不匹配导致乱码输出

**问题**: `_fetch_vision_embeds_rdma()` 使用 `torch.get_default_dtype()` = float32（4 bytes/element），但 EmbeddingPool buffer 是 bfloat16（2 bytes/element）。RDMA 按原始字节拷贝，导致偏移量翻倍、数据解读错误，模型输出全是乱码。

**修复**: 改用 `self.model.model.embed_tokens.weight.dtype` 获取实际模型 dtype。

### 2026-03-10: 移除 NANOCTRL_SCOPE / NANOCTRL_ADDRESS 环境变量读取

**问题**: 多个组件从 `os.getenv("NANOCTRL_SCOPE")` / `os.getenv("NANOCTRL_ADDRESS")` 读取配置，容易在 Ray actor 跨进程场景下出错（子进程不一定继承环境变量），且有隐式覆盖的隐患。

**修复**: 全部改为显式参数传递：

- `NanoDeploy/config.py`: 删除 `enable_nanoctrl` 字段和环境变量读取，纯依赖构造时传参
- `DLSlime/peer_agent.py`: `scope` 参数恢复为正式参数，不再读 env var
- `NanoDeploy/context/cache.py`: 新增 `nanoctrl_scope` 字段，通过 `set_cache_context()` 传入
- `NanoDeploy/worker/model_runner.py`: 删除 `os.environ["NANOCTRL_SCOPE"]` propagation hack
- `NanoDeployVL/encoder_engine.py`: 删除 `os.getenv("NANOCTRL_SCOPE")` fallback
- `test_ep_full.py`: 新增 `--nanoctrl_scope` CLI 参数

### 2026-03-10: NanoCtrl unregister_engine 幂等化

**问题**: 引擎 shutdown 时调用 `/unregister_engine`，若引擎已因 TTL 过期被清除或从未成功注册，返回 404 错误，NanoCtrl 日志打出 ERROR。

**修复**: `redis_repo.rs` 中 `unregister_engine` 不再抛 `NotFound`，改为 WARN 日志 + 返回成功（幂等语义）。

### 2026-03-10: Hybrid 模式跳过 PeerAgent 启动

**问题**: `CacheContext.start_peer_agent()` 在 hybrid 模式下也尝试启动 RDMA PeerAgent 并注册 MR，但 hybrid engine 不做 P2P KV 传输，没有 NanoCtrl 地址时导致不必要的初始化失败。

**修复**: `start_peer_agent(mode)` 新增 mode 参数，`mode == "hybrid"` 时直接 return。

### 2026-03-10: set_cache_context NameError

**问题**: `set_cache_context()` 中 `num_hidden_layers=num_kv_layers`，但函数参数名是 `num_hidden_layers`，导致 `NameError: name 'num_kv_layers' is not defined`。

**修复**: 改为 `num_hidden_layers=num_hidden_layers`。

### 2026-03-10: Qwen3Moe inputs_embeds 兼容

**问题**: `run_model()` 中 `self.model(input_ids, positions, inputs_embeds=inputs_embeds)` 始终传 `inputs_embeds` 关键字参数（即使为 None），但 `Qwen3MoeForCausalLM.forward()` 不接受该参数，导致 `TypeError`。

**修复**: 只在 `inputs_embeds is not None` 时传递该参数，否则只传 `input_ids, positions`。
