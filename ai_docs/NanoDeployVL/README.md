# NanoDeployVL — 视觉-语言推理引擎

> 分支: `feature/nanodeploy_vl`

## 概述

**NanoDeployVL** 为 NanoInfra 提供视觉-语言（VL）多模态推理能力。当前支持 **Qwen3.5-MoE** 系列 VLM 模型（35B-A3B / 397B-A17B-FP8）。

核心设计思路：**EP（Encoder-Prefill）分离架构** — 独立的 `EncoderEngine` 进程运行 Vision Encoder，通过 RDMA 将视觉 embedding 传输到 LLM workers 的 `EmbeddingPool`，由 ModelRunner 在 prefill 阶段注入。

## 架构

```
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

1. `EncoderEngine` 从 NanoCtrl 接收 encode 请求（含 `pixel_values` 及 vision slot 分配）
2. `VisionEncoder` 将 `pixel_values` 编码为 vision embeddings，写入 `EmbeddingPool` 的 slot
3. LLM worker 的 `ModelRunner.run_from_bytes()` 从 FlatBuffers 中提取 `VisionSlot` 信息
4. `_fetch_vision_embeds_rdma()` 通过 dlslime PeerAgent RDMA 读取 encoder 的 EmbeddingPool slot
5. `_inject_vision_embeds()` 在 prefill 阶段用 vision embeddings 替换占位符 token 的 text embedding
6. Prefill 完成后，LLM 端通过 P2P `FreeVisionSlots`（Action=4）通知 encoder 释放 slot

## 文件结构

### NanoDeployVL/

| 文件                                     | 说明                                                                                  |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| `nanodeployvl/__init__.py`               | 包入口，导出 VLConfig、EncoderConfig、EncoderEngine、VisionEncoder、ImageProcessor    |
| `nanodeployvl/config.py`                 | `VLConfig` — 继承 NanoDeploy `Config`，提取 vision_config 和特殊 token ID             |
| `nanodeployvl/encoder/encoder_config.py` | `EncoderConfig` — 独立 encoder 进程配置                                               |
| `nanodeployvl/encoder/encoder_engine.py` | `EncoderEngine` — 独立 vision encoder 进程，连接 NanoCtrl，管理 EmbeddingPool 及 RDMA |
| `nanodeployvl/vision/encoder.py`         | `VisionEncoder` + `VisionModel` — Qwen3VL ViT 实现                                    |
| `nanodeployvl/vision/processor.py`       | `ImageProcessor` — 封装 HF `Qwen3VLProcessor`                                         |

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
