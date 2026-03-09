# NanoDeployVL — 视觉-语言推理引擎

> 提交: `7f6e0623` (init vl) + `b484e71c` (update vl)
> 分支: `feature/nanodeploy_vl`
> 日期: 2026-03-09

## 概述

新增 **NanoDeployVL** 子项目，为 NanoInfra 提供视觉-语言（VL）多模态推理能力。当前支持 **Qwen3.5-MoE** 系列 VLM 模型（35B-A3B / 397B-A17B-FP8）。

核心设计思路：在 NanoDeploy 已有的 LLM 推理引擎之上，叠加一个**独立的 Vision Encoder 前端**，通过 side-channel 将视觉 embedding 注入到 LLM 的 prefill 阶段。

## 架构

```
┌──────────────────┐
│     VLEngine     │  VL 编排层
│                  │
│  ┌────────────┐  │
│  │ Processor  │  │  HF Qwen3VLProcessor — 图文分词 + 图片预处理
│  └────────────┘  │
│  ┌────────────┐  │
│  │  Vision    │  │  独立 ViT，运行在 driver GPU (cuda:0)
│  │  Encoder   │  │  加载 model.visual.* 权重
│  └────────────┘  │
│  ┌────────────┐  │
│  │ LLMEngine  │  │  NanoDeploy scheduler + Ray workers
│  │  (text)    │  │  调度、KV cache、解码
│  └────────────┘  │
└──────────────────┘
```

**数据流**:

1. 用户输入 `messages` + `images` → `ImageProcessor` 通过 HF Processor 分词并将图片转为 `pixel_values`，展开 `<|image_pad|>` 占位符
2. `VisionEncoder` 将 `pixel_values` 编码为 vision embeddings
3. Vision embeddings 通过 `RayExecutor.set_vision_embeds()` 推送到所有 model workers（CPU tensor → CUDA）
4. Workers 在 prefill 阶段 `_inject_vision_embeds()` 用 vision embeddings 替换占位符 token 的 text embedding
5. LLM 正常解码生成

## 新增文件

### NanoDeployVL/ （新子项目）

| 文件                               | 说明                                                                                                                                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pyproject.toml`                   | 项目配置，依赖 nanodeploy、transformers≥4.52.0、Pillow、safetensors                                                                                                                   |
| `nanodeployvl/__init__.py`         | 包入口，导出 VLConfig、VLEngine、VisionEncoder、ImageProcessor                                                                                                                        |
| `nanodeployvl/config.py`           | `VLConfig` — 继承 NanoDeploy `Config`，新增 vision_device / vision_dtype / vision_batch_size 等 VL 参数；自动从 HF config 提取 vision_config 和特殊 token ID                          |
| `nanodeployvl/engine/__init__.py`  | 引擎包入口                                                                                                                                                                            |
| `nanodeployvl/engine/vl_engine.py` | `VLEngine` — VL 推理主引擎，编排 Processor → VisionEncoder → LLMEngine 流程；提供 `generate_vl()` 单条和 `generate_vl_batch()` 批量接口                                               |
| `nanodeployvl/vision/__init__.py`  | 视觉包入口                                                                                                                                                                            |
| `nanodeployvl/vision/encoder.py`   | `VisionEncoder` + `VisionModel` — 独立 Qwen3VL ViT 实现（PatchEmbed → RotaryEmb → VisionBlock × N → PatchMerger），支持 deepstack 中间层特征提取；从 safetensors 加载 `visual.*` 权重 |
| `nanodeployvl/vision/processor.py` | `ImageProcessor` — 封装 HF `Qwen3VLProcessor`，提供 `apply_chat_template()` 和 `process()` 接口                                                                                       |
| `docs/testing.md`                  | 测试文档，包含环境准备、4 种测试用例（纯文本 / 单图 / URL 图 / 多 GPU FP8）、调试指南                                                                                                 |

### NanoDeploy/ （已有文件修改）

| 文件                                           | 改动说明                                                                                                                                                                                   |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `examples/encoder_llm.py`                      | **新增** — VL 推理示例脚本，支持 `--image_path`/ `--image_url`/ `--prompt` 参数                                                                                                            |
| `examples/run_mm.sh`                           | **新增** — 快速运行脚本                                                                                                                                                                    |
| `nanodeploy/engine/llm_engine.py`              | `exit()` 增加 `hasattr` 保护，避免 executor 未初始化时报错                                                                                                                                 |
| `nanodeploy/engine/ray_executor.py`            | **新增** `set_vision_embeds()` / `clear_vision_embeds()` — 通过 `collective_rpc` 广播视觉 embedding 到所有 workers                                                                         |
| `nanodeploy/models/qwen3_5_moe/qwen3_5_moe.py` | `Qwen3_5MoeModel.forward()` 和 `Qwen3_5MoeForConditionalGeneration.forward()` 新增 `inputs_embeds` 可选参数，支持跳过 `embed_tokens` 直接使用外部 embedding                                |
| `nanodeploy/worker/model_runner.py`            | **核心改动** — 新增 `_vision_embeds` side-channel 属性，`set_vision_embeds()` / `clear_vision_embeds()` / `_inject_vision_embeds()` 方法；`run_model` 中 prefill 路径注入 vision embedding |
| `nanodeploy/csrc/scheduler/block_manager.cpp`  | `can_append()` 和 `may_append()` 重构 — 简化块分配逻辑，基于实际 block_table 大小而非纯 dispatched_tokens 计算，修复边界条件                                                               |

## 关键设计决策

1. **Vision Encoder 独立于 LLM Workers**: ViT 在 driver 进程的单 GPU 上运行，LLM 由 Ray workers 处理。避免在每个 worker 上重复加载 vision 权重。
2. **Side-channel 注入**: Vision embedding 通过 Ray RPC 以 CPU tensor 形式广播到 workers，workers 在 prefill 时将其移到 CUDA 并替换占位符 token。
3. **HF Processor 复用**: 直接使用 HuggingFace `Qwen3VLProcessor` 处理图文分词和图像预处理，保持与上游兼容。
4. **Block Manager 修复**: 简化了 `can_append` / `may_append` 的块分配逻辑，使其基于已有 block_table 大小计算增量需求，更加清晰且避免了多模态长序列下的边界问题。

## 后续优化方向

- **MRoPE 支持**: 当前使用 1D positions，预留了 `get_rope_index()` 接口升级为 3D position IDs（temporal, height, width）
- **Vision Encoder 并行化**: 使用 Ray Actor 或 DP 分片并行化 encoder
- **Batch Vision Encoding**: 批量化 ViT 推理（当前逐图编码）
- **流式输出**: `generate_vl()` 当前同步，可扩展为 streaming token generation
- **视频支持**: 框架已预留 `encode_video` 接口，但未做端到端测试
