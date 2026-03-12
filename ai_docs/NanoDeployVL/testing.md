# NanoDeployVL 多模态推理测试文档

## 概述

NanoDeployVL 是基于 NanoDeploy 的视觉-语言（VL）推理引擎，支持 Qwen3.5-MoE 系列多模态模型。

### 支持的模型

| 模型                  | 路径                                          | GPU 需求 | 备注                   |
| --------------------- | --------------------------------------------- | -------- | ---------------------- |
| Qwen3.5-35B-A3B       | `/models/models-Qwen-Qwen3.5-35B-A3B`         | 1-2 GPU  | BF16，小模型，快速验证 |
| Qwen3.5-397B-A17B-FP8 | `/models/models--Qwen--Qwen3.5-397B-A17B-FP8` | 8 GPU    | FP8 量化，生产级       |

### 架构

```
┌──────────────┐
│  VLEngine    │  (VL orchestration layer)
│              │
│  ┌─────────┐ │
│  │Processor│ │  HF Qwen3VLProcessor – tokenize + image preprocess
│  └─────────┘ │
│  ┌─────────┐ │
│  │ Vision  │ │  Standalone ViT on driver GPU (cuda:0)
│  │ Encoder │ │  Loads model.visual.* weights
│  └─────────┘ │
│  ┌─────────┐ │
│  │LLMEngine│ │  NanoDeploy scheduler + Ray workers
│  │(text)   │ │  Handles scheduling, KV cache, decoding
│  └─────────┘ │
└──────────────┘
```

**数据流**:

1. 用户输入 messages + images → `ImageProcessor` 将图片转为 pixel_values，文本分词（展开 `<|image_pad|>` 占位符）
2. `VisionEncoder` 将 pixel_values 编码为 vision embeddings
3. Vision embeddings 通过 Ray 推送到所有 model workers（side-channel）
4. Workers 在 prefill 阶段用 vision embeddings 替换占位符 token 的 text embedding
5. LLM 正常解码生成

## 环境准备

```bash
# 1. 安装 NanoDeploy（如已安装跳过）
cd NanoDeploy
pip install -e .

# 2. 安装 NanoDeployVL 依赖
cd ../NanoDeployVL
pip install -e .

# 3. 确保 transformers 版本足够新（需要 Qwen3VLProcessor 支持）
pip install transformers>=4.52.0

# 4. 其他依赖
pip install Pillow qwen-vl-utils
```

## 测试用例

### 测试 1: 文本-only 模式（验证基础 pipeline）

不使用图片，验证 VLEngine 在纯文本模式下的正确性。

```bash
cd NanoDeploy/examples

python encoder_llm.py \
    --model /models/models-Qwen-Qwen3.5-35B-A3B \
    --attn_world_size 1 \
    --prompt "What is 1+1?" \
    --max_tokens 64 \
    --enforce_eager
```

**预期输出**: 模型正确回答「2」或类似的数学回答。

### 测试 2: 单图推理（核心功能验证）

使用本地图片进行 VL 推理。

```bash
python encoder_llm.py \
    --model /models/models-Qwen-Qwen3.5-35B-A3B \
    --attn_world_size 1 \
    --image_path /path/to/test_image.jpg \
    --prompt "Describe what you see in this image." \
    --max_tokens 256 \
    --enforce_eager
```

**预期输出**: 模型描述图片内容。

### 测试 3: URL 图片推理

使用网络图片进行推理。

```bash
python encoder_llm.py \
    --model /models/models-Qwen-Qwen3.5-35B-A3B \
    --attn_world_size 1 \
    --image_url "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
    --prompt "What is in this image?" \
    --max_tokens 256 \
    --enforce_eager
```

### 测试 4: 多 GPU FP8 模型推理

测试 397B FP8 大模型的多 GPU 推理。

```bash
python encoder_llm.py \
    --model /models/models--Qwen--Qwen3.5-397B-A17B-FP8 \
    --attn_world_size 8 \
    --attention_dp 1 \
    --attention_sp 1 \
    --image_path /path/to/test_image.jpg \
    --prompt "详细描述这张图片中的内容。" \
    --max_tokens 512 \
    --enforce_eager
```

## 调试指南

### 常见问题

1. **HF Processor 加载失败**

   - 确认模型目录包含 `preprocessor_config.json` 和 `tokenizer_config.json`
   - 确认 transformers 版本 >= 4.52.0

2. **Vision encoder 权重加载失败**

   - 检查 safetensors 文件中是否包含 `model.visual.*` 前缀的权重
   - 检查 `model.safetensors.index.json` 中的 weight_map

3. **图片 token 数量不匹配**

   - 错误: `Image token count (N) != image embedding count (M)`
   - 原因: HF Processor 生成的占位符 token 数量和 vision encoder 输出的 embedding 数量不一致
   - 检查: 图片预处理参数（patch_size, temporal_patch_size, spatial_merge_size）是否正确

4. **CUDA OOM**

   - Vision encoder 占用单独的 GPU 内存
   - 可以通过 `--vision_device cpu` 将 encoder 放到 CPU（但会更慢）
   - 大图片会生成大量 vision tokens，考虑缩小图片

### 日志级别调整

```python
from nanodeploy.logging import set_log_level
set_log_level("DEBUG")  # DEBUG, INFO, WARNING, ERROR
```

## 后续优化方向

1. **MRoPE 支持**: 当前使用 1D positions，后续升级为 3D position IDs（temporal, height, width），方法 `get_rope_index()` 已在 VisionEncoder 中预留接口。

2. **Vision encoder 并行**: 当前 encoder 运行在 driver 进程的单 GPU，可使用 Ray Actor 或 DP 分片并行化。

3. **Batch vision encoding**: 当前批量请求中的图片逐个编码，可批量化 ViT 推理。

4. **流式输出**: `generate_vl()` 当前是同步的，可扩展为 streaming token generation。

5. **视频支持**: 框架已预留 video encoding 接口（`encode_video`），但未做端到端测试。

## 文件清单

| 文件                                                      | 说明                                        |
| --------------------------------------------------------- | ------------------------------------------- |
| `NanoDeployVL/nanodeployvl/__init__.py`                   | Package 入口                                |
| `NanoDeployVL/nanodeployvl/config.py`                     | VLConfig 配置类                             |
| `NanoDeployVL/nanodeployvl/vision/encoder.py`             | Vision encoder（独立 ViT 实现）             |
| `NanoDeployVL/nanodeployvl/vision/processor.py`           | HF Processor 封装                           |
| `NanoDeployVL/nanodeployvl/engine/vl_engine.py`           | VL 推理引擎                                 |
| `NanoDeploy/nanodeploy/worker/model_runner.py`            | \[修改\] 添加 vision embedding side-channel |
| `NanoDeploy/nanodeploy/engine/ray_executor.py`            | \[修改\] 添加 set/clear_vision_embeds       |
| `NanoDeploy/nanodeploy/models/qwen3_5_moe/qwen3_5_moe.py` | \[修改\] 添加 inputs_embeds 参数            |
| `NanoDeploy/examples/encoder_llm.py`                      | 测试脚本                                    |
