# test_ep_full.py — EP 分离 VL 推理集成测试 Walkthrough

> Full end-to-end test for the EP (Encoder-Prefill) separated VL pipeline:
> EncoderEngine (ViT) + LLMComponent, connected via RDMA.

______________________________________________________________________

## 架构回顾

```
┌─────────────────────────────┐        RDMA         ┌───────────────────────────────┐
│       EncoderEngine         │ ──────────────────► │        LLMComponent           │
│  (driver process, cuda:0)   │   VisionSlotMeta     │  (Ray worker, 1+ GPU)         │
│                             │                      │                               │
│  ImageProcessor             │                      │  prefill: fetch RDMA embeds   │
│    → pixel_values           │                      │    + inject at image tokens   │
│  VisionEncoder (ViT)        │                      │  decode: autoregressive       │
│    → EmbeddingPool          │                      │                               │
│    → RDMA MR                │                      │  P2P FreeVisionSlots →        │
└─────────────────────────────┘ ◄──────────────────  └───────────────────────────────┘
                                    Action=4 free
```

**数据流**:

1. `ImageProcessor` 预处理图片 → `pixel_values [N_patches, patch_dim]`
2. `VisionEncoder.encode()` → vision embeddings 写入 `EmbeddingPool` + 注册 RDMA MR
3. 返回 `VisionSlotMeta`（slot_idx, num_tokens, hidden_size, RDMA handle）
4. `LLMComponent` 收到附有 slot meta 的 `Sequence`
5. Prefill 阶段：RDMA 拉取 vision embeddings，注入 image token 位置
6. Decode 正常自回归生成
7. 生成结束后，LLM 通过 P2P 消息释放 encoder 侧的 slot

______________________________________________________________________

## Prerequisites

| 服务                      | 说明                                     |
| ------------------------- | ---------------------------------------- |
| Redis `127.0.0.1:6379`    | NanoCtrl 依赖                            |
| NanoCtrl `127.0.0.1:3000` | Engine 注册中心                          |
| Ray cluster               | `ray start --head` 或已有 ray address    |
| dlslime                   | RDMA 传输库                              |
| Model checkpoint          | 含 `vision_config`（如 Qwen3.5-35B-A3B） |

______________________________________________________________________

## 运行方式

```bash
cd NanoDeploy/examples

# 最小运行（合成图片，1 GPU，eager mode）
python test_ep_full.py \
    --model /models/models-Qwen-Qwen3.5-35B-A3B \
    --enforce_eager

# 真实图片
python test_ep_full.py \
    --model /models/models-Qwen-Qwen3.5-35B-A3B \
    --image_path /tmp/test_image.jpg \
    --prompt "Describe this image in detail." \
    --max_tokens 128 \
    --enforce_eager

# 多 GPU
python test_ep_full.py \
    --model /models/models-Qwen-Qwen3.5-35B-A3B \
    --attn_world_size 2 \
    --image_path /tmp/test_image.jpg \
    --enforce_eager
```

______________________________________________________________________

## 代码结构（两个阶段）

### Phase 1: `step_encoder()`

```
EncoderConfig → EncoderEngine
ImageProcessor.apply_chat_template() → prompt_text
ImageProcessor.process()            → pixel_values, image_grid_thw, input_ids
engine.encode(pixel_values, image_grid_thw) → [VisionSlotMeta]
```

关键返回值：

- `slot_metas`: 每个图片一个 meta，含 RDMA 地址
- `token_ids`: 展开了 `<|image_pad|>` 占位符后的 full token list
- `processor`: 用于最终 decode 输出 text

### Phase 2: `step_llm()`

```
LLMComponent.as_remote(config)
Sequence(token_ids) + seq.add_vision_slot(...)
llm.add_request([seq])
llm.generate() → finished sequences
llm.send_free_vision_slots(encoder_id, slots)  # P2P 释放
```

______________________________________________________________________

## 预期时序（1280×960 图片，Qwen3.5-35B-A3B, 1 GPU）

| 阶段                          | 耗时    | 说明                      |
| ----------------------------- | ------- | ------------------------- |
| EncoderEngine 初始化          | ~2s     | 权重加载 + warmup         |
| `encode()`                    | ~0.06s  | ViT forward，4800 patches |
| LLMComponent 初始化           | ~30–60s | Ray actor 启动 + 权重加载 |
| Prefill + Decode (128 tokens) | ~5–15s  | 取决于 GPU 型号           |

> **注意**: encoder encode 时间在 patch_embed 优化后从 ~24s 降至 ~0.06s（见下方优化说明）。

______________________________________________________________________

## Patch Embed 优化

### 问题

`VisionPatchEmbed` 使用 `nn.Conv3d(kernel_size=stride=[temporal, patch, patch])` 做 patch projection。当输入为大图（1280×960 → 4800 patches），`Conv3d` 在 cuDNN 下对这种 batch-of-independent-patches 场景**极度低效**（24秒）。

### 根因

```python
# 原实现
x = x.view(-1, in_channels, temporal_patch_size, patch_size, patch_size)
x = self.proj(x)   # Conv3d: [4800, 3, 2, 14, 14] → slow
x = x.view(-1, embed_dim)
```

`kernel_size == stride`，每个 patch 完全独立，无重叠、无 padding。Conv3d 在这种情况下等价于对展平后的 patch 做 matmul。

### 修复

```python
# 优化后（encoder.py VisionPatchEmbed.forward）
x = x.to(dtype=self.proj.weight.dtype)
w = self.proj.weight.flatten(1)   # [embed_dim, in_ch*t*patch*patch]
return F.linear(x, w, self.proj.bias)  # single GEMM, [N, embed_dim]
```

### 效果（1280×960，4800 patches）

| 指标               | 优化前  | 优化后 | 加速比 |
| ------------------ | ------- | ------ | ------ |
| `patch_embed` 耗时 | 24.089s | 0.004s | ~6000x |
| 总 `encode()` 耗时 | 24.157s | 0.057s | ~424x  |

______________________________________________________________________

## 验证 Checklist

- [ ] `[Encoder] Encoded 1 image(s)` 在 1s 以内
- [ ] `[Encoder] Pool after generation: free=8/8` — slot 被 P2P 正确释放
- [ ] `Seq X: N generated tokens` 输出合理文本（非乱码）
- [ ] `[Encoder] All slots freed via P2P ✓` — P2P 消息正常到达

______________________________________________________________________

## 关键文件

| 文件                                                  | 作用                            |
| ----------------------------------------------------- | ------------------------------- |
| `NanoDeploy/examples/test_ep_full.py`                 | 本测试脚本                      |
| `NanoDeployVL/nanodeployvl/vision/encoder.py`         | ViT 实现 + patch embed 优化     |
| `NanoDeployVL/nanodeployvl/encoder/encoder_engine.py` | EncoderEngine (pool + RDMA)     |
| `NanoDeployVL/nanodeployvl/vision/processor.py`       | HF Processor 封装               |
| `NanoDeploy/nanodeploy/llm_component.py`              | LLM side (prefill/decode + P2P) |
