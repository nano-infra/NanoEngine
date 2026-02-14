# DeepSeek-V3 Support for NanoDeploy

> 开发完成日期：2026-02-14

## 概述

在 NanoDeploy 中完成了 DeepSeek-V3 模型的全面支持，包括 FP8 量化推理、MLA (Multi-head Latent Attention)、MoE (Mixture of Experts)、Prefill-Decode 分离（PD 分离）等核心功能。非分离模式和 PD 分离模式均已验证输出正确。

______________________________________________________________________

## 修改文件清单

| 文件                                                       | 改动类型      | 说明                                                                                         |
| ---------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------- |
| `nanodeploy/models/deepseek_v2.py`                         | **新增/重写** | DeepSeekV2/V3 模型实现，包含 MLA 注意力、MoE 路由、FP8 MLP                                   |
| `nanodeploy/worker/model_runner.py`                        | **修改**      | 注册 DeepseekV3 架构、MLA 模式检测、FlashMLA metadata 管理、CUDA Graph 适配、block_size 同步 |
| `nanodeploy/worker/loader.py`                              | **修改**      | 权重加载：expert 合并、kv_b_proj 分解、packed modules、FP8 scale、MTP 层跳过                 |
| `nanodeploy/layers/attention.py`                           | **修改**      | FlashMLAImpl decode 路径使用预计算 metadata；MLA prefill 转移到模型层处理                    |
| `nanodeploy/layers/rotary_embedding.py`                    | **修改**      | Yarn-aware RoPE 实现，手动缓存替代 lru_cache                                                 |
| `nanodeploy/llm_component.py`                              | **修改**      | 在 `__init__` 中注册 NanoCtrl，支持直接使用 LLMComponent 的 PD 分离                          |
| `nanodeploy/server/engine_server.py`                       | **修改**      | 适配 LLMComponent 注册逻辑变更                                                               |
| `nanodeploy/config.py`                                     | **修改**      | 优先使用内置 transformers config，避免 Ray actor 中 `transformers_modules` 导入失败          |
| `nanodeploy/engine/llm_engine.py`                          | **修改**      | 添加 `Sequence.set_block_size()` 调用                                                        |
| `NanoSequence/nanosequence/csrc/sequence/sequence.h`       | **修改**      | `block_size` 从 constexpr 改为 mutable static，添加 `set_block_size()`                       |
| `NanoSequence/nanosequence/csrc/sequence/sequence.cpp`     | **修改**      | 定义 static 成员                                                                             |
| `NanoSequence/nanosequence/csrc/bind/sequence_binding.cpp` | **修改**      | 暴露 `set_block_size` 到 Python                                                              |
| `nanodeploy/csrc/scheduler/scheduler.cpp`                  | **修改**      | 使用 `kvcache_block_size_` 替代 `Sequence::block_size`                                       |

______________________________________________________________________

## 关键技术实现

### 1. MLA (Multi-head Latent Attention)

DeepSeek-V3 使用 MLA 替代标准 GQA，将 KV 投影压缩到低秩空间：

- **KV 压缩**：`kv_a_proj_with_mqa` 将隐藏状态压缩到 `kv_lora_rank=512` 维 + `qk_rope_head_dim=64` 维（用于 RoPE），共 576 维
- **Absorbed Attention (Decode)**：将 `kv_b_proj` 的权重吸收到 Q 侧（BMM）和输出侧（BMM），避免在 decode 阶段展开 KV。使用 FlashMLA kernel 直接在压缩的 KV cache 上做注意力
- **Non-Absorbed Attention (Prefill)**：显式展开 Q/K/V 到 per-head 表示（`head_dim_qk=192`, `head_dim_vo=128`），使用 FA3 (`flash_attn_interface.flash_attn_varlen_func`) 计算，因为 FlashMLA 的 dense prefill kernel 仅支持 SM100 (Blackwell)

```
KV Cache 形状: (1, num_layers, num_blocks, block_size, 1, 576)
                ↑ kv_count=1（MLA 只有一份 compressed KV）
```

**文件**: `nanodeploy/models/deepseek_v2.py` — `DeepseekV2Attention`

### 2. RoPE 维度格式转换

**问题**：DeepSeek-V3 checkpoint 使用 `rope_interleave=True`（默认），投影权重产生的 PE 维度是交错格式 `(d0,d1), (d2,d3), ...`，但 NanoDeploy 的 `apply_rotary_emb` 期望半旋转格式 `(d0,d32), (d1,d33), ...`。

**解决方案**：在 `deepseek_v2.py` 中实现 `_interleaved_to_half` 函数，在 RoPE 应用前将 `q_pe` 和 `k_pe` 从交错格式转换为半旋转格式：

```python
def _interleaved_to_half(x):
    """Convert interleaved RoPE layout to half-rotation layout."""
    x2 = x.unflatten(-1, (-1, 2))            # (..., dim/2, 2)
    return torch.cat([x2[..., 0], x2[..., 1]], dim=-1)  # (..., dim)
```

### 3. Yarn RoPE

DeepSeek-V3 使用 Yarn (Yet Another RoPE extensioN) 来扩展上下文窗口：

- 实现了 `YarnRotaryEmbedding`，包含 `yarn_find_correction_range`、`yarn_linear_ramp_mask`、`yarn_get_mscale`
- 将 `rope_scaling` 字典参数传递给 `get_rope`，使用手动字典缓存（因为 dict 不可哈希，无法用 `@lru_cache`）
- 支持从 `config.rope_parameters`（内置 transformers config）或顶层属性（自定义 config）读取参数

**文件**: `nanodeploy/layers/rotary_embedding.py`

### 4. MoE 路由 (DeepSeek-V3 特有)

- **Sigmoid 评分**：替代 softmax，使用 `torch.sigmoid` 计算 expert scores
- **e_score_correction_bias**：在 sigmoid 后加上偏置校正
- **Group-limited Top-K**：将 experts 分组（`n_group=8`），先选 top groups，再在组内选 top experts
- **Shared Expert**：每层有 1 个 shared expert，单独计算后与 routed expert 输出相加

**文件**: `nanodeploy/models/deepseek_v2.py` — `DeepseekV2MoE`

### 5. FP8 权重加载

- **Expert 合并**：将 per-expert 权重 `experts.{i}.gate_proj.weight` / `up_proj.weight` / `down_proj.weight` 合并为 3D 张量 `gate_up_proj` / `down_proj`，同时合并对应的 `weight_scale_inv`
- **kv_b_proj 分解**：将 `kv_b_proj.weight` 分解为 `kc`（BMM 权重，shape `[nope_size, num_heads, kv_lora_rank]`）和 `vc`（BMM 权重，shape `[v_head_size, num_heads, kv_lora_rank]`）
- **Packed Modules**：处理 `gate_proj` + `up_proj` → `gate_up_proj` 的打包，包括 shared expert 的 FP8 权重及 scale
- **MTP 层跳过**：DeepSeek-V3 有 Multi-Token Prediction 层（layer index >= `num_hidden_layers`），在加载时跳过

**文件**: `nanodeploy/worker/loader.py`

### 6. FlashMLA 与 CUDA Graph

- **Decode**：使用 `flash_mla.flash_mla_with_kvcache` 在压缩的 KV cache 上直接计算注意力
- **Metadata 预计算**：`tile_scheduler_metadata` 和 `num_splits` 在 `model_runner.py` 的 `prepare_decode` 中预计算，存入 context，供 FlashMLA 使用
- **CUDA Graph 兼容**：预计算 metadata 存储在固定 buffer 中，CUDA graph replay 时通过 `copy_` 更新
- **num_splits 裁剪**：`FlashMLAImpl.forward` 中将 `num_splits` 裁剪到 `[:batch_size + 1]` 以匹配实际 batch size

**文件**: `nanodeploy/layers/attention.py`, `nanodeploy/worker/model_runner.py`

### 7. KV Cache Block Size 统一

**问题**：Python 端 `kvcache_block_size=64`，但 C++ 端 `Sequence::block_size` 硬编码为 256，导致 `slot_mapping` 计算错误，prefill 时 assert 失败。

**解决方案**：

1. C++ `Sequence::block_size` 改为 mutable static，添加 `set_block_size()` 方法
2. Python 端在 `ModelRunner.__init__` 和 `LLMEngine.__init__` 中调用 `Sequence.set_block_size(config.kvcache_block_size)`
3. C++ scheduler 使用 `kvcache_block_size_`（从 Python config 传入）替代 `Sequence::block_size`

### 8. PD 分离 NanoCtrl 注册

**问题**：`_register_with_nanoctrl()` 仅在 `engine_server.py` 中调用，直接使用 `LLMComponent.as_remote()` 的 disagg 脚本不走 engine_server，导致引擎未注册到 NanoCtrl，decode 端无法发现 prefill 端的 RDMA 地址，KV cache 迁移失败。

**解决方案**：在 `LLMComponent.__init__` 末尾直接调用 `_register_with_nanoctrl()`。`engine_server.py` 路径在绑定 P2P socket 后会重新注册以更新 `p2p_port`。

**文件**: `nanodeploy/llm_component.py`, `nanodeploy/server/engine_server.py`

### 9. Config 加载兼容性

**问题**：Ray actor 中使用 `AutoConfig.from_pretrained(..., trust_remote_code=True)` 会触发 `ModuleNotFoundError: No module named 'transformers_modules'`。

**解决方案**：先尝试 `trust_remote_code=False`（使用内置 transformers 的 `DeepseekV3Config`），仅在 `ValueError` 时 fallback 到 `trust_remote_code=True`。

**文件**: `nanodeploy/config.py`

______________________________________________________________________

## 关键 Bug 修复记录

| #   | 现象                                        | 根因                                                          | 修复                             |
| --- | ------------------------------------------- | ------------------------------------------------------------- | -------------------------------- |
| 1   | RoPE 未生效                                 | `rotary_emb` 返回值未写回 `query_states`/`key_states`         | 添加赋值语句                     |
| 2   | `ModuleNotFoundError: transformers_modules` | Ray actor 中 trust_remote_code 导入失败                       | 优先使用内置 config              |
| 3   | `AttributeError: rope_theta`                | 内置 DeepseekV3Config 的 rope 参数在 `rope_parameters` 字典内 | 兼容两种读取方式                 |
| 4   | `TypeError: unhashable type: dict`          | `rope_scaling` dict 传给 `@lru_cache`                         | 改用手动字典缓存                 |
| 5   | `ModuleList has no attribute 61`            | MTP 层（layer >= num_hidden_layers）未跳过                    | 添加层索引检查                   |
| 6   | `FlashAttention head dimension > 256`       | Absorbed MLA prefill 的 head_dim=576 超限                     | 改用 non-absorbed path + FA3     |
| 7   | `SM100 only kernel`                         | FlashMLA dense prefill kernel 仅支持 Blackwell                | 改用 FA3                         |
| 8   | Decode 输出乱码                             | RoPE 维度格式不匹配（interleaved vs half-rotation）           | 添加 `_interleaved_to_half` 转换 |
| 9   | `num_splits shape mismatch`                 | CUDA graph 中 num_splits buffer 尺寸不匹配                    | 裁剪到 `[:batch_size + 1]`       |
| 10  | `assert slot_mapping.numel() == N`          | C++ block_size=256 vs Python block_size=64                    | 统一为可配置 block_size          |
| 11  | PD 分离输出乱码                             | Engine 未注册到 NanoCtrl，RDMA 地址未知                       | 在 LLMComponent.__init__ 中注册  |

______________________________________________________________________

## 测试验证

### 非分离模式 (non-disagg)

```bash
python examples/deepseek_v3_non_disagg.py
```

- ✅ Prefill 正确
- ✅ Decode 输出连贯
- ✅ CUDA Graph 模式正常

### PD 分离模式 (disagg)

```bash
python examples/deepseek_v3_disagg.py
```

- ✅ NanoCtrl 注册成功
- ✅ Prefill 在 node 183 执行
- ✅ KV Cache RDMA 迁移到 node 179
- ✅ Decode 输出正确连贯
- ✅ 性能指标：Prefill ~4800 tok/s, Decode ~210 tok/s

### 示例输出

输入：一篇约 2700 token 的中文作文，要求打分

输出（256 tokens）：

> 这篇作文以细腻的笔触描绘了一个雨夜停电的场景，通过生动的细节描写和真挚的情感表达，展现了家庭温暖与幸福的主题。以下是对这篇作文的评分及评语：
>
> **评分：95/100**
>
> 1. **内容与主题（30/30）：** 作文紧扣"雨夜的暖光"这一主题...
> 2. **结构与逻辑（25/25）：** 文章结构清晰，层次分明...
> 3. **语言与表达（25/25）：** 语言优美，描写细腻...
> 4. **创意与深度（15/20）：** 作文通过停电这一日常生活中的小插曲...

______________________________________________________________________

## 配置参考

```python
Config(
    model="/models/deepseek-v3",
    enforce_eager=False,          # 启用 CUDA Graph
    attention_dp=8,
    attention_sp=1,
    attention_tp=1,
    ffn_dp=1,
    ffn_ep=8,                     # 8 卡 Expert Parallelism
    ffn_tp=1,
    kvcache_block_size=64,        # MLA 专用 block size
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    max_num_batched_tokens=4096,
    # PD 分离需要额外配置：
    mode="prefill" / "decode",
    nanoctrl_address="10.102.97.179:3000",
)
```

______________________________________________________________________

## 架构要点

```
DeepseekV3ForCausalLM
├── model (DeepseekV2Model)
│   ├── embed_tokens (VocabParallelEmbedding)
│   └── layers (ModuleList × 61)
│       ├── DeepseekV2DecoderLayer (dense MLP, layers 0-2)
│       │   ├── self_attn (DeepseekV2Attention)
│       │   │   ├── q_proj → q_a_proj + q_a_layernorm + q_b_proj
│       │   │   ├── kv_a_proj_with_mqa + kv_a_layernorm
│       │   │   ├── kc_bmm / vc_bmm (absorbed from kv_b_proj)
│       │   │   ├── rotary_emb (YarnRotaryEmbedding)
│       │   │   └── attn_fwd (Attention with FlashMLAImpl)
│       │   └── mlp (DeepseekV2MLP: gate_up_proj + down_proj)
│       └── DeepseekV2MoEDecoderLayer (MoE, layers 3-60)
│           ├── self_attn (same as above)
│           └── mlp (DeepseekV2MoE)
│               ├── gate (sigmoid + e_score_correction_bias + group top-k)
│               ├── shared_experts (DeepseekV2MLP × 1)
│               └── experts (DeepseekV2MLP × 256, EP across 8 GPUs)
└── lm_head (ParallelLMHead)
```
