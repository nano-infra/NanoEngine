# Bug Fixes: Qwen3.5 GDN State Migration & DeepSeek v3 Tokenizer

> 修复日期：2026-02-25

## 概述

本次修复解决了两个关键问题：

1. **Qwen3.5 PD 分离版本 GDN 状态传输不完整**：导致 decode 端输出胡言乱语
2. **DeepSeek v3 tokenizer 解码问题**：输出中出现 `Ġ` 字符而非空格

______________________________________________________________________

## Bug #1: Qwen3.5 PD 分离版本 GDN 状态传输不完整

### 问题描述

在 Qwen3.5-MoE 模型的 PD（Prefill-Decode）分离模式下，decode 端输出胡言乱语，而非 PD 分离模式输出正常。

**症状**：

- Qwen3.5 非 PD 分离版本：✅ 输出正确
- Qwen3.5 PD 分离版本：❌ 输出胡言乱语
- Qwen3 PD 分离版本：✅ 输出正确（Qwen3 没有 GDN 层）

### 根本原因

在 `nanodeploy/context/cache.py` 的 `migrate()` 方法中，GDN 状态传输循环使用了 `self.num_hidden_layers` 来遍历层数。但对于 Qwen3.5-MoE 混合注意力模型：

- `self.num_hidden_layers` = **15**（只有 full_attention 层需要 KV cache）
- GDN 状态实际分配了 **60** 层（所有模型层，包括 GDN 层）

**代码位置**：`nanodeploy/context/cache.py` 的 `migrate()` 方法

**问题代码**：

```python
for layer_idx in range(self.num_hidden_layers):  # ❌ 只有 15 层
    gdn_assigns[engine_id][peer_alias].append(
        (layer_idx, remote_state_slot, local_state_slot)
    )
```

**影响**：

- 只有 layer 0~14 的 GDN 状态被 RDMA 传输到 decode 端
- Layer 15~59 中的 33 个 GDN 层的 conv state 和 recurrent state **完全丢失**（全是零）
- Decode 端因为缺少大量 GDN 状态而产生胡言乱语

**为什么 Qwen3 PD 正确**：Qwen3 没有 GDN 层（纯 full attention），不需要传输 GDN 状态，所以 `num_hidden_layers` 的值无关紧要。

**为什么 Qwen3.5 非 PD 正确**：非 PD 模式下 prefill 和 decode 在同一个 GPU 上，GDN 状态直接在本地内存中，不需要 RDMA 传输。

### 修复方案

将循环改为使用 `self.gdn_recurrent_states.shape[0]`（=60），即 GDN 状态张量的实际层维度，而不是 `self.num_hidden_layers`（=15）。

**修改文件**：`nanodeploy/context/cache.py`

**修复代码**：

```python
# Use actual GDN state layer count (all model layers),
# NOT self.num_hidden_layers (which is only KV cache layers
# for mixed attention models like Qwen3.5-MoE).
num_gdn_layers = self.gdn_recurrent_states.shape[0]
for layer_idx in range(num_gdn_layers):  # ✅ 使用实际的 GDN 层数（60）
    gdn_assigns[engine_id][peer_alias].append(
        (layer_idx, remote_state_slot, local_state_slot)
    )
```

### 相关代码位置

- `nanodeploy/context/cache.py`：`migrate()` 方法（约第 570 行）
- `nanodeploy/worker/model_runner.py`：`preallocate_kvcache()` 方法中设置 `num_hidden_layers`（约第 293-299 行）

______________________________________________________________________

## Bug #2: DeepSeek v3 Tokenizer 解码问题（Ġ 字符乱码）

### 问题描述

使用 DeepSeek v3 模型时，解码输出中出现 `Ġ` 字符（U+0120）而非正常的空格，例如：

```
Prompt: 'What is 1+1?'
Completion: "TheĠsumĠofĠ1Ġ+Ġ1ĠisĠ**2**.ĠLetĠmeĠknow..."
```

期望输出：

```
"The sum of 1 + 1 is **2**. Let me know..."
```

### 根本原因

这是 **transformers 5.3.0.dev0 的 `LlamaTokenizer` bug**：

1. **DeepSeek v3 的 tokenizer.json** 使用 **ByteLevel** 编码（空格 = `Ġ`，U+0120）
2. **`AutoTokenizer.from_pretrained()`** 加载后实例化为 `LlamaTokenizer`
3. **`LlamaTokenizer.__init__`** 硬编码了 **Metaspace** pre-tokenizer/decoder（空格 = `▁`，U+2581）
4. **`convert_to_native_format`** 从 `tokenizer.json` 只提取了 vocab/merges，但 decoder 被 `__init__` 覆盖为 Metaspace 版本
5. **结果**：
   - **编码产生错误的 token ID**（`"sum"` → ID 5674 而非 `"Ġsum"` → ID 2595）
   - **解码不转换 `Ġ` 为空格**

**代码位置**：`transformers/src/transformers/models/llama/tokenization_llama.py`

**问题代码**：

```python
class LlamaTokenizer(TokenizersBackend):
    def __init__(self, ...):
        # ❌ 硬编码 Metaspace pre-tokenizer
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", ...
        )
        # ❌ 硬编码 Metaspace decoder
        sequence = [
            decoders.Replace("▁", " "),  # 期望 ▁，但 tokenizer.json 用的是 Ġ
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]
        self._tokenizer.decoder = decoders.Sequence(sequence)
```

**影响范围**：

- 不仅是显示问题！prompt 编码的 token ID 也是错的
- 简单 prompt（如 "What is 1+1?"）可能恰好没有暴露问题
- 复杂 prompt 可能会导致模型行为异常

### 修复方案

将所有 `AutoTokenizer.from_pretrained()` 替换为 `PreTrainedTokenizerFast.from_pretrained()`，它会直接加载 `tokenizer.json` 并正确使用 ByteLevel decoder。

**修改文件**：

1. `nanodeploy/engine/llm_engine.py`（核心引擎）
2. `examples/non_disagg.py`
3. `examples/disagg.py`
4. `examples/mixed_dataset.py`

**修复代码**：

```python
# 修改前
from transformers import AutoTokenizer
self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

# 修改后
from transformers import PreTrainedTokenizerFast
self.tokenizer = PreTrainedTokenizerFast.from_pretrained(config.model)
```

**验证**：

- `AutoTokenizer.from_pretrained` → `LlamaTokenizer` → ❌ `"Thesumof1+1is2"`
- `PreTrainedTokenizerFast.from_pretrained` → `TokenizersBackend` → ✅ `"The sum of 1 + 1 is 2"`

### 相关代码位置

- `nanodeploy/engine/llm_engine.py`：第 13、48 行
- `examples/non_disagg.py`：第 15、35 行
- `examples/disagg.py`：第 29、75 行
- `examples/mixed_dataset.py`：第 7、19 行

______________________________________________________________________

## 测试验证

### Qwen3.5 GDN 状态传输修复

**测试命令**：

```bash
python examples/disagg.py --log_level INFO \
    --model /models/models--Qwen--Qwen3.5-397B-A17B-FP8 \
    --ray_address 10.102.97.179:7078 \
    --attention_dp 8 --ffn_ep 8 \
    --kvcache_block_size 256 --max_num_seqs 32 \
    --prefill.master_address 10.102.97.183:6006 \
    --decode.master_address 10.102.97.179:6006 \
    --decode.loop_count 16 --prompt "Introduce yourself"
```

**预期结果**：✅ 输出正确的自我介绍文本，不再胡言乱语

### DeepSeek v3 Tokenizer 修复

**测试命令**：

```bash
python examples/non_disagg.py --ray_address 10.102.97.179:7078 \
    --master_address 10.102.97.179:6006 \
    --model /models/deepseek-v3 \
    --attention_dp 8 --ffn_ep 8 \
    --kvcache_block_size 64 \
    --max_tokens 64 --temperature 0.1 \
    --prompt "What is 1+1?" --max_num_seqs 16
```

**预期结果**：✅ 输出中空格正常显示，不再出现 `Ġ` 字符

______________________________________________________________________

## 技术细节

### GDN 状态传输机制

在 PD 分离模式下，prefill 端需要将以下状态迁移到 decode 端：

1. **KV Cache**：只传输 full_attention 层的 KV cache（15 层）
2. **GDN 状态**：需要传输所有层的 GDN 状态（60 层）
   - `gdn_conv_states`：形状 `(num_layers, batch_size, ...)`
   - `gdn_recurrent_states`：形状 `(num_layers, batch_size, ...)`

**关键点**：`CacheContext.num_hidden_layers` 只表示需要 KV cache 的层数，不表示模型的总层数。对于混合注意力模型（如 Qwen3.5-MoE），应该使用实际的状态张量形状来确定层数。

### Tokenizer 加载机制

**`AutoTokenizer.from_pretrained()`**：

- 根据 `tokenizer_config.json` 的 `tokenizer_class` 字段选择 tokenizer 类
- 对于 DeepSeek v3，选择 `LlamaTokenizer`
- `LlamaTokenizer` 有自定义 `__init__`，会覆盖 decoder 配置

**`PreTrainedTokenizerFast.from_pretrained()`**：

- 直接加载 `tokenizer.json` 文件
- 使用 tokenizers 库的原生 decoder（ByteLevel）
- 不会覆盖 decoder 配置

**建议**：对于使用 ByteLevel 编码的模型（如 DeepSeek v3、Qwen 系列），优先使用 `PreTrainedTokenizerFast` 以确保正确解码。

______________________________________________________________________

## 相关文件清单

| 文件                              | 修改类型 | 说明                      |
| --------------------------------- | -------- | ------------------------- |
| `nanodeploy/context/cache.py`     | **修改** | 修复 GDN 状态传输层数计算 |
| `nanodeploy/engine/llm_engine.py` | **修改** | 修复 tokenizer 加载方式   |
| `examples/non_disagg.py`          | **修改** | 修复 tokenizer 加载方式   |
| `examples/disagg.py`              | **修改** | 修复 tokenizer 加载方式   |
| `examples/mixed_dataset.py`       | **修改** | 修复 tokenizer 加载方式   |

______________________________________________________________________

## 后续建议

1. **统一 tokenizer 加载方式**：考虑在 `Config` 类或工具函数中统一 tokenizer 加载逻辑，避免在多个地方重复代码
2. **GDN 状态层数检查**：在 `CacheContext` 初始化时添加断言，确保 GDN 状态层数与模型配置一致
3. **测试覆盖**：为 PD 分离模式添加自动化测试，覆盖混合注意力模型（Qwen3.5-MoE）的 GDN 状态传输

______________________________________________________________________

## 参考资料

- [Transformers Tokenizers Documentation](https://huggingface.co/docs/transformers/main/en/tokenizer_summary)
- [HuggingFace Tokenizers Library](https://github.com/huggingface/tokenizers)
- Qwen3.5-MoE 模型配置：`/models/models--Qwen--Qwen3.5-397B-A17B-FP8/config.json`
- DeepSeek v3 tokenizer 配置：`/models/deepseek-v3/tokenizer_config.json`
