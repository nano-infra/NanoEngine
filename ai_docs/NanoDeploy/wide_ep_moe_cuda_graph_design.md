# WideEP (Attention TP + FFN EP) 混合并行与 CUDA Graph 兼容设计

本文档记录了在 NanoDeploy 中实现同层异构并行（Attention 采用 Tensor Parallelism, FFN 采用 Expert Parallelism），并在单机或多机环境下完全兼容 CUDA Graph 捕获的设计与实现细节。此方案主要针对 Qwen3 / Qwen3.5 / DeepseekV2 等 MoE 模型。

## 1. 背景与目标

在 MoE 模型的推理中，将 Attention 层配置为 TP（Tensor Parallelism），将 FFN / MoE 层配置为 EP（Expert Parallelism）可以最大化显存利用率和计算效率（即 WideEP 架构）。例如：配置 `attn_tp=2, ffn_ep=8`。

然而，传统的并行架构在 TP 和 EP 的跨界转换时，或者在非满配 EP（如单卡 EP=1）时，往往会触发 CPU 端的隐式同步（例如动态分配 Buffer、数据分发等）。CUDA Graph 捕获阶段严禁任何形式的 CPU 侧 Stream 同步或基于数据的动态控制流分支。

**核心目标：**
在维持 `attn_tp` 和 `ffn_ep` 可自由组合的前提下：

1. 实现无分支、无 CPU 同步的并行度转换。
2. 确保全链路兼容 CUDA Graph 捕获（即 Batch Size、Buffer Size 在捕获期严格固定对齐）。
3. 保持底层对 GPU-Native Kernel（如 DeepEP / DeepGEMM）的完美接驳。

______________________________________________________________________

## 2. 架构设计：转换层 (Transition Layers)

为了隔离 Attention 和 FFN 的并行域边界，我们在 `nanodeploy/layers/parallelism_transition.py` 中引入了两个轻量级的路由组件。这两个组件作为标准的 `nn.Module` 按需插入到 `DecoderLayer` 的前向计算图中。

### 2.1. AttnToFfnTransition (Scatter 阶段)

**触发位置**：Attention (带有 TP All-reduce 和 Post-LN) 之后，MoE (EP) 之前。
**核心逻辑**：
Attention 阶段由于 TP 的 All-reduce 属性，同一 TP 组内的所有 GPU 持有完全相同的全量 `hidden_states`。为了进入 FFN 的 EP 模式，每个 GPU 需要只处理自己负责的部分 Token。

- 该层利用 `chunking`（按序列/Batch 维度切分），每个 `attn_tp_rank` 直接截取总 Batch 中的 `1 / attn_tp` 份额。
- **Batch Padding (关键)**：为了兼容 CUDA Graph，Batch Size 必须是确定的。当输入 `batch_size` 无法被 `attn_tp` 整除时（极典型的如 Graph Capture 期间的 `bs=1, tp=2`），该层会**自动向上 Padding 零元素**对齐至 `attn_tp` 的倍数，再进行 Chunk。

### 2.2. FfnToAttnTransition (Gather 阶段)

**触发位置**：MoE (EP Combine 之后)，即 FFN 结束，下一个 Attention 层或输出头之前。
**核心逻辑**：
FFN 阶段结束后，各个 GPU 仅包含被处理过的属于自己的那部分 Token 输出。为了进入下一层的 Attention（需要完整的 Sequence / Batch），此层执行反向操作。

- 在当前的 `attn_tp_group` 内执行一次 `AllGather` 操作，将切片的数据原样拼回。
- 若前置的 `AttnToFfnTransition` 发生了 Padding 填充，这里会利用预存的 Original Batch Size 对 Padding 进行 **Slice** 切除，恢复原始的 Token 数量和真实维度。

> **向后兼容性**：在不需要转化的场景（如纯 TP 或纯单一卡），这两个层会完美退化为 `nn.Identity()`（开销为零）。

______________________________________________________________________

## 3. 具体修改与整合链路

整个链路的整合不侵入底层的线性算子计算逻辑代码。

### 3.1. Linear 层的 `parallel_context` 支持

在此之前，底层的 RowParallel 和 ColumnParallel Linear 层在初始化和计算时，强耦合了 `get_dist_context().attn_tp_*` 通信组。
为此，引入 `parallel_context`（`"attn"` 或 `"ffn"`）标签：

- 模型创建时，若该 Linear 层属于 Attention 则默认走 `"attn"`。
- 若属于 FFN/MLP（例如 MoE 内部的 expert 投影，或 Shared Expert 的 down proj），传入 `parallel_context="ffn"`。
- Linear 基类会根据该标签，正确绑定并使用 `ffn_tp_group` 作为 All-Reduce 的通信群组。

```python
# 例如 Qwen3MoeMLP 内部:
self.gate_up_proj = get_backend().get_merged_column_parallel_linear(
    ..., parallel_context="ffn"
)
```

### 3.2. DecoderLayer 的拓扑包裹

在各类模型的 `DecoderLayer` 中（如 Qwen3, Qwen3.5, DeepseekV2 等）：

```python
# 初始化：
if attn_tp > 1 and ffn_ep > 1:
    self.attn_to_ffn = AttnToFfnTransition()
    self.ffn_to_attn = FfnToAttnTransition(scatter_layer=self.attn_to_ffn)

# Forward 链路：
hidden_states = self.post_attention_layernorm(hidden_states, residual)
hidden_states = self.attn_to_ffn(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = self.ffn_to_attn(hidden_states)
```

### 3.3. Qwen3.5 线性注意力 GatedDeltaNet 的修复

Qwen3.5 采用了一种线性注意力组件（`GatedDeltaNet`），这是一个特例情况：

- 该算子的所有 Input Projections 本身是 **Replicated**（各个卡算出完全一样的输入和结果）。
- 该算子内部不进行基于 TP 的 Tensor 切分缩减，只做数学上的全局映射。
- 因此它的 Output Projection 曾经错误使用了 `RowParallelLinear`（这就导致了 `attn_tp=2` 时，输入依然是 8192 满维度，但由于 `RowParallel` 的行为被错误地当成 4096 去匹配 `matmul` 形状而报错）。
- **修复**：将其 `out_proj` 修改为 `ReplicatedLinear`。如此一来，即使开着 TP=2，线性注意力也仅仅是在每个局部卡上算全量并安全进入后续网络，无额外同步消耗。

______________________________________________________________________

## 4. 结论与总结

1. **兼容性**：新链路完美兼容 `batch_size < tp_size` 及不整除场景，使得 CUDA Graph Capture 在小批量 decode (如 batch=1) 时天然免疫 IndexError 和 Cuda-Stream-Sync-Violation 问题。
2. **正交扩展**：彻底解耦了 TP 和 EP 的算子绑定逻辑。目前可自由在配置文件中组合 `attn_dp`, `attn_tp`, `ffn_ep`, `ffn_tp` 拓扑矩阵。
3. **架构侵入极小**：封装极好地包裹在 Transition 模块与 Linear Mixin 层，所有主流和新增的模型只做寥寥数行的包裹改动即可支持庞大集群的并行策略。
