# Qwen3 MoE C++ Migration Progress

## Overview

本文档记录 Qwen3 MoE 模型向 C++ ModelRunner 引擎迁移的进度。

## 完成状态

### Phase 0: 单卡 FP16 Simplified (当前阶段) ✅ **已完成**

**目标**: 使用最简化的 C++ 实现 (仅依赖 LibTorch) 在单机单卡上跑通 Qwen3 MoE 的推理流程，不依赖 DeepEP/DeepGemm，确保基础逻辑正确。

**已完成工作 (2026-01-11)**:

1. **简化版模型实现** (`csrc/nanodeploy/models/qwen3_moe_simple.h`)

   - 参考 `nanovllm/models/qwen3_moe.py` 的逻辑实现。
   - 移除 DeepEP 和 DeepGemm 依赖，使用 `torch::topk`, `index_select`, `index_add_` 实现 MoE 路由。
   - `Qwen3MoeSparseMoeBlock`: 实现了专家路由和计算。
   - `Qwen3MoeMLP`: 标准 MLP 实现。

2. **ModelRunner 集成** (`csrc/nanodeploy/worker/model_runner.cpp`)

   - 切换包含 `qwen3_moe_simple.h`。
   - 调整权重加载逻辑：直接加载到 `std::vector<std::shared_ptr<Qwen3MoeMLP>> experts_` 中。
   - 移除 DeepEP buffer 初始化。

3. **测试工具更新** (`tests/test_qwen3_moe_runner.cpp`)

   - 支持命令行参数解析，适配 `tools/run_qwen_chat.py`。
   - 实现了与 Python Chat 脚本的交互接口。

4. **Correctness Verification (Debug Success)**

   - **问题**: 初始版本存在 "Dirty Generation" (重复生成 `...the...the`)。
   - **原因**: `Qwen3MoeDecoderLayer::forward` 中 Residual Logic 实现错误。当 `residual` 未定义（首层或简化调用）时，代码逻辑导致层计算被跳过或未正确累加。
   - **修复**: 统一 Residual Stream 逻辑，将 `hidden_states` 作为累加流 (`accumulator`)，确保 `Pre-Norm -> Attn -> Add -> Post-Norm -> MoE -> Add` 流程正确执行。
   - **结果**:
     - Prompt: "introduce yourself"
     - Output: "Hello! I'm Qwen, a large-scale language model developed by Alibaba Cloud. I'm designed to assist..."
     - 验证了单卡 FP16 MoE 通路的正确性。

### Phase 1: 单卡 FP16 (DeepEP/DeepGemm 集成) ⏳ 待启动

**计划**:

- 恢复 `qwen3_moe.h` (完整版实现)。
- 集成 DeepEP (Expert Parallel 通信库) 的单卡模拟模式或多卡模式。
- 集成 DeepGemm 以提升 GEMM 性能。

### Phase 2: 单卡 FP8 ⏳ 待验证

**计划**:

- 加载 FP8 权重和 scales
- 切换 `Qwen3MoeSparseMoeBlock` 使用 FP8 路径
- 验证 FP8 内核执行和精度

### Phase 3: 分布式 DP + EP ⏳ 待实现

**计划**:

- 多卡环境下的 DeepEP 通信
- 专家分片和路由
- 跨节点验证

## 关键设计与实现细节 (Phase 0)

为了快速定位问题，我们实现了 `qwen3_moe_simple.h`。

### 1. 路由逻辑 (Routing)

不使用 DeepEP 的 `dispatch` / `combine`，而是使用 PyTorch 原语模拟：

- **Gating**: `torch::topk` 选出 TopK 专家。
- **Dispatch**: 使用 `at::_unique` 找出当前 batch 激活的专家列表。
- **Compute**: 循环遍历激活的专家，使用 `torch::where` 找出对应的 token indices，执行 `expert->forward()`。
- **Combine**: 使用 `index_add_` 将专家输出累加回 `final_hidden_states`。

### 2. 权重存储

- 专家权重存储为 `std::vector<std::shared_ptr<Qwen3MoeMLP>> experts_`。
- 每个 `Qwen3MoeMLP` 包含独立的 `gate_up_proj` (MergedColumnParallelLinear) 和 `down_proj` (RowParallelLinear)。
- 这种结构方便单卡调试和逐个加载权重，但在大规模分布式场景下效率不如 DeepEP 的平铺布局。

### 3. Layer Implementation

- **Decoder Layer**: 遵循 Pre-Norm 结构。
  ```cpp
  normed = norm(hidden);
  hidden = hidden + attn(normed);
  normed = norm(hidden);
  hidden = hidden + moe(normed);
  ```
  *注意*: 即使是 "Simple" 版本，也必须严格遵守 Transformer 的 Residual Add 顺序。

## 验证方法

使用 Python 脚本驱动 C++ Runner 进行端对端测试：

```bash
python ./tools/run_qwen_chat.py \
    --model_path /models/Qwen3-30B-A3B-Instruct-2507 \
    --exe_path ./build/bin/test_qwen3_moe_runner \
    --agent_port 8888 --prompt "introduce yourself"
```
