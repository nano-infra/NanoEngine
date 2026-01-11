# Qwen3 MoE C++ Migration Progress

## Overview

本文档记录 Qwen3 MoE 模型向 C++ ModelRunner 引擎迁移的进度。

## 完成状态

### Phase 1: 单卡 FP16 (基础实现) ✅ 编译通过

**目标**: 使用标准 PyTorch 操作验证 MoE 基础逻辑

**已完成工作**:

1. **配置更新** (`csrc/nanodeploy/core/config.h`)
   - 添加 MoE 相关字段: `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `shared_expert_intermediate_size`
   - 从 HuggingFace `config.json` 自动解析

2. **分布式上下文** (`csrc/nanodeploy/worker/distributed.h/.cpp`)
   - 添加 EP (Expert Parallel) 支持: `ep_rank()`, `ep_world_size()`, `ffn_ep_world_size()`

3. **Qwen3 MoE 模型** (`csrc/nanodeploy/models/qwen3_moe.h`)
   - `Qwen3MoeAttention`: Prefill (SDPA) 和 Decode (FlashInfer) 分离
   - `Qwen3MoeMLP`: 稠密 MLP 层
   - `Qwen3MoeSparseMoeBlock`: 稀疏 MoE 层，集成 DeepEP dispatch/combine
   - `Qwen3MoeDecoderLayer`: 支持 `decoder_sparse_step` 配置
   - `Qwen3MoeModel` / `Qwen3MoeForCausalLM`: 完整模型组装

4. **ModelRunner 集成** (`csrc/nanodeploy/worker/model_runner.h/.cpp`)
   - 支持 MoE 模型加载和推理
   - DeepEP Buffer 初始化
   - MoE 权重加载逻辑 (专家权重堆叠)

5. **测试用例** (`tests/test_qwen3_moe_runner.cpp`)
   - 单卡 MoE 推理测试

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

## 关键设计决策

### 1. DeepGemm ODR 问题

**问题**: DeepGemm 头文件包含全局变量定义，多个编译单元包含会导致链接器报 "multiple definition" 错误。

**解决方案**: 
- Phase 1: 使用标准 PyTorch `torch::bmm` 替代 DeepGemm
- Phase 2+: 将 DeepGemm 调用封装在 `deep_gemm_runner.cpp` 中，其他文件通过接口调用

### 2. Prefill vs Decode 分离

| 阶段 | Attention | MoE Dispatch |
|------|-----------|--------------|
| Prefill | SDPA (`at::scaled_dot_product_attention`) | `low_latency_dispatch` |
| Decode | FlashInfer PagedAttention | `low_latency_dispatch` |

### 3. 权重布局

MoE 专家权重从 HuggingFace 格式转换为 DeepGemm 格式:
- GateUp: `[num_local_experts, moe_inter*2, hidden]`
- Down: `[num_local_experts, hidden, moe_inter]`

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `csrc/nanodeploy/core/config.h` | 修改 | 添加 MoE 配置字段 |
| `csrc/nanodeploy/worker/distributed.h` | 修改 | 添加 EP 支持 |
| `csrc/nanodeploy/worker/distributed.cpp` | 修改 | 实现 `ep_rank()` |
| `csrc/nanodeploy/models/qwen3_moe.h` | 新建 | Qwen3 MoE 完整实现 |
| `csrc/nanodeploy/worker/model_runner.h` | 修改 | 添加 MoE 模型成员 |
| `csrc/nanodeploy/worker/model_runner.cpp` | 修改 | MoE 模型加载和推理 |
| `tests/test_qwen3_moe_runner.cpp` | 新建 | MoE 测试用例 |
| `tests/CMakeLists.txt` | 修改 | 添加测试目标 |

## 下一步

1. 运行 `test_qwen3_moe_runner` 验证 Phase 1 正确性
2. 准备 FP8 权重和 scales 用于 Phase 2 测试
3. 实现 DeepGemm wrapper 用于 Phase 2 FP8 优化

## 参考

- [cpp_modelrunner_migration_plan.md](cpp_modelrunner_migration_plan.md) - 原始迁移计划
- [DeepEP](../third_party/DeepEP) - Expert Parallel 通信库
- [DeepGemm](../third_party/DeepGemm) - 高性能 GEMM 内核
