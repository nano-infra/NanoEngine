# NanoDeploy C++ ModelRunner 迁移计划

## 1. 现状分析

### 1.1 Python 模块结构

```
nanodeploy/
├── worker/
│   ├── model_runner.py      # 854 行 - Ray Actor, 推理主循环
│   ├── cache.py             # KV Cache 管理
│   ├── context.py           # 推理上下文
│   ├── distributed.py       # 分布式上下文
│   └── loader.py            # 权重加载
├── layers/
│   ├── activation.py        # SiluAndMul
│   ├── attention.py         # FlashAttention/MLA
│   ├── embed_head.py        # Embedding/LMHead
│   ├── layernorm.py         # RMSNorm
│   ├── linear.py            # QKV/ColumnParallel/RowParallel
│   ├── rotary_embedding.py  # RoPE
│   └── sampler.py           # Top-k/p 采样
└── models/
    └── qwen3_moe.py         # 548 行 - Qwen3MoE 模型定义
```

### 1.2 现有 C++ 代码

```
NanoDeploy/csrc/nanodeploy/
├── worker/
│   ├── dummy_runner.h       # ✅ 已有 - c10d 分布式初始化模板
│   └── dummy_runner_ipc.h   # ✅ 已有 - Spoke 序列化
│   └── model_runner_utils.h # ✅ 已有 - Metadata 结构体
├── sequence/                 # ✅ 已有 - Sequence 定义
├── scheduler/                # ✅ 已有 - C++ 调度器
├── executor/                 # ✅ 已有 - 执行器框架
└── logging.h                 # ✅ 已有 - 日志
```

______________________________________________________________________

## 2. Python → C++ 模块对应表

| Python 模块           | C++ 目标                | 依赖库                    | 优先级 |
| :-------------------- | :---------------------- | :------------------------ | :----: |
| `model_runner.py`     | `model_runner.h`        | libtorch, Spoke           |   P0   |
| `distributed.py`      | `distributed_context.h` | c10d                      |   P0   |
| `context.py`          | `inference_context.h`   | libtorch                  |   P0   |
| `cache.py`            | `kv_cache.h`            | FlashInfer page.cuh       |   P1   |
| `loader.py`           | `weight_loader.h`       | nlohmann/json, mmap       |   P1   |
| `linear.py`           | `layers/linear.h`       | **DeepGEMM** (FP8)        |   P1   |
| `attention.py`        | `layers/attention.h`    | FlashInfer Attention/MLA  |   P1   |
| `layernorm.py`        | `layers/rms_norm.h`     | FlashInfer norm.cuh       |   P2   |
| `rotary_embedding.py` | `layers/rotary_emb.h`   | FlashInfer pos_enc.cuh    |   P2   |
| `activation.py`       | `layers/activation.h`   | FlashInfer activation.cuh |   P2   |
| `embed_head.py`       | `layers/embedding.h`    | libtorch                  |   P2   |
| `sampler.py`          | `layers/sampler.h`      | FlashInfer sampling.cuh   |   P2   |
| `qwen3_moe.py`        | `models/qwen3_moe.h`    | **DeepGEMM + DeepEP**     |   P3   |

______________________________________________________________________

## 3. 详细设计

### 3.1 量化模板 + 工厂模式 (细化)

```cpp
// ==================== 量化类型枚举 ====================
enum class QuantType { FP16, BF16, FP8_E4M3, W8A8 };

// ==================== 权重描述 ====================
struct WeightDesc {
    std::string name;
    std::vector<int64_t> shape;
    torch::ScalarType dtype;
    bool is_scale = false;
};

// ==================== 线性层模板 ====================
template<QuantType Q>
struct LinearTraits;

template<> struct LinearTraits<QuantType::FP16> {
    using WeightType = torch::Tensor;  // FP16
    static constexpr bool HasScale = false;
    static constexpr torch::ScalarType Dtype = torch::kFloat16;
};

template<> struct LinearTraits<QuantType::FP8_E4M3> {
    using WeightType = torch::Tensor;  // FP8
    static constexpr bool HasScale = true;
    static constexpr torch::ScalarType Dtype = torch::kFloat8_e4m3fn;
};

template<QuantType Q>
class Linear {
    using Traits = LinearTraits<Q>;
    torch::Tensor weight_;
    std::conditional_t<Traits::HasScale, torch::Tensor, std::monostate> scale_;

public:
    std::vector<WeightDesc> weight_descs(const std::string& prefix) const {
        std::vector<WeightDesc> descs = {
            {prefix + ".weight", shape_, Traits::Dtype}
        };
        if constexpr (Traits::HasScale) {
            descs.push_back({prefix + ".weight_scale", scale_shape_, torch::kFloat32, true});
        }
        return descs;
    }

    void forward(torch::Tensor& x) {
        if constexpr (Q == QuantType::FP16 || Q == QuantType::BF16) {
            flashinfer::gemm_fp16(x, weight_);
        } else if constexpr (Q == QuantType::FP8_E4M3) {
            flashinfer::gemm_fp8(x, weight_, scale_);
        }
    }
};

// ==================== 工厂函数 ====================
class ILinear {
public:
    virtual std::vector<WeightDesc> weight_descs(const std::string& prefix) const = 0;
    virtual void set_weights(const std::map<std::string, torch::Tensor>& w) = 0;
    virtual void forward(torch::Tensor& x) = 0;
    virtual ~ILinear() = default;
};

std::unique_ptr<ILinear> create_linear(QuantType q, int in_features, int out_features) {
    switch (q) {
        case QuantType::FP16:
            return std::make_unique<Linear<QuantType::FP16>>(in_features, out_features);
        case QuantType::FP8_E4M3:
            return std::make_unique<Linear<QuantType::FP8_E4M3>>(in_features, out_features);
        // ...
    }
}
```

### 3.2 配置化权重映射 (细化)

```cpp
// ==================== 权重映射配置 ====================
// 从 JSON 或 YAML 加载

struct WeightMapping {
    std::string hf_name;       // HuggingFace 权重名
    std::string internal_name; // 内部名
    std::vector<std::string> pack_sources;  // 打包来源 (可选)
};

// qwen3_moe_weight_map.json:
// {
//   "model.layers.{layer}.self_attn.q_proj.weight": "layers.{layer}.attn.wq",
//   "model.layers.{layer}.self_attn.k_proj.weight": "layers.{layer}.attn.wk",
//   "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight": "layers.{layer}.moe.experts.{expert}.w1",
//   ...
// }

class WeightMappingConfig {
    std::vector<WeightMapping> mappings_;

public:
    static WeightMappingConfig load(const std::string& model_type) {
        std::string path = fmt::format("configs/{}_weight_map.json", model_type);
        auto j = nlohmann::json::parse(std::ifstream(path));
        // ...
    }

    std::string resolve(const std::string& internal_name, int layer, int expert = -1) {
        // 替换 {layer} 和 {expert} 占位符
    }
};
```

### 3.3 qwen3_moe.py → qwen3_moe.h 映射

| Python 类                | C++ 类                   |          权重数量           |
| :----------------------- | :----------------------- | :-------------------------: |
| `Qwen3MoeAttention`      | `Qwen3Attention<Q>`      |     4 (qkv, o) + 2 norm     |
| `Qwen3MoeMLP`            | `Qwen3MLP<Q>`            | 2 (gate_up, down) + scales  |
| `Qwen3MoeSparseMoeBlock` | `Qwen3MoE<Q>`            | gate + experts × (w1+w2+w3) |
| `Qwen3MoeDecoderLayer`   | `Qwen3DecoderLayer<Q>`   |   attn + mlp/moe + 2 norm   |
| `Qwen3MoeModel`          | `Qwen3Model<Q>`          | embed + layers + final_norm |
| `Qwen3MoeForCausalLM`    | `Qwen3MoeForCausalLM<Q>` |       model + lm_head       |

关键映射:

```cpp
// Python: self.qkv_proj = QKVParallelLinear(...)
// C++:
template<QuantType Q>
class QKVParallelLinear {
    Linear<Q> q_proj_, k_proj_, v_proj_;  // 或融合为 qkv_

    std::vector<WeightDesc> weight_descs(int layer) const {
        std::string prefix = fmt::format("model.layers.{}.self_attn.", layer);
        auto descs = q_proj_.weight_descs(prefix + "q_proj");
        concat(descs, k_proj_.weight_descs(prefix + "k_proj"));
        concat(descs, v_proj_.weight_descs(prefix + "v_proj"));
        return descs;
    }
};
```

______________________________________________________________________

## 4. 迁移阶段 (细化版)

### Phase 0: 文件架构与基础配置 (1 天)

- [ ] **建立目录结构**: 按照第 7.5 节的 CMake 模板配置 `third_party` 和 `csrc`。
- [ ] **初始化 Module 基类**: 定义 `Module` 接口，包含 `weight_specs()` 和 `set_weights()`。
- [ ] **实现配置加载**: 使用 `nlohmann/json` 解析 HF `config.json` 到 `ModelConfig`。
- [ ] **定义权重映射 (WeightMapping)**: 实现从内部统一命名到 HF 命名的映射器 (Section 3.2)。

### Phase 1: 权重加载与 KV Cache (3 天)

- [ ] **SafeTensorLoader**: 实现基于 `mmap` 的权重读取，返回 `torch::Tensor` (Section 3.2)。
- [ ] **WeightServer (Spoke Actor)**: 实现基于 RDMA 的权重下发服务，支持分片加载。
- [ ] **KVCache (C++ 实现)**: 迁移 Python `cache.py` 逻辑，对接 FlashInfer 的 `paged_kv_cache`。
- [ ] **验证**: `test_weight_loader` 确保能正确加载权重到 GPU。

### Phase 2: 基础算子层 (5 天)

- [ ] **Linear 模板与工厂 (DeepGEMM)**:
  - 实现 `LinearTraits` 特化 (Section 3.1)。
  - 对接 DeepGEMM 的 `fp8_gemm`。
  - **避坑**: 实现 `warmup_deep_gemm()` 预热 JIT 缓存 (Section 7.2)。
- [ ] **FlashInfer 算子封装**:
  - 实现 `RMSNorm`, `RoPE`, `Sampler` 的 C++ 封装。
  - **避坑**: 采用分离编译单元 (`.cu`) 减少模板膨胀 (Section 7.1)。
- [ ] **验证**: `test_layers` 对比 C++ 层输出与 Python 输出的一致性。

### Phase 3: MoE 层与 DeepEP 集成 (5 天)

- [ ] **DeepEP 初始化**: 按照 c10d -> NVSHMEM 的顺序初始化 (Section 7.3)。
- [ ] **MoE 核心逻辑**:
  - 实现 Gating + TopK。
  - 集成 DeepEP 的 `low_latency_dispatch`。
  - 对接 DeepGEMM 的 `m_grouped_fp8_gemm_nt_masked` (Section 3.1)。
- [ ] **验证**: `test_moe` 在多机环境下验证专家并行通信正确性。

### Phase 4: 模型组装与条带化 SP (3 天)

- [ ] **Qwen3MoE 模型类**: 组合 Attention, MLP/MoE, Norm。
- [ ] **条带化 SP 逻辑 (Section 9)**:
  - 在 Attention 中实现 `master_sp_rank` 路由判断。
  - 使用 FlashInfer 的 `merge_states` 合并 SP 结果。
  - **避坑**: 使用 `DLSlime` 的 `AllToAllIntraLLBuffer` 实现低延迟全对全通信 (Section 9.3.2)。

### Phase 5: ModelRunner 与现有代码集成 (5 天)

- [ ] **集成现有组件 (Section 8)**:
  - 接入现有 `Scheduler` 进行批调度。
  - 使用 `prepare_prefill_cpp` / `prepare_decode_cpp` 准备元数据。
  - 使用 `update_seqs_inner_loop` 更新 `Sequence` 状态。
- [ ] **实现核心推理循环**:
  - `run()` 函数处理 prefill/decode。
  - 实现 **Local/Global 双 Batch Capture** 策略捕获 CUDA 图 (Section 9.3.3)。
- [ ] **Spoke Actor 封装**: 将 `ModelRunner` 封装为 Spoke Actor，支持 RPC 调用。

### Phase 6: 端端验证与优化 (3 天)

- [ ] **精度验证**: 运行经典 prompt，对比生成的 token 与 Python 原版是否一致。
- [ ] **性能分析**: 评估 C++ 带来的 overhead 减少和 GPU 利用率提升。
- [ ] **部署**: 整理文件布局，确保 cache 目录可迁移 (Section 7.2)。

______________________________________________________________________

## 5. 时间线

| 阶段                 | 工作量 |   累计    |
| :------------------- | :----: | :-------: |
| Phase 0: 文件架构    |  1 天  |   1 天    |
| Phase 1: Config 加载 |  2 天  |   3 天    |
| Phase 2: 权重加载    |  3 天  |   6 天    |
| Phase 3: 基础层      |  5 天  |   11 天   |
| Phase 4: MoE 层      |  5 天  |   16 天   |
| Phase 5: 模型组装    |  3 天  |   19 天   |
| Phase 6: ModelRunner |  5 天  |   24 天   |
| Phase 7: 端到端测试  |  3 天  | **27 天** |

______________________________________________________________________

## 6. 风险与缓解

| 风险                     | 缓解措施                 |
| :----------------------- | :----------------------- |
| FlashInfer API 变动      | 锁定 commit, 封装适配层  |
| HuggingFace 权重命名变化 | 配置化映射表, 支持多版本 |
| 多种量化格式             | 模板特化 + 工厂模式      |
| DeepEP 编译复杂          | 预编译 .so, 文档化依赖   |

______________________________________________________________________

## 7. C++ 集成注意事项

### 7.1 FlashInfer (Header-Only CUDA)

| 问题                  | 影响                   | 解决方案                              |
| :-------------------- | :--------------------- | :------------------------------------ |
| **CUDA 模板膨胀**     | 编译时间长, .so 体积大 | 显式实例化常用类型, 分离编译单元      |
| **C++17 要求**        | 需要 nvcc 支持         | 确保 CUDA >= 11.0                     |
| **cuBLAS/cuDNN 冲突** | 符号重复               | 使用 `--relocatable-device-code=true` |

```cmake
# FlashInfer 集成示例
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# 显式实例化减少编译时间
add_library(flashinfer_kernels STATIC
    flashinfer_attention_fp16.cu
    flashinfer_attention_bf16.cu
    flashinfer_attention_fp8.cu
)
```

### 7.2 DeepGEMM (JIT 编译)

> \[!WARNING\]
> DeepGEMM 使用 **运行时 JIT 编译**，需要特殊处理

| 问题                | 影响             | 解决方案                           |
| :------------------ | :--------------- | :--------------------------------- |
| **JIT 首次编译慢**  | 第一次调用延迟高 | 预热阶段编译, 缓存到磁盘           |
| **需要 NVCC/NVRTC** | 部署时需编译器   | 使用 `DG_JIT_CACHE_DIR` 预编译缓存 |
| **Python 绑定**     | C++ 调用困难     | 直接使用 `csrc/` 下的 C++ 代码     |

```cpp
// DeepGEMM C++ 直接调用
#include "deep_gemm/jit/jit_context.hpp"
#include "deep_gemm/fp8_gemm.hpp"

// 预热 (首次会触发 JIT 编译)
void warmup_deep_gemm() {
    deep_gemm::JitContext::instance().set_cache_dir("/path/to/cache");

    // 预编译常用 shape
    auto dummy_a = torch::zeros({1024, 4096}, torch::kFloat8_e4m3fn);
    auto dummy_b = torch::zeros({4096, 4096}, torch::kFloat8_e4m3fn);
    deep_gemm::fp8_gemm_nt(dummy_a, dummy_b, ...);
}
```

**部署策略**:

1. 开发时: JIT 编译，灵活调试
2. 生产部署: 预编译所有 shape，拷贝 cache 目录

### 7.3 DeepEP (NVSHMEM 依赖)

| 问题                 | 影响                  | 解决方案               |
| :------------------- | :-------------------- | :--------------------- |
| **NVSHMEM 安装复杂** | 需要 RDMA 驱动配置    | 文档化安装流程         |
| **进程组要求**       | 需要 `nvshmem_init()` | 在 c10d 初始化后调用   |
| **符号冲突**         | 与 MPI 冲突           | 使用动态链接, 隔离加载 |

```cpp
// DeepEP 初始化顺序
void init_distributed() {
    // 1. 先初始化 c10d (TCPStore + NCCL)
    c10d::TCPStore store(...);
    c10d::ProcessGroupNCCL pg(...);

    // 2. 再初始化 NVSHMEM (DeepEP 内部处理)
    deep_ep::Buffer buffer(pg, nvl_bytes, rdma_bytes);
}
```

______________________________________________________________________

## 8. 现有 C++ 代码集成

### 8.1 已有模块概览

```
NanoDeploy/csrc/nanodeploy/
├── scheduler/
│   ├── scheduler.h          # ✅ Scheduler 类 - 批调度逻辑
│   ├── block_manager.h      # ✅ BlockManager - KV Cache 块管理
│   └── sp_state_manager.h   # ✅ Sequence Parallel 状态管理
├── sequence/
│   ├── sequence.h           # ✅ Sequence 类 - 请求/Token 管理
│   └── serialization.h      # ✅ 序列化 (与 Spoke 通信)
├── worker/
│   ├── dummy_runner.h       # ✅ DummyRunner - c10d 分布式模板
│   ├── dummy_runner_ipc.h   # ✅ Spoke 序列化特化
│   └── model_runner_utils.h # ✅ PrefillMetadata/DecodeMetadata
└── executor/
    └── spoke_executor.h     # ✅ Spoke Executor 框架
```

### 8.2 新 ModelRunner 集成点

```cpp
// worker/model_runner.h (新文件)
class ModelRunner {
    // === 核心组件 (Worker 侧) ===
    std::unique_ptr<Qwen3MoeForCausalLM<QuantType::FP8>> model_;
    std::unique_ptr<KVCache> kv_cache_;
    std::unique_ptr<WeightClient> weight_client_;

public:
    // 执行一轮推理 (由 Spoke Actor 调用)
    RunResp run_step(const RunReq& req) {
        // 1. 根据 req (序列化后的 Metadata) 进行前向
        // 2. 采样并返回 logits/tokens
        return resp;
    }
};
```

______________________________________________________________________

## 9. 条带化动态上下文并行 (Striped Context Parallel)

### 9.1 通信机制

> \[!IMPORTANT\]
> 采用 `DLSlime` 的 `AllToAllIntraLLBuffer` 替代原有的 P2P NCCL 通信。

```cpp
// 正确的 SP 通信模式: 使用 DLSlime 的 AllToAllIntraLLBuffer
void sp_communication(AllToAllIntraLLBuffer& buffer,
                      torch::Tensor input,
                      torch::Tensor mask,
                      bool is_transpose = false) {
    // 内部封装了基于 NVSHMEM/IPC 的低延迟 All-to-All (LL)
    auto output = buffer.allToAllLL2D(input, is_transpose, mask);
}
```

### 9.2 CUDA Graph 兼容性: Local/Global Dual Batch Capture

由于 SP 模式下，当前 rank 既作为 Master 处理 local 序列采样（`master_bs`），又作为参与者处理 global 范围内的 Attention 计算（`attn_bs`），我们需要捕获包含这两个 batch size 的图：

1. **预采样 Batch 列表**:

   ```cpp
   std::vector<int> master_bs_list = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
   std::vector<int> attn_bs_list   = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
   ```

2. **双循环捕获与跳过条件**:

   ```cpp
   for (int m_bs : master_bs_list) {
       for (int a_bs : attn_bs_list) {
           // 跳过无效组合，减少捕获时间
           if (a_bs < m_bs - max_num_send || a_bs > m_bs + max_num_recv) continue;

           // 捕获 (m_bs, a_bs) 组合
           capture_graph(m_bs, a_bs);
       }
   }
   ```

3. **运行时查找**:
   根据当前的 `bs` 和 `context.attention_compute_bs` 向上取整查找最匹配的已捕获图。

______________________________________________________________________

## 10. Engine-Worker 交互协议

1. **职责分离**:

   - **Engine (Master)**: 运行 `Scheduler`, `Sequence` 管理。
   - **Worker (ModelRunner)**: 运行 GPU 计算 (Attention/MoE), 采样。

2. **Spoke RPC 交互**:
   Engine 通过 Spoke 调用 Worker 的 `run_step`，传输精简后的 Metadata (RDMA 零拷贝)。
