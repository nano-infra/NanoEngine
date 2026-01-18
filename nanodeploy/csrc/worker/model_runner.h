#pragma once

#include <c10/cuda/CUDAStream.h>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "nanodeploy/csrc/context/attention_context.h"
#include "nanodeploy/csrc/context/deep_ep_context.h"
#include "nanodeploy/csrc/context/deep_gemm_context.h"
#include "nanodeploy/csrc/context/distributed_context.h"
#include "nanodeploy/csrc/core/config.h"
#include "nanodeploy/csrc/core/weight_mapping.h"
#include "nanodeploy/csrc/models/qwen3.h"
#include "nanodeploy/csrc/models/qwen3_moe.h"
#include "nanodeploy/csrc/ops/flashinfer_ops.h"
#include "nanodeploy/csrc/worker/model_runner_ipc.h"
#include "nanodeploy/csrc/worker/weight_loader.h"

// Third-party
#ifdef DEEPSEEK_MOE
#include "deep_ep.hpp"
#endif

namespace nanodeploy {

class WeightManager {
public:
    explicit WeightManager(const std::filesystem::path& model_dir, torch::Device device = torch::kCPU);

    torch::Tensor load(const std::string& param_name);

    // Helper to check existence
    bool has_param(const std::string& param_name);

private:
    std::filesystem::path                             model_dir_;
    torch::Device                                     device_;
    bool                                              is_sharded_ = false;
    std::unordered_map<std::string, std::string>      param_to_file_;
    std::unordered_map<std::string, SafeTensorLoader> loaders_;
};

class ModelRunner {
public:
    ModelRunner() = default;

    // Initialize model, KV cache, and load weights
    void init(const std::string& config_path, int rank, int world_size);
    void init(const std::string& config_path, const DistributedConfig& dconf);

    // Main execution entry point
    ModelRunResp run(ModelRunReq req);

    // DeepEP sync methods (for multi-GPU MoE)
    DeepEpInfoResp getDeepEpInfo();
    bool           syncDeepEp(DeepEpSyncReq req);

    // Initializer Extensions (Multi-Stage)
    KvCacheInitResp  init_kv_cache(KvCacheInitReq req);
    bool             warmup_moe();  // Standalone MoE/DeepGemm warmup
    GraphCaptureResp capture_decode_graphs(GraphCaptureReq req);

private:
    void init_internal(const std::string& config_path, int rank);
    void load_weights(const std::string& weight_path);

    // Helpers to load specific components
    void load_layer_weights(int layer_idx, models::Qwen3DecoderLayer<QuantType::FP16>* layer);
    void load_moe_layer_weights(int layer_idx, models::DeepSeekMoeDecoderLayer<QuantType::FP16>* layer);

private:
    std::unique_ptr<core::ModelConfig> config_;
    std::unique_ptr<WeightManager>     weight_manager_;

    // Contexts
    std::unique_ptr<DeepEpContext>    deep_ep_ctx_;
    std::unique_ptr<AttentionContext> attn_ctx_;
    std::unique_ptr<DeepGemmContext>  gemm_ctx_;

    // Dense Model
    std::unique_ptr<models::Qwen3ForCausalLM<QuantType::FP16>> model_;

    // MoE Model
    std::unique_ptr<models::Qwen3MoeForCausalLM<QuantType::FP16>> moe_model_;

    int           world_size_     = 1;
    int           rank_           = 0;
    int           max_batch_size_ = 0;
    torch::Device device_         = torch::kCPU;

    // CUDA Graph Support
    // Static inputs/buffers for graph execution
    torch::Tensor static_input_ids_;
    torch::Tensor static_positions_;
    torch::Tensor static_slot_mapping_;
    torch::Tensor static_block_tables_;
    torch::Tensor static_seq_lens_;

    // Graph storage: batch_size -> graph_exec
    std::map<int, cudaGraphExec_t> decode_graphs_;
    std::map<int, torch::Tensor>   decode_graph_outputs_;

    bool                  enable_cuda_graph_ = false;
    c10::cuda::CUDAStream capture_stream_    = c10::cuda::getDefaultCUDAStream();
};

}  // namespace nanodeploy
