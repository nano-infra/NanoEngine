#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "nanodeploy/core/config.h"
#include "nanodeploy/core/weight_mapping.h"
#include "nanodeploy/layers/flashinfer_handler.h"
#include "nanodeploy/models/qwen3.h"
#include "nanodeploy/models/qwen3_moe.h"
#include "nanodeploy/worker/distributed.h"
#include "nanodeploy/worker/kv_cache.h"
#include "nanodeploy/worker/model_runner_ipc.h"
#include "nanodeploy/worker/weight_loader.h"

// Third-party
#include "deep_ep.hpp"

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

private:
    void init_internal(const std::string& config_path);
    void load_weights(const std::string& weight_path);

    // Helpers to load specific components
    void load_layer_weights(int layer_idx, models::Qwen3DecoderLayer<QuantType::FP16>* layer);
    void load_moe_layer_weights(int layer_idx, models::Qwen3MoeDecoderLayer<QuantType::FP16>* layer);

private:
    std::unique_ptr<core::ModelConfig> config_;
    std::unique_ptr<WeightManager>     weight_manager_;

    // Dense Model
    std::unique_ptr<models::Qwen3ForCausalLM<QuantType::FP16>> model_;

    // MoE Model
    std::unique_ptr<models::Qwen3MoeForCausalLM<QuantType::FP16>> moe_model_;
    std::unique_ptr<deep_ep::Buffer>                              ep_buffer_;

    std::unique_ptr<KvCache>                   kv_cache_;
    std::unique_ptr<layers::FlashInferHandler> flashinfer_handler_;

    int           rank_       = 0;
    int           world_size_ = 1;
    torch::Device device_     = torch::kCPU;
};

}  // namespace nanodeploy
