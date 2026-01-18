#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "nanodeploy/csrc/context/attention_context.h"
#include "nanodeploy/csrc/context/distributed_context.h"
#include "nanodeploy/csrc/core/config.h"
#include "nanodeploy/csrc/models/qwen3_host.h"
#include "nanodeploy/csrc/worker/model_runner_ipc.h"
#include "nanodeploy/csrc/worker/weight_loader.h"

namespace nanodeploy {

// Forward declare WeightManager if reused, or redefine if needed.
// We can reuse the one in model_runner.h if we move it to a common header,
// OR just redefine/copy it for HostRunner to stay decoupled.
// Given strict separation request, let's duplicate/adapt minimal WeightManager or include it.
// Actually, WeightManager is coupled in model_runner.h.
// Let's copy it to here for now to avoid include mess, or move to common.
// Moving to common `weight_manager.h` would be cleaner but out of strict plan scope.
// I will include definition here for simplicity of "Just Host Runner".

class HostWeightManager {
public:
    explicit HostWeightManager(const std::filesystem::path& model_dir, torch::Device device = torch::kCPU);

    torch::Tensor load(const std::string& param_name);
    bool          has_param(const std::string& param_name);

private:
    std::filesystem::path                             model_dir_;
    torch::Device                                     device_;
    bool                                              is_sharded_ = false;
    std::unordered_map<std::string, std::string>      param_to_file_;
    std::unordered_map<std::string, SafeTensorLoader> loaders_;
};

class HostModelRunner {
public:
    HostModelRunner() = default;

    void init(const std::string& config_path, int rank, int world_size);
    void init(const std::string& config_path, const DistributedConfig& dconf);

    ModelRunResp run(ModelRunReq req);

    // DeepEP stubs (return empty/false for Host)
    DeepEpInfoResp getDeepEpInfo()
    {
        return {};
    }
    bool syncDeepEp(DeepEpSyncReq /*req*/)
    {
        return true;
    }

private:
    void init_internal(const std::string& config_path, int rank);
    void load_weights(const std::string& weight_path);
    void load_layer_weights(int layer_idx, models::Qwen3HostDecoderLayer<QuantType::FP16>* layer);

private:
    std::unique_ptr<core::ModelConfig> config_;
    std::unique_ptr<HostWeightManager> weight_manager_;

    // Host Model
    std::unique_ptr<models::Qwen3HostForCausalLM<QuantType::FP16>> model_;

    std::unique_ptr<KvCache> kv_cache_;

    int           rank_       = 0;
    int           world_size_ = 1;
    torch::Device device_     = torch::kCPU;
};

}  // namespace nanodeploy
