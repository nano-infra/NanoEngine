#pragma once

#include <memory>
#include <string>
#include <vector>

#include "nanodeploy/csrc/core/config.h"

#ifdef DEEPSEEK_MOE
#include "deep_ep.hpp"
#endif

#include "nanodeploy/csrc/worker/model_runner_ipc.h"

namespace nanodeploy {

class DeepEpContext {
public:
    // Singleton accessor
    static DeepEpContext& instance()
    {
        static DeepEpContext instance;
        return instance;
    }
    DeepEpContext() = default;
    ~DeepEpContext();

    // Initialize DeepEP buffer based on config
    void init(const core::ModelConfig& config, int ffn_ep_world_size, int ffn_ep_rank);

    // Get buffer pointer (returns nullptr if not initialized or single GPU)
#ifdef DEEPSEEK_MOE
    deep_ep::Buffer* get_buffer() const
    {
        return buffer_.get();
    }

    deep_ep::Config* get_config() const
    {
        return dep_config_.get();
    }
#else
    void* get_buffer() const
    {
        return nullptr;
    }
#endif

    // Sync operations
    bool           sync(const DeepEpSyncReq& req);
    DeepEpInfoResp get_info();

private:
#ifdef DEEPSEEK_MOE
    std::unique_ptr<deep_ep::Buffer> buffer_;
    std::unique_ptr<deep_ep::Config> dep_config_{};
#endif
};

// Global accessor function (for compatibility)
inline DeepEpContext& get_deep_ep_context()
{
    return DeepEpContext::instance();
}

}  // namespace nanodeploy
