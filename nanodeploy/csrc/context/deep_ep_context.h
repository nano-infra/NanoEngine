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
    DeepEpContext(int rank, int world_size);
    ~DeepEpContext();

    // Initialize DeepEP buffer based on config
    void init(const core::ModelConfig& config, int ffn_ep_world_size, int ffn_ep_rank);

    // Get buffer pointer (returns nullptr if not initialized or single GPU)
#ifdef DEEPSEEK_MOE
    deep_ep::Buffer* get_buffer() const
    {
        return buffer_.get();
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
    int rank_;
    int world_size_;

#ifdef DEEPSEEK_MOE
    std::unique_ptr<deep_ep::Buffer> buffer_;
#endif
};

}  // namespace nanodeploy
