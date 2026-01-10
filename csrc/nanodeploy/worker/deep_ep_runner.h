#pragma once

#include <memory>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "deep_ep.hpp"
#include "nanodeploy/worker/deep_ep_ipc.h"

namespace nanodeploy {

class DeepEPRunner {
public:
    DeepEPRunner() = default;
    ~DeepEPRunner();

    DeepEPInitResp init(const DeepEPInitReq& req);
    DeepEPInfoResp get_info();
    DeepEPSyncResp sync(const DeepEPSyncReq& req);
    DeepEPTestResp run_test(const DeepEPTestReq& req);

private:
    std::unique_ptr<deep_ep::Buffer> buffer_;
    int                              rank_       = -1;
    int                              world_size_ = -1;
    int                              device_id_  = 0;
};

}  // namespace nanodeploy
