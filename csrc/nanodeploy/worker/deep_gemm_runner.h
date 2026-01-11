#pragma once

#include "deep_gemm_ipc.h"
#include <memory>

// Forward declaration for pybind11 interpreter
namespace pybind11 {
class scoped_interpreter;
}

namespace nanodeploy {

class DeepGemmRunner {
public:
    DeepGemmRunner() = default;
    ~DeepGemmRunner();

    DeepGemmInitResp init(const DeepGemmInitReq& req);
    DeepGemmTestResp run_test(const DeepGemmTestReq& req);

private:
    int rank_       = -1;
    int world_size_ = -1;
    int device_id_  = 0;
};

}  // namespace nanodeploy
