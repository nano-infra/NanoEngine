#pragma once

#include "nanodeploy/csrc/core/config.h"

#ifdef DEEPSEEK_MOE
#include "nanodeploy/csrc/ops/deep_gemm_ops.h"
#endif

namespace nanodeploy {

class DeepGemmContext {
public:
    DeepGemmContext()  = default;
    ~DeepGemmContext() = default;

    void init(const core::ModelConfig& config);
};

}  // namespace nanodeploy
