#include "nanodeploy/csrc/context/deep_gemm_context.h"
#include "nanodeploy/csrc/logging.h"

namespace nanodeploy {

void DeepGemmContext::init(const core::ModelConfig& config)
{
#ifdef DEEPSEEK_MOE
    if (config.is_moe) {
        NANODEPLOY_LOG_INFO("Initializing DeepGemm utils...");
        ops::DeepGemmOps::init();
    }
#endif
}

}  // namespace nanodeploy
