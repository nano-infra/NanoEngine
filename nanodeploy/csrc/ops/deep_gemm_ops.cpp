#include "nanodeploy/csrc/ops/deep_gemm_ops.h"

#include <iostream>
#include <mutex>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include "apis/gemm.hpp"
#include "apis/runtime.hpp"
#include "jit_kernels/heuristics/sm100.hpp"

#include "nanodeploy/csrc/logging.h"

namespace nanodeploy {
namespace ops {

void DeepGemmOps::init()
{
    static std::mutex           mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static bool initialized = false;
    if (initialized)
        return;

    std::string deep_gemm_root = "/mnt/nvme1n1/ml_research/majinming/src/nano-deploy/third_party/DeepGemm/deep_gemm";
    std::string cuda_home      = "/usr/local/cuda";

    deep_gemm::Compiler::prepare_init(deep_gemm_root, cuda_home);
    deep_gemm::KernelRuntime::prepare_init(cuda_home);

    NANODEPLOY_LOG_INFO("DeepGemmOps initialized");
    initialized = true;
}

void DeepGemmOps::bf16_gemm_nt(const torch::Tensor&                a,
                               const torch::Tensor&                b,
                               const torch::Tensor&                d,
                               const std::optional<torch::Tensor>& c,
                               const std::string&                  compiled_dims)
{
    deep_gemm::gemm::bf16_gemm_nt(a, b, d, c, compiled_dims);
}

void DeepGemmOps::m_grouped_bf16_gemm_nt_contiguous(const torch::Tensor& a,
                                                    const torch::Tensor& b,
                                                    const torch::Tensor& d,
                                                    const torch::Tensor& m_indices,
                                                    const std::string&   compiled_dims)
{
    deep_gemm::gemm::m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices, compiled_dims);
}

void DeepGemmOps::m_grouped_bf16_gemm_nt_masked(const torch::Tensor& a,
                                                const torch::Tensor& b,
                                                const torch::Tensor& d,
                                                const torch::Tensor& masked_m,
                                                int                  expected_m,
                                                const std::string&   compiled_dims)
{
    deep_gemm::gemm::m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m, compiled_dims);
}

}  // namespace ops
}  // namespace nanodeploy
