#pragma once

#include <memory>
#include <optional>
#include <string>
#include <torch/torch.h>

namespace nanodeploy {
namespace ops {

/**
 * DeepGemmOps - Static operations for DeepGemm GEMM kernels
 */
class DeepGemmOps {
public:
    // Initialize DeepGemm utilities (compiler, runtime)
    static void init();

    // BF16 GEMM: D = A @ B.T + C
    static void bf16_gemm_nt(const torch::Tensor&                a,
                             const torch::Tensor&                b,
                             const torch::Tensor&                d,
                             const std::optional<torch::Tensor>& c             = std::nullopt,
                             const std::string&                  compiled_dims = "nk");

    // Grouped BF16 GEMM with contiguous m_indices
    static void m_grouped_bf16_gemm_nt_contiguous(const torch::Tensor& a,
                                                  const torch::Tensor& b,
                                                  const torch::Tensor& d,
                                                  const torch::Tensor& m_indices,
                                                  const std::string&   compiled_dims = "nk");

    // Masked grouped GEMM for Low Latency mode
    static void m_grouped_bf16_gemm_nt_masked(const torch::Tensor& a,
                                              const torch::Tensor& b,
                                              const torch::Tensor& d,
                                              const torch::Tensor& masked_m,
                                              int                  expected_m,
                                              const std::string&   compiled_dims = "nk");
};

}  // namespace ops
}  // namespace nanodeploy
