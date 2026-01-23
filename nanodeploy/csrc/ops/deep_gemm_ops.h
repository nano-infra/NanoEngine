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

    // FP8 GEMM: D = A @ B.T (with block scales)
    // a: pair<fp8_data, scale_inv>, b: pair<fp8_weight, weight_scale_inv>
    static void fp8_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor&                           d,
                            const std::optional<torch::Tensor>&            c             = std::nullopt,
                            const std::string&                             compiled_dims = "nk");

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

    // Grouped FP8 GEMM with contiguous m_indices
    static void m_grouped_fp8_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                 const std::pair<torch::Tensor, torch::Tensor>& b,
                                                 const torch::Tensor&                           d,
                                                 const torch::Tensor&                           m_indices,
                                                 const std::string&                             compiled_dims = "nk");

    // Masked grouped FP8 GEMM for Low Latency mode
    static void m_grouped_fp8_gemm_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor&                           d,
                                             const torch::Tensor&                           masked_m,
                                             int                                            expected_m,
                                             const std::string&                             compiled_dims = "nk");
};

}  // namespace ops
}  // namespace nanodeploy
