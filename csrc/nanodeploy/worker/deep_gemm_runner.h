#pragma once

#include "deep_gemm_ipc.h"
#include <memory>

// Forward declaration for pybind11 interpreter
namespace pybind11 {
class scoped_interpreter;
}

#include <optional>
#include <string>
#include <torch/torch.h>

namespace nanodeploy {

class DeepGemmRunner {
public:
    DeepGemmRunner() = default;
    ~DeepGemmRunner();

    DeepGemmInitResp init(const DeepGemmInitReq& req);
    DeepGemmTestResp run_test(const DeepGemmTestReq& req);

    // Static utilities for MoE models
    static void init_utils();
    static void bf16_gemm_nt(const torch::Tensor&                a,
                             const torch::Tensor&                b,
                             const torch::Tensor&                d,
                             const std::optional<torch::Tensor>& c             = std::nullopt,
                             const std::string&                  compiled_dims = "nk");

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

private:
    int rank_       = -1;
    int world_size_ = -1;
    int device_id_  = 0;
};

}  // namespace nanodeploy
