#include "deep_gemm_runner.h"

#include <chrono>
#include <iostream>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

// Fix for missing SM100ArchSpec in DeepGemm headers
#include "apis/gemm.hpp"
#include "apis/runtime.hpp"
#include "jit_kernels/heuristics/sm100.hpp"

#include "nanodeploy/logging.h"

namespace nanodeploy {

DeepGemmRunner::~DeepGemmRunner() {}

#include <mutex>

void DeepGemmRunner::init_utils()
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

    NANODEPLOY_LOG_INFO("DeepGemm utilities initialized");
    initialized = true;
}

void DeepGemmRunner::bf16_gemm_nt(const torch::Tensor&                a,
                                  const torch::Tensor&                b,
                                  const torch::Tensor&                d,
                                  const std::optional<torch::Tensor>& c,
                                  const std::string&                  compiled_dims)
{
    deep_gemm::gemm::bf16_gemm_nt(a, b, d, c, compiled_dims);
}

void DeepGemmRunner::m_grouped_bf16_gemm_nt_contiguous(const torch::Tensor& a,
                                                       const torch::Tensor& b,
                                                       const torch::Tensor& d,
                                                       const torch::Tensor& m_indices,
                                                       const std::string&   compiled_dims)
{
    deep_gemm::gemm::m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices, compiled_dims);
}

void DeepGemmRunner::m_grouped_bf16_gemm_nt_masked(const torch::Tensor& a,
                                                   const torch::Tensor& b,
                                                   const torch::Tensor& d,
                                                   const torch::Tensor& masked_m,
                                                   int                  expected_m,
                                                   const std::string&   compiled_dims)
{
    deep_gemm::gemm::m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m, compiled_dims);
}

DeepGemmInitResp DeepGemmRunner::init(const DeepGemmInitReq& req)
{
    try {
        rank_       = req.rank;
        world_size_ = req.world_size;

        // Reset device - REMOVED for safety in multi-threaded environment
        // cudaDeviceReset();
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        if (num_devices > 0) {
            device_id_ = rank_ % num_devices;
            cudaSetDevice(device_id_);
        }

        // Initialize DeepGemm
        init_utils();

        return {true, "Initialized successfully"};
    }
    catch (const std::exception& e) {
        return {false, std::string("Init failed: ") + e.what()};
    }
}

DeepGemmTestResp DeepGemmRunner::run_test(const DeepGemmTestReq& req)
{
    try {
        cudaSetDevice(device_id_);
        torch::manual_seed(req.seed);

        double total_time_us = 0.0;

        if (req.mode == (int)DeepGemmTestMode::kFp8Gemm) {
            int m = req.m;
            int n = req.n;
            int k = req.k;

            auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
            auto options_fp8  = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA);
            auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

            auto a_fp32 = torch::randn({m, k}, options_fp32);
            auto b_fp32 = torch::randn({n, k}, options_fp32);

            auto a = a_fp32.to(options_fp8);
            auto b = b_fp32.to(options_fp8);

            // Scales
            // DeepGemm requires Block-wise scaling factors.
            // Recipe default for FP32 SF is (1, 1, 128).
            // So shape should be [M, ceil_div(K, 128)] and [N, ceil_div(K, 128)]
            // K is 4096, 4096/128 = 32.
            // SFA: [M, 32] (gran_mn=1)
            // SFB: [ceil_div(N, 128), 32] (gran_mn=128 for FP32 SFB on SM100)
            int  k_blocks = (k + 127) / 128;
            int  n_blocks = (n + 127) / 128;
            auto a_scale  = torch::ones({m, k_blocks}, options_fp32);
            auto b_scale  = torch::ones({n_blocks, k_blocks}, options_fp32);

            auto d = torch::empty({m, n}, options_bf16);

            // Warmup
            for (int i = 0; i < req.warmup_iters; ++i) {
                // deep_gemm::gemm::fp8_gemm_nt(pair(a, a_scale), pair(b, b_scale), d, ...)
                deep_gemm::gemm::fp8_gemm_nt({a, a_scale},
                                             {b, b_scale},
                                             d,
                                             std::nullopt,  // c
                                             std::nullopt,  // recipe
                                             "nk",          // compiled_dims
                                             false          // disable_ue8m0_cast
                );
            }
            torch::cuda::synchronize();

            // Measure
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < req.test_iters; ++i) {
                deep_gemm::gemm::fp8_gemm_nt({a, a_scale}, {b, b_scale}, d, std::nullopt, std::nullopt, "nk", false);
            }
            torch::cuda::synchronize();
            auto end      = std::chrono::high_resolution_clock::now();
            total_time_us = std::chrono::duration<double, std::micro>(end - start).count();
        }
        else if (req.mode == (int)DeepGemmTestMode::kMaskedGroupGemm) {
            int m          = req.m;
            int n          = req.n;
            int k          = req.k;
            int num_groups = req.num_groups;
            int expected_m = m / 2;

            auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
            auto options_fp8  = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA);
            auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
            auto options_int  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

            auto a_fp32 = torch::randn({num_groups, m, k}, options_fp32);
            auto b_fp32 = torch::randn({num_groups, n, k}, options_fp32);

            auto a = a_fp32.to(options_fp8);
            auto b = b_fp32.to(options_fp8);

            int k_blocks = (k + 127) / 128;
            int n_blocks = (n + 127) / 128;  // For SFB in masked GEMM too?
            // In masked GEMM, A is [num_groups, m, k], B is [num_groups, n, k].
            // SFA: [num_groups, m, k_blocks] (gran_mn=1)
            // SFB: [num_groups, ceil_div(n, 128), k_blocks] (gran_mn=128)
            auto a_scale = torch::ones({num_groups, m, k_blocks}, options_fp32);
            auto b_scale = torch::ones({num_groups, n_blocks, k_blocks}, options_fp32);

            auto masked_m = torch::randint(1, m + 1, {num_groups}, options_int);

            auto d = torch::empty({num_groups, m, n}, options_bf16);

            // Warmup
            for (int i = 0; i < req.warmup_iters; ++i) {
                deep_gemm::gemm::m_grouped_fp8_gemm_nt_masked(
                    {a, a_scale}, {b, b_scale}, d, masked_m, expected_m, std::nullopt, "nk", false);
            }
            torch::cuda::synchronize();

            // Measure
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < req.test_iters; ++i) {
                deep_gemm::gemm::m_grouped_fp8_gemm_nt_masked(
                    {a, a_scale}, {b, b_scale}, d, masked_m, expected_m, std::nullopt, "nk", false);
            }
            torch::cuda::synchronize();
            auto end      = std::chrono::high_resolution_clock::now();
            total_time_us = std::chrono::duration<double, std::micro>(end - start).count();
        }
        else if (req.mode == (int)DeepGemmTestMode::kGroupedBf16Gemm) {
            int m_actual   = req.m;
            int n          = req.n;
            int k          = req.k;
            int num_groups = req.num_groups;

            // DeepGemm Grouped GEMM requires M to be a multiple of 128
            int m_padded = (m_actual + 127) / 128 * 128;

            auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
            auto options_int  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

            // A: [M_padded, K]
            auto a = torch::randn({m_padded, k}, options_bf16);
            // B: [num_groups, N, K]
            auto b = torch::randn({num_groups, n, k}, options_bf16);
            // D: [M_padded, N]
            auto d = torch::empty({m_padded, n}, options_bf16);

            // m_indices: [M_padded] -> expert ID for each row
            auto m_indices         = torch::empty({m_padded}, options_int);
            int  tokens_per_expert = m_padded / num_groups;
            for (int g = 0; g < num_groups; ++g) {
                int start = g * tokens_per_expert;
                int end   = (g == num_groups - 1) ? m_padded : (g + 1) * tokens_per_expert;
                m_indices.narrow(0, start, end - start).fill_(g);
            }

            // Warmup
            for (int i = 0; i < req.warmup_iters; ++i) {
                deep_gemm::gemm::m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices, "nk");
            }
            torch::cuda::synchronize();

            // Measure
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < req.test_iters; ++i) {
                deep_gemm::gemm::m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices, "nk");
            }
            torch::cuda::synchronize();
            auto end      = std::chrono::high_resolution_clock::now();
            total_time_us = std::chrono::duration<double, std::micro>(end - start).count();
        }
        else {
            return {false, 0.0, "Unknown test mode"};
        }

        return {true, total_time_us / req.test_iters, "Success"};
    }
    catch (const std::exception& e) {
        return {false, 0.0, std::string("Test failed: ") + e.what()};
    }
}

}  // namespace nanodeploy
