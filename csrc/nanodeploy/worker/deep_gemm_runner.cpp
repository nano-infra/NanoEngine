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

namespace nanodeploy {

DeepGemmRunner::~DeepGemmRunner() {}

DeepGemmInitResp DeepGemmRunner::init(const DeepGemmInitReq& req)
{
    try {
        rank_       = req.rank;
        world_size_ = req.world_size;

        // Reset device
        cudaDeviceReset();
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        if (num_devices > 0) {
            device_id_ = rank_ % num_devices;
            cudaSetDevice(device_id_);
        }

        // Initialize DeepGemm
        // We need the library root path (where include/ and kernels/ are)
        // The structure is DeepGemm/deep_gemm/include, so we point to DeepGemm/deep_gemm
        std::string deep_gemm_root =
            "/mnt/nvme1n1/ml_research/majinming/src/nano-deploy/third_party/DeepGemm/deep_gemm";
        std::string cuda_home = "/usr/local/cuda";  // As found by 'which nvcc'

        // Ensure paths exist? DeepGemm might throw if not.
        // deep_gemm::runtime::init does not exist as a C++ function, only pybind definition.
        // We call underlying init functions directly.
        deep_gemm::Compiler::prepare_init(deep_gemm_root, cuda_home);
        deep_gemm::KernelRuntime::prepare_init(cuda_home);

        // Also set num_sms if needed, or default is fine.
        // device_runtime->set_num_sms(...);

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
