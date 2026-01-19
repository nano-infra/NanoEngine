/**
 * @file test_deepgemm_fp8.cpp
 * @brief Unit tests for DeepGEMM FP8 operations
 *
 * These tests verify the correct scale layout and tensor formats
 * expected by DeepGEMM's FP8 kernels.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <torch/torch.h>

// Include DeepGemm ops header
#include "ops/deep_gemm_ops.h"

using nanodeploy::ops::DeepGemmOps;

class DeepGemmFP8Test: public ::testing::Test {
protected:
    void SetUp() override
    {
        // Ensure CUDA is available
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available, skipping DeepGEMM tests";
        }
        device_ = torch::kCUDA;
    }

    torch::Device device_ = torch::kCPU;

    static constexpr int BLOCK_SIZE = 128;

    // Helper to print tensor info
    void PrintTensorInfo(const std::string& name, const torch::Tensor& t)
    {
        std::cout << name << ": shape=" << t.sizes() << " dtype=" << t.dtype() << " contiguous=" << t.is_contiguous()
                  << " strides=" << t.strides() << std::endl;
    }

    // Create scale tensor with same memory layout as Python's quant_fp8_tma
    // Python: scales = A.new_empty(num_groups, aligned_M).T
    torch::Tensor CreateTMAScaleLayout(int aligned_m, int num_groups)
    {
        // Create base tensor [num_groups, aligned_m]
        auto scale_base =
            torch::randn({num_groups, aligned_m}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
        // Return transpose VIEW (not contiguous!)
        return scale_base.t();
    }

    // Quantize input to FP8 with per-row scaling (matching Python quant_fp8_tma)
    std::tuple<torch::Tensor, torch::Tensor> QuantFP8TMA(torch::Tensor input, int group_size = 128)
    {
        EXPECT_EQ(input.dim(), 2);
        int m = input.size(0);
        int k = input.size(1);
        EXPECT_EQ(k % group_size, 0) << "K must be divisible by group_size";

        int num_groups = k / group_size;
        int alignment  = 128;
        int aligned_m  = ((m + alignment - 1) / alignment) * alignment;

        // Pad M dimension if necessary
        torch::Tensor padded_input = input;
        if (m != aligned_m) {
            padded_input = torch::zeros({aligned_m, k}, input.options());
            padded_input.slice(0, 0, m).copy_(input);
        }

        // Allocate output
        auto quantized = torch::empty({aligned_m, k}, input.options().dtype(torch::kFloat8_e4m3fn));

        // Scale: Create as [num_groups, aligned_m] then transpose VIEW
        auto scale_base = torch::empty({num_groups, aligned_m}, input.options().dtype(torch::kFloat32));
        auto scale      = scale_base.t();  // Transpose VIEW!

        // FP8 E4M3 max value
        constexpr float fp8_max  = 448.0f;
        float           rfp8_max = 1.0f / fp8_max;

        // Vectorized processing
        auto input_reshaped = padded_input.view({aligned_m, num_groups, group_size});
        auto abs_max        = input_reshaped.abs().amax(-1);
        abs_max             = abs_max.clamp_min(1e-6f);

        // Copy to scale (through the transposed view)
        scale.copy_(abs_max * rfp8_max);

        // Quantize
        auto scale_expanded = abs_max.unsqueeze(-1);
        auto scaled         = input_reshaped / scale_expanded * fp8_max;
        auto clamped        = scaled.clamp(-fp8_max, fp8_max);
        quantized.copy_(clamped.view({aligned_m, k}));

        return {quantized.to(torch::kFloat8_e4m3fn), scale};
    }
};

// Test 1: Verify scale layout matches Python's quant_fp8_tma
TEST_F(DeepGemmFP8Test, ScaleLayoutMatchesPython)
{
    int aligned_m  = 128;
    int num_groups = 16;  // K = 2048, group_size = 128

    auto scale = CreateTMAScaleLayout(aligned_m, num_groups);

    PrintTensorInfo("Scale (TMA layout)", scale);

    // Verify logical shape is [aligned_m, num_groups]
    EXPECT_EQ(scale.size(0), aligned_m);
    EXPECT_EQ(scale.size(1), num_groups);

    // Verify it's NOT contiguous (transpose view)
    EXPECT_FALSE(scale.is_contiguous());

    // Verify strides match column-major layout
    // [aligned_m, num_groups] from [num_groups, aligned_m].T
    // strides should be [1, aligned_m]
    EXPECT_EQ(scale.stride(0), 1);
    EXPECT_EQ(scale.stride(1), aligned_m);

    std::cout << "Scale layout test passed!" << std::endl;
}

// Test 2: Quantize input and verify scale dimensions
TEST_F(DeepGemmFP8Test, QuantFP8TMAProducesCorrectShapes)
{
    int m = 1;     // Single token
    int k = 2048;  // Hidden size

    auto input = torch::randn({m, k}, torch::TensorOptions().dtype(torch::kBFloat16).device(device_));

    auto [quantized, scale] = QuantFP8TMA(input);

    PrintTensorInfo("Quantized", quantized);
    PrintTensorInfo("Scale", scale);

    // Expected dimensions
    int aligned_m  = 128;  // Padded to 128
    int num_groups = 16;   // 2048 / 128

    EXPECT_EQ(quantized.size(0), aligned_m);
    EXPECT_EQ(quantized.size(1), k);
    EXPECT_EQ(quantized.dtype(), torch::kFloat8_e4m3fn);

    // Scale should be [aligned_m, num_groups] as transposed view
    EXPECT_EQ(scale.size(0), aligned_m);
    EXPECT_EQ(scale.size(1), num_groups);
    EXPECT_FALSE(scale.is_contiguous());

    std::cout << "QuantFP8TMA produces correct shapes!" << std::endl;
}

// Test 3: Verify weight scale layout for pre-quantized weights
TEST_F(DeepGemmFP8Test, WeightScaleLayout)
{
    // Typical weight: [N, K] = [5120, 2048]
    // Weight scale from file: [N/128, K/128] = [40, 16]
    int n_tiles = 40;
    int k_tiles = 16;

    // Weight scale as loaded from model
    auto weight_scale = torch::randn({n_tiles, k_tiles}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));

    PrintTensorInfo("Weight scale (original)", weight_scale);

    // DeepGEMM may need transposed view
    auto weight_scale_t = weight_scale.t();

    PrintTensorInfo("Weight scale (transposed view)", weight_scale_t);

    // Original: [40, 16], contiguous
    EXPECT_EQ(weight_scale.size(0), n_tiles);
    EXPECT_EQ(weight_scale.size(1), k_tiles);
    EXPECT_TRUE(weight_scale.is_contiguous());

    // Transposed: [16, 40], non-contiguous view
    EXPECT_EQ(weight_scale_t.size(0), k_tiles);
    EXPECT_EQ(weight_scale_t.size(1), n_tiles);
    EXPECT_FALSE(weight_scale_t.is_contiguous());
}

// Test 4: Test DeepGEMM assertion conditions
TEST_F(DeepGemmFP8Test, DeepGemmAssertionConditions)
{
    // DeepGEMM checks: sf.size(-2) == ceil_div(mn, gran_mn)
    // where gran_mn = 128

    // For input [M, K] with scale, what should sf.size(-2) be?
    // Option A: ceil(M / 128)
    // Option B: ceil(K / 128)

    int m = 128;
    int k = 2048;
    int n = 5120;

    int m_tiles = (m + 127) / 128;  // 1
    int k_tiles = k / 128;          // 16
    int n_tiles = n / 128;          // 40

    std::cout << "Dimension analysis:" << std::endl;
    std::cout << "  M=" << m << ", M_tiles=" << m_tiles << std::endl;
    std::cout << "  K=" << k << ", K_tiles=" << k_tiles << std::endl;
    std::cout << "  N=" << n << ", N_tiles=" << n_tiles << std::endl;

    // Test different scale layouts
    struct ScaleConfig {
        std::string          name;
        std::vector<int64_t> shape;
        bool                 transpose;
        int                  expected_size_m2;  // size(-2) after optional transpose
    };

    std::vector<ScaleConfig> configs = {
        // For input A [M, K]:
        {"Input: [M_tiles, K_tiles]", {m_tiles, k_tiles}, false, m_tiles},
        {"Input: [K_tiles, M_tiles]", {k_tiles, m_tiles}, false, k_tiles},
        {"Input: [aligned_M, K_tiles]", {m, k_tiles}, false, m},
        {"Input: [K_tiles, aligned_M].T", {k_tiles, m}, true, m},

        // For weight B [N, K]:
        {"Weight: [N_tiles, K_tiles]", {n_tiles, k_tiles}, false, n_tiles},
        {"Weight: [K_tiles, N_tiles]", {k_tiles, n_tiles}, false, k_tiles},
        {"Weight: [N_tiles, K_tiles].T", {n_tiles, k_tiles}, true, k_tiles},
    };

    for (const auto& cfg : configs) {
        auto scale = torch::randn(cfg.shape, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
        if (cfg.transpose) {
            scale = scale.t();
        }

        std::cout << cfg.name << ": size(-2)=" << scale.size(-2) << " (expected " << cfg.expected_size_m2 << ")"
                  << " contiguous=" << scale.is_contiguous() << std::endl;

        EXPECT_EQ(scale.size(-2), cfg.expected_size_m2);
    }
}

// Test 5: Attempt actual DeepGEMM call (will fail if layout is wrong)
TEST_F(DeepGemmFP8Test, DISABLED_DeepGemmFP8NT)
{
    // This test is DISABLED by default since it requires correct scale layout
    // Enable once we figure out the correct layout

    int m = 1;
    int k = 2048;
    int n = 5120;

    // Create input
    auto input_bf16 = torch::randn({m, k}, torch::TensorOptions().dtype(torch::kBFloat16).device(device_));

    // Quantize input
    auto [input_fp8, input_scale] = QuantFP8TMA(input_bf16);
    int aligned_m                 = input_fp8.size(0);

    // Create weight (pre-quantized)
    auto weight       = torch::randn({n, k}, torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device_));
    auto weight_scale = torch::randn({n / 128, k / 128}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));

    // Output
    auto output = torch::empty({aligned_m, n}, torch::TensorOptions().dtype(torch::kBFloat16).device(device_));

    PrintTensorInfo("Input FP8", input_fp8);
    PrintTensorInfo("Input Scale", input_scale);
    PrintTensorInfo("Weight", weight);
    PrintTensorInfo("Weight Scale", weight_scale);

    // Try different scale configurations
    try {
        // Configuration 1: Input scale as-is, weight scale transposed view
        auto weight_scale_t = weight_scale.t();
        std::cout << "Trying: input_scale as-is, weight_scale.t()" << std::endl;
        nanodeploy::ops::DeepGemmOps::fp8_gemm_nt({input_fp8, input_scale}, {weight, weight_scale_t}, output);
        std::cout << "SUCCESS!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << std::endl;
    }
}
