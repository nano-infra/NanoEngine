#pragma once
#include "nanodeploy/csrc/core/common.h"
#include "nanodeploy/csrc/core/module.h"
#include "nanodeploy/csrc/ops/deep_gemm_ops.h"
#include <torch/torch.h>

namespace nanodeploy {
namespace layers {

// =============================================================================
// Base Linear template (BF16/FP16)
// =============================================================================
template<QuantType Q>
class Linear: public core::Module {
public:
    Linear(int in_features, int out_features, bool bias = false): in_features_(in_features), out_features_(out_features)
    {
        weight = this->register_parameter("weight", torch::empty({out_features, in_features}));
        if (bias) {
            this->bias = this->register_parameter("bias", torch::empty({out_features}));
        }
    }

    torch::Tensor forward(torch::Tensor input)
    {
        return torch::nn::functional::linear(input, weight, bias);
    }

    int           in_features_;
    int           out_features_;
    torch::Tensor weight;
    torch::Tensor bias;
};

// =============================================================================
// FP8_E4M3 Specialization - Uses DeepGEMM
// =============================================================================
template<>
class Linear<QuantType::FP8_E4M3>: public core::Module {
public:
    static constexpr int BLOCK_SIZE = 128;

    Linear(int in_features, int out_features, bool bias = false): in_features_(in_features), out_features_(out_features)
    {
        // FP8 weight: [out_features, in_features] as float8_e4m3fn
        weight = this->register_parameter("weight", torch::empty({out_features, in_features}, torch::kFloat8_e4m3fn));

        // Scale inverse: [out_features/block, in_features/block] as float32
        int scale_out = (out_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int scale_in  = (in_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
        weight_scale_inv =
            this->register_parameter("weight_scale_inv", torch::empty({scale_out, scale_in}, torch::kFloat32));

        if (bias) {
            this->bias = this->register_parameter("bias", torch::empty({out_features}, torch::kBFloat16));
        }
    }

    torch::Tensor forward(torch::Tensor input)
    {
        // Input: [batch, seq_len, in_features] or [num_tokens, in_features]
        // Ensure input is BF16 (model expects BF16, converts internally to FP8 for compute)
        if (input.scalar_type() != torch::kBFloat16) {
            input = input.to(torch::kBFloat16);
        }

        auto input_shape = input.sizes().vec();
        int  m           = 1;
        for (size_t i = 0; i < input_shape.size() - 1; ++i) {
            m *= input_shape[i];
        }
        int k = in_features_;
        int n = out_features_;

        // Flatten input to 2D: [M, K]
        auto input_2d = input.view({m, k}).contiguous();

        // Quantize input to FP8 dynamically (returns padded tensor)
        auto [input_fp8, input_scale, padded_m] = quant_fp8(input_2d);

        // Allocate PADDED output: [padded_M, N] as BF16
        auto output = torch::empty({padded_m, n}, input.options().dtype(torch::kBFloat16));

        // Call DeepGEMM FP8: D = A @ B.T
        // A = input_fp8 [padded_M, K], A_scale = [padded_M, K_tiles] (transposed view from quant_fp8)
        // B = weight [N, K], B_scale = [N_tiles, K_tiles] (directly from model, NO transpose)
        //
        // Python does NOT transpose weight_scale_inv, so we shouldn't either!

        // Debug prints
        std::cout << "[Linear::forward] Input: " << input_fp8.sizes() << " Scale: " << input_scale.sizes()
                  << " (contiguous=" << input_scale.is_contiguous() << ")" << std::endl;
        std::cout << "[Linear::forward] Weight: " << weight.sizes() << " Scale: " << weight_scale_inv.sizes()
                  << " (contiguous=" << weight_scale_inv.is_contiguous() << ")" << std::endl;
        ops::DeepGemmOps::fp8_gemm_nt({input_fp8, input_scale}, {weight, weight_scale_inv}, output);

        // Slice output back to original M
        output = output.slice(0, 0, m).contiguous();

        // Add bias if present
        if (bias.defined()) {
            output = output + bias;
        }

        // Reshape output to match input shape
        input_shape.back() = n;
        return output.view(input_shape);
    }

    // Quantize BF16 input to FP8 with per-row scaling (like quant_fp8_tma)
    // Each row has num_groups scale values, where num_groups = K / GROUP_SIZE
    // Input MUST be BF16 (caller ensures this)
    // Returns: (quantized_padded, scale, padded_m)
    std::tuple<torch::Tensor, torch::Tensor, int> quant_fp8(torch::Tensor input)
    {
        // Input: [M, K] as BF16
        int m = input.size(0);
        int k = input.size(1);

        // Calculate aligned dimensions
        int alignment  = BLOCK_SIZE;      // 128
        int num_groups = k / BLOCK_SIZE;  // K_tiles
        int aligned_m  = ((m + alignment - 1) / alignment) * alignment;

        // Pad M dimension if necessary
        torch::Tensor padded_input = input;
        if (m != aligned_m) {
            padded_input = torch::zeros({aligned_m, k}, input.options());
            padded_input.slice(0, 0, m).copy_(input);
        }

        // Allocate output
        auto quantized = torch::empty({aligned_m, k}, input.options().dtype(torch::kFloat8_e4m3fn));

        // Scale: Create as [num_groups, aligned_m] then transpose VIEW (not contiguous!)
        // This matches Python: scales = A.new_empty(num_groups, aligned_M).T
        auto scale_base = torch::empty({num_groups, aligned_m}, input.options().dtype(torch::kFloat32));
        auto scale      = scale_base.t();  // Transpose VIEW, NOT contiguous!

        // FP8 E4M3 max value
        constexpr float fp8_max  = 448.0f;
        float           rfp8_max = 1.0f / fp8_max;

        // Vectorized processing using tensor ops
        // Reshape input to [aligned_m, num_groups, GROUP_SIZE]
        auto input_reshaped = padded_input.view({aligned_m, num_groups, BLOCK_SIZE});

        // Compute max abs per group: [aligned_m, num_groups]
        auto abs_max = input_reshaped.abs().amax(-1);  // reduce last dim
        abs_max      = abs_max.clamp_min(1e-6f);

        // Scale = abs_max * rfp8_max
        scale.copy_(abs_max * rfp8_max);

        // Quantize all at once
        auto scale_expanded = abs_max.unsqueeze(-1);  // [aligned_m, num_groups, 1]
        auto scaled         = input_reshaped / scale_expanded * fp8_max;
        auto clamped        = scaled.clamp(-fp8_max, fp8_max);
        quantized.copy_(clamped.view({aligned_m, k}));

        // Scale inverse for DeepGEMM: use the transposed scale view directly
        return {quantized.to(torch::kFloat8_e4m3fn), scale, aligned_m};
    }

    int           in_features_;
    int           out_features_;
    torch::Tensor weight;            // FP8 [out, in]
    torch::Tensor weight_scale_inv;  // FP32 [out/block, in/block]
    torch::Tensor bias;              // BF16 [out]
};

// =============================================================================
// Parallel Linear classes
// =============================================================================
template<QuantType Q>
class ColumnParallelLinear: public Linear<Q> {
public:
    ColumnParallelLinear(int in_features, int out_features, bool bias = false):
        Linear<Q>(in_features, out_features, bias)
    {
    }
};

template<QuantType Q>
class RowParallelLinear: public Linear<Q> {
public:
    RowParallelLinear(int in_features, int out_features, bool bias = false): Linear<Q>(in_features, out_features, bias)
    {
    }
};

template<QuantType Q>
class QKVParallelLinear: public ColumnParallelLinear<Q> {
public:
    using ColumnParallelLinear<Q>::ColumnParallelLinear;
};

template<QuantType Q>
class MergedColumnParallelLinear: public ColumnParallelLinear<Q> {
public:
    using ColumnParallelLinear<Q>::ColumnParallelLinear;
};

}  // namespace layers
}  // namespace nanodeploy
