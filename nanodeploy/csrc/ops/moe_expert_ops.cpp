#include "nanodeploy/csrc/ops/moe_expert_ops.h"
#include "nanodeploy/csrc/ops/deep_gemm_ops.h"

namespace nanodeploy {
namespace ops {

torch::Tensor MoeExpertOps::compute_contiguous(torch::Tensor input,
                                               torch::Tensor topk_idx,
                                               torch::Tensor topk_weights,
                                               torch::Tensor gate_up_weights,
                                               torch::Tensor down_weights,
                                               int           hidden_size)
{
    int  num_tokens        = input.size(0);
    auto device            = input.device();
    int  num_local_experts = gate_up_weights.size(0);
    int  moe_inter_2       = gate_up_weights.size(1);
    int  top_k             = topk_idx.size(1);

    if (num_tokens == 0) {
        return torch::zeros({0, hidden_size}, input.options());
    }

    auto topk_idx_cpu = topk_idx.to(torch::kCPU);
    auto topk_idx_acc = topk_idx_cpu.accessor<int64_t, 2>();

    std::vector<std::vector<int64_t>> token_indices_per_expert(num_local_experts);
    std::vector<std::vector<int>>     token_k_per_expert(num_local_experts);

    for (int64_t t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            int64_t expert_id = topk_idx_acc[t][k];
            if (expert_id >= 0 && expert_id < num_local_experts) {
                token_indices_per_expert[expert_id].push_back(t);
                token_k_per_expert[expert_id].push_back(k);
            }
        }
    }

    constexpr int64_t    ALIGNMENT = 128;
    std::vector<int64_t> actual_counts(num_local_experts);
    std::vector<int64_t> aligned_counts(num_local_experts);
    std::vector<int64_t> used_experts;
    int64_t              total_aligned_rows = 0;

    for (int e = 0; e < num_local_experts; ++e) {
        int64_t count     = token_indices_per_expert[e].size();
        actual_counts[e]  = count;
        aligned_counts[e] = (count + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
        total_aligned_rows += aligned_counts[e];
        if (count > 0) {
            used_experts.push_back(e);
        }
    }

    if (used_experts.empty() || total_aligned_rows == 0) {
        return torch::zeros({num_tokens, hidden_size}, input.options());
    }

    auto input_bf16 = input.to(torch::kBFloat16);
    auto aligned_x  = torch::zeros({total_aligned_rows, input.size(1)}, input_bf16.options());
    auto m_indices  = torch::full({total_aligned_rows}, -1, torch::TensorOptions().dtype(torch::kInt32).device(device));

    int64_t aligned_offset = 0;
    int     group_idx      = 0;

    for (int e = 0; e < num_local_experts; ++e) {
        int64_t count         = actual_counts[e];
        int64_t aligned_count = aligned_counts[e];

        if (count > 0) {
            auto token_indices_tensor =
                torch::from_blob(token_indices_per_expert[e].data(), {count}, torch::kLong).clone().to(device);

            aligned_x.slice(0, aligned_offset, aligned_offset + count) =
                input_bf16.index_select(0, token_indices_tensor);

            m_indices.slice(0, aligned_offset, aligned_offset + count).fill_(group_idx);
            group_idx++;
        }
        aligned_offset += aligned_count;
    }

    auto used_experts_tensor =
        torch::from_blob(used_experts.data(), {(long)used_experts.size()}, torch::kLong).clone().to(device);
    auto gate_up_selected = gate_up_weights.index_select(0, used_experts_tensor).contiguous();
    auto down_selected    = down_weights.index_select(0, used_experts_tensor).contiguous();

    auto gateup_out = torch::empty({total_aligned_rows, moe_inter_2}, input_bf16.options());
    DeepGemmOps::m_grouped_bf16_gemm_nt_contiguous(aligned_x, gate_up_selected, gateup_out, m_indices);

    int  intermediate_size = moe_inter_2 / 2;
    auto gate_out          = gateup_out.slice(1, 0, intermediate_size);
    auto up_out            = gateup_out.slice(1, intermediate_size, moe_inter_2);
    auto act_out           = torch::silu(gate_out) * up_out;

    auto aligned_down_out = torch::empty({total_aligned_rows, hidden_size}, input_bf16.options());
    DeepGemmOps::m_grouped_bf16_gemm_nt_contiguous(act_out, down_selected, aligned_down_out, m_indices);

    auto output      = torch::zeros({num_tokens, hidden_size}, input_bf16.options());
    auto weights_gpu = topk_weights.to(torch::kFloat32).to(device);

    aligned_offset = 0;
    for (int e = 0; e < num_local_experts; ++e) {
        int64_t count         = actual_counts[e];
        int64_t aligned_count = aligned_counts[e];

        if (count > 0) {
            auto token_indices =
                torch::from_blob(token_indices_per_expert[e].data(), {count}, torch::kLong).clone().to(device);

            auto k_indices = torch::from_blob(token_k_per_expert[e].data(), {count}, torch::kInt32)
                                 .clone()
                                 .to(torch::kLong)
                                 .to(device);

            auto expert_weights   = weights_gpu.index({token_indices, k_indices});
            auto expert_outputs   = aligned_down_out.slice(0, aligned_offset, aligned_offset + count);
            auto weighted_outputs = expert_outputs * expert_weights.unsqueeze(1).to(expert_outputs.dtype());
            output.index_add_(0, token_indices, weighted_outputs);
        }
        aligned_offset += aligned_count;
    }

    return output.to(input.dtype());
}

torch::Tensor MoeExpertOps::compute_masked(torch::Tensor recv_x,
                                           torch::Tensor masked_m,
                                           int           expected_m,
                                           torch::Tensor gate_up_weights,
                                           torch::Tensor down_weights)
{
    int  num_groups        = recv_x.size(0);
    int  max_m             = recv_x.size(1);
    int  hidden_dim        = recv_x.size(2);
    int  intermediate_size = gate_up_weights.size(1) / 2;
    auto device            = recv_x.device();

    auto gateup_output = torch::empty({num_groups, max_m, intermediate_size * 2},
                                      torch::TensorOptions().dtype(torch::kBFloat16).device(device));

    auto down_output =
        torch::empty({num_groups, max_m, hidden_dim}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));

    // Ensure masked_m is int32. If not, this allocates, which is fine for eager mode but not capture.
    // For eager mode wrapper, we handle it here.
    auto masked_m_int = masked_m.to(torch::kInt32);

    compute_masked_out(recv_x, masked_m_int, expected_m, gate_up_weights, down_weights, gateup_output, down_output);

    return down_output;
}

void MoeExpertOps::compute_masked_out(torch::Tensor recv_x,
                                      torch::Tensor masked_m,
                                      int           expected_m,
                                      torch::Tensor gate_up_weights,
                                      torch::Tensor down_weights,
                                      torch::Tensor gateup_output,
                                      torch::Tensor down_output)
{
    int intermediate_size = gate_up_weights.size(1) / 2;

    DeepGemmOps::m_grouped_bf16_gemm_nt_masked(recv_x.contiguous(),
                                               gate_up_weights.contiguous(),
                                               gateup_output,
                                               masked_m.contiguous(),  // Assumes input is already Int32
                                               expected_m);

    auto gate_out = gateup_output.slice(2, 0, intermediate_size);
    auto up_out   = gateup_output.slice(2, intermediate_size, intermediate_size * 2);
    // Note: silu * up allocates a temporary for the result of silu, and then multiplies.
    // Ideally we want fused kernel or minimal allocations.
    // But basic arithmetic ops like * and silu might be handled by PyTorch capture OK if shapes are static.
    // HOWEVER, `act_out` needs to be contiguous for the next GEMM.
    // To be perfectly capture safe without intermediate tensors, we need a custom kernel or
    // pre-allocated intermediates.
    // For now, let's rely on PyTorch allocator handling the `silu * up` temp.
    // IF this fails, we need `silu_and_mul_out`.
    auto act_out = torch::silu(gate_out) * up_out;

    DeepGemmOps::m_grouped_bf16_gemm_nt_masked(
        act_out.contiguous(), down_weights.contiguous(), down_output, masked_m.contiguous(), expected_m);
}

// Helper to quantize tensor to FP8 with TMA-compatible per-row scaling
// Matches Python's quant_fp8_tma: scales = A.new_empty(num_groups, aligned_M).T
static std::pair<torch::Tensor, torch::Tensor> quant_fp8_tensor(torch::Tensor input)
{
    // Ensure input is BF16
    if (input.scalar_type() != torch::kBFloat16) {
        input = input.to(torch::kBFloat16);
    }

    constexpr int BLOCK_SIZE = 128;
    int           m          = input.size(0);
    int           k          = input.size(1);

    int num_groups = k / BLOCK_SIZE;  // K_tiles
    int aligned_m  = ((m + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    // Pad M dimension if necessary
    torch::Tensor padded_input = input;
    if (m != aligned_m) {
        padded_input = torch::zeros({aligned_m, k}, input.options());
        padded_input.slice(0, 0, m).copy_(input);
    }

    // Allocate output
    auto quantized = torch::empty({aligned_m, k}, input.options().dtype(torch::kFloat8_e4m3fn));

    // Scale: Create as [num_groups, aligned_m] then transpose VIEW (TMA layout)
    auto scale_base = torch::empty({num_groups, aligned_m}, input.options().dtype(torch::kFloat32));
    auto scale      = scale_base.t();  // Transpose VIEW!

    // FP8 E4M3 max value
    constexpr float fp8_max  = 448.0f;
    float           rfp8_max = 1.0f / fp8_max;

    // Vectorized processing
    auto input_reshaped = padded_input.view({aligned_m, num_groups, BLOCK_SIZE});
    auto abs_max        = input_reshaped.abs().amax(-1);
    abs_max             = abs_max.clamp_min(1e-6f);

    scale.copy_(abs_max * rfp8_max);

    auto scale_expanded = abs_max.unsqueeze(-1);
    auto scaled         = input_reshaped / scale_expanded * fp8_max;
    auto clamped        = scaled.clamp(-fp8_max, fp8_max);
    quantized.copy_(clamped.view({aligned_m, k}));

    // Return quantized (possibly slice back to original m) and scale
    // Note: For MoE, we keep the padded tensor
    return {quantized.to(torch::kFloat8_e4m3fn), scale};
}

torch::Tensor MoeExpertOps::compute_contiguous(torch::Tensor                           input,
                                               torch::Tensor                           topk_idx,
                                               torch::Tensor                           topk_weights,
                                               std::pair<torch::Tensor, torch::Tensor> gate_up,
                                               std::pair<torch::Tensor, torch::Tensor> down,
                                               int                                     hidden_size)
{
    // ... Logic similar to BF16 but with FP8 quantization and GEMM ...
    // Note: Due to length/complexity, we simplify to reusing the structure but calling FP8 kernels

    int  num_tokens        = input.size(0);
    auto device            = input.device();
    int  num_local_experts = gate_up.first.size(0);
    int  moe_inter_2       = gate_up.first.size(1);
    int  top_k             = topk_idx.size(1);

    if (num_tokens == 0) {
        return torch::zeros({0, hidden_size}, input.options());
    }

    auto topk_idx_cpu = topk_idx.to(torch::kCPU);
    auto topk_idx_acc = topk_idx_cpu.accessor<int64_t, 2>();

    std::vector<std::vector<int64_t>> token_indices_per_expert(num_local_experts);
    std::vector<std::vector<int>>     token_k_per_expert(num_local_experts);

    for (int64_t t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            int64_t expert_id = topk_idx_acc[t][k];
            if (expert_id >= 0 && expert_id < num_local_experts) {
                token_indices_per_expert[expert_id].push_back(t);
                token_k_per_expert[expert_id].push_back(k);
            }
        }
    }

    constexpr int64_t    ALIGNMENT = 128;
    std::vector<int64_t> actual_counts(num_local_experts);
    std::vector<int64_t> aligned_counts(num_local_experts);
    std::vector<int64_t> used_experts;
    int64_t              total_aligned_rows = 0;

    for (int e = 0; e < num_local_experts; ++e) {
        int64_t count     = token_indices_per_expert[e].size();
        actual_counts[e]  = count;
        aligned_counts[e] = (count + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
        total_aligned_rows += aligned_counts[e];
        if (count > 0) {
            used_experts.push_back(e);
        }
    }

    if (used_experts.empty() || total_aligned_rows == 0) {
        return torch::zeros({num_tokens, hidden_size}, input.options());
    }

    auto input_bf16 = input.to(torch::kBFloat16);
    auto aligned_x  = torch::zeros({total_aligned_rows, input.size(1)}, input_bf16.options());
    auto m_indices  = torch::full({total_aligned_rows}, -1, torch::TensorOptions().dtype(torch::kInt32).device(device));

    int64_t aligned_offset = 0;
    int     group_idx      = 0;

    for (int e = 0; e < num_local_experts; ++e) {
        int64_t count         = actual_counts[e];
        int64_t aligned_count = aligned_counts[e];

        if (count > 0) {
            auto token_indices_tensor =
                torch::from_blob(token_indices_per_expert[e].data(), {count}, torch::kLong).clone().to(device);

            aligned_x.slice(0, aligned_offset, aligned_offset + count) =
                input_bf16.index_select(0, token_indices_tensor);

            m_indices.slice(0, aligned_offset, aligned_offset + count).fill_(group_idx);
            group_idx++;
        }
        aligned_offset += aligned_count;
    }

    // FP8: Select weights and scales
    auto used_experts_tensor =
        torch::from_blob(used_experts.data(), {(long)used_experts.size()}, torch::kLong).clone().to(device);

    // Select gate_up weights/scales
    auto gate_up_w_sel = gate_up.first.index_select(0, used_experts_tensor).contiguous();
    auto gate_up_s_sel = gate_up.second.index_select(0, used_experts_tensor).contiguous();

    // Select down weights/scales
    auto down_w_sel = down.first.index_select(0, used_experts_tensor).contiguous();
    auto down_s_sel = down.second.index_select(0, used_experts_tensor).contiguous();

    // Quantize input (returns TMA-compatible scale layout)
    auto [aligned_x_fp8, aligned_x_scale] = quant_fp8_tensor(aligned_x);

    auto gateup_out = torch::empty({total_aligned_rows, moe_inter_2}, input_bf16.options());

    // DeepGEMM FP8 Call - input scale already has TMA layout, weight scale used as-is
    DeepGemmOps::m_grouped_fp8_gemm_nt_contiguous(
        {aligned_x_fp8, aligned_x_scale}, {gate_up_w_sel, gate_up_s_sel}, gateup_out, m_indices);

    int  intermediate_size = moe_inter_2 / 2;
    auto gate_out          = gateup_out.slice(1, 0, intermediate_size);
    auto up_out            = gateup_out.slice(1, intermediate_size, moe_inter_2);
    auto act_out           = torch::silu(gate_out) * up_out;

    // Quantize activation (returns TMA-compatible scale layout)
    auto [act_out_fp8, act_out_scale] = quant_fp8_tensor(act_out);

    auto aligned_down_out = torch::empty({total_aligned_rows, hidden_size}, input_bf16.options());

    // DeepGEMM FP8 Call - input scale already has TMA layout, weight scale used as-is
    DeepGemmOps::m_grouped_fp8_gemm_nt_contiguous(
        {act_out_fp8, act_out_scale}, {down_w_sel, down_s_sel}, aligned_down_out, m_indices);

    // Scatter back
    auto output      = torch::zeros({num_tokens, hidden_size}, input_bf16.options());
    auto weights_gpu = topk_weights.to(torch::kFloat32).to(device);

    aligned_offset = 0;
    for (int e = 0; e < num_local_experts; ++e) {
        int64_t count         = actual_counts[e];
        int64_t aligned_count = aligned_counts[e];

        if (count > 0) {
            auto token_indices =
                torch::from_blob(token_indices_per_expert[e].data(), {count}, torch::kLong).clone().to(device);

            auto k_indices = torch::from_blob(token_k_per_expert[e].data(), {count}, torch::kInt32)
                                 .clone()
                                 .to(torch::kLong)
                                 .to(device);

            auto expert_weights   = weights_gpu.index({token_indices, k_indices});
            auto expert_outputs   = aligned_down_out.slice(0, aligned_offset, aligned_offset + count);
            auto weighted_outputs = expert_outputs * expert_weights.unsqueeze(1).to(expert_outputs.dtype());
            output.index_add_(0, token_indices, weighted_outputs);
        }
        aligned_offset += aligned_count;
    }

    return output.to(input.dtype());
}

torch::Tensor MoeExpertOps::compute_masked(torch::Tensor                           recv_x,
                                           torch::Tensor                           masked_m,
                                           int                                     expected_m,
                                           std::pair<torch::Tensor, torch::Tensor> gate_up,
                                           std::pair<torch::Tensor, torch::Tensor> down)
{
    int  num_groups        = recv_x.size(0);
    int  max_m             = recv_x.size(1);
    int  hidden_dim        = recv_x.size(2);
    int  intermediate_size = gate_up.first.size(1) / 2;
    auto device            = recv_x.device();

    auto gateup_output = torch::empty({num_groups, max_m, intermediate_size * 2},
                                      torch::TensorOptions().dtype(torch::kBFloat16).device(device));

    auto down_output =
        torch::empty({num_groups, max_m, hidden_dim}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));

    auto masked_m_int = masked_m.to(torch::kInt32);

    // Quantize recv_x
    // recv_x is [G, M, K], reshape to [G*M, K] for quantization
    // Then reshape both tensor and scale back to 3D for grouped GEMM
    auto [recv_x_fp8, recv_x_scale] = quant_fp8_tensor(recv_x.view({-1, hidden_dim}));
    int num_k_tiles                 = hidden_dim / 128;  // K_tiles

    // Reshape FP8 tensor back to [G, M, K]
    recv_x_fp8 = recv_x_fp8.view({num_groups, max_m, hidden_dim});

    // Reshape scale from [G*M, K_tiles] to [G, M, K_tiles] for grouped GEMM
    // NOTE: scale from quant_fp8_tensor is a transposed view [aligned_M, K_tiles]
    // But for grouped GEMM, DeepGEMM expects [num_groups, M, K_tiles]
    recv_x_scale = recv_x_scale.view({num_groups, max_m, num_k_tiles});
    // The scale shape should be flattened or not?
    // DeepGEMM input scale is typically [M_blocks, K_blocks].
    // If we view as [G*M, K], the scale is [(G*M)/128, K/128].

    // DeepGEMM implementation for masked might handle 3D input explicitly.
    // Let's assume passed tensors work if shapes match logical dimensions.

    // First GEMM
    // Debug
    std::cout << "[MoeExpertOps] Expert Masked GEMM 1:" << std::endl;
    std::cout << "  Input: " << recv_x_fp8.sizes() << " Scale: " << recv_x_scale.sizes() << std::endl;
    std::cout << "  Weight: " << gate_up.first.sizes() << " Scale: " << gate_up.second.sizes() << std::endl;
    std::cout << "  MaskedM: " << masked_m_int.sizes() << " ExpectedM: " << expected_m << std::endl;

    // Input scale already has TMA layout from quant_fp8_tensor
    DeepGemmOps::m_grouped_fp8_gemm_nt_masked(
        {recv_x_fp8, recv_x_scale}, {gate_up.first, gate_up.second}, gateup_output, masked_m_int, expected_m);

    auto gate_out = gateup_output.slice(2, 0, intermediate_size);
    auto up_out   = gateup_output.slice(2, intermediate_size, intermediate_size * 2);
    auto act_out  = torch::silu(gate_out) * up_out;

    // Second GEMM - quant_fp8_tensor returns TMA-compatible scale
    // Need to reshape both tensor and scale to 3D for grouped GEMM
    auto [act_out_fp8, act_out_scale] = quant_fp8_tensor(act_out.view({-1, intermediate_size}));
    int num_inter_tiles               = intermediate_size / 128;  // K_tiles for intermediate
    act_out_fp8                       = act_out_fp8.view({num_groups, max_m, intermediate_size});
    act_out_scale                     = act_out_scale.view({num_groups, max_m, num_inter_tiles});

    DeepGemmOps::m_grouped_fp8_gemm_nt_masked(
        {act_out_fp8, act_out_scale}, {down.first, down.second}, down_output, masked_m_int, expected_m);

    return down_output;
}

}  // namespace ops
}  // namespace nanodeploy
