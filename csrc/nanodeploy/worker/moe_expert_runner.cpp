#include "nanodeploy/worker/moe_expert_runner.h"
#include "nanodeploy/worker/deep_gemm_runner.h"

namespace nanodeploy {

torch::Tensor MoeExpertRunner::compute_contiguous(torch::Tensor input,
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

    // Convert to CPU for routing logic
    auto topk_idx_cpu = topk_idx.to(torch::kCPU);
    auto topk_idx_acc = topk_idx_cpu.accessor<int64_t, 2>();

    // 1. Count tokens per expert and build scatter indices
    std::vector<std::vector<int64_t>> token_indices_per_expert(num_local_experts);
    std::vector<std::vector<int>>     token_k_per_expert(num_local_experts);

    for (int64_t t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            int64_t expert_id = topk_idx_acc[t][k];
            // expert_id < 0 means not assigned (e.g., -1 from DeepEP dispatch)
            if (expert_id >= 0 && expert_id < num_local_experts) {
                token_indices_per_expert[expert_id].push_back(t);
                token_k_per_expert[expert_id].push_back(k);
            }
        }
    }

    // 2. Compute aligned counts (128 alignment for DeepGemm)
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

    // 3. Scatter: build aligned input tensor and m_indices
    auto input_bf16 = input.to(torch::kBFloat16);
    auto aligned_x  = torch::zeros({total_aligned_rows, input.size(1)}, input_bf16.options());
    auto m_indices  = torch::full({total_aligned_rows}, -1, torch::TensorOptions().dtype(torch::kInt32).device(device));

    int64_t aligned_offset = 0;
    int     group_idx      = 0;

    for (int e = 0; e < num_local_experts; ++e) {
        int64_t count         = actual_counts[e];
        int64_t aligned_count = aligned_counts[e];

        if (count > 0) {
            // Copy tokens for this expert to aligned positions
            auto token_indices_tensor =
                torch::from_blob(token_indices_per_expert[e].data(), {count}, torch::kLong).clone().to(device);

            aligned_x.slice(0, aligned_offset, aligned_offset + count) =
                input_bf16.index_select(0, token_indices_tensor);

            // Set m_indices for valid rows (group index)
            m_indices.slice(0, aligned_offset, aligned_offset + count).fill_(group_idx);
            group_idx++;
        }
        aligned_offset += aligned_count;
    }

    // 4. Select weights for used experts only
    auto used_experts_tensor =
        torch::from_blob(used_experts.data(), {(long)used_experts.size()}, torch::kLong).clone().to(device);
    auto gate_up_selected = gate_up_weights.index_select(0, used_experts_tensor).contiguous();
    auto down_selected    = down_weights.index_select(0, used_experts_tensor).contiguous();

    // 5. GateUp GEMM
    auto gateup_out = torch::empty({total_aligned_rows, moe_inter_2}, input_bf16.options());
    DeepGemmRunner::m_grouped_bf16_gemm_nt_contiguous(aligned_x, gate_up_selected, gateup_out, m_indices);

    // 6. Activation (SiLU and Mul)
    int  intermediate_size = moe_inter_2 / 2;
    auto gate_out          = gateup_out.slice(1, 0, intermediate_size);
    auto up_out            = gateup_out.slice(1, intermediate_size, moe_inter_2);
    auto act_out           = torch::silu(gate_out) * up_out;

    // 7. Down GEMM
    auto aligned_down_out = torch::empty({total_aligned_rows, hidden_size}, input_bf16.options());
    DeepGemmRunner::m_grouped_bf16_gemm_nt_contiguous(act_out, down_selected, aligned_down_out, m_indices);

    // 8. Gather with weighted sum (scatter-add)
    auto output      = torch::zeros({num_tokens, hidden_size}, input_bf16.options());
    auto weights_gpu = topk_weights.to(torch::kFloat32).to(device);

    aligned_offset = 0;
    for (int e = 0; e < num_local_experts; ++e) {
        int64_t count         = actual_counts[e];
        int64_t aligned_count = aligned_counts[e];

        if (count > 0) {
            // Get token indices and k indices for this expert
            auto token_indices =
                torch::from_blob(token_indices_per_expert[e].data(), {count}, torch::kLong).clone().to(device);

            auto k_indices = torch::from_blob(token_k_per_expert[e].data(), {count}, torch::kInt32)
                                 .clone()
                                 .to(torch::kLong)
                                 .to(device);

            // Get weights for these token-expert pairs
            auto expert_weights = weights_gpu.index({token_indices, k_indices});  // [count]

            // Get expert outputs
            auto expert_outputs = aligned_down_out.slice(0, aligned_offset, aligned_offset + count);

            // Weight and scatter-add
            auto weighted_outputs = expert_outputs * expert_weights.unsqueeze(1).to(expert_outputs.dtype());
            output.index_add_(0, token_indices, weighted_outputs);
        }
        aligned_offset += aligned_count;
    }

    return output.to(input.dtype());
}

torch::Tensor MoeExpertRunner::compute_masked(torch::Tensor recv_x,
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

    // 1. GateUp GEMM with masking
    auto gateup_output = torch::empty({num_groups, max_m, intermediate_size * 2},
                                      torch::TensorOptions().dtype(torch::kBFloat16).device(device));

    DeepGemmRunner::m_grouped_bf16_gemm_nt_masked(recv_x.contiguous(),
                                                  gate_up_weights.contiguous(),
                                                  gateup_output,
                                                  masked_m.to(torch::kInt32).contiguous(),
                                                  expected_m,
                                                  "nk");

    // 2. Activation (SiLU and Mul)
    auto gate_out = gateup_output.slice(2, 0, intermediate_size);
    auto up_out   = gateup_output.slice(2, intermediate_size, intermediate_size * 2);
    auto act_out  = torch::silu(gate_out) * up_out;

    // 3. Down GEMM with masking
    auto down_output =
        torch::empty({num_groups, max_m, hidden_dim}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));

    DeepGemmRunner::m_grouped_bf16_gemm_nt_masked(act_out.contiguous(),
                                                  down_weights.contiguous(),
                                                  down_output,
                                                  masked_m.to(torch::kInt32).contiguous(),
                                                  expected_m,
                                                  "nk");

    return down_output;
}

}  // namespace nanodeploy
