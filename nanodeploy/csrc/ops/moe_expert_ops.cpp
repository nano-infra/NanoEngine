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

}  // namespace ops
}  // namespace nanodeploy
