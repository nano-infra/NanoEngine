#pragma once

#include <torch/torch.h>
#include <vector>

namespace nanodeploy {

/**
 * MoeExpertRunner - Unified interface for MoE expert computation
 *
 * Provides two modes:
 * 1. Contiguous mode: For normal dispatch with scattered tokens (used in prefill)
 * 2. Masked mode: For low-latency dispatch with pre-organized 3D tensors (used in decode)
 */
class MoeExpertRunner {
public:
    /**
     * Compute experts using contiguous grouped GEMM
     * Used for normal dispatch path where tokens are scattered
     *
     * @param input           [num_tokens, hidden_size] - input tokens
     * @param topk_idx        [num_tokens, top_k] - local expert IDs for each token
     * @param topk_weights    [num_tokens, top_k] - routing weights
     * @param gate_up_weights [num_local_experts, intermediate*2, hidden_size]
     * @param down_weights    [num_local_experts, hidden_size, intermediate]
     * @param hidden_size     hidden dimension
     * @return                [num_tokens, hidden_size] - weighted sum of expert outputs
     */
    static torch::Tensor compute_contiguous(torch::Tensor input,
                                            torch::Tensor topk_idx,
                                            torch::Tensor topk_weights,
                                            torch::Tensor gate_up_weights,
                                            torch::Tensor down_weights,
                                            int           hidden_size);

    /**
     * Compute experts using masked grouped GEMM
     * Used for low-latency dispatch path where data is pre-organized as [G, M, K]
     *
     * @param recv_x          [num_local_experts, max_m, hidden_size] - received tokens per expert
     * @param masked_m        [num_local_experts] - actual token counts per expert
     * @param expected_m      expected average tokens per expert (for GEMM tuning)
     * @param gate_up_weights [num_local_experts, intermediate*2, hidden_size]
     * @param down_weights    [num_local_experts, hidden_size, intermediate]
     * @return                [num_local_experts, max_m, hidden_size] - expert outputs
     */
    static torch::Tensor compute_masked(torch::Tensor recv_x,
                                        torch::Tensor masked_m,
                                        int           expected_m,
                                        torch::Tensor gate_up_weights,
                                        torch::Tensor down_weights);
};

}  // namespace nanodeploy
