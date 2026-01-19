#pragma once

#include <torch/torch.h>
#include <vector>

namespace nanodeploy {
namespace ops {

/**
 * MoeExpertOps - Static operations for MoE expert computation
 */
class MoeExpertOps {
public:
    /**
     * Compute experts using contiguous grouped GEMM
     * Used for normal dispatch path where tokens are scattered
     */
    static torch::Tensor compute_contiguous(torch::Tensor input,
                                            torch::Tensor topk_idx,
                                            torch::Tensor topk_weights,
                                            torch::Tensor gate_up_weights,
                                            torch::Tensor down_weights,
                                            int           hidden_size);

    // FP8 Overload
    static torch::Tensor compute_contiguous(torch::Tensor                           input,
                                            torch::Tensor                           topk_idx,
                                            torch::Tensor                           topk_weights,
                                            std::pair<torch::Tensor, torch::Tensor> gate_up,
                                            std::pair<torch::Tensor, torch::Tensor> down,
                                            int                                     hidden_size);

    /**
     * Compute experts using masked grouped GEMM
     * Used for low-latency dispatch path where data is pre-organized as [G, M, K]
     */
    static torch::Tensor compute_masked(torch::Tensor recv_x,
                                        torch::Tensor masked_m,
                                        int           expected_m,
                                        torch::Tensor gate_up_weights,
                                        torch::Tensor down_weights);

    // FP8 Overload
    static torch::Tensor compute_masked(torch::Tensor                           recv_x,
                                        torch::Tensor                           masked_m,
                                        int                                     expected_m,
                                        std::pair<torch::Tensor, torch::Tensor> gate_up,
                                        std::pair<torch::Tensor, torch::Tensor> down);

    /**
     * Compute experts using masked grouped GEMM (buffers provided)
     * Capture-friendly version that performs NO allocations.
     */
    static void compute_masked_out(torch::Tensor recv_x,
                                   torch::Tensor masked_m,
                                   int           expected_m,
                                   torch::Tensor gate_up_weights,
                                   torch::Tensor down_weights,
                                   torch::Tensor gateup_output,
                                   torch::Tensor down_output);
};

}  // namespace ops
}  // namespace nanodeploy
