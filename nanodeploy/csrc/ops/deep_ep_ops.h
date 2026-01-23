#pragma once

#include <memory>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "deep_ep.hpp"

namespace nanodeploy {
namespace ops {

// Handle containing metadata needed for combine after dispatch
struct DispatchHandle {
    bool                                use_internode = false;
    int                                 hidden_size   = 0;
    deep_ep::Config                     config{132, 1024, 2048, 1024, 2048};
    std::optional<deep_ep::EventHandle> prev_event;

    // Intranode metadata
    torch::Tensor intra_rank_prefix_matrix;
    torch::Tensor intra_recv_channel_prefix_matrix;
    torch::Tensor intra_send_head;
    torch::Tensor intra_recv_src_idx;

    // Internode metadata
    torch::Tensor is_token_in_rank;
    torch::Tensor recv_rdma_channel_prefix_matrix;
    torch::Tensor recv_rdma_rank_prefix_sum;
    torch::Tensor recv_gbl_channel_prefix_matrix;
    torch::Tensor recv_src_meta;
    torch::Tensor send_rdma_head;
    torch::Tensor send_nvl_head;
};

// Result of dispatch_normal
struct DispatchResult {
    torch::Tensor                recv_x;
    std::optional<torch::Tensor> recv_topk_idx;
    std::optional<torch::Tensor> recv_topk_weights;
    DispatchHandle               handle;
};

// Result of dispatch_normal_fp8 (FP8 data + scales)
struct DispatchResultFP8 {
    torch::Tensor                recv_x;         // FP8 tensor
    torch::Tensor                recv_x_scales;  // scales for FP8 data
    std::optional<torch::Tensor> recv_topk_idx;
    std::optional<torch::Tensor> recv_topk_weights;
    DispatchHandle               handle;
};

// Handle for Low Latency mode combine
struct LowLatencyDispatchHandle {
    torch::Tensor topk_idx;
    torch::Tensor topk_weights;
    torch::Tensor src_info;
    torch::Tensor layout_range;
    int           num_max_dispatch_tokens_per_rank = 0;
    int           num_experts                      = 0;
    int           hidden_size                      = 0;
};

// Result of dispatch_low_latency
struct LowLatencyDispatchResult {
    torch::Tensor            recv_x;
    torch::Tensor            masked_m;
    int                      expected_m = 0;
    LowLatencyDispatchHandle handle;
};

/**
 * DeepEpOps - Static operations for DeepEP dispatch/combine
 */
class DeepEpOps {
public:
    // Normal mode dispatch (BF16)
    static DispatchResult dispatch_normal(torch::Tensor hidden_states,
                                          torch::Tensor topk_ids,
                                          torch::Tensor topk_weights,
                                          int           num_experts,
                                          int           expert_alignment = 128);

    // Normal mode dispatch (FP8 + scales) - reduces bandwidth by 50%
    static DispatchResultFP8 dispatch_normal_fp8(torch::Tensor x_fp8,
                                                 torch::Tensor x_scales,
                                                 torch::Tensor topk_ids,
                                                 torch::Tensor topk_weights,
                                                 int           num_experts,
                                                 int           expert_alignment = 128);

    // Normal mode combine
    static torch::Tensor combine_normal(torch::Tensor expert_output, const DispatchHandle& handle);

    // Low Latency mode dispatch
    static LowLatencyDispatchResult dispatch_low_latency(torch::Tensor hidden_states,
                                                         torch::Tensor topk_ids,
                                                         torch::Tensor topk_weights,
                                                         int           num_max_dispatch_tokens_per_rank,
                                                         int           num_experts,
                                                         int           ep_world_size = 0);

    // Low Latency mode combine
    static torch::Tensor combine_low_latency(torch::Tensor expert_output, const LowLatencyDispatchHandle& handle);
};

}  // namespace ops
}  // namespace nanodeploy
