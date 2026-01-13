#pragma once

#include <memory>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "deep_ep.hpp"
#include "nanodeploy/worker/deep_ep_ipc.h"

namespace nanodeploy {

// Handle containing metadata needed for combine after dispatch
struct DispatchHandle {
    bool                                use_internode = false;
    int                                 hidden_size   = 0;
    deep_ep::Config                     config{132, 1024, 2048, 1024, 2048};  // Conservative default for large models
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
    torch::Tensor                recv_x;             // [num_recv_tokens, hidden_size]
    std::optional<torch::Tensor> recv_topk_idx;      // [num_recv_tokens, top_k] - local expert IDs
    std::optional<torch::Tensor> recv_topk_weights;  // [num_recv_tokens, top_k] - weights
    DispatchHandle               handle;             // metadata for combine
};

// Handle for Low Latency mode combine
struct LowLatencyDispatchHandle {
    torch::Tensor topk_idx;      // Original topk_idx for combine
    torch::Tensor topk_weights;  // Original topk_weights for combine
    torch::Tensor src_info;      // Source info from dispatch
    torch::Tensor layout_range;  // Layout range from dispatch
    int           num_max_dispatch_tokens_per_rank = 0;
    int           num_experts                      = 0;
    int           hidden_size                      = 0;
};

// Result of dispatch_low_latency
struct LowLatencyDispatchResult {
    torch::Tensor            recv_x;          // [num_local_experts, max_tokens_per_expert, hidden_size]
    torch::Tensor            masked_m;        // [num_local_experts] - actual token count per expert
    int                      expected_m = 0;  // Expected tokens per expert (for DeepGemm)
    LowLatencyDispatchHandle handle;
};

class DeepEPRunner {
public:
    DeepEPRunner() = default;
    ~DeepEPRunner();

    DeepEPInitResp init(const DeepEPInitReq& req);
    DeepEPInfoResp get_info();
    DeepEPSyncResp sync(const DeepEPSyncReq& req);
    DeepEPTestResp run_test(const DeepEPTestReq& req);

    // Normal mode dispatch: automatically selects intranode or internode
    // Args:
    //   hidden_states: [num_tokens, hidden_size] - input tokens
    //   topk_ids: [num_tokens, top_k] - global expert IDs
    //   topk_weights: [num_tokens, top_k] - weights (float32)
    //   num_experts: total number of experts
    //   expert_alignment: alignment for DeepGemm (default 128)
    // Returns:
    //   DispatchResult containing recv_x, recv_topk_idx, recv_topk_weights, and handle
    static DispatchResult dispatch_normal(deep_ep::Buffer* buffer,
                                          torch::Tensor    hidden_states,
                                          torch::Tensor    topk_ids,
                                          torch::Tensor    topk_weights,
                                          int              num_experts,
                                          int              expert_alignment = 128);

    // Normal mode combine: sends results back to original ranks
    // Args:
    //   expert_output: [num_recv_tokens, hidden_size] - weighted expert outputs
    //   handle: DispatchHandle from dispatch_normal
    // Returns:
    //   [num_tokens, hidden_size] - combined output
    static torch::Tensor
    combine_normal(deep_ep::Buffer* buffer, torch::Tensor expert_output, const DispatchHandle& handle);

    // Low Latency mode dispatch: for decode phase with few tokens
    // Args:
    //   hidden_states: [num_tokens, hidden_size] - input tokens
    //   topk_ids: [num_tokens, top_k] - global expert IDs (int64)
    //   topk_weights: [num_tokens, top_k] - weights (float32)
    //   num_max_dispatch_tokens_per_rank: max tokens per rank
    //   num_experts: total number of experts
    // Returns:
    //   LowLatencyDispatchResult with recv_x [G, M, K], masked_m, expected_m, handle
    static LowLatencyDispatchResult dispatch_low_latency(deep_ep::Buffer* buffer,
                                                         torch::Tensor    hidden_states,
                                                         torch::Tensor    topk_ids,
                                                         torch::Tensor    topk_weights,
                                                         int              num_max_dispatch_tokens_per_rank,
                                                         int              num_experts);

    // Low Latency mode combine: sends results back with weighted sum
    // Args:
    //   expert_output: [num_local_experts, max_m, hidden_size] - expert outputs
    //   handle: LowLatencyDispatchHandle from dispatch_low_latency
    // Returns:
    //   [num_tokens, hidden_size] - combined output
    static torch::Tensor
    combine_low_latency(deep_ep::Buffer* buffer, torch::Tensor expert_output, const LowLatencyDispatchHandle& handle);

private:
    std::unique_ptr<deep_ep::Buffer> buffer_;
    int                              rank_       = -1;
    int                              world_size_ = -1;
    int                              device_id_  = 0;
};

}  // namespace nanodeploy
