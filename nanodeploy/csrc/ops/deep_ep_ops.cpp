#include "nanodeploy/csrc/ops/deep_ep_ops.h"
#include "nanodeploy/csrc/context/distributed_context.h"
#include "nanodeploy/csrc/ops/deep_ep_utils.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace nanodeploy {
namespace ops {

DispatchResult DeepEpOps::dispatch_normal(deep_ep::Buffer* buffer,
                                          torch::Tensor    hidden_states,
                                          torch::Tensor    topk_ids,
                                          torch::Tensor    topk_weights,
                                          int              num_experts,
                                          int              expert_alignment)
{
    auto device = hidden_states.device();

    // Ensure hidden_states is BF16 (DeepEP intranode dispatch doesn't support FP8 directly)
    if (hidden_states.scalar_type() != torch::kBFloat16) {
        hidden_states = hidden_states.to(torch::kBFloat16);
    }

    auto hidden_flat = hidden_states.view({-1, hidden_states.size(-1)});

    std::optional<deep_ep::EventHandle> prev_event = std::nullopt;
    auto layout                       = buffer->get_dispatch_layout(topk_ids, num_experts, prev_event, false, false);
    auto num_tokens_per_rank          = std::get<0>(layout);
    auto num_tokens_per_rdma_rank     = std::get<1>(layout);
    auto num_tokens_per_expert_global = std::get<2>(layout);
    auto is_token_in_rank             = std::get<3>(layout);

    int num_sms    = 0;
    int device_idx = device.is_cuda() ? device.index() : 0;
    if (device_idx < 0)
        device_idx = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_idx);
    if (num_sms <= 0)
        num_sms = 132;

    deep_ep::Config config(num_sms, 6, 256, 6, 128);

    int  num_rdma_ranks = buffer->get_num_rdma_ranks();
    bool use_internode  = (num_rdma_ranks > 1);

    DispatchResult result;
    result.handle.use_internode    = use_internode;
    result.handle.hidden_size      = hidden_flat.size(1);
    result.handle.config           = config;
    result.handle.prev_event       = prev_event;
    result.handle.is_token_in_rank = is_token_in_rank;

    if (use_internode) {
        auto dispatch_ret = buffer->internode_dispatch(hidden_flat.contiguous(),
                                                       std::nullopt,
                                                       topk_ids,
                                                       topk_weights,
                                                       num_tokens_per_rank,
                                                       num_tokens_per_rdma_rank,
                                                       is_token_in_rank,
                                                       num_tokens_per_expert_global,
                                                       0,
                                                       0,
                                                       std::nullopt,
                                                       std::nullopt,
                                                       std::nullopt,
                                                       std::nullopt,
                                                       expert_alignment,
                                                       0,
                                                       config,
                                                       prev_event,
                                                       false,
                                                       false);

        result.recv_x                                 = std::get<0>(dispatch_ret);
        result.recv_topk_idx                          = std::get<2>(dispatch_ret);
        result.recv_topk_weights                      = std::get<3>(dispatch_ret);
        result.handle.recv_rdma_channel_prefix_matrix = std::get<7>(dispatch_ret).value();
        result.handle.recv_rdma_rank_prefix_sum       = std::get<8>(dispatch_ret);
        result.handle.recv_gbl_channel_prefix_matrix  = std::get<9>(dispatch_ret).value();
        result.handle.recv_src_meta                   = std::get<11>(dispatch_ret).value();
        result.handle.send_rdma_head                  = std::get<12>(dispatch_ret).value();
        result.handle.send_nvl_head                   = std::get<13>(dispatch_ret).value();
    }
    else {
        auto dispatch_ret = buffer->intranode_dispatch(hidden_flat.contiguous(),
                                                       std::nullopt,
                                                       topk_ids,
                                                       topk_weights,
                                                       num_tokens_per_rank,
                                                       is_token_in_rank,
                                                       num_tokens_per_expert_global,
                                                       0,
                                                       std::nullopt,
                                                       std::nullopt,
                                                       expert_alignment,
                                                       0,
                                                       config,
                                                       prev_event,
                                                       false,
                                                       false);

        result.recv_x                                  = std::get<0>(dispatch_ret);
        result.recv_topk_idx                           = std::get<2>(dispatch_ret);
        result.recv_topk_weights                       = std::get<3>(dispatch_ret);
        result.handle.intra_rank_prefix_matrix         = std::get<5>(dispatch_ret);
        result.handle.intra_recv_channel_prefix_matrix = std::get<7>(dispatch_ret);
        result.handle.intra_recv_src_idx               = std::get<8>(dispatch_ret);
        result.handle.intra_send_head                  = std::get<9>(dispatch_ret);
    }

    return result;
}

torch::Tensor
DeepEpOps::combine_normal(deep_ep::Buffer* buffer, torch::Tensor expert_output, const DispatchHandle& handle)
{
    auto device = expert_output.device();

    torch::Tensor output_for_combine  = expert_output;
    torch::Tensor src_idx_for_combine = handle.use_internode ? handle.recv_src_meta : handle.intra_recv_src_idx;

    if (expert_output.size(0) == 0) {
        output_for_combine  = torch::zeros({1, handle.hidden_size}, expert_output.options());
        src_idx_for_combine = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    }

    auto          prev_event = handle.prev_event;
    torch::Tensor combined_x;

    if (handle.use_internode) {
        auto combine_ret = buffer->internode_combine(output_for_combine,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     src_idx_for_combine,
                                                     handle.is_token_in_rank,
                                                     handle.recv_rdma_channel_prefix_matrix,
                                                     handle.recv_rdma_rank_prefix_sum,
                                                     handle.recv_gbl_channel_prefix_matrix,
                                                     handle.send_rdma_head,
                                                     handle.send_nvl_head,
                                                     handle.config,
                                                     prev_event,
                                                     false,
                                                     false);
        combined_x       = std::get<0>(combine_ret);
    }
    else {
        auto combine_ret = buffer->intranode_combine(output_for_combine,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     src_idx_for_combine,
                                                     handle.intra_rank_prefix_matrix,
                                                     handle.intra_recv_channel_prefix_matrix,
                                                     handle.intra_send_head,
                                                     handle.config,
                                                     prev_event,
                                                     false,
                                                     false);
        combined_x       = std::get<0>(combine_ret);
    }

    return combined_x;
}

LowLatencyDispatchResult DeepEpOps::dispatch_low_latency(deep_ep::Buffer* buffer,
                                                         torch::Tensor    hidden_states,
                                                         torch::Tensor    topk_ids,
                                                         torch::Tensor    topk_weights,
                                                         int              num_max_dispatch_tokens_per_rank,
                                                         int              num_experts,
                                                         int              ep_world_size)
{
    // Ensure hidden_states is BF16
    if (hidden_states.scalar_type() != torch::kBFloat16) {
        hidden_states = hidden_states.to(torch::kBFloat16);
    }

    auto hidden_flat = hidden_states.view({-1, hidden_states.size(-1)});
    int  num_tokens  = hidden_flat.size(0);
    int  hidden_size = hidden_flat.size(1);
    int  top_k       = topk_ids.size(1);

    int world_size = ep_world_size > 0 ? ep_world_size : get_dist_context().ffn_ep();

    auto topk_ids_i64 = topk_ids.to(torch::kLong);

    auto dispatch_ret = buffer->low_latency_dispatch(hidden_flat.contiguous(),
                                                     topk_ids_i64,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     num_max_dispatch_tokens_per_rank,
                                                     num_experts,
                                                     false,
                                                     false,
                                                     false,
                                                     false,
                                                     false);

    auto recv_x       = std::get<0>(dispatch_ret);
    auto masked_m     = std::get<2>(dispatch_ret);
    auto src_info     = std::get<3>(dispatch_ret);
    auto layout_range = std::get<4>(dispatch_ret);

    // Use fixed expected_m based on buffer max capacity for CUDA Graph compatibility
    // The masked_m tensor tells DeepGemm how many tokens are actually valid per expert
    int expected_m = static_cast<int>(recv_x.size(1));  // = num_max_dispatch_tokens_per_rank * world_size

    LowLatencyDispatchResult result;
    result.recv_x                                  = recv_x;
    result.masked_m                                = masked_m;
    result.expected_m                              = expected_m;
    result.handle.topk_idx                         = topk_ids_i64;
    result.handle.topk_weights                     = topk_weights;
    result.handle.src_info                         = src_info;
    result.handle.layout_range                     = layout_range;
    result.handle.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank;
    result.handle.num_experts                      = num_experts;
    result.handle.hidden_size                      = hidden_size;

    return result;
}

torch::Tensor DeepEpOps::combine_low_latency(deep_ep::Buffer*                buffer,
                                             torch::Tensor                   expert_output,
                                             const LowLatencyDispatchHandle& handle)
{
    auto combine_ret = buffer->low_latency_combine(expert_output,
                                                   handle.topk_idx,
                                                   handle.topk_weights,
                                                   handle.src_info,
                                                   handle.layout_range,
                                                   std::nullopt,
                                                   handle.num_max_dispatch_tokens_per_rank,
                                                   handle.num_experts,
                                                   false,
                                                   false,
                                                   false,
                                                   false,
                                                   std::nullopt);

    return std::get<0>(combine_ret);
}

}  // namespace ops
}  // namespace nanodeploy
