#include "nanodeploy/worker/deep_ep_runner.h"
#include "nanodeploy/worker/deep_ep_utils.h"
#include "nanodeploy/worker/distributed.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace nanodeploy {

// We need a global interpreter if we are to use pybind11 types in C++.
// Since this is running in a dedicated Actor process, this is acceptable for now.
// In a production C++ environment without Python, DeepEP would need refactoring.
static std::unique_ptr<pybind11::scoped_interpreter> g_interpreter;

DeepEPRunner::~DeepEPRunner()
{
    if (buffer_) {
        buffer_->destroy();
    }
}

DeepEPInitResp DeepEPRunner::init(const DeepEPInitReq& req)
{
    try {
        if (!g_interpreter) {
            // Check if interpreter is already running (maybe by another runner or embedding)
            if (!Py_IsInitialized()) {
                g_interpreter = std::make_unique<pybind11::scoped_interpreter>();
            }
        }

        rank_       = req.rank;
        world_size_ = req.world_size;

        // Clear invalid CUDA context inherited from parent process after fork()
        cudaDeviceReset();

        // Determine device
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        if (num_devices > 0) {
            device_id_ = rank_ % num_devices;
            cudaSetDevice(0);
        }

        // Setup NVSHMEM environment variables BEFORE creating buffer
        // num_qps_per_rank should be >= num_local_experts
        // Use single_card mode when world_size == 1
        setup_nvshmem_env(32, world_size_ == 1);

        buffer_ = std::make_unique<deep_ep::Buffer>(req.rank,
                                                    req.world_size,
                                                    req.num_nvl_bytes,
                                                    req.num_rdma_bytes,
                                                    req.low_latency_mode,
                                                    true,  // explicitly_destroy
                                                    true,  // enable_shrink (required for clean_low_latency_buffer)
                                                    false  // use_fabric
        );

        return {true, "Initialized successfully"};
    }
    catch (const std::exception& e) {
        return {false, std::string("Init failed: ") + e.what()};
    }
}

DeepEPInfoResp DeepEPRunner::get_info()
{
    DeepEPInfoResp resp;
    try {
        // IPC Handle
        auto        ipc_handle_py = buffer_->get_local_ipc_handle();
        std::string ipc_str       = static_cast<std::string>(ipc_handle_py);
        resp.ipc_handle.assign(ipc_str.begin(), ipc_str.end());

        // NVSHMEM ID (Rank 0 only usually, but we get it anyway)
        // DeepEP asserts rdma_rank == 0 for get_local_nvshmem_unique_id
        if (buffer_->get_rdma_rank() == 0) {
            auto        nv_id_py = buffer_->get_local_nvshmem_unique_id();
            std::string nv_str   = static_cast<std::string>(nv_id_py);
            resp.nvshmem_unique_id.assign(nv_str.begin(), nv_str.end());
        }
    }
    catch (const std::exception& e) {
        std::cerr << "get_info failed: " << e.what() << std::endl;
    }
    return resp;
}

DeepEPSyncResp DeepEPRunner::sync(const DeepEPSyncReq& req)
{
    try {
        std::cout << "[DeepEP] Rank " << rank_ << ": sync called. handle_size=" << req.handle_size
                  << ", valid_mask_size=" << req.valid_mask.size() << std::endl;

        std::vector<int> device_ids;
        for (int i = 0; i < world_size_; ++i) {
            device_ids.push_back(i % 8);  // Assume max 8 gpus per node
        }

        std::vector<std::optional<pybind11::bytearray>> all_handles;
        const uint8_t* ptr    = reinterpret_cast<const uint8_t*>(req.all_handles_flat.data());
        size_t         offset = 0;

        for (int i = 0; i < world_size_; ++i) {
            bool valid = (i < (int)req.valid_mask.size()) && req.valid_mask[i];
            if (valid) {
                if (offset + req.handle_size > req.all_handles_flat.size()) {
                    throw std::runtime_error("Invalid handle buffer size");
                }
                std::string s(reinterpret_cast<const char*>(ptr + offset), req.handle_size);
                all_handles.emplace_back(pybind11::bytearray(s));
                offset += req.handle_size;
            }
            else {
                all_handles.push_back(std::nullopt);
            }
        }

        std::optional<pybind11::bytearray> root_id_opt = std::nullopt;
        if (!req.nvshmem_unique_id.empty()) {
            std::string s(req.nvshmem_unique_id.begin(), req.nvshmem_unique_id.end());
            root_id_opt = pybind11::bytearray(s);
        }

        std::cout << "[DeepEP] Rank " << rank_ << ": invoking buffer_->sync..." << std::endl;
        buffer_->sync(device_ids, all_handles, root_id_opt);
        std::cout << "[DeepEP] Rank " << rank_ << ": buffer_->sync returned." << std::endl;

        return {true, "Sync successful"};
    }
    catch (const std::exception& e) {
        std::cerr << "[DeepEP] Rank " << rank_ << ": Sync exception: " << e.what() << std::endl;
        return {false, std::string("Sync failed: ") + e.what()};
    }
}

DeepEPTestResp DeepEPRunner::run_test(const DeepEPTestReq& req)
{
    try {
        cudaSetDevice(device_id_);
        torch::manual_seed(req.seed + rank_);

        // Prepare Data
        int num_tokens  = req.num_tokens;
        int hidden      = req.hidden;
        int num_experts = req.num_experts;
        int num_topk    = req.num_topk;

        auto options_bf16  = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
        auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto options_int   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

        // x: [num_tokens, hidden]
        auto x = torch::randn({num_tokens, hidden}, options_bf16);

        // topk_idx: [num_tokens, num_topk]
        auto scores       = torch::randn({num_tokens, num_experts}, options_float).abs() + 1;
        auto topk_ret     = torch::topk(scores, num_topk, -1, true, true);
        auto topk_idx     = std::get<1>(topk_ret).to(torch::kInt64);
        auto topk_weights = torch::randn({num_tokens, num_topk}, options_float).abs();

        // Clean low-latency buffer before use (required for low-latency mode)
        // num_max_dispatch_tokens_per_rank = num_tokens in this test
        buffer_->clean_low_latency_buffer(num_tokens, hidden, num_experts);
        cudaDeviceSynchronize();

        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto dispatch_ret = buffer_->low_latency_dispatch(x,
                                                              topk_idx,
                                                              std::nullopt,
                                                              std::nullopt,  // stats
                                                              num_tokens,
                                                              num_experts,
                                                              req.use_fp8,  // use_fp8
                                                              false,        // round_scale
                                                              false,        // use_ue8m0
                                                              false,        // async
                                                              false         // return_recv_hook
            );
        }

        // Measure Dispatch
        auto start     = std::chrono::high_resolution_clock::now();
        int  num_iters = 20;

        // We need to keep the result to pass to combine
        std::tuple<torch::Tensor,
                   std::optional<torch::Tensor>,
                   torch::Tensor,
                   torch::Tensor,
                   torch::Tensor,
                   std::optional<deep_ep::EventHandle>,
                   std::optional<std::function<void()>>>
            dispatch_result;

        for (int i = 0; i < num_iters; ++i) {
            dispatch_result = buffer_->low_latency_dispatch(x,
                                                            topk_idx,
                                                            std::nullopt,
                                                            std::nullopt,
                                                            num_tokens,
                                                            num_experts,
                                                            req.use_fp8,
                                                            false,
                                                            false,
                                                            false,
                                                            false);
        }
        torch::cuda::synchronize();
        auto   end          = std::chrono::high_resolution_clock::now();
        double dispatch_lat = std::chrono::duration<double, std::micro>(end - start).count() / num_iters;

        // Extract outputs for combine
        auto recv_x = std::get<0>(dispatch_result);

        // If FP8, unpack
        // DeepEP logic: if use_fp8, recv_x is tuple of (data, scale) or just tensor?
        // Let's check deep_ep.cpp.
        // It returns tuple<Tensor, optional<Tensor>...>.
        // If FP8, first tensor is INT8?

        // For combine, we need src_info and layout_range
        auto src_info     = std::get<3>(dispatch_result);
        auto layout_range = std::get<4>(dispatch_result);

        // Simulate Computation (Identity for now)
        // In real MoE, we would do GEMM here.
        // We need 'hidden_x' matching the recv_x layout/type.
        // If FP8, recv_x might be different.
        // For simplicity in this test, we just clone recv_x as input to combine.
        // DeepEP combine expects `x` which is the result of computation.

        auto hidden_x = recv_x.clone();

        // Measure Combine
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iters; ++i) {
            buffer_->low_latency_combine(hidden_x,
                                         topk_idx,
                                         topk_weights,
                                         src_info,
                                         layout_range,
                                         std::nullopt,  // cost stats
                                         num_tokens,
                                         num_experts,
                                         false,        // use_logfmt
                                         false,        // zero_copy
                                         false,        // async
                                         false,        // return_recv_hook
                                         std::nullopt  // out
            );
        }
        torch::cuda::synchronize();
        end                = std::chrono::high_resolution_clock::now();
        double combine_lat = std::chrono::duration<double, std::micro>(end - start).count() / num_iters;

        return {true, dispatch_lat, combine_lat, "Success"};
    }
    catch (const std::exception& e) {
        return {false, 0.0, 0.0, std::string("Test failed: ") + e.what()};
    }
}

DispatchResult DeepEPRunner::dispatch_normal(deep_ep::Buffer* buffer,
                                             torch::Tensor    hidden_states,
                                             torch::Tensor    topk_ids,
                                             torch::Tensor    topk_weights,
                                             int              num_experts,
                                             int              expert_alignment)
{
    auto device      = hidden_states.device();
    auto hidden_flat = hidden_states.view({-1, hidden_states.size(-1)});

    // Get dispatch layout
    std::optional<deep_ep::EventHandle> prev_event = std::nullopt;
    auto layout                       = buffer->get_dispatch_layout(topk_ids, num_experts, prev_event, false, false);
    auto num_tokens_per_rank          = std::get<0>(layout);
    auto num_tokens_per_rdma_rank     = std::get<1>(layout);
    auto num_tokens_per_expert_global = std::get<2>(layout);
    auto is_token_in_rank             = std::get<3>(layout);

    // Get config
    int num_sms    = 0;
    int device_idx = device.is_cuda() ? device.index() : 0;
    if (device_idx < 0)
        device_idx = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_idx);
    if (num_sms <= 0)
        num_sms = 132;  // H100/H200 default

    // Dynamically adjust config based on hidden_size to avoid buffer overflow
    int hidden_size     = hidden_flat.size(1);
    int nvl_send_tokens = 4096;
    int nvl_recv_tokens = 8192;
    if (hidden_size > 6000) {
        nvl_send_tokens = 1024;
        nvl_recv_tokens = 2048;
    }
    else if (hidden_size > 4000) {
        nvl_send_tokens = 2048;
        nvl_recv_tokens = 4096;
    }
    deep_ep::Config config(num_sms, nvl_send_tokens, nvl_recv_tokens, nvl_send_tokens, nvl_recv_tokens);

    // Determine dispatch path
    int  num_rdma_ranks = buffer->get_num_rdma_ranks();
    bool use_internode  = (num_rdma_ranks > 1);

    DispatchResult result;
    result.handle.use_internode    = use_internode;
    result.handle.hidden_size      = hidden_flat.size(1);
    result.handle.config           = config;
    result.handle.prev_event       = prev_event;
    result.handle.is_token_in_rank = is_token_in_rank;

    if (use_internode) {
        // Internode dispatch
        auto dispatch_ret = buffer->internode_dispatch(hidden_flat.contiguous(),
                                                       std::nullopt,  // x_scales
                                                       topk_ids,
                                                       topk_weights,  // Must be float32
                                                       num_tokens_per_rank,
                                                       num_tokens_per_rdma_rank,
                                                       is_token_in_rank,
                                                       num_tokens_per_expert_global,
                                                       0,  // cached_num_recv_tokens
                                                       0,  // cached_num_rdma_recv_tokens
                                                       std::nullopt,
                                                       std::nullopt,
                                                       std::nullopt,
                                                       std::nullopt,  // cached matrices
                                                       expert_alignment,
                                                       0,  // num_worst_tokens
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
        // Intranode dispatch
        auto dispatch_ret = buffer->intranode_dispatch(hidden_flat.contiguous(),
                                                       std::nullopt,  // x_scales
                                                       topk_ids,
                                                       topk_weights,  // Must be float32
                                                       num_tokens_per_rank,
                                                       is_token_in_rank,
                                                       num_tokens_per_expert_global,
                                                       0,  // cached_num_recv_tokens
                                                       std::nullopt,
                                                       std::nullopt,  // cached matrices
                                                       expert_alignment,
                                                       0,  // num_worst_tokens
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
DeepEPRunner::combine_normal(deep_ep::Buffer* buffer, torch::Tensor expert_output, const DispatchHandle& handle)
{
    auto device = expert_output.device();

    // Handle empty output
    torch::Tensor output_for_combine  = expert_output;
    torch::Tensor src_idx_for_combine = handle.use_internode ? handle.recv_src_meta : handle.intra_recv_src_idx;

    if (expert_output.size(0) == 0) {
        // Create dummy tensors with at least 1 element
        output_for_combine  = torch::zeros({1, handle.hidden_size}, expert_output.options());
        src_idx_for_combine = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    }

    // Create mutable copy of prev_event (DeepEP API requires non-const reference)
    auto prev_event = handle.prev_event;

    torch::Tensor combined_x;

    if (handle.use_internode) {
        auto combine_ret = buffer->internode_combine(output_for_combine,
                                                     std::nullopt,  // weights already applied
                                                     std::nullopt,
                                                     std::nullopt,  // bias
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
                                                     std::nullopt,  // weights already applied
                                                     std::nullopt,
                                                     std::nullopt,  // bias
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

LowLatencyDispatchResult DeepEPRunner::dispatch_low_latency(deep_ep::Buffer* buffer,
                                                            torch::Tensor    hidden_states,
                                                            torch::Tensor    topk_ids,
                                                            torch::Tensor    topk_weights,
                                                            int              num_max_dispatch_tokens_per_rank,
                                                            int              num_experts)
{
    auto hidden_flat = hidden_states.view({-1, hidden_states.size(-1)});
    int  num_tokens  = hidden_flat.size(0);
    int  hidden_size = hidden_flat.size(1);
    int  top_k       = topk_ids.size(1);

    // Get world_size from DistContext
    auto& dist_ctx   = get_dist_context();
    int   world_size = dist_ctx.ffn_ep_world_size();

    // Ensure topk_ids is int64 as required by DeepEP
    auto topk_ids_i64 = topk_ids.to(torch::kLong);

    // Call low_latency_dispatch
    auto dispatch_ret = buffer->low_latency_dispatch(hidden_flat.contiguous(),
                                                     topk_ids_i64,
                                                     std::nullopt,  // cumulative_local_expert_recv_stats
                                                     std::nullopt,  // dispatch_wait_recv_cost_stats
                                                     num_max_dispatch_tokens_per_rank,
                                                     num_experts,
                                                     false,  // use_fp8
                                                     false,  // round_scale
                                                     false,  // use_ue8m0
                                                     false,  // async
                                                     false   // return_recv_hook
    );

    auto recv_x       = std::get<0>(dispatch_ret);  // [num_local_experts, max_m, hidden_size]
    auto masked_m     = std::get<2>(dispatch_ret);  // [num_local_experts] - actual counts
    auto src_info     = std::get<3>(dispatch_ret);  // for combine
    auto layout_range = std::get<4>(dispatch_ret);  // for combine

    // Calculate expected_m (average tokens per expert)
    int expected_m = (num_tokens * world_size * top_k + num_experts - 1) / num_experts;
    expected_m     = std::min(expected_m, static_cast<int>(recv_x.size(1)));  // Cap at actual max

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

torch::Tensor DeepEPRunner::combine_low_latency(deep_ep::Buffer*                buffer,
                                                torch::Tensor                   expert_output,
                                                const LowLatencyDispatchHandle& handle)
{
    // Call low_latency_combine
    // expert_output: [num_local_experts, max_m, hidden_size]
    auto combine_ret = buffer->low_latency_combine(expert_output,
                                                   handle.topk_idx,
                                                   handle.topk_weights,
                                                   handle.src_info,
                                                   handle.layout_range,
                                                   std::nullopt,  // combine_wait_recv_cost_stats
                                                   handle.num_max_dispatch_tokens_per_rank,
                                                   handle.num_experts,
                                                   false,        // use_logfmt
                                                   false,        // zero_copy
                                                   false,        // async
                                                   false,        // return_recv_hook
                                                   std::nullopt  // out
    );

    return std::get<0>(combine_ret);
}

}  // namespace nanodeploy
