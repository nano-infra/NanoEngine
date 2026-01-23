#include "nanodeploy/csrc/context/deep_ep_context.h"
#include "nanodeploy/csrc/logging.h"
#include "nanodeploy/csrc/ops/deep_ep_utils.h"

#include <algorithm>
#include <cuda_runtime.h>

namespace nanodeploy {

DeepEpContext::~DeepEpContext() = default;

void DeepEpContext::init(const core::ModelConfig& config, int ffn_ep_world_size, int ffn_ep_rank)
{
#ifdef DEEPSEEK_MOE
    int ep_size = ffn_ep_world_size;

    if (ep_size > 1) {
        // Multi-GPU: Initialize DeepEP for expert parallel communication
        int num_experts = config.num_experts;
        int hidden_size = config.hidden_size;

        // Setup NVSHMEM environment variables BEFORE creating buffer
        int num_local_experts = num_experts / ep_size;
        int num_qps_per_rank  = std::max(32, num_local_experts);
        setup_nvshmem_env(num_qps_per_rank, false);  // multi-card mode

        // Calculate buffer sizes for COMMON buffer (supports both Normal and Low Latency)
        int num_sms = 0;
        // Assuming current device is set correctly before calling init
        int current_device;
        cudaGetDevice(&current_device);
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, current_device);
        num_sms = 20;

        int64_t hidden_size_bytes;
        if (config.quant_method == "fp8") {
            hidden_size_bytes = hidden_size;  // FP8_E4M3FN
        }
        else {
            hidden_size_bytes = hidden_size * 2;  // BF16
        }

        // Use recommended configs from DeepEP
        int nvl_chunk_send  = 6;
        int nvl_chunk_recv  = 256;
        int rdma_chunk_send = 6;
        int rdma_chunk_recv = 128;

        dep_config_ = std::make_unique<deep_ep::Config>(
            num_sms, nvl_chunk_send, nvl_chunk_recv, rdma_chunk_send, rdma_chunk_recv);

        // NVL buffer for Normal mode
        // NOTE: FP8 mode requires extra space for scale buffers, multiply by 2x as safety margin
        int64_t num_nvl_bytes = dep_config_->get_nvl_buffer_size_hint(hidden_size_bytes, ep_size);

        // RDMA buffer: max of Normal and Low Latency requirements
        int     num_max_dispatch_tokens_per_rank = 256;  // Typical decode batch size
        int64_t normal_rdma_bytes                = dep_config_->get_rdma_buffer_size_hint(hidden_size_bytes, ep_size);
        // For low latency mode, we use hidden_size_bytes to ensure sufficient buffer
        // NOTE: There may be a mismatch with LowLatencyLayout which uses dimension,
        // but using bytes here provides larger allocation which is safer
        int64_t ll_rdma_bytes = deep_ep::get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank, hidden_size_bytes, ep_size, num_experts);

        int64_t num_rdma_bytes = std::max(normal_rdma_bytes, ll_rdma_bytes);

        NANODEPLOY_LOG_INFO("DeepEP Buffer params: ep_size=", ep_size, " hidden=", hidden_size, " num_sms=", num_sms);

        buffer_ = std::make_unique<deep_ep::Buffer>(ffn_ep_rank,
                                                    ep_size,
                                                    num_nvl_bytes,
                                                    num_rdma_bytes,
                                                    true,  // low_latency_mode = true
                                                    true,  // explicitly_destroy
                                                    true,  // enable_shrink
                                                    false  // use_fabric
        );

        NANODEPLOY_LOG_INFO("DeepEP Buffer initialized for DeepSeek MoE (ep_size=", ep_size, ")");
    }
    else {
        NANODEPLOY_LOG_INFO("DeepSeek MoE running in single-card mode (no DeepEP)");
    }
#else
    // No-op if DEEPSEEK_MOE not defined
#endif
}

DeepEpInfoResp DeepEpContext::get_info()
{
    DeepEpInfoResp resp{};
#ifdef DEEPSEEK_MOE
    if (buffer_) {
        resp.device_id      = buffer_->get_local_device_id();
        resp.num_rdma_ranks = buffer_->get_num_rdma_ranks();
        resp.rdma_rank      = buffer_->get_rdma_rank();
        resp.root_rdma_rank = buffer_->get_root_rdma_rank(true);

        // Get IPC handle
        auto ipc_handle_py = buffer_->get_local_ipc_handle();
        resp.ipc_handle    = ipc_handle_py.cast<std::string>();

        // Get NVSHMEM unique ID
        if (resp.rdma_rank == resp.root_rdma_rank) {
            try {
                auto nvshmem_id_py     = buffer_->get_local_nvshmem_unique_id();
                resp.nvshmem_unique_id = nvshmem_id_py.cast<std::string>();
            }
            catch (const std::exception& e) {
                NANODEPLOY_LOG_WARN("Could not get NVSHMEM unique ID: ", e.what());
            }
        }
    }
#endif
    return resp;
}

bool DeepEpContext::sync(const DeepEpSyncReq& req)
{
#ifdef DEEPSEEK_MOE
    if (!buffer_) {
        // Safe to return false or true? If no buffer, maybe no sync needed.
        return false;
    }

    // Convert to pybind11 types
    std::vector<std::optional<pybind11::bytearray>> all_handles;
    all_handles.reserve(req.ipc_handles.size());
    for (const auto& h : req.ipc_handles) {
        all_handles.push_back(pybind11::bytearray(h.data(), h.size()));
    }

    std::optional<pybind11::bytearray> root_unique_id;
    if (!req.root_nvshmem_unique_id.empty()) {
        root_unique_id = pybind11::bytearray(req.root_nvshmem_unique_id);
    }

    try {
        buffer_->sync(req.device_ids, all_handles, root_unique_id);
        return true;
    }
    catch (const std::exception& e) {
        NANODEPLOY_LOG_ERROR("DeepEP sync failed: ", e.what());
        return false;
    }
#else
    return false;
#endif
}

}  // namespace nanodeploy
