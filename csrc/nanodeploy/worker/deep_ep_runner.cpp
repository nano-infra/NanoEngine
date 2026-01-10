#include "nanodeploy/worker/deep_ep_runner.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace nanodeploy {

// Set NVSHMEM environment variables required for DeepEP low-latency mode
// Must be called BEFORE creating the Buffer (which initializes NVSHMEM)
static void setup_nvshmem_env(int num_qps_per_rank = 32)
{
    // Enable IBGDA (InfiniBand GPUDirect Async)
    setenv("NVSHMEM_IB_ENABLE_IBGDA", "1", 0);

    // Number of QPs per rank - should be >= number of local experts
    setenv("NVSHMEM_IBGDA_NUM_RC_PER_PE", std::to_string(num_qps_per_rank).c_str(), 0);

    // Allow P2P (NVLink) traffic
    setenv("NVSHMEM_DISABLE_P2P", "0", 0);

    // QP depth - must be larger than on-flight WRs
    setenv("NVSHMEM_QP_DEPTH", "1024", 0);

    // Reduce GPU memory usage
    setenv("NVSHMEM_MAX_TEAMS", "7", 0);

    // Disable NVLink SHArP
    setenv("NVSHMEM_DISABLE_NVLS", "1", 0);

    // NVSHMEM initialization requires at least 256 MiB granularity
    setenv("NVSHMEM_CUMEM_GRANULARITY", "536870912", 0);  // 2^29 = 512 MiB

    // Disable multi-node NVLink detection (for single node testing)
    setenv("NVSHMEM_DISABLE_MNNVL", "1", 0);
}

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
        setup_nvshmem_env(32);

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

}  // namespace nanodeploy
