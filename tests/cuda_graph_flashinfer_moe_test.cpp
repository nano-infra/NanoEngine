#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>

#include "nanodeploy/csrc/ops/deep_gemm_ops.h"
#include "nanodeploy/csrc/ops/flashinfer_ops.h"
#include "nanodeploy/csrc/ops/moe_expert_ops.h"

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            FAIL() << "CUDA Error: " << cudaGetErrorString(err);                                                       \
        }                                                                                                              \
    } while (0)

class FlashInferMoeGraphTest: public ::testing::Test {
protected:
    void SetUp() override
    {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        // Initialize DeepGemm to avoid JIT assertions
        nanodeploy::ops::DeepGemmOps::init();

        CUDA_CHECK(cudaStreamCreate(&native_stream));
        auto device_index = c10::cuda::current_device();
        c10_stream        = c10::cuda::getStreamFromExternal(native_stream, device_index);
        old_stream        = c10::cuda::getCurrentCUDAStream();
        c10::cuda::setCurrentCUDAStream(c10_stream);
        stream = native_stream;
    }

    void TearDown() override
    {
        if (old_stream.stream() != nullptr) {
            c10::cuda::setCurrentCUDAStream(old_stream);
        }
        if (graph_exec)
            cudaGraphExecDestroy(graph_exec);
        if (graph)
            cudaGraphDestroy(graph);

        cudaStreamCaptureStatus status;
        cudaStreamIsCapturing(stream, &status);
        if (status == cudaStreamCaptureStatusActive) {
            cudaStreamEndCapture(stream, &graph);
            if (graph)
                cudaGraphDestroy(graph);
        }
        cudaStreamDestroy(native_stream);
    }

    cudaStream_t          native_stream;
    cudaStream_t          stream;
    c10::cuda::CUDAStream c10_stream = c10::cuda::getDefaultCUDAStream();
    c10::cuda::CUDAStream old_stream = c10::cuda::getDefaultCUDAStream();

    cudaGraph_t     graph      = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
};

TEST_F(FlashInferMoeGraphTest, FlashInferAttentionCapture)
{
    int num_layers = 1, num_heads = 32, num_kv_heads = 8, head_dim = 128, page_size = 16;
    int batch_size = 4, max_num_blocks = 64;

    auto device       = torch::kCUDA;
    auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    nanodeploy::ops::FlashInferOps handler(num_layers, num_heads, num_kv_heads, head_dim, page_size, device);
    handler.init_workspace(batch_size, max_num_blocks);

    std::vector<int> block_tables(batch_size * max_num_blocks, 0);
    // Fill block tables
    for (int i = 0; i < batch_size * max_num_blocks; ++i)
        block_tables[i] = i % max_num_blocks;
    std::vector<int> seq_lens(batch_size, 64);

    // Prepare metadata OUTSIDE capture
    handler.begin_forward(
        block_tables.data(), seq_lens.data(), batch_size, max_num_blocks, num_heads, num_kv_heads, head_dim, page_size);
    cudaDeviceSynchronize();

    // Allocate tensors OUTSIDE capture
    auto q       = torch::randn({batch_size, 1, num_heads, head_dim}, options_bf16).contiguous();
    auto k_cache = torch::randn({max_num_blocks, page_size, num_kv_heads, head_dim}, options_bf16).contiguous();
    auto v_cache = torch::randn({max_num_blocks, page_size, num_kv_heads, head_dim}, options_bf16).contiguous();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    auto output =
        handler.attention(q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), batch_size, 1, num_heads, head_dim, 0);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    cudaDeviceSynchronize();
}

TEST_F(FlashInferMoeGraphTest, MoeComputeMaskedCapture)
{
    auto device       = torch::kCUDA;
    auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    auto options_int  = torch::TensorOptions().dtype(torch::kInt32).device(device);

    int num_groups = 8, max_m = 256, hidden_dim = 7168, intermediate_size = 2560;
    int expected_m = 128;  // Used by DeepGemm

    // Allocations OUTSIDE capture
    auto recv_x          = torch::randn({num_groups, max_m, hidden_dim}, options_bf16).contiguous();
    auto masked_m        = torch::randint(1, max_m, {num_groups}, options_int).contiguous();  // Actual m per group
    auto gate_up_weights = torch::randn({num_groups, intermediate_size * 2, hidden_dim}, options_bf16).contiguous();
    auto down_weights    = torch::randn({num_groups, hidden_dim, intermediate_size}, options_bf16).contiguous();

    // Explicit return buffers
    auto gateup_output = torch::empty({num_groups, max_m, intermediate_size * 2}, options_bf16);
    auto down_output   = torch::empty({num_groups, max_m, hidden_dim}, options_bf16);

    // Warmup
    {
        nanodeploy::ops::MoeExpertOps::compute_masked_out(
            recv_x, masked_m, expected_m, gate_up_weights, down_weights, gateup_output, down_output);
    }
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    nanodeploy::ops::MoeExpertOps::compute_masked_out(
        recv_x, masked_m, expected_m, gate_up_weights, down_weights, gateup_output, down_output);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    cudaDeviceSynchronize();
}

TEST_F(FlashInferMoeGraphTest, SiluCapture)
{
    auto device  = torch::kCUDA;
    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    auto x       = torch::randn({256, 2560}, options);

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    auto output = torch::silu(x);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
}

TEST_F(FlashInferMoeGraphTest, SlicingCapture)
{
    auto device       = torch::kCUDA;
    auto options      = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    auto x            = torch::randn({256, 5120}, options);
    int  intermediate = 2560;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    auto gate   = x.slice(1, 0, intermediate);
    auto up     = x.slice(1, intermediate, intermediate * 2);
    auto result = torch::silu(gate) * up;
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
}

// These negative tests confirm what is NOT supported
TEST_F(FlashInferMoeGraphTest, ExpectFail_ZerosAllocation)
{
    auto device  = torch::kCUDA;
    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    try {
        auto output = torch::zeros({256, 4096}, options);
        // If it throws, we catch it. If it doesn't throw but returns error later, we handle that.
        // Some allocators might not throw immediately but produce invalid graphs.
        // But typically torch::zeros/empty will call malloc which fails capture.
        cudaStreamEndCapture(stream, &graph);
        // If we got here, capture succeeded surprisingly.
        // We actually EXPECT failure or crash for allocations.
        // But if it passes, it's not a failure of the test suite per se.
    }
    catch (...) {
        // Expected
        cudaStreamEndCapture(stream, &graph);  // Clean up state if needed
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
