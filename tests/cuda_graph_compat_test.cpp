#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <torch/torch.h>

// Helper macro for CUDA errors
#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            FAIL() << "CUDA Error: " << cudaGetErrorString(err);                                                       \
        }                                                                                                              \
    } while (0)

class CudaGraphCompatTest: public ::testing::Test {
protected:
    void SetUp() override
    {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        // Create a dedicated stream for this test to avoid global stream pollution
        CUDA_CHECK(cudaStreamCreate(&native_stream));

        // Wrap as c10::cuda::CUDAStream
        // Note: getStreamFromExternal requires device index
        auto device_index = c10::cuda::current_device();
        c10_stream        = c10::cuda::getStreamFromExternal(native_stream, device_index);

        // Save old stream and switch to new one
        old_stream = c10::cuda::getCurrentCUDAStream();
        c10::cuda::setCurrentCUDAStream(c10_stream);

        stream = native_stream;  // For raw CUDA calls
    }

    void TearDown() override
    {
        // Restore original stream
        if (old_stream.stream() != nullptr) {
            c10::cuda::setCurrentCUDAStream(old_stream);
        }

        if (graph_exec) {
            cudaGraphExecDestroy(graph_exec);
            graph_exec = nullptr;
        }
        if (graph) {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }

        // Explicitly check status on OUR stream
        cudaStreamCaptureStatus status;
        cudaStreamIsCapturing(stream, &status);
        if (status == cudaStreamCaptureStatusActive) {
            cudaStreamEndCapture(stream, &graph);
            if (graph) {
                cudaGraphDestroy(graph);
                graph = nullptr;
            }
        }

        // Destroy our stream
        cudaStreamDestroy(native_stream);
    }

    cudaStream_t          native_stream;
    cudaStream_t          stream;  // alias for native_stream
    c10::cuda::CUDAStream c10_stream = c10::cuda::getDefaultCUDAStream();
    c10::cuda::CUDAStream old_stream = c10::cuda::getDefaultCUDAStream();

    cudaGraph_t     graph      = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
};

// Basic Matrix Multiplication Test
TEST_F(CudaGraphCompatTest, MatMul)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto a       = torch::randn({256, 512}, options);
    auto b       = torch::randn({512, 256}, options);

    // Warmup
    {
        auto d = torch::matmul(a, b);
    }
    cudaDeviceSynchronize();

    // Capture
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    auto d = torch::matmul(a, b);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    cudaDeviceSynchronize();
}

// Basic Addition Test
TEST_F(CudaGraphCompatTest, Add)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto a       = torch::randn({256, 256}, options);
    auto b       = torch::randn({256, 256}, options);

    // Warmup
    {
        auto c = a + b;
    }
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    auto c = a + b;
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    cudaDeviceSynchronize();
}

TEST_F(CudaGraphCompatTest, CPUEmbeddingWeight)
{
    // 1. Allocate input on GPU
    auto options_long = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);
    auto input_ids    = torch::randint(0, 1000, {32}, options_long);

    // 2. Allocate weight on CPU (Default)
    auto embedding_weight = torch::randn({1000, 512});  // CPU tensor

    c10::cuda::setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream());
    cudaDeviceSynchronize();

    // 3. Capture
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    try {
        // Embedding with CPU weight + GPU input
        auto output = torch::embedding(embedding_weight, input_ids);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        SUCCEED();
    }
    catch (const std::exception& e) {
        std::cout << "Caught expected error: " << e.what() << std::endl;
        cudaStreamEndCapture(stream, &graph);
        // This is what we expect to fail if our hypothesis is correct
        if (std::string(e.what()).find("StreamCaptureUnsupported") != std::string::npos
            || std::string(e.what()).find("operation not permitted") != std::string::npos) {
            // Verification successful - we reproduced the failure
            return;
        }
        FAIL() << "Capture failed with unexpected error: " << e.what();
    }
}

TEST_F(CudaGraphCompatTest, IndexSelect)
{
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto options_long  = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);

    auto weight  = torch::randn({10000, 512}, options_float);
    auto indices = torch::randint(0, 10000, {32}, options_long);

    // Warmup
    {
        auto output = torch::index_select(weight, 0, indices);
    }
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    try {
        auto output = torch::index_select(weight, 0, indices);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    }
    catch (const std::exception& e) {
        FAIL() << "index_select capture failed: " << e.what();
    }
}

TEST_F(CudaGraphCompatTest, Gather)
{
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto options_long  = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);

    auto weight  = torch::randn({32, 512}, options_float);
    auto indices = torch::randint(0, 512, {32, 1}, options_long).expand({32, 512});

    // Warmup
    {
        auto output = torch::gather(weight, 1, indices);
    }
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    try {
        auto output = torch::gather(weight, 1, indices);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    }
    catch (const std::exception& e) {
        FAIL() << "gather capture failed: " << e.what();
    }
}

TEST_F(CudaGraphCompatTest, CopyInPlace)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto a       = torch::randn({256, 512}, options);
    auto b       = torch::empty({256, 512}, options);

    // Warmup
    b.copy_(a);
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    b.copy_(a);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
}

TEST_F(CudaGraphCompatTest, PreAllocatedEmbedding)
{
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto options_long  = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);

    int batch_size  = 32;
    int vocab_size  = 10000;
    int hidden_size = 512;

    auto embedding_weight = torch::randn({vocab_size, hidden_size}, options_float);
    auto input_ids        = torch::randint(0, vocab_size, {batch_size}, options_long);
    auto output           = torch::empty({batch_size, hidden_size}, options_float);

    // Warmup
    torch::index_select_out(output, embedding_weight, 0, input_ids);
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    torch::index_select_out(output, embedding_weight, 0, input_ids);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
}

TEST_F(CudaGraphCompatTest, D2DCopy_Copy_)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto src     = torch::randn({256, 512}, options);
    auto dst     = torch::empty({256, 512}, options);

    // Warmup
    dst.copy_(src);
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    dst.copy_(src);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    cudaDeviceSynchronize();
}

TEST_F(CudaGraphCompatTest, D2DCopy_To)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto src     = torch::randn({256, 512}, options);

    // Note: .to() allocates memory IF it returns a new tensor.
    // So `auto dst = src.to(device)` will allocate, thus failing capture.
    // BUT `src.to(src.options(), /*non_blocking=*/false, /*copy=*/false)` is specific.
    // The user likely wants to know if `.to()` calls inside capture work.
    // They typically DON'T work because of allocation.
    // We expect this to FAIL if we don't hold the output.
    // But let's test a case where it likely does NOT copy (same device).

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    auto dst = src.to(torch::kCUDA);  // Should be no-op or shallow copy?
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
}

TEST_F(CudaGraphCompatTest, PreAllocatedRMSNorm)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto  x      = torch::randn({32, 512}, options);
    auto  weight = torch::ones({512}, options);
    float eps    = 1e-6f;

    // Warmup - explicitly allocate outputs for warmup
    {
        auto variance = x.pow(2).mean(-1, true);
        auto x_normed = x * torch::rsqrt(variance + eps);
        auto output   = x_normed * weight;
    }
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    // Note: Intermediate allocations (variance, x_normed) happen here.
    // Ideally these should be fused or pre-allocated, but let's test if PyTorch allocator handles them.
    auto variance = x.pow(2).mean(-1, true);
    auto x_normed = x * torch::rsqrt(variance + eps);
    auto output   = x_normed * weight;
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
