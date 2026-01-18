#include <cassert>
#include <cmath>
#include <iostream>
#include <torch/torch.h>
#include <vector>

#include "nanodeploy/csrc/ops/flashinfer_ops.h"

using namespace nanodeploy;

// Simple SDPA Reference Implementation (CPU/Float32 for precision)
// Inputs:
//   q: [1, NumHeads, HeadDim] (Single query token)
//   k: [SeqLen, NumKVHeads, HeadDim] (Contiguous context)
//   v: [SeqLen, NumKVHeads, HeadDim] (Contiguous context)
// Returns: [1, NumHeads, HeadDim]
torch::Tensor manual_attention_ref(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    // Convert to Float32 for reference calculation
    q = q.to(torch::kFloat32);
    k = k.to(torch::kFloat32);
    v = v.to(torch::kFloat32);

    int64_t B    = 1;
    int64_t Sq   = 1;
    int64_t Sk   = k.size(0);
    int64_t H    = q.size(1);
    int64_t D    = q.size(2);
    int64_t H_kv = k.size(1);

    // Expand GQA
    if (H != H_kv) {
        int64_t group = H / H_kv;
        // Repeat interleave on dim 1 (Heads)
        k = k.repeat_interleave(group, 1);
        v = v.repeat_interleave(group, 1);
    }

    // [1, H, D] -> [1, H, 1, D] -> permute -> [1, H, 1, D]
    auto q_ = q.view({B, H, Sq, D});
    // [Sk, H, D] -> [1, Sk, H, D] -> permute -> [1, H, Sk, D]
    auto k_ = k.unsqueeze(0).permute({0, 2, 1, 3});
    auto v_ = v.unsqueeze(0).permute({0, 2, 1, 3});

    // Score: [1, H, 1, D] @ [1, H, D, Sk] -> [1, H, 1, Sk]
    auto scores = torch::matmul(q_, k_.transpose(-2, -1));
    scores      = scores / std::sqrt((float)D);
    auto attn   = torch::softmax(scores, -1);

    // Output: [1, H, 1, Sk] @ [1, H, Sk, D] -> [1, H, 1, D]
    auto output = torch::matmul(attn, v_);

    return output.view({1, H, D});
}

void test_single_token_decode_correctness()
{
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA not available, skipping test." << std::endl;
        return;
    }
    torch::Device device(torch::kCUDA);
    torch::manual_seed(42);

    std::cout << "[Test] Starting SingleTokenDecodeCorrectness..." << std::endl;

    // Config
    int batch_size     = 1;
    int num_heads      = 16;
    int num_kv_heads   = 8;  // GQA case
    int head_dim       = 128;
    int page_size      = 16;
    int seq_len        = 17;   // Total tokens (0..16)
    int max_num_blocks = 128;  // Sufficiently large

    // Initialize Handler
    auto handler = std::make_unique<ops::FlashInferOps>(1, num_heads, num_kv_heads, head_dim, page_size, device);

    // Create Data (BFloat16)
    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    // Q: [1, H, D]
    auto q = torch::randn({1, num_heads, head_dim}, opts);

    // KV Cache: [NumBlocks, NumKVHeads, PageSize, HeadDim]
    // We need 2 blocks for 17 tokens.
    // Block 0: ID 0
    // Block 1: ID 1
    int  total_blocks = 16;
    auto k_cache      = torch::randn({total_blocks, num_kv_heads, page_size, head_dim}, opts);
    auto v_cache      = torch::randn({total_blocks, num_kv_heads, page_size, head_dim}, opts);

    // Prepare inputs for Handler
    // Block Table: [1, MaxBlocks] -> [0, 1, 0, 0...]
    std::vector<int> block_table_host(max_num_blocks, 0);
    block_table_host[0] = 0;
    block_table_host[1] = 1;

    auto block_tables = torch::tensor(block_table_host, torch::TensorOptions().dtype(torch::kInt32).device(device));
    block_tables      = block_tables.unsqueeze(0);  // [1, MaxBlocks]

    auto seq_lens = torch::tensor({seq_len}, torch::TensorOptions().dtype(torch::kInt32).device(device));  // [1]

    std::cout << "[Test] Running begin_forward..." << std::endl;
    // Run Forward
    handler->begin_forward(block_tables.data_ptr<int>(),
                           seq_lens.data_ptr<int>(),
                           batch_size,
                           max_num_blocks,
                           num_heads,
                           num_kv_heads,
                           head_dim,
                           page_size);

    std::cout << "[Test] Running attention kernel..." << std::endl;
    auto output = handler->attention(q.data_ptr(),
                                     k_cache.data_ptr(),
                                     v_cache.data_ptr(),
                                     batch_size,
                                     /*seq=*/1,  // Decode is seq=1 query
                                     num_heads,
                                     head_dim);

    // -----------------------------------------------------------------
    // Reference Calculation
    // -----------------------------------------------------------------
    std::cout << "[Test] Computing reference..." << std::endl;

    auto k_cpu = k_cache.to(torch::kCPU);
    auto v_cpu = v_cache.to(torch::kCPU);

    // [NumKVHeads, PageSize, HeadDim] -> [PageSize, NumKVHeads, HeadDim] (NHD)
    // Wait, our cache is HND: [NumKVHeads, PageSize, HeadDim]
    // We need [SeqLen, NumKVHeads, HeadDim]

    // Block 0: k_cpu[0] -> [KV, 16, D] -> Transpose(0,1) -> [16, KV, D]
    auto k_b0 = k_cpu[0].permute({1, 0, 2});
    auto v_b0 = v_cpu[0].permute({1, 0, 2});

    // Block 1: k_cpu[1].slice(1, 0, 1) -> [KV, 1, D] -> Transpose -> [1, KV, D]
    // 17-1 = 16. 16%16 + 1 = 1. Last page len is 1.
    auto k_b1 = k_cpu[1].slice(1, 0, 1).permute({1, 0, 2});
    auto v_b1 = v_cpu[1].slice(1, 0, 1).permute({1, 0, 2});

    auto k_cont = torch::cat({k_b0, k_b1}, 0);  // [17, KV, D]
    auto v_cont = torch::cat({v_b0, v_b1}, 0);  // [17, KV, D]

    auto q_cpu = q.to(torch::kCPU);

    auto ref_out = manual_attention_ref(q_cpu, k_cont, v_cont);

    // Compare
    auto out_cpu = output.to(torch::kCPU).to(torch::kFloat32);  // [1, 1, H, D] -> [1, H, D]
    out_cpu      = out_cpu.squeeze(1);                          // [1, H, D]

    float diff = (out_cpu - ref_out).abs().max().item<float>();
    std::cout << "Max Diff: " << diff << std::endl;

    bool passed = torch::allclose(out_cpu, ref_out, /*rtol=*/1e-2, /*atol=*/1e-2);
    if (passed) {
        std::cout << "[Test] PASSED! Outputs match." << std::endl;
    }
    else {
        std::cerr << "[Test] FAILED! Outputs mismatch." << std::endl;
        std::cerr << "Reference[0][0][0]: " << ref_out[0][0][0].item<float>() << std::endl;
        std::cerr << "Output[0][0][0]:    " << out_cpu[0][0][0].item<float>() << std::endl;
        exit(1);
    }
}

int main()
{
    try {
        test_single_token_decode_correctness();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
