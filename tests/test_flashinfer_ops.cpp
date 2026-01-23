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

    auto block_tables = torch::tensor(
        block_table_host, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    block_tables = block_tables.unsqueeze(0);  // [1, MaxBlocks]

    auto seq_lens = torch::tensor(
        {seq_len}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true));  // [1]

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

void test_prefill_correctness()
{
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA not available, skipping test." << std::endl;
        return;
    }
    torch::Device device(torch::kCUDA);
    torch::manual_seed(42);

    std::cout << "[Test] Starting PrefillCorrectness..." << std::endl;

    // Config
    int batch_size   = 2;
    int num_heads    = 16;
    int num_kv_heads = 8;
    int head_dim     = 128;
    int page_size    = 16;

    // Seq lengths: 17, 33
    // Total tokens: 50
    std::vector<int> seq_lens_host = {17, 33};
    std::vector<int> q_indptr_host = {0, 17, 50};
    int              total_tokens  = 50;

    // Max blocks per seq
    // 17 -> 2 blocks
    // 33 -> 3 blocks
    int max_num_blocks = 3;

    // Handler
    auto handler = std::make_unique<ops::FlashInferOps>(1, num_heads, num_kv_heads, head_dim, page_size, device);

    // Data (BF16)
    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    // Q: [TotalTokens, H, D]
    auto q = torch::randn({total_tokens, num_heads, head_dim}, opts);

    // KV Cache
    // Total blocks needed = 2 + 3 = 5.
    // Let's allocate 16 blocks to be safe.
    int  total_blocks = 16;
    auto k_cache      = torch::randn({total_blocks, num_kv_heads, page_size, head_dim}, opts);
    auto v_cache      = torch::randn({total_blocks, num_kv_heads, page_size, head_dim}, opts);

    // Block Tables
    // Seq 0: [0, 1]
    // Seq 1: [2, 3, 4]
    std::vector<int> block_tables_host = {0, 1, 0, 2, 3, 4};  // [Batch, MaxBlocks] -> [2, 3] flattened?
    // Wait, begin_forward_prefill expects raw pointer.
    // Assuming row-major [Batch, MaxBlocks] where MaxBlocks is passed.

    auto block_tables = torch::tensor(block_tables_host,
                                      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));  // Host tensor
    auto q_indptr =
        torch::tensor(q_indptr_host, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));  // Host tensor

    // Seq 0: 17 tokens, page 16 -> last page len 1
    // Seq 1: 33 tokens, page 16 -> last page len 1
    std::vector<int> last_page_lens_host = {1, 1};
    auto             last_page_lens =
        torch::tensor(last_page_lens_host, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

    std::cout << "[Test] Running begin_forward_prefill..." << std::endl;
    handler->begin_forward_prefill(q_indptr.data_ptr<int>(),
                                   block_tables.data_ptr<int>(),
                                   last_page_lens.data_ptr<int>(),
                                   batch_size,
                                   max_num_blocks,
                                   num_heads,
                                   num_kv_heads,
                                   head_dim,
                                   page_size);

    std::cout << "[Test] Running prefill kernel..." << std::endl;
    auto output = handler->prefill(q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), total_tokens);

    // -----------------------------------------------------------------
    // Reference Check
    // -----------------------------------------------------------------
    std::cout << "[Test] Computing reference..." << std::endl;

    auto k_cpu = k_cache.to(torch::kCPU);  // [TotalBlocks, KV, Page, D]
    auto v_cpu = v_cache.to(torch::kCPU);
    auto q_cpu = q.to(torch::kCPU);  // [TotalTokens, H, D]

    // Split Q and Compute for each sequence
    // Result Accumulator
    auto ref_out = torch::zeros_like(q_cpu);

    // Helper to reconstruct Contiguous K/V from blocks
    auto reconstruct_kv = [&](const std::vector<int>& blocks, int seq_len) {
        std::vector<torch::Tensor> k_parts, v_parts;
        int                        remaining = seq_len;
        for (int b_idx : blocks) {
            int valid = std::min(remaining, page_size);
            if (valid <= 0)
                break;
            // blocks are [KV, Page, D]. Need [Page, KV, D] for processing logic usually?
            // manual_ref expects [Seq, KV, D].
            // cache is [KV, Page, D] (HND).
            auto k_blk = k_cpu[b_idx].permute({1, 0, 2}).slice(0, 0, valid);  // [Page, KV, D] sliced
            auto v_blk = v_cpu[b_idx].permute({1, 0, 2}).slice(0, 0, valid);
            k_parts.push_back(k_blk);
            v_parts.push_back(v_blk);
            remaining -= valid;
        }
        return std::make_pair(torch::cat(k_parts, 0), torch::cat(v_parts, 0));
    };

    // Seq 0
    {
        int  start = 0;
        int  len   = 17;
        auto q_seq = q_cpu.slice(0, start, start + len).unsqueeze(0);  // [1, Seq, H, D]
        // Manual ref expects q: [1, H, D] ?? No, manual_attention_ref expects single token Q?
        // My manual_attention_ref implementation (line 17) takes q [1, H, D].
        // It does: q_.view({B, H, Sq, D}). If Sq > 1, it handles it!
        // Line 40: Sq=1 hardcoded in comments but `Sq` variable logic?
        // Let's check `manual_attention_ref` again.
        // Line 25: `int64_t Sq = 1;`
        // Line 40: `q_ = q.view({B, H, Sq, D});`
        // If I pass q with Sq > 1, `q.view` will fail or misfit if q is [1, H, S, D].
        // If q is [1, H, D] (from arguments q size), Sq is inferred as D size??
        // `q.size(1)` is H. `q.size(2)` is D.
        // It assumes input is [1, H, D].

        // I need to update `manual_attention_ref` to support sequence Q?
        // Or loop tokens in verification.
        // Let's loop tokens for verification to reuse existing ref.

        auto kv_pair = reconstruct_kv({0, 1}, len);
        auto k_cont  = kv_pair.first;
        auto v_cont  = kv_pair.second;

        // Causal Masking is tricky with manual ref?
        // manual_ref computes `scores`. It does NOT apply causal mask.
        // FlashInfer prefill IS CAUSAL.
        // So I must mask `scores` in manual calculation.
        // Since `manual_attention_ref` doesn't support causal mask, I must verify token-by-token.
        // For token i (0..16), it attends to 0..i.

        for (int i = 0; i < len; ++i) {
            auto q_tok = q_seq.slice(1, i, i + 1).squeeze(1);  // [1, H, D]
            auto k_ctx = k_cont.slice(0, 0, i + 1);            // [i+1, KV, D]
            auto v_ctx = v_cont.slice(0, 0, i + 1);

            auto out_tok       = manual_attention_ref(q_tok, k_ctx, v_ctx);  // [1, H, D]
            ref_out[start + i] = out_tok.squeeze(0);
        }
    }

    // Seq 1
    {
        int  start   = 17;
        int  len     = 33;
        auto q_seq   = q_cpu.slice(0, start, start + len).unsqueeze(0);
        auto kv_pair = reconstruct_kv({2, 3, 4}, len);
        auto k_cont  = kv_pair.first;
        auto v_cont  = kv_pair.second;

        for (int i = 0; i < len; ++i) {
            auto q_tok = q_seq.slice(1, i, i + 1).squeeze(1);
            auto k_ctx = k_cont.slice(0, 0, i + 1);
            auto v_ctx = v_cont.slice(0, 0, i + 1);

            auto out_tok       = manual_attention_ref(q_tok, k_ctx, v_ctx);
            ref_out[start + i] = out_tok.squeeze(0);
        }
    }

    // Compare
    auto out_cpu = output.to(torch::kCPU).to(torch::kFloat32);

    float diff = (out_cpu - ref_out).abs().max().item<float>();
    std::cout << "Max Diff: " << diff << std::endl;
    // BF16 precision is low, especially for accumulation.
    // 1e-2 might be tight?
    // Usually < 0.5 for large values, but for attention (softmax 0..1) -> values are weighted sums of V.
    // V is random normal (mean 0 var 1).
    bool passed = diff < 0.1;  // Relaxed for BF16/accumulation

    if (passed) {
        std::cout << "[Test] PASSED! Outputs match." << std::endl;
    }
    else {
        std::cerr << "[Test] FAILED! Outputs mismatch." << std::endl;
        // Find first mismatch
        auto diff_t    = (out_cpu - ref_out).abs();
        auto flat_diff = diff_t.view({-1});
        auto flat_out  = out_cpu.view({-1});
        auto flat_ref  = ref_out.view({-1});
        for (int i = 0; i < flat_diff.numel(); ++i) {
            if (flat_diff[i].item<float>() > 0.5) {  // Threshold
                std::cerr << "Mismatch at flat index " << i << ": Ref=" << flat_ref[i].item<float>()
                          << " Out=" << flat_out[i].item<float>() << " Diff=" << flat_diff[i].item<float>()
                          << std::endl;
                break;
            }
        }
        exit(1);
    }
}

int main()
{
    try {
        test_single_token_decode_correctness();
        test_prefill_correctness();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
