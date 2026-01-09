#include <cassert>
#include <cmath>
#include <iostream>
#include <torch/torch.h>
#include <vector>

#include "nanodeploy/layers/rotary_embedding.h"

using namespace nanodeploy;

// Reference RoPE Implementation
// x: [B, S, H, D]
// pos: [B, S]
std::tuple<torch::Tensor, torch::Tensor>
manual_rope_ref(torch::Tensor q, torch::Tensor k, torch::Tensor pos, int head_dim, float base)
{
    // 1. Compute Cos/Sin
    // inv_freq = 1.0 / (base ^ (i / dim)) for i = 0, 2, 4...
    auto opts = q.options().dtype(torch::kFloat32);  // Calc in FP32

    int  dim      = head_dim;
    auto inv_freq = 1.0 / torch::pow(base, torch::arange(0, dim, 2, opts).div(dim));

    // Outer product: pos [B*S] x inv_freq [D/2] -> [B*S, D/2]
    auto t = torch::arange(dim / 2, opts);

    // We need to broadcast pos carefully
    // Flatten pos: [TotalTokens]
    auto pos_flat = pos.flatten().to(torch::kFloat32);  // [T]

    // freqs = pos.outer(inv_freq)
    auto freqs = torch::outer(pos_flat, inv_freq);  // [T, D/2]

    // emb = cat(freqs, freqs) -> [T, D]
    auto emb = torch::cat({freqs, freqs}, -1);

    // cos, sin
    auto cos = emb.cos().unsqueeze(1);  // [T, 1, D] (Broadcast over heads)
    auto sin = emb.sin().unsqueeze(1);  // [T, 1, D]

    // 2. Rotate Q, K
    // Flash Attention style rotation:
    // x_rot = [-x2, x1]
    // out = x * cos + x_rot * sin
    // Where x is [..., D]. x1 = x[..., :D/2], x2 = x[..., D/2:]

    // Reshape q, k to [TotalTokens, H, D]
    auto q_flat = q.view({-1, q.size(2), q.size(3)}).to(torch::kFloat32);
    auto k_flat = k.view({-1, k.size(2), k.size(3)}).to(torch::kFloat32);

    auto rotate_half = [](torch::Tensor x) {
        auto x1 = x.slice(-1, 0, x.size(-1) / 2);
        auto x2 = x.slice(-1, x.size(-1) / 2);
        return torch::cat({-x2, x1}, -1);
    };

    auto q_embed = (q_flat * cos) + (rotate_half(q_flat) * sin);
    auto k_embed = (k_flat * cos) + (rotate_half(k_flat) * sin);

    return {q_embed.view_as(q), k_embed.view_as(k)};
}

void test_rope_correctness()
{
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA not available." << std::endl;
        return;
    }
    torch::Device device(torch::kCUDA);
    torch::manual_seed(42);

    std::cout << "[Test] Starting RoPE Correctness..." << std::endl;

    int   head_dim   = 128;
    int   num_heads  = 4;
    int   seq_len    = 17;
    int   batch_size = 1;
    float base       = 1000000.0f;  // Qwen 2.5/3 standard? Or 10000? Let's test generic first.

    // Initialize Module
    // We assume max_pos=2048
    auto rope = std::make_shared<layers::RotaryEmbedding>(head_dim, 2048, base, device);
    // rope->to(device); // Not needed/supported, handled by constructor or auto-move

    // Inputs (BFloat16)
    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    auto q    = torch::randn({batch_size, seq_len, num_heads, head_dim}, opts);
    auto k    = torch::randn({batch_size, seq_len, num_heads, head_dim}, opts);
    auto pos =
        torch::arange(seq_len, torch::TensorOptions().dtype(torch::kLong).device(device)).unsqueeze(0);  // [1, S]

    // Run Module
    // Run Module
    std::cout << "[Test] Running C++ RoPE..." << std::endl;
    auto [q_out, k_out] = rope->forward(pos,
                                        q.transpose(1, 2),  // [B, S, H, D] -> [B, H, S, D]
                                        k.transpose(1, 2));

    // Output is [B, H, S, D]. Transpose back to [B, S, H, D] for comparison
    q_out = q_out.transpose(1, 2);
    k_out = k_out.transpose(1, 2);

    // Run Reference
    std::cout << "[Test] Running Reference RoPE..." << std::endl;
    auto [q_ref, k_ref] = manual_rope_ref(q.to(torch::kFloat32), k.to(torch::kFloat32), pos, head_dim, base);

    // Compare
    auto q_cpu     = q_out.to(torch::kCPU).to(torch::kFloat32);
    auto k_cpu     = k_out.to(torch::kCPU).to(torch::kFloat32);
    auto q_ref_cpu = q_ref.to(torch::kCPU);

    float diff = (q_cpu - q_ref_cpu).abs().max().item<float>();
    std::cout << "Max Diff: " << diff << std::endl;

    // RoPE computed in BF16/Half vs FP32 ref might have larger error
    bool passed = torch::allclose(q_cpu, q_ref_cpu, /*rtol=*/1e-2, /*atol=*/2e-2);

    if (passed) {
        std::cout << "[Test] PASSED! Outputs match." << std::endl;
    }
    else {
        std::cerr << "[Test] FAILED! Outputs mismatch." << std::endl;
        exit(1);
    }
}

int main()
{
    test_rope_correctness();
    return 0;
}
