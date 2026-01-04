#pragma once
#include "nanodeploy/core/module.h"
#include <torch/torch.h>

namespace nanodeploy {
namespace layers {

class Attention: public core::Module {
public:
    Attention(
        int num_heads, int head_dim, float scaling, int num_kv_heads, int q_head_dim, const std::string& type = "GQA")
    {
    }

    torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v)
    {
        // q: [batch, heads, seq, head_dim] (after transpose)
        // k, v: [batch, heads, seq, head_dim]
        // This impl expects q, k, v to be already in [B, H, S, D] format or broadcastable.
        // For PyTorch SDPA, inputs are typically (query, key, value)

        // C++ SDPA API: torch::nn::functional::scaled_dot_product_attention(query, key, value, attn_mask, dropout_p,
        // is_causal)
        return at::scaled_dot_product_attention(q,
                                                k,
                                                v,
                                                /*attn_mask=*/{},
                                                /*dropout_p=*/0.0,
                                                /*is_causal=*/true);
    }
};

}  // namespace layers
}  // namespace nanodeploy
