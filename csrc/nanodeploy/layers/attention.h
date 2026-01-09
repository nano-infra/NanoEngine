#pragma once
#include "nanodeploy/core/module.h"
#include <torch/torch.h>

namespace nanodeploy {
namespace layers {

class Attention: public core::Module {
public:
#include <cstdint>  // For int64_t
#include <limits>   // For numeric_limits

    Attention(int   num_heads,
              int   head_dim,
              float scaling,
              int   num_kv_heads,
              int   q_head_dim,
              const std::string& /*type*/ = "GQA"):
        scaling_(scaling)
    {
    }

    torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v)
    {
        // Use SDPA
        // Note: sdp_attention expects [B, H, S, D]
        // We must ensure inputs are contiguous for safe execution on some backends
        q = q.contiguous();
        k = k.contiguous();
        v = v.contiguous();

        return at::scaled_dot_product_attention(q,
                                                k,
                                                v,
                                                /*attn_mask=*/{},
                                                /*dropout_p=*/0.0,
                                                /*is_causal=*/true,
                                                /*scale=*/scaling_);
    }

private:
    float scaling_;
};

}  // namespace layers
}  // namespace nanodeploy
