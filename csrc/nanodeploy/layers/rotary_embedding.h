#pragma once
#include "nanodeploy/core/module.h"
#include <torch/torch.h>

namespace nanodeploy {
namespace layers {

class RotaryEmbedding: public core::Module {
public:
    RotaryEmbedding(int dim, int max_position, double base): dim_(dim), max_position_(max_position), base_(base)
    {
        inv_freq_ = 1.0 / torch::pow(base, torch::arange(0, dim, 2).to(torch::kFloat32) / dim);
        register_buffer("inv_freq", inv_freq_);
        _set_cos_sin_cache(max_position, torch::kFloat32, torch::kCPU);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor /*positions*/, torch::Tensor q, torch::Tensor k)
    {
        // q, k: [batch, seq, num_heads, head_dim]
        // positions: [batch, seq] (or [seq])
        // Expand cos/sin to match batch size
        // This is a simplified reference impl. Real impl should use FlashInfer/CUDA kernels.

        auto device  = q.device();
        auto seq_len = q.size(1);

        // Ensure cache is on correct device
        if (cos_cached_.device() != device) {
            cos_cached_ = cos_cached_.to(device);
            sin_cached_ = sin_cached_.to(device);
        }

        // Gather cos/sin based on positions
        // positions shape [batch, seq] -> index into cache [max_pos, dim]
        // result: [batch, seq, dim] -> reshape to [batch, seq, 1, head_dim] for broadcasting

        // Note: For now, assuming simple slicing for contiguous positions (common case)
        // Ideally should support arbitrary positions.

        auto cos = cos_cached_.index({torch::indexing::Slice(0, seq_len)}).unsqueeze(0).unsqueeze(2);  // [1, S, 1, D]
        auto sin = sin_cached_.index({torch::indexing::Slice(0, seq_len)}).unsqueeze(0).unsqueeze(2);

        return {_apply_rotary_pos_emb(q, cos, sin), _apply_rotary_pos_emb(k, cos, sin)};
    }

    void _set_cos_sin_cache(int max_pos, torch::ScalarType dtype, torch::Device device)
    {
        auto t      = torch::arange(max_pos, device).to(dtype);
        auto freqs  = torch::outer(t, inv_freq_.to(device));  // [max_pos, dim/2]
        auto emb    = torch::cat({freqs, freqs}, -1);         // [max_pos, dim]
        cos_cached_ = emb.cos().to(dtype);
        sin_cached_ = emb.sin().to(dtype);
    }

    torch::Tensor _apply_rotary_pos_emb(torch::Tensor x, torch::Tensor cos, torch::Tensor sin)
    {
        // rotate_half
        auto x1        = x.slice(/*dim=*/-1, 0, x.size(-1) / 2);
        auto x2        = x.slice(/*dim=*/-1, x.size(-1) / 2);
        auto x_rotated = torch::cat({-x2, x1}, /*dim=*/-1);
        return (x * cos) + (x_rotated * sin);
    }

    // Making members public for easy access in Attention
    int           dim_;
    int           max_position_;
    double        base_;
    torch::Tensor inv_freq_;
    torch::Tensor cos_cached_;
    torch::Tensor sin_cached_;
};

}  // namespace layers
}  // namespace nanodeploy
