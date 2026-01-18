#pragma once
#include "nanodeploy/csrc/core/module.h"
#include "nanodeploy/csrc/logging.h"
#include <torch/torch.h>

namespace nanodeploy {
namespace layers {

class RotaryEmbedding: public core::Module {
public:
    RotaryEmbedding(int dim, int max_position, double base, torch::Device device = torch::kCPU):
        dim_(dim), max_position_(max_position), base_(base)
    {
        inv_freq_ =
            1.0 / torch::pow(base, torch::arange(0, dim, 2, torch::dtype(torch::kFloat32).device(device)) / dim);
        register_buffer("inv_freq", inv_freq_);
        _set_cos_sin_cache(max_position, torch::kFloat32, device);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor positions, torch::Tensor q, torch::Tensor k)
    {
        // positions: [batch, seq] (or [seq])
        // Expand cos/sin to match batch size
        // This is a simplified reference impl. Real impl should use FlashInfer/CUDA kernels.

        auto seq_len = q.size(2);

        // Ensure cache is on correct device
        // Ensure cache is on correct device and dtype
        if (cos_cached_.device() != q.device() || cos_cached_.scalar_type() != q.scalar_type()) {
            cos_cached_ = cos_cached_.to(q.device(), q.scalar_type());
            sin_cached_ = sin_cached_.to(q.device(), q.scalar_type());
        }

        // Validate and Clamp positions
        // ... (commented out code remains commented)

        // Gather cos/sin based on positions
        // positions: [B, S] or [TotalTokens]
        // Flatten positions to handle both 1D and 2D
        auto flat_pos = positions.flatten();

        auto cos = cos_cached_.index({flat_pos});  // [Total, Dim]
        auto sin = sin_cached_.index({flat_pos});

        // Reshape explicitly to match Q structure [B, S, D]
        // Q: [Batch, Heads, Seq, HeadDim]
        int64_t batch = q.size(0);
        int64_t seq   = q.size(2);  // Should match seq_len
        int64_t dim   = cos.size(-1);

        cos = cos.view({batch, seq, dim});
        sin = sin.view({batch, seq, dim});

        // Unsqueeze for broadcasting over Heads [B, H, S, D]
        // Target: [B, 1, S, D]
        cos = cos.unsqueeze(1);
        sin = sin.unsqueeze(1);

        // Cast to input dtype (important for BF16/FP16)
        if (cos.scalar_type() != q.scalar_type()) {
            cos = cos.to(q.scalar_type());
            sin = sin.to(q.scalar_type());
        }

        auto q_out = _apply_rotary_pos_emb(q, cos, sin);
        auto k_out = _apply_rotary_pos_emb(k, cos, sin);
        return {q_out, k_out};
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
