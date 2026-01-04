#pragma once
#include "nanodeploy/core/module.h"
#include <torch/torch.h>

namespace nanodeploy {
namespace layers {

class RMSNorm: public core::Module {
public:
    RMSNorm(int hidden_size, double eps = 1e-6): eps_(eps)
    {
        weight = register_parameter("weight", torch::ones({hidden_size}));
    }

    torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor residual = {})
    {
        // Pytorch Reference:
        // input_dtype = hidden_states.dtype
        // hidden_states = hidden_states.to(torch.float32)
        // variance = hidden_states.pow(2).mean(-1, keepdim=True)
        // hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        // return weight * hidden_states.to(input_dtype)

        auto input_dtype = hidden_states.scalar_type();
        auto x           = hidden_states.to(torch::kFloat32);
        auto variance    = x.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
        x                = x * torch::rsqrt(variance + eps_);
        return (weight * x).to(input_dtype);
    }

private:
    double        eps_;
    torch::Tensor weight;
};

}  // namespace layers
}  // namespace nanodeploy
