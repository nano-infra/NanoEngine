#pragma once
#include "nanodeploy/csrc/core/module.h"
#include <torch/torch.h>

namespace nanodeploy {
namespace layers {

class SiluAndMul: public core::Module {
public:
    torch::Tensor forward(torch::Tensor x)
    {
        // x shape: [batch, seq, 2 * intermediate_size]
        auto chunks = x.chunk(2, /*dim=*/-1);
        return torch::nn::functional::silu(chunks[0]) * chunks[1];
    }
};

}  // namespace layers
}  // namespace nanodeploy
