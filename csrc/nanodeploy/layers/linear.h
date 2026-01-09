#pragma once
#include "nanodeploy/core/common.h"
#include "nanodeploy/core/module.h"
#include <torch/torch.h>

namespace nanodeploy {
namespace layers {

// Base class or interface for Linear layers
template<QuantType Q>
class Linear: public core::Module {
public:
    Linear(int in_features, int out_features, bool bias = false): in_features_(in_features), out_features_(out_features)
    {
        // Use empty initialization for speed (we load weights immediately after)
        weight = this->register_parameter("weight", torch::empty({out_features, in_features}));
        if (bias) {
            this->bias = this->register_parameter("bias", torch::empty({out_features}));
        }
    }

    torch::Tensor forward(torch::Tensor input)
    {
        // Using at::linear or torch::nn::functional::linear if available or manual
        return torch::nn::functional::linear(input, weight, bias);
    }

    // protected:
    int           in_features_;
    int           out_features_;
    torch::Tensor weight;
    torch::Tensor bias;
};

// Parallel Linear Stubs
// Parallel Linear Stubs (Currently just wrappers around Linear)
template<QuantType Q>
class ColumnParallelLinear: public Linear<Q> {
public:
    ColumnParallelLinear(int in_features, int out_features, bool bias = false):
        Linear<Q>(in_features, out_features, bias)
    {
    }
    // TODO: Implement gather/scatter logic for column parallel
};

template<QuantType Q>
class RowParallelLinear: public Linear<Q> {
public:
    RowParallelLinear(int in_features, int out_features, bool bias = false): Linear<Q>(in_features, out_features, bias)
    {
    }
    // TODO: Implement gather/scatter logic for row parallel
};

template<QuantType Q>
class QKVParallelLinear: public ColumnParallelLinear<Q> {
public:
    using ColumnParallelLinear<Q>::ColumnParallelLinear;
};

template<QuantType Q>
class MergedColumnParallelLinear: public ColumnParallelLinear<Q> {
public:
    using ColumnParallelLinear<Q>::ColumnParallelLinear;
};

}  // namespace layers
}  // namespace nanodeploy
