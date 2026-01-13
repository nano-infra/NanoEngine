#pragma once
#include "nanodeploy/core/common.h"
#include "nanodeploy/core/module.h"
#include <torch/torch.h>

namespace nanodeploy {
namespace layers {

template<QuantType Q>
class Linear: public core::Module {
public:
    Linear(int in_features, int out_features, bool bias = false): in_features_(in_features), out_features_(out_features)
    {
        weight = this->register_parameter("weight", torch::empty({out_features, in_features}));
        if (bias) {
            this->bias = this->register_parameter("bias", torch::empty({out_features}));
        }
    }

    torch::Tensor forward(torch::Tensor input)
    {
        return torch::nn::functional::linear(input, weight, bias);
    }

    int           in_features_;
    int           out_features_;
    torch::Tensor weight;
    torch::Tensor bias;
};

template<QuantType Q>
class ColumnParallelLinear: public Linear<Q> {
public:
    ColumnParallelLinear(int in_features, int out_features, bool bias = false):
        Linear<Q>(in_features, out_features, bias)
    {
    }
};

template<QuantType Q>
class RowParallelLinear: public Linear<Q> {
public:
    RowParallelLinear(int in_features, int out_features, bool bias = false): Linear<Q>(in_features, out_features, bias)
    {
    }
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
