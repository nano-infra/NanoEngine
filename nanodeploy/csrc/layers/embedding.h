#pragma once
#include "nanodeploy/csrc/core/module.h"
#include <torch/torch.h>

namespace nanodeploy {
namespace layers {

class Embedding: public core::Module {
public:
    Embedding(int num_embeddings, int embedding_dim, torch::Device device = torch::kCPU)
    {
        weight = register_parameter(
            "weight", torch::randn({num_embeddings, embedding_dim}, torch::TensorOptions().device(device)));
    }

    torch::Tensor forward(torch::Tensor input)
    {
        return torch::nn::functional::embedding(input, weight);
    }

public:
    torch::Tensor weight;
};

class VocabParallelEmbedding: public Embedding {
public:
    using Embedding::Embedding;
    // For now, no parallel logic, just inheriting forward
};

class ParallelLMHead: public core::Module {
public:
    ParallelLMHead(int vocab_size, int hidden_size, torch::Device device = torch::kCPU)
    {
        weight = register_parameter("weight",
                                    torch::randn({vocab_size, hidden_size}, torch::TensorOptions().device(device)));
    }

    torch::Tensor forward(torch::Tensor input)
    {
        return torch::nn::functional::linear(input, weight);
    }

    torch::Tensor weight;
};

}  // namespace layers
}  // namespace nanodeploy
