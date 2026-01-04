#pragma once
#include "nanodeploy/core/module.h"
#include <torch/torch.h>

namespace nanodeploy {
namespace layers {

class Embedding: public core::Module {
public:
    Embedding(int num_embeddings, int embedding_dim)
    {
        weight = register_parameter("weight", torch::randn({num_embeddings, embedding_dim}));
    }

    torch::Tensor forward(torch::Tensor input)
    {
        return torch::nn::functional::embedding(input, weight);
    }

protected:
    torch::Tensor weight;
};

class VocabParallelEmbedding: public Embedding {
public:
    using Embedding::Embedding;
    // For now, no parallel logic, just inheriting forward
};

class ParallelLMHead: public core::Module {
public:
    ParallelLMHead(int vocab_size, int hidden_size) {}
};

}  // namespace layers
}  // namespace nanodeploy
