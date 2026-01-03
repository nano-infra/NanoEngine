#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "nanodeploy/sequence/sequence.h"

namespace nanodeploy {

// Define request and response types for easier usage
using RunReq  = std::vector<std::shared_ptr<Sequence>>;
using RunResp = torch::Tensor;

class DummyRunner {
public:
    DummyRunner() = default;

    // Pure logic: run sequences and return tensor
    RunResp run(RunReq dp_seqs)
    {
        int num_sequence = dp_seqs.size();
        return torch::zeros({num_sequence, 16}, torch::kFloat32);
    }
};

}  // namespace nanodeploy
