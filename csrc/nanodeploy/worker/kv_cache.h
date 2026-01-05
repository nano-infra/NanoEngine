#pragma once

#include <torch/torch.h>
#include <vector>

namespace nanodeploy {

class KvCache {
public:
    KvCache(int /*num_layers*/, int /*num_heads*/, int /*head_dim*/, int /*max_batch_size*/, int /*max_seq_len*/)
    {
        // Allocation placeholder
    }

    void step(const std::vector<int>& /*sequence_ids*/, const std::vector<int>& /*positions*/)
    {
        // Update cache logic
    }
};

}  // namespace nanodeploy
