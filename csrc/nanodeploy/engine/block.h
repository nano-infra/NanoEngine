#pragma once
#include <cstdint>
#include <vector>

namespace nanodeploy {

class Block {
public:
    explicit Block(int block_id);

    void update(int64_t hash, const std::vector<int>& token_ids);
    void reset();

    int              block_id;
    int              ref_count = 0;
    int64_t          hash      = -1;
    std::vector<int> token_ids;
};

}  // namespace nanodeploy
