#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nanodeploy {

class Block {
public:
    explicit Block(int block_id, int block_size = 256);

    void update(int64_t hash, const std::vector<int>& token_ids);
    void update(int64_t hash, const int* token_ids, size_t size);
    void reset();

    int              block_id;
    int              ref_count = 0;
    int64_t          hash      = -1;
    std::vector<int> token_ids;
};

}  // namespace nanodeploy
