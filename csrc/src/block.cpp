#include "block.h"

namespace nanodeploy {

Block::Block(int block_id) : block_id(block_id) {}

void Block::update(int64_t hash, const std::vector<int>& token_ids) {
    this->hash = hash;
    this->token_ids = token_ids;
}

void Block::reset() {
    ref_count = 1;
    hash = -1;
    token_ids.clear();
}

} // namespace nanodeploy
