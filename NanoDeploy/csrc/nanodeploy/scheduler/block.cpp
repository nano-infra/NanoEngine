#include "block.h"

namespace nanodeploy {

Block::Block(int block_id, int block_size): block_id(block_id)
{
    token_ids.reserve(block_size);
}

void Block::update(int64_t hash, const std::vector<int>& token_ids)
{
    this->hash = hash;
    this->token_ids.assign(token_ids.begin(), token_ids.end());
}

void Block::update(int64_t hash, const int* token_ids, size_t size)
{
    this->hash = hash;
    this->token_ids.assign(token_ids, token_ids + size);
}

void Block::reset()
{
    ref_count = 1;
    hash      = -1;
    token_ids.clear();
}

}  // namespace nanodeploy
