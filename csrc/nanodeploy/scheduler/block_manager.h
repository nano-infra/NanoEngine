#pragma once

#include <list>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "block.h"
#include "nanodeploy/sequence/sequence.h"

namespace nanodeploy {

class Sequence;

class BlockManager {
public:
    BlockManager(const std::string& engine_id, int sp_idx, int num_blocks, int block_size);

    // Static hash calculation (using xxhash)
    static int64_t compute_hash(const std::vector<int>& token_ids, int64_t prefix = -1);
    static int64_t compute_hash(const int* token_ids, size_t size, int64_t prefix = -1);

    // Block allocation and deallocation
    bool can_allocate(Sequence& seq) const;
    void allocate(Sequence& seq, int token_idx_from = -1, int token_idx_to = -1);
    // LS-Decode-Core dummy admission deliberately bypasses prefix reuse so
    // logical placement and physical capacity accounting remain identical.
    void allocate_uncached(Sequence& seq);
    void deallocate(Sequence& seq, BlockContextSlot slot);
    void trim_blocks_to_token_count(Sequence& seq, BlockContextSlot slot, int token_count);

    // Append related
    bool can_append(Sequence& seq, int num_tokens = 1) const;
    bool may_append(Sequence& seq, int num_tokens = 1);

    // Accessors
    std::vector<int> free_block_ids() const
    {
        return std::vector<int>(free_block_ids_.begin(), free_block_ids_.end());
    }
    int num_free_blocks() const
    {
        return static_cast<int>(free_block_ids_.size());
    }
    const std::vector<Block>& blocks() const
    {
        return blocks_;
    }

private:
    Block& allocate_block(int block_id);
    void   deallocate_block(int block_id);

    std::string                           engine_id_;
    int                                   sp_idx_;
    int                                   block_size_;
    std::vector<Block>                    blocks_;
    std::unordered_map<int64_t, int>      hash_to_block_id_;
    std::list<int>                        free_block_ids_;
    std::vector<std::list<int>::iterator> block_id_to_free_list_it_;
    std::unordered_set<int>               used_block_ids_;
};

}  // namespace nanodeploy
