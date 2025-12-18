#pragma once
#include <deque>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "block.h"

namespace nanodeploy {

class Sequence;

class BlockManager {
public:
    BlockManager(const std::optional<std::string>& engine_id,
                 int sp_idx,
                 int num_blocks,
                 int block_size);
    
    // Static hash calculation (using xxhash)
    static int64_t compute_hash(const std::vector<int>& token_ids, 
                                int64_t prefix = -1);
    
    // Block allocation and deallocation
    bool can_allocate(Sequence& seq) const;
    void allocate(Sequence& seq, int token_idx_from = -1, int token_idx_to = -1);
    void deallocate(Sequence& seq);
    
    // Append related
    bool can_append(Sequence& seq, int num_tokens = 1) const;
    void may_append(Sequence& seq, int num_tokens = 1);
    
    // Accessors
    const std::deque<int>& free_block_ids() const { return free_block_ids_; }
    int num_free_blocks() const { return static_cast<int>(free_block_ids_.size()); }
    const std::vector<Block>& blocks() const { return blocks_; }
    
private:
    Block& allocate_block(int block_id);
    void deallocate_block(int block_id);
    
    std::optional<std::string> engine_id_;
    int sp_idx_;
    int block_size_;
    std::vector<Block> blocks_;
    std::unordered_map<int64_t, int> hash_to_block_id_;
    std::deque<int> free_block_ids_;
    std::unordered_set<int> used_block_ids_;
};

} // namespace nanodeploy
