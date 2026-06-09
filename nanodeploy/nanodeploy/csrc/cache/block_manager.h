#pragma once

#include <list>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nanodeploy/csrc/sequence/sequence.h"

#include "block.h"

namespace nanodeploy {

class Sequence;

class BlockManager {
public:
    BlockManager(const std::string& engine_id, int group_id, int num_blocks, int block_size);

    // Static hash calculation (using xxhash)
    static int64_t compute_hash(const std::vector<int>& token_ids, int64_t prefix = -1);
    static int64_t compute_hash(const int* token_ids, size_t size, int64_t prefix = -1);

    // Block allocation and deallocation.
    // can_allocate returns the number of prefix cache hits (>= 0) on success,
    // or -1 if allocation is impossible.  The caller can forward this value as
    // `prefix_hint` to allocate() to avoid a redundant hash scan.
    int  can_allocate(Sequence& seq) const;
    void allocate(Sequence& seq, int prefix_hint = -1);

    // Count consecutive leading blocks whose hash matches an already-active
    // (shared) block.  These don't need a free slot — we just bump ref_count.
    int  count_active_prefix_hits(Sequence& seq) const;
    void deallocate(Sequence& seq, BlockContextSlot slot);

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
    int                                   group_id_;
    int                                   block_size_;
    std::vector<Block>                    blocks_;
    std::unordered_map<int64_t, int>      hash_to_block_id_;
    std::list<int>                        free_block_ids_;
    std::vector<std::list<int>::iterator> block_id_to_free_list_it_;
    std::unordered_set<int>               used_block_ids_;
};

}  // namespace nanodeploy
