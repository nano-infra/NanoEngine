#pragma once

#include <list>
#include <string>
#include <unordered_set>
#include <vector>

#include "nanodeploy/csrc/sequence/sequence.h"

namespace nanodeploy {

class Sequence;

// Paged block allocator for a DSv4 compressed KV cache (one instance per
// compression ratio).  Simpler than the SWA BlockManager — no prefix caching
// and no ref-counting, since each page is owned by exactly one sequence.
//
// Pattern mirrors GDNStateManager: intrusive std::list free-list with an
// iterator map for O(1) removal.
class CompressedBlockManager {
public:
    CompressedBlockManager(
        const std::string& engine_id, int ratio, int num_pages, int page_size, int max_blocks_per_seq);

    // Query whether the pool has at least `num_blocks` free pages.
    bool can_allocate(int num_blocks) const;

    // Reserve `num_blocks` pages for `seq` (ACTIVE slot).  Block IDs are
    // stored on `seq.compressed_block_table(ACTIVE, ratio_)`.
    void allocate(Sequence& seq, int num_blocks);

    // Grow the reservation for `seq` by `additional_blocks`.  Returns false
    // without any state change if the pool has insufficient free pages.
    bool may_append(Sequence& seq, int additional_blocks);

    // Release every page currently held by `seq` in `slot` back to the pool
    // and clear the sequence's per-ratio block table for that slot.
    void deallocate(Sequence& seq, BlockContextSlot slot);

    // Accessors
    int ratio() const
    {
        return ratio_;
    }
    int page_size() const
    {
        return page_size_;
    }
    int max_blocks_per_seq() const
    {
        return max_blocks_per_seq_;
    }
    int num_free_pages() const
    {
        return static_cast<int>(free_pages_.size());
    }

private:
    int  allocate_page();          // pop from free_pages_
    void deallocate_page(int id);  // push to free_pages_

    std::string engine_id_;
    int         ratio_;
    int         num_pages_;
    int         page_size_;
    int         max_blocks_per_seq_;

    std::list<int>                        free_pages_;
    std::vector<std::list<int>::iterator> page_id_to_free_list_it_;
    std::unordered_set<int>               used_pages_;
};

}  // namespace nanodeploy
