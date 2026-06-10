#pragma once

#include <string>
#include <vector>

#include "dlengine/csrc/sequence/sequence.h"

namespace dlengine {

class Sequence;

// Paged block allocator for a DSv4 compressed KV cache (one instance per
// compression ratio).  Simpler than the SWA BlockManager — no prefix caching
// and no ref-counting, since each page is owned by exactly one sequence.
//
// Free list is a `std::vector<int>` stack (pop_back / push_back) — every op
// is O(1), data is contiguous, and per-page overhead is just 4 bytes.  A
// parallel `std::vector<uint8_t>` tracks in-use state for double-free guards.
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
    int  allocate_page();          // pop_back from free_pages_
    void deallocate_page(int id);  // push_back to free_pages_

    std::string engine_id_;
    int         ratio_;
    int         num_pages_;
    int         page_size_;
    int         max_blocks_per_seq_;

    std::vector<int>     free_pages_;  // stack of free page IDs
    std::vector<uint8_t> in_use_;      // 1 = in use, 0 = free; indexed by page_id
};

}  // namespace dlengine
