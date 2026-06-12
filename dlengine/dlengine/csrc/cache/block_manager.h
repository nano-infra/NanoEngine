#pragma once

#include <list>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dlengine/csrc/sequence/sequence.h"

#include "block.h"

namespace dlengine {

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

    // --- L3 (3FS) tiered KV cache hooks -----------------------------------
    // All L3 behavior is inert unless set_l3_enabled(true) has been called.
    //
    // Lifecycle (driven from Python LLMEngine.step()):
    //   1. set_l3_enabled(true)                       once at startup
    //   2. set_l3_resident_hashes(known_l3_hashes)    before schedule()
    //   3. scheduler.schedule() -> allocate()         emits pending_loads_
    //   4. drain_pending_loads() -> worker USRBIO read into the fresh blocks
    //      BEFORE the prefill forward runs (the leading prefix is marked
    //      "cached" so recompute is skipped — the data MUST be loaded).
    //   5. deallocate() of finished seqs              emits pending_offloads_
    //   6. drain_pending_offloads() -> worker USRBIO write of the GPU block,
    //      then mark_l3_resident(stored_hashes).
    void set_l3_enabled(bool enabled)
    {
        l3_enabled_ = enabled;
    }
    bool l3_enabled() const
    {
        return l3_enabled_;
    }

    // Replace / extend the set of block hashes known to be durable in L3.
    void set_l3_resident_hashes(const std::vector<int64_t>& hashes);
    void mark_l3_resident(const std::vector<int64_t>& hashes);
    bool is_l3_resident(int64_t hash) const
    {
        return l3_resident_hashes_.count(hash) > 0;
    }

    // Streaming full-block hashes for the ACTIVE slot (same hash chain as
    // allocate()), so Python can probe L3 membership with exact keys.
    std::vector<int64_t> compute_block_hashes(Sequence& seq) const;

    // Drain (return + clear) the pending (hash, block_id) work queues.
    std::vector<std::pair<int64_t, int>> drain_pending_loads();
    std::vector<std::pair<int64_t, int>> drain_pending_offloads();

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

    // --- L3 (3FS) tiered KV cache state -----------------------------------
    bool                                 l3_enabled_ = false;
    std::unordered_set<int64_t>          l3_resident_hashes_;
    std::vector<std::pair<int64_t, int>> pending_loads_;     // (hash, block_id)
    std::vector<std::pair<int64_t, int>> pending_offloads_;  // (hash, block_id)
};

}  // namespace dlengine
