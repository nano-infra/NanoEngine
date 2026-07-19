#pragma once

#include <cstdint>
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
    class PreparedBlockMutation {
    public:
        enum class State {
            PREPARED,
            COMMITTED,
            ABORTED
        };

        PreparedBlockMutation(const PreparedBlockMutation&)            = delete;
        PreparedBlockMutation& operator=(const PreparedBlockMutation&) = delete;
        PreparedBlockMutation(PreparedBlockMutation&& other) noexcept;
        PreparedBlockMutation& operator=(PreparedBlockMutation&& other) noexcept;
        ~PreparedBlockMutation() noexcept;

        // The IDs are in caller-requested order for an exact allocation and
        // current free-list order for a count-based allocation. They are
        // reserved at prepare time but are not attached to a Sequence or group.
        const std::vector<int>& block_ids() const noexcept
        {
            return block_ids_;
        }

        State state() const noexcept
        {
            return state_;
        }

        // Both operations are idempotent. All allocating work and validation
        // happened in prepare_*(), so these publication primitives never throw.
        void commit_noexcept() noexcept;
        void abort_noexcept() noexcept;

    private:
        friend class BlockManager;

        enum class Kind {
            ALLOCATE,
            RELEASE
        };

        struct BlockSnapshot {
            int              block_id = -1;
            int              ref_count = 0;
            int64_t          hash = -1;
            std::vector<int> token_ids;
        };

        PreparedBlockMutation(BlockManager* manager, Kind kind) noexcept;
        void take_from(PreparedBlockMutation&& other) noexcept;

        BlockManager*                  manager_ = nullptr;
        Kind                           kind_    = Kind::ALLOCATE;
        State                          state_   = State::ABORTED;
        std::vector<int>               block_ids_;
        std::vector<BlockSnapshot>     allocation_snapshots_;
        std::list<int>                 staged_free_nodes_;
        std::vector<std::list<int>::iterator> release_node_iters_;
    };

    // Atomically replaces a set of caller-owned references with an exact
    // number of new ownership references on this rank. Refcount-1 releases are
    // reusable by the same transaction, so a full rank can exchange disjoint
    // pending-only blocks without publishing an intermediate free state.
    class PreparedBlockRebalance {
    public:
        enum class State {
            PREPARED,
            COMMITTED,
            ABORTED
        };

        PreparedBlockRebalance(const PreparedBlockRebalance&)            = delete;
        PreparedBlockRebalance& operator=(const PreparedBlockRebalance&) = delete;
        PreparedBlockRebalance(PreparedBlockRebalance&& other) noexcept;
        PreparedBlockRebalance& operator=(PreparedBlockRebalance&& other) noexcept;
        ~PreparedBlockRebalance() noexcept;

        // Stable caller order: reusable release IDs first, then current
        // free-list order. These IDs can be installed in shadow block tables
        // before publication.
        const std::vector<int>& allocation_block_ids() const noexcept
        {
            return allocation_block_ids_;
        }

        const std::vector<int>& release_block_ids() const noexcept
        {
            return release_block_ids_;
        }

        State state() const noexcept
        {
            return state_;
        }

        void commit_noexcept() noexcept;
        void abort_noexcept() noexcept;

    private:
        friend class BlockManager;

        struct BlockSnapshot {
            int              block_id  = -1;
            int              ref_count = 0;
            int64_t          hash      = -1;
            std::vector<int> token_ids;
        };

        explicit PreparedBlockRebalance(BlockManager* manager) noexcept;
        void     take_from(PreparedBlockRebalance&& other) noexcept;

        BlockManager*              manager_ = nullptr;
        State                      state_   = State::ABORTED;
        std::vector<int>           release_block_ids_;
        std::vector<int>           allocation_block_ids_;
        std::vector<int>           reused_block_ids_;
        std::vector<int>           newly_allocated_block_ids_;
        std::vector<BlockSnapshot> allocation_snapshots_;
        std::list<int>             detached_free_nodes_;
        std::list<int>             staged_release_nodes_;
    };

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
    // Reserve free blocks without publishing them into a Sequence. The
    // count-based overload follows current free-list order; the exact overload
    // preserves the caller's ID order in PreparedBlockMutation::block_ids().
    PreparedBlockMutation prepare_allocate_uncached(int count);
    PreparedBlockMutation prepare_allocate_uncached(const std::vector<int>& exact_block_ids);
    // Prepare one ownership-reference release for each unique block ID. LS
    // uncached allocations do not share block IDs, so duplicate IDs are rejected
    // instead of being interpreted as multiple reference releases.
    PreparedBlockMutation prepare_release(const std::vector<int>& owned_block_ids);
    // Prepare a rank-local ownership rebalance. `release_block_ids` contains
    // one unique currently-owned reference per ID; `allocation_count` is the
    // number of references that the caller's final shadow tables need in
    // addition to their immutable historical prefix. The returned allocation
    // IDs may reuse refcount-1 releases and therefore need no free headroom.
    PreparedBlockRebalance prepare_rebalance(const std::vector<int>& release_block_ids,
                                              int                     allocation_count);
    void deallocate(Sequence& seq, BlockContextSlot slot);
    void trim_blocks_to_token_count(Sequence& seq, BlockContextSlot slot, int token_count);
    // Reserve physical blocks without publishing them in a sequence block
    // table. KV consolidation uses this to keep ACTIVE metadata unchanged
    // until the device-to-device copy has completed successfully.
    std::vector<int> reserve_blocks(int count);
    void             release_blocks(const std::vector<int>& block_ids);

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
    // A prepared mutation owns these IDs until commit/abort. Existing mutating
    // APIs fail before touching such blocks, preventing stale prepared commits.
    std::unordered_set<int>               prepared_block_ids_;
    // Free-list nodes are kept in ascending order of this key. Allocation abort
    // can therefore restore exact ordering even when other disjoint mutations
    // are prepared concurrently and temporarily hide neighboring nodes.
    std::vector<uint64_t>                 free_order_keys_;
    uint64_t                              next_free_order_key_ = 0;
};

}  // namespace nanodeploy
