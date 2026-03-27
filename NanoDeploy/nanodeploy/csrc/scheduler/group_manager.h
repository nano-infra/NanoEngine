#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "nanodeploy/csrc/cache/block_manager.h"
#include "nanodeploy/csrc/cache/gdn_state_manager.h"
#include "nanodeploy/csrc/sequence/sequence.h"

namespace nanodeploy {

enum class RoutingStrategy {
    RoundRobin,
    LeastBatch,
    LeastCache
};

struct AllocResult {
    int chunk_end;   // num_tokens boundary for the current batch
    int new_tokens;  // budget consumed (= chunk_end - num_cached_tokens)
};

class GroupManager {
public:
    static constexpr int segment_size = 256;

    GroupManager(const std::string& engine_id,
                 int                group_size,
                 int                num_kvcache_blocks,
                 int                kvcache_block_size,
                 int                max_num_seqs,
                 int                max_num_batched_tokens);

    // State queries
    bool is_empty() const
    {
        return running.empty();
    }

    // Block management delegation
    bool can_append(Sequence& seq, int num_tokens = 1);
    bool may_append(Sequence& seq, int num_tokens = 1);

    // Allocation logic
    // num_seqs and num_batched_tokens are maps from dp_idx to count/tokens
    // But wait, in Python:
    // num_seqs: dict[int, int] -> maps master_group to count?
    // Let's check Python code:
    // num_seqs[selected_dp_idx][block_ctx.master_group_id] += 1
    // So passed to can_allocate is num_seqs[selected_dp_idx], which is dict[int, int] (group_id -> count)
    bool can_allocate(Sequence&                           seq,
                      const std::unordered_map<int, int>& num_seqs,
                      const std::unordered_map<int, int>& num_batched_tokens);

    void allocate(Sequence& seq);
    void deallocate(Sequence& seq, BlockContextSlot slot = BlockContextSlot::ACTIVE);

    // Atomic budget-check + full-prompt allocation + chunk computation.
    // Internally: saves/restores num_tokens, sets full_len for block allocation,
    // computes chunk boundary from prefix hits + budget, restricts dispatch for
    // chunked sequences to master group.
    // On success: blocks allocated for full prompt, num_tokens = chunk_end,
    //             returns {chunk_end, new_tokens}.
    // On failure: num_tokens restored, no side effects, returns nullopt.
    std::optional<AllocResult> try_allocate(Sequence&                           seq,
                                            const std::unordered_map<int, int>& num_seqs,
                                            const std::unordered_map<int, int>& num_batched_tokens);

    // Load tracking
    /// \brief Returns the total number of sequences currently running on this engine.
    ///
    /// This aggregates the number of active sequences across all block group
    /// (block group) partitions managed by this GroupManager.
    ///
    /// \note This class does not provide internal synchronization. Callers must
    ///       ensure external synchronization if accessed from multiple threads.
    int num_running_seqs() const
    {
        return num_running_seqs_;
    }

    /// \brief Returns the total number of tokens currently being processed.
    ///
    /// The returned value is the sum of running tokens across all running
    /// sequences and all SP partitions in this manager.
    ///
    /// \note This class does not provide internal synchronization. Callers must
    ///       ensure external synchronization if accessed from multiple threads.
    int num_running_tokens() const
    {
        return num_running_tokens_;
    }

    /// \brief Returns the number of running sequences assigned to a given group index.
    ///
    /// \param group_id The zero-based block group index for which to query
    ///               the number of running sequences.
    /// \return The number of currently running sequences mapped to \p group_id.
    ///
    /// \warning No bounds checking is performed on \p group_id; callers must ensure
    ///          that it is within the valid range of group indices for this engine.
    /// \note This class does not provide internal synchronization. Callers must
    ///       ensure external synchronization if accessed from multiple threads.
    int num_running_seqs_per_group(int group_id) const
    {
        return num_running_seqs_per_group_[group_id];
    }

    /// \brief Returns the number of running tokens assigned to a given group index.
    ///
    /// \param group_id The zero-based block group index for which to query
    ///               the number of running tokens.
    /// \return The number of tokens currently being processed on \p group_id.
    ///
    /// \warning No bounds checking is performed on \p group_id; callers must ensure
    ///          that it is within the valid range of group indices for this engine.
    /// \note This class does not provide internal synchronization. Callers must
    ///       ensure external synchronization if accessed from multiple threads.
    int num_running_tokens_per_group(int group_id) const
    {
        return num_running_tokens_per_group_[group_id];
    }

    // WARNING: This method modifies shared state without thread safety protection.
    // If called concurrently from multiple threads (e.g., in worker_func),
    // this will cause race conditions on the counters.

    /// \brief Adjusts the number of running tokens for a given group index.
    ///
    /// This updates both the global running-token count and the per-group running
    /// token count for the specified \p group_id.
    ///
    /// \param group_id The zero-based block group index whose token count
    ///               should be updated.
    /// \param count  The number of tokens to add. Implementations may pass a
    ///               negative value to decrement the counters when tokens are
    ///               completed or removed.
    ///
    /// \warning No bounds checking is performed on \p group_id; callers must ensure
    ///          that it is within the valid range of group indices for this engine.
    /// \note This class does not provide internal synchronization. Callers must
    ///       ensure external synchronization if accessed from multiple threads.
    void add_running_tokens(int group_id, int count)
    {
        num_running_tokens_ += count;
        num_running_tokens_per_group_[group_id] += count;
    }

    std::unordered_map<int, std::shared_ptr<BlockManager>> block_manager;

    GDNStateManager gdn_state_manager_;

    std::deque<std::shared_ptr<Sequence>>  running;
    std::vector<std::shared_ptr<Sequence>> dummy_seqs;

    RoutingStrategy routing_strategy = RoutingStrategy::RoundRobin;

private:
    void initialize_dummy_seqs();
    int  next_group_id();  // Round-robin counter

    std::string engine_id_;
    int         group_size_;
    int         max_num_seqs_;
    int         max_num_batched_tokens_;

    int kvcache_block_size_;
    int num_kvcache_blocks_;

    int              group_rr_counter_   = 0;
    int              num_running_seqs_   = 0;
    int              num_running_tokens_ = 0;
    std::vector<int> num_running_seqs_per_group_;
    std::vector<int> num_running_tokens_per_group_;

    // Per-group prefix hit counts cached between can_allocate() and allocate().
    // Populated by can_allocate on success; consumed (moved) by allocate.
    std::vector<int> cached_prefix_hints_;
};

}  // namespace nanodeploy
