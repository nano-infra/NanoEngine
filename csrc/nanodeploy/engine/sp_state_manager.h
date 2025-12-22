#pragma once
#include "block_manager.h"
#include "sequence.h"
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nanodeploy {

enum class RoutingStrategy {
    RoundRobin,
    LeastBatch,
    LeastCache
};

class SPStateManager {
public:
    static constexpr int segment_size = 1024;

    SPStateManager(const std::optional<std::string>& engine_id,
                   int                               attention_sp,
                   int                               num_kvcache_blocks,
                   int                               kvcache_block_size,
                   int                               max_num_seqs,
                   int                               max_num_batched_tokens);

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
    // num_seqs: dict[int, int] -> maps master_sp_rank to count?
    // Let's check Python code:
    // num_seqs[selected_dp_idx][block_ctx.master_sp_idx] += 1
    // So passed to can_allocate is num_seqs[selected_dp_idx], which is dict[int, int] (sp_idx -> count)
    bool can_allocate(Sequence&                           seq,
                      const std::unordered_map<int, int>& num_seqs,
                      const std::unordered_map<int, int>& num_batched_tokens);

    void allocate(Sequence& seq);
    void deallocate(Sequence& seq);

    // Load tracking
    /// \brief Returns the total number of sequences currently running on this engine.
    ///
    /// This aggregates the number of active sequences across all sequence-parallel
    /// (SP) partitions managed by this SPStateManager.
    ///
    /// \note This class does not provide internal synchronization. Callers must
    ///       ensure external synchronization if accessed from multiple threads.
    int num_running_seqs() const { return num_running_seqs_; }

    /// \brief Returns the total number of tokens currently being processed.
    ///
    /// The returned value is the sum of running tokens across all running
    /// sequences and all SP partitions in this manager.
    ///
    /// \note This class does not provide internal synchronization. Callers must
    ///       ensure external synchronization if accessed from multiple threads.
    int num_running_tokens() const { return num_running_tokens_; }

    /// \brief Returns the number of running sequences assigned to a given SP index.
    ///
    /// \param sp_idx The zero-based sequence-parallel index for which to query
    ///               the number of running sequences.
    /// \return The number of currently running sequences mapped to \p sp_idx.
    ///
    /// \warning No bounds checking is performed on \p sp_idx; callers must ensure
    ///          that it is within the valid range of SP indices for this engine.
    /// \note This class does not provide internal synchronization. Callers must
    ///       ensure external synchronization if accessed from multiple threads.
    int num_running_seqs_per_sp(int sp_idx) const { return num_running_seqs_per_sp_[sp_idx]; }

    /// \brief Returns the number of running tokens assigned to a given SP index.
    ///
    /// \param sp_idx The zero-based sequence-parallel index for which to query
    ///               the number of running tokens.
    /// \return The number of tokens currently being processed on \p sp_idx.
    ///
    /// \warning No bounds checking is performed on \p sp_idx; callers must ensure
    ///          that it is within the valid range of SP indices for this engine.
    /// \note This class does not provide internal synchronization. Callers must
    ///       ensure external synchronization if accessed from multiple threads.
    int num_running_tokens_per_sp(int sp_idx) const { return num_running_tokens_per_sp_[sp_idx]; }

    /// \brief Adjusts the number of running tokens for a given SP index.
    ///
    /// This updates both the global running-token count and the per-SP running
    /// token count for the specified \p sp_idx.
    ///
    /// \param sp_idx The zero-based sequence-parallel index whose token count
    ///               should be updated.
    /// \param count  The number of tokens to add. Implementations may pass a
    ///               negative value to decrement the counters when tokens are
    ///               completed or removed.
    ///
    /// \warning No bounds checking is performed on \p sp_idx; callers must ensure
    ///          that it is within the valid range of SP indices for this engine.
    /// \note This class does not provide internal synchronization. Callers must
    ///       ensure external synchronization if accessed from multiple threads.
    void add_running_tokens(int sp_idx, int count) {
        num_running_tokens_ += count;
        num_running_tokens_per_sp_[sp_idx] += count;
    }

    // Public members to be exposed to Python
    std::unordered_map<int, std::shared_ptr<BlockManager>> block_manager;
    std::deque<std::shared_ptr<Sequence>>                  running;
    std::vector<std::shared_ptr<Sequence>>                 dummy_seqs;

    RoutingStrategy routing_strategy = RoutingStrategy::RoundRobin;

private:
    void initialize_dummy_seqs();
    int  next_sp_idx();  // Round-robin counter

    std::optional<std::string> engine_id_;
    int                        attention_sp_;
    int                        max_num_seqs_;
    int                        max_num_batched_tokens_;

    int sp_rr_counter_ = 0;
    int num_running_seqs_ = 0;
    int num_running_tokens_ = 0;
    std::vector<int> num_running_seqs_per_sp_;
    std::vector<int> num_running_tokens_per_sp_;
};

}  // namespace nanodeploy
