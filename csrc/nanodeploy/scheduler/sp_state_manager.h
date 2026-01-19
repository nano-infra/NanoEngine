#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "nanodeploy/sequence/sequence.h"

#include "block_manager.h"

namespace nanodeploy {

enum class RoutingStrategy {
    RoundRobin,
    LeastBatch,
    LeastCache
};

enum class SPMasterSelector {
    RoundRobin,
    LeastBatch,
    LeastCache
};

class SPStateManager {
public:
    SPStateManager(const std::string& engine_id,
                   int                attention_sp,
                   int                num_kvcache_blocks,
                   int                kvcache_block_size,
                   int                max_num_seqs,
                   int                max_num_batched_tokens,
                   int                max_num_recv_seqs,
                   double             reserved_blocks_per_req,
                   int                segment_size,
                   bool               enable_dynamic_sp_size,
                   bool               enable_non_uniform_split,
                   const std::string& sp_master_selector,
                   bool               sp_debug = false);

    void set_dp_idx(int dp_idx)
    {
        dp_idx_ = dp_idx;
    }

    int segment_size() const
    {
        return segment_size_;
    }

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
    void deallocate(Sequence& seq, BlockContextSlot slot = BlockContextSlot::ACTIVE);

    // Load tracking
    /// \brief Returns the total number of sequences currently running on this engine.
    ///
    /// This aggregates the number of active sequences across all sequence-parallel
    /// (SP) partitions managed by this SPStateManager.
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

    int num_recv_seqs_per_sp(int sp_idx) const
    {
        return num_recv_seqs_per_sp_[sp_idx];
    }

    // WARNING: This method modifies shared state without thread safety protection.
    // If called concurrently from multiple threads (e.g., in worker_func),
    // this will cause race conditions on the counters.
    void add_running_tokens(int sp_idx, int count)
    {
        (void)sp_idx;  // Unused parameter (reserved for future use)
        num_running_tokens_ += count;
    }

    // Public members to be exposed to Python
    std::unordered_map<int, std::shared_ptr<BlockManager>> block_manager;
    std::deque<std::shared_ptr<Sequence>>                  running;
    std::vector<std::shared_ptr<Sequence>>                 dummy_seqs;
    
    // Waiting queues for decentralized scheduler mode
    std::deque<std::shared_ptr<Sequence>>                  waiting;
    std::deque<std::shared_ptr<Sequence>>                  waiting_migration;

    RoutingStrategy routing_strategy = RoutingStrategy::RoundRobin;
    
    // Helper methods for decentralized scheduler
    bool is_waiting_empty() const
    {
        return waiting.empty() && waiting_migration.empty();
    }
    
    int get_waiting_queue_size() const
    {
        return static_cast<int>(waiting.size() + waiting_migration.size());
    }
    
    int get_total_load() const
    {
        return num_running_seqs_ + get_waiting_queue_size();
    }

private:
    void initialize_dummy_seqs();
    int  select_master_rank();

    std::string engine_id_;
    int         dp_idx_ = -1;
    int         attention_sp_;
    int         max_num_seqs_;
    int         max_num_batched_tokens_;
    int         max_num_recv_seqs_;
    double      reserved_blocks_per_req_;

    int kvcache_block_size_;
    int segment_size_;

    int              sp_rr_counter_      = 0;
    int              num_running_seqs_   = 0;
    int              num_running_tokens_ = 0;
    std::vector<int> num_recv_seqs_per_sp_;

    bool enable_dynamic_sp_size_;
    bool enable_non_uniform_split_;
    bool sp_debug_;

    SPMasterSelector master_selector_;
    std::vector<int> master_seq_counts_;
};

}  // namespace nanodeploy
