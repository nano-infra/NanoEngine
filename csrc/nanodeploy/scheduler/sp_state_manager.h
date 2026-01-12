#pragma once

#include <deque>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "nanodeploy/sequence/sequence.h"

#include "block_manager.h"
#include "load_statistics.h"
#include "sp_size_policy.h"

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
                   // SP size policy parameters
                   const std::string& sp_size_mode              = "segment",
                   float              initial_avg_prompt_length = 1024.0f,
                   float              initial_avg_output_length = 256.0f,
                   int                stats_window_size         = 1000);

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
    void add_running_tokens([[maybe_unused]] int sp_idx, int count)
    {
        num_running_tokens_ += count;
    }

    // Public members to be exposed to Python
    std::unordered_map<int, std::shared_ptr<BlockManager>> block_manager;
    std::deque<std::shared_ptr<Sequence>>                  running;
    std::vector<std::shared_ptr<Sequence>>                 dummy_seqs;

    RoutingStrategy routing_strategy = RoutingStrategy::RoundRobin;

    // === Load Statistics Access ===
    LoadStatistics&       load_stats() { return load_stats_; }
    const LoadStatistics& load_stats() const { return load_stats_; }
    
    // === SP Size Policy Access ===
    SPSizePolicy&       sp_size_policy() { return sp_size_policy_; }
    const SPSizePolicy& sp_size_policy() const { return sp_size_policy_; }

    // === Helper Methods for SP Size Decision ===
    
    /**
     * @brief Get free blocks per rank
     */
    std::vector<int> get_free_blocks_per_rank() const;
    
    /**
     * @brief Get used blocks per rank
     */
    std::vector<int> get_used_blocks_per_rank() const;
    
    /**
     * @brief Get current batch size (master seq count) per rank
     */
    std::vector<int> get_batch_size_per_rank() const;
    
    /**
     * @brief Record waiting queue size for load statistics
     */
    void record_waiting_queue_size(int queue_size);
    
    /**
     * @brief Record a trace sample for offline analysis
     * 
     * @param prompt_length Request prompt length
     * @param free_blocks_per_rank Free blocks on each rank
     * @param batch_size_per_rank Batch size on each rank
     * @param long_used_per_rank Long request blocks on each rank
     * @param decision SP size decision made
     */
    void record_trace_sample(
        int                      prompt_length,
        const std::vector<int>&  free_blocks_per_rank,
        const std::vector<int>&  batch_size_per_rank,
        const std::vector<int>&  long_used_per_rank,
        const SPSizeDecision&    decision);
    
    /**
     * @brief Set trace export path (enables trace collection)
     */
    void set_trace_export_path(const std::string& path);
    
    /**
     * @brief Flush and close trace file
     */
    void flush_trace_file();

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

    SPMasterSelector master_selector_;
    std::vector<int> master_seq_counts_;

    // === Load-aware SP size policy ===
    int            num_kvcache_blocks_;  // Total blocks per rank
    LoadStatistics load_stats_;
    SPSizePolicy   sp_size_policy_;
    
    // === Scheduling log ===
    bool enable_scheduling_log_ = true;  // Log each scheduling decision
    
    // === Trace collection for offline analysis ===
    std::string trace_export_path_;
    std::ofstream trace_file_;
    bool trace_enabled_ = false;
};

}  // namespace nanodeploy
