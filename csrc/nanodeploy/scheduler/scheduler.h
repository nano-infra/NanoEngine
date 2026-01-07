#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "nanodeploy/sequence/sequence.h"

#include "sp_state_manager.h"
#include "thread_pool.h"

namespace nanodeploy {

// Forward declaration
class MetricsManager;

// Result of a single scheduling step.
// This struct is returned by `schedule()` and summarizes which sequences
// should be executed on each data-parallel (DP) worker (and, if applicable,
// on each sequence-parallel (SP) shard) for the current iteration.
struct ScheduleResult {
    // Sequences scheduled per DP worker for this step.
    // Outer index: DP worker index.
    // Inner vector: sequences assigned to that DP worker.
    std::vector<std::vector<std::shared_ptr<Sequence>>> dp_seqs;

    // Sequences laid out per (DP, SP) shard for this step.
    // Outer index: DP worker index.
    // Inner vector: sequences assigned to that DP worker after applying
    // sequence-parallel (SP) partitioning / layout.
    std::vector<std::vector<std::shared_ptr<Sequence>>> dp_sp_seqs;

    // Filtered subset of `dp_sp_seqs` that will actually be executed in this
    // iteration (for example, after removing finished / paused sequences or
    // enforcing per-step limits on tokens or sequences).
    // Same indexing convention as `dp_sp_seqs`.
    std::vector<std::vector<std::shared_ptr<Sequence>>> filtered_dp_sp_seqs;

    // Indicates whether this scheduling step is a prefill step (true) or a
    // decode step (false). Callers can use this to select the appropriate
    // execution path.
    bool is_prefill;

    // SP counts
    std::vector<std::vector<int>> sp_send_counts;
    std::vector<std::vector<int>> sp_recv_counts;

    // Matrix of SP communication counts.
    // Dimensions: [dp_idx][master_sp_rank][participant_sp_rank]
    // Value: Number of requests sent from master_sp_rank to participant_sp_rank.
    // std::vector<std::vector<std::vector<int>>> sp_comm_matrix;

    // Matrix of Q communication counts (Master -> Participant).
    // Dimensions: [dp_idx][master_sp_rank][participant_sp_rank]
    // Value: Number of Q requests sent from master_sp_rank to participant_sp_rank.
    std::vector<std::vector<std::vector<int>>> sp_q_matrix;

    // Matrix of Res communication counts (Participant -> Master).
    // Dimensions: [dp_idx][participant_sp_rank][master_sp_rank]
    // Value: Number of Res requests sent from participant_sp_rank to master_sp_rank.
    // std::vector<std::vector<std::vector<int>>> sp_res_matrix;

    // Metrics for waiting queue blocks
    int waiting_head_blocks  = 0;
    int waiting_total_blocks = 0;
};

class Scheduler {
public:
    Scheduler(const std::string& engine_id,
              int                loop_count,
              int                max_num_seqs,
              int                max_num_batched_tokens,
              int                max_num_recv_seqs,
              int                eos,
              int                attention_dp,
              int                attention_sp,
              int                num_kvcache_blocks,
              int                kvcache_block_size,
              const std::string& mode,
              double             reserved_blocks_per_req,
              int                segment_size,
              bool               enable_dynamic_sp_size,
              bool               enable_non_uniform_split,
              const std::string& sp_master_selector);

    // Queue management
    void add(std::shared_ptr<Sequence> seq);

    // Main scheduling functions
    ScheduleResult schedule();

    // Postprocessing
    void postprocess(const std::vector<std::vector<std::shared_ptr<Sequence>>>& dp_sp_seqs,
                     const std::vector<std::vector<std::vector<int>>>&          dp_sp_token_ids,
                     bool                                                       update_metrics,
                     double                                                     step_duration_ms,
                     int                                                        loop_count);

    // State queries
    bool is_finished() const;

    // Preemption
    void preempt(int dp_idx, std::shared_ptr<Sequence> seq);

    // Migration management
    void free_to_be_migrated(std::shared_ptr<Sequence> seq);
    void free_to_be_migrated(const std::vector<std::shared_ptr<Sequence>>& seqs);

    // Access to running sequences
    std::deque<std::shared_ptr<Sequence>>&       running(int dp_idx);
    const std::deque<std::shared_ptr<Sequence>>& running(int dp_idx) const;

    // Access to block managers
    std::unordered_map<int, std::shared_ptr<BlockManager>>&       block_manager(int dp_idx);
    const std::unordered_map<int, std::shared_ptr<BlockManager>>& block_manager(int dp_idx) const;

    // Public members exposed to Python
    std::deque<std::shared_ptr<Sequence>>                              waiting;
    std::deque<std::shared_ptr<Sequence>>                              waiting_migration;
    std::vector<std::shared_ptr<SPStateManager>>                       worker_state;
    std::unordered_map<int, std::pair<std::shared_ptr<Sequence>, int>> to_be_migrated;

    // Configuration
    RoutingStrategy routing_strategy = RoutingStrategy::RoundRobin;

private:
    // Internal scheduling logic
    std::vector<std::vector<std::shared_ptr<Sequence>>> _schedule_prefill();
    std::vector<std::vector<std::shared_ptr<Sequence>>> _schedule_decode();

    // Round-robin counter for DP
    int next_dp_idx();

    // Configuration
    std::string engine_id_;
    int         loop_count_;
    int         max_num_seqs_;
    int         max_num_batched_tokens_;
    int         max_num_recv_seqs_;
    int         eos_;
    int         attention_dp_;
    int         attention_sp_;
    std::string mode_;
    double      reserved_blocks_per_req_;
    int         segment_size_;
    bool        enable_dynamic_sp_size_;
    bool        enable_non_uniform_split_;

    int dp_rr_counter_ = 0;

    std::unique_ptr<ThreadPool> thread_pool_;
};

}  // namespace nanodeploy
