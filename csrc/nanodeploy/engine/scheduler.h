#pragma once
#include "sequence.h"
#include "sp_state_manager.h"
#include "thread_pool.h"
#include <deque>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace nanodeploy {

// Forward declaration
class MetricsManager;

struct ScheduleResult {
    std::vector<std::vector<std::shared_ptr<Sequence>>> dp_seqs;
    std::vector<std::vector<std::shared_ptr<Sequence>>> dp_sp_seqs;
    std::vector<std::vector<std::shared_ptr<Sequence>>> filtered_dp_sp_seqs;
    bool                                                is_prefill;
};

class Scheduler {
public:
    Scheduler(const std::optional<std::string>& engine_id,
              int                                loop_count,
              int                                max_num_seqs,
              int                                max_num_batched_tokens,
              int                                eos,
              int                                attention_dp,
              int                                attention_sp,
              int                                num_kvcache_blocks,
              int                                kvcache_block_size,
              const std::string&                 mode);

    // Queue management
    void add(std::shared_ptr<Sequence> seq);

    // Main scheduling functions
    ScheduleResult schedule();

    // Postprocessing
    void postprocess(
        const std::vector<std::vector<std::shared_ptr<Sequence>>>& dp_sp_seqs,
        const std::vector<std::vector<std::vector<int>>>&          dp_sp_token_ids,
        bool                                                       update_metrics = true);

    // State queries
    bool is_finished() const;

    // Preemption
    void preempt(int dp_idx, std::shared_ptr<Sequence> seq);

    // Migration management
    void free_to_be_migrated(std::shared_ptr<Sequence> seq);
    void free_to_be_migrated(const std::vector<std::shared_ptr<Sequence>>& seqs);

    // Access to running sequences
    std::deque<std::shared_ptr<Sequence>>& running(int dp_idx);
    const std::deque<std::shared_ptr<Sequence>>& running(int dp_idx) const;

    // Access to block managers
    std::unordered_map<int, std::shared_ptr<BlockManager>>& block_manager(int dp_idx);
    const std::unordered_map<int, std::shared_ptr<BlockManager>>& block_manager(int dp_idx) const;

    // Public members exposed to Python
    std::deque<std::shared_ptr<Sequence>>                waiting;
    std::deque<std::shared_ptr<Sequence>>                waiting_migration;
    std::vector<std::shared_ptr<SPStateManager>>         worker_state;
    std::unordered_map<std::string, std::pair<std::shared_ptr<Sequence>, int>> to_be_migrated;

    // Configuration
    RoutingStrategy routing_strategy = RoutingStrategy::RoundRobin;

private:
    // Internal scheduling logic
    std::vector<std::vector<std::shared_ptr<Sequence>>> _schedule_prefill();
    std::vector<std::vector<std::shared_ptr<Sequence>>> _schedule_decode();

    // Round-robin counter for DP
    int next_dp_idx();

    // Configuration
    std::optional<std::string> engine_id_;
    int                        loop_count_;
    int                        max_num_seqs_;
    int                        max_num_batched_tokens_;
    int                        eos_;
    int                        attention_dp_;
    int                        attention_sp_;
    std::string                mode_;

    int dp_rr_counter_ = 0;

    std::unique_ptr<ThreadPool> thread_pool_;
};

}  // namespace nanodeploy
