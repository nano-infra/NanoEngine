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
    LeastCache,
    VLLMLoadBalance  // vLLM-style: score = waiting * 4 + running
};

enum class SPMasterSelector {
    RoundRobin,
    LeastBatch,
    LeastCache
};

enum class DynamicSPSizeStrategy {
    Legacy,
    LongShortSP8,
    Bucket
};

struct SPBucketInterval {
    int sp_size = 1;
    int seq_len_low = 0;
    int seq_len_high = 0;
};

class SPStateManager {
public:
    struct StageModel {
        double a = 1.0;
        double b = 0.0;

        double predict(double load) const
        {
            return a * load + b;
        }
    };

    struct CostModel {
        StageModel attention;
        StageModel q;
        StageModel res;
        StageModel lse;
    };

    struct TrafficModel {
        int q_bytes_per_edge = 1;
        int res_bytes_per_edge = 1;
        int lse_bytes_per_edge = 1;
    };

    struct LatencyBreakdown {
        double total = 0.0;
        double attention = 0.0;
        double q = 0.0;
        double res = 0.0;
        double lse = 0.0;
        double max_tokens = 0.0;
        double max_q_bytes = 0.0;
        double max_res_bytes = 0.0;
        double max_lse_bytes = 0.0;
    };

    struct PlannedPlacement {
        int              master_sp_idx = 0;
        std::vector<int> num_dispatched_tokens;
    };

    struct DecodeBatchPlan {
        std::vector<PlannedPlacement> placements;
        LatencyBreakdown              latency;
        int                           max_tokens = 0;
        int                           total_overflow = 0;
        int                           extra_participants = 0;
    };

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
                   const std::string& dynamic_sp_size_strategy,
                   int                dynamic_sp_long_request_threshold,
                   int                dynamic_sp_long_request_size,
                   bool               enable_dynamic_sp_bucket_policy,
                   const std::string& dynamic_sp_bucket_policy,
                   double             attention_cost_a,
                   double             attention_cost_b,
                   double             q_cost_a,
                   double             q_cost_b,
                   double             res_cost_a,
                   double             res_cost_b,
                   double             lse_cost_a,
                   double             lse_cost_b,
                   int                q_bytes_per_edge,
                   int                res_bytes_per_edge,
                   int                lse_bytes_per_edge,
                   bool               enable_non_uniform_split,
                   const std::string& sp_master_selector,
                   bool               sp_debug = false,
                   int                fixed_sp_segments = 0);

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

    std::optional<DecodeBatchPlan> plan_decode_batch(
        const std::vector<std::shared_ptr<Sequence>>& pending_seqs) const;

    void apply_planned_placement(Sequence& seq, const PlannedPlacement& placement);

    // Build and reuse immutable running-state snapshots within a single
    // scheduler step. This avoids rescanning all running sequences for each
    // tentative decode-batch plan in the latency-aware scheduler.
    void begin_decode_planning() const;
    void end_decode_planning() const;

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
    struct PlanningState {
        std::vector<int> tokens;
        std::vector<int> master_counts;
        std::vector<int> recv_counts;
        std::vector<int> free_blocks;
        std::vector<int> batch_tokens;
        std::vector<int> send_q;
        std::vector<int> recv_q;
        std::vector<int> send_res;
        std::vector<int> recv_res;
        std::vector<int> send_lse;
        std::vector<int> recv_lse;
        int              group_max = 0;
        int              rr_cursor = 0;
    };

    void initialize_dummy_seqs();
    int  select_master_rank();
    std::optional<int> select_bucket_sp_size(int seq_len) const;
    void add_communication(PlanningState& state,
                           int            master_sp_idx,
                           const std::vector<int>& dispatched_tokens) const;
    PlanningState build_running_state_snapshot() const;

    std::string engine_id_;
    int         dp_idx_ = -1;
    int         attention_sp_;
    int         max_num_seqs_;
    int         max_num_batched_tokens_;
    int         max_num_recv_seqs_;
    double      reserved_blocks_per_req_;

    int kvcache_block_size_;
    int segment_size_;
    DynamicSPSizeStrategy dynamic_sp_size_strategy_;
    int                   long_request_sp_threshold_;
    int                   long_request_sp_size_;
    bool                  enable_dynamic_sp_bucket_policy_;
    std::vector<SPBucketInterval> dynamic_sp_bucket_policy_;

    int              sp_rr_counter_      = 0;
    int              num_running_seqs_   = 0;
    int              num_running_tokens_ = 0;
    std::vector<int> num_recv_seqs_per_sp_;

    bool enable_dynamic_sp_size_;
    CostModel cost_model_;
    TrafficModel traffic_model_;
    bool enable_non_uniform_split_;
    bool sp_debug_;
    int  fixed_sp_segments_;

    SPMasterSelector master_selector_;
    std::vector<int> master_seq_counts_;
    mutable std::optional<PlanningState> cached_running_state_;
};

}  // namespace nanodeploy
