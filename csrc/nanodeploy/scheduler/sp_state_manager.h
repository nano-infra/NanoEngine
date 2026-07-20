#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
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
    int sp_size      = 1;
    int seq_len_low  = 0;
    int seq_len_high = 0;
};

class SPStateManager {
public:
    class PreparedLSInitialBatch {
    public:
        enum class State {
            PREPARED,
            COMMITTED,
            ABORTED
        };

        PreparedLSInitialBatch(const PreparedLSInitialBatch&)            = delete;
        PreparedLSInitialBatch& operator=(const PreparedLSInitialBatch&) = delete;
        PreparedLSInitialBatch(PreparedLSInitialBatch&& other) noexcept;
        PreparedLSInitialBatch& operator=(PreparedLSInitialBatch&& other) noexcept;
        ~PreparedLSInitialBatch() noexcept;

        State state() const noexcept
        {
            return state_;
        }

        // The caller must validate immediately before entering its no-throw
        // publication region. A false result means stable Sequence/counter
        // state changed since prepare and the mutation must be aborted.
        bool validate_precommit_noexcept() const noexcept;

        // Idempotent, allocation-free publication/rollback primitives.
        void commit_noexcept() noexcept;
        void abort_noexcept() noexcept;

    private:
        friend class SPStateManager;
        struct Impl;

        explicit PreparedLSInitialBatch(std::unique_ptr<Impl> impl) noexcept;
        void     take_from(PreparedLSInitialBatch&& other) noexcept;

        std::unique_ptr<Impl> impl_;
        State                 state_ = State::ABORTED;
    };

    class PreparedLSRelease {
    public:
        enum class State {
            PREPARED,
            COMMITTED,
            ABORTED
        };

        PreparedLSRelease(const PreparedLSRelease&)            = delete;
        PreparedLSRelease& operator=(const PreparedLSRelease&) = delete;
        PreparedLSRelease(PreparedLSRelease&& other) noexcept;
        PreparedLSRelease& operator=(PreparedLSRelease&& other) noexcept;
        ~PreparedLSRelease() noexcept;

        State state() const noexcept
        {
            return state_;
        }

        bool validate_precommit_noexcept() const noexcept;
        void commit_noexcept() noexcept;
        void abort_noexcept() noexcept;

    private:
        friend class SPStateManager;
        struct Impl;

        explicit PreparedLSRelease(std::unique_ptr<Impl> impl) noexcept;
        void     take_from(PreparedLSRelease&& other) noexcept;

        std::unique_ptr<Impl> impl_;
        State                 state_ = State::ABORTED;
    };

    class PreparedLSIterationMasterPlan {
    public:
        enum class State {
            PREPARED,
            COMMITTED,
            ABORTED
        };

        PreparedLSIterationMasterPlan(const PreparedLSIterationMasterPlan&)            = delete;
        PreparedLSIterationMasterPlan& operator=(const PreparedLSIterationMasterPlan&) = delete;
        PreparedLSIterationMasterPlan(PreparedLSIterationMasterPlan&& other) noexcept;
        PreparedLSIterationMasterPlan& operator=(PreparedLSIterationMasterPlan&& other) noexcept;
        ~PreparedLSIterationMasterPlan() noexcept;

        State state() const noexcept
        {
            return state_;
        }

        bool validate_precommit_noexcept() const noexcept;
        void commit_noexcept() noexcept;
        void abort_noexcept() noexcept;

    private:
        friend class SPStateManager;
        struct Impl;

        explicit PreparedLSIterationMasterPlan(std::unique_ptr<Impl> impl) noexcept;
        void     take_from(PreparedLSIterationMasterPlan&& other) noexcept;

        std::unique_ptr<Impl> impl_;
        State                 state_ = State::ABORTED;
    };

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
        int q_bytes_per_edge   = 1;
        int res_bytes_per_edge = 1;
        int lse_bytes_per_edge = 1;
    };

    struct LatencyBreakdown {
        double total         = 0.0;
        double attention     = 0.0;
        double q             = 0.0;
        double res           = 0.0;
        double lse           = 0.0;
        double max_tokens    = 0.0;
        double max_q_bytes   = 0.0;
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
        int                           max_tokens         = 0;
        int                           total_overflow     = 0;
        int                           extra_participants = 0;
    };

    struct LSDecodeMasterPlan {
        bool             success = false;
        std::string      failure_reason;
        std::string      scale_reason        = "none";
        std::string      assignment_strategy = "source_greedy";
        std::vector<int> allocation;
        std::vector<int> master_ranks;
        std::vector<int> master_batch_sizes;
        // Parallel to the stable request list passed to the planner.
        std::vector<int> sequence_master_ranks;
        std::vector<int> new_allocation_ranks;
        std::vector<int> group_used_kv_tokens;
        std::vector<int> group_used_kv_blocks;
    };

    struct KVTokenRangeMove {
        uint64_t seq_id           = 0;
        int      dp_idx           = -1;
        int      src_sp_rank      = -1;
        int      dst_sp_rank      = -1;
        int      src_block_id     = -1;
        int      src_token_offset = 0;
        int      dst_block_id     = -1;
        int      dst_token_offset = 0;
        int      num_tokens       = 0;
    };

    struct LSKVConsolidationPlan {
        enum class State {
            REJECTED,
            RESERVED,
            DISPATCHED,
            COMMITTED,
            ABORTED
        };

        struct PreparedBlockStage {
            int                                 rank = -1;
            BlockManager::PreparedBlockMutation mutation;

            PreparedBlockStage(int rank, BlockManager::PreparedBlockMutation&& mutation) noexcept:
                rank(rank), mutation(std::move(mutation))
            {
            }

            PreparedBlockStage(const PreparedBlockStage&)            = delete;
            PreparedBlockStage& operator=(const PreparedBlockStage&) = delete;
            PreparedBlockStage(PreparedBlockStage&&) noexcept         = default;
            PreparedBlockStage& operator=(PreparedBlockStage&&) noexcept = default;
        };

        struct SequenceStage {
            std::shared_ptr<Sequence>                sequence;
            BlockContext                             old_context;
            BlockContext                             staged_context;
            std::vector<int>                         source_blocks;
            std::vector<PreparedBlockStage>          destination_allocations;
            std::optional<BlockManager::PreparedBlockMutation> source_release;
        };

        struct SequenceSnapshot {
            std::shared_ptr<Sequence> sequence;
            BlockContext              context;
            SequenceStatus            status = SequenceStatus::WAITING;
        };

        bool                          success = false;
        std::string                   failure_reason;
        uint64_t                      transaction_id = 0;
        uint64_t                      group_id       = 0;
        int                           dp_idx         = -1;
        int                           source_rank    = -1;
        std::vector<int>              retained_ranks;
        std::vector<uint64_t>         group_sequence_ids;
        std::vector<KVTokenRangeMove> moves;
        int64_t                       num_tokens = 0;
        State                         state      = State::REJECTED;
        std::vector<SequenceSnapshot> sequence_snapshots;
        // Prepared mutations keep raw BlockManager pointers. A plan can be
        // retained by Python after Scheduler teardown, so keep every manager
        // alive until sequence_stages (declared after this guard) is destroyed.
        std::vector<std::shared_ptr<BlockManager>> block_manager_lifetime_guards;
        std::vector<SequenceStage>    sequence_stages;
        std::vector<int>              master_seq_counts_before;
        std::vector<int>              master_seq_counts_after;
        std::vector<int>              recv_seq_counts_before;
        std::vector<int>              recv_seq_counts_after;
        int                           running_seqs_before   = 0;
        int                           running_tokens_before = 0;

        // Scheduler publication shadows are populated while the transaction
        // is still RESERVED. The post-copy commit only swaps these vectors and
        // scalar metadata; it performs no allocation.
        std::vector<int> scheduler_allocation_before;
        std::vector<int> scheduler_allocation_after;
        std::vector<int> scheduler_last_iteration_masters_after;
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
                   bool               sp_debug      = false,
                   int                fixed_sp_size = 0);

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
    bool can_append_on_sp(Sequence& seq, int sp_idx, int num_tokens = 1) const;
    bool may_append_on_sp(Sequence& seq, int sp_idx, int num_tokens = 1);

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

    std::optional<DecodeBatchPlan> plan_decode_batch(const std::vector<std::shared_ptr<Sequence>>& pending_seqs) const;

    void apply_planned_placement(Sequence& seq, const PlannedPlacement& placement);

    void               allocate_ls_initial(Sequence& seq);
    // Formal LS path. placement_contexts contain the complete logical prompt
    // placement but no physical block IDs. prepare reserves exact physical
    // blocks, including one fixed dummy-bootstrap token of headroom on each
    // sequence's master, and builds complete shadow ACTIVE contexts without
    // changing stable Sequence state. commit_noexcept() swaps those contexts
    // and publishes counters including that future dummy token
    // (baseline + sum(prompt_tokens + 1)); the caller must append/mark the
    // already-reserved dummy without calling add_running_tokens(). It never
    // calls legacy allocate or may_append APIs.
    PreparedLSInitialBatch
    prepare_ls_initial_batch(const std::vector<std::shared_ptr<Sequence>>& batch,
                             const std::vector<BlockContext>&              placement_contexts);
    // Convenience overload for callers that already installed placement-only
    // ACTIVE contexts. New formal callers should use the explicit-shadow
    // overload above so prepare leaves stable contexts untouched.
    PreparedLSInitialBatch prepare_ls_initial_batch(const std::vector<std::shared_ptr<Sequence>>& batch);

    // Prepare an exact, allocation-free-at-commit ACTIVE release for OFFLOAD
    // and bootstrap-finished cleanup. Other slots remain on the legacy path.
    PreparedLSRelease prepare_ls_release(const std::shared_ptr<Sequence>& sequence,
                                         BlockContextSlot slot = BlockContextSlot::ACTIVE);

    void               allocate_ls_initial_batch(const std::vector<std::shared_ptr<Sequence>>& batch,
                                                 int failure_after_allocations_for_test = -1);
    void               set_decode_master(Sequence& seq, int master_sp_idx);
    int                estimate_pending_append_capacity(int                                           rank,
                                                        const std::vector<std::shared_ptr<Sequence>>& requests,
                                                        const std::vector<std::shared_ptr<Sequence>>& group_sequences,
                                                        int num_output_tokens = 1) const;
    LSDecodeMasterPlan plan_iteration_masters_source_greedy(const std::vector<std::shared_ptr<Sequence>>& requests,
                                                            const std::vector<int>&                       allocation,
                                                            const std::vector<int>&                       extra_ranks,
                                                            int  batch_per_master,
                                                            bool enable_memory_scale_up,
                                                            int  num_output_tokens = 1) const;
    bool               validate_iteration_master_plan(const std::vector<std::shared_ptr<Sequence>>& requests,
                                                      const LSDecodeMasterPlan&                     plan,
                                                      std::string*                                  error = nullptr,
                                                      int num_output_tokens = 1) const;
    // Formal LS Decode publication adapter. It builds complete shadow ACTIVE
    // contexts and rank-local prepared rebalances from a validated plan. The
    // current pending frontier and its output headroom can be transferred on a
    // completely full rank; prepare never changes Sequence/counter state and
    // commit/abort are idempotent, allocation-free and noexcept.
    PreparedLSIterationMasterPlan prepare_iteration_master_plan(const std::vector<std::shared_ptr<Sequence>>& requests,
                                                                const LSDecodeMasterPlan&                     plan,
                                                                int num_output_tokens = 1);
    // Validate the final manager counters for one atomic pool step that may
    // publish a survivor admission before applying the existing Decode
    // iteration's role deltas. Both transactions must still be PREPARED and
    // individually fresh. The check is allocation-free and noexcept so the
    // scheduler can repeat it immediately before entering publication.
    bool validate_ls_pool_step_composition_noexcept(const PreparedLSInitialBatch*        initial,
                                                    const PreparedLSIterationMasterPlan* iteration) const noexcept;
    bool reassign_pending_append(Sequence& seq, int target_sp_idx, int num_output_tokens = 1);
    bool commit_iteration_master_plan(const std::vector<std::shared_ptr<Sequence>>& requests,
                                      const LSDecodeMasterPlan&                     plan,
                                      int num_output_tokens = 1);
    std::shared_ptr<LSKVConsolidationPlan>
                     plan_kv_consolidation(uint64_t                                      transaction_id,
                                           uint64_t                                      group_id,
                                           int                                           dp_idx,
                                           const std::vector<std::shared_ptr<Sequence>>& sequences,
                                           int                                           source_rank,
                                           const std::vector<int>&                       retained_ranks);
    bool             commit_kv_consolidation(const std::shared_ptr<LSKVConsolidationPlan>& plan) noexcept;
    void             abort_kv_consolidation(const std::shared_ptr<LSKVConsolidationPlan>& plan) noexcept;
    std::vector<int> group_used_kv_tokens(const std::vector<std::shared_ptr<Sequence>>& seqs) const;
    std::vector<int> group_used_kv_blocks(const std::vector<std::shared_ptr<Sequence>>& seqs) const;
    int              get_active_master_count(const std::vector<std::shared_ptr<Sequence>>& seqs) const;
    int              get_kv_participant_count(const std::vector<std::shared_ptr<Sequence>>& seqs) const;
    void             rebuild_decode_role_counters();

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

    int master_seq_count(int sp_idx) const
    {
        return master_seq_counts_[sp_idx];
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
    std::deque<std::shared_ptr<Sequence>> waiting;
    std::deque<std::shared_ptr<Sequence>> waiting_migration;

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

    void               initialize_dummy_seqs();
    int                select_master_rank();
    int                effective_target_sp_size(int requested_sp_size, int num_tokens) const;
    std::optional<int> select_bucket_sp_size(int seq_len) const;
    void add_communication(PlanningState& state, int master_sp_idx, const std::vector<int>& dispatched_tokens) const;
    PlanningState build_running_state_snapshot() const;

    std::string engine_id_;
    int         dp_idx_ = -1;
    int         attention_sp_;
    int         max_num_seqs_;
    int         max_num_batched_tokens_;
    int         max_num_recv_seqs_;
    double      reserved_blocks_per_req_;

    int                           kvcache_block_size_;
    int                           segment_size_;
    DynamicSPSizeStrategy         dynamic_sp_size_strategy_;
    int                           long_request_sp_threshold_;
    int                           long_request_sp_size_;
    bool                          enable_dynamic_sp_bucket_policy_;
    std::vector<SPBucketInterval> dynamic_sp_bucket_policy_;

    int              sp_rr_counter_      = 0;
    int              num_running_seqs_   = 0;
    int              num_running_tokens_ = 0;
    std::vector<int> num_recv_seqs_per_sp_;

    bool         enable_dynamic_sp_size_;
    CostModel    cost_model_;
    TrafficModel traffic_model_;
    bool         enable_non_uniform_split_;
    bool         sp_debug_;
    int          fixed_sp_size_;

    SPMasterSelector                     master_selector_;
    std::vector<int>                     master_seq_counts_;
    mutable std::optional<PlanningState> cached_running_state_;
};

}  // namespace nanodeploy
