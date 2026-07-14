#pragma once

#include <cstdint>
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

enum class SchedulerMode {
    CENTRALIZED,
    DECENTRALIZED
};

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
    // Per-DP histogram of active SP sizes.
    // Dimensions: [dp_idx][sp_size], where sp_size is the number of
    // non-zero entries in num_dispatched_tokens for a request.
    std::vector<std::vector<int>> sp_size_hist_per_dp;

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
    std::vector<std::vector<std::vector<int>>> sp_res_matrix;

    // Metrics for waiting queue blocks (per DP worker)
    std::vector<int> waiting_head_blocks;
    std::vector<int> waiting_total_blocks;

    // LS-Decode-Core admission records (one entry per batch admitted in this
    // scheduler step).
    std::vector<uint64_t>                      ls_initial_batch_ids;
    std::vector<uint64_t>                      ls_initial_group_ids;
    std::vector<int>                           ls_initial_kv_dops;
    std::vector<std::vector<int>>              ls_initial_kv_ranks;
    std::vector<std::vector<uint64_t>>         ls_initial_sequence_ids;
    std::vector<std::vector<std::vector<int>>> ls_initial_prompt_kv_tokens;
    std::vector<std::vector<int>>              ls_initial_provisional_pending_targets;
    std::vector<uint64_t>                      ls_initial_admission_orders;
    std::vector<uint32_t>                      ls_initial_admission_attempts;
    std::vector<bool>                          ls_initial_is_recovery_batch;
    std::vector<std::optional<uint64_t>>       ls_initial_parent_batch_ids;
    std::vector<std::string>                   ls_initial_admission_kinds;

    // LS-Decode-Core sealed/pending logical batch observations.
    std::vector<uint64_t>              ls_sealed_batch_ids;
    std::vector<std::vector<uint64_t>> ls_sealed_batch_sequence_ids;
    int                                ls_pending_batch_count             = 0;
    int                                ls_pending_request_count           = 0;
    uint64_t                           ls_oldest_pending_batch_age_steps  = 0;
    uint32_t                           ls_max_pending_batch_attempts      = 0;
    uint64_t                           ls_atomic_admission_no_fit_count   = 0;
    uint64_t                           ls_atomic_admission_merge_count    = 0;
    uint64_t                           ls_atomic_admission_rollback_count = 0;

    // LS-Decode-Core per-iteration group records.
    std::vector<uint64_t>              ls_group_ids;
    std::vector<int>                   ls_group_dp_indices;
    std::vector<int>                   ls_real_batch_sizes;
    std::vector<int>                   ls_master_dops;
    std::vector<int>                   ls_kv_dops;
    std::vector<std::vector<int>>      ls_master_ranks;
    std::vector<std::vector<int>>      ls_master_batch_sizes;
    std::vector<std::vector<int>>      ls_group_rank_allocations;
    std::vector<std::vector<int>>      ls_group_used_kv_tokens;
    std::vector<std::vector<int>>      ls_group_used_kv_blocks;
    std::vector<std::vector<uint64_t>> ls_iteration_sequence_ids;
    std::vector<std::vector<int>>      ls_iteration_master_assignments;
    std::vector<std::vector<int>>      ls_pending_append_blocks_per_master;
    std::vector<std::vector<int>>      ls_new_master_ranks;
    std::vector<std::vector<int>>      ls_reused_passive_master_ranks;
    std::vector<std::string>           ls_scale_reasons;
    std::vector<int64_t>               ls_historical_kv_migration_bytes;
    std::vector<uint64_t>              ls_preempted_sequence_ids;
    std::vector<std::string>           ls_preemption_reasons;
    double                             ls_planning_latency_ms = 0.0;
};

struct InitialBatchPlacement {
    uint64_t                      batch_id = 0;
    uint64_t                      group_id = 0;
    std::vector<uint64_t>         sequence_ids;
    int                           initial_kv_dop = 0;
    std::vector<int>              initial_kv_ranks;
    std::vector<std::vector<int>> prompt_kv_tokens;
    std::vector<int>              provisional_pending_targets;
    uint64_t                      admission_order    = 0;
    uint32_t                      admission_attempts = 0;
    bool                          is_recovery_batch  = false;
    std::optional<uint64_t>       parent_batch_id;
    std::string                   admission_kind = "standalone";
};

enum class PendingDecodeBatchState {
    QUEUED,
    COMMITTING,
    ADMITTED
};

struct PendingDecodeBatch {
    uint64_t                               batch_id = 0;
    std::vector<std::shared_ptr<Sequence>> sequences;
    uint64_t                               enqueue_order      = 0;
    uint64_t                               enqueue_step       = 0;
    uint32_t                               admission_attempts = 0;
    bool                                   is_recovery_batch  = false;
    std::optional<uint64_t>                parent_batch_id;
    PendingDecodeBatchState                state = PendingDecodeBatchState::QUEUED;
};

struct DecodeGroupState {
    uint64_t                               group_id = 0;
    int                                    dp_idx   = -1;
    std::vector<std::shared_ptr<Sequence>> sequences;
    std::vector<InitialBatchPlacement>     initial_batch_placements;
    std::vector<int>                       allocated_attention_ranks;
    std::vector<int>                       last_iteration_masters;
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
              bool               use_new_decode_dynamic_sp_scheduler,
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
              bool               sp_debug,
              int                fixed_sp_size,
              bool               enable_ls_decode_core_scheduler  = false,
              int                ls_decode_initial_kv_dop         = 0,
              int                ls_decode_batch_per_master       = 64,
              bool               ls_decode_enable_memory_scale_up = true,
              const std::string& scheduler_mode                   = "centralized");

    // Queue management
    void add(std::shared_ptr<Sequence> seq);

    // Main scheduling functions
    ScheduleResult schedule();

    // Postprocessing
    void postprocess(const std::vector<std::vector<std::shared_ptr<Sequence>>>& dp_sp_seqs,
                     const std::vector<std::vector<std::vector<int>>>&          dp_sp_token_ids,
                     bool                                                       update_metrics,
                     double                                                     accumulated_step_time_ms,
                     int                                                        loop_count);

    // State queries
    bool is_finished() const;

    // Get total waiting queue sizes (for metrics/logging)
    int get_total_waiting_size() const;
    int get_total_waiting_migration_size() const;

    // Read-only LS logical-batch snapshots for tests and diagnostics.
    std::vector<uint64_t>                           get_ls_pending_batch_ids() const;
    std::vector<std::vector<uint64_t>>              get_ls_pending_batch_sequence_ids() const;
    std::vector<uint32_t>                           get_ls_pending_batch_attempts() const;
    std::vector<bool>                               get_ls_pending_batch_is_recovery() const;
    std::vector<std::optional<uint64_t>>            get_ls_pending_batch_parent_batch_ids() const;
    std::vector<uint64_t>                           get_ls_group_ids() const;
    std::vector<std::vector<uint64_t>>              get_ls_group_sequence_ids() const;
    std::vector<std::vector<uint64_t>>              get_ls_group_initial_batch_ids() const;
    std::vector<std::vector<uint64_t>>              get_ls_group_initial_admission_orders() const;
    std::vector<std::vector<std::vector<uint64_t>>> get_ls_group_initial_sequence_ids() const;
    std::vector<std::pair<uint64_t, uint64_t>>      get_ls_active_batch_owners() const;

    // Test-only failure injection. A non-negative value throws after that
    // many complete sequence allocations inside the next LS admission.
    void set_ls_admission_failure_after_allocations_for_test(int value);
    void set_ls_admission_failure_after_publications_for_test(int value);

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
    std::vector<std::vector<std::shared_ptr<Sequence>>> _schedule_ls_decode_admission();
    std::vector<std::vector<std::shared_ptr<Sequence>>> _schedule_ls_decode();
    std::vector<std::vector<std::shared_ptr<Sequence>>> _schedule_decode_prefill_latency_aware();
    std::optional<std::pair<std::vector<int>, std::vector<std::vector<int>>>>
                     _plan_ls_initial_placement(int                                           dp_idx,
                                                const std::vector<std::shared_ptr<Sequence>>& batch,
                                                const std::vector<int>&                       rank_pool,
                                                const std::vector<std::shared_ptr<Sequence>>& existing_sequences = {},
                                                const std::vector<int>&                       base_allocation = {}) const;
    std::vector<int> _ls_unallocated_ranks(int dp_idx, std::optional<uint64_t> excluding_group = std::nullopt) const;
    void             _merge_ls_groups(uint64_t lhs_group_id, uint64_t rhs_group_id);
    void             _remove_seq_from_ls_group(uint64_t seq_id, bool clear_batch_owner = true);
    void             _reconcile_ls_groups();
    void             _seal_ls_decode_arrivals();
    bool             _ls_batch_fits_empty_system(const std::vector<std::shared_ptr<Sequence>>& batch) const;

    // Decentralized scheduling logic
    ScheduleResult                         _schedule_decentralized();
    std::vector<std::shared_ptr<Sequence>> _schedule_prefill_for_worker(int dp_idx);
    std::vector<std::shared_ptr<Sequence>> _schedule_decode_for_worker(int dp_idx);

    // Routing function for decentralized mode
    int select_dp_worker_for_routing(Sequence& seq);

    // Round-robin counter for DP
    int next_dp_idx();

    // Configuration
    std::string      engine_id_;
    int              loop_count_;
    int              max_num_seqs_;
    int              max_num_batched_tokens_;
    int              max_num_recv_seqs_;
    int              eos_;
    int              attention_dp_;
    int              attention_sp_;
    std::string      mode_;
    double           reserved_blocks_per_req_;
    int              segment_size_;
    bool             enable_dynamic_sp_size_;
    bool             use_new_decode_dynamic_sp_scheduler_;
    std::string      dynamic_sp_size_strategy_;
    int              dynamic_sp_long_request_threshold_;
    int              dynamic_sp_long_request_size_;
    bool             enable_non_uniform_split_;
    bool             sp_debug_;
    bool             enable_ls_decode_core_scheduler_;
    int              ls_decode_initial_kv_dop_;
    int              ls_decode_batch_per_master_;
    bool             ls_decode_enable_memory_scale_up_;
    std::vector<int> ls_empty_system_free_blocks_per_rank_;

    std::string sp_master_selector_;

    SchedulerMode scheduler_mode_ = SchedulerMode::CENTRALIZED;

    int dp_rr_counter_ = 0;

    std::unique_ptr<ThreadPool> thread_pool_;

    uint64_t                                                next_ls_group_id_        = 0;
    uint64_t                                                next_ls_batch_id_        = 0;
    uint64_t                                                next_ls_admission_order_ = 0;
    uint64_t                                                next_ls_enqueue_order_   = 0;
    uint64_t                                                ls_schedule_step_        = 0;
    std::unordered_map<uint64_t, DecodeGroupState>          ls_groups_;
    std::vector<std::vector<uint64_t>>                      ls_group_ids_by_dp_;
    std::unordered_map<uint64_t, uint64_t>                  ls_seq_to_group_;
    std::unordered_map<uint64_t, uint64_t>                  ls_seq_to_batch_;
    std::deque<PendingDecodeBatch>                          ls_pending_decode_batches_;
    std::vector<InitialBatchPlacement>                      ls_step_initial_records_;
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> ls_step_sealed_batches_;
    std::vector<SPStateManager::LSDecodeMasterPlan>         ls_step_group_plans_;
    std::vector<uint64_t>                                   ls_step_group_plan_ids_;
    std::vector<std::vector<int>>                           ls_step_reused_passive_masters_;
    std::vector<uint64_t>                                   ls_step_preempted_sequence_ids_;
    std::vector<std::string>                                ls_step_preemption_reasons_;
    uint64_t                                                ls_step_atomic_no_fit_count_                      = 0;
    uint64_t                                                ls_step_atomic_merge_count_                       = 0;
    uint64_t                                                ls_step_atomic_rollback_count_                    = 0;
    int                                                     ls_admission_failure_after_allocations_for_test_  = -1;
    int                                                     ls_admission_failure_after_publications_for_test_ = -1;
    double                                                  ls_step_planning_latency_ms_                      = 0.0;
};

}  // namespace nanodeploy
