#pragma once

#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

enum class ScheduleAction {
    ADMISSION,
    DECODE,
    KV_CONSOLIDATION
};

enum class LSAddError {
    NONE,
    ALREADY_ADDED_OR_ASSIGNED,
    IGNORE_EOS_REQUIRED,
    INVALID_MAX_TOKENS,
    FUTURE_TOKEN_NO_FIT,
    CURRENT_EXACT_NO_FIT
};

struct LSAddResult {
    bool        accepted    = true;
    int         assigned_dp = -1;
    LSAddError  error       = LSAddError::NONE;
    std::string reason;
};

enum class LSAdmissionKind {
    FRESH,
    OFFLOAD_READMIT
};

enum class LSAdmissionTargetKind {
    STANDALONE,
    CAPACITY_APPEND
};

struct LSAdmissionRecord {
    std::shared_ptr<Sequence> sequence;
    int                       dp_idx   = -1;
    uint64_t                  batch_id = 0;
    std::optional<uint64_t>   group_id_after_commit;
    LSAdmissionKind           admission_kind = LSAdmissionKind::FRESH;
    LSAdmissionTargetKind     target_kind    = LSAdmissionTargetKind::STANDALONE;
    int                       planned_kv_dop = 0;
    std::vector<int>          planned_kv_ranks;
    bool                      bootstrap_finished = false;
    int                       bootstrap_token_id = 0;
};

enum class LSFatalCode {
    NO_PROGRESS_INVARIANT,
    UNRECOVERABLE_CAPACITY,
    DECODE_PREPARE_OR_VALIDATE_FAILED,
    METRIC_COMMIT_FAILED,
    KV_CONSOLIDATION_FAILED,
    POST_PUBLICATION_INVARIANT
};

class LSSchedulerFatalError: public std::runtime_error {
public:
    LSSchedulerFatalError(LSFatalCode code, const std::string& message): std::runtime_error(message), code_(code) {}

    LSFatalCode fatal_code() const noexcept
    {
        return code_;
    }

private:
    LSFatalCode code_;
};

// Result of a single scheduling step.
// This struct is returned by `schedule()` and summarizes which sequences
// should be executed on each data-parallel (DP) worker (and, if applicable,
// on each sequence-parallel (SP) shard) for the current iteration.
struct ScheduleResult {
    ScheduleAction action = ScheduleAction::DECODE;

    // Actual number of model-forward loops for this action. Decode uses the
    // configured fixed loop count; scheduler-only actions use one.
    int execution_loop_count = 1;

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
    bool is_prefill = false;

    // Set only for an exclusive KV_CONSOLIDATION maintenance step. The plan
    // is already RESERVED and must be executed directly by the engine.
    std::shared_ptr<SPStateManager::LSKVConsolidationPlan> kv_consolidation_plan;

    // Typed LoongServe-style Decode-only ABI. Admissions are scheduler/KV
    // side effects and may accompany a Decode action for the step-entry
    // running snapshot. Newly admitted requests start Decode on the next step.
    std::vector<LSAdmissionRecord>     ls_admission_records;
    std::vector<std::vector<uint64_t>> ls_real_decode_ids_by_dp;
    std::vector<std::vector<uint64_t>> ls_running_ids_by_dp_after_commit;
    // Resource-version telemetry. Epochs are engine-lifetime monotonic pool
    // versions; reservation/rollback/NO_FIT never increments them.
    std::vector<uint64_t> ls_pool_resource_epoch_before;
    std::vector<uint64_t> ls_pool_resource_epoch_after;

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
    // Optional NANODEPLOY_LS_SCHEDULER_PHASE_TIMING diagnostics. Top-level
    // phase values are mutually exclusive; nested hotspot values are inclusive
    // and therefore must not be added to the top-level values as percentages.
    bool                                      ls_scheduler_phase_timing_enabled = false;
    std::unordered_map<std::string, double>   ls_scheduler_phase_timing_ms;
    std::unordered_map<std::string, uint64_t> ls_scheduler_phase_timing_counts;

    // Automatic KV-consolidation decision telemetry. Shadow mode populates
    // these fields without reserving blocks or changing placement.
    bool        ls_kv_consolidation_candidate       = false;
    int64_t     ls_kv_consolidation_group_id        = -1;
    int         ls_kv_consolidation_source_rank     = -1;
    int         ls_kv_consolidation_target_dop      = -1;
    uint64_t    ls_kv_consolidation_stable_steps    = 0;
    double      ls_kv_consolidation_group_util      = 0.0;
    std::string ls_kv_consolidation_decision_reason = "off";
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

struct DecodeGroupState {
    uint64_t                               group_id = 0;
    int                                    dp_idx   = -1;
    std::vector<std::shared_ptr<Sequence>> sequences;
    std::vector<InitialBatchPlacement>     initial_batch_placements;
    std::vector<int>                       allocated_attention_ranks;
    std::vector<int>                       last_iteration_masters;
    int                                    kv_candidate_target_dop   = -1;
    uint64_t                               kv_candidate_stable_steps = 0;
    std::vector<uint64_t>                  kv_candidate_member_ids;
    std::vector<int>                       kv_candidate_allocation;
    uint64_t                               last_scale_up_step      = 0;
    uint64_t                               last_consolidation_step = 0;
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
              bool               enable_ls_decode_core_scheduler                 = false,
              int                ls_decode_initial_kv_dop                        = 0,
              int                ls_decode_batch_per_master                      = 64,
              bool               ls_decode_enable_memory_scale_up                = true,
              const std::string& scheduler_mode                                  = "centralized",
              const std::string& ls_kv_consolidation_mode                        = "off",
              double             ls_kv_consolidation_candidate_util              = 0.50,
              double             ls_kv_consolidation_target_high_watermark       = 0.80,
              int                ls_kv_consolidation_stable_steps                = 32,
              int                ls_kv_consolidation_cooldown_steps              = 64,
              int                ls_kv_consolidation_check_interval_steps        = 8,
              int                ls_kv_consolidation_max_source_blocks_per_event = 0,
              bool               ls_decode_enable_future_kv_admission            = true,
              int                ls_max_num_ooe                                  = 10,
              int                ls_running_max_req_size                         = 1000,
              int                ls_admission_max_tokens_per_pool                = 0,
              int                ls_min_comp_bound_decoding_batch_size           = 128);

    // Queue management
    LSAddResult add(std::shared_ptr<Sequence> seq);
    LSAddResult precheck_add_identity(const std::shared_ptr<Sequence>& seq) const;

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
    std::vector<int>                                get_ls_group_allocated_ranks(uint64_t group_id) const;
    std::vector<std::vector<uint64_t>>              get_ls_waiting_sequence_ids_by_dp() const;
    std::vector<int>                                get_ls_num_ooe() const;
    std::vector<uint64_t>                           get_ls_pool_resource_epochs() const;
    std::optional<uint64_t>                         get_ls_arrival_order(uint64_t seq_id) const;

    void                       latch_ls_fatal(LSFatalCode code) noexcept;
    std::optional<LSFatalCode> ls_fatal_code() const noexcept;
    bool                       ls_decode_core_enabled() const noexcept
    {
        return enable_ls_decode_core_scheduler_;
    }

    // Manual, stop-the-world LS KV scale-down transaction. Planning reserves
    // destination blocks but leaves ACTIVE metadata untouched. The caller must
    // run the returned physical moves on every worker before commit.
    std::shared_ptr<SPStateManager::LSKVConsolidationPlan> plan_ls_kv_scale_down(uint64_t group_id, int source_rank);
    bool mark_ls_kv_scale_down_dispatched(const std::shared_ptr<SPStateManager::LSKVConsolidationPlan>& plan) noexcept;
    bool commit_ls_kv_scale_down(const std::shared_ptr<SPStateManager::LSKVConsolidationPlan>& plan);
    void abort_ls_kv_scale_down(const std::shared_ptr<SPStateManager::LSKVConsolidationPlan>& plan);

    // Test-only failure injection. Zero fails the next isolated pool
    // admission; a positive value allows that many complete pool admission
    // prepares and then fails the following pool. Publication is never entered.
    void set_ls_admission_failure_after_allocations_for_test(int value);
    void set_ls_admission_failure_after_publications_for_test(int value);
    void set_ls_post_admission_component_failure_for_test(int value);

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
    using LSBlockContextOverrides = std::unordered_map<const Sequence*, const BlockContext*>;

    // Internal scheduling logic
    std::vector<std::vector<std::shared_ptr<Sequence>>> _schedule_prefill();
    std::vector<std::vector<std::shared_ptr<Sequence>>> _schedule_decode();
    std::vector<std::vector<std::shared_ptr<Sequence>>> _schedule_ls_decode_admission();
    std::vector<std::vector<std::shared_ptr<Sequence>>>
    _schedule_ls_decode(const std::unordered_set<uint64_t>* eligible_sequence_ids = nullptr);
    std::vector<std::vector<std::shared_ptr<Sequence>>>
    _schedule_ls_combined_pool_step(const std::unordered_set<uint64_t>&                     eligible_sequence_ids,
                                    std::shared_ptr<SPStateManager::LSKVConsolidationPlan>* kv_consolidation_plan);
    std::vector<std::vector<std::shared_ptr<Sequence>>> _schedule_decode_prefill_latency_aware();

    std::optional<std::pair<std::vector<int>, std::vector<std::vector<int>>>>
                     _plan_ls_initial_placement(int                                           dp_idx,
                                                const std::vector<std::shared_ptr<Sequence>>& batch,
                                                const std::vector<int>&                       rank_pool,
                                                const std::vector<std::shared_ptr<Sequence>>& existing_sequences = {},
                                                const std::vector<int>&                       base_allocation    = {},
                                                const std::vector<int>&                       free_block_adjustments = {},
                                                const LSBlockContextOverrides&                context_overrides = {}) const;
    std::vector<int> _ls_unallocated_ranks(int dp_idx, std::optional<uint64_t> excluding_group = std::nullopt) const;
    std::optional<int64_t> _ls_future_kv_peak_tokens(const std::vector<std::shared_ptr<Sequence>>& sequences) const;

    bool _ls_future_kv_fits(int                                           dp_idx,
                            const std::vector<std::shared_ptr<Sequence>>& batch,
                            const std::vector<std::shared_ptr<Sequence>>& existing_sequences,
                            const std::vector<int>&                       future_rank_pool,
                            const std::vector<int>&                       free_block_adjustments = {},
                            const LSBlockContextOverrides&                context_overrides      = {}) const;

    bool _ls_future_kv_fits_empty_system(int                                           dp_idx,
                                         const std::vector<std::shared_ptr<Sequence>>& batch,
                                         const std::vector<int>&                       future_rank_pool) const;
    void _merge_ls_groups(uint64_t lhs_group_id, uint64_t rhs_group_id);

    void    _remove_seq_from_ls_group(uint64_t seq_id);
    void    _reconcile_ls_groups();
    bool    _ls_batch_fits_empty_system(int dp_idx, const std::vector<std::shared_ptr<Sequence>>& batch) const;
    bool    _ls_current_admission_fits(int dp_idx, const std::vector<std::shared_ptr<Sequence>>& batch) const;
    int64_t _ls_admission_need_tokens(const Sequence& sequence) const;
    int64_t _ls_pool_token_capacity(int dp_idx) const;
    bool    _ls_pool_future_kv_fits(int dp_idx, const std::vector<std::shared_ptr<Sequence>>& tentative) const;
    bool    _ls_rank_is_truly_idle(int dp_idx, int sp_idx) const;
    // `force_new_boundary` is for public publications that happen after
    // schedule() returned (manual consolidation commit / explicit preempt).
    // In-step publications keep the default de-duplication semantics.
    void _mark_ls_pool_resource_mutated(int dp_idx, bool force_new_boundary = false) noexcept;
    std::vector<std::shared_ptr<Sequence>> _ls_running_sequences_in_pool(int dp_idx) const;
    bool                                   _ensure_ls_decode_memory_safety();
    bool                                   _ls_offload_one_victim(int dp_idx, const std::string& reason);
    std::shared_ptr<SPStateManager::LSKVConsolidationPlan> _maybe_plan_ls_kv_consolidation();
    bool _ls_kv_consolidation_watermark_ok(const std::shared_ptr<SPStateManager::LSKVConsolidationPlan>& plan) const;
    void _populate_ls_kv_consolidation_telemetry(ScheduleResult& result) const;
    void _populate_ls_scheduler_phase_timing(ScheduleResult& result) const;

    // Decentralized scheduling logic
    ScheduleResult                         _schedule_decentralized();
    std::vector<std::shared_ptr<Sequence>> _schedule_prefill_for_worker(int dp_idx);
    std::vector<std::shared_ptr<Sequence>> _schedule_decode_for_worker(int dp_idx);

    // Routing function for decentralized mode
    int select_dp_worker_for_routing(Sequence& seq);

    // Round-robin counter for DP
    int next_dp_idx();

    // Configuration
    std::string                   engine_id_;
    int                           loop_count_;
    int                           max_num_seqs_;
    int                           max_num_batched_tokens_;
    int                           max_num_recv_seqs_;
    int                           eos_;
    int                           attention_dp_;
    int                           attention_sp_;
    std::string                   mode_;
    double                        reserved_blocks_per_req_;
    int                           segment_size_;
    bool                          enable_dynamic_sp_size_;
    bool                          use_new_decode_dynamic_sp_scheduler_;
    std::string                   dynamic_sp_size_strategy_;
    int                           dynamic_sp_long_request_threshold_;
    int                           dynamic_sp_long_request_size_;
    bool                          enable_non_uniform_split_;
    bool                          sp_debug_;
    bool                          enable_ls_decode_core_scheduler_;
    bool                          ls_scheduler_phase_timing_enabled_;
    int                           ls_decode_initial_kv_dop_;
    int                           ls_decode_batch_per_master_;
    bool                          ls_decode_enable_memory_scale_up_;
    bool                          ls_decode_enable_future_kv_admission_;
    int                           ls_max_num_ooe_;
    int                           ls_running_max_req_size_;
    int                           ls_admission_max_tokens_per_pool_;
    int                           ls_min_comp_bound_decoding_batch_size_;
    std::string                   ls_kv_consolidation_mode_;
    double                        ls_kv_consolidation_candidate_util_;
    double                        ls_kv_consolidation_target_high_watermark_;
    int                           ls_kv_consolidation_stable_steps_;
    int                           ls_kv_consolidation_cooldown_steps_;
    int                           ls_kv_consolidation_check_interval_steps_;
    int                           ls_kv_consolidation_max_source_blocks_per_event_;
    std::vector<std::vector<int>> ls_empty_system_free_blocks_per_rank_;

    std::string sp_master_selector_;

    SchedulerMode scheduler_mode_ = SchedulerMode::CENTRALIZED;

    int dp_rr_counter_ = 0;

    std::unique_ptr<ThreadPool> thread_pool_;

    uint64_t                                        next_ls_group_id_        = 0;
    uint64_t                                        next_ls_batch_id_        = 0;
    uint64_t                                        next_ls_admission_order_ = 0;
    uint64_t                                        ls_schedule_step_        = 0;
    std::unordered_map<uint64_t, DecodeGroupState>  ls_groups_;
    std::vector<std::vector<uint64_t>>              ls_group_ids_by_dp_;
    std::unordered_map<uint64_t, uint64_t>          ls_seq_to_group_;
    std::vector<InitialBatchPlacement>              ls_step_initial_records_;
    std::vector<SPStateManager::LSDecodeMasterPlan> ls_step_group_plans_;
    std::vector<uint64_t>                           ls_step_group_plan_ids_;
    std::vector<std::vector<uint64_t>>              ls_step_group_plan_sequence_ids_;
    std::vector<std::vector<int>>                   ls_step_reused_passive_masters_;
    std::vector<uint64_t>                           ls_step_preempted_sequence_ids_;
    std::vector<std::string>                        ls_step_preemption_reasons_;
    uint64_t                                        ls_step_atomic_no_fit_count_                      = 0;
    uint64_t                                        ls_step_atomic_merge_count_                       = 0;
    uint64_t                                        ls_step_atomic_rollback_count_                    = 0;
    int                                             ls_admission_failure_after_allocations_for_test_  = -1;
    int                                             ls_admission_failure_after_publications_for_test_ = -1;
    int                                             ls_post_admission_component_failure_for_test_     = -1;
    double                                          ls_step_planning_latency_ms_                      = 0.0;
    struct LSSchedulerPhaseTimingState {
        double   total_ms                     = 0.0;
        double   snapshot_copy_ms             = 0.0;
        double   mandatory_safety_ms          = 0.0;
        double   admission_ms                 = 0.0;
        double   kv_consolidation_ms          = 0.0;
        double   decode_plan_prepare_ms       = 0.0;
        double   publication_ms               = 0.0;
        double   unattributed_ms              = 0.0;
        double   rollback_shadow_copy_ms      = 0.0;
        double   admission_scan_ms            = 0.0;
        double   future_kv_pool_ms            = 0.0;
        double   future_kv_empty_system_ms    = 0.0;
        double   empty_system_fit_ms          = 0.0;
        double   initial_placement_ms         = 0.0;
        double   admission_plan_ms            = 0.0;
        uint64_t prepare_attempts             = 0;
        uint64_t snapshot_copies              = 0;
        uint64_t rollback_shadow_copies       = 0;
        uint64_t admission_pool_attempts      = 0;
        uint64_t waiting_candidates_scanned   = 0;
        uint64_t future_kv_pool_calls         = 0;
        uint64_t future_kv_empty_system_calls = 0;
        uint64_t empty_system_fit_calls       = 0;
        uint64_t initial_placement_calls      = 0;
        uint64_t admission_plan_calls         = 0;
        uint64_t decode_pool_plan_attempts    = 0;
    };
    mutable LSSchedulerPhaseTimingState                    ls_step_phase_timing_;
    uint64_t                                               next_ls_kv_transaction_id_ = 1;
    std::shared_ptr<SPStateManager::LSKVConsolidationPlan> active_ls_kv_transaction_;
    bool                                                   ls_step_kv_candidate_       = false;
    int64_t                                                ls_step_kv_group_id_        = -1;
    int                                                    ls_step_kv_source_rank_     = -1;
    int                                                    ls_step_kv_target_dop_      = -1;
    uint64_t                                               ls_step_kv_stable_steps_    = 0;
    double                                                 ls_step_kv_group_util_      = 0.0;
    std::string                                            ls_step_kv_decision_reason_ = "off";

    // Canonical request-level dispatch state for the Decode-only baseline.
    std::vector<std::list<std::shared_ptr<Sequence>>> ls_waiting_by_dp_;
    std::vector<int>                                  ls_num_ooe_;
    uint64_t                                          next_ls_arrival_order_ = 0;
    int                                               next_ls_dp_rr_         = 0;
    std::unordered_map<uint64_t, uint64_t>            ls_arrival_order_by_seq_id_;
    std::unordered_set<uint64_t>                      seen_ls_seq_ids_;
    std::vector<LSAdmissionRecord>                    ls_step_admission_records_;
    std::vector<std::vector<uint64_t>>                ls_step_real_decode_ids_by_dp_;
    int                                               ls_step_execution_loop_count_ = 1;
    std::vector<uint64_t>                             ls_pool_resource_epoch_;
    std::vector<uint64_t>                             ls_step_pool_resource_epoch_before_;
    std::vector<bool>                                 ls_step_pool_resource_mutated_;
    bool                                              ls_step_offload_committed_ = false;
    // Set immediately before the first no-throw resource/scheduler publication
    // in a schedule() call. The function-try-boundary maps any later
    // unexpected exception (including ScheduleResult/telemetry allocation) to
    // POST_PUBLICATION_INVARIANT instead of misclassifying it as a recoverable
    // Decode prepare failure.
    bool                       ls_step_publication_started_ = false;
    std::optional<LSFatalCode> ls_fatal_;
};

}  // namespace nanodeploy
