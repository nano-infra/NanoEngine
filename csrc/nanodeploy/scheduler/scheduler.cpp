#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unordered_set>

#include "nanodeploy/metrics/sequence_metric.h"
#include "nanodeploy/sequence/sequence.h"

#include "scheduler_utils.h"

#include "scheduler.h"

namespace nanodeploy {

namespace {

bool has_remote_committed_kv(const Sequence& seq, int attention_sp)
{
    const auto& ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
    for (int sp_idx = 0; sp_idx < attention_sp; ++sp_idx) {
        if (sp_idx != ctx.master_sp_idx_ && seq.committed_context_len(BlockContextSlot::ACTIVE, sp_idx) > 0) {
            return true;
        }
    }
    return false;
}

int committed_kv_rank_count(const Sequence& seq, int attention_sp)
{
    int count = 0;
    for (int sp_idx = 0; sp_idx < attention_sp; ++sp_idx) {
        if (seq.committed_context_len(BlockContextSlot::ACTIVE, sp_idx) > 0) {
            count++;
        }
    }
    return count;
}

}  // namespace

Scheduler::Scheduler(const std::string& engine_id,
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
                     bool               enable_ls_decode_core_scheduler,
                     int                ls_decode_initial_kv_dop,
                     int                ls_decode_batch_per_master,
                     bool               ls_decode_enable_memory_scale_up,
                     const std::string& scheduler_mode,
                     const std::string& ls_kv_consolidation_mode,
                     double             ls_kv_consolidation_candidate_util,
                     double             ls_kv_consolidation_target_high_watermark,
                     int                ls_kv_consolidation_stable_steps,
                     int                ls_kv_consolidation_cooldown_steps,
                     int                ls_kv_consolidation_check_interval_steps,
                     int                ls_kv_consolidation_max_source_blocks_per_event):
    engine_id_(engine_id),
    loop_count_(loop_count),
    max_num_seqs_(max_num_seqs),
    max_num_batched_tokens_(max_num_batched_tokens),
    max_num_recv_seqs_(max_num_recv_seqs),
    eos_(eos),
    attention_dp_(attention_dp),
    attention_sp_(attention_sp),
    mode_(mode),
    reserved_blocks_per_req_(reserved_blocks_per_req),
    segment_size_(segment_size),
    enable_dynamic_sp_size_(enable_dynamic_sp_size),
    use_new_decode_dynamic_sp_scheduler_(use_new_decode_dynamic_sp_scheduler),
    dynamic_sp_size_strategy_(dynamic_sp_size_strategy),
    dynamic_sp_long_request_threshold_(dynamic_sp_long_request_threshold),
    dynamic_sp_long_request_size_(dynamic_sp_long_request_size),
    enable_non_uniform_split_(enable_non_uniform_split),
    sp_debug_(sp_debug),
    enable_ls_decode_core_scheduler_(enable_ls_decode_core_scheduler),
    ls_decode_initial_kv_dop_(ls_decode_initial_kv_dop),
    ls_decode_batch_per_master_(ls_decode_batch_per_master),
    ls_decode_enable_memory_scale_up_(ls_decode_enable_memory_scale_up),
    ls_kv_consolidation_mode_(ls_kv_consolidation_mode),
    ls_kv_consolidation_candidate_util_(ls_kv_consolidation_candidate_util),
    ls_kv_consolidation_target_high_watermark_(ls_kv_consolidation_target_high_watermark),
    ls_kv_consolidation_stable_steps_(ls_kv_consolidation_stable_steps),
    ls_kv_consolidation_cooldown_steps_(ls_kv_consolidation_cooldown_steps),
    ls_kv_consolidation_check_interval_steps_(ls_kv_consolidation_check_interval_steps),
    ls_kv_consolidation_max_source_blocks_per_event_(ls_kv_consolidation_max_source_blocks_per_event),
    sp_master_selector_(sp_master_selector)
{
    if (ls_kv_consolidation_mode_ != "off" && ls_kv_consolidation_mode_ != "shadow"
        && ls_kv_consolidation_mode_ != "execute") {
        throw std::invalid_argument("ls_kv_consolidation_mode must be one of: off, shadow, execute");
    }
    if (ls_kv_consolidation_mode_ != "off" && !enable_ls_decode_core_scheduler_) {
        throw std::invalid_argument("automatic KV consolidation requires the LS Decode core scheduler");
    }
    if (!(ls_kv_consolidation_candidate_util_ > 0.0 && ls_kv_consolidation_candidate_util_ <= 1.0)) {
        throw std::invalid_argument("ls_kv_consolidation_candidate_util must be in (0, 1]");
    }
    if (!(ls_kv_consolidation_target_high_watermark_ > 0.0 && ls_kv_consolidation_target_high_watermark_ <= 1.0)) {
        throw std::invalid_argument("ls_kv_consolidation_target_high_watermark must be in (0, 1]");
    }
    if (ls_kv_consolidation_stable_steps_ <= 0 || ls_kv_consolidation_cooldown_steps_ < 0
        || ls_kv_consolidation_check_interval_steps_ <= 0 || ls_kv_consolidation_max_source_blocks_per_event_ < 0) {
        throw std::invalid_argument("invalid automatic KV consolidation step threshold");
    }
    if (ls_kv_consolidation_mode_ == "execute" && ls_kv_consolidation_max_source_blocks_per_event_ == 0) {
        throw std::invalid_argument("automatic KV consolidation execute requires a positive source-block budget");
    }
    Sequence::block_size = kvcache_block_size;
    // Initialize worker states
    worker_state.reserve(attention_dp_);
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        auto sp_manager = std::make_shared<SPStateManager>(engine_id_,
                                                           attention_sp_,
                                                           num_kvcache_blocks,
                                                           kvcache_block_size,
                                                           max_num_seqs_,
                                                           max_num_batched_tokens_,
                                                           max_num_recv_seqs_,
                                                           reserved_blocks_per_req_,
                                                           segment_size_,
                                                           enable_dynamic_sp_size_,
                                                           dynamic_sp_size_strategy_,
                                                           dynamic_sp_long_request_threshold_,
                                                           dynamic_sp_long_request_size_,
                                                           enable_dynamic_sp_bucket_policy,
                                                           dynamic_sp_bucket_policy,
                                                           attention_cost_a,
                                                           attention_cost_b,
                                                           q_cost_a,
                                                           q_cost_b,
                                                           res_cost_a,
                                                           res_cost_b,
                                                           lse_cost_a,
                                                           lse_cost_b,
                                                           q_bytes_per_edge,
                                                           res_bytes_per_edge,
                                                           lse_bytes_per_edge,
                                                           enable_non_uniform_split,
                                                           sp_master_selector,
                                                           sp_debug_,
                                                           fixed_sp_size);

        sp_manager->set_dp_idx(dp_idx);
        worker_state.push_back(sp_manager);
    }
    ls_empty_system_free_blocks_per_rank_.resize(attention_sp_);
    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        ls_empty_system_free_blocks_per_rank_[sp_idx] =
            worker_state.front()->block_manager.at(sp_idx)->num_free_blocks();
    }
    // Set scheduler mode
    if (scheduler_mode == "decentralized") {
        scheduler_mode_ = SchedulerMode::DECENTRALIZED;
    }
    else {
        scheduler_mode_ = SchedulerMode::CENTRALIZED;
    }

    std::cerr << "[Scheduler] Initialized with segment_size=" << segment_size_ << ", fixed_sp_size=" << fixed_sp_size
              << ", use_new_decode_dynamic_sp_scheduler=" << use_new_decode_dynamic_sp_scheduler_
              << ", dynamic_sp_size_strategy=" << dynamic_sp_size_strategy_
              << ", dynamic_sp_long_request_threshold=" << dynamic_sp_long_request_threshold_
              << ", dynamic_sp_long_request_size=" << dynamic_sp_long_request_size_ << ", scheduler_mode="
              << (scheduler_mode_ == SchedulerMode::DECENTRALIZED ? "decentralized" : "centralized") << std::endl;
    thread_pool_ = std::make_unique<ThreadPool>(attention_dp_);
    ls_group_ids_by_dp_.resize(attention_dp_);
}

void Scheduler::add(std::shared_ptr<Sequence> seq)
{
    if (active_ls_kv_transaction_) {
        throw std::runtime_error("cannot add a sequence while an LS KV scale-down transaction is reserved");
    }
    seq->active(engine_id_, attention_sp_, attention_dp_);

    if (seq->metric) {
        seq->metric->record_arrival();
    }

    if (scheduler_mode_ == SchedulerMode::DECENTRALIZED) {
        // Decentralized mode: immediately route to selected DP worker
        int   selected_dp_idx = select_dp_worker_for_routing(*seq);
        auto& target_queue    = (mode_ == "decode") ? worker_state[selected_dp_idx]->waiting_migration :
                                                      worker_state[selected_dp_idx]->waiting;
        target_queue.push_back(seq);

        // Set dp_idx (though resources haven't been allocated yet)
        seq->block_ctx(BlockContextSlot::ACTIVE).dp_idx_ = selected_dp_idx;

        if (mode_ == "decode" && seq->metric) {
            seq->metric->record_decode_arrival();
        }
    }
    else {
        // Centralized mode: add to global queue (original logic)
        if (mode_ == "decode") {
            waiting_migration.push_back(seq);
            if (seq->metric) {
                seq->metric->record_decode_arrival();
            }
        }
        else {
            waiting.push_back(seq);
        }
    }
}

bool Scheduler::is_finished() const
{
    if (scheduler_mode_ == SchedulerMode::DECENTRALIZED) {
        // Check all workers' queues and running state
        for (const auto& ws : worker_state) {
            if (!ws->is_waiting_empty() || !ws->is_empty()) {
                return false;
            }
        }
        return true;
    }
    else {
        // Centralized mode: original logic
        const auto& wait_queue = (mode_ != "decode") ? waiting : waiting_migration;
        if (!wait_queue.empty())
            return false;
        if (enable_ls_decode_core_scheduler_ && !ls_pending_decode_batches_.empty()) {
            return false;
        }
        if (enable_ls_decode_core_scheduler_ && !to_be_migrated.empty()) {
            return false;
        }

        for (const auto& ws : worker_state) {
            if (!ws->is_empty())
                return false;
        }
        return true;
    }
}

int Scheduler::get_total_waiting_size() const
{
    if (scheduler_mode_ == SchedulerMode::DECENTRALIZED) {
        int total = 0;
        for (const auto& ws : worker_state) {
            total += static_cast<int>(ws->waiting.size());
        }
        return total;
    }
    else {
        return static_cast<int>(waiting.size());
    }
}

int Scheduler::get_total_waiting_migration_size() const
{
    if (scheduler_mode_ == SchedulerMode::DECENTRALIZED) {
        int total = 0;
        for (const auto& ws : worker_state) {
            total += static_cast<int>(ws->waiting_migration.size());
        }
        return total;
    }
    else {
        int total = static_cast<int>(waiting_migration.size());
        if (enable_ls_decode_core_scheduler_) {
            for (const auto& batch : ls_pending_decode_batches_) {
                total += static_cast<int>(batch.sequences.size());
            }
        }
        return total;
    }
}

std::vector<uint64_t> Scheduler::get_ls_pending_batch_ids() const
{
    std::vector<uint64_t> result;
    result.reserve(ls_pending_decode_batches_.size());
    for (const auto& batch : ls_pending_decode_batches_) {
        result.push_back(batch.batch_id);
    }
    return result;
}

std::vector<std::vector<uint64_t>> Scheduler::get_ls_pending_batch_sequence_ids() const
{
    std::vector<std::vector<uint64_t>> result;
    result.reserve(ls_pending_decode_batches_.size());
    for (const auto& batch : ls_pending_decode_batches_) {
        std::vector<uint64_t> sequence_ids;
        sequence_ids.reserve(batch.sequences.size());
        for (const auto& seq : batch.sequences) {
            sequence_ids.push_back(seq->seq_id);
        }
        result.push_back(std::move(sequence_ids));
    }
    return result;
}

std::vector<uint32_t> Scheduler::get_ls_pending_batch_attempts() const
{
    std::vector<uint32_t> result;
    result.reserve(ls_pending_decode_batches_.size());
    for (const auto& batch : ls_pending_decode_batches_) {
        result.push_back(batch.admission_attempts);
    }
    return result;
}

std::vector<bool> Scheduler::get_ls_pending_batch_is_recovery() const
{
    std::vector<bool> result;
    result.reserve(ls_pending_decode_batches_.size());
    for (const auto& batch : ls_pending_decode_batches_) {
        result.push_back(batch.is_recovery_batch);
    }
    return result;
}

std::vector<std::optional<uint64_t>> Scheduler::get_ls_pending_batch_parent_batch_ids() const
{
    std::vector<std::optional<uint64_t>> result;
    result.reserve(ls_pending_decode_batches_.size());
    for (const auto& batch : ls_pending_decode_batches_) {
        result.push_back(batch.parent_batch_id);
    }
    return result;
}

std::vector<uint64_t> Scheduler::get_ls_group_ids() const
{
    std::vector<uint64_t> result;
    result.reserve(ls_groups_.size());
    for (const auto& [group_id, group] : ls_groups_) {
        (void)group;
        result.push_back(group_id);
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<std::vector<uint64_t>> Scheduler::get_ls_group_sequence_ids() const
{
    std::vector<std::vector<uint64_t>> result;
    for (uint64_t group_id : get_ls_group_ids()) {
        std::vector<uint64_t> sequence_ids;
        for (const auto& seq : ls_groups_.at(group_id).sequences) {
            if (seq && seq->status == SequenceStatus::RUNNING) {
                sequence_ids.push_back(seq->seq_id);
            }
        }
        result.push_back(std::move(sequence_ids));
    }
    return result;
}

std::vector<std::vector<uint64_t>> Scheduler::get_ls_group_initial_batch_ids() const
{
    std::vector<std::vector<uint64_t>> result;
    for (uint64_t group_id : get_ls_group_ids()) {
        std::vector<uint64_t> batch_ids;
        for (const auto& record : ls_groups_.at(group_id).initial_batch_placements) {
            batch_ids.push_back(record.batch_id);
        }
        result.push_back(std::move(batch_ids));
    }
    return result;
}

std::vector<std::vector<uint64_t>> Scheduler::get_ls_group_initial_admission_orders() const
{
    std::vector<std::vector<uint64_t>> result;
    for (uint64_t group_id : get_ls_group_ids()) {
        std::vector<uint64_t> admission_orders;
        for (const auto& record : ls_groups_.at(group_id).initial_batch_placements) {
            admission_orders.push_back(record.admission_order);
        }
        result.push_back(std::move(admission_orders));
    }
    return result;
}

std::vector<std::vector<std::vector<uint64_t>>> Scheduler::get_ls_group_initial_sequence_ids() const
{
    std::vector<std::vector<std::vector<uint64_t>>> result;
    for (uint64_t group_id : get_ls_group_ids()) {
        std::vector<std::vector<uint64_t>> records;
        for (const auto& record : ls_groups_.at(group_id).initial_batch_placements) {
            records.push_back(record.sequence_ids);
        }
        result.push_back(std::move(records));
    }
    return result;
}

std::vector<std::pair<uint64_t, uint64_t>> Scheduler::get_ls_active_batch_owners() const
{
    std::vector<std::pair<uint64_t, uint64_t>> result(ls_seq_to_batch_.begin(), ls_seq_to_batch_.end());
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<int> Scheduler::get_ls_group_allocated_ranks(uint64_t group_id) const
{
    auto group = ls_groups_.find(group_id);
    if (group == ls_groups_.end()) {
        throw std::runtime_error("unknown LS Decode group");
    }
    return group->second.allocated_attention_ranks;
}

std::shared_ptr<SPStateManager::LSKVConsolidationPlan> Scheduler::plan_ls_kv_scale_down(uint64_t group_id,
                                                                                        int      source_rank)
{
    auto rejected = [&](const std::string& reason) {
        auto plan            = std::make_shared<SPStateManager::LSKVConsolidationPlan>();
        plan->transaction_id = next_ls_kv_transaction_id_++;
        plan->group_id       = group_id;
        plan->source_rank    = source_rank;
        plan->failure_reason = reason;
        return plan;
    };
    if (!enable_ls_decode_core_scheduler_) {
        return rejected("LS Decode core scheduler is disabled");
    }
    if (active_ls_kv_transaction_) {
        return rejected("another LS KV scale-down transaction is already reserved");
    }
    auto group_it = ls_groups_.find(group_id);
    if (group_it == ls_groups_.end()) {
        return rejected("unknown LS Decode group");
    }

    auto& group = group_it->second;
    if (std::find(group.allocated_attention_ranks.begin(), group.allocated_attention_ranks.end(), source_rank)
        == group.allocated_attention_ranks.end()) {
        return rejected("source rank is not allocated to the LS Decode group");
    }
    std::vector<int> retained;
    for (int rank : group.allocated_attention_ranks) {
        if (rank != source_rank) {
            retained.push_back(rank);
        }
    }

    auto plan = worker_state.at(group.dp_idx)
                    ->plan_kv_consolidation(
                        next_ls_kv_transaction_id_++, group_id, group.dp_idx, group.sequences, source_rank, retained);
    if (plan->success) {
        active_ls_kv_transaction_ = plan;
    }
    return plan;
}

bool Scheduler::commit_ls_kv_scale_down(const std::shared_ptr<SPStateManager::LSKVConsolidationPlan>& plan)
{
    if (!plan || !active_ls_kv_transaction_ || active_ls_kv_transaction_.get() != plan.get()) {
        return false;
    }
    auto group_it = ls_groups_.find(plan->group_id);
    if (group_it == ls_groups_.end() || group_it->second.dp_idx != plan->dp_idx) {
        return false;
    }
    auto& group = group_it->second;

    std::vector<uint64_t> current_sequence_ids;
    current_sequence_ids.reserve(group.sequences.size());
    for (const auto& sequence : group.sequences) {
        if (sequence) {
            current_sequence_ids.push_back(sequence->seq_id);
        }
    }
    auto planned_sequence_ids = plan->group_sequence_ids;
    std::sort(current_sequence_ids.begin(), current_sequence_ids.end());
    std::sort(planned_sequence_ids.begin(), planned_sequence_ids.end());
    if (current_sequence_ids != planned_sequence_ids) {
        return false;
    }

    std::vector<int> expected_allocation = plan->retained_ranks;
    expected_allocation.push_back(plan->source_rank);
    auto current_allocation = group.allocated_attention_ranks;
    std::sort(expected_allocation.begin(), expected_allocation.end());
    std::sort(current_allocation.begin(), current_allocation.end());
    if (expected_allocation != current_allocation) {
        return false;
    }
    if (!worker_state.at(plan->dp_idx)->commit_kv_consolidation(plan)) {
        return false;
    }

    group.allocated_attention_ranks = plan->retained_ranks;
    group.last_iteration_masters.erase(
        std::remove(group.last_iteration_masters.begin(), group.last_iteration_masters.end(), plan->source_rank),
        group.last_iteration_masters.end());
    group.kv_candidate_target_dop   = -1;
    group.kv_candidate_stable_steps = 0;
    group.last_consolidation_step   = ls_schedule_step_;
    active_ls_kv_transaction_.reset();
    return true;
}

void Scheduler::abort_ls_kv_scale_down(const std::shared_ptr<SPStateManager::LSKVConsolidationPlan>& plan)
{
    if (!plan || !active_ls_kv_transaction_ || active_ls_kv_transaction_.get() != plan.get()) {
        return;
    }
    worker_state.at(plan->dp_idx)->abort_kv_consolidation(plan);
    active_ls_kv_transaction_.reset();
}

bool Scheduler::_ls_kv_consolidation_watermark_ok(
    const std::shared_ptr<SPStateManager::LSKVConsolidationPlan>& plan) const
{
    if (!plan || !plan->success || plan->dp_idx < 0 || plan->dp_idx >= attention_dp_) {
        return false;
    }
    for (int rank : plan->retained_ranks) {
        const auto& manager = worker_state.at(plan->dp_idx)->block_manager.at(rank);
        int         total   = static_cast<int>(manager->blocks().size());
        if (total <= 0) {
            return false;
        }
        double utilization = static_cast<double>(total - manager->num_free_blocks()) / static_cast<double>(total);
        if (utilization > ls_kv_consolidation_target_high_watermark_) {
            return false;
        }
    }
    return true;
}

void Scheduler::_populate_ls_kv_consolidation_telemetry(ScheduleResult& result) const
{
    result.ls_kv_consolidation_candidate       = ls_step_kv_candidate_;
    result.ls_kv_consolidation_group_id        = ls_step_kv_group_id_;
    result.ls_kv_consolidation_source_rank     = ls_step_kv_source_rank_;
    result.ls_kv_consolidation_target_dop      = ls_step_kv_target_dop_;
    result.ls_kv_consolidation_stable_steps    = ls_step_kv_stable_steps_;
    result.ls_kv_consolidation_group_util      = ls_step_kv_group_util_;
    result.ls_kv_consolidation_decision_reason = ls_step_kv_decision_reason_;
}

std::shared_ptr<SPStateManager::LSKVConsolidationPlan> Scheduler::_maybe_plan_ls_kv_consolidation()
{
    if (ls_kv_consolidation_mode_ == "off") {
        ls_step_kv_decision_reason_ = "off";
        return nullptr;
    }
    if (!ls_pending_decode_batches_.empty()) {
        for (auto& [group_id, group] : ls_groups_) {
            (void)group_id;
            group.kv_candidate_target_dop   = -1;
            group.kv_candidate_stable_steps = 0;
        }
        ls_step_kv_decision_reason_ = "pending_admission_not_proven";
        return nullptr;
    }

    _reconcile_ls_groups();

    struct Candidate {
        uint64_t         group_id     = 0;
        int              dp_idx       = -1;
        int              target_dop   = -1;
        double           utilization  = 0.0;
        uint64_t         stable_steps = 0;
        std::vector<int> source_ranks;
    };
    std::vector<Candidate> candidates;

    auto group_ids = get_ls_group_ids();
    for (uint64_t group_id : group_ids) {
        auto& group       = ls_groups_.at(group_id);
        auto  used_blocks = worker_state.at(group.dp_idx)->group_used_kv_blocks(group.sequences);
        auto  used_tokens = worker_state.at(group.dp_idx)->group_used_kv_tokens(group.sequences);

        std::vector<int> participants;
        int64_t          used_block_sum   = 0;
        int64_t          usable_block_sum = 0;
        for (int rank : group.allocated_attention_ranks) {
            if (rank >= 0 && rank < attention_sp_ && used_blocks.at(rank) > 0) {
                participants.push_back(rank);
                used_block_sum += used_blocks[rank];
                usable_block_sum += ls_empty_system_free_blocks_per_rank_.at(rank);
            }
        }

        int real_batch =
            static_cast<int>(std::count_if(group.sequences.begin(), group.sequences.end(), [](const auto& sequence) {
                return sequence && sequence->status == SequenceStatus::RUNNING;
            }));
        int    target_dop  = std::max(1, (real_batch + ls_decode_batch_per_master_ - 1) / ls_decode_batch_per_master_);
        double utilization = usable_block_sum > 0 ? static_cast<double>(used_block_sum) / usable_block_sum : 1.0;

        bool has_unreclaimed_empty_rank = participants.size() != group.allocated_attention_ranks.size();
        bool below_candidate_threshold  = utilization < ls_kv_consolidation_candidate_util_;
        bool dop_can_shrink             = static_cast<int>(participants.size()) > target_dop;
        if (real_batch == 0 || has_unreclaimed_empty_rank || !below_candidate_threshold || !dop_can_shrink) {
            group.kv_candidate_target_dop   = -1;
            group.kv_candidate_stable_steps = 0;
            continue;
        }

        std::vector<int> sources;
        for (int rank : participants) {
            bool active_or_pending_master = std::any_of(
                group.sequences.begin(), group.sequences.end(), [&](const std::shared_ptr<Sequence>& sequence) {
                    if (!sequence || sequence->status != SequenceStatus::RUNNING) {
                        return false;
                    }
                    const auto& context = sequence->block_ctx(BlockContextSlot::ACTIVE);
                    return context.master_sp_idx_ == rank
                           || (context.pending_token_present_ && context.pending_token_target_sp_ == rank);
                });
            if (!active_or_pending_master) {
                sources.push_back(rank);
            }
        }
        std::sort(sources.begin(), sources.end(), [&](int lhs, int rhs) {
            return std::tie(used_blocks[lhs], used_tokens[lhs], lhs)
                   < std::tie(used_blocks[rhs], used_tokens[rhs], rhs);
        });
        if (sources.empty()) {
            group.kv_candidate_target_dop   = -1;
            group.kv_candidate_stable_steps = 0;
            continue;
        }
        if (group.kv_candidate_target_dop == target_dop) {
            group.kv_candidate_stable_steps++;
        }
        else {
            group.kv_candidate_target_dop   = target_dop;
            group.kv_candidate_stable_steps = 1;
        }
        candidates.push_back(
            {group_id, group.dp_idx, target_dop, utilization, group.kv_candidate_stable_steps, std::move(sources)});
    }

    if (candidates.empty()) {
        ls_step_kv_decision_reason_ = "no_candidate";
        return nullptr;
    }

    bool check_due = ls_schedule_step_ % static_cast<uint64_t>(ls_kv_consolidation_check_interval_steps_) == 0;
    for (const auto& candidate : candidates) {
        const auto& group        = ls_groups_.at(candidate.group_id);
        ls_step_kv_candidate_    = true;
        ls_step_kv_group_id_     = static_cast<int64_t>(candidate.group_id);
        ls_step_kv_source_rank_  = candidate.source_ranks.front();
        ls_step_kv_target_dop_   = candidate.target_dop;
        ls_step_kv_stable_steps_ = candidate.stable_steps;
        ls_step_kv_group_util_   = candidate.utilization;

        if (candidate.stable_steps < static_cast<uint64_t>(ls_kv_consolidation_stable_steps_)) {
            ls_step_kv_decision_reason_ = "stable_window";
            continue;
        }
        uint64_t cooldown = static_cast<uint64_t>(ls_kv_consolidation_cooldown_steps_);
        if (ls_schedule_step_ < group.last_scale_up_step + cooldown
            || ls_schedule_step_ < group.last_consolidation_step + cooldown) {
            ls_step_kv_decision_reason_ = "cooldown";
            continue;
        }
        if (!check_due) {
            ls_step_kv_decision_reason_ = "check_interval";
            continue;
        }
        if (ls_kv_consolidation_mode_ == "shadow") {
            ls_step_kv_decision_reason_ = "shadow_candidate";
            return nullptr;
        }

        for (int source_rank : candidate.source_ranks) {
            auto plan = plan_ls_kv_scale_down(candidate.group_id, source_rank);
            if (!plan->success) {
                ls_step_kv_decision_reason_ = plan->failure_reason;
                continue;
            }
            int source_blocks = 0;
            for (const auto& stage : plan->sequence_stages) {
                source_blocks += static_cast<int>(stage.source_blocks.size());
            }
            if (source_blocks > ls_kv_consolidation_max_source_blocks_per_event_) {
                abort_ls_kv_scale_down(plan);
                ls_step_kv_decision_reason_ = "source_block_budget";
                continue;
            }
            if (!_ls_kv_consolidation_watermark_ok(plan)) {
                abort_ls_kv_scale_down(plan);
                ls_step_kv_decision_reason_ = "target_high_watermark";
                continue;
            }
            ls_step_kv_source_rank_     = source_rank;
            ls_step_kv_decision_reason_ = "execute";
            return plan;
        }
    }
    return nullptr;
}

void Scheduler::set_ls_admission_failure_after_allocations_for_test(int value)
{
    ls_admission_failure_after_allocations_for_test_ = value;
}

void Scheduler::set_ls_admission_failure_after_publications_for_test(int value)
{
    ls_admission_failure_after_publications_for_test_ = value;
}

std::deque<std::shared_ptr<Sequence>>& Scheduler::running(int dp_idx)
{
    return worker_state[dp_idx]->running;
}

const std::deque<std::shared_ptr<Sequence>>& Scheduler::running(int dp_idx) const
{
    return worker_state[dp_idx]->running;
}

std::unordered_map<int, std::shared_ptr<BlockManager>>& Scheduler::block_manager(int dp_idx)
{
    return worker_state[dp_idx]->block_manager;
}

const std::unordered_map<int, std::shared_ptr<BlockManager>>& Scheduler::block_manager(int dp_idx) const
{
    return worker_state[dp_idx]->block_manager;
}

int Scheduler::next_dp_idx()
{
    int idx        = dp_rr_counter_;
    dp_rr_counter_ = (dp_rr_counter_ + 1) % attention_dp_;
    return idx;
}

int Scheduler::select_dp_worker_for_routing(Sequence& seq)
{
    if (routing_strategy == RoutingStrategy::RoundRobin) {
        return next_dp_idx();
    }
    else if (routing_strategy == RoutingStrategy::LeastBatch) {
        int best_idx = 0;
        int min_load = std::numeric_limits<int>::max();
        for (int i = 0; i < attention_dp_; ++i) {
            int load = worker_state[i]->get_total_load();
            if (load < min_load) {
                min_load = load;
                best_idx = i;
            }
        }
        return best_idx;
    }
    else if (routing_strategy == RoutingStrategy::LeastCache) {
        int best_idx = 0;
        int max_free = -1;
        for (int i = 0; i < attention_dp_; ++i) {
            // Get free blocks from the first SP rank (or aggregate if needed)
            int free_blocks = 0;
            if (!worker_state[i]->block_manager.empty()) {
                // Use the first available block manager to get free blocks
                auto it = worker_state[i]->block_manager.begin();
                if (it != worker_state[i]->block_manager.end()) {
                    free_blocks = it->second->num_free_blocks();
                }
            }
            if (free_blocks > max_free) {
                max_free = free_blocks;
                best_idx = i;
            }
        }
        return best_idx;
    }
    else if (routing_strategy == RoutingStrategy::VLLMLoadBalance) {
        // vLLM-style load balancing: score = waiting * 4 + running
        // This matches vLLM's load balancing algorithm in DPLBAsyncMPClient
        int best_idx  = 0;
        int min_score = std::numeric_limits<int>::max();

        for (int i = 0; i < attention_dp_; ++i) {
            int waiting = worker_state[i]->get_waiting_queue_size();
            int running = worker_state[i]->num_running_seqs();

            // vLLM formula: score = waiting * 4 + running
            // waiting has 4x weight compared to running
            int score = waiting * 4 + running;

            if (score < min_score) {
                min_score = score;
                best_idx  = i;
            }
        }
        return best_idx;
    }
    return 0;
}

ScheduleResult Scheduler::schedule()
{
    if (active_ls_kv_transaction_) {
        throw std::runtime_error("cannot schedule Decode while an LS KV scale-down transaction is reserved");
    }
    if (enable_ls_decode_core_scheduler_) {
        ls_schedule_step_++;
    }
    ls_step_initial_records_.clear();
    ls_step_sealed_batches_.clear();
    ls_step_group_plans_.clear();
    ls_step_group_plan_ids_.clear();
    ls_step_reused_passive_masters_.clear();
    ls_step_preempted_sequence_ids_.clear();
    ls_step_preemption_reasons_.clear();
    ls_step_atomic_no_fit_count_   = 0;
    ls_step_atomic_merge_count_    = 0;
    ls_step_atomic_rollback_count_ = 0;
    ls_step_planning_latency_ms_   = 0.0;
    ls_step_kv_candidate_          = false;
    ls_step_kv_group_id_           = -1;
    ls_step_kv_source_rank_        = -1;
    ls_step_kv_target_dop_         = -1;
    ls_step_kv_stable_steps_       = 0;
    ls_step_kv_group_util_         = 0.0;
    ls_step_kv_decision_reason_    = ls_kv_consolidation_mode_ == "off" ? "off" : "no_candidate";
    std::vector<std::vector<std::shared_ptr<Sequence>>> dp_seqs;
    bool                                                has_prefill = false;

    if (scheduler_mode_ == SchedulerMode::DECENTRALIZED) {
        return _schedule_decentralized();
    }
    else {
        // Centralized mode: original logic
        // Try prefill first
        dp_seqs = enable_ls_decode_core_scheduler_ ? _schedule_ls_decode_admission() : _schedule_prefill();

        // Check if any sequences were scheduled in prefill
        for (const auto& seqs : dp_seqs) {
            if (!seqs.empty()) {
                has_prefill = true;
                break;
            }
        }

        if (!has_prefill && enable_ls_decode_core_scheduler_) {
            auto consolidation_plan = _maybe_plan_ls_kv_consolidation();
            if (consolidation_plan) {
                ScheduleResult maintenance;
                maintenance.action                = ScheduleAction::KV_CONSOLIDATION;
                maintenance.is_prefill            = false;
                maintenance.kv_consolidation_plan = std::move(consolidation_plan);
                _populate_ls_kv_consolidation_telemetry(maintenance);
                return maintenance;
            }
        }

        if (!has_prefill) {
            // No prefill sequences, schedule decode
            dp_seqs = enable_ls_decode_core_scheduler_ ? _schedule_ls_decode() : _schedule_decode();
        }
    }

    ScheduleResult result;
    result.dp_seqs    = dp_seqs;
    result.is_prefill = has_prefill;
    result.action     = has_prefill ? ScheduleAction::ADMISSION : ScheduleAction::DECODE;

    // Prepare dp_sp_seqs and filtered_dp_sp_seqs
    result.dp_sp_seqs.reserve(attention_dp_ * attention_sp_);
    result.filtered_dp_sp_seqs.reserve(attention_dp_ * attention_sp_);

    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            // dp_sp_seqs is just dp_seqs[dp_idx] repeated for each sp_idx
            result.dp_sp_seqs.push_back(dp_seqs[dp_idx]);

            // filtered_dp_sp_seqs is dp_seqs[dp_idx] filtered by master_sp_idx
            std::vector<std::shared_ptr<Sequence>> filtered;
            for (const auto& seq : dp_seqs[dp_idx]) {
                if (seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_ == sp_idx) {
                    filtered.push_back(seq);
                }
            }
            result.filtered_dp_sp_seqs.push_back(std::move(filtered));
        }
    }

    result.sp_send_counts.resize(attention_dp_);
    result.sp_recv_counts.resize(attention_dp_);
    result.sp_size_hist_per_dp.resize(attention_dp_);
    result.sp_res_matrix.clear();

    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        result.sp_send_counts[dp_idx].resize(attention_sp_);
        result.sp_recv_counts[dp_idx].resize(attention_sp_);
        result.sp_size_hist_per_dp[dp_idx].assign(attention_sp_ + 1, 0);

        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            // SP Send Count: Number of sequences where this SP rank is MASTER (initiator)
            // AND the sequence is actually distributed (has blocks on > 1 ranks).
            int         send_count = 0;
            const auto& sp_seqs    = result.filtered_dp_sp_seqs[dp_idx * attention_sp_ + sp_idx];
            for (const auto& seq : sp_seqs) {
                if (has_remote_committed_kv(*seq, attention_sp_)) {
                    send_count++;
                }
            }
            result.sp_send_counts[dp_idx][sp_idx] = send_count;

            // SP Recv Count: Number of sequences where this SP rank PARTICIPATES
            // AND the sequence is actually distributed.
            int recv_count = 0;
            for (const auto& seq : dp_seqs[dp_idx]) {
                bool is_dummy = false;
                for (const auto& dummy : worker_state[dp_idx]->dummy_seqs) {
                    if (seq == dummy) {
                        is_dummy = true;
                        break;
                    }
                }

                if (!is_dummy) {
                    const auto& block_ctx     = seq->block_ctx(BlockContextSlot::ACTIVE);
                    int         master_sp_idx = block_ctx.master_sp_idx_;

                    if (seq->committed_context_len(BlockContextSlot::ACTIVE, sp_idx) > 0 && master_sp_idx != sp_idx) {
                        recv_count++;
                    }
                }
            }
            result.sp_recv_counts[dp_idx][sp_idx] = recv_count;
        }

        for (const auto& seq : dp_seqs[dp_idx]) {
            bool is_dummy = false;
            for (const auto& dummy : worker_state[dp_idx]->dummy_seqs) {
                if (seq == dummy) {
                    is_dummy = true;
                    break;
                }
            }

            if (is_dummy) {
                continue;
            }

            int active_ranks = committed_kv_rank_count(*seq, attention_sp_);

            if (active_ranks >= 0 && active_ranks <= attention_sp_) {
                result.sp_size_hist_per_dp[dp_idx][active_ranks]++;
            }
        }

        // SP Communication Matrix Logic
        // Initialize matrix for this DP rank: [attention_sp_][attention_sp_]
        // result.sp_comm_matrix.push_back(
        // std::vector<std::vector<int>>(attention_sp_, std::vector<int>(attention_sp_, 0)));

        result.sp_q_matrix.push_back(std::vector<std::vector<int>>(attention_sp_, std::vector<int>(attention_sp_, 0)));

        result.sp_res_matrix.push_back(
            std::vector<std::vector<int>>(attention_sp_, std::vector<int>(attention_sp_, 0)));

        for (const auto& seq : dp_seqs[dp_idx]) {
            bool is_dummy = false;
            for (const auto& dummy : worker_state[dp_idx]->dummy_seqs) {
                if (seq == dummy) {
                    is_dummy = true;
                    break;
                }
            }
            if (is_dummy)
                continue;

            if (has_remote_committed_kv(*seq, attention_sp_)) {
                int master_sp_idx = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;

                // For each participating rank:
                for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
                    if (seq->committed_context_len(BlockContextSlot::ACTIVE, sp_idx) > 0) {
                        if (sp_idx != master_sp_idx) {
                            // Q Matrix: Master sends Q to each participant.
                            result.sp_q_matrix[dp_idx][master_sp_idx][sp_idx]++;

                            // Res Matrix: Each participant sends one result back to the master.
                            result.sp_res_matrix[dp_idx][sp_idx][master_sp_idx]++;
                        }
                    }
                }
            }
        }
    }

    // Calculate waiting queue block metrics (centralized: replicate global values to all DPs)
    auto& wait_queue = (mode_ != "decode") ? waiting : waiting_migration;

    int head_blocks  = 0;
    int total_blocks = 0;

    if (!wait_queue.empty()) {
        auto head_seq = wait_queue.front();
        head_blocks   = (head_seq->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
    }

    for (const auto& seq : wait_queue) {
        total_blocks += (seq->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
    }
    if (enable_ls_decode_core_scheduler_) {
        if (!ls_pending_decode_batches_.empty() && !ls_pending_decode_batches_.front().sequences.empty()) {
            auto head_seq = ls_pending_decode_batches_.front().sequences.front();
            head_blocks   = (head_seq->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
        }
        for (const auto& batch : ls_pending_decode_batches_) {
            for (const auto& seq : batch.sequences) {
                total_blocks += (seq->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
            }
        }
    }

    // Replicate to all DP workers (centralized has shared global queue)
    result.waiting_head_blocks.resize(attention_dp_, head_blocks);
    result.waiting_total_blocks.resize(attention_dp_, total_blocks);

    if (enable_ls_decode_core_scheduler_) {
        for (const auto& record : ls_step_initial_records_) {
            result.ls_initial_batch_ids.push_back(record.batch_id);
            result.ls_initial_group_ids.push_back(record.group_id);
            result.ls_initial_kv_dops.push_back(record.initial_kv_dop);
            result.ls_initial_kv_ranks.push_back(record.initial_kv_ranks);
            result.ls_initial_sequence_ids.push_back(record.sequence_ids);
            result.ls_initial_prompt_kv_tokens.push_back(record.prompt_kv_tokens);
            result.ls_initial_provisional_pending_targets.push_back(record.provisional_pending_targets);
            result.ls_initial_admission_orders.push_back(record.admission_order);
            result.ls_initial_admission_attempts.push_back(record.admission_attempts);
            result.ls_initial_is_recovery_batch.push_back(record.is_recovery_batch);
            result.ls_initial_parent_batch_ids.push_back(record.parent_batch_id);
            result.ls_initial_admission_kinds.push_back(record.admission_kind);
        }
        for (const auto& [batch_id, sequence_ids] : ls_step_sealed_batches_) {
            result.ls_sealed_batch_ids.push_back(batch_id);
            result.ls_sealed_batch_sequence_ids.push_back(sequence_ids);
        }
        for (size_t idx = 0; idx < ls_step_group_plans_.size(); ++idx) {
            uint64_t group_id = ls_step_group_plan_ids_[idx];
            auto     group_it = ls_groups_.find(group_id);
            if (group_it == ls_groups_.end()) {
                continue;
            }
            const auto& group = group_it->second;
            const auto& plan  = ls_step_group_plans_[idx];
            result.ls_group_ids.push_back(group_id);
            result.ls_group_dp_indices.push_back(group.dp_idx);
            result.ls_real_batch_sizes.push_back(static_cast<int>(plan.sequence_master_ranks.size()));
            result.ls_master_dops.push_back(static_cast<int>(plan.master_ranks.size()));
            result.ls_kv_dops.push_back(worker_state[group.dp_idx]->get_kv_participant_count(group.sequences));
            result.ls_master_ranks.push_back(plan.master_ranks);
            result.ls_master_batch_sizes.push_back(plan.master_batch_sizes);
            result.ls_group_rank_allocations.push_back(group.allocated_attention_ranks);
            result.ls_group_used_kv_tokens.push_back(plan.group_used_kv_tokens);
            result.ls_group_used_kv_blocks.push_back(plan.group_used_kv_blocks);
            std::vector<uint64_t> sequence_ids;
            std::vector<int>      pending_blocks(attention_sp_, 0);
            size_t                assignment_idx = 0;
            for (const auto& seq : group.sequences) {
                if (!seq || seq->status != SequenceStatus::RUNNING) {
                    continue;
                }
                sequence_ids.push_back(seq->seq_id);
                int master           = plan.sequence_master_ranks.at(assignment_idx++);
                int committed        = seq->committed_context_len(BlockContextSlot::ACTIVE, master);
                int committed_blocks = (committed + Sequence::block_size - 1) / Sequence::block_size;
                int table_blocks     = static_cast<int>(seq->block_table(BlockContextSlot::ACTIVE, master).size());
                pending_blocks[master] += std::max(0, table_blocks - committed_blocks);
            }
            result.ls_iteration_sequence_ids.push_back(std::move(sequence_ids));
            result.ls_iteration_master_assignments.push_back(plan.sequence_master_ranks);
            result.ls_pending_append_blocks_per_master.push_back(std::move(pending_blocks));
            result.ls_new_master_ranks.push_back(plan.new_allocation_ranks);
            result.ls_reused_passive_master_ranks.push_back(idx < ls_step_reused_passive_masters_.size() ?
                                                                ls_step_reused_passive_masters_[idx] :
                                                                std::vector<int>{});
            result.ls_scale_reasons.push_back(plan.scale_reason);
            result.ls_historical_kv_migration_bytes.push_back(0);
        }
        result.ls_preempted_sequence_ids = ls_step_preempted_sequence_ids_;
        result.ls_preemption_reasons     = ls_step_preemption_reasons_;
        result.ls_planning_latency_ms    = ls_step_planning_latency_ms_;
        result.ls_pending_batch_count    = static_cast<int>(ls_pending_decode_batches_.size());
        for (const auto& batch : ls_pending_decode_batches_) {
            result.ls_pending_request_count += static_cast<int>(batch.sequences.size());
            result.ls_oldest_pending_batch_age_steps =
                std::max(result.ls_oldest_pending_batch_age_steps, ls_schedule_step_ - batch.enqueue_step);
            result.ls_max_pending_batch_attempts =
                std::max(result.ls_max_pending_batch_attempts, batch.admission_attempts);
        }
        result.ls_atomic_admission_no_fit_count   = ls_step_atomic_no_fit_count_;
        result.ls_atomic_admission_merge_count    = ls_step_atomic_merge_count_;
        result.ls_atomic_admission_rollback_count = ls_step_atomic_rollback_count_;
    }

    _populate_ls_kv_consolidation_telemetry(result);

    return result;
}

std::vector<std::vector<std::shared_ptr<Sequence>>> Scheduler::_schedule_decode_prefill_latency_aware()
{
    std::vector<std::vector<std::shared_ptr<Sequence>>>         scheduled_seqs(attention_dp_);
    std::vector<std::vector<std::shared_ptr<Sequence>>>         tentative_batches(attention_dp_);
    std::vector<std::optional<SPStateManager::DecodeBatchPlan>> tentative_plans(attention_dp_);

    struct PlanningCacheGuard {
        std::vector<std::shared_ptr<SPStateManager>>& workers;
        explicit PlanningCacheGuard(std::vector<std::shared_ptr<SPStateManager>>& worker_state): workers(worker_state)
        {
            for (auto& worker : workers) {
                worker->begin_decode_planning();
            }
        }
        ~PlanningCacheGuard()
        {
            for (auto& worker : workers) {
                worker->end_decode_planning();
            }
        }
    } planning_cache_guard(worker_state);

    auto& waiting_queue = waiting_migration;

    while (!waiting_queue.empty()) {
        auto seq = waiting_queue.front();

        std::vector<std::pair<int, int>> dp_order;
        dp_order.reserve(attention_dp_);
        for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
            int projected_batch =
                worker_state[dp_idx]->num_running_seqs() + static_cast<int>(tentative_batches[dp_idx].size());
            dp_order.push_back({projected_batch, dp_idx});
        }
        std::sort(dp_order.begin(), dp_order.end());

        bool admitted = false;
        for (const auto& entry : dp_order) {
            int   dp_idx          = entry.second;
            auto& candidate_batch = tentative_batches[dp_idx];
            candidate_batch.push_back(seq);

            auto candidate_plan = worker_state[dp_idx]->plan_decode_batch(candidate_batch);
            if (!candidate_plan.has_value()) {
                candidate_batch.pop_back();
                continue;
            }

            tentative_plans[dp_idx] = std::move(candidate_plan);
            waiting_queue.pop_front();
            admitted = true;
            break;
        }

        if (!admitted) {
            break;
        }
    }

    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        if (!tentative_plans[dp_idx].has_value()) {
            continue;
        }

        auto& plan  = *tentative_plans[dp_idx];
        auto& batch = tentative_batches[dp_idx];
        for (size_t i = 0; i < batch.size(); ++i) {
            auto& seq = batch[i];
            worker_state[dp_idx]->apply_planned_placement(*seq, plan.placements[i]);

            auto& block_ctx   = seq->block_ctx(BlockContextSlot::ACTIVE);
            block_ctx.dp_idx_ = dp_idx;

            worker_state[dp_idx]->allocate(*seq);
            seq->status = SequenceStatus::RUNNING;
            worker_state[dp_idx]->running.push_back(seq);
            scheduled_seqs[dp_idx].push_back(seq);

            if (seq->metric) {
                seq->metric->record_first_scheduled();
                seq->metric->record_decode_scheduled();
            }
        }
    }

    return scheduled_seqs;
}

std::vector<int> Scheduler::_ls_unallocated_ranks(int dp_idx, std::optional<uint64_t> excluding_group) const
{
    std::vector<bool> allocated(attention_sp_, false);
    if (dp_idx < 0 || dp_idx >= attention_dp_) {
        return {};
    }
    for (uint64_t group_id : ls_group_ids_by_dp_[dp_idx]) {
        if (excluding_group.has_value() && group_id == *excluding_group) {
            continue;
        }
        auto it = ls_groups_.find(group_id);
        if (it == ls_groups_.end()) {
            continue;
        }
        for (int rank : it->second.allocated_attention_ranks) {
            if (rank >= 0 && rank < attention_sp_) {
                allocated[rank] = true;
            }
        }
    }
    std::vector<int> result;
    for (int rank = 0; rank < attention_sp_; ++rank) {
        if (!allocated[rank]) {
            result.push_back(rank);
        }
    }
    return result;
}

std::optional<std::pair<std::vector<int>, std::vector<std::vector<int>>>>
Scheduler::_plan_ls_initial_placement(int                                           dp_idx,
                                      const std::vector<std::shared_ptr<Sequence>>& batch,
                                      const std::vector<int>&                       rank_pool,
                                      const std::vector<std::shared_ptr<Sequence>>& existing_sequences,
                                      const std::vector<int>&                       base_allocation) const
{
    if (batch.empty() || static_cast<int>(batch.size()) > max_num_seqs_) {
        return std::nullopt;
    }
    std::vector<int> ordered_pool = rank_pool;
    std::sort(ordered_pool.begin(), ordered_pool.end(), [&](int lhs, int rhs) {
        int lhs_free = worker_state[dp_idx]->block_manager.at(lhs)->num_free_blocks();
        int rhs_free = worker_state[dp_idx]->block_manager.at(rhs)->num_free_blocks();
        return lhs_free != rhs_free ? lhs_free > rhs_free : lhs < rhs;
    });
    ordered_pool.erase(std::unique(ordered_pool.begin(), ordered_pool.end()), ordered_pool.end());

    int first_d = ls_decode_initial_kv_dop_ == 0 ? 1 : ls_decode_initial_kv_dop_;
    int last_d  = ls_decode_initial_kv_dop_ == 0 ? std::min(attention_sp_, static_cast<int>(ordered_pool.size())) :
                                                   ls_decode_initial_kv_dop_;
    for (int d = first_d; d <= last_d; ++d) {
        if (d <= 0 || d > static_cast<int>(ordered_pool.size())) {
            continue;
        }
        std::vector<int>              ranks(ordered_pool.begin(), ordered_pool.begin() + d);
        std::vector<std::vector<int>> placements(batch.size(), std::vector<int>(attention_sp_, 0));
        std::vector<int>              needed_blocks(attention_sp_, 0);
        for (size_t seq_idx = 0; seq_idx < batch.size(); ++seq_idx) {
            int prompt_tokens = batch[seq_idx]->num_tokens;
            int base          = prompt_tokens / d;
            int remainder     = prompt_tokens % d;
            for (int rank_idx = 0; rank_idx < d; ++rank_idx) {
                int tokens                = base + (rank_idx < remainder ? 1 : 0);
                int rank                  = ranks[rank_idx];
                placements[seq_idx][rank] = tokens;
                needed_blocks[rank] += (tokens + Sequence::block_size - 1) / Sequence::block_size;
            }
            // Dummy Prefill appends one pending input token to the provisional
            // master immediately after admission. Charge a new tail block when
            // the prompt ends exactly on a block boundary.
            int provisional_master = ranks.front();
            if (placements[seq_idx][provisional_master] % Sequence::block_size == 0) {
                needed_blocks[provisional_master]++;
            }
        }

        bool feasible = true;
        for (int rank : ranks) {
            int free_blocks = worker_state[dp_idx]->block_manager.at(rank)->num_free_blocks();
            if (free_blocks < needed_blocks[rank]) {
                feasible = false;
                break;
            }
        }
        // Shadow a deterministic round-robin master assignment to ensure the
        // initial placement has at least one receiver-metadata-feasible first
        // Decode iteration. The real source-greedy planner may choose a better
        // assignment, but admission must never commit a placement with no
        // legal receiver shape.
        if (feasible) {
            std::vector<int> planning_ranks = base_allocation;
            // D_init does not constrain first-iteration master DoP: any rank
            // currently available to the group can be added without history
            // migration by the Decode planner.
            for (int rank : ordered_pool) {
                if (std::find(planning_ranks.begin(), planning_ranks.end(), rank) == planning_ranks.end()) {
                    planning_ranks.push_back(rank);
                }
            }
            std::vector<int> receiver_load(attention_sp_, 0);
            std::vector<int> master_load(attention_sp_, 0);
            size_t           master_cursor = 0;
            for (const auto& seq : existing_sequences) {
                if (!seq || seq->status != SequenceStatus::RUNNING) {
                    continue;
                }
                int master = planning_ranks[master_cursor++ % planning_ranks.size()];
                master_load[master]++;
                for (int owner = 0; owner < attention_sp_; ++owner) {
                    if (owner != master && seq->committed_context_len(BlockContextSlot::ACTIVE, owner) > 0) {
                        receiver_load[owner]++;
                    }
                }
            }
            for (size_t seq_idx = 0; seq_idx < batch.size(); ++seq_idx) {
                int master = planning_ranks[master_cursor++ % planning_ranks.size()];
                master_load[master]++;
                for (int owner = 0; owner < attention_sp_; ++owner) {
                    if (owner != master && placements[seq_idx][owner] > 0) {
                        receiver_load[owner]++;
                    }
                }
            }
            feasible = std::all_of(receiver_load.begin(),
                                   receiver_load.end(),
                                   [&](int count) { return count <= max_num_recv_seqs_; })
                       && std::all_of(master_load.begin(), master_load.end(), [&](int count) {
                              return count <= std::min(max_num_seqs_, max_num_batched_tokens_);
                          });
            if (feasible) {
                for (int rank : planning_ranks) {
                    int free_blocks = worker_state[dp_idx]->block_manager.at(rank)->num_free_blocks();
                    int headroom    = static_cast<int>(std::ceil(master_load[rank] * reserved_blocks_per_req_));
                    if (free_blocks < needed_blocks[rank] + headroom) {
                        feasible = false;
                        break;
                    }
                }
            }
        }
        if (feasible) {
            return std::make_pair(std::move(ranks), std::move(placements));
        }
        if (ls_decode_initial_kv_dop_ != 0) {
            break;
        }
    }
    return std::nullopt;
}

bool Scheduler::_ls_batch_fits_empty_system(const std::vector<std::shared_ptr<Sequence>>& batch) const
{
    if (batch.empty() || static_cast<int>(batch.size()) > max_num_seqs_) {
        return false;
    }

    int first_d = ls_decode_initial_kv_dop_ == 0 ? 1 : ls_decode_initial_kv_dop_;
    int last_d  = ls_decode_initial_kv_dop_ == 0 ? attention_sp_ : ls_decode_initial_kv_dop_;
    for (int d = first_d; d <= last_d; ++d) {
        if (d <= 0 || d > attention_sp_) {
            continue;
        }
        std::vector<std::vector<int>> placements(batch.size(), std::vector<int>(attention_sp_, 0));
        std::vector<int>              needed_blocks(attention_sp_, 0);
        for (size_t seq_idx = 0; seq_idx < batch.size(); ++seq_idx) {
            int prompt_tokens = batch[seq_idx]->num_tokens;
            int base          = prompt_tokens / d;
            int remainder     = prompt_tokens % d;
            for (int rank = 0; rank < d; ++rank) {
                int tokens                = base + (rank < remainder ? 1 : 0);
                placements[seq_idx][rank] = tokens;
                needed_blocks[rank] += (tokens + Sequence::block_size - 1) / Sequence::block_size;
            }
            if (placements[seq_idx][0] % Sequence::block_size == 0) {
                needed_blocks[0]++;
            }
        }

        std::vector<int> receiver_load(attention_sp_, 0);
        std::vector<int> master_load(attention_sp_, 0);
        for (size_t seq_idx = 0; seq_idx < batch.size(); ++seq_idx) {
            int master = static_cast<int>(seq_idx % attention_sp_);
            master_load[master]++;
            for (int owner = 0; owner < attention_sp_; ++owner) {
                if (owner != master && placements[seq_idx][owner] > 0) {
                    receiver_load[owner]++;
                }
            }
        }

        bool feasible = std::all_of(
            receiver_load.begin(), receiver_load.end(), [&](int count) { return count <= max_num_recv_seqs_; });
        feasible = feasible && std::all_of(master_load.begin(), master_load.end(), [&](int count) {
                       return count <= std::min(max_num_seqs_, max_num_batched_tokens_);
                   });
        for (int rank = 0; feasible && rank < attention_sp_; ++rank) {
            int headroom = static_cast<int>(std::ceil(master_load[rank] * reserved_blocks_per_req_));
            feasible     = needed_blocks[rank] + headroom <= ls_empty_system_free_blocks_per_rank_[rank];
        }
        if (feasible) {
            return true;
        }
        if (ls_decode_initial_kv_dop_ != 0) {
            break;
        }
    }
    return false;
}

void Scheduler::_seal_ls_decode_arrivals()
{
    if (waiting_migration.empty()) {
        return;
    }

    const size_t seal_limit  = static_cast<size_t>(attention_dp_) * static_cast<size_t>(max_num_seqs_);
    const size_t n           = std::min(waiting_migration.size(), seal_limit);
    const size_t num_batches = std::min(static_cast<size_t>(attention_dp_), n);
    const size_t base        = n / num_batches;
    const size_t remainder   = n % num_batches;

    for (size_t batch_idx = 0; batch_idx < num_batches && !waiting_migration.empty(); ++batch_idx) {
        size_t candidate_size = std::min(base + (batch_idx < remainder ? 1U : 0U), waiting_migration.size());
        std::vector<std::shared_ptr<Sequence>> candidate;
        while (candidate_size > 0) {
            candidate.assign(waiting_migration.begin(), waiting_migration.begin() + candidate_size);
            if (_ls_batch_fits_empty_system(candidate)) {
                break;
            }
            candidate_size--;
        }
        if (candidate_size == 0) {
            throw std::runtime_error("LS Decode request cannot fit an otherwise empty DP; refusing to seal a "
                                     "permanently blocked logical batch");
        }

        std::unordered_set<uint64_t> candidate_sequence_ids;
        for (const auto& seq : candidate) {
            if (!seq || !candidate_sequence_ids.insert(seq->seq_id).second || ls_seq_to_batch_.count(seq->seq_id)) {
                throw std::runtime_error("LS Decode seal found duplicate live sequence ownership");
            }
        }

        PendingDecodeBatch batch;
        batch.batch_id      = next_ls_batch_id_;
        batch.sequences     = candidate;
        batch.enqueue_order = next_ls_enqueue_order_;
        batch.enqueue_step  = ls_schedule_step_;
        std::vector<uint64_t> sequence_ids;
        sequence_ids.reserve(batch.sequences.size());
        for (const auto& seq : batch.sequences) {
            sequence_ids.push_back(seq->seq_id);
        }

        std::vector<uint64_t> inserted_sequence_ids;
        inserted_sequence_ids.reserve(sequence_ids.size());
        bool sealed_event_published  = false;
        bool pending_batch_published = false;
        try {
            ls_step_sealed_batches_.push_back({batch.batch_id, sequence_ids});
            sealed_event_published = true;
            ls_pending_decode_batches_.push_back(batch);
            pending_batch_published = true;
            for (uint64_t seq_id : sequence_ids) {
                auto [it, inserted] = ls_seq_to_batch_.emplace(seq_id, batch.batch_id);
                (void)it;
                if (!inserted) {
                    throw std::runtime_error("LS Decode seal found duplicate live sequence ownership");
                }
                inserted_sequence_ids.push_back(seq_id);
            }
        }
        catch (...) {
            for (uint64_t seq_id : inserted_sequence_ids) {
                ls_seq_to_batch_.erase(seq_id);
            }
            if (pending_batch_published) {
                ls_pending_decode_batches_.pop_back();
            }
            if (sealed_event_published) {
                ls_step_sealed_batches_.pop_back();
            }
            throw;
        }
        for (size_t idx = 0; idx < candidate_size; ++idx) {
            waiting_migration.pop_front();
        }
        next_ls_batch_id_++;
        next_ls_enqueue_order_++;
    }
}

void Scheduler::_merge_ls_groups(uint64_t lhs_group_id, uint64_t rhs_group_id)
{
    if (lhs_group_id == rhs_group_id) {
        return;
    }
    uint64_t survivor_id = std::min(lhs_group_id, rhs_group_id);
    uint64_t removed_id  = std::max(lhs_group_id, rhs_group_id);
    auto     survivor_it = ls_groups_.find(survivor_id);
    auto     removed_it  = ls_groups_.find(removed_id);
    if (survivor_it == ls_groups_.end() || removed_it == ls_groups_.end()) {
        return;
    }
    auto& survivor = survivor_it->second;
    auto& removed  = removed_it->second;
    if (survivor.dp_idx != removed.dp_idx) {
        throw std::runtime_error("cannot merge LS decode groups across DP domains");
    }

    std::unordered_map<uint64_t, std::shared_ptr<Sequence>> sequences_by_id;
    std::vector<std::shared_ptr<Sequence>>                  prior_sequence_order;
    for (const auto& seq : survivor.sequences) {
        sequences_by_id.emplace(seq->seq_id, seq);
        prior_sequence_order.push_back(seq);
    }
    for (const auto& seq : removed.sequences) {
        sequences_by_id.emplace(seq->seq_id, seq);
        prior_sequence_order.push_back(seq);
    }
    auto merged_records = survivor.initial_batch_placements;
    merged_records.insert(
        merged_records.end(), removed.initial_batch_placements.begin(), removed.initial_batch_placements.end());
    std::stable_sort(merged_records.begin(), merged_records.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.admission_order, lhs.batch_id) < std::tie(rhs.admission_order, rhs.batch_id);
    });
    std::vector<std::shared_ptr<Sequence>> merged_sequences;
    std::unordered_set<uint64_t>           emitted_sequence_ids;
    for (const auto& record : merged_records) {
        for (uint64_t seq_id : record.sequence_ids) {
            auto seq          = sequences_by_id.find(seq_id);
            auto active_batch = ls_seq_to_batch_.find(seq_id);
            if (seq != sequences_by_id.end() && active_batch != ls_seq_to_batch_.end()
                && active_batch->second == record.batch_id && emitted_sequence_ids.insert(seq_id).second) {
                merged_sequences.push_back(seq->second);
            }
        }
    }
    for (const auto& seq : prior_sequence_order) {
        if (seq && seq->status == SequenceStatus::RUNNING && emitted_sequence_ids.insert(seq->seq_id).second) {
            throw std::runtime_error("live LS sequence has no active initial-batch placement record");
        }
    }
    auto merged_allocation = survivor.allocated_attention_ranks;
    for (int rank : removed.allocated_attention_ranks) {
        if (std::find(merged_allocation.begin(), merged_allocation.end(), rank) == merged_allocation.end()) {
            merged_allocation.push_back(rank);
        }
    }
    survivor.initial_batch_placements  = std::move(merged_records);
    survivor.sequences                 = std::move(merged_sequences);
    survivor.allocated_attention_ranks = std::move(merged_allocation);
    survivor.last_scale_up_step        = ls_schedule_step_;
    survivor.kv_candidate_target_dop   = -1;
    survivor.kv_candidate_stable_steps = 0;
    for (const auto& seq : removed.sequences) {
        ls_seq_to_group_[seq->seq_id] = survivor_id;
    }

    auto& group_ids = ls_group_ids_by_dp_[survivor.dp_idx];
    group_ids.erase(std::remove(group_ids.begin(), group_ids.end(), removed_id), group_ids.end());
    std::sort(group_ids.begin(), group_ids.end());
    ls_groups_.erase(removed_it);

    std::unordered_map<uint64_t, uint64_t> active_batch_group;
    for (const auto& [group_id, group] : ls_groups_) {
        for (const auto& seq : group.sequences) {
            auto owner = ls_seq_to_batch_.find(seq->seq_id);
            if (owner == ls_seq_to_batch_.end()) {
                throw std::runtime_error("running LS sequence has no active batch owner");
            }
            auto [it, inserted] = active_batch_group.emplace(owner->second, group_id);
            if (!inserted && it->second != group_id) {
                throw std::runtime_error("active LS logical batch is split across Decode groups");
            }
        }
    }
}

void Scheduler::_remove_seq_from_ls_group(uint64_t seq_id, bool clear_batch_owner)
{
    auto owner = ls_seq_to_group_.find(seq_id);
    if (owner == ls_seq_to_group_.end()) {
        return;
    }
    uint64_t group_id = owner->second;
    ls_seq_to_group_.erase(owner);
    if (clear_batch_owner) {
        ls_seq_to_batch_.erase(seq_id);
    }
    auto group_it = ls_groups_.find(group_id);
    if (group_it == ls_groups_.end()) {
        return;
    }
    auto& seqs = group_it->second.sequences;
    seqs.erase(std::remove_if(seqs.begin(), seqs.end(), [&](const auto& seq) { return !seq || seq->seq_id == seq_id; }),
               seqs.end());
    if (!seqs.empty()) {
        group_it->second.kv_candidate_target_dop   = -1;
        group_it->second.kv_candidate_stable_steps = 0;
        return;
    }
    int   dp_idx    = group_it->second.dp_idx;
    auto& group_ids = ls_group_ids_by_dp_[dp_idx];
    group_ids.erase(std::remove(group_ids.begin(), group_ids.end(), group_id), group_ids.end());
    ls_groups_.erase(group_it);
}

void Scheduler::_reconcile_ls_groups()
{
    std::vector<uint64_t> stale_sequences;
    for (const auto& [seq_id, group_id] : ls_seq_to_group_) {
        auto group_it = ls_groups_.find(group_id);
        if (group_it == ls_groups_.end()) {
            stale_sequences.push_back(seq_id);
            continue;
        }
        auto seq_it = std::find_if(group_it->second.sequences.begin(),
                                   group_it->second.sequences.end(),
                                   [&](const auto& seq) { return seq && seq->seq_id == seq_id; });
        if (seq_it == group_it->second.sequences.end() || (*seq_it)->status != SequenceStatus::RUNNING) {
            stale_sequences.push_back(seq_id);
        }
    }
    for (uint64_t seq_id : stale_sequences) {
        _remove_seq_from_ls_group(seq_id);
    }
}

std::vector<std::vector<std::shared_ptr<Sequence>>> Scheduler::_schedule_ls_decode_admission()
{
    std::vector<std::vector<std::shared_ptr<Sequence>>> scheduled(attention_dp_);
    _reconcile_ls_groups();
    _seal_ls_decode_arrivals();
    if (ls_pending_decode_batches_.empty()) {
        return scheduled;
    }

    struct AdmissionCandidate {
        int                           dp_idx = -1;
        std::optional<uint64_t>       merge_target;
        std::vector<int>              ranks;
        std::vector<std::vector<int>> placements;
        int                           projected_live_sequences = 0;
        int                           new_allocation_ranks     = 0;
    };

    std::vector<bool>     dp_admitted(attention_dp_, false);
    std::vector<uint64_t> pending_ids;
    pending_ids.reserve(ls_pending_decode_batches_.size());
    for (const auto& batch : ls_pending_decode_batches_) {
        pending_ids.push_back(batch.batch_id);
    }

    for (uint64_t pending_id : pending_ids) {
        if (std::all_of(dp_admitted.begin(), dp_admitted.end(), [](bool used) { return used; })) {
            break;
        }
        auto pending_it = std::find_if(ls_pending_decode_batches_.begin(),
                                       ls_pending_decode_batches_.end(),
                                       [&](const auto& batch) { return batch.batch_id == pending_id; });
        if (pending_it == ls_pending_decode_batches_.end()) {
            continue;
        }
        auto& batch = *pending_it;
        if (batch.state != PendingDecodeBatchState::QUEUED) {
            throw std::runtime_error("LS pending batch is not queued at planning time");
        }
        batch.admission_attempts++;

        std::optional<AdmissionCandidate> selected;
        for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
            if (dp_admitted[dp_idx]) {
                continue;
            }
            auto rank_pool = _ls_unallocated_ranks(dp_idx);
            auto plan      = _plan_ls_initial_placement(dp_idx, batch.sequences, rank_pool);
            if (!plan.has_value()) {
                continue;
            }
            AdmissionCandidate candidate;
            candidate.dp_idx     = dp_idx;
            candidate.ranks      = std::move(plan->first);
            candidate.placements = std::move(plan->second);
            candidate.projected_live_sequences =
                worker_state[dp_idx]->num_running_seqs() + static_cast<int>(batch.sequences.size());
            candidate.new_allocation_ranks = static_cast<int>(candidate.ranks.size());
            if (!selected.has_value()
                || std::tie(candidate.projected_live_sequences, candidate.new_allocation_ranks, candidate.dp_idx)
                       < std::tie(
                           selected->projected_live_sequences, selected->new_allocation_ranks, selected->dp_idx)) {
                selected = std::move(candidate);
            }
        }

        if (!selected.has_value()) {
            for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
                if (dp_admitted[dp_idx]) {
                    continue;
                }
                std::vector<uint64_t> group_ids = ls_group_ids_by_dp_[dp_idx];
                std::sort(group_ids.begin(), group_ids.end());
                for (uint64_t group_id : group_ids) {
                    const auto&      group     = ls_groups_.at(group_id);
                    std::vector<int> rank_pool = group.allocated_attention_ranks;
                    for (int rank : _ls_unallocated_ranks(dp_idx)) {
                        if (std::find(rank_pool.begin(), rank_pool.end(), rank) == rank_pool.end()) {
                            rank_pool.push_back(rank);
                        }
                    }
                    auto plan = _plan_ls_initial_placement(
                        dp_idx, batch.sequences, rank_pool, group.sequences, group.allocated_attention_ranks);
                    if (!plan.has_value()) {
                        continue;
                    }
                    AdmissionCandidate candidate;
                    candidate.dp_idx       = dp_idx;
                    candidate.merge_target = group_id;
                    candidate.ranks        = std::move(plan->first);
                    candidate.placements   = std::move(plan->second);
                    candidate.projected_live_sequences =
                        static_cast<int>(group.sequences.size()) + static_cast<int>(batch.sequences.size());
                    candidate.new_allocation_ranks =
                        static_cast<int>(std::count_if(candidate.ranks.begin(), candidate.ranks.end(), [&](int rank) {
                            return std::find(group.allocated_attention_ranks.begin(),
                                             group.allocated_attention_ranks.end(),
                                             rank)
                                   == group.allocated_attention_ranks.end();
                        }));
                    if (!selected.has_value()
                        || std::tie(candidate.new_allocation_ranks,
                                    candidate.projected_live_sequences,
                                    group_id,
                                    candidate.dp_idx)
                               < std::tie(selected->new_allocation_ranks,
                                          selected->projected_live_sequences,
                                          *selected->merge_target,
                                          selected->dp_idx)) {
                        selected = std::move(candidate);
                    }
                }
            }
        }

        if (!selected.has_value()) {
            ls_step_atomic_no_fit_count_++;
            continue;
        }

        for (const auto& seq : batch.sequences) {
            if (!seq) {
                throw std::runtime_error("LS pending batch contains a null sequence");
            }
            auto active_batch = ls_seq_to_batch_.find(seq->seq_id);
            if (seq->status != SequenceStatus::WAITING || active_batch == ls_seq_to_batch_.end()
                || active_batch->second != batch.batch_id || ls_seq_to_group_.count(seq->seq_id)) {
                throw std::runtime_error("LS pending batch changed after placement planning");
            }
        }
        if (selected->placements.size() != batch.sequences.size() || selected->ranks.empty()) {
            throw std::runtime_error("LS whole-batch plan has incomplete placement output");
        }

        const int                       dp_idx   = selected->dp_idx;
        const bool                      is_merge = selected->merge_target.has_value();
        const uint64_t                  group_id = is_merge ? *selected->merge_target : next_ls_group_id_;
        std::optional<DecodeGroupState> original_group;
        if (is_merge) {
            original_group = ls_groups_.at(group_id);
        }
        std::vector<BlockContext> original_contexts;
        original_contexts.reserve(batch.sequences.size());
        for (const auto& seq : batch.sequences) {
            original_contexts.push_back(seq->block_ctx(BlockContextSlot::ACTIVE));
        }

        bool physical_allocation_committed = false;
        try {
            batch.state = PendingDecodeBatchState::COMMITTING;
            for (size_t seq_idx = 0; seq_idx < batch.sequences.size(); ++seq_idx) {
                auto& ctx                    = batch.sequences[seq_idx]->block_ctx(BlockContextSlot::ACTIVE);
                ctx.dp_idx_                  = dp_idx;
                ctx.master_sp_idx_           = selected->ranks.front();
                ctx.pending_token_present_   = false;
                ctx.pending_token_target_sp_ = -1;
                ctx.num_dispatched_tokens    = selected->placements[seq_idx];
                ctx.block_location.clear();
                ctx.sp_block_table.assign(attention_sp_, {});
            }

            int failure_after                                = ls_admission_failure_after_allocations_for_test_;
            ls_admission_failure_after_allocations_for_test_ = -1;
            worker_state[dp_idx]->allocate_ls_initial_batch(batch.sequences, failure_after);
            physical_allocation_committed = true;

            int failure_after_publications                    = ls_admission_failure_after_publications_for_test_;
            ls_admission_failure_after_publications_for_test_ = -1;
            if (failure_after_publications == 0) {
                throw std::runtime_error("injected LS admission publication failure");
            }

            if (!is_merge) {
                DecodeGroupState group;
                group.group_id = group_id;
                group.dp_idx   = dp_idx;
                ls_groups_.emplace(group_id, std::move(group));
                ls_group_ids_by_dp_[dp_idx].push_back(group_id);
                std::sort(ls_group_ids_by_dp_[dp_idx].begin(), ls_group_ids_by_dp_[dp_idx].end());
            }
            auto& group = ls_groups_.at(group_id);

            InitialBatchPlacement record;
            record.batch_id         = batch.batch_id;
            record.group_id         = group_id;
            record.initial_kv_dop   = static_cast<int>(selected->ranks.size());
            record.initial_kv_ranks = selected->ranks;
            record.prompt_kv_tokens = selected->placements;
            record.provisional_pending_targets.assign(batch.sequences.size(), selected->ranks.front());
            record.admission_order    = next_ls_admission_order_;
            record.admission_attempts = batch.admission_attempts;
            record.is_recovery_batch  = batch.is_recovery_batch;
            record.parent_batch_id    = batch.parent_batch_id;
            record.admission_kind     = batch.is_recovery_batch ? "recovery" : (is_merge ? "merge" : "standalone");

            int published_sequences = 0;
            for (const auto& seq : batch.sequences) {
                seq->status = SequenceStatus::RUNNING;
                worker_state[dp_idx]->running.push_back(seq);
                group.sequences.push_back(seq);
                ls_seq_to_group_[seq->seq_id] = group_id;
                record.sequence_ids.push_back(seq->seq_id);
                published_sequences++;
                if (failure_after_publications >= 0 && published_sequences == failure_after_publications) {
                    throw std::runtime_error("injected LS admission publication failure");
                }
            }
            for (int rank : selected->ranks) {
                if (std::find(group.allocated_attention_ranks.begin(), group.allocated_attention_ranks.end(), rank)
                    == group.allocated_attention_ranks.end()) {
                    group.allocated_attention_ranks.push_back(rank);
                }
            }
            if (!is_merge || selected->new_allocation_ranks > 0) {
                group.last_scale_up_step = ls_schedule_step_;
            }
            group.kv_candidate_target_dop   = -1;
            group.kv_candidate_stable_steps = 0;
            group.initial_batch_placements.push_back(record);
            ls_step_initial_records_.push_back(record);

            batch.state         = PendingDecodeBatchState::ADMITTED;
            scheduled[dp_idx]   = batch.sequences;
            dp_admitted[dp_idx] = true;
            next_ls_admission_order_++;
            if (!is_merge) {
                next_ls_group_id_++;
            }
            else {
                ls_step_atomic_merge_count_++;
            }
            for (const auto& seq : batch.sequences) {
                if (seq->metric) {
                    seq->metric->record_first_scheduled();
                    seq->metric->record_decode_scheduled();
                }
            }
            ls_pending_decode_batches_.erase(pending_it);
        }
        catch (const std::exception& error) {
            bool injected_failure = std::string_view(error.what()).find("injected LS") != std::string_view::npos;
            if (physical_allocation_committed) {
                for (const auto& seq : batch.sequences) {
                    worker_state[dp_idx]->deallocate(*seq, BlockContextSlot::ACTIVE);
                }
            }
            auto& running_queue = worker_state[dp_idx]->running;
            for (size_t seq_idx = 0; seq_idx < batch.sequences.size(); ++seq_idx) {
                const auto& seq = batch.sequences[seq_idx];
                running_queue.erase(std::remove(running_queue.begin(), running_queue.end(), seq), running_queue.end());
                ls_seq_to_group_.erase(seq->seq_id);
                seq->block_ctx(BlockContextSlot::ACTIVE) = std::move(original_contexts[seq_idx]);
                seq->status                              = SequenceStatus::WAITING;
            }
            if (is_merge) {
                ls_groups_[group_id] = std::move(*original_group);
            }
            else {
                ls_groups_.erase(group_id);
                auto& group_ids = ls_group_ids_by_dp_[dp_idx];
                group_ids.erase(std::remove(group_ids.begin(), group_ids.end(), group_id), group_ids.end());
            }
            ls_step_initial_records_.erase(
                std::remove_if(ls_step_initial_records_.begin(),
                               ls_step_initial_records_.end(),
                               [&](const auto& record) { return record.batch_id == batch.batch_id; }),
                ls_step_initial_records_.end());
            scheduled[dp_idx].clear();
            batch.state = PendingDecodeBatchState::QUEUED;
            ls_step_atomic_rollback_count_++;
            std::cerr << "LS atomic admission rolled back batch=" << batch.batch_id << ": " << error.what()
                      << std::endl;
            if (!injected_failure) {
                throw;
            }
        }
    }
    return scheduled;
}

std::vector<std::vector<std::shared_ptr<Sequence>>> Scheduler::_schedule_ls_decode()
{
    auto                                                planning_start = std::chrono::steady_clock::now();
    std::vector<std::vector<std::shared_ptr<Sequence>>> scheduled(attention_dp_);
    _reconcile_ls_groups();

    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        bool planned     = false;
        int  retry_limit = attention_sp_ + 1;
        while (!planned && retry_limit-- > 0) {
            auto group_ids = ls_group_ids_by_dp_[dp_idx];
            std::sort(group_ids.begin(), group_ids.end());
            if (group_ids.empty()) {
                planned = true;
                break;
            }

            // Reclaim only ranks that hold neither committed KV nor a pending
            // token for this group. No historical KV is moved.
            for (uint64_t group_id : group_ids) {
                auto& group = ls_groups_.at(group_id);
                auto  used  = worker_state[dp_idx]->group_used_kv_tokens(group.sequences);
                group.allocated_attention_ranks.erase(
                    std::remove_if(group.allocated_attention_ranks.begin(),
                                   group.allocated_attention_ranks.end(),
                                   [&](int rank) {
                                       bool pending = std::any_of(
                                           group.sequences.begin(), group.sequences.end(), [&](const auto& seq) {
                                               const auto& ctx = seq->block_ctx(BlockContextSlot::ACTIVE);
                                               return ctx.pending_token_present_
                                                      && ctx.pending_token_target_sp_ == rank;
                                           });
                                       return used[rank] == 0 && !pending;
                                   }),
                    group.allocated_attention_ranks.end());
            }

            // If ranks assigned to other groups are the only reason a group
            // cannot reach its threshold-sized compute demand, merge with the
            // oldest compatible group before generating any iteration plan.
            int  globally_unallocated = static_cast<int>(_ls_unallocated_ranks(dp_idx).size());
            bool merged_for_compute   = false;
            for (uint64_t group_id : group_ids) {
                const auto& group = ls_groups_.at(group_id);
                int         real_batch =
                    static_cast<int>(std::count_if(group.sequences.begin(), group.sequences.end(), [](const auto& seq) {
                        return seq && seq->status == SequenceStatus::RUNNING;
                    }));
                int desired = std::min(attention_sp_,
                                       (real_batch + ls_decode_batch_per_master_ - 1) / ls_decode_batch_per_master_);
                int directly_available =
                    static_cast<int>(group.allocated_attention_ranks.size()) + globally_unallocated;
                if (desired > directly_available && group_ids.size() > 1) {
                    uint64_t merge_with = group_ids.front() == group_id ? group_ids[1] : group_ids.front();
                    _merge_ls_groups(group_id, merge_with);
                    merged_for_compute = true;
                    break;
                }
            }
            if (merged_for_compute) {
                continue;
            }

            std::unordered_set<int> owned;
            for (uint64_t group_id : group_ids) {
                for (int rank : ls_groups_.at(group_id).allocated_attention_ranks) {
                    owned.insert(rank);
                }
            }

            struct PendingPlan {
                uint64_t                               group_id;
                std::vector<std::shared_ptr<Sequence>> requests;
                SPStateManager::LSDecodeMasterPlan     plan;
                std::vector<int>                       reused_passive;
            };
            std::vector<PendingPlan> pending_plans;
            std::optional<uint64_t>  failed_group;
            std::string              failure_reason;

            for (uint64_t group_id : group_ids) {
                auto&                                  group = ls_groups_.at(group_id);
                std::vector<std::shared_ptr<Sequence>> requests;
                for (const auto& seq : group.sequences) {
                    if (seq && seq->status == SequenceStatus::RUNNING) {
                        requests.push_back(seq);
                    }
                }
                if (requests.empty()) {
                    continue;
                }

                std::vector<int> extras;
                for (int rank = 0; rank < attention_sp_; ++rank) {
                    if (!owned.count(rank)) {
                        extras.push_back(rank);
                    }
                }
                auto plan =
                    worker_state[dp_idx]->plan_iteration_masters_source_greedy(requests,
                                                                               group.allocated_attention_ranks,
                                                                               extras,
                                                                               ls_decode_batch_per_master_,
                                                                               ls_decode_enable_memory_scale_up_);
                std::string validation_error;
                if (!plan.success
                    || !worker_state[dp_idx]->validate_iteration_master_plan(requests, plan, &validation_error)) {
                    failed_group   = group_id;
                    failure_reason = plan.success ? validation_error : plan.failure_reason;
                    break;
                }
                for (int rank : plan.allocation) {
                    owned.insert(rank);
                }

                std::vector<int> reused_passive;
                for (int rank : plan.master_ranks) {
                    bool was_last_master =
                        std::find(group.last_iteration_masters.begin(), group.last_iteration_masters.end(), rank)
                        != group.last_iteration_masters.end();
                    if (!was_last_master && plan.group_used_kv_tokens[rank] > 0
                        && std::find(
                               group.allocated_attention_ranks.begin(), group.allocated_attention_ranks.end(), rank)
                               != group.allocated_attention_ranks.end()) {
                        reused_passive.push_back(rank);
                    }
                }
                pending_plans.push_back({group_id, std::move(requests), std::move(plan), std::move(reused_passive)});
            }

            if (failed_group.has_value()) {
                if (group_ids.size() > 1) {
                    uint64_t merge_with = group_ids.front() == *failed_group ? group_ids[1] : group_ids.front();
                    _merge_ls_groups(*failed_group, merge_with);
                    continue;
                }

                std::cerr << "LS-Decode-Core plan failed for group=" << *failed_group << ": " << failure_reason
                          << std::endl;
                auto& victims = ls_groups_.at(*failed_group).sequences;
                auto  victim  = std::find_if(victims.rbegin(), victims.rend(), [](const auto& seq) {
                    return seq && seq->status == SequenceStatus::RUNNING;
                });
                if (victim != victims.rend()) {
                    ls_step_preempted_sequence_ids_.push_back((*victim)->seq_id);
                    ls_step_preemption_reasons_.push_back(failure_reason);
                    preempt(dp_idx, *victim);
                }
                _reconcile_ls_groups();
                continue;
            }

            std::vector<int> master_load(attention_sp_, 0);
            for (auto& pending : pending_plans) {
                auto& group = ls_groups_.at(pending.group_id);
                if (!worker_state[dp_idx]->commit_iteration_master_plan(pending.requests, pending.plan)) {
                    throw std::runtime_error("validated LS iteration plan commit failed");
                }
                for (size_t seq_idx = 0; seq_idx < pending.requests.size(); ++seq_idx) {
                    int master = pending.plan.sequence_master_ranks[seq_idx];
                    scheduled[dp_idx].push_back(pending.requests[seq_idx]);
                    master_load[master]++;
                }
                bool allocation_grew = pending.plan.allocation.size() > group.allocated_attention_ranks.size();
                group.allocated_attention_ranks = pending.plan.allocation;
                group.last_iteration_masters    = pending.plan.master_ranks;
                if (allocation_grew) {
                    group.last_scale_up_step        = ls_schedule_step_;
                    group.kv_candidate_target_dop   = -1;
                    group.kv_candidate_stable_steps = 0;
                }
                ls_step_group_plan_ids_.push_back(pending.group_id);
                ls_step_group_plans_.push_back(pending.plan);
                ls_step_reused_passive_masters_.push_back(pending.reused_passive);
            }
            for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
                if (master_load[sp_idx] == 0) {
                    scheduled[dp_idx].push_back(worker_state[dp_idx]->dummy_seqs[sp_idx]);
                }
            }
            planned = true;
        }
    }

    // Every DP must enter the fixed EP/SP collective cadence, including a
    // DP with no live Decode group. Fill missing master ranks uniformly here
    // so no early-exit or failure path can return an empty worker batch.
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        std::vector<bool> has_master(attention_sp_, false);
        for (const auto& seq : scheduled[dp_idx]) {
            int master = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
            if (master >= 0 && master < attention_sp_) {
                has_master[master] = true;
            }
        }
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            if (!has_master[sp_idx]) {
                scheduled[dp_idx].push_back(worker_state[dp_idx]->dummy_seqs[sp_idx]);
            }
        }
    }

    ls_step_planning_latency_ms_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - planning_start).count();
    return scheduled;
}

std::vector<std::vector<std::shared_ptr<Sequence>>> Scheduler::_schedule_prefill()
{
    if (scheduler_mode_ == SchedulerMode::CENTRALIZED && mode_ == "decode" && enable_dynamic_sp_size_
        && use_new_decode_dynamic_sp_scheduler_ && routing_strategy == RoutingStrategy::LeastBatch) {
        return _schedule_decode_prefill_latency_aware();
    }

    std::vector<std::vector<std::shared_ptr<Sequence>>> scheduled_seqs(attention_dp_);

    // num_seqs and num_batched_tokens track per-DP, per-SP-rank counts for the CURRENT batch
    std::vector<std::unordered_map<int, int>> num_seqs(attention_dp_);
    std::vector<std::unordered_map<int, int>> num_batched_tokens(attention_dp_);

    // Initialize with default values of 0
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            num_seqs[dp_idx][sp_idx]           = 0;
            num_batched_tokens[dp_idx][sp_idx] = 0;
        }
    }

    auto& waiting_queue = (mode_ != "decode") ? waiting : waiting_migration;

    // For LeastBatch and LeastCache, we maintain a set to act as a min-heap
    std::set<std::pair<int, int>> dp_load_set;
    if (routing_strategy == RoutingStrategy::LeastBatch) {
        for (int i = 0; i < attention_dp_; ++i) {
            dp_load_set.insert({worker_state[i]->num_running_seqs(), i});
        }
    }
    else if (routing_strategy == RoutingStrategy::LeastCache) {
        for (int i = 0; i < attention_dp_; ++i) {
            dp_load_set.insert({worker_state[i]->num_running_tokens(), i});
        }
    }

    while (!waiting_queue.empty()) {
        auto seq       = waiting_queue.front();
        bool scheduled = false;

        if (routing_strategy == RoutingStrategy::RoundRobin) {
            // Try all DP ranks in round-robin order
            for (int attempt = 0; attempt < attention_dp_; ++attempt) {
                int selected_dp_idx = next_dp_idx();

                // Check if this DP rank can allocate the sequence
                bool can_allocate = worker_state[selected_dp_idx]->can_allocate(
                    *seq, num_seqs[selected_dp_idx], num_batched_tokens[selected_dp_idx]);

                if (!can_allocate) {
                    continue;
                }

                // Allocate the sequence
                worker_state[selected_dp_idx]->allocate(*seq);

                // Update tracking
                auto& block_ctx   = seq->block_ctx(BlockContextSlot::ACTIVE);
                block_ctx.dp_idx_ = selected_dp_idx;
                int master_sp_idx = block_ctx.master_sp_idx_;

                num_seqs[selected_dp_idx][master_sp_idx] += 1;
                num_batched_tokens[selected_dp_idx][master_sp_idx] += (seq->num_tokens - seq->num_cached_tokens);

                // Update sequence status
                seq->status = SequenceStatus::RUNNING;

                // Add to scheduled and running queues
                waiting_queue.pop_front();
                worker_state[selected_dp_idx]->running.push_back(seq);
                scheduled_seqs[selected_dp_idx].push_back(seq);

                // Record metrics
                if (seq->metric) {
                    seq->metric->record_first_scheduled();
                    if (mode_ == "decode") {
                        seq->metric->record_decode_scheduled();
                    }
                }

                scheduled = true;
                break;
            }
        }
        else if (routing_strategy == RoutingStrategy::LeastBatch || routing_strategy == RoutingStrategy::LeastCache) {
            // Iterate through DP ranks in increasing order of load
            for (auto it = dp_load_set.begin(); it != dp_load_set.end(); ++it) {
                int selected_dp_idx = it->second;

                bool can_allocate = worker_state[selected_dp_idx]->can_allocate(
                    *seq, num_seqs[selected_dp_idx], num_batched_tokens[selected_dp_idx]);

                if (!can_allocate) {
                    continue;
                }

                // WARNING: erase(it) invalidates the iterator. This is safe here because
                // we break the loop immediately after. If refactoring to remove the break
                // or making dp_load_set a member variable, ensure thread-safety and
                // correct iterator management.
                dp_load_set.erase(it);
                worker_state[selected_dp_idx]->allocate(*seq);
                int new_load = (routing_strategy == RoutingStrategy::LeastBatch) ?
                                   worker_state[selected_dp_idx]->num_running_seqs() :
                                   worker_state[selected_dp_idx]->num_running_tokens();
                dp_load_set.insert({new_load, selected_dp_idx});

                auto& block_ctx   = seq->block_ctx(BlockContextSlot::ACTIVE);
                block_ctx.dp_idx_ = selected_dp_idx;
                int master_sp_idx = block_ctx.master_sp_idx_;

                num_seqs[selected_dp_idx][master_sp_idx] += 1;
                num_batched_tokens[selected_dp_idx][master_sp_idx] += (seq->num_tokens - seq->num_cached_tokens);

                seq->status = SequenceStatus::RUNNING;

                waiting_queue.pop_front();
                worker_state[selected_dp_idx]->running.push_back(seq);
                scheduled_seqs[selected_dp_idx].push_back(seq);

                if (seq->metric) {
                    seq->metric->record_first_scheduled();
                    if (mode_ == "decode") {
                        seq->metric->record_decode_scheduled();
                    }
                }

                scheduled = true;
                break;
            }
        }
        else {
            throw std::runtime_error("Unknown routing strategy");
        }

        if (!scheduled) {
            // Cannot schedule any more sequences
            break;
        }
    }

    return scheduled_seqs;
}

std::vector<std::vector<std::shared_ptr<Sequence>>> Scheduler::_schedule_decode()
{
    std::vector<std::vector<std::shared_ptr<Sequence>>> scheduled_seqs(attention_dp_);

    for (int selected_dp_idx = 0; selected_dp_idx < attention_dp_; ++selected_dp_idx) {
        auto& running_queue = worker_state[selected_dp_idx]->running;

        std::unordered_map<int, int>          num_seqs;
        std::deque<std::shared_ptr<Sequence>> skipped;
        std::vector<int>                      sp_lens(attention_sp_, 0);

        while (!running_queue.empty()) {
            auto seq = running_queue.front();
            running_queue.pop_front();

            int master_rank = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;

            // Check if we've reached the max sequences for this SP rank
            if (num_seqs[master_rank] >= max_num_seqs_) {
                skipped.push_back(seq);
                continue;
            }

            // Try to ensure we can append tokens
            while (!worker_state[selected_dp_idx]->can_append(*seq, loop_count_)) {
                // Need to preempt to free up space
                if (!running_queue.empty()) {
                    auto victim = running_queue.back();
                    running_queue.pop_back();
                    preempt(selected_dp_idx, victim);
                }
                else if (!skipped.empty()) {
                    auto victim = skipped.back();
                    skipped.pop_back();
                    preempt(selected_dp_idx, victim);
                }
                else {
                    // Preempt current sequence itself
                    preempt(selected_dp_idx, seq);
                    seq = nullptr;
                    break;
                }
            }

            if (seq) {
                // Successfully ensured space for this sequence
                num_seqs[master_rank] += 1;
                if (!worker_state[selected_dp_idx]->may_append(*seq, loop_count_)) {
                    // This should not happen if can_append is correct, but handle it gracefully
                    preempt(selected_dp_idx, seq);
                }
                else {
                    scheduled_seqs[selected_dp_idx].push_back(seq);
                    sp_lens[master_rank] += seq->num_tokens;
                }
            }
        }

        // Put skipped and scheduled sequences back to running queue
        for (auto it = scheduled_seqs[selected_dp_idx].rbegin(); it != scheduled_seqs[selected_dp_idx].rend(); ++it) {
            running_queue.push_front(*it);
        }
        for (auto it = skipped.rbegin(); it != skipped.rend(); ++it) {
            running_queue.push_front(*it);
        }

        // Add dummy sequences for SP ranks with no work
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            if (sp_lens[sp_idx] == 0) {
                scheduled_seqs[selected_dp_idx].push_back(worker_state[selected_dp_idx]->dummy_seqs[sp_idx]);
            }
        }
    }

    return scheduled_seqs;
}

void Scheduler::preempt(int dp_idx, std::shared_ptr<Sequence> seq)
{
    if (active_ls_kv_transaction_) {
        throw std::runtime_error("cannot preempt while an LS KV scale-down transaction is reserved");
    }
    std::cerr << "Preemption happens for seq_id=" << seq->seq_id << std::endl;

    if (enable_ls_decode_core_scheduler_) {
        auto group_owner = ls_seq_to_group_.find(seq->seq_id);
        if (group_owner == ls_seq_to_group_.end()) {
            throw std::runtime_error("cannot preempt an LS sequence that is not owned by a running group");
        }
        auto group = ls_groups_.find(group_owner->second);
        if (group == ls_groups_.end()) {
            throw std::runtime_error("cannot preempt an LS sequence whose running group is missing");
        }
        if (group->second.dp_idx != dp_idx) {
            throw std::runtime_error("cannot preempt an LS sequence from a DP that does not own its running group");
        }
        auto batch_owner = ls_seq_to_batch_.find(seq->seq_id);
        if (batch_owner == ls_seq_to_batch_.end()) {
            throw std::runtime_error("cannot preempt an LS sequence without an active batch owner");
        }

        auto& running_queue = worker_state[dp_idx]->running;
        if (std::find(running_queue.begin(), running_queue.end(), seq) == running_queue.end()) {
            throw std::runtime_error("cannot preempt an LS sequence that is absent from its DP running queue");
        }

        // Prepare every allocation that recovery publication needs before
        // mutating sequence, group, or block-manager state. Keep the existing
        // batch-owner map node alive so publishing the replacement owner below
        // cannot allocate.
        const uint64_t     recovery_batch_id      = next_ls_batch_id_;
        const uint64_t     recovery_enqueue_order = next_ls_enqueue_order_;
        PendingDecodeBatch recovery;
        recovery.batch_id          = recovery_batch_id;
        recovery.sequences         = {seq};
        recovery.enqueue_order     = recovery_enqueue_order;
        recovery.enqueue_step      = ls_schedule_step_;
        recovery.is_recovery_batch = true;
        recovery.parent_batch_id   = batch_owner->second;
        BlockContext fresh_context(engine_id_, attention_sp_, attention_dp_);

        ls_pending_decode_batches_.push_front(std::move(recovery));
        try {
            // Deallocate before resetting num_tokens so running-token
            // accounting uses the actual preempted length.
            worker_state[dp_idx]->deallocate(*seq);
        }
        catch (...) {
            if (ls_pending_decode_batches_.empty()
                || ls_pending_decode_batches_.front().batch_id != recovery_batch_id) {
                std::terminate();
            }
            ls_pending_decode_batches_.pop_front();
            throw;
        }

        _remove_seq_from_ls_group(seq->seq_id, false);
        running_queue.erase(std::remove(running_queue.begin(), running_queue.end(), seq), running_queue.end());

        int prompt_len = seq->num_prompt_tokens;
        seq->status    = SequenceStatus::WAITING;
        seq->token_ids.resize(prompt_len);
        seq->num_tokens                          = prompt_len;
        seq->num_checkpointed_tokens             = prompt_len;
        seq->last_token                          = seq->token_ids.empty() ? 0 : seq->token_ids.back();
        seq->block_ctx(BlockContextSlot::ACTIVE) = std::move(fresh_context);

        batch_owner->second = recovery_batch_id;
        ++next_ls_batch_id_;
        ++next_ls_enqueue_order_;

        if (seq->metric) {
            seq->metric->on_preemption();
        }
        return;
    }

    // Reset metrics for fresh start
    if (seq->metric) {
        seq->metric->on_preemption();
    }

    // Deallocate before resetting num_tokens so running-token accounting uses
    // the actual preempted length.
    int prompt_len = seq->num_prompt_tokens;
    seq->status    = SequenceStatus::WAITING;
    worker_state[dp_idx]->deallocate(*seq);

    // Reset sequence to prompt-only state (discard generated tokens)
    seq->token_ids.resize(prompt_len);
    seq->num_tokens              = prompt_len;
    seq->num_checkpointed_tokens = prompt_len;
    seq->last_token              = seq->token_ids.empty() ? 0 : seq->token_ids.back();
    // Re-initialize BlockContext for fresh scheduling
    seq->active(engine_id_, attention_sp_, attention_dp_);

    if (scheduler_mode_ == SchedulerMode::DECENTRALIZED) {
        // Decentralized mode: put back to the worker's queue
        auto& target_queue =
            (mode_ == "decode") ? worker_state[dp_idx]->waiting_migration : worker_state[dp_idx]->waiting;
        target_queue.push_front(seq);
        // Keep the dp_idx that was set during routing
        seq->block_ctx(BlockContextSlot::ACTIVE).dp_idx_ = dp_idx;
    }
    else {
        // Centralized mode: put back to global queue
        if (mode_ == "decode") {
            waiting_migration.push_front(seq);
        }
        else {
            waiting.push_front(seq);
        }
    }
}

void Scheduler::postprocess(const std::vector<std::vector<std::shared_ptr<Sequence>>>& dp_sp_seqs,
                            const std::vector<std::vector<std::vector<int>>>&          dp_sp_token_ids,
                            bool                                                       update_metrics,
                            double                                                     accumulated_step_time_ms,
                            int                                                        loop_count)
{
    if (active_ls_kv_transaction_) {
        throw std::runtime_error("cannot postprocess while an LS KV scale-down transaction is reserved");
    }
    // Call the C++ postprocess_sequences utility directly with shared_ptrs
    auto migrations = postprocess_sequences(worker_state,
                                            dp_sp_seqs,
                                            dp_sp_token_ids,
                                            eos_,
                                            mode_ == "prefill",
                                            update_metrics,
                                            accumulated_step_time_ms,
                                            loop_count,
                                            thread_pool_.get());

    // Store migrations
    for (const auto& [seq_shared, dp_idx] : migrations) {
        to_be_migrated[seq_shared->seq_id] = {seq_shared, dp_idx};
    }
    if (enable_ls_decode_core_scheduler_) {
        _reconcile_ls_groups();
    }
}

void Scheduler::free_to_be_migrated(std::shared_ptr<Sequence> seq)
{
    if (active_ls_kv_transaction_) {
        throw std::runtime_error("cannot free migration KV while an LS KV scale-down transaction is reserved");
    }
    auto it = to_be_migrated.find(seq->seq_id);
    if (it == to_be_migrated.end()) {
        throw std::runtime_error("Sequence " + std::to_string(seq->seq_id) + " not found in to_be_migrated");
    }

    int selected_dp_idx = it->second.second;
    worker_state[selected_dp_idx]->deallocate(*seq, BlockContextSlot::MIGRATE);
    to_be_migrated.erase(it);
}

void Scheduler::free_to_be_migrated(const std::vector<std::shared_ptr<Sequence>>& seqs)
{
    for (const auto& seq : seqs) {
        free_to_be_migrated(seq);
    }
}

ScheduleResult Scheduler::_schedule_decentralized()
{
    std::vector<std::vector<std::shared_ptr<Sequence>>> scheduled_seqs(attention_dp_);
    bool                                                has_prefill = false;

    // Each DP worker independently schedules its own queue
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        // Try prefill first (from waiting queue or waiting_migration queue)
        // This should always be attempted, regardless of mode_
        // _schedule_prefill_for_worker will select the correct queue based on mode_
        scheduled_seqs[dp_idx] = _schedule_prefill_for_worker(dp_idx);
        if (!scheduled_seqs[dp_idx].empty()) {
            has_prefill = true;
        }

        // If no prefill, schedule decode
        if (scheduled_seqs[dp_idx].empty()) {
            scheduled_seqs[dp_idx] = _schedule_decode_for_worker(dp_idx);
        }
    }

    ScheduleResult result;
    result.dp_seqs    = scheduled_seqs;
    result.is_prefill = has_prefill;
    result.action     = has_prefill ? ScheduleAction::ADMISSION : ScheduleAction::DECODE;

    // Prepare dp_sp_seqs and filtered_dp_sp_seqs (same as centralized mode)
    result.dp_sp_seqs.reserve(attention_dp_ * attention_sp_);
    result.filtered_dp_sp_seqs.reserve(attention_dp_ * attention_sp_);

    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            // dp_sp_seqs is just dp_seqs[dp_idx] repeated for each sp_idx
            result.dp_sp_seqs.push_back(scheduled_seqs[dp_idx]);

            // filtered_dp_sp_seqs is dp_seqs[dp_idx] filtered by master_sp_idx
            std::vector<std::shared_ptr<Sequence>> filtered;
            for (const auto& seq : scheduled_seqs[dp_idx]) {
                if (seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_ == sp_idx) {
                    filtered.push_back(seq);
                }
            }
            result.filtered_dp_sp_seqs.push_back(std::move(filtered));
        }
    }

    // Calculate SP counts (same as centralized mode)
    result.sp_send_counts.resize(attention_dp_);
    result.sp_recv_counts.resize(attention_dp_);
    result.sp_size_hist_per_dp.resize(attention_dp_);
    result.sp_res_matrix.clear();

    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        result.sp_send_counts[dp_idx].resize(attention_sp_);
        result.sp_recv_counts[dp_idx].resize(attention_sp_);
        result.sp_size_hist_per_dp[dp_idx].assign(attention_sp_ + 1, 0);

        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            // SP Send Count
            int         send_count = 0;
            const auto& sp_seqs    = result.filtered_dp_sp_seqs[dp_idx * attention_sp_ + sp_idx];
            for (const auto& seq : sp_seqs) {
                if (has_remote_committed_kv(*seq, attention_sp_)) {
                    send_count++;
                }
            }
            result.sp_send_counts[dp_idx][sp_idx] = send_count;

            // SP Recv Count
            int recv_count = 0;
            for (const auto& seq : scheduled_seqs[dp_idx]) {
                bool is_dummy = false;
                for (const auto& dummy : worker_state[dp_idx]->dummy_seqs) {
                    if (seq == dummy) {
                        is_dummy = true;
                        break;
                    }
                }
                if (!is_dummy) {
                    const auto& block_ctx     = seq->block_ctx(BlockContextSlot::ACTIVE);
                    int         master_sp_idx = block_ctx.master_sp_idx_;
                    if (seq->committed_context_len(BlockContextSlot::ACTIVE, sp_idx) > 0 && master_sp_idx != sp_idx) {
                        recv_count++;
                    }
                }
            }
            result.sp_recv_counts[dp_idx][sp_idx] = recv_count;
        }

        for (const auto& seq : scheduled_seqs[dp_idx]) {
            bool is_dummy = false;
            for (const auto& dummy : worker_state[dp_idx]->dummy_seqs) {
                if (seq == dummy) {
                    is_dummy = true;
                    break;
                }
            }
            if (is_dummy) {
                continue;
            }

            int active_ranks = committed_kv_rank_count(*seq, attention_sp_);
            if (active_ranks >= 0 && active_ranks <= attention_sp_) {
                result.sp_size_hist_per_dp[dp_idx][active_ranks]++;
            }
        }

        // SP Q Matrix
        result.sp_q_matrix.push_back(std::vector<std::vector<int>>(attention_sp_, std::vector<int>(attention_sp_, 0)));
        result.sp_res_matrix.push_back(
            std::vector<std::vector<int>>(attention_sp_, std::vector<int>(attention_sp_, 0)));

        for (const auto& seq : scheduled_seqs[dp_idx]) {
            bool is_dummy = false;
            for (const auto& dummy : worker_state[dp_idx]->dummy_seqs) {
                if (seq == dummy) {
                    is_dummy = true;
                    break;
                }
            }
            if (is_dummy)
                continue;

            if (has_remote_committed_kv(*seq, attention_sp_)) {
                int master_sp_idx = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
                for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
                    if (seq->committed_context_len(BlockContextSlot::ACTIVE, sp_idx) > 0 && sp_idx != master_sp_idx) {
                        result.sp_q_matrix[dp_idx][master_sp_idx][sp_idx]++;
                        result.sp_res_matrix[dp_idx][sp_idx][master_sp_idx]++;
                    }
                }
            }
        }
    }

    // Calculate waiting queue block metrics (per-DP)
    result.waiting_head_blocks.resize(attention_dp_, 0);
    result.waiting_total_blocks.resize(attention_dp_, 0);

    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        auto& worker     = worker_state[dp_idx];
        auto& wait_queue = (mode_ != "decode") ? worker->waiting : worker->waiting_migration;

        if (!wait_queue.empty()) {
            auto head_seq = wait_queue.front();
            result.waiting_head_blocks[dp_idx] =
                (head_seq->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
        }

        for (const auto& seq : wait_queue) {
            result.waiting_total_blocks[dp_idx] += (seq->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
        }
    }

    return result;
}

std::vector<std::shared_ptr<Sequence>> Scheduler::_schedule_prefill_for_worker(int dp_idx)
{
    std::vector<std::shared_ptr<Sequence>> scheduled_seqs;
    auto&                                  worker = worker_state[dp_idx];
    auto& waiting_queue                           = (mode_ != "decode") ? worker->waiting : worker->waiting_migration;

    std::unordered_map<int, int> num_seqs;
    std::unordered_map<int, int> num_batched_tokens;

    // Initialize counts
    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        num_seqs[sp_idx]           = 0;
        num_batched_tokens[sp_idx] = 0;
    }

    // Schedule from this worker's queue
    while (!waiting_queue.empty()) {
        auto seq = waiting_queue.front();

        // Check if can allocate
        bool can_allocate = worker->can_allocate(*seq, num_seqs, num_batched_tokens);

        if (!can_allocate) {
            break;  // Cannot allocate more, stop scheduling
        }

        // Allocate resources
        worker->allocate(*seq);

        // Update counts
        auto& block_ctx     = seq->block_ctx(BlockContextSlot::ACTIVE);
        int   master_sp_idx = block_ctx.master_sp_idx_;
        num_seqs[master_sp_idx] += 1;
        num_batched_tokens[master_sp_idx] += (seq->num_tokens - seq->num_cached_tokens);

        // Move to running queue
        seq->status = SequenceStatus::RUNNING;
        waiting_queue.pop_front();
        worker->running.push_back(seq);
        scheduled_seqs.push_back(seq);

        // Record metrics
        if (seq->metric) {
            seq->metric->record_first_scheduled();
            if (mode_ == "decode") {
                seq->metric->record_decode_scheduled();
            }
        }
    }

    return scheduled_seqs;
}

std::vector<std::shared_ptr<Sequence>> Scheduler::_schedule_decode_for_worker(int dp_idx)
{
    std::vector<std::shared_ptr<Sequence>> scheduled_seqs;
    auto&                                  worker        = worker_state[dp_idx];
    auto&                                  running_queue = worker->running;

    std::unordered_map<int, int>          num_seqs;
    std::deque<std::shared_ptr<Sequence>> skipped;
    std::vector<int>                      sp_lens(attention_sp_, 0);

    while (!running_queue.empty()) {
        auto seq = running_queue.front();
        running_queue.pop_front();

        int master_rank = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;

        // Check if we've reached the max sequences for this SP rank
        if (num_seqs[master_rank] >= max_num_seqs_) {
            skipped.push_back(seq);
            continue;
        }

        // Try to ensure we can append tokens
        while (!worker->can_append(*seq, loop_count_)) {
            // Need to preempt to free up space
            if (!running_queue.empty()) {
                auto victim = running_queue.back();
                running_queue.pop_back();
                preempt(dp_idx, victim);
            }
            else if (!skipped.empty()) {
                auto victim = skipped.back();
                skipped.pop_back();
                preempt(dp_idx, victim);
            }
            else {
                // Preempt current sequence itself
                preempt(dp_idx, seq);
                seq = nullptr;
                break;
            }
        }

        if (seq) {
            // Successfully ensured space for this sequence
            num_seqs[master_rank] += 1;
            if (!worker->may_append(*seq, loop_count_)) {
                // This should not happen if can_append is correct, but handle it gracefully
                preempt(dp_idx, seq);
            }
            else {
                scheduled_seqs.push_back(seq);
                sp_lens[master_rank] += seq->num_tokens;
            }
        }
    }

    // Put skipped and scheduled sequences back to running queue
    for (auto it = scheduled_seqs.rbegin(); it != scheduled_seqs.rend(); ++it) {
        running_queue.push_front(*it);
    }
    for (auto it = skipped.rbegin(); it != skipped.rend(); ++it) {
        running_queue.push_front(*it);
    }

    // Add dummy sequences for SP ranks with no work
    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        if (sp_lens[sp_idx] == 0) {
            scheduled_seqs.push_back(worker->dummy_seqs[sp_idx]);
        }
    }

    return scheduled_seqs;
}

}  // namespace nanodeploy
