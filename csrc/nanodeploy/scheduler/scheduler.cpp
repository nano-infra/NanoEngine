#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <utility>

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
                     int                ls_kv_consolidation_max_source_blocks_per_event,
                     bool               ls_decode_enable_future_kv_admission,
                     int                ls_max_num_ooe,
                     int                ls_running_max_req_size,
                     int                ls_admission_max_tokens_per_pool,
                     int                ls_min_comp_bound_decoding_batch_size):
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
    ls_decode_enable_future_kv_admission_(ls_decode_enable_future_kv_admission),
    ls_max_num_ooe_(ls_max_num_ooe),
    ls_running_max_req_size_(ls_running_max_req_size),
    ls_admission_max_tokens_per_pool_(ls_admission_max_tokens_per_pool),
    ls_min_comp_bound_decoding_batch_size_(ls_min_comp_bound_decoding_batch_size),
    ls_kv_consolidation_mode_(ls_kv_consolidation_mode),
    ls_kv_consolidation_candidate_util_(ls_kv_consolidation_candidate_util),
    ls_kv_consolidation_target_high_watermark_(ls_kv_consolidation_target_high_watermark),
    ls_kv_consolidation_stable_steps_(ls_kv_consolidation_stable_steps),
    ls_kv_consolidation_cooldown_steps_(ls_kv_consolidation_cooldown_steps),
    ls_kv_consolidation_check_interval_steps_(ls_kv_consolidation_check_interval_steps),
    ls_kv_consolidation_max_source_blocks_per_event_(ls_kv_consolidation_max_source_blocks_per_event),
    sp_master_selector_(sp_master_selector)
{
    if (ls_max_num_ooe_ < 0 || ls_running_max_req_size_ <= 0 || ls_admission_max_tokens_per_pool_ < 0
        || ls_min_comp_bound_decoding_batch_size_ <= 0) {
        throw std::invalid_argument("invalid LoongServe-style Decode-only admission configuration");
    }
    if (enable_ls_decode_core_scheduler_ && (loop_count_ <= 0 || loop_count_ >= kvcache_block_size)) {
        throw std::invalid_argument(
            "LoongServe-style Decode loop_count must be positive and smaller than kvcache_block_size");
    }
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
    ls_empty_system_free_blocks_per_rank_.assign(attention_dp_, std::vector<int>(attention_sp_, 0));
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            ls_empty_system_free_blocks_per_rank_[dp_idx][sp_idx] =
                worker_state[dp_idx]->block_manager.at(sp_idx)->num_free_blocks();
        }
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
    ls_waiting_by_dp_.resize(attention_dp_);
    ls_num_ooe_.assign(attention_dp_, 0);
    ls_step_real_decode_ids_by_dp_.resize(attention_dp_);
    ls_pool_resource_epoch_.assign(attention_dp_, 0);
    ls_step_pool_resource_epoch_before_.assign(attention_dp_, 0);
    ls_step_pool_resource_mutated_.assign(attention_dp_, false);
    if (ls_admission_max_tokens_per_pool_ == 0) {
        const int64_t auto_limit = std::max<int64_t>(
            max_num_batched_tokens_, _ls_pool_token_capacity(0) / 6);
        ls_admission_max_tokens_per_pool_ =
            static_cast<int>(std::min<int64_t>(auto_limit, std::numeric_limits<int>::max()));
    }
}

LSAddResult Scheduler::precheck_add_identity(const std::shared_ptr<Sequence>& seq) const
{
    if (!seq) {
        return {false, -1, LSAddError::ALREADY_ADDED_OR_ASSIGNED, "sequence is null"};
    }
    if (!enable_ls_decode_core_scheduler_) {
        return {true, -1, LSAddError::NONE, {}};
    }
    if (seq->assigned_dp != -1) {
        return {false,
                seq->assigned_dp,
                LSAddError::ALREADY_ADDED_OR_ASSIGNED,
                "sequence is already assigned to a LoongServe-style DP pool"};
    }
    if (seen_ls_seq_ids_.count(seq->seq_id)) {
        return {false,
                -1,
                LSAddError::ALREADY_ADDED_OR_ASSIGNED,
                "sequence ID has already been used by this scheduler"};
    }
    return {true, -1, LSAddError::NONE, {}};
}

LSAddResult Scheduler::add(std::shared_ptr<Sequence> seq)
{
    if (ls_fatal_.has_value()) {
        throw LSSchedulerFatalError(*ls_fatal_, "LoongServe-style scheduler is permanently fatal");
    }
    if (active_ls_kv_transaction_) {
        throw std::runtime_error("cannot add a sequence while an LS KV scale-down transaction is reserved");
    }

    LSAddResult identity = precheck_add_identity(seq);
    if (!identity.accepted) {
        return identity;
    }

    if (enable_ls_decode_core_scheduler_) {
        const int assigned_dp = next_ls_dp_rr_;
        LSAddResult result{true, assigned_dp, LSAddError::NONE, {}};

        if (!seq->ignore_eos) {
            result = {false,
                      assigned_dp,
                      LSAddError::IGNORE_EOS_REQUIRED,
                      "LoongServe-style Decode-only requires ignore_eos=true"};
        }
        else if (seq->max_tokens < 1) {
            result = {false,
                      assigned_dp,
                      LSAddError::INVALID_MAX_TOKENS,
                      "LoongServe-style Decode-only requires max_tokens >= 1"};
        }
        else if (_ls_admission_need_tokens(*seq) > ls_admission_max_tokens_per_pool_
                 || !_ls_future_kv_fits_empty_system(assigned_dp, {seq}, [&] {
                        std::vector<int> ranks(attention_sp_);
                        std::iota(ranks.begin(), ranks.end(), 0);
                        return ranks;
                    }())) {
            result = {false,
                      assigned_dp,
                      LSAddError::FUTURE_TOKEN_NO_FIT,
                      "request exceeds the fixed SP-pool future-token policy"};
        }
        else if (!_ls_batch_fits_empty_system(assigned_dp, {seq})) {
            result = {false,
                      assigned_dp,
                      LSAddError::CURRENT_EXACT_NO_FIT,
                      "request cannot fit the full empty SP pool exactly"};
        }

        // Prepare every allocating container operation before the ingress
        // publication boundary. Node-handle insertion and list splice below do
        // not allocate after the scheduler consumes the RR slot.
        std::unordered_set<uint64_t> prepared_seen_source;
        prepared_seen_source.insert(seq->seq_id);
        auto prepared_seen_node = prepared_seen_source.extract(seq->seq_id);
        seen_ls_seq_ids_.reserve(seen_ls_seq_ids_.size() + 1);

        std::list<std::shared_ptr<Sequence>> prepared_waiting_node;
        std::unordered_map<uint64_t, uint64_t> prepared_arrival_source;
        std::unordered_map<uint64_t, uint64_t>::node_type prepared_arrival_node;
        std::optional<BlockContext> prepared_active_context;
        if (result.accepted) {
            prepared_waiting_node.push_back(seq);
            prepared_arrival_source.emplace(seq->seq_id, next_ls_arrival_order_);
            prepared_arrival_node = prepared_arrival_source.extract(seq->seq_id);
            ls_arrival_order_by_seq_id_.reserve(ls_arrival_order_by_seq_id_.size() + 1);
            prepared_active_context.emplace(engine_id_, attention_sp_, attention_dp_);
            prepared_active_context->dp_idx_ = assigned_dp;
        }

        // Every structurally valid first attempt consumes exactly one RR slot,
        // including singleton validation rejection. Seen IDs intentionally live
        // until scheduler teardown.
        auto seen_insert = seen_ls_seq_ids_.insert(std::move(prepared_seen_node));
        if (!seen_insert.inserted) {
            std::terminate();
        }
        next_ls_dp_rr_ = (next_ls_dp_rr_ + 1) % attention_dp_;
        seq->assigned_dp = assigned_dp;
        if (!result.accepted) {
            return result;
        }

        std::swap(seq->block_ctx(BlockContextSlot::ACTIVE), *prepared_active_context);
        seq->status = SequenceStatus::WAITING;
        ls_waiting_by_dp_[assigned_dp].splice(ls_waiting_by_dp_[assigned_dp].end(), prepared_waiting_node);
        auto arrival_insert = ls_arrival_order_by_seq_id_.insert(std::move(prepared_arrival_node));
        if (!arrival_insert.inserted) {
            std::terminate();
        }
        ++next_ls_arrival_order_;

        return result;
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
    return {true, -1, LSAddError::NONE, {}};
}

bool Scheduler::is_finished() const
{
    if (enable_ls_decode_core_scheduler_) {
        for (const auto& queue : ls_waiting_by_dp_) {
            if (!queue.empty()) {
                return false;
            }
        }
        for (const auto& ws : worker_state) {
            if (!ws->is_empty()) {
                return false;
            }
        }
        return true;
    }
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
        for (const auto& ws : worker_state) {
            if (!ws->is_empty())
                return false;
        }
        return true;
    }
}

int Scheduler::get_total_waiting_size() const
{
    if (enable_ls_decode_core_scheduler_) {
        int total = 0;
        for (const auto& queue : ls_waiting_by_dp_) {
            total += static_cast<int>(queue.size());
        }
        return total;
    }
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
    if (enable_ls_decode_core_scheduler_) {
        int total = 0;
        for (const auto& queue : ls_waiting_by_dp_) {
            total += static_cast<int>(queue.size());
        }
        return total;
    }
    if (scheduler_mode_ == SchedulerMode::DECENTRALIZED) {
        int total = 0;
        for (const auto& ws : worker_state) {
            total += static_cast<int>(ws->waiting_migration.size());
        }
        return total;
    }
    else {
        return static_cast<int>(waiting_migration.size());
    }
}

std::vector<std::vector<uint64_t>> Scheduler::get_ls_waiting_sequence_ids_by_dp() const
{
    std::vector<std::vector<uint64_t>> result(attention_dp_);
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        result[dp_idx].reserve(ls_waiting_by_dp_[dp_idx].size());
        for (const auto& sequence : ls_waiting_by_dp_[dp_idx]) {
            if (sequence) {
                result[dp_idx].push_back(sequence->seq_id);
            }
        }
    }
    return result;
}

std::vector<int> Scheduler::get_ls_num_ooe() const
{
    return ls_num_ooe_;
}

std::vector<uint64_t> Scheduler::get_ls_pool_resource_epochs() const
{
    return ls_pool_resource_epoch_;
}

std::optional<uint64_t> Scheduler::get_ls_arrival_order(uint64_t seq_id) const
{
    auto arrival = ls_arrival_order_by_seq_id_.find(seq_id);
    return arrival == ls_arrival_order_by_seq_id_.end() ? std::nullopt : std::optional<uint64_t>(arrival->second);
}

void Scheduler::latch_ls_fatal(LSFatalCode code) noexcept
{
    if (!ls_fatal_.has_value()) {
        ls_fatal_ = code;
    }
}

std::optional<LSFatalCode> Scheduler::ls_fatal_code() const noexcept
{
    return ls_fatal_;
}

std::vector<uint64_t> Scheduler::get_ls_pending_batch_ids() const
{
    return {};
}

std::vector<std::vector<uint64_t>> Scheduler::get_ls_pending_batch_sequence_ids() const
{
    return {};
}

std::vector<uint32_t> Scheduler::get_ls_pending_batch_attempts() const
{
    return {};
}

std::vector<bool> Scheduler::get_ls_pending_batch_is_recovery() const
{
    return {};
}

std::vector<std::optional<uint64_t>> Scheduler::get_ls_pending_batch_parent_batch_ids() const
{
    return {};
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
    // Kept as a compatibility probe. The source-aligned path has no
    // persistent fresh-batch ownership; batch IDs exist only in admission
    // records and immutable placement telemetry.
    return {};
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
    if (ls_fatal_.has_value()) {
        throw LSSchedulerFatalError(*ls_fatal_, "LoongServe-style scheduler is permanently fatal");
    }
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
        try {
            plan->scheduler_allocation_before = group.allocated_attention_ranks;
            plan->scheduler_allocation_after = plan->retained_ranks;
            plan->scheduler_last_iteration_masters_after.reserve(group.last_iteration_masters.size());
            std::copy_if(group.last_iteration_masters.begin(),
                         group.last_iteration_masters.end(),
                         std::back_inserter(plan->scheduler_last_iteration_masters_after),
                         [&](int rank) { return rank != plan->source_rank; });
        }
        catch (...) {
            worker_state.at(group.dp_idx)->abort_kv_consolidation(plan);
            throw;
        }
        active_ls_kv_transaction_ = plan;
    }
    return plan;
}

bool Scheduler::mark_ls_kv_scale_down_dispatched(
    const std::shared_ptr<SPStateManager::LSKVConsolidationPlan>& plan) noexcept
{
    if (ls_fatal_.has_value() || !plan || !active_ls_kv_transaction_
        || active_ls_kv_transaction_.get() != plan.get()
        || plan->state != SPStateManager::LSKVConsolidationPlan::State::RESERVED) {
        return false;
    }
    plan->state = SPStateManager::LSKVConsolidationPlan::State::DISPATCHED;
    return true;
}

bool Scheduler::commit_ls_kv_scale_down(const std::shared_ptr<SPStateManager::LSKVConsolidationPlan>& plan)
{
    auto fail_dispatched = [&](const std::string& reason) -> bool {
        if (plan && plan->state == SPStateManager::LSKVConsolidationPlan::State::DISPATCHED) {
            latch_ls_fatal(LSFatalCode::KV_CONSOLIDATION_FAILED);
            throw LSSchedulerFatalError(LSFatalCode::KV_CONSOLIDATION_FAILED, reason);
        }
        return false;
    };
    if (!plan || !active_ls_kv_transaction_ || active_ls_kv_transaction_.get() != plan.get()) {
        return fail_dispatched("dispatched LS KV scale-down transaction identity changed before commit");
    }
    if (plan->state != SPStateManager::LSKVConsolidationPlan::State::DISPATCHED) {
        return false;
    }
    auto group_it = ls_groups_.find(plan->group_id);
    if (group_it == ls_groups_.end() || group_it->second.dp_idx != plan->dp_idx) {
        return fail_dispatched("dispatched LS KV scale-down group changed before commit");
    }
    auto& group = group_it->second;

    if (group.sequences.size() != plan->group_sequence_ids.size()) {
        return fail_dispatched("dispatched LS KV scale-down membership became stale before commit");
    }
    for (size_t idx = 0; idx < group.sequences.size(); ++idx) {
        if (!group.sequences[idx] || group.sequences[idx]->seq_id != plan->group_sequence_ids[idx]) {
            return fail_dispatched("dispatched LS KV scale-down membership became stale before commit");
        }
    }
    if (group.allocated_attention_ranks != plan->scheduler_allocation_before) {
        return fail_dispatched("dispatched LS KV scale-down allocation became stale before commit");
    }
    if (!worker_state.at(plan->dp_idx)->commit_kv_consolidation(plan)) {
        return fail_dispatched("dispatched LS KV scale-down worker state became stale before commit");
    }

    // Every allocating scheduler shadow was built while RESERVED. Worker and
    // scheduler publication below is a no-throw tail after physical copy.
    group.allocated_attention_ranks.swap(plan->scheduler_allocation_after);
    group.last_iteration_masters.swap(plan->scheduler_last_iteration_masters_after);
    group.kv_candidate_target_dop   = -1;
    group.kv_candidate_stable_steps = 0;
    group.kv_candidate_member_ids.clear();
    group.kv_candidate_allocation.clear();
    group.last_consolidation_step   = ls_schedule_step_;
    // Coordinator commit runs after the reserving schedule() returned, so it
    // is a fresh publication boundary even if the previous step marked this DP.
    _mark_ls_pool_resource_mutated(plan->dp_idx, true);
    active_ls_kv_transaction_.reset();
    return true;
}

void Scheduler::abort_ls_kv_scale_down(const std::shared_ptr<SPStateManager::LSKVConsolidationPlan>& plan)
{
    if (!plan || !active_ls_kv_transaction_ || active_ls_kv_transaction_.get() != plan.get()) {
        return;
    }
    if (plan->state == SPStateManager::LSKVConsolidationPlan::State::DISPATCHED) {
        latch_ls_fatal(LSFatalCode::KV_CONSOLIDATION_FAILED);
        throw LSSchedulerFatalError(
            LSFatalCode::KV_CONSOLIDATION_FAILED,
            "cannot abort an LS KV scale-down transaction after worker dispatch");
    }
    if (plan->state != SPStateManager::LSKVConsolidationPlan::State::RESERVED) {
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

    std::vector<uint64_t> group_ids;
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        group_ids.insert(group_ids.end(), ls_group_ids_by_dp_[dp_idx].begin(), ls_group_ids_by_dp_[dp_idx].end());
    }
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
                usable_block_sum += static_cast<int64_t>(
                    worker_state.at(group.dp_idx)->block_manager.at(rank)->blocks().size());
            }
        }

        int real_batch =
            static_cast<int>(std::count_if(group.sequences.begin(), group.sequences.end(), [](const auto& sequence) {
                return sequence && sequence->status == SequenceStatus::RUNNING;
            }));
        int target_dop = 1;
        while (target_dop < static_cast<int>(participants.size())
               && real_batch / target_dop > ls_min_comp_bound_decoding_batch_size_) {
            ++target_dop;
        }
        double utilization = usable_block_sum > 0 ? static_cast<double>(used_block_sum) / usable_block_sum : 1.0;

        bool has_unreclaimed_empty_rank = participants.size() != group.allocated_attention_ranks.size();
        bool below_candidate_threshold  = utilization < ls_kv_consolidation_candidate_util_;
        bool dop_can_shrink             = static_cast<int>(participants.size()) > target_dop;
        if (real_batch == 0 || has_unreclaimed_empty_rank || !dop_can_shrink
            || !below_candidate_threshold) {
            group.kv_candidate_target_dop   = -1;
            group.kv_candidate_stable_steps = 0;
            group.kv_candidate_member_ids.clear();
            group.kv_candidate_allocation.clear();
            continue;
        }

        std::vector<uint64_t> member_ids;
        for (const auto& sequence : group.sequences) {
            if (sequence && sequence->status == SequenceStatus::RUNNING) {
                member_ids.push_back(sequence->seq_id);
            }
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
            group.kv_candidate_member_ids.clear();
            group.kv_candidate_allocation.clear();
            continue;
        }
        const bool same_identity = group.kv_candidate_target_dop == target_dop
                                   && group.kv_candidate_member_ids == member_ids
                                   && group.kv_candidate_allocation == group.allocated_attention_ranks;
        if (same_identity) {
            group.kv_candidate_stable_steps++;
        }
        else {
            group.kv_candidate_target_dop   = target_dop;
            group.kv_candidate_stable_steps = 1;
            group.kv_candidate_member_ids   = std::move(member_ids);
            group.kv_candidate_allocation   = group.allocated_attention_ranks;
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
            if (ls_kv_consolidation_mode_ == "execute"
                && source_blocks > ls_kv_consolidation_max_source_blocks_per_event_) {
                abort_ls_kv_scale_down(plan);
                ls_step_kv_decision_reason_ = "source_block_budget";
                continue;
            }
            if (!_ls_kv_consolidation_watermark_ok(plan)) {
                abort_ls_kv_scale_down(plan);
                ls_step_kv_decision_reason_ = "target_high_watermark";
                continue;
            }
            if (ls_kv_consolidation_mode_ == "shadow") {
                abort_ls_kv_scale_down(plan);
                ls_step_kv_source_rank_     = source_rank;
                ls_step_kv_decision_reason_ = "shadow_candidate";
                return nullptr;
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

void Scheduler::set_ls_post_admission_component_failure_for_test(int value)
{
    ls_post_admission_component_failure_for_test_ = value;
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
try
{
    // This is per-call state. Clear it before any entry guard so a RESERVED
    // transaction cannot inherit publication state from the preceding step.
    ls_step_publication_started_ = false;
    if (ls_fatal_.has_value()) {
        throw LSSchedulerFatalError(*ls_fatal_, "LoongServe-style scheduler is permanently fatal");
    }
    if (active_ls_kv_transaction_) {
        throw std::runtime_error("cannot schedule Decode while an LS KV scale-down transaction is reserved");
    }
    if (enable_ls_decode_core_scheduler_) {
        ls_schedule_step_++;
    }
    ls_step_pool_resource_epoch_before_ = ls_pool_resource_epoch_;
    ls_step_pool_resource_mutated_.assign(attention_dp_, false);
    ls_step_initial_records_.clear();
    ls_step_group_plans_.clear();
    ls_step_group_plan_ids_.clear();
    ls_step_group_plan_sequence_ids_.clear();
    ls_step_reused_passive_masters_.clear();
    ls_step_preempted_sequence_ids_.clear();
    ls_step_preemption_reasons_.clear();
    ls_step_atomic_no_fit_count_   = 0;
    ls_step_atomic_merge_count_    = 0;
    ls_step_atomic_rollback_count_ = 0;
    ls_step_planning_latency_ms_ = 0.0;
    ls_step_kv_candidate_        = false;
    ls_step_kv_group_id_         = -1;
    ls_step_kv_source_rank_      = -1;
    ls_step_kv_target_dop_       = -1;
    ls_step_kv_stable_steps_     = 0;
    ls_step_kv_group_util_       = 0.0;
    ls_step_kv_decision_reason_  = ls_kv_consolidation_mode_ == "off" ? "off" : "no_candidate";
    ls_step_admission_records_.clear();
    ls_step_real_decode_ids_by_dp_.assign(attention_dp_, {});
    ls_step_execution_loop_count_ = 1;
    ls_step_offload_committed_ = false;
    std::vector<std::vector<std::shared_ptr<Sequence>>> dp_seqs;
    bool                                                has_prefill = false;
    bool                                                has_step_entry_decode = false;

    if (scheduler_mode_ == SchedulerMode::DECENTRALIZED) {
        return _schedule_decentralized();
    }
    else {
        if (enable_ls_decode_core_scheduler_) {
            std::unordered_set<uint64_t> step_entry_ids;
            for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
                for (const auto& sequence : _ls_running_sequences_in_pool(dp_idx)) {
                    const int remaining_tokens = sequence->max_tokens - sequence->num_completed_tokens();
                    if (remaining_tokens <= 0) {
                        latch_ls_fatal(LSFatalCode::POST_PUBLICATION_INVARIANT);
                        throw LSSchedulerFatalError(
                            LSFatalCode::POST_PUBLICATION_INVARIANT,
                            "LoongServe-style RUNNING sequence has no remaining output tokens");
                    }
                    step_entry_ids.insert(sequence->seq_id);
                    has_step_entry_decode = true;
                }
            }
            if (has_step_entry_decode) {
                // Match NanoDeploy's original chunked Decode semantics: every
                // Decode dispatch executes the configured loop count. Requests
                // that reach max_tokens or EOS inside the chunk are finalized by
                // postprocess(), which discards the unused tail tokens instead of
                // shrinking the whole batch to the shortest remaining request.
                ls_step_execution_loop_count_ = loop_count_;
            }

            std::shared_ptr<SPStateManager::LSKVConsolidationPlan> consolidation_plan;
            dp_seqs = _schedule_ls_combined_pool_step(step_entry_ids, &consolidation_plan);
            if (consolidation_plan) {
                try {
                    ScheduleResult maintenance;
                    maintenance.action                = ScheduleAction::KV_CONSOLIDATION;
                    maintenance.is_prefill            = false;
                    maintenance.kv_consolidation_plan = consolidation_plan;
                    maintenance.ls_pool_resource_epoch_before = ls_step_pool_resource_epoch_before_;
                    maintenance.ls_pool_resource_epoch_after  = ls_pool_resource_epoch_;
                    _populate_ls_kv_consolidation_telemetry(maintenance);
                    return maintenance;
                }
                catch (...) {
                    // The plan has not been returned and no worker RPC can have
                    // started. Restore its prepared reservations before the
                    // exception leaves schedule(); otherwise the active guard
                    // would become unreachable and permanently wedge the engine.
                    const auto original_error = std::current_exception();
                    try {
                        abort_ls_kv_scale_down(consolidation_plan);
                    }
                    catch (...) {
                        latch_ls_fatal(LSFatalCode::DECODE_PREPARE_OR_VALIDATE_FAILED);
                        throw LSSchedulerFatalError(
                            LSFatalCode::DECODE_PREPARE_OR_VALIDATE_FAILED,
                            "failed to abort an unpublished LS KV consolidation reservation");
                    }
                    std::rethrow_exception(original_error);
                }
            }
        }
        else {
            // Centralized legacy mode keeps its original admission-first path.
            dp_seqs = _schedule_prefill();

            for (const auto& seqs : dp_seqs) {
                if (!seqs.empty()) {
                    has_prefill = true;
                    break;
                }
            }
            if (!has_prefill) {
                dp_seqs = _schedule_decode();
            }
        }
    }

    if (enable_ls_decode_core_scheduler_ && !has_step_entry_decode && ls_step_admission_records_.empty()
        && !ls_step_offload_committed_) {
        if (get_total_waiting_migration_size() > 0) {
            latch_ls_fatal(LSFatalCode::NO_PROGRESS_INVARIANT);
            throw LSSchedulerFatalError(LSFatalCode::NO_PROGRESS_INVARIANT,
                                        "outstanding LS requests produced no admission, OFFLOAD, or Decode plan");
        }
        throw std::runtime_error("cannot schedule an empty LoongServe-style engine");
    }

    ScheduleResult result;
    result.dp_seqs    = dp_seqs;
    result.is_prefill = has_prefill;
    result.execution_loop_count = enable_ls_decode_core_scheduler_ ? ls_step_execution_loop_count_ :
                                                                    (has_prefill ? 1 : loop_count_);
    result.action = enable_ls_decode_core_scheduler_
                        ? (has_step_entry_decode && !ls_step_offload_committed_ ? ScheduleAction::DECODE
                                                                               : ScheduleAction::ADMISSION)
                        : (has_prefill ? ScheduleAction::ADMISSION : ScheduleAction::DECODE);

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

    if (enable_ls_decode_core_scheduler_) {
        result.waiting_head_blocks.assign(attention_dp_, 0);
        result.waiting_total_blocks.assign(attention_dp_, 0);
        for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
            const auto& queue = ls_waiting_by_dp_[dp_idx];
            if (!queue.empty()) {
                result.waiting_head_blocks[dp_idx] =
                    (queue.front()->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
            }
            for (const auto& sequence : queue) {
                result.waiting_total_blocks[dp_idx] +=
                    (sequence->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
            }
        }
    }
    else {
        // Legacy centralized mode reports the shared queue on every DP.
        auto& wait_queue = (mode_ != "decode") ? waiting : waiting_migration;
        int head_blocks  = 0;
        int total_blocks = 0;
        if (!wait_queue.empty()) {
            head_blocks = (wait_queue.front()->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
        }
        for (const auto& sequence : wait_queue) {
            total_blocks += (sequence->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
        }
        result.waiting_head_blocks.resize(attention_dp_, head_blocks);
        result.waiting_total_blocks.resize(attention_dp_, total_blocks);
    }

    if (enable_ls_decode_core_scheduler_) {
        result.ls_admission_records = ls_step_admission_records_;
        result.ls_real_decode_ids_by_dp = ls_step_real_decode_ids_by_dp_;
        result.ls_pool_resource_epoch_before = ls_step_pool_resource_epoch_before_;
        result.ls_pool_resource_epoch_after  = ls_pool_resource_epoch_;
        result.ls_running_ids_by_dp_after_commit.resize(attention_dp_);
        for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
            for (const auto& sequence : _ls_running_sequences_in_pool(dp_idx)) {
                result.ls_running_ids_by_dp_after_commit[dp_idx].push_back(sequence->seq_id);
            }
        }
        for (size_t idx = 0; idx < ls_step_group_plans_.size(); ++idx) {
            uint64_t group_id = ls_step_group_plan_ids_[idx];
            auto     group_it = ls_groups_.find(group_id);
            if (group_it == ls_groups_.end()) {
                continue;
            }
            const auto& group = group_it->second;
            const auto& plan  = ls_step_group_plans_[idx];
            const auto& planned_sequence_ids = ls_step_group_plan_sequence_ids_.at(idx);
            if (planned_sequence_ids.size() != plan.sequence_master_ranks.size()) {
                latch_ls_fatal(LSFatalCode::POST_PUBLICATION_INVARIANT);
                throw LSSchedulerFatalError(LSFatalCode::POST_PUBLICATION_INVARIANT,
                                            "LS iteration telemetry membership does not match the committed plan");
            }
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
            sequence_ids.reserve(planned_sequence_ids.size());
            for (size_t assignment_idx = 0; assignment_idx < planned_sequence_ids.size(); ++assignment_idx) {
                const uint64_t seq_id = planned_sequence_ids[assignment_idx];
                auto seq_it = std::find_if(group.sequences.begin(), group.sequences.end(), [&](const auto& seq) {
                    return seq && seq->seq_id == seq_id;
                });
                if (seq_it == group.sequences.end() || (*seq_it)->status != SequenceStatus::RUNNING) {
                    latch_ls_fatal(LSFatalCode::POST_PUBLICATION_INVARIANT);
                    throw LSSchedulerFatalError(LSFatalCode::POST_PUBLICATION_INVARIANT,
                                                "LS iteration telemetry references a non-running plan member");
                }
                const auto& seq       = *seq_it;
                sequence_ids.push_back(seq_id);
                int master           = plan.sequence_master_ranks[assignment_idx];
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
        result.ls_pending_batch_count = 0;
        for (const auto& queue : ls_waiting_by_dp_) {
            result.ls_pending_request_count += static_cast<int>(queue.size());
        }
        result.ls_atomic_admission_no_fit_count   = ls_step_atomic_no_fit_count_;
        result.ls_atomic_admission_merge_count    = ls_step_atomic_merge_count_;
        result.ls_atomic_admission_rollback_count = ls_step_atomic_rollback_count_;
    }

    _populate_ls_kv_consolidation_telemetry(result);

    if (enable_ls_decode_core_scheduler_ && result.action == ScheduleAction::ADMISSION) {
        result.dp_seqs.clear();
        result.dp_sp_seqs.clear();
        result.filtered_dp_sp_seqs.clear();
        result.sp_send_counts.clear();
        result.sp_recv_counts.clear();
        result.sp_size_hist_per_dp.clear();
        result.sp_q_matrix.clear();
        result.sp_res_matrix.clear();
        result.waiting_head_blocks.clear();
        result.waiting_total_blocks.clear();
        result.ls_real_decode_ids_by_dp.clear();
    }

    return result;
}
catch (const LSSchedulerFatalError&) {
    throw;
}
catch (...) {
    if (enable_ls_decode_core_scheduler_ && ls_step_publication_started_) {
        latch_ls_fatal(LSFatalCode::POST_PUBLICATION_INVARIANT);
        throw LSSchedulerFatalError(LSFatalCode::POST_PUBLICATION_INVARIANT,
                                    "unexpected failure after LS scheduler publication began");
    }
    throw;
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
        if (!allocated[rank] && _ls_rank_is_truly_idle(dp_idx, rank)) {
            result.push_back(rank);
        }
    }
    return result;
}

int64_t Scheduler::_ls_admission_need_tokens(const Sequence& sequence) const
{
    if (sequence.status == SequenceStatus::PAUSED_OFFLOAD) {
        return sequence.num_tokens;
    }
    return sequence.num_prompt_tokens;
}

int64_t Scheduler::_ls_pool_token_capacity(int dp_idx) const
{
    if (dp_idx < 0 || dp_idx >= attention_dp_) {
        return 0;
    }
    int64_t blocks = 0;
    for (int rank = 0; rank < attention_sp_; ++rank) {
        auto manager = worker_state[dp_idx]->block_manager.find(rank);
        if (manager == worker_state[dp_idx]->block_manager.end()) {
            return 0;
        }
        blocks += static_cast<int64_t>(manager->second->blocks().size());
    }
    return blocks * static_cast<int64_t>(Sequence::block_size);
}

std::vector<std::shared_ptr<Sequence>> Scheduler::_ls_running_sequences_in_pool(int dp_idx) const
{
    std::vector<std::shared_ptr<Sequence>> result;
    std::unordered_set<uint64_t>           emitted;
    if (dp_idx < 0 || dp_idx >= attention_dp_) {
        return result;
    }
    for (uint64_t group_id : ls_group_ids_by_dp_[dp_idx]) {
        auto group = ls_groups_.find(group_id);
        if (group == ls_groups_.end() || group->second.dp_idx != dp_idx) {
            throw std::runtime_error("LS canonical group order references a missing or cross-pool group");
        }
        for (const auto& sequence : group->second.sequences) {
            if (!sequence || sequence->status != SequenceStatus::RUNNING || sequence->assigned_dp != dp_idx
                || sequence->block_ctx(BlockContextSlot::ACTIVE).dp_idx_ != dp_idx) {
                throw std::runtime_error("LS canonical group contains an invalid running sequence");
            }
            auto owner = ls_seq_to_group_.find(sequence->seq_id);
            if (owner == ls_seq_to_group_.end() || owner->second != group_id
                || !emitted.insert(sequence->seq_id).second) {
                throw std::runtime_error("LS running sequence has missing or duplicate canonical ownership");
            }
            result.push_back(sequence);
        }
    }
    return result;
}

bool Scheduler::_ls_pool_future_kv_fits(int                                           dp_idx,
                                        const std::vector<std::shared_ptr<Sequence>>& tentative) const
{
    if (!ls_decode_enable_future_kv_admission_) {
        return true;
    }
    if (dp_idx < 0 || dp_idx >= attention_dp_) {
        return false;
    }

    std::vector<std::shared_ptr<Sequence>> envelope;
    std::unordered_set<uint64_t>           selected_ids;
    for (const auto& sequence : _ls_running_sequences_in_pool(dp_idx)) {
        if (sequence && selected_ids.insert(sequence->seq_id).second) {
            envelope.push_back(sequence);
        }
    }
    for (const auto& sequence : tentative) {
        if (!sequence || sequence->assigned_dp != dp_idx) {
            return false;
        }
        if (selected_ids.insert(sequence->seq_id).second) {
            envelope.push_back(sequence);
        }
    }

    int unselected_paused = 0;
    for (const auto& sequence : ls_waiting_by_dp_[dp_idx]) {
        if (sequence && sequence->status == SequenceStatus::PAUSED_OFFLOAD
            && !selected_ids.count(sequence->seq_id)) {
            ++unselected_paused;
        }
    }
    if (static_cast<int>(envelope.size()) + unselected_paused > ls_running_max_req_size_) {
        return false;
    }
    auto peak = _ls_future_kv_peak_tokens(envelope);
    return peak.has_value() && *peak <= _ls_pool_token_capacity(dp_idx);
}

bool Scheduler::_ls_rank_is_truly_idle(int dp_idx, int sp_idx) const
{
    if (dp_idx < 0 || dp_idx >= attention_dp_ || sp_idx < 0 || sp_idx >= attention_sp_) {
        return false;
    }
    for (uint64_t group_id : ls_group_ids_by_dp_[dp_idx]) {
        auto group = ls_groups_.find(group_id);
        if (group != ls_groups_.end()
            && std::find(group->second.allocated_attention_ranks.begin(),
                         group->second.allocated_attention_ranks.end(),
                         sp_idx)
                   != group->second.allocated_attention_ranks.end()) {
            return false;
        }
    }
    for (const auto& sequence : worker_state[dp_idx]->running) {
        if (!sequence || sequence->status != SequenceStatus::RUNNING) {
            continue;
        }
        const auto& context = sequence->block_ctx(BlockContextSlot::ACTIVE);
        if (context.master_sp_idx_ == sp_idx
            || (context.pending_token_present_ && context.pending_token_target_sp_ == sp_idx)
            || (sp_idx < static_cast<int>(context.num_dispatched_tokens.size())
                && context.num_dispatched_tokens[sp_idx] != 0)
            || (sp_idx < static_cast<int>(context.sp_block_table.size())
                && !context.sp_block_table[sp_idx].empty())) {
            return false;
        }
    }
    // Each SPStateManager owns one permanent cadence dummy per rank. Its
    // allocator block is fixed infrastructure, not live request KV and must
    // not make an otherwise unowned rank unavailable to LS planning. Active
    // transaction reservations are guarded globally by the scheduler.
    return true;
}

void Scheduler::_mark_ls_pool_resource_mutated(int dp_idx, bool force_new_boundary) noexcept
{
    if (dp_idx < 0 || dp_idx >= attention_dp_
        || dp_idx >= static_cast<int>(ls_pool_resource_epoch_.size())
        || dp_idx >= static_cast<int>(ls_step_pool_resource_mutated_.size())) {
        return;
    }
    if (force_new_boundary || !ls_step_pool_resource_mutated_[dp_idx]) {
        ls_step_pool_resource_mutated_[dp_idx] = true;
        ++ls_pool_resource_epoch_[dp_idx];
    }
}

std::optional<int64_t>
Scheduler::_ls_future_kv_peak_tokens(const std::vector<std::shared_ptr<Sequence>>& sequences) const
{
    // Source-equivalent to LoongServe ReqQueue::_can_add_new_req(): represent
    // every live request as (KV held now, remaining Decode iterations), sort by
    // remaining iterations, and evaluate the aggregate KV high-water mark at
    // every request-completion boundary. NanoDeploy's LS workload uses
    // ignore_eos=True, and LoongServe's default/busy path also uses the full
    // user-provided maximum output length, so max_tokens is deliberately not
    // shortened using an average-output heuristic here.
    std::vector<std::pair<int64_t, int64_t>> lengths;
    lengths.reserve(sequences.size());
    for (const auto& seq : sequences) {
        if (!seq) {
            return std::nullopt;
        }
        if (seq->status == SequenceStatus::FINISHED) {
            continue;
        }

        const int64_t generated  = std::max(0, seq->num_completed_tokens());
        const int64_t max_output = std::max(0, seq->max_tokens);
        int64_t       held_tokens;
        int64_t       remaining_iterations;
        if (seq->status == SequenceStatus::RUNNING) {
            held_tokens          = static_cast<int64_t>(seq->num_prompt_tokens) + generated;
            remaining_iterations = std::max<int64_t>(0, max_output - generated - 1);
        }
        else if (seq->status == SequenceStatus::WAITING) {
            // LoongServe charges the first sampled token at admission and then
            // excludes the final token, whose KV is never needed after finish.
            held_tokens          = static_cast<int64_t>(seq->num_prompt_tokens) + 1;
            remaining_iterations = std::max<int64_t>(0, max_output - 2);
        }
        else if (seq->status == SequenceStatus::PAUSED_OFFLOAD) {
            held_tokens = static_cast<int64_t>(seq->num_prompt_tokens) + generated + 1;
            remaining_iterations = std::max<int64_t>(0, max_output - generated - 2);
        }
        else {
            return std::nullopt;
        }
        lengths.emplace_back(held_tokens, remaining_iterations);
    }

    if (lengths.empty()) {
        return int64_t{0};
    }
    std::stable_sort(
        lengths.begin(), lengths.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    int64_t prefix_held = 0;
    int64_t peak_tokens = 0;
    for (size_t idx = 0; idx < lengths.size(); ++idx) {
        prefix_held += lengths[idx].first;
        const int64_t active_requests = static_cast<int64_t>(idx) + 1;
        peak_tokens                   = std::max(peak_tokens, prefix_held + active_requests * lengths[idx].second);
    }
    return peak_tokens;
}

bool Scheduler::_ls_future_kv_fits(int                                           dp_idx,
                                   const std::vector<std::shared_ptr<Sequence>>& batch,
                                   const std::vector<std::shared_ptr<Sequence>>& existing_sequences,
                                   const std::vector<int>&                       future_rank_pool,
                                   const std::vector<int>&                       free_block_adjustments,
                                   const LSBlockContextOverrides&                context_overrides) const
{
    (void)existing_sequences;
    (void)future_rank_pool;
    (void)free_block_adjustments;
    (void)context_overrides;
    return _ls_pool_future_kv_fits(dp_idx, batch);
#if 0
    if (!ls_decode_enable_future_kv_admission_) {
        return true;
    }
    if (dp_idx < 0 || dp_idx >= attention_dp_ || future_rank_pool.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Sequence>> projected_sequences = existing_sequences;
    projected_sequences.insert(projected_sequences.end(), batch.begin(), batch.end());
    auto peak_tokens = _ls_future_kv_peak_tokens(projected_sequences);
    if (!peak_tokens.has_value()) {
        return false;
    }

    // Only capacity already owned by this prospective group plus currently
    // free blocks is admissible. This prevents the aggregate test from
    // borrowing blocks held by another LS group on the same DP domain.
    std::unordered_set<int> unique_ranks;
    int64_t                 accessible_blocks = 0;
    for (int rank : future_rank_pool) {
        if (rank < 0 || rank >= attention_sp_ || !unique_ranks.insert(rank).second) {
            if (rank < 0 || rank >= attention_sp_) {
                return false;
            }
            continue;
        }
        auto manager = worker_state[dp_idx]->block_manager.find(rank);
        if (manager == worker_state[dp_idx]->block_manager.end()) {
            return false;
        }
        int free_adjustment = rank < static_cast<int>(free_block_adjustments.size()) ? free_block_adjustments[rank] : 0;
        accessible_blocks += manager->second->num_free_blocks() + free_adjustment;
        for (const auto& seq : existing_sequences) {
            if (!seq || seq->status == SequenceStatus::FINISHED) {
                continue;
            }
            auto        context = context_overrides.find(seq.get());
            const auto& tables  = context == context_overrides.end() ?
                                      seq->block_ctx(BlockContextSlot::ACTIVE).sp_block_table :
                                      context->second->sp_block_table;
            if (rank >= static_cast<int>(tables.size())) {
                return false;
            }
            accessible_blocks += static_cast<int64_t>(tables[rank].size());
        }
    }
    const int64_t accessible_tokens = accessible_blocks * static_cast<int64_t>(Sequence::block_size);
    return *peak_tokens <= accessible_tokens;
#endif
}

bool Scheduler::_ls_future_kv_fits_empty_system(int                                           dp_idx,
                                                const std::vector<std::shared_ptr<Sequence>>& batch,
                                                const std::vector<int>&                       future_rank_pool) const
{
    if (!ls_decode_enable_future_kv_admission_) {
        return true;
    }
    if (dp_idx < 0 || dp_idx >= attention_dp_) {
        return false;
    }
    auto peak_tokens = _ls_future_kv_peak_tokens(batch);
    if (!peak_tokens.has_value() || future_rank_pool.empty()) {
        return false;
    }

    std::unordered_set<int> unique_ranks;
    int64_t                 capacity_blocks = 0;
    for (int rank : future_rank_pool) {
        if (rank < 0 || rank >= attention_sp_ || !unique_ranks.insert(rank).second) {
            if (rank < 0 || rank >= attention_sp_) {
                return false;
            }
            continue;
        }
        // Future-KV uses the allocator's full usable pool capacity. Permanent
        // cadence dummies, reserved headroom and current block reservations are
        // exact-gate concerns and must not be deducted a second time here.
        capacity_blocks += static_cast<int64_t>(worker_state[dp_idx]->block_manager.at(rank)->blocks().size());
    }
    const int64_t capacity_tokens = capacity_blocks * static_cast<int64_t>(Sequence::block_size);
    return *peak_tokens <= capacity_tokens;
}

std::optional<std::pair<std::vector<int>, std::vector<std::vector<int>>>>
Scheduler::_plan_ls_initial_placement(int                                           dp_idx,
                                      const std::vector<std::shared_ptr<Sequence>>& batch,
                                      const std::vector<int>&                       rank_pool,
                                      const std::vector<std::shared_ptr<Sequence>>& existing_sequences,
                                      const std::vector<int>&                       base_allocation,
                                      const std::vector<int>&                       free_block_adjustments,
                                      const LSBlockContextOverrides&                context_overrides) const
{
    if (batch.empty() || static_cast<int>(batch.size()) > max_num_seqs_) {
        return std::nullopt;
    }
    if (!_ls_pool_future_kv_fits(dp_idx, batch)) {
        return std::nullopt;
    }

    std::vector<int> ordered_pool = rank_pool;
    auto             adjusted_free_blocks = [&](int rank) {
        int adjustment = rank < static_cast<int>(free_block_adjustments.size()) ? free_block_adjustments[rank] : 0;
        return worker_state[dp_idx]->block_manager.at(rank)->num_free_blocks() + adjustment;
    };
    auto pool_running = _ls_running_sequences_in_pool(dp_idx);
    auto used_tokens  = worker_state[dp_idx]->group_used_kv_tokens(pool_running);
    // LoongServe selects instances by ascending used tokens. The stable rank
    // key makes the first exact-feasible DoP deterministic.
    std::sort(ordered_pool.begin(), ordered_pool.end(), [&](int lhs, int rhs) {
        return std::tie(used_tokens[lhs], lhs) < std::tie(used_tokens[rhs], rhs);
    });
    ordered_pool.erase(std::unique(ordered_pool.begin(), ordered_pool.end()), ordered_pool.end());

    int first_d = ls_decode_initial_kv_dop_ == 0 ? 1 : ls_decode_initial_kv_dop_;
    int last_d  = ls_decode_initial_kv_dop_ == 0 ? std::min(attention_sp_, static_cast<int>(ordered_pool.size())) :
                                                   ls_decode_initial_kv_dop_;
    for (int d = first_d; d <= last_d; ++d) {
        if (d <= 0 || d > static_cast<int>(ordered_pool.size())) {
            continue;
        }
        std::vector<int> ranks(ordered_pool.begin(), ordered_pool.begin() + d);
        std::vector<std::vector<int>> placements(batch.size(), std::vector<int>(attention_sp_, 0));
        std::vector<int>              needed_blocks(attention_sp_, 0);

        // Placement is the source packed-interval direction: fill the most
        // occupied selected rank first, then cross a rank boundary only when
        // the current rank's exact free-token capacity is exhausted.
        std::vector<int> packing_ranks = ranks;
        std::stable_sort(packing_ranks.begin(), packing_ranks.end(), [&](int lhs, int rhs) {
            return used_tokens[lhs] != used_tokens[rhs] ? used_tokens[lhs] > used_tokens[rhs] : lhs < rhs;
        });
        std::vector<int> existing_master_load(attention_sp_, 0);
        std::vector<int> new_master_load(attention_sp_, 0);
        for (const auto& seq : existing_sequences) {
            if (seq && seq->status == SequenceStatus::RUNNING) {
                existing_master_load[seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_]++;
            }
        }
        for (size_t seq_idx = 0; seq_idx < batch.size(); ++seq_idx) {
            new_master_load[ranks[seq_idx % ranks.size()]]++;
        }
        std::vector<int> master_load = existing_master_load;
        for (int rank = 0; rank < attention_sp_; ++rank) {
            master_load[rank] += new_master_load[rank];
        }

        std::vector<int64_t> free_tokens(attention_sp_, 0);
        bool                 feasible = true;
        for (int rank : packing_ranks) {
            const int free_blocks = adjusted_free_blocks(rank);
            const int reserved_blocks =
                static_cast<int>(std::ceil(master_load[rank] * reserved_blocks_per_req_));
            const int usable_blocks = free_blocks - reserved_blocks;
            const int bootstrap_tokens = new_master_load[rank];
            if (usable_blocks < 0
                || static_cast<int64_t>(usable_blocks) * Sequence::block_size < bootstrap_tokens) {
                feasible = false;
                break;
            }
            // Initial placement publishes one fixed bootstrap token per new
            // master and must preserve reserved Decode headroom. Remove both
            // from the packing capacity up front so a long prompt cannot fill
            // a rank and make the otherwise feasible cross-rank plan fail its
            // exact block check afterwards.
            free_tokens[rank] = static_cast<int64_t>(usable_blocks) * Sequence::block_size - bootstrap_tokens;
        }

        for (size_t seq_idx = 0; seq_idx < batch.size(); ++seq_idx) {
            if (!feasible) {
                break;
            }
            int64_t remaining = _ls_admission_need_tokens(*batch[seq_idx]);
            for (int rank : packing_ranks) {
                int64_t placed = std::min(remaining, free_tokens[rank]);
                if (placed <= 0) {
                    continue;
                }
                placements[seq_idx][rank] = static_cast<int>(placed);
                free_tokens[rank] -= placed;
                remaining -= placed;
                if (remaining == 0) {
                    break;
                }
            }
            if (remaining != 0) {
                feasible = false;
                break;
            }
            for (int rank : ranks) {
                int tokens = placements[seq_idx][rank];
                needed_blocks[rank] += (tokens + Sequence::block_size - 1) / Sequence::block_size;
            }
            int provisional_master = ranks[seq_idx % ranks.size()];
            if (placements[seq_idx][provisional_master] % Sequence::block_size == 0) {
                needed_blocks[provisional_master]++;
            }
        }

        for (int rank : ranks) {
            int free_blocks = adjusted_free_blocks(rank);
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
            for (int rank : ranks) {
                if (std::find(planning_ranks.begin(), planning_ranks.end(), rank) == planning_ranks.end()) {
                    planning_ranks.push_back(rank);
                }
            }
            std::vector<int> receiver_load(attention_sp_, 0);
            for (const auto& seq : existing_sequences) {
                if (!seq || seq->status != SequenceStatus::RUNNING) {
                    continue;
                }
                int         master  = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
                auto        context = context_overrides.find(seq.get());
                const auto& ctx =
                    context == context_overrides.end() ? seq->block_ctx(BlockContextSlot::ACTIVE) : *context->second;
                for (int owner = 0; owner < attention_sp_; ++owner) {
                    int committed = ctx.num_dispatched_tokens[owner];
                    if (ctx.pending_token_present_ && ctx.pending_token_target_sp_ == owner) {
                        committed--;
                    }
                    if (owner != master && committed > 0) {
                        receiver_load[owner]++;
                    }
                }
            }
            for (size_t seq_idx = 0; seq_idx < batch.size(); ++seq_idx) {
                int master = ranks[seq_idx % ranks.size()];
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
                for (int rank : ranks) {
                    int free_blocks = adjusted_free_blocks(rank);
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

bool Scheduler::_ls_batch_fits_empty_system(
    int dp_idx, const std::vector<std::shared_ptr<Sequence>>& batch) const
{
    if (dp_idx < 0 || dp_idx >= attention_dp_ || batch.empty()
        || static_cast<int>(batch.size()) > max_num_seqs_) {
        return false;
    }

    int first_d = ls_decode_initial_kv_dop_ == 0 ? 1 : ls_decode_initial_kv_dop_;
    int last_d  = ls_decode_initial_kv_dop_ == 0 ? attention_sp_ : ls_decode_initial_kv_dop_;
    for (int d = first_d; d <= last_d; ++d) {
        if (d <= 0 || d > attention_sp_) {
            continue;
        }
        std::vector<int> future_rank_pool;
        int              future_dop = ls_decode_enable_memory_scale_up_ ? attention_sp_ : d;
        future_rank_pool.reserve(future_dop);
        for (int rank = 0; rank < future_dop; ++rank) {
            future_rank_pool.push_back(rank);
        }
        if (!_ls_future_kv_fits_empty_system(dp_idx, batch, future_rank_pool)) {
            continue;
        }
        std::vector<std::vector<int>> placements(batch.size(), std::vector<int>(attention_sp_, 0));
        std::vector<int>              master_load(attention_sp_, 0);
        for (size_t seq_idx = 0; seq_idx < batch.size(); ++seq_idx) {
            master_load[seq_idx % d]++;
        }
        std::vector<int64_t>          free_tokens(attention_sp_, 0);
        bool                          placement_feasible = true;
        for (int rank = 0; rank < d; ++rank) {
            const int free_blocks = ls_empty_system_free_blocks_per_rank_[dp_idx][rank];
            const int reserved_blocks =
                static_cast<int>(std::ceil(master_load[rank] * reserved_blocks_per_req_));
            const int usable_blocks = free_blocks - reserved_blocks;
            const int bootstrap_tokens = master_load[rank];
            if (usable_blocks < 0
                || static_cast<int64_t>(usable_blocks) * Sequence::block_size < bootstrap_tokens) {
                placement_feasible = false;
                break;
            }
            free_tokens[rank] = static_cast<int64_t>(usable_blocks) * Sequence::block_size - bootstrap_tokens;
        }
        for (size_t seq_idx = 0; seq_idx < batch.size(); ++seq_idx) {
            if (!placement_feasible) {
                break;
            }
            int64_t remaining = _ls_admission_need_tokens(*batch[seq_idx]);
            for (int rank = 0; rank < d && remaining > 0; ++rank) {
                int64_t placed = std::min(remaining, free_tokens[rank]);
                placements[seq_idx][rank] = static_cast<int>(placed);
                free_tokens[rank] -= placed;
                remaining -= placed;
            }
            if (remaining != 0) {
                placement_feasible = false;
                break;
            }
        }
        if (!placement_feasible) {
            continue;
        }

        std::vector<int> needed_blocks(attention_sp_, 0);
        std::vector<int> receiver_load(attention_sp_, 0);
        for (size_t seq_idx = 0; seq_idx < batch.size(); ++seq_idx) {
            int master = static_cast<int>(seq_idx % d);
            for (int owner = 0; owner < attention_sp_; ++owner) {
                const int tokens_with_bootstrap = placements[seq_idx][owner] + (owner == master ? 1 : 0);
                needed_blocks[owner] +=
                    (tokens_with_bootstrap + Sequence::block_size - 1) / Sequence::block_size;
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
            feasible = needed_blocks[rank] + headroom
                       <= ls_empty_system_free_blocks_per_rank_[dp_idx][rank];
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

bool Scheduler::_ls_current_admission_fits(
    int dp_idx, const std::vector<std::shared_ptr<Sequence>>& batch) const
{
    if (dp_idx < 0 || dp_idx >= attention_dp_ || batch.empty()
        || !_ls_batch_fits_empty_system(dp_idx, batch)) {
        return false;
    }

    std::vector<std::shared_ptr<Sequence>> ordered = batch;
    std::stable_sort(ordered.begin(), ordered.end(), [&](const auto& lhs, const auto& rhs) {
        return _ls_admission_need_tokens(*lhs) > _ls_admission_need_tokens(*rhs);
    });

    auto idle_ranks = _ls_unallocated_ranks(dp_idx);
    int64_t idle_capacity = 0;
    for (int rank : idle_ranks) {
        idle_capacity += static_cast<int64_t>(worker_state[dp_idx]->block_manager.at(rank)->blocks().size())
                         * Sequence::block_size;
    }
    int64_t admission_sum = 0;
    for (const auto& sequence : ordered) {
        admission_sum += _ls_admission_need_tokens(*sequence);
    }

    std::vector<int> rank_pool = idle_ranks;
    std::vector<std::shared_ptr<Sequence>> donor_sequences;
    std::vector<int> donor_allocation;
    if (admission_sum > idle_capacity) {
        struct Donor {
            uint64_t group_id = 0;
            int64_t  slack    = 0;
        };
        std::vector<Donor> donors;
        for (uint64_t group_id : ls_group_ids_by_dp_[dp_idx]) {
            const auto& group = ls_groups_.at(group_id);
            auto used = worker_state[dp_idx]->group_used_kv_tokens(group.sequences);
            int64_t capacity = 0;
            int64_t occupied = 0;
            for (int rank : group.allocated_attention_ranks) {
                capacity += static_cast<int64_t>(worker_state[dp_idx]->block_manager.at(rank)->blocks().size())
                            * Sequence::block_size;
                occupied += used[rank];
            }
            donors.push_back({group_id, capacity - occupied});
        }
        std::stable_sort(donors.begin(), donors.end(), [](const Donor& lhs, const Donor& rhs) {
            return lhs.slack < rhs.slack;
        });

        int64_t covered = 0;
        const int64_t deficit = admission_sum - idle_capacity;
        while (!donors.empty() && covered < deficit) {
            Donor donor = donors.back();
            donors.pop_back();
            if (donor.slack <= 0) {
                continue;
            }
            covered += donor.slack;
            const auto& group = ls_groups_.at(donor.group_id);
            donor_sequences.insert(donor_sequences.end(), group.sequences.begin(), group.sequences.end());
            for (int rank : group.allocated_attention_ranks) {
                if (std::find(rank_pool.begin(), rank_pool.end(), rank) == rank_pool.end()) {
                    rank_pool.push_back(rank);
                }
                if (std::find(donor_allocation.begin(), donor_allocation.end(), rank) == donor_allocation.end()) {
                    donor_allocation.push_back(rank);
                }
            }
        }
        if (covered < deficit) {
            return false;
        }
    }

    return _plan_ls_initial_placement(dp_idx, ordered, rank_pool, donor_sequences, donor_allocation).has_value();
}

void Scheduler::_merge_ls_groups(uint64_t lhs_group_id, uint64_t rhs_group_id)
{
    if (lhs_group_id == rhs_group_id) {
        return;
    }
    // Memory-deficit merge direction is source-shaped: the constrained lhs
    // survives and donors append in pop/union order.
    uint64_t survivor_id = lhs_group_id;
    uint64_t removed_id  = rhs_group_id;
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

    std::unordered_set<uint64_t> emitted_sequence_ids;
    for (const auto& sequence : survivor.sequences) {
        if (sequence) {
            emitted_sequence_ids.insert(sequence->seq_id);
        }
    }
    for (const auto& sequence : removed.sequences) {
        if (sequence && sequence->status == SequenceStatus::RUNNING
            && emitted_sequence_ids.insert(sequence->seq_id).second) {
            survivor.sequences.push_back(sequence);
        }
    }
    auto merged_allocation = survivor.allocated_attention_ranks;
    for (int rank : removed.allocated_attention_ranks) {
        if (std::find(merged_allocation.begin(), merged_allocation.end(), rank) == merged_allocation.end()) {
            merged_allocation.push_back(rank);
        }
    }
    survivor.initial_batch_placements.insert(survivor.initial_batch_placements.end(),
                                             removed.initial_batch_placements.begin(),
                                             removed.initial_batch_placements.end());
    survivor.allocated_attention_ranks = std::move(merged_allocation);
    survivor.last_scale_up_step        = ls_schedule_step_;
    survivor.kv_candidate_target_dop   = -1;
    survivor.kv_candidate_stable_steps = 0;
    survivor.kv_candidate_member_ids.clear();
    survivor.kv_candidate_allocation.clear();
    for (const auto& seq : removed.sequences) {
        ls_seq_to_group_[seq->seq_id] = survivor_id;
    }

    auto& group_ids = ls_group_ids_by_dp_[survivor.dp_idx];
    group_ids.erase(std::remove(group_ids.begin(), group_ids.end(), removed_id), group_ids.end());
    ls_groups_.erase(removed_it);
}

void Scheduler::_remove_seq_from_ls_group(uint64_t seq_id)
{
    auto owner = ls_seq_to_group_.find(seq_id);
    if (owner == ls_seq_to_group_.end()) {
        return;
    }
    uint64_t group_id = owner->second;
    ls_seq_to_group_.erase(owner);
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
        group_it->second.kv_candidate_member_ids.clear();
        group_it->second.kv_candidate_allocation.clear();
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
        ls_arrival_order_by_seq_id_.erase(seq_id);
    }

    for (auto& [group_id, group] : ls_groups_) {
        (void)group_id;
        auto used = worker_state[group.dp_idx]->group_used_kv_tokens(group.sequences);
        group.allocated_attention_ranks.erase(
            std::remove_if(group.allocated_attention_ranks.begin(),
                           group.allocated_attention_ranks.end(),
                           [&](int rank) {
                               bool protected_role = std::any_of(group.sequences.begin(), group.sequences.end(), [&](const auto& sequence) {
                                   if (!sequence || sequence->status != SequenceStatus::RUNNING) {
                                       return false;
                                   }
                                   const auto& context = sequence->block_ctx(BlockContextSlot::ACTIVE);
                                   return context.master_sp_idx_ == rank
                                          || (context.pending_token_present_
                                              && context.pending_token_target_sp_ == rank);
                               });
                               return used[rank] == 0 && !protected_role;
                           }),
            group.allocated_attention_ranks.end());
    }
}

std::vector<std::vector<std::shared_ptr<Sequence>>> Scheduler::_schedule_ls_decode_admission()
{
    std::vector<std::vector<std::shared_ptr<Sequence>>> scheduled(attention_dp_);
    _reconcile_ls_groups();
    struct AdmissionPlan {
        LSAdmissionTargetKind               target_kind = LSAdmissionTargetKind::STANDALONE;
        std::vector<uint64_t>               planned_donors;
        std::vector<int>                    ranks;
        std::vector<std::vector<int>>       placements;
    };

    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        auto& waiting_queue = ls_waiting_by_dp_[dp_idx];
        if (waiting_queue.empty()) {
            continue;
        }

        std::vector<std::shared_ptr<Sequence>> selected_fifo;
        std::optional<uint64_t>                first_blocker;
        std::unordered_set<uint64_t>           selected_after_blocker;
        int64_t                                selected_tokens = 0;
        const bool allow_ooe = ls_num_ooe_[dp_idx] < ls_max_num_ooe_;
        const int running_count = static_cast<int>(_ls_running_sequences_in_pool(dp_idx).size());

        for (const auto& sequence : waiting_queue) {
            if (!sequence || sequence->assigned_dp != dp_idx
                || (sequence->status != SequenceStatus::WAITING
                    && sequence->status != SequenceStatus::PAUSED_OFFLOAD)) {
                latch_ls_fatal(LSFatalCode::POST_PUBLICATION_INVARIANT);
                throw LSSchedulerFatalError(LSFatalCode::POST_PUBLICATION_INVARIANT,
                                            "invalid sequence in an LS pool-local waiting queue");
            }
            const int64_t need = _ls_admission_need_tokens(*sequence);
            auto tentative = selected_fifo;
            tentative.push_back(sequence);
            bool feasible = static_cast<int>(tentative.size()) <= max_num_seqs_
                            && running_count + static_cast<int>(tentative.size()) <= ls_running_max_req_size_
                            && selected_tokens + need <= ls_admission_max_tokens_per_pool_
                            && _ls_pool_future_kv_fits(dp_idx, tentative)
                            && _ls_current_admission_fits(dp_idx, {sequence});
            if (feasible) {
                selected_fifo.push_back(sequence);
                selected_tokens += need;
                if (first_blocker.has_value()) {
                    selected_after_blocker.insert(sequence->seq_id);
                }
                continue;
            }
            if (!first_blocker.has_value()) {
                first_blocker = sequence->seq_id;
            }
            if (!allow_ooe) {
                break;
            }
        }

        std::optional<AdmissionPlan> plan;
        while (!selected_fifo.empty() && !plan.has_value()) {
            std::vector<std::shared_ptr<Sequence>> ordered = selected_fifo;
            std::stable_sort(ordered.begin(), ordered.end(), [&](const auto& lhs, const auto& rhs) {
                return _ls_admission_need_tokens(*lhs) > _ls_admission_need_tokens(*rhs);
            });

            // Empty-system feasibility is a permanent singleton/batch safety
            // boundary. It is evaluated after FIFO membership and sorting, but
            // any shrink is always applied to selected_fifo below.
            if (!_ls_batch_fits_empty_system(dp_idx, ordered)) {
                selected_fifo.pop_back();
                ++ls_step_atomic_no_fit_count_;
                continue;
            }

            auto idle_ranks = _ls_unallocated_ranks(dp_idx);
            int64_t idle_capacity = 0;
            for (int rank : idle_ranks) {
                idle_capacity += static_cast<int64_t>(worker_state[dp_idx]->block_manager.at(rank)->blocks().size())
                                 * Sequence::block_size;
            }
            int64_t admission_sum = 0;
            for (const auto& sequence : ordered) {
                admission_sum += _ls_admission_need_tokens(*sequence);
            }

            AdmissionPlan candidate;
            std::vector<int> rank_pool = idle_ranks;
            std::vector<std::shared_ptr<Sequence>> donor_sequences;
            std::vector<int> donor_allocation;
            if (admission_sum > idle_capacity) {
                candidate.target_kind = LSAdmissionTargetKind::CAPACITY_APPEND;
                const int64_t deficit = admission_sum - idle_capacity;
                struct Donor {
                    uint64_t group_id = 0;
                    int64_t  slack    = 0;
                    size_t   pool_order = 0;
                };
                std::vector<Donor> donors;
                for (size_t idx = 0; idx < ls_group_ids_by_dp_[dp_idx].size(); ++idx) {
                    uint64_t group_id = ls_group_ids_by_dp_[dp_idx][idx];
                    const auto& group = ls_groups_.at(group_id);
                    auto used = worker_state[dp_idx]->group_used_kv_tokens(group.sequences);
                    int64_t capacity = 0;
                    int64_t occupied = 0;
                    for (int rank : group.allocated_attention_ranks) {
                        capacity += static_cast<int64_t>(
                                        worker_state[dp_idx]->block_manager.at(rank)->blocks().size())
                                    * Sequence::block_size;
                        occupied += used[rank];
                    }
                    donors.push_back({group_id, capacity - occupied, idx});
                }
                std::stable_sort(donors.begin(), donors.end(), [](const Donor& lhs, const Donor& rhs) {
                    return lhs.slack < rhs.slack;
                });
                int64_t covered = 0;
                while (!donors.empty() && covered < deficit) {
                    Donor donor = donors.back();
                    donors.pop_back();
                    if (donor.slack <= 0) {
                        continue;
                    }
                    covered += donor.slack;
                    candidate.planned_donors.push_back(donor.group_id);
                    const auto& group = ls_groups_.at(donor.group_id);
                    donor_sequences.insert(donor_sequences.end(), group.sequences.begin(), group.sequences.end());
                    for (int rank : group.allocated_attention_ranks) {
                        if (std::find(rank_pool.begin(), rank_pool.end(), rank) == rank_pool.end()) {
                            rank_pool.push_back(rank);
                        }
                        if (std::find(donor_allocation.begin(), donor_allocation.end(), rank)
                            == donor_allocation.end()) {
                            donor_allocation.push_back(rank);
                        }
                    }
                }
                if (covered < deficit) {
                    selected_fifo.pop_back();
                    ++ls_step_atomic_no_fit_count_;
                    continue;
                }
            }

            auto placement = _plan_ls_initial_placement(
                dp_idx, ordered, rank_pool, donor_sequences, donor_allocation);
            if (!placement.has_value()) {
                selected_fifo.pop_back();
                ++ls_step_atomic_no_fit_count_;
                continue;
            }
            candidate.ranks      = std::move(placement->first);
            candidate.placements = std::move(placement->second);
            plan = std::move(candidate);
            selected_fifo = std::move(ordered);
        }
        if (!plan.has_value()) {
            continue;
        }

        // IDs are monotonic reservations. A failed transaction may leave a
        // gap but never publishes ownership for that ID.
        const uint64_t batch_id = next_ls_batch_id_++;
        const uint64_t new_group_id = next_ls_group_id_++;
        const uint64_t admission_order = next_ls_admission_order_++;
        const int failure_after_allocations =
            std::exchange(ls_admission_failure_after_allocations_for_test_, -1);
        const int failure_after_publications =
            std::exchange(ls_admission_failure_after_publications_for_test_, -1);
        bool publication_started = false;
        try {
            if (plan->ranks.empty() || plan->placements.size() != selected_fifo.size()) {
                throw std::runtime_error("invalid LS admission placement cardinality");
            }

            std::vector<SequenceStatus> original_statuses;
            std::vector<BlockContext>   placement_contexts;
            std::vector<bool>           bootstrap_finishes;
            std::vector<std::shared_ptr<Sequence>> survivors;
            std::vector<BlockContext>              survivor_contexts;
            std::vector<std::shared_ptr<Sequence>> finished;
            std::vector<BlockContext>              finished_contexts;
            original_statuses.reserve(selected_fifo.size());
            placement_contexts.reserve(selected_fifo.size());
            bootstrap_finishes.reserve(selected_fifo.size());
            survivors.reserve(selected_fifo.size());
            survivor_contexts.reserve(selected_fifo.size());
            finished.reserve(selected_fifo.size());
            finished_contexts.reserve(selected_fifo.size());

            for (size_t idx = 0; idx < selected_fifo.size(); ++idx) {
                const auto& sequence = selected_fifo[idx];
                original_statuses.push_back(sequence->status);
                BlockContext context(engine_id_, attention_sp_, attention_dp_);
                context.dp_idx_                  = dp_idx;
                context.master_sp_idx_           = plan->ranks[idx % plan->ranks.size()];
                context.pending_token_present_   = false;
                context.pending_token_target_sp_ = -1;
                context.num_dispatched_tokens    = plan->placements[idx];
                placement_contexts.push_back(context);

                const bool finishes = sequence->num_completed_tokens() + 1 >= sequence->max_tokens;
                bootstrap_finishes.push_back(finishes);
                if (finishes) {
                    finished.push_back(sequence);
                    finished_contexts.push_back(std::move(context));
                }
                else {
                    survivors.push_back(sequence);
                    survivor_contexts.push_back(std::move(context));
                }

                sequence->token_ids.reserve(sequence->token_ids.size() + 1);
                if (sequence->token_ids.capacity() <= sequence->token_ids.size()) {
                    throw std::runtime_error("LS admission failed to reserve dummy token storage");
                }
                if (sequence->metric && sequence->status == SequenceStatus::PAUSED_OFFLOAD
                    && sequence->metric->last_token_time.has_value()) {
                    sequence->metric->itl_samples.reserve(sequence->metric->itl_samples.size() + 1);
                    if (sequence->metric->itl_samples.capacity() <= sequence->metric->itl_samples.size()) {
                        throw std::runtime_error("LS readmission failed to reserve ITL storage");
                    }
                }
            }

            // Reserve the complete exact batch first. Bootstrap-finished
            // requests validate the same physical placement but abort their
            // reservation before survivor publication, matching the
            // post-bootstrap filter without ever publishing transient owner
            // state.
            std::optional<SPStateManager::PreparedLSInitialBatch> prepared_finished;
            std::optional<SPStateManager::PreparedLSInitialBatch> prepared_survivors;
            if (!finished.empty()) {
                prepared_finished.emplace(
                    worker_state[dp_idx]->prepare_ls_initial_batch(finished, finished_contexts));
            }
            if (!survivors.empty()) {
                prepared_survivors.emplace(
                    worker_state[dp_idx]->prepare_ls_initial_batch(survivors, survivor_contexts));
            }

            if (failure_after_allocations >= 0) {
                throw std::runtime_error("injected LS admission allocation-prepare failure");
            }

            std::vector<uint64_t> actual_donors;
            for (uint64_t donor_id : plan->planned_donors) {
                const auto& donor = ls_groups_.at(donor_id);
                bool overlap = false;
                for (size_t idx = 0; idx < selected_fifo.size(); ++idx) {
                    if (bootstrap_finishes[idx]) {
                        continue;
                    }
                    const auto& context = placement_contexts[idx];
                    for (int rank : donor.allocated_attention_ranks) {
                        if (context.num_dispatched_tokens[rank] > 0
                            || context.master_sp_idx_ == rank) {
                            overlap = true;
                            break;
                        }
                    }
                    if (overlap) {
                        break;
                    }
                }
                if (overlap) {
                    actual_donors.push_back(donor_id);
                }
            }

            std::optional<uint64_t> survivor_group;
            InitialBatchPlacement initial_record;
            initial_record.batch_id         = batch_id;
            initial_record.group_id         = new_group_id;
            initial_record.initial_kv_dop   = static_cast<int>(plan->ranks.size());
            initial_record.initial_kv_ranks = plan->ranks;
            initial_record.prompt_kv_tokens = plan->placements;
            for (size_t idx = 0; idx < selected_fifo.size(); ++idx) {
                initial_record.sequence_ids.push_back(selected_fifo[idx]->seq_id);
                initial_record.provisional_pending_targets.push_back(
                    plan->ranks[idx % plan->ranks.size()]);
            }
            initial_record.admission_order = admission_order;
            initial_record.admission_kind = plan->target_kind == LSAdmissionTargetKind::STANDALONE
                                                ? "standalone"
                                                : "capacity_append";

            // Every allocating standard-library operation below belongs to
            // prepare. The no-throw publication region swaps these complete
            // shadows after the prepared block transaction commits.
            auto prepared_groups           = ls_groups_;
            auto prepared_group_ids_by_dp   = ls_group_ids_by_dp_;
            auto prepared_seq_to_group      = ls_seq_to_group_;
            auto prepared_arrival_orders    = ls_arrival_order_by_seq_id_;
            auto prepared_waiting           = waiting_queue;
            auto prepared_running           = worker_state[dp_idx]->running;
            auto prepared_admission_records = ls_step_admission_records_;
            auto prepared_initial_records   = ls_step_initial_records_;

            if (!survivors.empty()) {
                DecodeGroupState group;
                group.group_id = new_group_id;
                group.dp_idx   = dp_idx;
                group.sequences = survivors;
                group.initial_batch_placements.push_back(initial_record);
                for (size_t idx = 0; idx < selected_fifo.size(); ++idx) {
                    if (bootstrap_finishes[idx]) {
                        continue;
                    }
                    const auto& sequence = selected_fifo[idx];
                    prepared_seq_to_group[sequence->seq_id] = new_group_id;
                    for (int rank = 0; rank < attention_sp_; ++rank) {
                        const auto& context = placement_contexts[idx];
                        if ((context.num_dispatched_tokens[rank] > 0
                             || context.master_sp_idx_ == rank)
                            && std::find(group.allocated_attention_ranks.begin(),
                                         group.allocated_attention_ranks.end(), rank)
                                   == group.allocated_attention_ranks.end()) {
                            group.allocated_attention_ranks.push_back(rank);
                        }
                    }
                }
                for (uint64_t donor_id : actual_donors) {
                    auto donor = prepared_groups.find(donor_id);
                    if (donor == prepared_groups.end()) {
                        throw std::runtime_error("planned LS admission donor disappeared");
                    }
                    for (const auto& sequence : donor->second.sequences) {
                        group.sequences.push_back(sequence);
                        prepared_seq_to_group[sequence->seq_id] = new_group_id;
                    }
                    group.initial_batch_placements.insert(group.initial_batch_placements.end(),
                                                          donor->second.initial_batch_placements.begin(),
                                                          donor->second.initial_batch_placements.end());
                    for (int rank : donor->second.allocated_attention_ranks) {
                        if (std::find(group.allocated_attention_ranks.begin(),
                                     group.allocated_attention_ranks.end(), rank)
                            == group.allocated_attention_ranks.end()) {
                            group.allocated_attention_ranks.push_back(rank);
                        }
                    }
                    auto& group_ids = prepared_group_ids_by_dp[dp_idx];
                    group_ids.erase(std::remove(group_ids.begin(), group_ids.end(), donor_id), group_ids.end());
                    prepared_groups.erase(donor);
                }
                group.last_scale_up_step = ls_schedule_step_;
                if (!prepared_groups.emplace(new_group_id, std::move(group)).second) {
                    throw std::runtime_error("reserved LS admission group ID already exists");
                }
                prepared_group_ids_by_dp[dp_idx].push_back(new_group_id);
                survivor_group = new_group_id;
            }

            for (size_t idx = 0; idx < selected_fifo.size(); ++idx) {
                const auto& sequence = selected_fifo[idx];
                auto queue_it = std::find(prepared_waiting.begin(), prepared_waiting.end(), sequence);
                if (queue_it == prepared_waiting.end()) {
                    throw std::runtime_error("committed LS sequence disappeared from its pool queue");
                }
                prepared_waiting.erase(queue_it);
                if (bootstrap_finishes[idx]) {
                    prepared_arrival_orders.erase(sequence->seq_id);
                }
                else {
                    prepared_running.push_back(sequence);
                }
                LSAdmissionRecord record;
                record.sequence = sequence;
                record.dp_idx = dp_idx;
                record.batch_id = batch_id;
                record.group_id_after_commit = bootstrap_finishes[idx] ? std::nullopt : survivor_group;
                record.admission_kind = original_statuses[idx] == SequenceStatus::PAUSED_OFFLOAD
                                            ? LSAdmissionKind::OFFLOAD_READMIT
                                            : LSAdmissionKind::FRESH;
                record.target_kind = plan->target_kind;
                record.planned_kv_dop = static_cast<int>(plan->ranks.size());
                record.planned_kv_ranks = plan->ranks;
                record.bootstrap_finished = bootstrap_finishes[idx];
                prepared_admission_records.push_back(std::move(record));
            }
            prepared_initial_records.push_back(initial_record);

            bool bypass = first_blocker.has_value()
                          && std::any_of(selected_fifo.begin(), selected_fifo.end(), [&](const auto& sequence) {
                                 return selected_after_blocker.count(sequence->seq_id) != 0;
                             });
            const int prepared_ooe = bypass ? ls_num_ooe_[dp_idx] + 1 : 0;

            if (failure_after_publications >= 0) {
                throw std::runtime_error("injected LS admission pre-publication failure");
            }

            if (prepared_finished && !prepared_finished->validate_precommit_noexcept()) {
                throw std::runtime_error("prepared bootstrap-finished LS reservation became stale");
            }
            if (prepared_survivors && !prepared_survivors->validate_precommit_noexcept()) {
                throw std::runtime_error("prepared survivor LS reservation became stale");
            }
            if (prepared_finished) {
                prepared_finished->abort_noexcept();
            }
            if (prepared_survivors && !prepared_survivors->validate_precommit_noexcept()) {
                throw std::runtime_error("prepared survivor LS reservation changed after finished abort");
            }

            const double bootstrap_commit_time =
                std::chrono::duration_cast<std::chrono::duration<double>>(
                    std::chrono::high_resolution_clock::now().time_since_epoch())
                    .count();

            // No operation below allocates or has a recoverable fault point.
            // PreparedLSInitialBatch already publishes prompt+dummy counters;
            // direct dummy append must not call add_running_tokens().
            publication_started = true;
            ls_step_publication_started_ = true;
            _mark_ls_pool_resource_mutated(dp_idx);
            if (prepared_survivors) {
                prepared_survivors->commit_noexcept();
            }
            for (size_t idx = 0; idx < selected_fifo.size(); ++idx) {
                auto& sequence = selected_fifo[idx];
                sequence->token_ids.push_back(0);
                sequence->last_token = 0;
                sequence->num_tokens++;
                if (bootstrap_finishes[idx]) {
                    sequence->status = SequenceStatus::FINISHED;
                }
                else {
                    auto& context = sequence->block_ctx(BlockContextSlot::ACTIVE);
                    const int master = context.master_sp_idx_;
                    context.num_dispatched_tokens[master]++;
                    context.pending_token_present_   = true;
                    context.pending_token_target_sp_ = master;
                    sequence->status = SequenceStatus::RUNNING;
                }
                if (sequence->metric) {
                    if (original_statuses[idx] == SequenceStatus::PAUSED_OFFLOAD) {
                        if (sequence->metric->last_token_time.has_value()) {
                            sequence->metric->itl_samples.push_back(
                                (bootstrap_commit_time - *sequence->metric->last_token_time) * 1000.0);
                        }
                    }
                    else {
                        if (!sequence->metric->first_scheduled_time.has_value()) {
                            sequence->metric->first_scheduled_time = bootstrap_commit_time;
                        }
                        if (!sequence->metric->decode_scheduled_time.has_value()) {
                            sequence->metric->decode_scheduled_time = bootstrap_commit_time;
                        }
                        if (!sequence->metric->first_token_time.has_value()) {
                            sequence->metric->first_token_time = bootstrap_commit_time;
                        }
                    }
                    sequence->metric->last_token_time = bootstrap_commit_time;
                    sequence->metric->num_generated_tokens++;
                }
            }

            worker_state[dp_idx]->running.swap(prepared_running);
            waiting_queue.swap(prepared_waiting);
            ls_groups_.swap(prepared_groups);
            ls_group_ids_by_dp_.swap(prepared_group_ids_by_dp);
            ls_seq_to_group_.swap(prepared_seq_to_group);
            ls_arrival_order_by_seq_id_.swap(prepared_arrival_orders);
            ls_step_admission_records_.swap(prepared_admission_records);
            ls_step_initial_records_.swap(prepared_initial_records);
            ls_num_ooe_[dp_idx] = prepared_ooe;
            if (!actual_donors.empty()) {
                ++ls_step_atomic_merge_count_;
            }
        }
        catch (const std::exception&) {
            if (publication_started) {
                latch_ls_fatal(LSFatalCode::POST_PUBLICATION_INVARIANT);
                throw LSSchedulerFatalError(LSFatalCode::POST_PUBLICATION_INVARIANT,
                                            "LS admission failed after entering no-throw publication");
            }
            ++ls_step_atomic_rollback_count_;
        }
    }
    return scheduled;
}

bool Scheduler::_ensure_ls_decode_memory_safety()
{
    _reconcile_ls_groups();
    auto prepared_groups          = ls_groups_;
    auto prepared_group_ids_by_dp = ls_group_ids_by_dp_;
    auto prepared_seq_to_group    = ls_seq_to_group_;
    bool graph_changed            = false;
    std::vector<bool> pool_graph_changed(attention_dp_, false);

    auto decode_idle_tokens = [&](int dp_idx, uint64_t group_id) -> int64_t {
        const auto& group = prepared_groups.at(group_id);
        auto        used  = worker_state[dp_idx]->group_used_kv_tokens(group.sequences);
        int64_t     capacity = 0;
        int64_t     occupied = 0;
        int64_t     running_requests = 0;
        for (int rank : group.allocated_attention_ranks) {
            capacity += static_cast<int64_t>(worker_state[dp_idx]->block_manager.at(rank)->blocks().size())
                        * Sequence::block_size;
            occupied += used[rank];
        }
        for (const auto& sequence : group.sequences) {
            if (sequence && sequence->status == SequenceStatus::RUNNING) {
                ++running_requests;
            }
        }
        return capacity - occupied - running_requests;
    };

    auto merge_shadow_groups = [&](uint64_t constrained_id, uint64_t donor_id) {
        auto constrained = prepared_groups.find(constrained_id);
        auto donor       = prepared_groups.find(donor_id);
        if (constrained == prepared_groups.end() || donor == prepared_groups.end()
            || constrained->second.dp_idx != donor->second.dp_idx) {
            throw std::runtime_error("invalid prospective LS memory-deficit group merge");
        }
        auto& survivor = constrained->second;
        std::unordered_set<uint64_t> emitted_ids;
        emitted_ids.reserve(survivor.sequences.size() + donor->second.sequences.size());
        for (const auto& sequence : survivor.sequences) {
            if (sequence) {
                emitted_ids.insert(sequence->seq_id);
            }
        }
        for (const auto& sequence : donor->second.sequences) {
            if (sequence && sequence->status == SequenceStatus::RUNNING
                && emitted_ids.insert(sequence->seq_id).second) {
                survivor.sequences.push_back(sequence);
            }
            if (sequence) {
                prepared_seq_to_group[sequence->seq_id] = constrained_id;
            }
        }
        survivor.initial_batch_placements.insert(survivor.initial_batch_placements.end(),
                                                 donor->second.initial_batch_placements.begin(),
                                                 donor->second.initial_batch_placements.end());
        for (int rank : donor->second.allocated_attention_ranks) {
            if (std::find(survivor.allocated_attention_ranks.begin(),
                          survivor.allocated_attention_ranks.end(), rank)
                == survivor.allocated_attention_ranks.end()) {
                survivor.allocated_attention_ranks.push_back(rank);
            }
        }
        survivor.last_scale_up_step        = ls_schedule_step_;
        survivor.kv_candidate_target_dop   = -1;
        survivor.kv_candidate_stable_steps = 0;
        survivor.kv_candidate_member_ids.clear();
        survivor.kv_candidate_allocation.clear();
        auto& group_ids = prepared_group_ids_by_dp[survivor.dp_idx];
        group_ids.erase(std::remove(group_ids.begin(), group_ids.end(), donor_id), group_ids.end());
        prepared_groups.erase(donor);
        graph_changed = true;
        pool_graph_changed[survivor.dp_idx] = true;
    };

    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        auto group_ids = prepared_group_ids_by_dp[dp_idx];

        // Stable zero-live cleanup is part of mandatory safety and does not
        // move historical KV or reorder the canonical group list.
        for (uint64_t group_id : group_ids) {
            auto& group = prepared_groups.at(group_id);
            auto  used  = worker_state[dp_idx]->group_used_kv_tokens(group.sequences);
            const size_t allocation_size_before = group.allocated_attention_ranks.size();
            group.allocated_attention_ranks.erase(
                std::remove_if(group.allocated_attention_ranks.begin(),
                               group.allocated_attention_ranks.end(),
                               [&](int rank) {
                                   bool protected_role = std::any_of(
                                       group.sequences.begin(), group.sequences.end(), [&](const auto& sequence) {
                                           if (!sequence || sequence->status != SequenceStatus::RUNNING) {
                                               return false;
                                           }
                                           const auto& context = sequence->block_ctx(BlockContextSlot::ACTIVE);
                                           return context.master_sp_idx_ == rank
                                                  || (context.pending_token_present_
                                                      && context.pending_token_target_sp_ == rank);
                                       });
                                   return used[rank] == 0 && !protected_role;
                               }),
                group.allocated_attention_ranks.end());
            if (group.allocated_attention_ranks.size() != allocation_size_before) {
                graph_changed = true;
                pool_graph_changed[dp_idx] = true;
            }
        }

        // LoongServe source order: stable-sort can/cannot lists by slack, pop
        // donors from the can tail, keep the constrained group as survivor,
        // append each feasible union without re-sorting, then consume truly
        // idle ranks in ascending SP order for the residual deficit.
        std::vector<uint64_t> can_decode;
        std::vector<uint64_t> cannot_decode;
        for (uint64_t group_id : group_ids) {
            (decode_idle_tokens(dp_idx, group_id) >= 0 ? can_decode : cannot_decode).push_back(group_id);
        }
        auto slack_less = [&](uint64_t lhs, uint64_t rhs) {
            return decode_idle_tokens(dp_idx, lhs) < decode_idle_tokens(dp_idx, rhs);
        };
        std::stable_sort(can_decode.begin(), can_decode.end(), slack_less);
        std::stable_sort(cannot_decode.begin(), cannot_decode.end(), slack_less);

        for (uint64_t constrained_id : cannot_decode) {
            int64_t slack = decode_idle_tokens(dp_idx, constrained_id);
            while (slack < 0 && !can_decode.empty()) {
                uint64_t donor_id = can_decode.back();
                can_decode.pop_back();
                if (donor_id == constrained_id || !prepared_groups.count(donor_id)) {
                    continue;
                }
                merge_shadow_groups(constrained_id, donor_id);
                slack = decode_idle_tokens(dp_idx, constrained_id);
            }
            if (slack < 0) {
                std::vector<int> idle_ranks;
                for (int rank = 0; rank < attention_sp_; ++rank) {
                    bool owned = false;
                    for (uint64_t group_id : prepared_group_ids_by_dp[dp_idx]) {
                        const auto& allocation = prepared_groups.at(group_id).allocated_attention_ranks;
                        if (std::find(allocation.begin(), allocation.end(), rank) != allocation.end()) {
                            owned = true;
                            break;
                        }
                    }
                    if (!owned) {
                        idle_ranks.push_back(rank);
                    }
                }
                auto& constrained = prepared_groups.at(constrained_id);
                for (int rank : idle_ranks) {
                    constrained.allocated_attention_ranks.push_back(rank);
                    graph_changed = true;
                    pool_graph_changed[dp_idx] = true;
                    slack += static_cast<int64_t>(worker_state[dp_idx]->block_manager.at(rank)->blocks().size())
                             * Sequence::block_size;
                    constrained.last_scale_up_step        = ls_schedule_step_;
                    constrained.kv_candidate_target_dop   = -1;
                    constrained.kv_candidate_stable_steps = 0;
                    constrained.kv_candidate_member_ids.clear();
                    constrained.kv_candidate_allocation.clear();
                    if (slack >= 0) {
                        break;
                    }
                }
            }
            if (slack < 0) {
                if (!_ls_offload_one_victim(dp_idx, "OFFLOAD memory-token deficit")) {
                    latch_ls_fatal(LSFatalCode::UNRECOVERABLE_CAPACITY);
                    throw LSSchedulerFatalError(LSFatalCode::UNRECOVERABLE_CAPACITY,
                                                "pool capacity cannot recover a Decode group");
                }
                return false;
            }
            can_decode.push_back(constrained_id);
        }
        if (prepared_group_ids_by_dp[dp_idx] != can_decode) {
            graph_changed = true;
            pool_graph_changed[dp_idx] = true;
        }
        prepared_group_ids_by_dp[dp_idx] = std::move(can_decode);
    }

    // All pools passed the capacity boundary. Publication is a single
    // allocation-free graph swap; an OFFLOAD path above always operated on
    // the untouched stable graph.
    if (graph_changed) {
        ls_step_publication_started_ = true;
        for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
            if (pool_graph_changed[dp_idx]) {
                _mark_ls_pool_resource_mutated(dp_idx);
            }
        }
        ls_groups_.swap(prepared_groups);
        ls_group_ids_by_dp_.swap(prepared_group_ids_by_dp);
        ls_seq_to_group_.swap(prepared_seq_to_group);
    }
    return true;
}

bool Scheduler::_ls_offload_one_victim(int dp_idx, const std::string& reason)
{
    if (dp_idx < 0 || dp_idx >= attention_dp_ || ls_step_offload_committed_) {
        return false;
    }
    std::vector<std::shared_ptr<Sequence>> candidates = _ls_running_sequences_in_pool(dp_idx);
    std::stable_sort(candidates.begin(), candidates.end(), [&](const auto& lhs, const auto& rhs) {
        uint64_t lhs_arrival = ls_arrival_order_by_seq_id_.at(lhs->seq_id);
        uint64_t rhs_arrival = ls_arrival_order_by_seq_id_.at(rhs->seq_id);
        return std::tie(lhs_arrival, lhs->seq_id) > std::tie(rhs_arrival, rhs->seq_id);
    });

    auto context_empty = [](const BlockContext& context) {
        return !context.pending_token_present_ && context.pending_token_target_sp_ == -1
               && context.block_location.empty()
               && std::all_of(context.sp_block_table.begin(), context.sp_block_table.end(), [](const auto& table) {
                      return table.empty();
                  })
               && std::all_of(context.num_dispatched_tokens.begin(),
                              context.num_dispatched_tokens.end(),
                              [](int tokens) { return tokens == 0; });
    };

    for (const auto& victim : candidates) {
        if (!victim) {
            continue;
        }
        auto paused_shadow = std::make_shared<Sequence>(*victim);
        paused_shadow->status = SequenceStatus::PAUSED_OFFLOAD;
        if (!context_empty(victim->block_ctx(BlockContextSlot::MIGRATE))
            || !context_empty(victim->block_ctx(BlockContextSlot::SWAP))
            || !_ls_batch_fits_empty_system(dp_idx, {paused_shadow})) {
            continue;
        }
        auto group_owner = ls_seq_to_group_.find(victim->seq_id);
        if (group_owner == ls_seq_to_group_.end()) {
            continue;
        }

        // Build every scheduler publication shadow before reserving the ACTIVE
        // release. OFFLOAD commit is then only an allocator publication,
        // scalar status update and no-throw container swaps.
        auto prepared_running           = worker_state[dp_idx]->running;
        auto prepared_waiting           = ls_waiting_by_dp_[dp_idx];
        auto prepared_groups            = ls_groups_;
        auto prepared_group_ids_by_dp    = ls_group_ids_by_dp_;
        auto prepared_seq_to_group       = ls_seq_to_group_;
        auto prepared_preempted_ids      = ls_step_preempted_sequence_ids_;
        auto prepared_preemption_reasons = ls_step_preemption_reasons_;

        prepared_running.erase(std::remove(prepared_running.begin(), prepared_running.end(), victim),
                               prepared_running.end());
        prepared_waiting.push_front(victim);
        prepared_preempted_ids.push_back(victim->seq_id);
        prepared_preemption_reasons.push_back("OFFLOAD: " + reason);

        const uint64_t group_id = group_owner->second;
        prepared_seq_to_group.erase(victim->seq_id);
        auto prepared_group = prepared_groups.find(group_id);
        if (prepared_group == prepared_groups.end()) {
            continue;
        }
        auto& remaining = prepared_group->second.sequences;
        remaining.erase(std::remove_if(remaining.begin(), remaining.end(), [&](const auto& sequence) {
                            return !sequence || sequence->seq_id == victim->seq_id;
                        }),
                        remaining.end());
        if (remaining.empty()) {
            auto& group_ids = prepared_group_ids_by_dp[dp_idx];
            group_ids.erase(std::remove(group_ids.begin(), group_ids.end(), group_id), group_ids.end());
            prepared_groups.erase(prepared_group);
        }
        else {
            auto& group = prepared_group->second;
            group.kv_candidate_target_dop   = -1;
            group.kv_candidate_stable_steps = 0;
            group.kv_candidate_member_ids.clear();
            group.kv_candidate_allocation.clear();
            auto used = worker_state[dp_idx]->group_used_kv_tokens(group.sequences);
            group.allocated_attention_ranks.erase(
                std::remove_if(group.allocated_attention_ranks.begin(),
                               group.allocated_attention_ranks.end(),
                               [&](int rank) {
                                   bool protected_role = std::any_of(
                                       group.sequences.begin(), group.sequences.end(), [&](const auto& sequence) {
                                           if (!sequence || sequence->status != SequenceStatus::RUNNING) {
                                               return false;
                                           }
                                           const auto& context = sequence->block_ctx(BlockContextSlot::ACTIVE);
                                           return context.master_sp_idx_ == rank
                                                  || (context.pending_token_present_
                                                      && context.pending_token_target_sp_ == rank);
                                       });
                                   return used[rank] == 0 && !protected_role;
                               }),
                group.allocated_attention_ranks.end());
        }

        auto prepared_release = worker_state[dp_idx]->prepare_ls_release(victim);
        if (!prepared_release.validate_precommit_noexcept()) {
            continue;
        }
        ls_step_publication_started_ = true;
        _mark_ls_pool_resource_mutated(dp_idx);
        prepared_release.commit_noexcept();
        victim->status = SequenceStatus::PAUSED_OFFLOAD;
        worker_state[dp_idx]->running.swap(prepared_running);
        ls_waiting_by_dp_[dp_idx].swap(prepared_waiting);
        ls_groups_.swap(prepared_groups);
        ls_group_ids_by_dp_.swap(prepared_group_ids_by_dp);
        ls_seq_to_group_.swap(prepared_seq_to_group);
        ls_step_preempted_sequence_ids_.swap(prepared_preempted_ids);
        ls_step_preemption_reasons_.swap(prepared_preemption_reasons);
        ls_step_offload_committed_ = true;

        const auto& active = victim->block_ctx(BlockContextSlot::ACTIVE);
        if (victim->assigned_dp != dp_idx || active.dp_idx_ != dp_idx || active.master_sp_idx_ != -1
            || !context_empty(active)) {
            latch_ls_fatal(LSFatalCode::POST_PUBLICATION_INVARIANT);
            throw LSSchedulerFatalError(LSFatalCode::POST_PUBLICATION_INVARIANT,
                                        "prepared LS OFFLOAD published an invalid paused request");
        }
        return true;
    }
    return false;
}

std::vector<std::vector<std::shared_ptr<Sequence>>>
Scheduler::_schedule_ls_decode(const std::unordered_set<uint64_t>* eligible_sequence_ids)
try
{
    auto planning_start = std::chrono::steady_clock::now();
    _reconcile_ls_groups();
    auto is_eligible = [&](const std::shared_ptr<Sequence>& sequence) {
        return sequence && sequence->status == SequenceStatus::RUNNING
               && (!eligible_sequence_ids || eligible_sequence_ids->count(sequence->seq_id) != 0);
    };

    auto fail_required_decode = [&](const std::string& message) -> void {
        const LSFatalCode code = ls_step_publication_started_
                                     ? LSFatalCode::POST_PUBLICATION_INVARIANT
                                     : LSFatalCode::DECODE_PREPARE_OR_VALIDATE_FAILED;
        latch_ls_fatal(code);
        throw LSSchedulerFatalError(code, message);
    };
    auto is_capacity_no_fit = [](const std::string& reason) {
        return reason.find("capacity") != std::string::npos
               || reason.find("infeasible") != std::string::npos
               || reason.find("exhausted") != std::string::npos;
    };

    struct PendingGroupPlan {
        uint64_t                               group_id = 0;
        std::vector<std::shared_ptr<Sequence>> requests;
        std::vector<uint64_t>                  sequence_ids;
        SPStateManager::LSDecodeMasterPlan     plan;
        std::vector<int>                       reused_passive;
    };
    struct PendingDPPlan {
        std::vector<PendingGroupPlan>                              groups;
        std::vector<std::shared_ptr<Sequence>>                     combined_requests;
        SPStateManager::LSDecodeMasterPlan                         combined_plan;
        std::optional<SPStateManager::PreparedLSIterationMasterPlan> prepared;
    };
    std::vector<PendingDPPlan> pending_by_dp(attention_dp_);

    // Policy planning is mutation-free. In particular, no DP publishes a
    // pending/master transition until every other DP has produced a valid
    // required Decode component.
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        const auto group_ids = ls_group_ids_by_dp_[dp_idx];
        std::unordered_set<int> owned;
        for (uint64_t group_id : group_ids) {
            for (int rank : ls_groups_.at(group_id).allocated_attention_ranks) {
                owned.insert(rank);
            }
        }

        auto& pending_dp = pending_by_dp[dp_idx];
        pending_dp.groups.reserve(group_ids.size());
        for (uint64_t group_id : group_ids) {
            const auto& group = ls_groups_.at(group_id);
            std::vector<std::shared_ptr<Sequence>> requests;
            for (const auto& seq : group.sequences) {
                if (is_eligible(seq)) {
                    requests.push_back(seq);
                }
            }
            if (requests.empty()) {
                continue;
            }

            std::vector<int> extras = _ls_unallocated_ranks(dp_idx);
            extras.erase(std::remove_if(extras.begin(), extras.end(), [&](int rank) { return owned.count(rank); }),
                         extras.end());
            std::sort(extras.begin(), extras.end(), std::greater<int>());
            auto plan = worker_state[dp_idx]->plan_iteration_masters_source_greedy(
                requests,
                group.allocated_attention_ranks,
                extras,
                ls_min_comp_bound_decoding_batch_size_,
                ls_decode_enable_memory_scale_up_,
                ls_step_execution_loop_count_);
            if (!plan.success) {
                const std::string reason = plan.failure_reason.empty() ? "planner returned failure" : plan.failure_reason;
                if (!is_capacity_no_fit(reason)) {
                    fail_required_decode("LS required Decode planner failed internally: " + reason);
                }
                if (ls_step_publication_started_) {
                    fail_required_decode("LS required Decode capacity no-fit followed an earlier publication");
                }
                std::cerr << "LS-Decode-Core capacity no-fit for group=" << group_id << ": " << reason
                          << std::endl;
                if (!_ls_offload_one_victim(dp_idx, reason)) {
                    latch_ls_fatal(LSFatalCode::UNRECOVERABLE_CAPACITY);
                    throw LSSchedulerFatalError(LSFatalCode::UNRECOVERABLE_CAPACITY,
                                                "no recoverable OFFLOAD victim for an LS Decode capacity deficit");
                }
                ls_step_planning_latency_ms_ =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - planning_start)
                        .count();
                return std::vector<std::vector<std::shared_ptr<Sequence>>>(attention_dp_);
            }

            std::string validation_error;
            if (!worker_state[dp_idx]->validate_iteration_master_plan(
                    requests, plan, &validation_error, ls_step_execution_loop_count_)) {
                fail_required_decode("LS required Decode plan validation failed: " + validation_error);
            }
            for (int rank : plan.allocation) {
                owned.insert(rank);
            }

            std::vector<int> reused_passive;
            for (int rank : plan.master_ranks) {
                bool was_last_master =
                    std::find(group.last_iteration_masters.begin(), group.last_iteration_masters.end(), rank)
                    != group.last_iteration_masters.end();
                if (!was_last_master && plan.group_used_kv_tokens.at(rank) > 0
                    && std::find(group.allocated_attention_ranks.begin(),
                                 group.allocated_attention_ranks.end(), rank)
                           != group.allocated_attention_ranks.end()) {
                    reused_passive.push_back(rank);
                }
            }
            std::vector<uint64_t> sequence_ids;
            sequence_ids.reserve(requests.size());
            for (const auto& request : requests) {
                sequence_ids.push_back(request->seq_id);
            }
            pending_dp.groups.push_back(
                {group_id, std::move(requests), std::move(sequence_ids), std::move(plan), std::move(reused_passive)});
        }

        if (pending_dp.groups.empty()) {
            continue;
        }

        auto& combined = pending_dp.combined_plan;
        combined.success             = true;
        combined.scale_reason        = "combined_pool_iteration";
        combined.assignment_strategy = "source_greedy";
        std::vector<int> combined_master_load(attention_sp_, 0);
        std::unordered_set<int> combined_allocation;
        std::unordered_set<int> combined_new_allocation;
        for (const auto& pending : pending_dp.groups) {
            pending_dp.combined_requests.insert(pending_dp.combined_requests.end(),
                                                pending.requests.begin(), pending.requests.end());
            combined.sequence_master_ranks.insert(combined.sequence_master_ranks.end(),
                                                  pending.plan.sequence_master_ranks.begin(),
                                                  pending.plan.sequence_master_ranks.end());
            for (int rank : pending.plan.allocation) {
                if (combined_allocation.insert(rank).second) {
                    combined.allocation.push_back(rank);
                }
            }
            for (int rank : pending.plan.new_allocation_ranks) {
                if (combined_new_allocation.insert(rank).second) {
                    combined.new_allocation_ranks.push_back(rank);
                }
            }
            for (int master : pending.plan.sequence_master_ranks) {
                if (master < 0 || master >= attention_sp_) {
                    fail_required_decode("LS required Decode plan produced an invalid master rank");
                }
                combined_master_load[master]++;
            }
        }
        for (int rank = 0; rank < attention_sp_; ++rank) {
            if (combined_master_load[rank] > 0) {
                combined.master_ranks.push_back(rank);
                combined.master_batch_sizes.push_back(combined_master_load[rank]);
            }
        }
        combined.group_used_kv_tokens =
            worker_state[dp_idx]->group_used_kv_tokens(pending_dp.combined_requests);
        combined.group_used_kv_blocks =
            worker_state[dp_idx]->group_used_kv_blocks(pending_dp.combined_requests);
        std::string combined_validation_error;
        if (!worker_state[dp_idx]->validate_iteration_master_plan(
                pending_dp.combined_requests,
                combined,
                &combined_validation_error,
                ls_step_execution_loop_count_)) {
            fail_required_decode("LS combined pool Decode validation failed: " + combined_validation_error);
        }
    }

    // Build every scheduler-owned publication container before reserving or
    // publishing physical iteration state.
    auto prepared_groups = ls_groups_;
    std::vector<std::vector<std::shared_ptr<Sequence>>> prepared_scheduled(attention_dp_);
    std::vector<std::vector<uint64_t>> prepared_real_decode_ids(attention_dp_);
    std::vector<SPStateManager::LSDecodeMasterPlan> prepared_group_plans;
    std::vector<uint64_t> prepared_group_plan_ids;
    std::vector<std::vector<uint64_t>> prepared_group_plan_sequence_ids;
    std::vector<std::vector<int>> prepared_reused_passive_masters;
    size_t total_group_plans = 0;
    for (const auto& pending_dp : pending_by_dp) {
        total_group_plans += pending_dp.groups.size();
    }
    prepared_group_plans.reserve(total_group_plans);
    prepared_group_plan_ids.reserve(total_group_plans);
    prepared_group_plan_sequence_ids.reserve(total_group_plans);
    prepared_reused_passive_masters.reserve(total_group_plans);

    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        std::vector<int> master_load(attention_sp_, 0);
        for (const auto& pending : pending_by_dp[dp_idx].groups) {
            auto group_it = prepared_groups.find(pending.group_id);
            if (group_it == prepared_groups.end()) {
                fail_required_decode("LS required Decode group disappeared during prepare");
            }
            auto& group = group_it->second;
            const bool allocation_grew = pending.plan.allocation.size() > group.allocated_attention_ranks.size();
            group.allocated_attention_ranks = pending.plan.allocation;
            group.last_iteration_masters    = pending.plan.master_ranks;
            if (allocation_grew) {
                group.last_scale_up_step        = ls_schedule_step_;
                group.kv_candidate_target_dop   = -1;
                group.kv_candidate_stable_steps = 0;
                group.kv_candidate_member_ids.clear();
                group.kv_candidate_allocation.clear();
            }
            for (size_t sequence_idx = 0; sequence_idx < pending.requests.size(); ++sequence_idx) {
                const int master = pending.plan.sequence_master_ranks.at(sequence_idx);
                prepared_scheduled[dp_idx].push_back(pending.requests[sequence_idx]);
                prepared_real_decode_ids[dp_idx].push_back(pending.requests[sequence_idx]->seq_id);
                master_load[master]++;
            }
            prepared_group_plan_ids.push_back(pending.group_id);
            prepared_group_plan_sequence_ids.push_back(pending.sequence_ids);
            prepared_group_plans.push_back(pending.plan);
            prepared_reused_passive_masters.push_back(pending.reused_passive);
        }
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            if (master_load[sp_idx] == 0) {
                prepared_scheduled[dp_idx].push_back(worker_state[dp_idx]->dummy_seqs[sp_idx]);
            }
        }
    }

    // A single prepared transaction per DP aggregates every disjoint group so
    // manager counters and full-rank pending-block transfers validate against
    // one prospective pool state. RAII aborts all earlier DP reservations if a
    // later prepare throws.
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        auto& pending_dp = pending_by_dp[dp_idx];
        if (!pending_dp.combined_requests.empty()) {
            pending_dp.prepared.emplace(worker_state[dp_idx]->prepare_iteration_master_plan(
                pending_dp.combined_requests, pending_dp.combined_plan, ls_step_execution_loop_count_));
        }
    }
    for (const auto& pending_dp : pending_by_dp) {
        if (pending_dp.prepared && !pending_dp.prepared->validate_precommit_noexcept()) {
            fail_required_decode("prepared LS pool Decode transaction became stale before publication");
        }
    }

    // From here through the swaps there is no recoverable fault point.
    ls_step_publication_started_ = true;
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        auto& pending_dp = pending_by_dp[dp_idx];
        if (pending_dp.prepared) {
            _mark_ls_pool_resource_mutated(dp_idx);
            pending_dp.prepared->commit_noexcept();
        }
    }
    ls_groups_.swap(prepared_groups);
    ls_step_real_decode_ids_by_dp_.swap(prepared_real_decode_ids);
    ls_step_group_plans_.swap(prepared_group_plans);
    ls_step_group_plan_ids_.swap(prepared_group_plan_ids);
    ls_step_group_plan_sequence_ids_.swap(prepared_group_plan_sequence_ids);
    ls_step_reused_passive_masters_.swap(prepared_reused_passive_masters);

    bool postpublication_valid = true;
    for (int dp_idx = 0; dp_idx < attention_dp_ && postpublication_valid; ++dp_idx) {
        const auto& pending_dp = pending_by_dp[dp_idx];
        if (pending_dp.prepared
            && pending_dp.prepared->state()
                   != SPStateManager::PreparedLSIterationMasterPlan::State::COMMITTED) {
            postpublication_valid = false;
            break;
        }
        size_t combined_idx = 0;
        for (const auto& pending : pending_dp.groups) {
            for (const auto& sequence : pending.requests) {
                if (!sequence || combined_idx >= pending_dp.combined_plan.sequence_master_ranks.size()) {
                    postpublication_valid = false;
                    break;
                }
                const int master = pending_dp.combined_plan.sequence_master_ranks[combined_idx++];
                const auto& context = sequence->block_ctx(BlockContextSlot::ACTIVE);
                if (sequence->status != SequenceStatus::RUNNING || context.master_sp_idx_ != master
                    || !context.pending_token_present_ || context.pending_token_target_sp_ != master) {
                    postpublication_valid = false;
                    break;
                }
            }
            if (!postpublication_valid) {
                break;
            }
        }
        if (combined_idx != pending_dp.combined_requests.size()) {
            postpublication_valid = false;
        }
    }
    if (!postpublication_valid) {
        latch_ls_fatal(LSFatalCode::POST_PUBLICATION_INVARIANT);
        throw LSSchedulerFatalError(LSFatalCode::POST_PUBLICATION_INVARIANT,
                                    "LS pool Decode publication failed its postcondition");
    }

    ls_step_planning_latency_ms_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - planning_start).count();
    return prepared_scheduled;
}
catch (const LSSchedulerFatalError&) {
    throw;
}
catch (...) {
    const LSFatalCode code = ls_step_publication_started_ ? LSFatalCode::POST_PUBLICATION_INVARIANT :
                                                            LSFatalCode::DECODE_PREPARE_OR_VALIDATE_FAILED;
    latch_ls_fatal(code);
    throw LSSchedulerFatalError(code,
                                code == LSFatalCode::POST_PUBLICATION_INVARIANT ?
                                    "unexpected failure after LS Decode publication began" :
                                    "unexpected failure while preparing required LS Decode");
}

std::vector<std::vector<std::shared_ptr<Sequence>>> Scheduler::_schedule_ls_combined_pool_step(
    const std::unordered_set<uint64_t>&                     eligible_sequence_ids,
    std::shared_ptr<SPStateManager::LSKVConsolidationPlan>* kv_consolidation_plan)
try {
    if (!kv_consolidation_plan) {
        throw std::invalid_argument("LS combined pool step requires a consolidation-plan output");
    }
    kv_consolidation_plan->reset();
    const auto planning_start = std::chrono::steady_clock::now();

    enum class AttemptFailureKind {
        NONE,
        TENTATIVE,
        CAPACITY_NO_FIT,
        INTERNAL,
        KV_CONSOLIDATION
    };
    struct AttemptError {
        AttemptFailureKind kind   = AttemptFailureKind::INTERNAL;
        int                dp_idx = -1;
        std::string        reason;
    };
    struct AdmissionPlan {
        LSAdmissionTargetKind         target_kind = LSAdmissionTargetKind::STANDALONE;
        std::vector<uint64_t>         planned_donors;
        std::vector<int>              ranks;
        std::vector<std::vector<int>> placements;
    };
    struct AdmissionSchedulerShadow {
        std::unordered_map<uint64_t, DecodeGroupState>     groups;
        std::vector<std::vector<uint64_t>>                 group_ids_by_dp;
        std::unordered_map<uint64_t, uint64_t>             seq_to_group;
        std::unordered_map<uint64_t, uint64_t>             arrival_orders;
        std::vector<std::list<std::shared_ptr<Sequence>>>  waiting;
        std::vector<std::deque<std::shared_ptr<Sequence>>> running;
        std::vector<int>                                   ooe;
        std::vector<LSAdmissionRecord>                     admission_records;
        std::vector<InitialBatchPlacement>                 initial_records;
        std::vector<bool>                                  pool_resource_changed;
        uint64_t                                           atomic_merge_count      = 0;
        bool                                               has_admission           = false;
        bool                                               had_tentative_admission = false;
    };
    struct PreparedAdmission {
        int                                                   dp_idx = -1;
        uint64_t                                              batch_id = 0;
        uint64_t                                              merge_count_delta = 0;
        std::vector<std::shared_ptr<Sequence>>                sequences;
        std::vector<SequenceStatus>                           original_statuses;
        std::vector<bool>                                     bootstrap_finishes;
        std::optional<SPStateManager::PreparedLSInitialBatch> prepared_survivors;
        std::unique_ptr<AdmissionSchedulerShadow>             rollback_base;
        double                                                bootstrap_commit_time = 0.0;
    };
    struct PendingGroupPlan {
        uint64_t                               group_id = 0;
        std::vector<std::shared_ptr<Sequence>> requests;
        std::vector<uint64_t>                  sequence_ids;
        SPStateManager::LSDecodeMasterPlan     plan;
        std::vector<int>                       reused_passive;
    };
    struct PendingDPPlan {
        std::vector<PendingGroupPlan>                                groups;
        std::vector<std::shared_ptr<Sequence>>                       combined_requests;
        SPStateManager::LSDecodeMasterPlan                           combined_plan;
        std::optional<SPStateManager::PreparedLSIterationMasterPlan> prepared;
    };
    struct PreparedStep {
        std::unordered_map<uint64_t, DecodeGroupState>      groups;
        std::vector<std::vector<uint64_t>>                  group_ids_by_dp;
        std::unordered_map<uint64_t, uint64_t>              seq_to_group;
        std::unordered_map<uint64_t, uint64_t>              arrival_orders;
        std::vector<std::list<std::shared_ptr<Sequence>>>   waiting;
        std::vector<std::deque<std::shared_ptr<Sequence>>>  running;
        std::vector<int>                                    ooe;
        std::vector<LSAdmissionRecord>                      admission_records;
        std::vector<InitialBatchPlacement>                  initial_records;
        std::vector<std::optional<PreparedAdmission>>       admissions;
        std::vector<PendingDPPlan>                          decode_by_dp;
        std::vector<std::vector<std::shared_ptr<Sequence>>> scheduled;
        std::vector<std::vector<uint64_t>>                  real_decode_ids;
        std::vector<SPStateManager::LSDecodeMasterPlan>     group_plans;
        std::vector<uint64_t>                               group_plan_ids;
        std::vector<std::vector<uint64_t>>                  group_plan_sequence_ids;
        std::vector<std::vector<int>>                       reused_passive_masters;
        std::vector<bool>                                   pool_resource_changed;
        uint64_t                                            atomic_merge_count      = 0;
        bool                                                mandatory_graph_changed = false;
        bool                                                has_admission           = false;
        bool                                                had_tentative_admission = false;
    };
    struct AttemptResult {
        std::unique_ptr<PreparedStep>                          step;
        AttemptFailureKind                                     failure = AttemptFailureKind::NONE;
        int                                                    dp_idx  = -1;
        std::string                                            reason;
        std::shared_ptr<SPStateManager::LSKVConsolidationPlan> consolidation;
        bool                                                   had_tentative_admission = false;
    };
    auto is_explicit_decode_capacity_no_fit = [](std::string_view reason) noexcept {
        // Only authoritative planner NO_FIT outcomes may select OFFLOAD.
        // Prepare/validation exceptions are consistency failures even when
        // their diagnostic happens to mention blocks or capacity.
        return reason.starts_with("receiver capacity proven infeasible:")
               || reason.starts_with("decode metadata capacity cannot cover")
               || reason.starts_with("append capacity cannot ")
               || reason.starts_with("append/receiver joint capacity proven infeasible after");
    };

    int                                                injected_allocation_failure  = -2;
    int                                                injected_publication_failure = -2;
    bool                                               injection_captured           = false;
    auto prepare_attempt = [&](bool include_admission) -> AttemptResult {
        AttemptResult result;
        auto          step      = std::make_unique<PreparedStep>();
        step->groups            = ls_groups_;
        step->group_ids_by_dp   = ls_group_ids_by_dp_;
        step->seq_to_group      = ls_seq_to_group_;
        step->arrival_orders    = ls_arrival_order_by_seq_id_;
        step->waiting           = ls_waiting_by_dp_;
        step->ooe               = ls_num_ooe_;
        step->admission_records = ls_step_admission_records_;
        step->initial_records   = ls_step_initial_records_;
        step->admissions.resize(attention_dp_);
        step->decode_by_dp.resize(attention_dp_);
        step->scheduled.resize(attention_dp_);
        step->real_decode_ids.resize(attention_dp_);
        step->pool_resource_changed.assign(attention_dp_, false);
        step->running.reserve(attention_dp_);
        for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
            step->running.push_back(worker_state[dp_idx]->running);
        }

        auto capture_admission_scheduler_shadow = [&]() {
            auto shadow                     = std::make_unique<AdmissionSchedulerShadow>();
            shadow->groups                  = step->groups;
            shadow->group_ids_by_dp         = step->group_ids_by_dp;
            shadow->seq_to_group            = step->seq_to_group;
            shadow->arrival_orders          = step->arrival_orders;
            shadow->waiting                 = step->waiting;
            shadow->running                 = step->running;
            shadow->ooe                     = step->ooe;
            shadow->admission_records       = step->admission_records;
            shadow->initial_records         = step->initial_records;
            shadow->pool_resource_changed   = step->pool_resource_changed;
            shadow->atomic_merge_count      = step->atomic_merge_count;
            shadow->has_admission           = step->has_admission;
            shadow->had_tentative_admission = step->had_tentative_admission;
            return shadow;
        };
        auto restore_admission_scheduler_shadow = [&](AdmissionSchedulerShadow& shadow) noexcept {
            step->groups.swap(shadow.groups);
            step->group_ids_by_dp.swap(shadow.group_ids_by_dp);
            step->seq_to_group.swap(shadow.seq_to_group);
            step->arrival_orders.swap(shadow.arrival_orders);
            step->waiting.swap(shadow.waiting);
            step->running.swap(shadow.running);
            step->ooe.swap(shadow.ooe);
            step->admission_records.swap(shadow.admission_records);
            step->initial_records.swap(shadow.initial_records);
            step->pool_resource_changed.swap(shadow.pool_resource_changed);
            std::swap(step->atomic_merge_count, shadow.atomic_merge_count);
            std::swap(step->has_admission, shadow.has_admission);
            std::swap(step->had_tentative_admission, shadow.had_tentative_admission);
        };

        try {
            auto rank_is_idle_in_graph = [&](int dp_idx, int rank) {
                if (dp_idx < 0 || dp_idx >= attention_dp_ || rank < 0 || rank >= attention_sp_) {
                    return false;
                }
                for (uint64_t group_id : step->group_ids_by_dp[dp_idx]) {
                    auto group = step->groups.find(group_id);
                    if (group == step->groups.end() || group->second.dp_idx != dp_idx) {
                        throw std::runtime_error("prospective LS graph has invalid canonical group ownership");
                    }
                    if (std::find(group->second.allocated_attention_ranks.begin(),
                                  group->second.allocated_attention_ranks.end(),
                                  rank)
                        != group->second.allocated_attention_ranks.end()) {
                        return false;
                    }
                }
                for (const auto& sequence : worker_state[dp_idx]->running) {
                    if (!sequence || sequence->status != SequenceStatus::RUNNING) {
                        continue;
                    }
                    const auto& context = sequence->block_ctx(BlockContextSlot::ACTIVE);
                    if (context.master_sp_idx_ == rank
                        || (context.pending_token_present_ && context.pending_token_target_sp_ == rank)
                        || (rank < static_cast<int>(context.num_dispatched_tokens.size())
                            && context.num_dispatched_tokens[rank] != 0)
                        || (rank < static_cast<int>(context.sp_block_table.size())
                            && !context.sp_block_table[rank].empty())) {
                        return false;
                    }
                }
                return true;
            };
            auto unallocated_ranks = [&](int dp_idx) {
                std::vector<int> ranks;
                for (int rank = 0; rank < attention_sp_; ++rank) {
                    if (rank_is_idle_in_graph(dp_idx, rank)) {
                        ranks.push_back(rank);
                    }
                }
                return ranks;
            };
            auto decode_idle_tokens = [&](int dp_idx, uint64_t group_id) -> int64_t {
                const auto& group            = step->groups.at(group_id);
                auto        used             = worker_state[dp_idx]->group_used_kv_tokens(group.sequences);
                int64_t     capacity         = 0;
                int64_t     occupied         = 0;
                int64_t     running_requests = 0;
                for (int rank : group.allocated_attention_ranks) {
                    capacity += static_cast<int64_t>(worker_state[dp_idx]->block_manager.at(rank)->blocks().size())
                                * Sequence::block_size;
                    occupied += used.at(rank);
                }
                for (const auto& sequence : group.sequences) {
                    if (sequence && sequence->status == SequenceStatus::RUNNING) {
                        ++running_requests;
                    }
                }
                return capacity - occupied - running_requests;
            };
            auto merge_shadow_groups = [&](uint64_t constrained_id, uint64_t donor_id) {
                auto constrained = step->groups.find(constrained_id);
                auto donor       = step->groups.find(donor_id);
                if (constrained == step->groups.end() || donor == step->groups.end()
                    || constrained->second.dp_idx != donor->second.dp_idx) {
                    throw std::runtime_error("invalid prospective LS memory-deficit group merge");
                }
                auto&                        survivor = constrained->second;
                std::unordered_set<uint64_t> emitted_ids;
                emitted_ids.reserve(survivor.sequences.size() + donor->second.sequences.size());
                for (const auto& sequence : survivor.sequences) {
                    if (sequence) {
                        emitted_ids.insert(sequence->seq_id);
                    }
                }
                for (const auto& sequence : donor->second.sequences) {
                    if (sequence && sequence->status == SequenceStatus::RUNNING
                        && emitted_ids.insert(sequence->seq_id).second) {
                        survivor.sequences.push_back(sequence);
                    }
                    if (sequence) {
                        step->seq_to_group[sequence->seq_id] = constrained_id;
                    }
                }
                survivor.initial_batch_placements.insert(survivor.initial_batch_placements.end(),
                                                         donor->second.initial_batch_placements.begin(),
                                                         donor->second.initial_batch_placements.end());
                for (int rank : donor->second.allocated_attention_ranks) {
                    if (std::find(
                            survivor.allocated_attention_ranks.begin(), survivor.allocated_attention_ranks.end(), rank)
                        == survivor.allocated_attention_ranks.end()) {
                        survivor.allocated_attention_ranks.push_back(rank);
                    }
                }
                survivor.last_scale_up_step        = ls_schedule_step_;
                survivor.kv_candidate_target_dop   = -1;
                survivor.kv_candidate_stable_steps = 0;
                survivor.kv_candidate_member_ids.clear();
                survivor.kv_candidate_allocation.clear();
                auto& group_ids = step->group_ids_by_dp[survivor.dp_idx];
                group_ids.erase(std::remove(group_ids.begin(), group_ids.end(), donor_id), group_ids.end());
                step->groups.erase(donor);
                step->mandatory_graph_changed                = true;
                step->pool_resource_changed[survivor.dp_idx] = true;
            };

            // Mandatory safety is prospective only. No group/owner publication
            // occurs until every required Decode component below validates.
            for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
                auto group_ids = step->group_ids_by_dp[dp_idx];
                for (uint64_t group_id : group_ids) {
                    auto&        group  = step->groups.at(group_id);
                    auto         used   = worker_state[dp_idx]->group_used_kv_tokens(group.sequences);
                    const size_t before = group.allocated_attention_ranks.size();
                    group.allocated_attention_ranks.erase(
                        std::remove_if(
                            group.allocated_attention_ranks.begin(),
                            group.allocated_attention_ranks.end(),
                            [&](int rank) {
                                bool protected_role = std::any_of(
                                    group.sequences.begin(), group.sequences.end(), [&](const auto& sequence) {
                                        if (!sequence || sequence->status != SequenceStatus::RUNNING) {
                                            return false;
                                        }
                                        const auto& context = sequence->block_ctx(BlockContextSlot::ACTIVE);
                                        return context.master_sp_idx_ == rank
                                               || (context.pending_token_present_
                                                   && context.pending_token_target_sp_ == rank);
                                    });
                                return used.at(rank) == 0 && !protected_role;
                            }),
                        group.allocated_attention_ranks.end());
                    if (group.allocated_attention_ranks.size() != before) {
                        step->mandatory_graph_changed       = true;
                        step->pool_resource_changed[dp_idx] = true;
                    }
                }

                std::vector<uint64_t> can_decode;
                std::vector<uint64_t> cannot_decode;
                for (uint64_t group_id : group_ids) {
                    (decode_idle_tokens(dp_idx, group_id) >= 0 ? can_decode : cannot_decode).push_back(group_id);
                }
                auto slack_less = [&](uint64_t lhs, uint64_t rhs) {
                    return decode_idle_tokens(dp_idx, lhs) < decode_idle_tokens(dp_idx, rhs);
                };
                std::stable_sort(can_decode.begin(), can_decode.end(), slack_less);
                std::stable_sort(cannot_decode.begin(), cannot_decode.end(), slack_less);
                for (uint64_t constrained_id : cannot_decode) {
                    int64_t slack = decode_idle_tokens(dp_idx, constrained_id);
                    while (slack < 0 && !can_decode.empty()) {
                        uint64_t donor_id = can_decode.back();
                        can_decode.pop_back();
                        if (donor_id == constrained_id || !step->groups.count(donor_id)) {
                            continue;
                        }
                        merge_shadow_groups(constrained_id, donor_id);
                        slack = decode_idle_tokens(dp_idx, constrained_id);
                    }
                    if (slack < 0) {
                        auto& constrained = step->groups.at(constrained_id);
                        for (int rank : unallocated_ranks(dp_idx)) {
                            constrained.allocated_attention_ranks.push_back(rank);
                            constrained.last_scale_up_step        = ls_schedule_step_;
                            constrained.kv_candidate_target_dop   = -1;
                            constrained.kv_candidate_stable_steps = 0;
                            constrained.kv_candidate_member_ids.clear();
                            constrained.kv_candidate_allocation.clear();
                            step->mandatory_graph_changed       = true;
                            step->pool_resource_changed[dp_idx] = true;
                            slack += static_cast<int64_t>(worker_state[dp_idx]->block_manager.at(rank)->blocks().size())
                                     * Sequence::block_size;
                            if (slack >= 0) {
                                break;
                            }
                        }
                    }
                    if (slack < 0) {
                        throw AttemptError{include_admission ? AttemptFailureKind::TENTATIVE :
                                                               AttemptFailureKind::CAPACITY_NO_FIT,
                                           dp_idx,
                                           "capacity no-fit for mandatory Decode memory safety"};
                    }
                    can_decode.push_back(constrained_id);
                }
                if (step->group_ids_by_dp[dp_idx] != can_decode) {
                    step->mandatory_graph_changed       = true;
                    step->pool_resource_changed[dp_idx] = true;
                }
                step->group_ids_by_dp[dp_idx] = std::move(can_decode);
            }

            if (include_admission) {
                for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
                    if (step->waiting[dp_idx].empty()) {
                        continue;
                    }

                    // Admission is optional per pool. Keep a post-mandatory
                    // scheduler snapshot so this pool alone can roll back a
                    // failed exact prepare while earlier pools retain their
                    // valid reservations for the global Decode validation.
                    auto rollback_base = capture_admission_scheduler_shadow();
                    try {

                        auto make_admission_plan =
                            [&](const std::vector<std::shared_ptr<Sequence>>& ordered) -> std::optional<AdmissionPlan> {
                            if (ordered.empty() || !_ls_batch_fits_empty_system(dp_idx, ordered)
                                || !_ls_pool_future_kv_fits(dp_idx, ordered)) {
                                return std::nullopt;
                            }
                            auto    idle_ranks    = unallocated_ranks(dp_idx);
                            int64_t idle_capacity = 0;
                            int64_t admission_sum = 0;
                            for (int rank : idle_ranks) {
                                idle_capacity +=
                                    static_cast<int64_t>(worker_state[dp_idx]->block_manager.at(rank)->blocks().size())
                                    * Sequence::block_size;
                            }
                            for (const auto& sequence : ordered) {
                                admission_sum += _ls_admission_need_tokens(*sequence);
                            }

                            AdmissionPlan                          candidate;
                            std::vector<int>                       rank_pool = idle_ranks;
                            std::vector<std::shared_ptr<Sequence>> donor_sequences;
                            std::vector<int>                       donor_allocation;
                            if (admission_sum > idle_capacity) {
                                candidate.target_kind = LSAdmissionTargetKind::CAPACITY_APPEND;
                                struct Donor {
                                    uint64_t group_id = 0;
                                    int64_t  slack    = 0;
                                };
                                std::vector<Donor> donors;
                                for (uint64_t group_id : step->group_ids_by_dp[dp_idx]) {
                                    const auto& group    = step->groups.at(group_id);
                                    auto        used     = worker_state[dp_idx]->group_used_kv_tokens(group.sequences);
                                    int64_t     capacity = 0;
                                    int64_t     occupied = 0;
                                    for (int rank : group.allocated_attention_ranks) {
                                        capacity += static_cast<int64_t>(
                                                        worker_state[dp_idx]->block_manager.at(rank)->blocks().size())
                                                    * Sequence::block_size;
                                        occupied += used.at(rank);
                                    }
                                    donors.push_back({group_id, capacity - occupied});
                                }
                                std::stable_sort(donors.begin(), donors.end(), [](const Donor& lhs, const Donor& rhs) {
                                    return lhs.slack < rhs.slack;
                                });
                                const int64_t deficit = admission_sum - idle_capacity;
                                int64_t       covered = 0;
                                while (!donors.empty() && covered < deficit) {
                                    Donor donor = donors.back();
                                    donors.pop_back();
                                    if (donor.slack <= 0) {
                                        continue;
                                    }
                                    covered += donor.slack;
                                    candidate.planned_donors.push_back(donor.group_id);
                                    const auto& group = step->groups.at(donor.group_id);
                                    donor_sequences.insert(
                                        donor_sequences.end(), group.sequences.begin(), group.sequences.end());
                                    for (int rank : group.allocated_attention_ranks) {
                                        if (std::find(rank_pool.begin(), rank_pool.end(), rank) == rank_pool.end()) {
                                            rank_pool.push_back(rank);
                                        }
                                        if (std::find(donor_allocation.begin(), donor_allocation.end(), rank)
                                            == donor_allocation.end()) {
                                            donor_allocation.push_back(rank);
                                        }
                                    }
                                }
                                if (covered < deficit) {
                                    return std::nullopt;
                                }
                            }
                            auto placement = _plan_ls_initial_placement(
                                dp_idx, ordered, rank_pool, donor_sequences, donor_allocation);
                            if (!placement) {
                                return std::nullopt;
                            }
                            candidate.ranks      = std::move(placement->first);
                            candidate.placements = std::move(placement->second);
                            return candidate;
                        };

                        std::vector<std::shared_ptr<Sequence>> selected_fifo;
                        std::optional<uint64_t>                first_blocker;
                        std::unordered_set<uint64_t>           selected_after_blocker;
                        int64_t                                selected_tokens = 0;
                        const bool                             allow_ooe       = step->ooe[dp_idx] < ls_max_num_ooe_;
                        const int running_count = static_cast<int>(_ls_running_sequences_in_pool(dp_idx).size());
                        for (const auto& sequence : step->waiting[dp_idx]) {
                            if (!sequence || sequence->assigned_dp != dp_idx
                                || (sequence->status != SequenceStatus::WAITING
                                    && sequence->status != SequenceStatus::PAUSED_OFFLOAD)) {
                                throw std::runtime_error("invalid sequence in an LS pool-local waiting queue");
                            }
                            const int64_t need      = _ls_admission_need_tokens(*sequence);
                            auto          tentative = selected_fifo;
                            tentative.push_back(sequence);
                            bool feasible =
                                static_cast<int>(tentative.size()) <= max_num_seqs_
                                && running_count + static_cast<int>(tentative.size()) <= ls_running_max_req_size_
                                && selected_tokens + need <= ls_admission_max_tokens_per_pool_
                                && _ls_pool_future_kv_fits(dp_idx, tentative)
                                && make_admission_plan({sequence}).has_value();
                            if (feasible) {
                                selected_fifo.push_back(sequence);
                                selected_tokens += need;
                                if (first_blocker) {
                                    selected_after_blocker.insert(sequence->seq_id);
                                }
                                continue;
                            }
                            if (!first_blocker) {
                                first_blocker = sequence->seq_id;
                            }
                            if (!allow_ooe) {
                                break;
                            }
                        }

                        std::optional<AdmissionPlan> admission_plan;
                        while (!selected_fifo.empty() && !admission_plan) {
                            std::vector<std::shared_ptr<Sequence>> ordered = selected_fifo;
                            std::stable_sort(ordered.begin(), ordered.end(), [&](const auto& lhs, const auto& rhs) {
                                return _ls_admission_need_tokens(*lhs) > _ls_admission_need_tokens(*rhs);
                            });
                            admission_plan = make_admission_plan(ordered);
                            if (!admission_plan) {
                                selected_fifo.pop_back();
                                ++ls_step_atomic_no_fit_count_;
                            }
                            else {
                                selected_fifo = std::move(ordered);
                            }
                        }
                        if (!admission_plan) {
                            continue;
                        }
                        step->had_tentative_admission = true;
                        if (!injection_captured) {
                            injected_allocation_failure =
                                std::exchange(ls_admission_failure_after_allocations_for_test_, -1);
                            injected_publication_failure =
                                std::exchange(ls_post_admission_component_failure_for_test_, -1);
                            injection_captured = true;
                        }

                        const uint64_t batch_id        = next_ls_batch_id_++;
                        const uint64_t new_group_id    = next_ls_group_id_++;
                        const uint64_t admission_order = next_ls_admission_order_++;
                        if (admission_plan->ranks.empty()
                            || admission_plan->placements.size() != selected_fifo.size()) {
                            throw std::runtime_error("invalid LS admission placement cardinality");
                        }

                        PreparedAdmission prepared_admission;
                        prepared_admission.dp_idx     = dp_idx;
                        prepared_admission.batch_id  = batch_id;
                        prepared_admission.sequences = selected_fifo;
                        prepared_admission.original_statuses.reserve(selected_fifo.size());
                        prepared_admission.bootstrap_finishes.reserve(selected_fifo.size());
                        std::vector<BlockContext>              placement_contexts;
                        std::vector<std::shared_ptr<Sequence>> survivors;
                        std::vector<BlockContext>              survivor_contexts;
                        std::vector<std::shared_ptr<Sequence>> finished;
                        std::vector<BlockContext>              finished_contexts;
                        placement_contexts.reserve(selected_fifo.size());
                        survivors.reserve(selected_fifo.size());
                        survivor_contexts.reserve(selected_fifo.size());
                        finished.reserve(selected_fifo.size());
                        finished_contexts.reserve(selected_fifo.size());
                        for (size_t idx = 0; idx < selected_fifo.size(); ++idx) {
                            const auto& sequence = selected_fifo[idx];
                            prepared_admission.original_statuses.push_back(sequence->status);
                            BlockContext context(engine_id_, attention_sp_, attention_dp_);
                            context.dp_idx_                = dp_idx;
                            context.master_sp_idx_         = admission_plan->ranks[idx % admission_plan->ranks.size()];
                            context.pending_token_present_ = false;
                            context.pending_token_target_sp_ = -1;
                            context.num_dispatched_tokens    = admission_plan->placements[idx];
                            placement_contexts.push_back(context);
                            const bool finishes = sequence->num_completed_tokens() + 1 >= sequence->max_tokens;
                            prepared_admission.bootstrap_finishes.push_back(finishes);
                            if (finishes) {
                                finished.push_back(sequence);
                                finished_contexts.push_back(std::move(context));
                            }
                            else {
                                survivors.push_back(sequence);
                                survivor_contexts.push_back(std::move(context));
                            }
                            sequence->token_ids.reserve(sequence->token_ids.size() + 1);
                            if (sequence->token_ids.capacity() <= sequence->token_ids.size()) {
                                throw std::runtime_error("LS admission failed to reserve dummy token storage");
                            }
                            if (sequence->metric && sequence->status == SequenceStatus::PAUSED_OFFLOAD
                                && sequence->metric->last_token_time) {
                                sequence->metric->itl_samples.reserve(sequence->metric->itl_samples.size() + 1);
                                if (sequence->metric->itl_samples.capacity() <= sequence->metric->itl_samples.size()) {
                                    throw std::runtime_error("LS readmission failed to reserve ITL storage");
                                }
                            }
                        }

                        std::optional<SPStateManager::PreparedLSInitialBatch> prepared_finished;
                        if (!finished.empty()) {
                            prepared_finished.emplace(
                                worker_state[dp_idx]->prepare_ls_initial_batch(finished, finished_contexts));
                        }
                        if (!survivors.empty()) {
                            prepared_admission.prepared_survivors.emplace(
                                worker_state[dp_idx]->prepare_ls_initial_batch(survivors, survivor_contexts));
                        }
                        if (injected_allocation_failure == 0) {
                            injected_allocation_failure = -1;
                            throw std::runtime_error("injected LS admission allocation-prepare failure");
                        }
                        // A positive countdown represents complete isolated pool
                        // prepares. Thus value 1 lets DP0 survive and makes DP1 the
                        // next injected failure in a two-pool test.
                        if (injected_allocation_failure > 0) {
                            --injected_allocation_failure;
                        }
                        if ((prepared_finished && !prepared_finished->validate_precommit_noexcept())
                            || (prepared_admission.prepared_survivors
                                && !prepared_admission.prepared_survivors->validate_precommit_noexcept())) {
                            throw std::runtime_error("prepared LS initial reservation became stale");
                        }
                        if (prepared_finished) {
                            prepared_finished->abort_noexcept();
                        }
                        if (prepared_admission.prepared_survivors
                            && !prepared_admission.prepared_survivors->validate_precommit_noexcept()) {
                            throw std::runtime_error("prepared LS survivor reservation changed after finished abort");
                        }

                        std::vector<uint64_t> actual_donors;
                        for (uint64_t donor_id : admission_plan->planned_donors) {
                            const auto& donor   = step->groups.at(donor_id);
                            bool        overlap = false;
                            for (size_t idx = 0; idx < selected_fifo.size() && !overlap; ++idx) {
                                if (prepared_admission.bootstrap_finishes[idx]) {
                                    continue;
                                }
                                for (int rank : donor.allocated_attention_ranks) {
                                    if (placement_contexts[idx].num_dispatched_tokens[rank] > 0
                                        || placement_contexts[idx].master_sp_idx_ == rank) {
                                        overlap = true;
                                        break;
                                    }
                                }
                            }
                            if (overlap) {
                                actual_donors.push_back(donor_id);
                            }
                        }

                        InitialBatchPlacement initial_record;
                        initial_record.batch_id         = batch_id;
                        initial_record.group_id         = new_group_id;
                        initial_record.initial_kv_dop   = static_cast<int>(admission_plan->ranks.size());
                        initial_record.initial_kv_ranks = admission_plan->ranks;
                        initial_record.prompt_kv_tokens = admission_plan->placements;
                        for (size_t idx = 0; idx < selected_fifo.size(); ++idx) {
                            initial_record.sequence_ids.push_back(selected_fifo[idx]->seq_id);
                            initial_record.provisional_pending_targets.push_back(
                                admission_plan->ranks[idx % admission_plan->ranks.size()]);
                        }
                        initial_record.admission_order = admission_order;
                        initial_record.admission_kind =
                            admission_plan->target_kind == LSAdmissionTargetKind::STANDALONE ? "standalone" :
                                                                                               "capacity_append";

                        std::optional<uint64_t> survivor_group;
                        if (!survivors.empty()) {
                            DecodeGroupState group;
                            group.group_id  = new_group_id;
                            group.dp_idx    = dp_idx;
                            group.sequences = survivors;
                            group.initial_batch_placements.push_back(initial_record);
                            for (size_t idx = 0; idx < selected_fifo.size(); ++idx) {
                                if (prepared_admission.bootstrap_finishes[idx]) {
                                    continue;
                                }
                                const auto& sequence                 = selected_fifo[idx];
                                step->seq_to_group[sequence->seq_id] = new_group_id;
                                for (int rank = 0; rank < attention_sp_; ++rank) {
                                    if ((placement_contexts[idx].num_dispatched_tokens[rank] > 0
                                         || placement_contexts[idx].master_sp_idx_ == rank)
                                        && std::find(group.allocated_attention_ranks.begin(),
                                                     group.allocated_attention_ranks.end(),
                                                     rank)
                                               == group.allocated_attention_ranks.end()) {
                                        group.allocated_attention_ranks.push_back(rank);
                                    }
                                }
                            }
                            for (uint64_t donor_id : actual_donors) {
                                auto donor = step->groups.find(donor_id);
                                if (donor == step->groups.end()) {
                                    throw std::runtime_error("planned LS admission donor disappeared");
                                }
                                for (const auto& sequence : donor->second.sequences) {
                                    group.sequences.push_back(sequence);
                                    step->seq_to_group[sequence->seq_id] = new_group_id;
                                }
                                group.initial_batch_placements.insert(group.initial_batch_placements.end(),
                                                                      donor->second.initial_batch_placements.begin(),
                                                                      donor->second.initial_batch_placements.end());
                                for (int rank : donor->second.allocated_attention_ranks) {
                                    if (std::find(group.allocated_attention_ranks.begin(),
                                                  group.allocated_attention_ranks.end(),
                                                  rank)
                                        == group.allocated_attention_ranks.end()) {
                                        group.allocated_attention_ranks.push_back(rank);
                                    }
                                }
                                auto& group_ids = step->group_ids_by_dp[dp_idx];
                                group_ids.erase(std::remove(group_ids.begin(), group_ids.end(), donor_id),
                                                group_ids.end());
                                step->groups.erase(donor);
                            }
                            group.last_scale_up_step = ls_schedule_step_;
                            if (!step->groups.emplace(new_group_id, std::move(group)).second) {
                                throw std::runtime_error("reserved LS admission group ID already exists");
                            }
                            step->group_ids_by_dp[dp_idx].push_back(new_group_id);
                            survivor_group = new_group_id;
                        }

                        for (size_t idx = 0; idx < selected_fifo.size(); ++idx) {
                            const auto& sequence = selected_fifo[idx];
                            auto        queue_it =
                                std::find(step->waiting[dp_idx].begin(), step->waiting[dp_idx].end(), sequence);
                            if (queue_it == step->waiting[dp_idx].end()) {
                                throw std::runtime_error("prepared LS sequence disappeared from its pool queue");
                            }
                            step->waiting[dp_idx].erase(queue_it);
                            if (prepared_admission.bootstrap_finishes[idx]) {
                                step->arrival_orders.erase(sequence->seq_id);
                            }
                            else {
                                step->running[dp_idx].push_back(sequence);
                            }
                            LSAdmissionRecord record;
                            record.sequence = sequence;
                            record.dp_idx   = dp_idx;
                            record.batch_id = batch_id;
                            record.group_id_after_commit =
                                prepared_admission.bootstrap_finishes[idx] ? std::nullopt : survivor_group;
                            record.admission_kind =
                                prepared_admission.original_statuses[idx] == SequenceStatus::PAUSED_OFFLOAD ?
                                    LSAdmissionKind::OFFLOAD_READMIT :
                                    LSAdmissionKind::FRESH;
                            record.target_kind        = admission_plan->target_kind;
                            record.planned_kv_dop     = static_cast<int>(admission_plan->ranks.size());
                            record.planned_kv_ranks   = admission_plan->ranks;
                            record.bootstrap_finished = prepared_admission.bootstrap_finishes[idx];
                            step->admission_records.push_back(std::move(record));
                        }
                        step->initial_records.push_back(initial_record);
                        const bool bypass =
                            first_blocker
                            && std::any_of(selected_fifo.begin(), selected_fifo.end(), [&](const auto& sequence) {
                                   return selected_after_blocker.count(sequence->seq_id) != 0;
                               });
                        step->ooe[dp_idx] = bypass ? step->ooe[dp_idx] + 1 : 0;
                        prepared_admission.bootstrap_commit_time =
                            std::chrono::duration_cast<std::chrono::duration<double>>(
                                std::chrono::high_resolution_clock::now().time_since_epoch())
                                .count();
                        prepared_admission.merge_count_delta = actual_donors.empty() ? 0 : 1;
                        prepared_admission.rollback_base     = std::move(rollback_base);
                        step->admissions[dp_idx].emplace(std::move(prepared_admission));
                        step->has_admission                 = true;
                        step->pool_resource_changed[dp_idx] = true;
                        step->atomic_merge_count += actual_donors.empty() ? 0 : 1;
                    }
                    catch (const std::exception&) {
                        step->admissions[dp_idx].reset();
                        if (rollback_base) {
                            restore_admission_scheduler_shadow(*rollback_base);
                        }
                        ++ls_step_atomic_rollback_count_;
                    }
                }
            }

            // Low-KV is exclusive and may only be considered after all
            // tentative admission owners have ended and no mandatory shadow
            // mutation exists. A rejected candidate is followed by Decode.
            if (include_admission && !step->has_admission && !step->mandatory_graph_changed) {
                auto consolidation = _maybe_plan_ls_kv_consolidation();
                if (consolidation) {
                    result.failure       = AttemptFailureKind::KV_CONSOLIDATION;
                    result.consolidation = std::move(consolidation);
                    return result;
                }
                // Candidate stability metadata may have advanced in the stable
                // graph; preserve it in the later scheduler publication shadow.
                step->groups          = ls_groups_;
                step->group_ids_by_dp = ls_group_ids_by_dp_;
                step->seq_to_group    = ls_seq_to_group_;
            }

            auto make_pool_admission_rollback_shadow = [&](int dp_idx) {
                if (dp_idx < 0 || dp_idx >= attention_dp_ || !step->admissions[dp_idx]
                    || !step->admissions[dp_idx]->rollback_base) {
                    throw std::runtime_error("missing LS pool-local admission rollback shadow");
                }
                const auto& admission = *step->admissions[dp_idx];
                const auto& base      = *admission.rollback_base;
                auto        rollback  = capture_admission_scheduler_shadow();

                std::unordered_set<uint64_t> pool_sequence_ids;
                auto collect_group_sequences = [&](const auto& groups, const auto& group_ids) {
                    for (uint64_t group_id : group_ids) {
                        auto group = groups.find(group_id);
                        if (group == groups.end() || group->second.dp_idx != dp_idx) {
                            throw std::runtime_error("invalid LS pool-local admission rollback group owner");
                        }
                        for (const auto& sequence : group->second.sequences) {
                            if (!sequence || sequence->assigned_dp != dp_idx) {
                                throw std::runtime_error("invalid LS pool-local admission rollback sequence owner");
                            }
                            pool_sequence_ids.insert(sequence->seq_id);
                        }
                    }
                };
                collect_group_sequences(step->groups, step->group_ids_by_dp[dp_idx]);
                collect_group_sequences(base.groups, base.group_ids_by_dp[dp_idx]);
                for (const auto& sequence : step->waiting[dp_idx]) {
                    if (sequence) {
                        pool_sequence_ids.insert(sequence->seq_id);
                    }
                }
                for (const auto& sequence : base.waiting[dp_idx]) {
                    if (sequence) {
                        pool_sequence_ids.insert(sequence->seq_id);
                    }
                }
                for (const auto& sequence : admission.sequences) {
                    if (sequence) {
                        pool_sequence_ids.insert(sequence->seq_id);
                    }
                }

                for (auto group = rollback->groups.begin(); group != rollback->groups.end();) {
                    if (group->second.dp_idx == dp_idx) {
                        group = rollback->groups.erase(group);
                    }
                    else {
                        ++group;
                    }
                }
                for (uint64_t group_id : base.group_ids_by_dp[dp_idx]) {
                    rollback->groups.insert_or_assign(group_id, base.groups.at(group_id));
                }
                rollback->group_ids_by_dp[dp_idx] = base.group_ids_by_dp[dp_idx];

                for (auto owner = rollback->seq_to_group.begin(); owner != rollback->seq_to_group.end();) {
                    auto group = step->groups.find(owner->second);
                    if (group == step->groups.end()) {
                        throw std::runtime_error("LS pool-local admission rollback found a stale owner index");
                    }
                    if (group->second.dp_idx == dp_idx) {
                        owner = rollback->seq_to_group.erase(owner);
                    }
                    else {
                        ++owner;
                    }
                }
                for (const auto& [seq_id, group_id] : base.seq_to_group) {
                    auto group = base.groups.find(group_id);
                    if (group != base.groups.end() && group->second.dp_idx == dp_idx) {
                        rollback->seq_to_group.insert_or_assign(seq_id, group_id);
                    }
                }

                for (uint64_t seq_id : pool_sequence_ids) {
                    rollback->arrival_orders.erase(seq_id);
                }
                for (const auto& [seq_id, arrival_order] : base.arrival_orders) {
                    if (pool_sequence_ids.count(seq_id) != 0) {
                        rollback->arrival_orders.insert_or_assign(seq_id, arrival_order);
                    }
                }
                rollback->waiting[dp_idx] = base.waiting[dp_idx];
                rollback->running[dp_idx] = base.running[dp_idx];
                rollback->ooe[dp_idx]     = base.ooe[dp_idx];
                rollback->admission_records.erase(
                    std::remove_if(rollback->admission_records.begin(),
                                   rollback->admission_records.end(),
                                   [&](const auto& record) { return record.dp_idx == dp_idx; }),
                    rollback->admission_records.end());
                rollback->initial_records.erase(
                    std::remove_if(rollback->initial_records.begin(),
                                   rollback->initial_records.end(),
                                   [&](const auto& record) { return record.batch_id == admission.batch_id; }),
                    rollback->initial_records.end());
                rollback->pool_resource_changed[dp_idx] = base.pool_resource_changed[dp_idx];
                if (rollback->atomic_merge_count < admission.merge_count_delta) {
                    throw std::runtime_error("LS pool-local admission rollback merge count underflow");
                }
                rollback->atomic_merge_count -= admission.merge_count_delta;
                rollback->has_admission = std::any_of(
                    step->admissions.begin(), step->admissions.end(), [&](const auto& candidate) {
                        return candidate.has_value() && candidate->dp_idx != dp_idx;
                    });
                return rollback;
            };
            auto rollback_pool_admission = [&](int dp_idx, AdmissionSchedulerShadow& rollback) noexcept {
                step->admissions[dp_idx].reset();
                restore_admission_scheduler_shadow(rollback);
                ++ls_step_atomic_rollback_count_;
            };

            auto is_eligible = [&](const std::shared_ptr<Sequence>& sequence) {
                return sequence && sequence->status == SequenceStatus::RUNNING
                       && eligible_sequence_ids.count(sequence->seq_id) != 0;
            };
            auto plan_decode_component = [&](int dp_idx, bool tentative_component) {
                auto& pending_dp = step->decode_by_dp[dp_idx];
                pending_dp       = PendingDPPlan{};
                if (tentative_component && injected_publication_failure == 0) {
                    injected_publication_failure = -1;
                    throw AttemptError{AttemptFailureKind::TENTATIVE,
                                       dp_idx,
                                       "injected LS post-admission Decode-component failure"};
                }
                if (tentative_component && injected_publication_failure > 0) {
                    --injected_publication_failure;
                }

                const auto              group_ids = step->group_ids_by_dp[dp_idx];
                std::unordered_set<int> owned;
                for (uint64_t group_id : group_ids) {
                    for (int rank : step->groups.at(group_id).allocated_attention_ranks) {
                        owned.insert(rank);
                    }
                }
                pending_dp.groups.reserve(group_ids.size());
                for (uint64_t group_id : group_ids) {
                    const auto&                            group = step->groups.at(group_id);
                    std::vector<std::shared_ptr<Sequence>> requests;
                    for (const auto& sequence : group.sequences) {
                        if (is_eligible(sequence)) {
                            requests.push_back(sequence);
                        }
                    }
                    if (requests.empty()) {
                        continue;
                    }

                    std::vector<int> extras;
                    for (int rank = 0; rank < attention_sp_; ++rank) {
                        if (!owned.count(rank) && rank_is_idle_in_graph(dp_idx, rank)) {
                            extras.push_back(rank);
                        }
                    }
                    std::sort(extras.begin(), extras.end(), std::greater<int>());
                    auto plan = worker_state[dp_idx]->plan_iteration_masters_source_greedy(
                        requests,
                        group.allocated_attention_ranks,
                        extras,
                        ls_min_comp_bound_decoding_batch_size_,
                        ls_decode_enable_memory_scale_up_,
                        ls_step_execution_loop_count_);
                    if (!plan.success) {
                        const std::string reason =
                            plan.failure_reason.empty() ? "planner returned failure" : plan.failure_reason;
                        throw AttemptError{tentative_component ?
                                               AttemptFailureKind::TENTATIVE :
                                               (is_explicit_decode_capacity_no_fit(reason) ?
                                                    AttemptFailureKind::CAPACITY_NO_FIT :
                                                    AttemptFailureKind::INTERNAL),
                                           dp_idx,
                                           reason};
                    }
                    std::string validation_error;
                    if (!worker_state[dp_idx]->validate_iteration_master_plan(
                            requests, plan, &validation_error, ls_step_execution_loop_count_)) {
                        throw AttemptError{tentative_component ? AttemptFailureKind::TENTATIVE :
                                                               AttemptFailureKind::INTERNAL,
                                           dp_idx,
                                           "required Decode plan validation failed: " + validation_error};
                    }
                    for (int rank : plan.allocation) {
                        owned.insert(rank);
                    }

                    std::vector<int> reused_passive;
                    for (int rank : plan.master_ranks) {
                        const bool was_last_master =
                            std::find(group.last_iteration_masters.begin(), group.last_iteration_masters.end(), rank)
                            != group.last_iteration_masters.end();
                        if (!was_last_master && plan.group_used_kv_tokens.at(rank) > 0
                            && std::find(
                                   group.allocated_attention_ranks.begin(), group.allocated_attention_ranks.end(), rank)
                                   != group.allocated_attention_ranks.end()) {
                            reused_passive.push_back(rank);
                        }
                    }
                    std::vector<uint64_t> sequence_ids;
                    sequence_ids.reserve(requests.size());
                    for (const auto& request : requests) {
                        sequence_ids.push_back(request->seq_id);
                    }
                    pending_dp.groups.push_back({group_id,
                                                 std::move(requests),
                                                 std::move(sequence_ids),
                                                 std::move(plan),
                                                 std::move(reused_passive)});
                }
                if (pending_dp.groups.empty()) {
                    return;
                }

                auto& combined               = pending_dp.combined_plan;
                combined.success             = true;
                combined.scale_reason        = "combined_pool_iteration";
                combined.assignment_strategy = "source_greedy";
                std::vector<int>        combined_master_load(attention_sp_, 0);
                std::unordered_set<int> combined_allocation;
                std::unordered_set<int> combined_new_allocation;
                for (const auto& pending : pending_dp.groups) {
                    pending_dp.combined_requests.insert(
                        pending_dp.combined_requests.end(), pending.requests.begin(), pending.requests.end());
                    combined.sequence_master_ranks.insert(combined.sequence_master_ranks.end(),
                                                          pending.plan.sequence_master_ranks.begin(),
                                                          pending.plan.sequence_master_ranks.end());
                    for (int rank : pending.plan.allocation) {
                        if (combined_allocation.insert(rank).second) {
                            combined.allocation.push_back(rank);
                        }
                    }
                    for (int rank : pending.plan.new_allocation_ranks) {
                        if (combined_new_allocation.insert(rank).second) {
                            combined.new_allocation_ranks.push_back(rank);
                        }
                    }
                    for (int master : pending.plan.sequence_master_ranks) {
                        if (master < 0 || master >= attention_sp_) {
                            throw AttemptError{tentative_component ? AttemptFailureKind::TENTATIVE :
                                                                   AttemptFailureKind::INTERNAL,
                                               dp_idx,
                                               "required Decode plan produced an invalid master rank"};
                        }
                        combined_master_load[master]++;
                    }
                }
                for (int rank = 0; rank < attention_sp_; ++rank) {
                    if (combined_master_load[rank] > 0) {
                        combined.master_ranks.push_back(rank);
                        combined.master_batch_sizes.push_back(combined_master_load[rank]);
                    }
                }
                combined.group_used_kv_tokens =
                    worker_state[dp_idx]->group_used_kv_tokens(pending_dp.combined_requests);
                combined.group_used_kv_blocks =
                    worker_state[dp_idx]->group_used_kv_blocks(pending_dp.combined_requests);
                std::string combined_error;
                if (!worker_state[dp_idx]->validate_iteration_master_plan(
                        pending_dp.combined_requests,
                        combined,
                        &combined_error,
                        ls_step_execution_loop_count_)) {
                    throw AttemptError{tentative_component ? AttemptFailureKind::TENTATIVE :
                                                            AttemptFailureKind::INTERNAL,
                                       dp_idx,
                                       "combined pool Decode validation failed: " + combined_error};
                }
            };

            auto prepare_decode_component = [&](int dp_idx, bool tentative_component) {
                auto& pending_dp = step->decode_by_dp[dp_idx];
                if (!pending_dp.combined_requests.empty()) {
                    try {
                        pending_dp.prepared.emplace(worker_state[dp_idx]->prepare_iteration_master_plan(
                            pending_dp.combined_requests,
                            pending_dp.combined_plan,
                            ls_step_execution_loop_count_));
                    }
                    catch (const std::exception& error) {
                        throw AttemptError{tentative_component ? AttemptFailureKind::TENTATIVE :
                                                                 AttemptFailureKind::INTERNAL,
                                           dp_idx,
                                           error.what()};
                    }
                }
                auto* initial   = step->admissions[dp_idx] && step->admissions[dp_idx]->prepared_survivors ?
                                      &*step->admissions[dp_idx]->prepared_survivors :
                                      nullptr;
                auto* iteration = pending_dp.prepared ? &*pending_dp.prepared : nullptr;
                if (!worker_state[dp_idx]->validate_ls_pool_step_composition_noexcept(initial, iteration)) {
                    throw AttemptError{tentative_component ? AttemptFailureKind::TENTATIVE :
                                                            AttemptFailureKind::INTERNAL,
                                       dp_idx,
                                       "prepared LS pool-step composition became stale or exceeded role capacity"};
                }
            };

            for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
                for (;;) {
                    const bool tentative_component = include_admission && step->admissions[dp_idx].has_value();
                    auto rollback = tentative_component ? make_pool_admission_rollback_shadow(dp_idx) : nullptr;
                    try {
                        plan_decode_component(dp_idx, tentative_component);
                        prepare_decode_component(dp_idx, tentative_component);
                        break;
                    }
                    catch (const AttemptError& error) {
                        if (!tentative_component || error.kind != AttemptFailureKind::TENTATIVE
                            || error.dp_idx != dp_idx) {
                            throw;
                        }

                        auto& pending_dp = step->decode_by_dp[dp_idx];
                        // Destroy the old-overlay reservation before restoring
                        // the post-mandatory pool graph. Its RAII abort releases
                        // every prepared block; it must never be cached/reused.
                        pending_dp.prepared.reset();
                        rollback_pool_admission(dp_idx, *rollback);
                        // Retry only this required component against the same
                        // post-mandatory graph without its optional admission.
                    }
                }
            }

            size_t total_group_plans = 0;
            for (const auto& pending_dp : step->decode_by_dp) {
                total_group_plans += pending_dp.groups.size();
            }
            step->group_plans.reserve(total_group_plans);
            step->group_plan_ids.reserve(total_group_plans);
            step->group_plan_sequence_ids.reserve(total_group_plans);
            step->reused_passive_masters.reserve(total_group_plans);
            const bool has_global_decode = std::any_of(
                step->decode_by_dp.begin(), step->decode_by_dp.end(), [](const auto& pending_dp) {
                    return !pending_dp.groups.empty();
                });
            for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
                std::vector<int> master_load(attention_sp_, 0);
                for (const auto& pending : step->decode_by_dp[dp_idx].groups) {
                    auto group_it = step->groups.find(pending.group_id);
                    if (group_it == step->groups.end()) {
                        throw AttemptError{AttemptFailureKind::INTERNAL,
                                           dp_idx,
                                           "required Decode group disappeared after final component validation"};
                    }
                    auto&      group = group_it->second;
                    const bool allocation_grew =
                        pending.plan.allocation.size() > group.allocated_attention_ranks.size();
                    if (pending.plan.allocation != group.allocated_attention_ranks) {
                        step->pool_resource_changed[dp_idx] = true;
                    }
                    group.allocated_attention_ranks = pending.plan.allocation;
                    group.last_iteration_masters    = pending.plan.master_ranks;
                    if (allocation_grew) {
                        group.last_scale_up_step        = ls_schedule_step_;
                        group.kv_candidate_target_dop   = -1;
                        group.kv_candidate_stable_steps = 0;
                        group.kv_candidate_member_ids.clear();
                        group.kv_candidate_allocation.clear();
                    }
                    for (size_t idx = 0; idx < pending.requests.size(); ++idx) {
                        const int master = pending.plan.sequence_master_ranks.at(idx);
                        step->scheduled[dp_idx].push_back(pending.requests[idx]);
                        step->real_decode_ids[dp_idx].push_back(pending.requests[idx]->seq_id);
                        master_load[master]++;
                    }
                    step->group_plan_ids.push_back(pending.group_id);
                    step->group_plan_sequence_ids.push_back(pending.sequence_ids);
                    step->group_plans.push_back(pending.plan);
                    step->reused_passive_masters.push_back(pending.reused_passive);
                }
                for (int rank = 0; rank < attention_sp_; ++rank) {
                    // Every DP pool participates in the globally synchronized
                    // FFN cadence. An idle pool must therefore receive the
                    // scheduler-owned, KV-backed rank dummy instead of leaving
                    // ModelRunner to fabricate an unallocated local Sequence.
                    if (master_load[rank] == 0 && has_global_decode) {
                        step->scheduled[dp_idx].push_back(worker_state[dp_idx]->dummy_seqs[rank]);
                    }
                }
            }

            result.step = std::move(step);
            return result;
        }
        catch (const AttemptError& error) {
            result.failure = error.kind;
            result.dp_idx  = error.dp_idx;
            result.reason  = error.reason;
        }
        catch (const std::exception& error) {
            // Every expected optional-pool failure is converted to a
            // dp-scoped TENTATIVE result inside the admission/component loop.
            // An exception escaping that boundary (for example while building
            // a no-self rollback shadow) cannot safely discard other pools'
            // valid admission owners via a global decode-only retry.
            result.failure = AttemptFailureKind::INTERNAL;
            result.reason  = error.what();
        }
        catch (...) {
            result.failure = AttemptFailureKind::INTERNAL;
            result.reason  = "unknown LS combined attempt failure";
        }
        result.had_tentative_admission = step->had_tentative_admission;
        result.step                    = std::move(step);
        return result;
    };

    AttemptResult attempt = prepare_attempt(true);
    if (attempt.failure == AttemptFailureKind::KV_CONSOLIDATION) {
        *kv_consolidation_plan = std::move(attempt.consolidation);
        ls_step_planning_latency_ms_ =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - planning_start).count();
        return std::vector<std::vector<std::shared_ptr<Sequence>>>(attention_dp_);
    }
    if (attempt.failure == AttemptFailureKind::TENTATIVE) {
        const bool rolled_back_admission = attempt.had_tentative_admission;
        // Reset aborts every prepared owner before constructing a different
        // stable graph. Cross-overlay prepared reservations are never reused.
        attempt.step.reset();
        if (rolled_back_admission) {
            ++ls_step_atomic_rollback_count_;
        }
        attempt = prepare_attempt(false);
    }
    if (attempt.failure != AttemptFailureKind::NONE) {
        const auto        failure = attempt.failure;
        const int         dp_idx  = attempt.dp_idx;
        const std::string reason  = attempt.reason.empty() ? "required Decode attempt failed" : attempt.reason;
        attempt.step.reset();
        if (failure == AttemptFailureKind::CAPACITY_NO_FIT) {
            if (dp_idx < 0 || dp_idx >= attention_dp_ || !_ls_offload_one_victim(dp_idx, reason)) {
                latch_ls_fatal(LSFatalCode::UNRECOVERABLE_CAPACITY);
                throw LSSchedulerFatalError(LSFatalCode::UNRECOVERABLE_CAPACITY,
                                            "no recoverable OFFLOAD victim for decode-only capacity no-fit");
            }
            ls_step_planning_latency_ms_ =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - planning_start).count();
            return std::vector<std::vector<std::shared_ptr<Sequence>>>(attention_dp_);
        }
        latch_ls_fatal(LSFatalCode::DECODE_PREPARE_OR_VALIDATE_FAILED);
        throw LSSchedulerFatalError(LSFatalCode::DECODE_PREPARE_OR_VALIDATE_FAILED,
                                    "stable decode-only prepare/validate failed: " + reason);
    }

    auto& step = *attempt.step;
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        auto* initial   = step.admissions[dp_idx] && step.admissions[dp_idx]->prepared_survivors ?
                              &*step.admissions[dp_idx]->prepared_survivors :
                              nullptr;
        auto* iteration = step.decode_by_dp[dp_idx].prepared ? &*step.decode_by_dp[dp_idx].prepared : nullptr;
        if (!worker_state[dp_idx]->validate_ls_pool_step_composition_noexcept(initial, iteration)) {
            latch_ls_fatal(LSFatalCode::DECODE_PREPARE_OR_VALIDATE_FAILED);
            throw LSSchedulerFatalError(LSFatalCode::DECODE_PREPARE_OR_VALIDATE_FAILED,
                                        "prepared LS pool step became stale before publication");
        }
    }

    bool has_publication = step.has_admission || step.mandatory_graph_changed;
    for (const auto& pending_dp : step.decode_by_dp) {
        has_publication = has_publication || pending_dp.prepared.has_value();
    }
    if (has_publication) {
        ls_step_publication_started_ = true;
        for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
            auto& admission = step.admissions[dp_idx];
            auto& iteration = step.decode_by_dp[dp_idx].prepared;
            if (step.pool_resource_changed[dp_idx] || admission || iteration) {
                _mark_ls_pool_resource_mutated(dp_idx);
            }
            if (admission && admission->prepared_survivors) {
                admission->prepared_survivors->commit_noexcept();
            }
            if (iteration) {
                iteration->commit_noexcept();
            }
            if (!admission) {
                continue;
            }
            for (size_t idx = 0; idx < admission->sequences.size(); ++idx) {
                auto& sequence = admission->sequences[idx];
                sequence->token_ids.push_back(0);
                sequence->last_token = 0;
                sequence->num_tokens++;
                if (admission->bootstrap_finishes[idx]) {
                    sequence->status = SequenceStatus::FINISHED;
                }
                else {
                    auto&     context = sequence->block_ctx(BlockContextSlot::ACTIVE);
                    const int master  = context.master_sp_idx_;
                    context.num_dispatched_tokens[master]++;
                    context.pending_token_present_   = true;
                    context.pending_token_target_sp_ = master;
                    sequence->status                 = SequenceStatus::RUNNING;
                }
                if (sequence->metric) {
                    if (admission->original_statuses[idx] == SequenceStatus::PAUSED_OFFLOAD) {
                        if (sequence->metric->last_token_time) {
                            sequence->metric->itl_samples.push_back(
                                (admission->bootstrap_commit_time - *sequence->metric->last_token_time) * 1000.0);
                        }
                    }
                    else {
                        if (!sequence->metric->first_scheduled_time) {
                            sequence->metric->first_scheduled_time = admission->bootstrap_commit_time;
                        }
                        if (!sequence->metric->decode_scheduled_time) {
                            sequence->metric->decode_scheduled_time = admission->bootstrap_commit_time;
                        }
                        if (!sequence->metric->first_token_time) {
                            sequence->metric->first_token_time = admission->bootstrap_commit_time;
                        }
                    }
                    sequence->metric->last_token_time = admission->bootstrap_commit_time;
                    sequence->metric->num_generated_tokens++;
                }
            }
        }

        for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
            worker_state[dp_idx]->running.swap(step.running[dp_idx]);
        }
        ls_waiting_by_dp_.swap(step.waiting);
        ls_groups_.swap(step.groups);
        ls_group_ids_by_dp_.swap(step.group_ids_by_dp);
        ls_seq_to_group_.swap(step.seq_to_group);
        ls_arrival_order_by_seq_id_.swap(step.arrival_orders);
        ls_num_ooe_.swap(step.ooe);
        ls_step_admission_records_.swap(step.admission_records);
        ls_step_initial_records_.swap(step.initial_records);
        ls_step_real_decode_ids_by_dp_.swap(step.real_decode_ids);
        ls_step_group_plans_.swap(step.group_plans);
        ls_step_group_plan_ids_.swap(step.group_plan_ids);
        ls_step_group_plan_sequence_ids_.swap(step.group_plan_sequence_ids);
        ls_step_reused_passive_masters_.swap(step.reused_passive_masters);
        ls_step_atomic_merge_count_ = step.atomic_merge_count;
    }

    ls_step_planning_latency_ms_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - planning_start).count();
    return std::move(step.scheduled);
}
catch (const LSSchedulerFatalError&) {
    throw;
}
catch (...) {
    const LSFatalCode code = ls_step_publication_started_ ? LSFatalCode::POST_PUBLICATION_INVARIANT :
                                                            LSFatalCode::DECODE_PREPARE_OR_VALIDATE_FAILED;
    latch_ls_fatal(code);
    throw LSSchedulerFatalError(code,
                                code == LSFatalCode::POST_PUBLICATION_INVARIANT ?
                                    "unexpected failure after LS combined pool publication began" :
                                    "unexpected failure while preparing the LS combined pool step");
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
    if (enable_ls_decode_core_scheduler_ && ls_fatal_.has_value()) {
        throw LSSchedulerFatalError(*ls_fatal_, "LoongServe-style scheduler is permanently fatal");
    }
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
        auto& running_queue = worker_state[dp_idx]->running;
        if (std::find(running_queue.begin(), running_queue.end(), seq) == running_queue.end()) {
            throw std::runtime_error("cannot preempt an LS sequence that is absent from its DP running queue");
        }

        auto prepared_running        = running_queue;
        auto prepared_waiting        = ls_waiting_by_dp_[dp_idx];
        auto prepared_groups         = ls_groups_;
        auto prepared_group_ids      = ls_group_ids_by_dp_;
        auto prepared_seq_to_group   = ls_seq_to_group_;
        prepared_running.erase(std::remove(prepared_running.begin(), prepared_running.end(), seq),
                               prepared_running.end());
        prepared_waiting.push_front(seq);

        const uint64_t group_id = group_owner->second;
        prepared_seq_to_group.erase(seq->seq_id);
        auto prepared_group = prepared_groups.find(group_id);
        if (prepared_group == prepared_groups.end()) {
            throw std::runtime_error("cannot prepare LS preemption for a missing group");
        }
        auto& remaining = prepared_group->second.sequences;
        remaining.erase(std::remove_if(remaining.begin(), remaining.end(), [&](const auto& candidate) {
                            return !candidate || candidate->seq_id == seq->seq_id;
                        }),
                        remaining.end());
        if (remaining.empty()) {
            auto& ids = prepared_group_ids[dp_idx];
            ids.erase(std::remove(ids.begin(), ids.end(), group_id), ids.end());
            prepared_groups.erase(prepared_group);
        }
        else {
            auto& prepared_state = prepared_group->second;
            prepared_state.kv_candidate_target_dop   = -1;
            prepared_state.kv_candidate_stable_steps = 0;
            prepared_state.kv_candidate_member_ids.clear();
            prepared_state.kv_candidate_allocation.clear();
            auto used = worker_state[dp_idx]->group_used_kv_tokens(prepared_state.sequences);
            prepared_state.allocated_attention_ranks.erase(
                std::remove_if(prepared_state.allocated_attention_ranks.begin(),
                               prepared_state.allocated_attention_ranks.end(),
                               [&](int rank) {
                                   bool protected_role = std::any_of(prepared_state.sequences.begin(),
                                                              prepared_state.sequences.end(),
                                                              [&](const auto& candidate) {
                                                                  if (!candidate
                                                                      || candidate->status
                                                                             != SequenceStatus::RUNNING) {
                                                                      return false;
                                                                  }
                                                                  const auto& context = candidate->block_ctx(
                                                                      BlockContextSlot::ACTIVE);
                                                                  return context.master_sp_idx_ == rank
                                                                         || (context.pending_token_present_
                                                                             && context.pending_token_target_sp_
                                                                                    == rank);
                                                              });
                                   return used[rank] == 0 && !protected_role;
                               }),
                prepared_state.allocated_attention_ranks.end());
        }

        auto prepared_release = worker_state[dp_idx]->prepare_ls_release(seq);
        if (!prepared_release.validate_precommit_noexcept()) {
            throw std::runtime_error("prepared LS preemption became stale before publication");
        }
        ls_step_publication_started_ = true;
        // Explicit preempt is called outside schedule() and therefore starts a
        // new resource-publication boundary of its own.
        _mark_ls_pool_resource_mutated(dp_idx, true);
        prepared_release.commit_noexcept();
        seq->status = SequenceStatus::PAUSED_OFFLOAD;
        running_queue.swap(prepared_running);
        ls_waiting_by_dp_[dp_idx].swap(prepared_waiting);
        ls_groups_.swap(prepared_groups);
        ls_group_ids_by_dp_.swap(prepared_group_ids);
        ls_seq_to_group_.swap(prepared_seq_to_group);
        const auto& active = seq->block_ctx(BlockContextSlot::ACTIVE);
        const bool context_empty = active.master_sp_idx_ == -1 && !active.pending_token_present_
                                   && active.pending_token_target_sp_ == -1 && active.block_location.empty()
                                   && std::all_of(active.sp_block_table.begin(),
                                                  active.sp_block_table.end(),
                                                  [](const auto& table) { return table.empty(); })
                                   && std::all_of(active.num_dispatched_tokens.begin(),
                                                  active.num_dispatched_tokens.end(),
                                                  [](int tokens) { return tokens == 0; });
        if (active.dp_idx_ != dp_idx || !context_empty) {
            latch_ls_fatal(LSFatalCode::POST_PUBLICATION_INVARIANT);
            throw LSSchedulerFatalError(LSFatalCode::POST_PUBLICATION_INVARIANT,
                                        "prepared LS preemption published an invalid paused request");
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
    if (enable_ls_decode_core_scheduler_ && ls_fatal_.has_value()) {
        throw LSSchedulerFatalError(*ls_fatal_, "LoongServe-style scheduler is permanently fatal");
    }
    if (active_ls_kv_transaction_) {
        throw std::runtime_error("cannot postprocess while an LS KV scale-down transaction is reserved");
    }
    try {
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
    catch (const LSSchedulerFatalError&) {
        throw;
    }
    catch (...) {
        if (enable_ls_decode_core_scheduler_) {
            latch_ls_fatal(LSFatalCode::POST_PUBLICATION_INVARIANT);
            throw LSSchedulerFatalError(LSFatalCode::POST_PUBLICATION_INVARIANT,
                                        "unexpected LS postprocess publication failure");
        }
        throw;
    }
}

void Scheduler::free_to_be_migrated(std::shared_ptr<Sequence> seq)
{
    if (enable_ls_decode_core_scheduler_ && ls_fatal_.has_value()) {
        throw LSSchedulerFatalError(*ls_fatal_, "LoongServe-style scheduler is permanently fatal");
    }
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
