#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "nanodeploy/metrics/sequence_metric.h"
#include "nanodeploy/sequence/sequence.h"

#include "scheduler_utils.h"

#include "scheduler.h"

namespace nanodeploy {

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
                     bool               use_new_decode_dynamic_sp_scheduler,
                     const std::string& dynamic_sp_size_strategy,
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
                     int                fixed_sp_size) :
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
    use_new_decode_dynamic_sp_scheduler_(use_new_decode_dynamic_sp_scheduler),
    dynamic_sp_size_strategy_(dynamic_sp_size_strategy),
    enable_non_uniform_split_(enable_non_uniform_split),
    sp_master_selector_(sp_master_selector)
{
    Sequence::block_size = kvcache_block_size;
    // Initialize worker states
    worker_state.reserve(attention_dp_);
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        auto sp_manager = std::make_shared<SPStateManager>(
            engine_id_, attention_sp_, num_kvcache_blocks, kvcache_block_size, 
            max_num_seqs_, max_num_batched_tokens_, max_num_recv_seqs_,
            reserved_blocks_per_req_, segment_size_, dynamic_sp_size_strategy_,
            enable_dynamic_sp_bucket_policy, dynamic_sp_bucket_policy,
            attention_cost_a, attention_cost_b,
            q_cost_a, q_cost_b,
            res_cost_a, res_cost_b,
            lse_cost_a, lse_cost_b,
            q_bytes_per_edge, res_bytes_per_edge, lse_bytes_per_edge,
            enable_non_uniform_split,
            sp_master_selector, fixed_sp_size);
        
        sp_manager->set_dp_idx(dp_idx);
        worker_state.push_back(sp_manager);
    }
    std::cerr << "[Scheduler] Initialized with segment_size=" << segment_size_ 
              << ", fixed_sp_size=" << fixed_sp_size
              << ", use_new_decode_dynamic_sp_scheduler=" << use_new_decode_dynamic_sp_scheduler_
              << ", dynamic_sp_size_strategy=" << dynamic_sp_size_strategy_
              << std::endl;
    thread_pool_ = std::make_unique<ThreadPool>(attention_dp_);
}

void Scheduler::add(std::shared_ptr<Sequence> seq)
{
    seq->active(engine_id_, attention_sp_, attention_dp_);

    if (seq->metric) {
        seq->metric->record_arrival();
    }

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

bool Scheduler::is_finished() const
{
    const auto& wait_queue = (mode_ != "decode") ? waiting : waiting_migration;
    if (!wait_queue.empty())
        return false;

    for (const auto& ws : worker_state) {
        if (!ws->is_empty())
            return false;
    }
    return true;
}

int Scheduler::get_total_waiting_size() const
{
    return static_cast<int>(waiting.size());
}

int Scheduler::get_total_waiting_migration_size() const
{
    return static_cast<int>(waiting_migration.size());
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

std::vector<std::vector<std::shared_ptr<Sequence>>> Scheduler::admit()
{
    return _schedule_prefill();
}

std::vector<std::vector<std::shared_ptr<Sequence>>> Scheduler::plan_decode()
{
    return _schedule_decode();
}

void Scheduler::append_missing_control_dummies(
    std::vector<std::vector<std::shared_ptr<Sequence>>>& dp_seqs)
{
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        std::vector<bool> has_master(attention_sp_, false);
        for (const auto& seq : dp_seqs[dp_idx]) {
            int master_sp_idx =
                seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
            if (master_sp_idx >= 0 && master_sp_idx < attention_sp_) {
                has_master[master_sp_idx] = true;
            }
        }
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            if (!has_master[sp_idx]) {
                dp_seqs[dp_idx].push_back(
                    worker_state[dp_idx]->dummy_seqs[sp_idx]);
            }
        }
    }
}

ScheduleResult Scheduler::schedule()
{
    std::vector<std::vector<std::shared_ptr<Sequence>>> dp_seqs;
    bool has_prefill = false;

    dp_seqs = admit();

    for (const auto& seqs : dp_seqs) {
        if (!seqs.empty()) {
            has_prefill = true;
            break;
        }
    }

    if (has_prefill) {
        append_missing_control_dummies(dp_seqs);
    }
    else {
        dp_seqs = plan_decode();
    }

    ScheduleResult result;
    result.dp_seqs    = dp_seqs;
    result.is_prefill = has_prefill;

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
                const auto& tokens       = seq->block_ctx(BlockContextSlot::ACTIVE).num_dispatched_tokens;
                int         active_ranks = 0;
                for (int count : tokens) {
                    if (count > 0)
                        active_ranks++;
                }

                if (active_ranks > 1) {
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
                    const auto& block_ctx    = seq->block_ctx(BlockContextSlot::ACTIVE);
                    const auto& tokens       = block_ctx.num_dispatched_tokens;
                    
                    int         active_ranks = 0;
                    for (int count : tokens) {
                        if (count > 0)
                            active_ranks++;
                    }

                    int master_sp_idx = block_ctx.master_sp_idx_;

                    if (active_ranks > 1 && tokens[sp_idx] > 0 && master_sp_idx != sp_idx) {
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

            const auto& tokens = seq->block_ctx(BlockContextSlot::ACTIVE).num_dispatched_tokens;
            int         active_ranks = 0;
            for (int count : tokens) {
                if (count > 0) {
                    active_ranks++;
                }
            }

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

            const auto& tokens       = seq->block_ctx(BlockContextSlot::ACTIVE).num_dispatched_tokens;
            int         active_ranks = 0;
            for (int count : tokens) {
                if (count > 0)
                    active_ranks++;
            }

            // Only count if SP is truly enabled (distributed across > 1 ranks)
            if (active_ranks > 1) {
                int master_sp_idx = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;

                // For each participating rank:
                for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
                    if (tokens[sp_idx] > 0) {
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
    
    int head_blocks = 0;
    int total_blocks = 0;
    
    if (!wait_queue.empty()) {
        auto head_seq = wait_queue.front();
        head_blocks = (head_seq->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
    }
    
    for (const auto& seq : wait_queue) {
        total_blocks += (seq->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
    }
    
    // Replicate to all DP workers (centralized has shared global queue)
    result.waiting_head_blocks.resize(attention_dp_, head_blocks);
    result.waiting_total_blocks.resize(attention_dp_, total_blocks);

    return result;
}

std::vector<std::vector<std::shared_ptr<Sequence>>> Scheduler::_schedule_decode_prefill_latency_aware()
{
    std::vector<std::vector<std::shared_ptr<Sequence>>> scheduled_seqs(attention_dp_);
    std::vector<std::vector<std::shared_ptr<Sequence>>> tentative_batches(attention_dp_);
    std::vector<std::optional<SPStateManager::DecodeBatchPlan>> tentative_plans(attention_dp_);

    struct PlanningCacheGuard {
        std::vector<std::shared_ptr<SPStateManager>>& workers;
        explicit PlanningCacheGuard(std::vector<std::shared_ptr<SPStateManager>>& worker_state) : workers(worker_state)
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
            int projected_batch = worker_state[dp_idx]->num_running_seqs()
                                  + static_cast<int>(tentative_batches[dp_idx].size());
            dp_order.push_back({projected_batch, dp_idx});
        }
        std::sort(dp_order.begin(), dp_order.end());

        bool admitted = false;
        for (const auto& entry : dp_order) {
            int dp_idx = entry.second;
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

        auto& plan = *tentative_plans[dp_idx];
        auto& batch = tentative_batches[dp_idx];
        for (size_t i = 0; i < batch.size(); ++i) {
            auto& seq = batch[i];
            worker_state[dp_idx]->apply_planned_placement(*seq, plan.placements[i]);

            auto& block_ctx = seq->block_ctx(BlockContextSlot::ACTIVE);
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

std::vector<std::vector<std::shared_ptr<Sequence>>> Scheduler::_schedule_prefill()
{
    if (mode_ == "decode"
        && use_new_decode_dynamic_sp_scheduler_
        && routing_strategy == RoutingStrategy::LeastBatch) {
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
    std::cerr << "Preemption happens for seq_id=" << seq->seq_id << std::endl;

    // Reset metrics for fresh start
    if (seq->metric) {
        seq->metric->on_preemption();
    }

    // Release accounting and KV state while the sequence still reflects every
    // allocated token, including the hierarchical bootstrap token.
    worker_state[dp_idx]->deallocate(*seq);

    // Reset sequence to prompt-only state (discard generated tokens)
    int prompt_len = seq->num_prompt_tokens;
    seq->token_ids.resize(prompt_len);
    seq->num_tokens = prompt_len;
    seq->num_bootstrap_tokens = 0;
    seq->num_checkpointed_tokens = prompt_len;
    seq->last_token = seq->token_ids.empty() ? 0 : seq->token_ids.back();

    seq->status = SequenceStatus::WAITING;

    // Re-initialize BlockContext for fresh scheduling
    seq->active(engine_id_, attention_sp_, attention_dp_);

    if (mode_ == "decode") {
        waiting_migration.push_front(seq);
    }
    else {
        waiting.push_front(seq);
    }
}

void Scheduler::postprocess(const std::vector<std::vector<std::shared_ptr<Sequence>>>& dp_sp_seqs,
                            const std::vector<std::vector<std::vector<int>>>&          dp_sp_token_ids,
                            bool                                                       update_metrics,
                            double                                                     accumulated_step_time_ms,
                            int                                                        loop_count)
{
    // Call the C++ postprocess_sequences utility directly with shared_ptrs
    auto migrations = postprocess_sequences(
        worker_state, dp_sp_seqs, dp_sp_token_ids, eos_, mode_ == "prefill", update_metrics, accumulated_step_time_ms, loop_count, thread_pool_.get());

    // Store migrations
    for (const auto& [seq_shared, dp_idx] : migrations) {
        to_be_migrated[seq_shared->seq_id] = {seq_shared, dp_idx};
    }
}

void Scheduler::free_to_be_migrated(std::shared_ptr<Sequence> seq)
{
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

}  // namespace nanodeploy
