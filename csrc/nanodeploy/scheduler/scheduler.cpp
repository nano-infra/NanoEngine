#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
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
                     const std::string& scheduler_mode,
                     bool               loongserve_decode_scheduler,
                     bool               loongserve_enable_kv_migration,
                     const std::string& loongserve_migration_granularity,
                     int                loongserve_min_comp_bound_batch_size) :
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
    sp_master_selector_(sp_master_selector),
    loongserve_decode_scheduler_(loongserve_decode_scheduler),
    loongserve_enable_kv_migration_(loongserve_enable_kv_migration),
    loongserve_migration_granularity_(loongserve_migration_granularity),
    loongserve_min_comp_bound_batch_size_(std::max(1, loongserve_min_comp_bound_batch_size))
{
    if (loongserve_migration_granularity_ != "block") {
        throw std::runtime_error("loongserve_migration_granularity currently only supports 'block'");
    }
    if (loongserve_decode_scheduler_ && loop_count_ != 1) {
        throw std::runtime_error(
            "loongserve_decode_scheduler requires loop_count=1 because no-migration elastic "
            "scale-up/down changes decode KV ownership at scheduler-step granularity");
    }
    if (loongserve_decode_scheduler_ && scheduler_mode != "centralized") {
        throw std::runtime_error("loongserve_decode_scheduler currently supports centralized scheduler_mode only");
    }

    Sequence::block_size = kvcache_block_size;
    // Initialize worker states
    worker_state.reserve(attention_dp_);
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        auto sp_manager = std::make_shared<SPStateManager>(
            engine_id_, attention_sp_, num_kvcache_blocks, kvcache_block_size, 
            max_num_seqs_, max_num_batched_tokens_, max_num_recv_seqs_,
            reserved_blocks_per_req_, segment_size_, enable_dynamic_sp_size_,
            dynamic_sp_size_strategy_, dynamic_sp_long_request_threshold_,
            dynamic_sp_long_request_size_,
            enable_dynamic_sp_bucket_policy, dynamic_sp_bucket_policy,
            attention_cost_a, attention_cost_b,
            q_cost_a, q_cost_b,
            res_cost_a, res_cost_b,
            lse_cost_a, lse_cost_b,
            q_bytes_per_edge, res_bytes_per_edge, lse_bytes_per_edge,
            enable_non_uniform_split,
            sp_master_selector, sp_debug_, fixed_sp_size);
        
        sp_manager->set_dp_idx(dp_idx);
        worker_state.push_back(sp_manager);
    }
    // Set scheduler mode
    if (scheduler_mode == "decentralized") {
        scheduler_mode_ = SchedulerMode::DECENTRALIZED;
    } else {
        scheduler_mode_ = SchedulerMode::CENTRALIZED;
    }
    
    std::cerr << "[Scheduler] Initialized with segment_size=" << segment_size_ 
              << ", fixed_sp_size=" << fixed_sp_size
              << ", use_new_decode_dynamic_sp_scheduler=" << use_new_decode_dynamic_sp_scheduler_
              << ", dynamic_sp_size_strategy=" << dynamic_sp_size_strategy_
              << ", dynamic_sp_long_request_threshold=" << dynamic_sp_long_request_threshold_
              << ", dynamic_sp_long_request_size=" << dynamic_sp_long_request_size_
              << ", scheduler_mode=" << (scheduler_mode_ == SchedulerMode::DECENTRALIZED ? "decentralized" : "centralized")
              << ", loongserve_decode_scheduler=" << loongserve_decode_scheduler_
              << ", loongserve_enable_kv_migration=" << loongserve_enable_kv_migration_
              << ", loongserve_min_comp_bound_batch_size=" << loongserve_min_comp_bound_batch_size_
              << std::endl;
    thread_pool_ = std::make_unique<ThreadPool>(attention_dp_);
}

void Scheduler::add(std::shared_ptr<Sequence> seq)
{
    seq->active(engine_id_, attention_sp_, attention_dp_);

    if (seq->metric) {
        seq->metric->record_arrival();
    }

    if (scheduler_mode_ == SchedulerMode::DECENTRALIZED) {
        // Decentralized mode: immediately route to selected DP worker
        int selected_dp_idx = select_dp_worker_for_routing(*seq);
        auto& target_queue = (mode_ == "decode") ? 
            worker_state[selected_dp_idx]->waiting_migration : 
            worker_state[selected_dp_idx]->waiting;
        target_queue.push_back(seq);
        
        // Set dp_idx (though resources haven't been allocated yet)
        seq->block_ctx(BlockContextSlot::ACTIVE).dp_idx_ = selected_dp_idx;
        
        if (mode_ == "decode" && seq->metric) {
            seq->metric->record_decode_arrival();
        }
    } else {
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
    } else {
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
    if (scheduler_mode_ == SchedulerMode::DECENTRALIZED) {
        int total = 0;
        for (const auto& ws : worker_state) {
            total += static_cast<int>(ws->waiting.size());
        }
        return total;
    } else {
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
    } else {
        return static_cast<int>(waiting_migration.size());
    }
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
        int best_idx = 0;
        int min_score = std::numeric_limits<int>::max();
        
        for (int i = 0; i < attention_dp_; ++i) {
            int waiting = worker_state[i]->get_waiting_queue_size();
            int running = worker_state[i]->num_running_seqs();
            
            // vLLM formula: score = waiting * 4 + running
            // waiting has 4x weight compared to running
            int score = waiting * 4 + running;
            
            if (score < min_score) {
                min_score = score;
                best_idx = i;
            }
        }
        return best_idx;
    }
    return 0;
}

ScheduleResult Scheduler::schedule()
{
    std::vector<std::vector<std::shared_ptr<Sequence>>> dp_seqs;
    bool has_prefill = false;
    
    if (scheduler_mode_ == SchedulerMode::DECENTRALIZED) {
        return _schedule_decentralized();
    } else {
        // Centralized mode: original logic
        // Try prefill first
        dp_seqs = _schedule_prefill();

        // Check if any sequences were scheduled in prefill
        for (const auto& seqs : dp_seqs) {
            if (!seqs.empty()) {
                has_prefill = true;
                break;
            }
        }

        if (!has_prefill) {
            // No prefill sequences, schedule decode
            dp_seqs = _schedule_decode();
        }
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

    result.loongserve_occupied_instances.resize(attention_dp_);
    result.loongserve_append_instances.resize(attention_dp_);
    result.loongserve_draining_instances.resize(attention_dp_);
    if (loongserve_decode_scheduler_) {
        for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
            auto it = decode_batch_state_by_dp_.find(dp_idx);
            if (it == decode_batch_state_by_dp_.end()) {
                continue;
            }
            result.loongserve_occupied_instances[dp_idx] = it->second.occupied_instances;
            result.loongserve_append_instances[dp_idx] = it->second.append_sp_for_step;
            result.loongserve_draining_instances[dp_idx] = it->second.draining_instances;
        }
    }

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
    if (scheduler_mode_ == SchedulerMode::CENTRALIZED
        && mode_ == "decode"
        && enable_dynamic_sp_size_
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
    if (mode_ == "decode" && attention_sp_ > 1 && loongserve_decode_scheduler_) {
        return _schedule_loongserve_decode();
    }

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

std::vector<std::vector<std::shared_ptr<Sequence>>> Scheduler::_schedule_loongserve_decode()
{
    std::vector<std::vector<std::shared_ptr<Sequence>>> scheduled_seqs(attention_dp_);
    pending_decode_kv_migration_plans_.clear();

    auto contains_rank = [](const std::vector<int>& ranks, int rank) {
        return std::find(ranks.begin(), ranks.end(), rank) != ranks.end();
    };

    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        auto& worker = worker_state[dp_idx];
        auto& running_queue = worker->running;

        std::vector<std::shared_ptr<Sequence>> candidates;
        candidates.reserve(running_queue.size());
        while (!running_queue.empty()) {
            candidates.push_back(running_queue.front());
            running_queue.pop_front();
        }

        if (candidates.empty()) {
            decode_batch_state_by_dp_.erase(dp_idx);
            for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
                scheduled_seqs[dp_idx].push_back(worker->dummy_seqs[sp_idx]);
            }
            continue;
        }

        DecodeBatchState batch_state;
        batch_state.batch_id = static_cast<uint64_t>(dp_idx);
        batch_state.dp_idx = dp_idx;
        batch_state.seqs = candidates;
        batch_state.batch_used_tokens_per_sp = worker->decode_batch_used_tokens_per_sp(candidates);

        std::vector<int> total_used_tokens_per_sp = worker->decode_total_used_tokens_per_sp();
        std::vector<int> free_tokens_per_sp = worker->decode_free_tokens_per_sp();
        std::vector<int> existing_kv_ranks = worker->decode_occupied_instances(batch_state.batch_used_tokens_per_sp);

        int decode_tokens_needed = static_cast<int>(candidates.size()) * std::max(1, loop_count_);

        int desired_compute_masters = 1;
        if ((int)candidates.size() >= loongserve_min_comp_bound_batch_size_) {
            desired_compute_masters =
                ((int)candidates.size() + loongserve_min_comp_bound_batch_size_ - 1)
                / loongserve_min_comp_bound_batch_size_;
            desired_compute_masters = std::min(desired_compute_masters, attention_sp_);
        }

        std::vector<int> append_instances;
        int append_free_tokens = 0;
        auto add_append_instance = [&](int sp_idx) {
            if (sp_idx < 0 || sp_idx >= attention_sp_ || free_tokens_per_sp[sp_idx] <= 0
                || contains_rank(append_instances, sp_idx)) {
                return false;
            }
            append_instances.push_back(sp_idx);
            append_free_tokens += free_tokens_per_sp[sp_idx];
            return true;
        };

        std::vector<int> existing_append_candidates = existing_kv_ranks;
        std::sort(existing_append_candidates.begin(), existing_append_candidates.end(), [&](int lhs, int rhs) {
            if (free_tokens_per_sp[lhs] != free_tokens_per_sp[rhs]) {
                return free_tokens_per_sp[lhs] > free_tokens_per_sp[rhs];
            }
            int lhs_used = lhs < (int)batch_state.batch_used_tokens_per_sp.size()
                               ? batch_state.batch_used_tokens_per_sp[lhs]
                               : 0;
            int rhs_used = rhs < (int)batch_state.batch_used_tokens_per_sp.size()
                               ? batch_state.batch_used_tokens_per_sp[rhs]
                               : 0;
            if (lhs_used != rhs_used) {
                return lhs_used > rhs_used;
            }
            return lhs < rhs;
        });

        for (int sp_idx : existing_append_candidates) {
            if ((int)append_instances.size() >= desired_compute_masters
                && append_free_tokens >= decode_tokens_needed) {
                break;
            }
            add_append_instance(sp_idx);
        }

        if ((int)append_instances.size() < desired_compute_masters
            || append_free_tokens < decode_tokens_needed) {
            std::vector<int> scale_up_candidates;
            for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
                if (!contains_rank(append_instances, sp_idx) && free_tokens_per_sp[sp_idx] > 0) {
                    scale_up_candidates.push_back(sp_idx);
                }
            }
            std::sort(scale_up_candidates.begin(), scale_up_candidates.end(), [&](int lhs, int rhs) {
                bool lhs_has_batch_kv = contains_rank(existing_kv_ranks, lhs);
                bool rhs_has_batch_kv = contains_rank(existing_kv_ranks, rhs);
                if (lhs_has_batch_kv != rhs_has_batch_kv) {
                    return lhs_has_batch_kv;
                }
                bool lhs_idle = lhs < (int)total_used_tokens_per_sp.size() && total_used_tokens_per_sp[lhs] == 0;
                bool rhs_idle = rhs < (int)total_used_tokens_per_sp.size() && total_used_tokens_per_sp[rhs] == 0;
                if (lhs_idle != rhs_idle) {
                    return lhs_idle;
                }
                if (free_tokens_per_sp[lhs] != free_tokens_per_sp[rhs]) {
                    return free_tokens_per_sp[lhs] > free_tokens_per_sp[rhs];
                }
                return lhs < rhs;
            });
            for (int sp_idx : scale_up_candidates) {
                if ((int)append_instances.size() >= desired_compute_masters
                    && append_free_tokens >= decode_tokens_needed) {
                    break;
                }
                add_append_instance(sp_idx);
            }
        }

        std::vector<int> occupied = existing_kv_ranks;
        for (int sp_idx : append_instances) {
            if (!contains_rank(occupied, sp_idx)) {
                occupied.push_back(sp_idx);
            }
        }
        std::sort(occupied.begin(), occupied.end());

        batch_state.append_sp_for_step = append_instances;
        for (int sp_idx : existing_kv_ranks) {
            if (!contains_rank(append_instances, sp_idx)) {
                batch_state.draining_instances.push_back(sp_idx);
            }
        }
        std::sort(batch_state.draining_instances.begin(), batch_state.draining_instances.end());

        if (append_instances.empty()) {
            for (const auto& seq : candidates) {
                preempt(dp_idx, seq);
            }
            decode_batch_state_by_dp_[dp_idx] = std::move(batch_state);
            continue;
        }

        int chunk_size = ((int)candidates.size() + (int)append_instances.size() - 1) / (int)append_instances.size();

        std::vector<int> master_scheduled_counts(attention_sp_, 0);
        std::vector<int> remote_recv_scheduled_counts(attention_sp_, 0);
        std::vector<int> sp_lens(attention_sp_, 0);
        std::deque<std::shared_ptr<Sequence>> skipped;

        auto choose_current_master = [&](const Sequence& seq) {
            const auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
            const auto& tokens = block_ctx.num_dispatched_tokens;
            int current_master = block_ctx.master_sp_idx_;
            if (current_master >= 0 && current_master < attention_sp_
                && current_master < (int)tokens.size() && tokens[current_master] > 0) {
                return current_master;
            }
            int fallback = -1;
            int max_tokens = -1;
            for (int sp_idx = 0; sp_idx < std::min(attention_sp_, (int)tokens.size()); ++sp_idx) {
                if (tokens[sp_idx] > 0 && tokens[sp_idx] > max_tokens) {
                    fallback = sp_idx;
                    max_tokens = tokens[sp_idx];
                }
            }
            return fallback;
        };

        for (size_t seq_idx = 0; seq_idx < candidates.size(); ++seq_idx) {
            auto& seq = candidates[seq_idx];
            int preferred_master_idx = std::min(
                static_cast<int>(append_instances.size()) - 1,
                static_cast<int>(seq_idx) / std::max(1, chunk_size));

            int current_master = choose_current_master(*seq);
            if (current_master < 0 || current_master >= attention_sp_) {
                if (loongserve_enable_kv_migration_) {
                    DecodeKVMigrationPlan plan;
                    plan.dp_idx = dp_idx;
                    pending_decode_kv_migration_plans_.push_back(std::move(plan));
                }
                preempt(dp_idx, seq);
                continue;
            }

            if (master_scheduled_counts[current_master] >= max_num_seqs_) {
                skipped.push_back(seq);
                continue;
            }

            const auto& tokens = seq->block_ctx(BlockContextSlot::ACTIVE).num_dispatched_tokens;
            bool exceeds_remote_recv_capacity = false;
            for (int sp_idx = 0; sp_idx < std::min(attention_sp_, (int)tokens.size()); ++sp_idx) {
                if (sp_idx == current_master || tokens[sp_idx] <= 0) {
                    continue;
                }
                if (remote_recv_scheduled_counts[sp_idx] >= max_num_recv_seqs_) {
                    exceeds_remote_recv_capacity = true;
                    break;
                }
            }
            if (exceeds_remote_recv_capacity) {
                skipped.push_back(seq);
                continue;
            }

            int selected_append_sp = -1;
            if (loop_count_ == 1) {
                for (int attempt = 0; attempt < static_cast<int>(append_instances.size()); ++attempt) {
                    int append_sp = append_instances[(preferred_master_idx + attempt) % append_instances.size()];
                    if (!worker->can_append_on_sp(*seq, append_sp, loop_count_)) {
                        continue;
                    }
                    selected_append_sp = append_sp;
                    break;
                }
            }
            else if (worker->can_append_on_sp(*seq, current_master, loop_count_)) {
                selected_append_sp = current_master;
            }

            if (selected_append_sp < 0) {
                if (worker->can_append_on_sp(*seq, current_master, loop_count_)) {
                    selected_append_sp = current_master;
                }
                else if (loongserve_enable_kv_migration_) {
                    DecodeKVMigrationPlan plan;
                    plan.dp_idx = dp_idx;
                    pending_decode_kv_migration_plans_.push_back(std::move(plan));
                }
            }

            if (selected_append_sp < 0) {
                preempt(dp_idx, seq);
                continue;
            }

            worker->set_decode_master(*seq, current_master);

            auto& block_ctx = seq->block_ctx(BlockContextSlot::ACTIVE);
            block_ctx.append_sp_idx_ = selected_append_sp != current_master ? selected_append_sp : -1;

            if (!worker->may_append_on_sp(*seq, selected_append_sp, loop_count_)) {
                block_ctx.append_sp_idx_ = -1;
                preempt(dp_idx, seq);
                continue;
            }

            master_scheduled_counts[current_master]++;
            for (int sp_idx = 0; sp_idx < std::min(attention_sp_, (int)tokens.size()); ++sp_idx) {
                if (sp_idx != current_master && tokens[sp_idx] > 0) {
                    remote_recv_scheduled_counts[sp_idx]++;
                }
            }
            scheduled_seqs[dp_idx].push_back(seq);
            sp_lens[current_master] += seq->num_tokens;
        }

        int range_begin = -1;
        int range_master = -1;
        for (int seq_idx = 0; seq_idx < static_cast<int>(scheduled_seqs[dp_idx].size()); ++seq_idx) {
            int master = scheduled_seqs[dp_idx][seq_idx]->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
            if (master != range_master) {
                if (range_begin >= 0) {
                    batch_state.master_sp_for_step.push_back(range_master);
                    batch_state.mini_batch_ranges.push_back({range_begin, seq_idx});
                }
                range_begin = seq_idx;
                range_master = master;
            }
        }
        if (range_begin >= 0) {
            batch_state.master_sp_for_step.push_back(range_master);
            batch_state.mini_batch_ranges.push_back({range_begin, static_cast<int>(scheduled_seqs[dp_idx].size())});
        }

        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            bool is_append_instance = contains_rank(append_instances, sp_idx);
            bool has_batch_kv = sp_idx < (int)batch_state.batch_used_tokens_per_sp.size()
                                && batch_state.batch_used_tokens_per_sp[sp_idx] > 0;
            if ((has_batch_kv || is_append_instance) && !contains_rank(batch_state.occupied_instances, sp_idx)) {
                batch_state.occupied_instances.push_back(sp_idx);
            }
        }
        std::sort(batch_state.occupied_instances.begin(), batch_state.occupied_instances.end());

        for (auto it = scheduled_seqs[dp_idx].rbegin(); it != scheduled_seqs[dp_idx].rend(); ++it) {
            running_queue.push_front(*it);
        }
        for (auto it = skipped.rbegin(); it != skipped.rend(); ++it) {
            running_queue.push_front(*it);
        }

        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            if (sp_lens[sp_idx] == 0) {
                scheduled_seqs[dp_idx].push_back(worker->dummy_seqs[sp_idx]);
            }
        }

        decode_batch_state_by_dp_[dp_idx] = std::move(batch_state);
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

    // Reset sequence to prompt-only state (discard generated tokens)
    int prompt_len = seq->num_prompt_tokens;
    seq->token_ids.resize(prompt_len);
    seq->num_tokens = prompt_len;
    seq->num_checkpointed_tokens = prompt_len;
    seq->last_token = seq->token_ids.empty() ? 0 : seq->token_ids.back();

    seq->status = SequenceStatus::WAITING;
    worker_state[dp_idx]->deallocate(*seq);
    
    // Re-initialize BlockContext for fresh scheduling
    seq->active(engine_id_, attention_sp_, attention_dp_);

    if (scheduler_mode_ == SchedulerMode::DECENTRALIZED) {
        // Decentralized mode: put back to the worker's queue
        auto& target_queue = (mode_ == "decode") ? 
            worker_state[dp_idx]->waiting_migration : 
            worker_state[dp_idx]->waiting;
        target_queue.push_front(seq);
        // Keep the dp_idx that was set during routing
        seq->block_ctx(BlockContextSlot::ACTIVE).dp_idx_ = dp_idx;
    } else {
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

ScheduleResult Scheduler::_schedule_decentralized()
{
    std::vector<std::vector<std::shared_ptr<Sequence>>> scheduled_seqs(attention_dp_);
    bool has_prefill = false;

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
            int send_count = 0;
            const auto& sp_seqs = result.filtered_dp_sp_seqs[dp_idx * attention_sp_ + sp_idx];
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

        // SP Q Matrix
        result.sp_q_matrix.push_back(std::vector<std::vector<int>>(attention_sp_, std::vector<int>(attention_sp_, 0)));
        result.sp_res_matrix.push_back(std::vector<std::vector<int>>(attention_sp_, std::vector<int>(attention_sp_, 0)));

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

            const auto& tokens       = seq->block_ctx(BlockContextSlot::ACTIVE).num_dispatched_tokens;
            int         active_ranks = 0;
            for (int count : tokens) {
                if (count > 0)
                    active_ranks++;
            }

            if (active_ranks > 1) {
                int master_sp_idx = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
                for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
                    if (tokens[sp_idx] > 0 && sp_idx != master_sp_idx) {
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
        auto& worker = worker_state[dp_idx];
        auto& wait_queue = (mode_ != "decode") ? worker->waiting : worker->waiting_migration;
        
        if (!wait_queue.empty()) {
            auto head_seq = wait_queue.front();
            result.waiting_head_blocks[dp_idx] = (head_seq->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
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
    auto& worker = worker_state[dp_idx];
    auto& waiting_queue = (mode_ != "decode") ? worker->waiting : worker->waiting_migration;

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
        auto& block_ctx   = seq->block_ctx(BlockContextSlot::ACTIVE);
        int master_sp_idx = block_ctx.master_sp_idx_;
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
    auto& worker = worker_state[dp_idx];
    auto& running_queue = worker->running;

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
