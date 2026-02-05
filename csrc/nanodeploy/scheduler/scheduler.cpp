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
                     bool               enable_dynamic_sp_size,
                     bool               enable_non_uniform_split,
                     const std::string& sp_master_selector,
                     bool               sp_debug,
                     int                fixed_sp_segments,
                     const std::string& scheduler_mode) :
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
    enable_non_uniform_split_(enable_non_uniform_split),
    sp_debug_(sp_debug),
    sp_master_selector_(sp_master_selector)
{
    Sequence::block_size = kvcache_block_size;
    // Initialize worker states
    worker_state.reserve(attention_dp_);
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        auto sp_manager = std::make_shared<SPStateManager>(
            engine_id_, attention_sp_, num_kvcache_blocks, kvcache_block_size, 
            max_num_seqs_, max_num_batched_tokens_, max_num_recv_seqs_,
            reserved_blocks_per_req_, segment_size_, enable_dynamic_sp_size_, 
            enable_non_uniform_split,
            sp_master_selector, sp_debug_, fixed_sp_segments);
        
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
              << ", fixed_sp_segments=" << fixed_sp_segments
              << ", scheduler_mode=" << (scheduler_mode_ == SchedulerMode::DECENTRALIZED ? "decentralized" : "centralized")
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

    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        result.sp_send_counts[dp_idx].resize(attention_sp_);
        result.sp_recv_counts[dp_idx].resize(attention_sp_);

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

        // SP Communication Matrix Logic
        // Initialize matrix for this DP rank: [attention_sp_][attention_sp_]
        // result.sp_comm_matrix.push_back(
        // std::vector<std::vector<int>>(attention_sp_, std::vector<int>(attention_sp_, 0)));

        result.sp_q_matrix.push_back(std::vector<std::vector<int>>(attention_sp_, std::vector<int>(attention_sp_, 0)));

        // result.sp_res_matrix.push_back(
        //     std::vector<std::vector<int>>(attention_sp_, std::vector<int>(attention_sp_, 0)));

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
                        // Original matrix (Master -> Participant) - kept for compatibility if needed
                        // result.sp_comm_matrix[dp_idx][master_sp_idx][sp_idx]++;

                        // Q Matrix: Master broadcast to all Participants
                        // Master sends Q to Participant
                        result.sp_q_matrix[dp_idx][master_sp_idx][sp_idx]++;

                        // Res Matrix: Participant sends results back to Master
                        // Participant sends Res to Master
                        // result.sp_res_matrix[dp_idx][sp_idx][master_sp_idx]++;
                    }
                }
            }
        }
    }

    // Calculate waiting queue block metrics
    auto& wait_queue = (mode_ != "decode") ? waiting : waiting_migration;

    if (!wait_queue.empty()) {
        auto head_seq = wait_queue.front();
        // Calculate blocks for head sequence: ceil(num_tokens / block_size)
        // Note: We use Sequence::block_size which is static constexpr int block_size = 256;
        result.waiting_head_blocks = (head_seq->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
    }

    int total_blocks = 0;
    for (const auto& seq : wait_queue) {
        total_blocks += (seq->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
    }
    result.waiting_total_blocks = total_blocks;

    return result;
}

std::vector<std::vector<std::shared_ptr<Sequence>>> Scheduler::_schedule_prefill()
{
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

    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        result.sp_send_counts[dp_idx].resize(attention_sp_);
        result.sp_recv_counts[dp_idx].resize(attention_sp_);

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

        // SP Q Matrix
        result.sp_q_matrix.push_back(std::vector<std::vector<int>>(attention_sp_, std::vector<int>(attention_sp_, 0)));

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
                    if (tokens[sp_idx] > 0) {
                        result.sp_q_matrix[dp_idx][master_sp_idx][sp_idx]++;
                    }
                }
            }
        }
    }

    // Calculate waiting queue block metrics (aggregate across all workers)
    int total_waiting_head_blocks = 0;
    int total_waiting_blocks = 0;
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        auto& worker = worker_state[dp_idx];
        auto& wait_queue = (mode_ != "decode") ? worker->waiting : worker->waiting_migration;
        
        if (!wait_queue.empty()) {
            auto head_seq = wait_queue.front();
            total_waiting_head_blocks += (head_seq->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
        }
        
        for (const auto& seq : wait_queue) {
            total_waiting_blocks += (seq->num_tokens + Sequence::block_size - 1) / Sequence::block_size;
        }
    }
    result.waiting_head_blocks = total_waiting_head_blocks;
    result.waiting_total_blocks = total_waiting_blocks;

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
