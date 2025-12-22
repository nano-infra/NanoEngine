#include "scheduler.h"
#include "scheduler_utils.h"
#include "nanodeploy/metrics/sequence_metric.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace nanodeploy {

Scheduler::Scheduler(const std::optional<std::string>& engine_id,
                     int                                loop_count,
                     int                                max_num_seqs,
                     int                                max_num_batched_tokens,
                     int                                eos,
                     int                                attention_dp,
                     int                                attention_sp,
                     int                                num_kvcache_blocks,
                     int                                kvcache_block_size,
                     const std::string&                 mode)
    : engine_id_(engine_id)
    , loop_count_(loop_count)
    , max_num_seqs_(max_num_seqs)
    , max_num_batched_tokens_(max_num_batched_tokens)
    , eos_(eos)
    , attention_dp_(attention_dp)
    , attention_sp_(attention_sp)
    , mode_(mode)
{
    // Initialize worker states for each DP rank
    worker_state.reserve(attention_dp_);
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        worker_state.push_back(std::make_shared<SPStateManager>(
            engine_id_, attention_sp_, num_kvcache_blocks, kvcache_block_size, max_num_seqs_, max_num_batched_tokens_));
    }
    // Initialize thread pool with attention_dp_ threads
    thread_pool_ = std::make_unique<ThreadPool>(attention_dp_);}

void Scheduler::add(std::shared_ptr<Sequence> seq)
{
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
    int idx       = dp_rr_counter_;
    dp_rr_counter_ = (dp_rr_counter_ + 1) % attention_dp_;
    return idx;
}

ScheduleResult Scheduler::schedule()
{
    // Try prefill first
    auto dp_seqs = _schedule_prefill();

    // Check if any sequences were scheduled in prefill
    bool has_prefill = false;
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

    ScheduleResult result;
    result.dp_seqs = dp_seqs;
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
                if (seq->block_ctx(engine_id_).master_sp_idx == sp_idx) {
                    filtered.push_back(seq);
                }
            }
            result.filtered_dp_sp_seqs.push_back(std::move(filtered));
        }
    }

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
    } else if (routing_strategy == RoutingStrategy::LeastCache) {
        for (int i = 0; i < attention_dp_; ++i) {
            dp_load_set.insert({worker_state[i]->num_running_tokens(), i});
        }
    }

    while (!waiting_queue.empty()) {
        auto seq = waiting_queue.front();
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
                auto& block_ctx         = seq->block_ctx(engine_id_);
                block_ctx.dp_idx        = selected_dp_idx;
                int master_sp_idx       = block_ctx.master_sp_idx;

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

                // Update load set: remove old load, allocate, insert new load
                dp_load_set.erase(it);
                worker_state[selected_dp_idx]->allocate(*seq);
                int new_load = (routing_strategy == RoutingStrategy::LeastBatch) 
                               ? worker_state[selected_dp_idx]->num_running_seqs()
                               : worker_state[selected_dp_idx]->num_running_tokens();
                dp_load_set.insert({new_load, selected_dp_idx});

                auto& block_ctx         = seq->block_ctx(engine_id_);
                block_ctx.dp_idx        = selected_dp_idx;
                int master_sp_idx       = block_ctx.master_sp_idx;

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

        std::unordered_map<int, int> num_seqs;
        std::deque<std::shared_ptr<Sequence>> skipped;

        while (!running_queue.empty()) {
            auto seq = running_queue.front();
            running_queue.pop_front();

            int master_rank = seq->block_ctx(engine_id_).master_sp_idx;

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
        std::vector<int> sp_lens(attention_sp_, 0);
        for (const auto& seq : scheduled_seqs[selected_dp_idx]) {
            sp_lens[seq->block_ctx(engine_id_).master_sp_idx] += seq->num_tokens;
        }

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
    seq->status = SequenceStatus::WAITING;
    worker_state[dp_idx]->deallocate(*seq);
    seq->num_checkpointed_tokens = static_cast<int>(seq->token_ids.size());
    waiting.push_front(seq);
}

void Scheduler::postprocess(
    const std::vector<std::vector<std::shared_ptr<Sequence>>>& dp_sp_seqs,
    const std::vector<std::vector<std::vector<int>>>&          dp_sp_token_ids,
    bool                                                       update_metrics)
{
    // Call the C++ postprocess_sequences utility directly with shared_ptrs
    auto migrations = postprocess_sequences(
        worker_state, dp_sp_seqs, dp_sp_token_ids, engine_id_.value_or(""), eos_, mode_ == "prefill", update_metrics, thread_pool_.get());

    // Store migrations
    for (const auto& [seq_shared, dp_idx] : migrations) {
        to_be_migrated[seq_shared->seq_id] = {seq_shared, dp_idx};
    }
}

void Scheduler::free_to_be_migrated(std::shared_ptr<Sequence> seq)
{
    auto it = to_be_migrated.find(seq->seq_id);
    if (it == to_be_migrated.end()) {
        throw std::runtime_error("Sequence " + seq->seq_id + " not found in to_be_migrated");
    }

    int selected_dp_idx = it->second.second;
    worker_state[selected_dp_idx]->deallocate(*seq);
    to_be_migrated.erase(it);
}

void Scheduler::free_to_be_migrated(const std::vector<std::shared_ptr<Sequence>>& seqs)
{
    for (const auto& seq : seqs) {
        free_to_be_migrated(seq);
    }
}

}  // namespace nanodeploy
