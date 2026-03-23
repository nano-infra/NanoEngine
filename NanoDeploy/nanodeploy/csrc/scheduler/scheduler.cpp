#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "nanodeploy/csrc/metrics/sequence_metric.h"
#include "nanodeploy/csrc/sequence/sequence.h"
#include "sequence_generated.h"

#include "scheduler_utils.h"

#include "scheduler.h"

namespace nanodeploy {

Scheduler::Scheduler(const std::string& engine_id,
                     int                loop_count,
                     int                max_num_seqs,
                     int                max_num_batched_tokens,
                     int                max_model_len,
                     int                eos,
                     int                attention_dp,
                     int                attention_sp,
                     int                num_kvcache_blocks,
                     int                kvcache_block_size,
                     const std::string& mode):
    engine_id_(engine_id),
    loop_count_(loop_count),
    max_num_seqs_(max_num_seqs),
    max_num_batched_tokens_(max_num_batched_tokens),
    eos_(eos),
    attention_dp_(attention_dp),
    attention_sp_(attention_sp),
    max_model_len_(max_model_len),
    num_kvcache_blocks_(num_kvcache_blocks),
    kvcache_block_size_(kvcache_block_size),
    mode_(mode)
{
    // Initialize worker states for each DP rank
    worker_state.reserve(attention_dp_);
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        worker_state.push_back(std::make_shared<SPStateManager>(
            engine_id_, attention_sp_, num_kvcache_blocks, kvcache_block_size, max_num_seqs_, max_num_batched_tokens_));
    }
    // Initialize thread pool with attention_dp_ threads
    thread_pool_ = std::make_unique<ThreadPool>(attention_dp_);
}

void Scheduler::add(std::shared_ptr<Sequence> seq)
{
    int prompt_len = seq->num_prompt_tokens();
    if (prompt_len > max_model_len_) {
        throw std::runtime_error("Prompt length (" + std::to_string(prompt_len) + ") exceeds max_model_len ("
                                 + std::to_string(max_model_len_)
                                 + "). Increase --max_model_len or shorten the prompt.");
    }

    seq->active(engine_id_, attention_sp_, attention_dp_, num_kvcache_blocks_);

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
    if (!prefilling.empty())
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
    int idx        = dp_rr_counter_;
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
                if (seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx == sp_idx) {
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
                    const auto& tokens       = seq->block_ctx(BlockContextSlot::ACTIVE).num_dispatched_tokens;
                    int         active_ranks = 0;
                    for (int count : tokens) {
                        if (count > 0)
                            active_ranks++;
                    }

                    if (active_ranks > 1 && tokens[sp_idx] > 0) {
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
                int master_sp_idx = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx;

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
        result.waiting_head_blocks = (head_seq->num_tokens() + Sequence::block_size - 1) / Sequence::block_size;
    }

    int total_blocks = 0;
    for (const auto& seq : wait_queue) {
        total_blocks += (seq->num_tokens() + Sequence::block_size - 1) / Sequence::block_size;
    }
    result.waiting_total_blocks = total_blocks;

    return result;
}

int Scheduler::compute_chunk_end(const Sequence&                                  seq,
                                 int                                              dp_idx,
                                 const std::vector<std::unordered_map<int, int>>& num_batched_tokens) const
{
    // Use the tightest per-SP budget across all SP ranks for this DP rank.
    int budget = max_num_batched_tokens_;
    for (auto& [sp, tok] : num_batched_tokens[dp_idx]) {
        budget = std::min(budget, max_num_batched_tokens_ - tok);
    }
    // For preempted decode sequences, num_checkpointed_tokens > num_prompt_tokens
    // because it includes generated tokens that must be re-prefilled.
    // For fresh sequences, num_checkpointed_tokens is 0, so we fall back to num_prompt_tokens.
    int target     = std::max(seq.num_prompt_tokens(), seq.num_checkpointed_tokens());
    int num_cached = seq.num_cached_tokens();
    int chunk_end  = num_cached + std::min(budget, target - num_cached);
    return (chunk_end > num_cached) ? chunk_end : -1;
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

    // -----------------------------------------------------------------------
    // Step 1: Schedule PREFILLING sequences (hold allocated blocks, higher
    // priority).  Process before fresh WAITING sequences.
    // -----------------------------------------------------------------------
    std::deque<std::shared_ptr<Sequence>> not_scheduled_prefilling;
    while (!prefilling.empty()) {
        auto seq = prefilling.front();
        prefilling.pop_front();
        auto& block_ctx = seq->block_ctx(BlockContextSlot::ACTIVE);
        int   dp_idx    = block_ctx.dp_idx;
        int   master_sp = block_ctx.master_sp_idx;

        int prev_tokens      = seq->num_tokens();
        int budget_remaining = max_num_batched_tokens_ - num_batched_tokens[dp_idx][master_sp];
        int new_tokens       = std::min(budget_remaining, seq->num_prompt_tokens() - prev_tokens);
        if (new_tokens <= 0) {
            not_scheduled_prefilling.push_back(seq);
            continue;
        }

        // Shrink-before-preempt: only preempt if truly saturated (zero free blocks).
        {
            int free_blocks        = worker_state[dp_idx]->block_manager[master_sp]->num_free_blocks();
            int max_appendable_tok = free_blocks * kvcache_block_size_;
            if (max_appendable_tok <= 0) {
                // Cache is saturated — preempt to avoid starvation deadlock
                preempt(dp_idx, seq);
                continue;
            }
            new_tokens = std::min(new_tokens, max_appendable_tok);
        }

        // Advance num_tokens to the new chunk endpoint and allocate new blocks
        seq->set_num_tokens(prev_tokens + new_tokens);
        worker_state[dp_idx]->block_manager[master_sp]->may_append(*seq, new_tokens);
        block_ctx.num_dispatched_tokens[master_sp] = seq->num_tokens();

        num_seqs[dp_idx][master_sp] += 1;
        num_batched_tokens[dp_idx][master_sp] += new_tokens;

        worker_state[dp_idx]->running.push_back(seq);
        scheduled_seqs[dp_idx].push_back(seq);
    }
    // Put back budget-exhausted prefilling sequences at the front (preserve order)
    for (auto it = not_scheduled_prefilling.rbegin(); it != not_scheduled_prefilling.rend(); ++it)
        prefilling.push_front(*it);

    // -----------------------------------------------------------------------
    // Step 2: Schedule fresh WAITING sequences with chunking
    // -----------------------------------------------------------------------

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
        auto seq             = waiting_queue.front();
        bool scheduled       = false;
        int  orig_num_tokens = seq->num_tokens();  // save for restore on failure

        if (routing_strategy == RoutingStrategy::RoundRobin) {
            // Try all DP ranks in round-robin order
            for (int attempt = 0; attempt < attention_dp_; ++attempt) {
                int selected_dp_idx = next_dp_idx();

                // Compute how many tokens to process in this chunk.
                // Temporarily set num_tokens = chunk_end for can_allocate / allocate.
                int chunk_end = compute_chunk_end(*seq, selected_dp_idx, num_batched_tokens);
                if (chunk_end < 0)
                    continue;
                seq->set_num_tokens(chunk_end);

                // Check if this DP rank can allocate the sequence
                bool can_alloc = worker_state[selected_dp_idx]->can_allocate(
                    *seq, num_seqs[selected_dp_idx], num_batched_tokens[selected_dp_idx]);

                if (!can_alloc) {
                    // Restore num_tokens on failure before trying next DP rank
                    seq->set_num_tokens(orig_num_tokens);
                    continue;
                }

                // Allocate the sequence (also calls set_num_cached_tokens via prefix hits)
                worker_state[selected_dp_idx]->allocate(*seq);

                // Update tracking
                auto& block_ctx   = seq->block_ctx(BlockContextSlot::ACTIVE);
                block_ctx.dp_idx  = selected_dp_idx;
                int master_sp_idx = block_ctx.master_sp_idx;

                num_seqs[selected_dp_idx][master_sp_idx] += 1;
                num_batched_tokens[selected_dp_idx][master_sp_idx] += (seq->num_tokens() - seq->num_cached_tokens());

                // Update sequence status
                seq->set_status(SequenceStatus::RUNNING);

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

                // Compute chunk size
                int chunk_end = compute_chunk_end(*seq, selected_dp_idx, num_batched_tokens);
                if (chunk_end < 0)
                    continue;
                seq->set_num_tokens(chunk_end);

                bool can_alloc = worker_state[selected_dp_idx]->can_allocate(
                    *seq, num_seqs[selected_dp_idx], num_batched_tokens[selected_dp_idx]);

                if (!can_alloc) {
                    seq->set_num_tokens(orig_num_tokens);
                    continue;
                }

                // WARNING: erase(it) invalidates the iterator. This is safe here because
                // we break the loop immediately after.
                dp_load_set.erase(it);
                worker_state[selected_dp_idx]->allocate(*seq);
                int new_load = (routing_strategy == RoutingStrategy::LeastBatch) ?
                                   worker_state[selected_dp_idx]->num_running_seqs() :
                                   worker_state[selected_dp_idx]->num_running_tokens();
                dp_load_set.insert({new_load, selected_dp_idx});

                auto& block_ctx   = seq->block_ctx(BlockContextSlot::ACTIVE);
                block_ctx.dp_idx  = selected_dp_idx;
                int master_sp_idx = block_ctx.master_sp_idx;

                num_seqs[selected_dp_idx][master_sp_idx] += 1;
                num_batched_tokens[selected_dp_idx][master_sp_idx] += (seq->num_tokens() - seq->num_cached_tokens());

                seq->set_status(SequenceStatus::RUNNING);

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

            int master_rank = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx;

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
                    sp_lens[master_rank] += seq->num_tokens();
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
    std::cerr << "Preemption happens for seq_id=" << seq->seq_id() << std::endl;
    seq->set_status(SequenceStatus::WAITING);
    worker_state[dp_idx]->deallocate(*seq);
    // Record the full context length (prompt + any generated tokens) so that
    // re-prefill will rebuild KV for ALL tokens, not just the original prompt.
    // For PREFILLING sequences token_ids().size() == num_prompt_tokens(), so
    // this is equivalent to the old reset.  For decode sequences this correctly
    // preserves the generated continuation.
    int total_tokens = static_cast<int>(seq->token_ids().size());
    seq->set_num_tokens(total_tokens);
    seq->set_num_checkpointed_tokens(total_tokens);
    waiting.push_front(seq);
}

void Scheduler::postprocess(const std::vector<std::vector<std::shared_ptr<Sequence>>>& dp_sp_seqs,
                            const std::vector<std::vector<std::vector<int>>>&          dp_sp_token_ids,
                            bool                                                       update_metrics)
{
    // Call the C++ postprocess_sequences utility directly with shared_ptrs
    auto result = postprocess_sequences(
        worker_state, dp_sp_seqs, dp_sp_token_ids, eos_, mode_ == "prefill", update_metrics, thread_pool_.get());

    // Store migrations
    for (const auto& [seq_shared, dp_idx] : result.migrations) {
        to_be_migrated[seq_shared->seq_id()] = {seq_shared, dp_idx};
    }

    // Route non-final prefill chunks back to the prefilling queue
    for (auto& seq : result.continuations) {
        prefilling.push_back(seq);
    }
}

void Scheduler::free_to_be_migrated(std::shared_ptr<Sequence> seq)
{
    auto it = to_be_migrated.find(seq->seq_id());
    if (it == to_be_migrated.end()) {
        throw std::runtime_error("Sequence " + std::to_string(seq->seq_id()) + " not found in to_be_migrated");
    }

    // IMPORTANT: Use the ORIGINAL sequence from the map, not the passed-in seq.
    // The caller (engine_server.py) creates a minimal Sequence with only seq_id set
    // and empty block tables. Deallocating that would be a no-op, leaking all KV blocks.
    auto& original_seq    = it->second.first;
    int   selected_dp_idx = it->second.second;
    worker_state[selected_dp_idx]->deallocate(*original_seq, BlockContextSlot::MIGRATE);
    to_be_migrated.erase(it);
}

void Scheduler::free_to_be_migrated(const std::vector<std::shared_ptr<Sequence>>& seqs)
{
    for (const auto& seq : seqs) {
        free_to_be_migrated(seq);
    }
}

}  // namespace nanodeploy
