#include <algorithm>
#include <exception>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

#include "nanodeploy/csrc/metrics/sequence_metric.h"
#include "nanodeploy/csrc/sequence/sequence.h"
#include "sequence_generated.h"

#include "scheduler.h"

namespace nanodeploy {

Scheduler::Scheduler(const std::string& engine_id,
                     int                loop_count,
                     int                max_num_seqs,
                     int                max_num_batched_tokens,
                     int                max_model_len,
                     int                eos,
                     int                attention_dp,
                     int                group_size,
                     int                num_kvcache_blocks,
                     int                kvcache_block_size,
                     const std::string& mode):
    engine_id_(engine_id),
    loop_count_(loop_count),
    max_num_seqs_(max_num_seqs),
    max_num_batched_tokens_(max_num_batched_tokens),
    eos_(eos),
    attention_dp_(attention_dp),
    group_size_(group_size),
    max_model_len_(max_model_len),
    num_kvcache_blocks_(num_kvcache_blocks),
    kvcache_block_size_(kvcache_block_size),
    mode_(mode)
{
    // Initialize worker states for each DP rank
    worker_state.reserve(attention_dp_);
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        worker_state.push_back(std::make_shared<GroupManager>(
            engine_id_, group_size_, num_kvcache_blocks, kvcache_block_size, max_num_seqs_, max_num_batched_tokens_));
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

    seq->active(engine_id_, group_size_, attention_dp_, num_kvcache_blocks_);

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

    // Prepare dp_group_seqs and filtered_dp_group_seqs
    result.dp_group_seqs.reserve(attention_dp_ * group_size_);
    result.filtered_dp_group_seqs.reserve(attention_dp_ * group_size_);

    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        for (int group_id = 0; group_id < group_size_; ++group_id) {
            // dp_group_seqs is just dp_seqs[dp_idx] repeated for each group_id
            result.dp_group_seqs.push_back(dp_seqs[dp_idx]);

            // filtered_dp_group_seqs is dp_seqs[dp_idx] filtered by master_group_id
            std::vector<std::shared_ptr<Sequence>> filtered;
            for (const auto& seq : dp_seqs[dp_idx]) {
                if (seq->block_ctx(BlockContextSlot::ACTIVE).master_group_id == group_id) {
                    filtered.push_back(seq);
                }
            }
            result.filtered_dp_group_seqs.push_back(std::move(filtered));
        }
    }

    result.group_send_counts.resize(attention_dp_);
    result.group_recv_counts.resize(attention_dp_);

    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        result.group_send_counts[dp_idx].resize(group_size_);
        result.group_recv_counts[dp_idx].resize(group_size_);

        for (int group_id = 0; group_id < group_size_; ++group_id) {
            // SP Send Count: Number of sequences where this SP rank is MASTER (initiator)
            // AND the sequence is actually distributed (has blocks on > 1 ranks).
            int         send_count = 0;
            const auto& group_seqs = result.filtered_dp_group_seqs[dp_idx * group_size_ + group_id];
            for (const auto& seq : group_seqs) {
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
            result.group_send_counts[dp_idx][group_id] = send_count;

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

                    if (active_ranks > 1 && tokens[group_id] > 0) {
                        recv_count++;
                    }
                }
            }
            result.group_recv_counts[dp_idx][group_id] = recv_count;
        }

        // SP Communication Matrix Logic
        // Initialize matrix for this DP rank: [group_size_][group_size_]
        // result.group_comm_matrix.push_back(
        // std::vector<std::vector<int>>(group_size_, std::vector<int>(group_size_, 0)));

        result.group_q_matrix.push_back(std::vector<std::vector<int>>(group_size_, std::vector<int>(group_size_, 0)));

        // result.group_res_matrix.push_back(
        //     std::vector<std::vector<int>>(group_size_, std::vector<int>(group_size_, 0)));

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
                int master_group_id = seq->block_ctx(BlockContextSlot::ACTIVE).master_group_id;

                // For each participating rank:
                for (int group_id = 0; group_id < group_size_; ++group_id) {
                    if (tokens[group_id] > 0) {
                        // Original matrix (Master -> Participant) - kept for compatibility if needed
                        // result.group_comm_matrix[dp_idx][master_group_id][group_id]++;

                        // Q Matrix: Master broadcast to all Participants
                        // Master sends Q to Participant
                        result.group_q_matrix[dp_idx][master_group_id][group_id]++;

                        // Res Matrix: Participant sends results back to Master
                        // Participant sends Res to Master
                        // result.group_res_matrix[dp_idx][group_id][master_group_id]++;
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

std::vector<std::vector<std::shared_ptr<Sequence>>> Scheduler::_schedule_prefill()
{
    std::vector<std::vector<std::shared_ptr<Sequence>>> scheduled_seqs(attention_dp_);

    // num_seqs and num_batched_tokens track per-DP, per-SP-rank counts for the CURRENT batch
    std::vector<std::unordered_map<int, int>> num_seqs(attention_dp_);
    std::vector<std::unordered_map<int, int>> num_batched_tokens(attention_dp_);

    // Initialize with default values of 0
    for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
        for (int group_id = 0; group_id < group_size_; ++group_id) {
            num_seqs[dp_idx][group_id]           = 0;
            num_batched_tokens[dp_idx][group_id] = 0;
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
        auto& block_ctx    = seq->block_ctx(BlockContextSlot::ACTIVE);
        int   dp_idx       = block_ctx.dp_idx;
        int   master_group = block_ctx.master_group_id;

        int prev_tokens      = seq->num_tokens();
        int budget_remaining = max_num_batched_tokens_ - num_batched_tokens[dp_idx][master_group];
        int new_tokens       = std::min(budget_remaining, seq->num_prompt_tokens() - prev_tokens);
        if (new_tokens <= 0) {
            not_scheduled_prefilling.push_back(seq);
            continue;
        }

        // Advance num_tokens to the new chunk endpoint (blocks pre-allocated at admission)
        seq->set_num_tokens(prev_tokens + new_tokens);
        block_ctx.num_dispatched_tokens[master_group] = seq->num_tokens();

        num_seqs[dp_idx][master_group] += 1;
        num_batched_tokens[dp_idx][master_group] += new_tokens;

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
        auto seq       = waiting_queue.front();
        bool scheduled = false;

        if (routing_strategy == RoutingStrategy::RoundRobin) {
            for (int attempt = 0; attempt < attention_dp_; ++attempt) {
                int  selected_dp_idx = next_dp_idx();
                auto result          = worker_state[selected_dp_idx]->try_allocate(
                    *seq, num_seqs[selected_dp_idx], num_batched_tokens[selected_dp_idx]);
                if (!result)
                    continue;

                auto& block_ctx  = seq->block_ctx(BlockContextSlot::ACTIVE);
                block_ctx.dp_idx = selected_dp_idx;
                int master_group = block_ctx.master_group_id;

                num_seqs[selected_dp_idx][master_group] += 1;
                num_batched_tokens[selected_dp_idx][master_group] += result->new_tokens;

                seq->set_status(SequenceStatus::RUNNING);
                waiting_queue.pop_front();
                worker_state[selected_dp_idx]->running.push_back(seq);
                scheduled_seqs[selected_dp_idx].push_back(seq);

                if (seq->metric) {
                    seq->metric->record_first_scheduled();
                    if (mode_ == "decode")
                        seq->metric->record_decode_scheduled();
                }
                scheduled = true;
                break;
            }
        }
        else if (routing_strategy == RoutingStrategy::LeastBatch || routing_strategy == RoutingStrategy::LeastCache) {
            for (auto it = dp_load_set.begin(); it != dp_load_set.end(); ++it) {
                int  selected_dp_idx = it->second;
                auto result          = worker_state[selected_dp_idx]->try_allocate(
                    *seq, num_seqs[selected_dp_idx], num_batched_tokens[selected_dp_idx]);
                if (!result)
                    continue;

                // erase(it) invalidates the iterator; safe because we break immediately.
                dp_load_set.erase(it);
                int new_load = (routing_strategy == RoutingStrategy::LeastBatch) ?
                                   worker_state[selected_dp_idx]->num_running_seqs() :
                                   worker_state[selected_dp_idx]->num_running_tokens();
                dp_load_set.insert({new_load, selected_dp_idx});

                auto& block_ctx  = seq->block_ctx(BlockContextSlot::ACTIVE);
                block_ctx.dp_idx = selected_dp_idx;
                int master_group = block_ctx.master_group_id;

                num_seqs[selected_dp_idx][master_group] += 1;
                num_batched_tokens[selected_dp_idx][master_group] += result->new_tokens;

                seq->set_status(SequenceStatus::RUNNING);
                waiting_queue.pop_front();
                worker_state[selected_dp_idx]->running.push_back(seq);
                scheduled_seqs[selected_dp_idx].push_back(seq);

                if (seq->metric) {
                    seq->metric->record_first_scheduled();
                    if (mode_ == "decode")
                        seq->metric->record_decode_scheduled();
                }
                scheduled = true;
                break;
            }
        }
        else {
            throw std::runtime_error("Unknown routing strategy");
        }

        if (!scheduled) {
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
        std::vector<int>                      group_lens(group_size_, 0);

        while (!running_queue.empty()) {
            auto seq = running_queue.front();
            running_queue.pop_front();

            int master_rank = seq->block_ctx(BlockContextSlot::ACTIVE).master_group_id;

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
                    group_lens[master_rank] += seq->num_tokens();
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
        for (int group_id = 0; group_id < group_size_; ++group_id) {
            if (group_lens[group_id] == 0) {
                scheduled_seqs[selected_dp_idx].push_back(worker_state[selected_dp_idx]->dummy_seqs[group_id]);
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

void Scheduler::postprocess_worker_func(std::shared_ptr<GroupManager>   state_manager,
                                        const PostprocessWorkerContext* ctx,
                                        PostprocessWorkerContext*       result_ctx,
                                        int                             eos_id,
                                        bool                            is_prefill,
                                        bool                            update_metrics)
{
    try {
        std::unordered_set<std::shared_ptr<Sequence>> dummy_set;
        for (const auto& dummy : state_manager->dummy_seqs) {
            dummy_set.insert(dummy);
        }

        for (const auto& task : ctx->tasks) {
            std::shared_ptr<Sequence> seq = task.seq;

            if (dummy_set.count(seq))
                continue;

            // Detect non-final prefill chunk: num_tokens was set to the chunk
            // endpoint during scheduling; if it's still less than the re-prefill
            // target the sequence is not done prefilling yet.
            // For fresh sequences the target is num_prompt_tokens.
            // For preempted decode sequences num_checkpointed_tokens includes
            // generated tokens that must also be re-prefilled.
            // NOTE: This must NOT be guarded by is_prefill (which reflects the
            // scheduler mode, e.g. disaggregated "prefill" vs "decode").  In a
            // combined (non-disaggregated) scheduler the mode is never "prefill",
            // but chunked sequences still need to continue prefilling.
            int prefill_target = std::max(seq->num_prompt_tokens(), seq->num_checkpointed_tokens());
            if (seq->num_tokens() < prefill_target) {
                // KV has been computed for this chunk's tokens; advance the
                // cached token pointer so the next chunk starts here.
                seq->set_num_cached_tokens(seq->num_tokens());
                seq->set_status(SequenceStatus::PREFILLING);
                result_ctx->chunk_continuations.push_back(seq);
                continue;  // don't process token_ids for this sequence
            }

            // Final prefill chunk completed — transition to RUNNING so the
            // sequence stays in the running queue for subsequent decode steps.
            if (seq->status() == SequenceStatus::PREFILLING) {
                seq->set_status(SequenceStatus::RUNNING);
            }

            for (int token_id : *task.tokens) {

                int master_group_id = seq->block_ctx().master_group_id;
                if (task.group_id != master_group_id) {
                    throw std::runtime_error("group_id mismatch: task.group_id=" + std::to_string(task.group_id)
                                             + " != master_group_id=" + std::to_string(master_group_id)
                                             + " for seq_id=" + std::to_string(seq->seq_id()));
                }

                seq->append_token(token_id, BlockContextSlot::ACTIVE, task.group_id);
                state_manager->add_running_tokens(task.group_id, 1);

                if (update_metrics && seq->metric) {
                    if (seq->metric->num_generated_tokens == 0) {
                        seq->metric->record_first_token();
                        seq->metric->num_generated_tokens = 1;
                    }
                    else {
                        seq->metric->record_token();
                    }
                }

                bool finished = (!seq->sampling_params().ignore_eos && token_id == eos_id)
                                || (seq->num_completed_tokens() >= seq->sampling_params().max_tokens);

                if (finished) {
                    seq->set_status(SequenceStatus::FINISHED);
                    state_manager->deallocate(*seq);
                    break;
                }
                else if (is_prefill) {
                    seq->set_status(SequenceStatus::TO_BE_MIGRATED);
                    seq->migrate();
                    result_ctx->migration_candidates.push_back({seq, result_ctx->dp_idx});
                    break;
                }
            }
        }

        auto& running = state_manager->running;
        if (!running.empty()) {
            running.erase(std::remove_if(running.begin(),
                                         running.end(),
                                         [](const std::shared_ptr<Sequence>& s) {
                                             return s->status() == SequenceStatus::FINISHED
                                                    || s->status() == SequenceStatus::TO_BE_MIGRATED
                                                    || s->status() == SequenceStatus::PREFILLING;
                                         }),
                          running.end());
        }
    }
    catch (...) {
        result_ctx->eptr = std::current_exception();
    }
}

PostprocessResult
Scheduler::postprocess_sequences_impl(const std::vector<std::vector<std::shared_ptr<Sequence>>>& dp_group_seqs,
                                      const std::vector<std::vector<std::vector<int>>>&          dp_group_token_ids,
                                      bool                                                       is_prefill,
                                      bool                                                       update_metrics)
{
    size_t num_dp    = worker_state.size();
    size_t num_dp_sp = dp_group_seqs.size();
    if (num_dp == 0)
        return {};
    if (num_dp_sp % num_dp != 0) {
        throw std::runtime_error("dp_group_seqs size is not a multiple of num_dp");
    }
    size_t num_groups = num_dp_sp / num_dp;

    if (dp_group_token_ids.size() != num_dp_sp) {
        throw std::runtime_error("dp_group_token_ids length mismatch with dp_group_seqs");
    }

    std::vector<PostprocessWorkerContext> contexts(num_dp);

    for (size_t dp_idx = 0; dp_idx < num_dp; ++dp_idx) {
        auto& ctx  = contexts[dp_idx];
        ctx.dp_idx = static_cast<int>(dp_idx);

        for (size_t group_id = 0; group_id < num_groups; ++group_id) {
            size_t      idx          = dp_idx * num_groups + group_id;
            const auto& batch_seqs   = dp_group_seqs[idx];
            const auto& batch_tokens = dp_group_token_ids[idx];

            if (batch_seqs.size() > batch_tokens.size()) {
                throw std::runtime_error("batch_seqs size mismatch with batch_tokens: not enough tokens");
            }

            size_t batch_size = batch_seqs.size();

            for (size_t i = 0; i < batch_size; ++i) {
                ctx.tasks.push_back({batch_seqs[i], &batch_tokens[i], (int)group_id});
            }
        }
    }

    if (thread_pool_) {
        std::vector<std::future<void>> futures;
        futures.reserve(num_dp);

        for (size_t dp_idx = 0; dp_idx < num_dp; ++dp_idx) {
            futures.push_back(thread_pool_->enqueue(postprocess_worker_func,
                                                    worker_state[dp_idx],
                                                    &contexts[dp_idx],
                                                    &contexts[dp_idx],
                                                    eos_,
                                                    is_prefill,
                                                    update_metrics));
        }

        for (auto& f : futures) {
            f.get();
        }
    }
    else {
        std::vector<std::thread> threads;
        threads.reserve(num_dp);

        for (size_t dp_idx = 0; dp_idx < num_dp; ++dp_idx) {
            threads.emplace_back(postprocess_worker_func,
                                 worker_state[dp_idx],
                                 &contexts[dp_idx],
                                 &contexts[dp_idx],
                                 eos_,
                                 is_prefill,
                                 update_metrics);
        }

        for (auto& t : threads) {
            if (t.joinable())
                t.join();
        }
    }

    PostprocessResult result;
    for (const auto& ctx : contexts) {
        if (ctx.eptr) {
            std::rethrow_exception(ctx.eptr);
        }
        result.migrations.insert(
            result.migrations.end(), ctx.migration_candidates.begin(), ctx.migration_candidates.end());
        result.continuations.insert(
            result.continuations.end(), ctx.chunk_continuations.begin(), ctx.chunk_continuations.end());
    }

    return result;
}

void Scheduler::postprocess(const std::vector<std::vector<std::shared_ptr<Sequence>>>& dp_group_seqs,
                            const std::vector<std::vector<std::vector<int>>>&          dp_group_token_ids,
                            bool                                                       update_metrics)
{
    auto result = postprocess_sequences_impl(dp_group_seqs, dp_group_token_ids, mode_ == "prefill", update_metrics);

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
