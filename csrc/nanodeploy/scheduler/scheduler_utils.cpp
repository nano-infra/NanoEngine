
#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_set>

#include "nanodeploy/metrics/sequence_metric.h"
#include "nanodeploy/sequence/sequence.h"

#include "thread_pool.h"

#include "scheduler_utils.h"

namespace nanodeploy {

struct Task {
    std::shared_ptr<Sequence> seq;
    const std::vector<int>*   tokens;
    int                       sp_idx;
};

struct WorkerContext {
    std::vector<Task>  tasks;
    MigrationList      migration_candidates;
    std::exception_ptr eptr = nullptr;
    int                dp_idx;

    void reserve(size_t n)
    {
        tasks.reserve(n);
    }
};

static void worker_func(std::shared_ptr<SPStateManager> state_manager,
                        const WorkerContext*            ctx,
                        WorkerContext*                  result_ctx,
                        int                             eos_id,
                        bool                            is_prefill,
                        bool                            update_metrics,
                        double                          step_itl_ms)
{
    try {
        std::unordered_set<std::shared_ptr<Sequence>> dummy_set;
        for (const auto& dummy : state_manager->dummy_seqs) {
            dummy_set.insert(dummy);
        }

        // Track the number of tokens generated per sequence in this step
        std::unordered_map<std::shared_ptr<Sequence>, int> seq_tokens_this_step;
        std::unordered_map<std::shared_ptr<Sequence>, bool> seq_is_first_token;

        for (const auto& task : ctx->tasks) {
            std::shared_ptr<Sequence> seq = task.seq;

            if (dummy_set.count(seq))
                continue;

            // Check if this is the first token for this sequence
            if (seq_tokens_this_step.find(seq) == seq_tokens_this_step.end()) {
                seq_tokens_this_step[seq] = 0;
                seq_is_first_token[seq] = (seq->metric && (seq->metric->num_generated_tokens == 0 ||
                                                           !seq->metric->first_token_time.has_value()));
            }

            for (int token_id : *task.tokens) {

                int master_sp_idx = seq->block_ctx().master_sp_idx_;
                if (task.sp_idx != master_sp_idx) {
                    throw std::runtime_error("sp_idx mismatch: task.sp_idx=" + std::to_string(task.sp_idx)
                                             + " != master_sp_idx=" + std::to_string(master_sp_idx)
                                             + " for seq_id=" + std::to_string(seq->seq_id));
                }

                seq->append_token(token_id, BlockContextSlot::ACTIVE, task.sp_idx);
                state_manager->add_running_tokens(task.sp_idx, 1);
                seq_tokens_this_step[seq]++;

                bool finished =
                    (!seq->ignore_eos && token_id == eos_id) || (seq->num_completed_tokens() >= seq->max_tokens);

                if (finished) {
                    seq->status = SequenceStatus::FINISHED;
                    state_manager->deallocate(*seq);
                    break;
                }
                else if (is_prefill) {
                    seq->status = SequenceStatus::TO_BE_MIGRATED;
                    seq->migrate();
                    std::cout << "migrating" << std::endl;
                    result_ctx->migration_candidates.push_back({seq, result_ctx->dp_idx});
                    break;
                }
            }
        }

        // Record metrics for all sequences processed in this step
        if (update_metrics) {
            for (const auto& [seq, num_tokens] : seq_tokens_this_step) {
                if (seq->metric && num_tokens > 0) {
                    if (seq_is_first_token[seq]) {
                        seq->metric->record_first_token();
                    }
                    seq->metric->record_step_tokens(num_tokens, step_itl_ms);
                }
            }
        }

        auto& running = state_manager->running;
        if (!running.empty()) {
            running.erase(std::remove_if(running.begin(),
                                         running.end(),
                                         [](const std::shared_ptr<Sequence>& s) {
                                             return s->status == SequenceStatus::FINISHED
                                                    || s->status == SequenceStatus::TO_BE_MIGRATED;
                                         }),
                          running.end());
        }
    }
    catch (...) {
        result_ctx->eptr = std::current_exception();
    }
}

MigrationList postprocess_sequences(std::vector<std::shared_ptr<SPStateManager>>               worker_states,
                                    const std::vector<std::vector<std::shared_ptr<Sequence>>>& dp_sp_seqs,
                                    const std::vector<std::vector<std::vector<int>>>&          dp_sp_token_ids,
                                    int                                                        eos_id,
                                    bool                                                       is_prefill,
                                    bool                                                       update_metrics,
                                    double                                                     step_duration_ms,
                                    int                                                        loop_count,
                                    ThreadPool*                                                thread_pool)
{
    size_t num_dp    = worker_states.size();
    size_t num_dp_sp = dp_sp_seqs.size();
    if (num_dp == 0)
        return {};
    if (num_dp_sp % num_dp != 0) {
        throw std::runtime_error("dp_sp_seqs size is not a multiple of num_dp");
    }
    size_t num_sp = num_dp_sp / num_dp;

    if (dp_sp_token_ids.size() != num_dp_sp) {
        throw std::runtime_error("dp_sp_token_ids length mismatch with dp_sp_seqs");
    }

    std::vector<WorkerContext> contexts(num_dp);

    for (size_t dp_idx = 0; dp_idx < num_dp; ++dp_idx) {
        auto& ctx  = contexts[dp_idx];
        ctx.dp_idx = static_cast<int>(dp_idx);

        for (size_t sp_idx = 0; sp_idx < num_sp; ++sp_idx) {
            size_t      idx          = dp_idx * num_sp + sp_idx;
            const auto& batch_seqs   = dp_sp_seqs[idx];
            const auto& batch_tokens = dp_sp_token_ids[idx];

            if (batch_seqs.size() > batch_tokens.size()) {
                throw std::runtime_error("batch_seqs size mismatch with batch_tokens: not enough tokens");
            }

            size_t batch_size = batch_seqs.size();

            for (size_t i = 0; i < batch_size; ++i) {
                ctx.tasks.push_back({batch_seqs[i], &batch_tokens[i], (int)sp_idx});
            }
        }
    }

    // Calculate step ITL: fair share of step time per token slot
    double step_itl_ms = (loop_count > 0) ? (step_duration_ms / loop_count) : 0.0;

    if (thread_pool) {
        std::vector<std::future<void>> futures;
        futures.reserve(num_dp);

        for (size_t dp_idx = 0; dp_idx < num_dp; ++dp_idx) {
            futures.push_back(thread_pool->enqueue(worker_func,
                                                   worker_states[dp_idx],
                                                   &contexts[dp_idx],
                                                   &contexts[dp_idx],
                                                   eos_id,
                                                   is_prefill,
                                                   update_metrics,
                                                   step_itl_ms));
        }

        for (auto& f : futures) {
            f.get();
        }
    }
    else {
        std::vector<std::thread> threads;
        threads.reserve(num_dp);

        for (size_t dp_idx = 0; dp_idx < num_dp; ++dp_idx) {
            threads.emplace_back(worker_func,
                                 worker_states[dp_idx],
                                 &contexts[dp_idx],
                                 &contexts[dp_idx],
                                 eos_id,
                                 is_prefill,
                                 update_metrics,
                                 step_itl_ms);
        }

        for (auto& t : threads) {
            if (t.joinable())
                t.join();
        }
    }

    MigrationList all_migrations;
    for (const auto& ctx : contexts) {
        if (ctx.eptr) {
            std::rethrow_exception(ctx.eptr);
        }
        all_migrations.insert(all_migrations.end(), ctx.migration_candidates.begin(), ctx.migration_candidates.end());
    }

    return all_migrations;
}

}  // namespace nanodeploy
