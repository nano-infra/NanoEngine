#include "scheduler_utils.h"
#include "nanodeploy/metrics/sequence_metric.h"
#include <thread>
#include <algorithm>
#include <unordered_set>
#include <iostream>
#include <exception>
#include <stdexcept>

namespace nanodeploy {

struct Task {
    Sequence* seq;
    const std::vector<int>* tokens;
    int sp_idx;
};

struct WorkerContext {
    std::vector<Task> tasks;
    MigrationList migration_candidates;
    std::exception_ptr eptr = nullptr;
    int dp_idx;

    void reserve(size_t n) { tasks.reserve(n); }
};

static void worker_func(
    SPStateManager* state_manager,
    const WorkerContext* ctx,
    WorkerContext* result_ctx,
    const std::string& engine_id,
    int eos_id,
    bool is_prefill,
    bool update_metrics
) {
    try {
        std::unordered_set<Sequence*> dummy_set;
        for (const auto& dummy : state_manager->dummy_seqs) {
            dummy_set.insert(dummy.get());
        }

        for (const auto& task : ctx->tasks) {
            Sequence* seq = task.seq;

            if (dummy_set.count(seq)) continue;

            for (int token_id : *task.tokens) {
                
                seq->append_token(token_id, engine_id, task.sp_idx);

                if (update_metrics && seq->metric) {
                    if (seq->metric->num_generated_tokens == 0) {
                         seq->metric->record_first_token();
                         seq->metric->num_generated_tokens = 1;
                    } else {
                         seq->metric->record_token();
                    }
                }
                
                bool finished = (!seq->ignore_eos && token_id == eos_id) || 
                                (seq->num_completed_tokens() == seq->max_tokens);
                                
                if (finished) {
                    seq->status = SequenceStatus::FINISHED;
                    state_manager->deallocate(*seq);
                    break; 
                } else if (is_prefill) {
                    seq->status = SequenceStatus::TO_BE_MIGRATED;
                    seq->backup_engine_id = seq->active_engine_id;
                    seq->active_engine_id = std::nullopt;
                    
                    result_ctx->migration_candidates.push_back({seq, result_ctx->dp_idx});
                    break;
                }
            }
        }

        auto& running = state_manager->running;
        if (!running.empty()) {
            running.erase(
                std::remove_if(running.begin(), running.end(),
                    [](const std::shared_ptr<Sequence>& s) {
                        return s->status == SequenceStatus::FINISHED || 
                               s->status == SequenceStatus::TO_BE_MIGRATED;
                    }
                ),
                running.end()
            );
        }
    } catch (...) {
        result_ctx->eptr = std::current_exception();
    }
}

MigrationList postprocess_sequences(
    std::vector<SPStateManager*> worker_states,
    const std::vector<std::vector<std::vector<Sequence*>>>& dp_seqs,
    const std::vector<std::vector<std::vector<std::vector<int>>>>& dp_token_ids,
    const std::string& engine_id,
    int eos_id,
    bool is_prefill,
    bool update_metrics
) {
    size_t num_dp = dp_seqs.size();
    if (worker_states.size() != num_dp) {
         throw std::runtime_error("dp_seqs length mismatch with worker_states");
    }
    if (dp_token_ids.size() != num_dp) {
         throw std::runtime_error("dp_token_ids length mismatch with dp_seqs");
    }

    std::vector<WorkerContext> contexts(num_dp);
    
    for (size_t dp_idx = 0; dp_idx < num_dp; ++dp_idx) {
        auto& ctx = contexts[dp_idx];
        ctx.dp_idx = static_cast<int>(dp_idx);
        
        const auto& sp_seqs = dp_seqs[dp_idx];
        const auto& sp_tokens = dp_token_ids[dp_idx];
        
        if (sp_seqs.size() != sp_tokens.size()) {
            throw std::runtime_error("sp_seqs size mismatch with sp_tokens");
        }

        size_t num_sp = sp_seqs.size();
        for (size_t sp_idx = 0; sp_idx < num_sp; ++sp_idx) {
            const auto& batch_seqs = sp_seqs[sp_idx];
            const auto& batch_tokens = sp_tokens[sp_idx];

            // Begin modification: Remove strict equality check, check if token count is sufficient
            // Python: zip(seqs, tokens) stops when seqs is exhausted, ignoring extra tokens
            if (batch_seqs.size() > batch_tokens.size()) {
                throw std::runtime_error("batch_seqs size mismatch with batch_tokens: not enough tokens");
            }
            // If batch_tokens.size() > batch_seqs.size(), we allow this case (ignore extra tokens)

            size_t batch_size = batch_seqs.size(); // Use the number of seqs as batch size
            // End modification

            for (size_t i = 0; i < batch_size; ++i) {
                ctx.tasks.push_back({
                    batch_seqs[i],
                    &batch_tokens[i],
                    (int)sp_idx
                });
            }
        }
    }

    {
        std::vector<std::thread> threads;
        threads.reserve(num_dp);

        for (size_t dp_idx = 0; dp_idx < num_dp; ++dp_idx) {
            threads.emplace_back(
                worker_func,
                worker_states[dp_idx],
                &contexts[dp_idx],
                &contexts[dp_idx], 
                engine_id,
                eos_id,
                is_prefill,
                update_metrics
            );
        }

        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }
    }

    MigrationList all_migrations;
    for (const auto& ctx : contexts) {
        if (ctx.eptr) {
            std::rethrow_exception(ctx.eptr);
        }
        all_migrations.insert(
            all_migrations.end(),
            ctx.migration_candidates.begin(),
            ctx.migration_candidates.end()
        );
    }

    return all_migrations;
}

} // namespace nanodeploy