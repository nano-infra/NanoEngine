#include <Python.h>
#include "scheduler_utils.h"
#include "sequence.h"
#include "sequence_metric.h"
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <cassert>
#include <exception>

namespace nanodeploy {

struct CompactTask {
    Sequence* seq_ptr;
    PyObject* seq_obj;
    size_t token_offset;
    uint32_t token_count;
    uint16_t sp_idx;
    uint16_t dp_idx;
};

struct WorkerContext {
    std::vector<CompactTask> tasks;
    std::vector<std::pair<PyObject*, int>> migration_candidates;
    std::exception_ptr eptr = nullptr;

    void reserve(size_t n) { tasks.reserve(n); }
};

void optimized_worker_func(
    SPStateManager* state_manager,
    const WorkerContext* ctx,
    const int* token_storage,
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
            Sequence* seq = task.seq_ptr;

            if (dummy_set.count(seq)) continue;

            #ifndef NDEBUG
            {
                assert(task.sp_idx == seq->block_ctx(engine_id).master_sp_idx);
            }
            #endif

            const int* tokens_begin = token_storage + task.token_offset;
            const int* tokens_end = tokens_begin + task.token_count;

            for (const int* it = tokens_begin; it != tokens_end; ++it) {
                int token_id = *it;
                
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
                    
                    result_ctx->migration_candidates.push_back({task.seq_obj, (int)task.dp_idx});
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

void postprocess_sequences(
    std::vector<SPStateManager*> worker_states,
    py::list dp_seqs,
    py::list dp_token_ids,
    const std::string& engine_id,
    int eos_id,
    bool is_prefill,
    py::dict to_be_migrated,
    bool update_metrics
) {
    Py_ssize_t num_dp = PyList_GET_SIZE(dp_seqs.ptr());
    if (worker_states.size() != (size_t)num_dp) {
         throw std::runtime_error("dp_seqs length mismatch with worker_states");
    }

    std::vector<WorkerContext> contexts(num_dp);
    std::vector<int> global_token_storage;
    
    global_token_storage.reserve(num_dp * 256); 

    for (Py_ssize_t dp_idx = 0; dp_idx < num_dp; ++dp_idx) {
        PyObject* sp_seqs_list = PyList_GET_ITEM(dp_seqs.ptr(), dp_idx);
        PyObject* sp_tokens_list = PyList_GET_ITEM(dp_token_ids.ptr(), dp_idx);
        
        Py_ssize_t num_sp = PyList_GET_SIZE(sp_seqs_list);
        
        auto& ctx = contexts[dp_idx];
        if (num_sp > 0) {
            PyObject* first_batch = PyList_GET_ITEM(sp_seqs_list, 0);
            ctx.reserve(num_sp * PyList_GET_SIZE(first_batch));
        }

        for (Py_ssize_t sp_idx = 0; sp_idx < num_sp; ++sp_idx) {
            PyObject* seq_batch = PyList_GET_ITEM(sp_seqs_list, sp_idx);
            PyObject* token_batch = PyList_GET_ITEM(sp_tokens_list, sp_idx);
            
            Py_ssize_t batch_size = PyList_GET_SIZE(seq_batch);

            for (Py_ssize_t i = 0; i < batch_size; ++i) {
                PyObject* py_seq = PyList_GET_ITEM(seq_batch, i);
                PyObject* py_tokens = PyList_GET_ITEM(token_batch, i);

                Sequence* seq_ptr = py::handle(py_seq).cast<Sequence*>();

                Py_ssize_t n_tokens = PyList_GET_SIZE(py_tokens);
                size_t start_offset = global_token_storage.size();
                
                auto add_token = [&](PyObject* item) {
                    long val = PyLong_AsLong(item);
                    if (val == -1 && PyErr_Occurred()) {
                        throw py::error_already_set();
                    }
                    global_token_storage.push_back((int)val);
                };

                if (n_tokens == 1) {
                    add_token(PyList_GET_ITEM(py_tokens, 0));
                } else {
                    for (Py_ssize_t j = 0; j < n_tokens; ++j) {
                        add_token(PyList_GET_ITEM(py_tokens, j));
                    }
                }

                ctx.tasks.push_back({
                    seq_ptr,
                    py_seq, 
                    start_offset,
                    (uint32_t)n_tokens,
                    (uint16_t)sp_idx,
                    (uint16_t)dp_idx
                });
            }
        }
    }

    {
        py::gil_scoped_release release;
        std::vector<std::thread> threads;
        threads.reserve(num_dp);

        const int* token_ptr = global_token_storage.data();

        for (size_t dp_idx = 0; dp_idx < (size_t)num_dp; ++dp_idx) {
            threads.emplace_back(
                optimized_worker_func,
                worker_states[dp_idx],
                &contexts[dp_idx],
                token_ptr,
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

    for (const auto& ctx : contexts) {
        if (ctx.eptr) {
            std::rethrow_exception(ctx.eptr);
        }
    }

    for (const auto& ctx : contexts) {
        for (const auto& item : ctx.migration_candidates) {
            PyObject* seq_obj = item.first;
            Sequence* seq_ptr = py::handle(seq_obj).cast<Sequence*>();
            to_be_migrated[py::str(seq_ptr->seq_id)] = py::make_tuple(py::handle(seq_obj), item.second);
        }
    }
}

} // namespace nanodeploy