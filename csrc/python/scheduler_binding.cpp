#include "nanodeploy/scheduler/scheduler.h"
#include "nanodeploy/scheduler/scheduler_utils.h"
#include "opaque_types.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace nanodeploy;

// Make opaque types for Scheduler's containers
PYBIND11_MAKE_OPAQUE(std::deque<std::shared_ptr<Sequence>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<SPStateManager>>);
PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string, std::pair<std::shared_ptr<Sequence>, int>>);

void bind_scheduler_utils(py::module_& m)
{
    py::enum_<ScheduleAction>(m, "ScheduleAction")
        .value("ADMISSION", ScheduleAction::ADMISSION)
        .value("DECODE", ScheduleAction::DECODE)
        .value("KV_CONSOLIDATION", ScheduleAction::KV_CONSOLIDATION)
        .export_values();

    // Bind the postprocess_sequences utility function
    m.def("postprocess_sequences",
          &postprocess_sequences,
          py::arg("worker_states"),
          py::arg("dp_sp_seqs"),
          py::arg("dp_sp_token_ids"),
          py::arg("eos_id"),
          py::arg("is_prefill"),
          py::arg("update_metrics"),
          py::arg("step_duration_ms"),
          py::arg("loop_count"),
          py::arg("thread_pool") = nullptr,
          py::call_guard<py::gil_scoped_release>());

    // Bind the SPStateManagerList type
    py::class_<std::vector<std::shared_ptr<SPStateManager>>>(m, "SPStateManagerList")
        .def(py::init<>())
        .def("__len__", [](const std::vector<std::shared_ptr<SPStateManager>>& v) { return v.size(); })
        .def("__getitem__",
             [](std::vector<std::shared_ptr<SPStateManager>>& v, size_t idx) -> std::shared_ptr<SPStateManager> {
                 if (idx >= v.size())
                     throw py::index_error();
                 return v[idx];
             })
        .def("__setitem__",
             [](std::vector<std::shared_ptr<SPStateManager>>& v, size_t idx, std::shared_ptr<SPStateManager> val) {
                 if (idx >= v.size())
                     throw py::index_error();
                 v[idx] = val;
             })
        .def(
            "__iter__",
            [](std::vector<std::shared_ptr<SPStateManager>>& v) { return py::make_iterator(v.begin(), v.end()); },
            py::keep_alive<0, 1>());

    // Bind the to_be_migrated map type
    py::class_<std::unordered_map<std::string, std::pair<std::shared_ptr<Sequence>, int>>>(m, "MigrationMap")
        .def(py::init<>())
        .def("__len__",
             [](const std::unordered_map<std::string, std::pair<std::shared_ptr<Sequence>, int>>& m) {
                 return m.size();
             })
        .def("__getitem__",
             [](const std::unordered_map<std::string, std::pair<std::shared_ptr<Sequence>, int>>& m,
                const std::string&                                                                key) {
                 auto it = m.find(key);
                 if (it == m.end())
                     throw py::key_error("key '" + key + "' not found");
                 return it->second;
             })
        .def("__setitem__",
             [](std::unordered_map<std::string, std::pair<std::shared_ptr<Sequence>, int>>& m,
                const std::string&                                                          key,
                const std::pair<std::shared_ptr<Sequence>, int>&                            value) { m[key] = value; })
        .def("__contains__",
             [](const std::unordered_map<std::string, std::pair<std::shared_ptr<Sequence>, int>>& m,
                const std::string& key) { return m.count(key) > 0; })
        .def("__delitem__",
             [](std::unordered_map<std::string, std::pair<std::shared_ptr<Sequence>, int>>& m, const std::string& key) {
                 auto it = m.find(key);
                 if (it == m.end())
                     throw py::key_error("key '" + key + "' not found");
                 m.erase(it);
             })
        .def("keys",
             [](const std::unordered_map<std::string, std::pair<std::shared_ptr<Sequence>, int>>& m) {
                 py::list keys;
                 for (const auto& kv : m) {
                     keys.append(kv.first);
                 }
                 return keys;
             })
        .def("items", [](const std::unordered_map<std::string, std::pair<std::shared_ptr<Sequence>, int>>& m) {
            py::list items;
            for (const auto& kv : m) {
                items.append(py::make_tuple(kv.first, kv.second));
            }
            return items;
        });

    // Bind the ScheduleResult struct
    py::class_<ScheduleResult>(m, "ScheduleResult")
        .def_readonly("action", &ScheduleResult::action)
        .def_readwrite("dp_seqs", &ScheduleResult::dp_seqs)
        .def_readwrite("dp_sp_seqs", &ScheduleResult::dp_sp_seqs)
        .def_readwrite("filtered_dp_sp_seqs", &ScheduleResult::filtered_dp_sp_seqs)
        .def_readwrite("is_prefill", &ScheduleResult::is_prefill)
        .def_readonly("kv_consolidation_plan", &ScheduleResult::kv_consolidation_plan)
        .def_readonly("sp_send_counts", &ScheduleResult::sp_send_counts)
        .def_readonly("sp_recv_counts", &ScheduleResult::sp_recv_counts)
        .def_readonly("sp_size_hist_per_dp", &ScheduleResult::sp_size_hist_per_dp)
        // .def_readonly("sp_comm_matrix", &ScheduleResult::sp_comm_matrix)
        .def_readonly("sp_q_matrix", &ScheduleResult::sp_q_matrix)
        .def_readonly("sp_res_matrix", &ScheduleResult::sp_res_matrix)
        .def_readonly("waiting_head_blocks", &ScheduleResult::waiting_head_blocks)
        .def_readonly("waiting_total_blocks", &ScheduleResult::waiting_total_blocks)
        .def_readonly("ls_initial_batch_ids", &ScheduleResult::ls_initial_batch_ids)
        .def_readonly("ls_initial_group_ids", &ScheduleResult::ls_initial_group_ids)
        .def_readonly("ls_initial_kv_dops", &ScheduleResult::ls_initial_kv_dops)
        .def_readonly("ls_initial_kv_ranks", &ScheduleResult::ls_initial_kv_ranks)
        .def_readonly("ls_initial_sequence_ids", &ScheduleResult::ls_initial_sequence_ids)
        .def_readonly("ls_initial_prompt_kv_tokens", &ScheduleResult::ls_initial_prompt_kv_tokens)
        .def_readonly("ls_initial_provisional_pending_targets", &ScheduleResult::ls_initial_provisional_pending_targets)
        .def_readonly("ls_initial_admission_orders", &ScheduleResult::ls_initial_admission_orders)
        .def_readonly("ls_initial_admission_attempts", &ScheduleResult::ls_initial_admission_attempts)
        .def_readonly("ls_initial_is_recovery_batch", &ScheduleResult::ls_initial_is_recovery_batch)
        .def_readonly("ls_initial_parent_batch_ids", &ScheduleResult::ls_initial_parent_batch_ids)
        .def_readonly("ls_initial_admission_kinds", &ScheduleResult::ls_initial_admission_kinds)
        .def_readonly("ls_sealed_batch_ids", &ScheduleResult::ls_sealed_batch_ids)
        .def_readonly("ls_sealed_batch_sequence_ids", &ScheduleResult::ls_sealed_batch_sequence_ids)
        .def_readonly("ls_pending_batch_count", &ScheduleResult::ls_pending_batch_count)
        .def_readonly("ls_pending_request_count", &ScheduleResult::ls_pending_request_count)
        .def_readonly("ls_oldest_pending_batch_age_steps", &ScheduleResult::ls_oldest_pending_batch_age_steps)
        .def_readonly("ls_max_pending_batch_attempts", &ScheduleResult::ls_max_pending_batch_attempts)
        .def_readonly("ls_atomic_admission_no_fit_count", &ScheduleResult::ls_atomic_admission_no_fit_count)
        .def_readonly("ls_atomic_admission_merge_count", &ScheduleResult::ls_atomic_admission_merge_count)
        .def_readonly("ls_atomic_admission_rollback_count", &ScheduleResult::ls_atomic_admission_rollback_count)
        .def_readonly("ls_group_ids", &ScheduleResult::ls_group_ids)
        .def_readonly("ls_group_dp_indices", &ScheduleResult::ls_group_dp_indices)
        .def_readonly("ls_real_batch_sizes", &ScheduleResult::ls_real_batch_sizes)
        .def_readonly("ls_master_dops", &ScheduleResult::ls_master_dops)
        .def_readonly("ls_kv_dops", &ScheduleResult::ls_kv_dops)
        .def_readonly("ls_master_ranks", &ScheduleResult::ls_master_ranks)
        .def_readonly("ls_master_batch_sizes", &ScheduleResult::ls_master_batch_sizes)
        .def_readonly("ls_group_rank_allocations", &ScheduleResult::ls_group_rank_allocations)
        .def_readonly("ls_group_used_kv_tokens", &ScheduleResult::ls_group_used_kv_tokens)
        .def_readonly("ls_group_used_kv_blocks", &ScheduleResult::ls_group_used_kv_blocks)
        .def_readonly("ls_iteration_sequence_ids", &ScheduleResult::ls_iteration_sequence_ids)
        .def_readonly("ls_iteration_master_assignments", &ScheduleResult::ls_iteration_master_assignments)
        .def_readonly("ls_pending_append_blocks_per_master", &ScheduleResult::ls_pending_append_blocks_per_master)
        .def_readonly("ls_new_master_ranks", &ScheduleResult::ls_new_master_ranks)
        .def_readonly("ls_reused_passive_master_ranks", &ScheduleResult::ls_reused_passive_master_ranks)
        .def_readonly("ls_scale_reasons", &ScheduleResult::ls_scale_reasons)
        .def_readonly("ls_historical_kv_migration_bytes", &ScheduleResult::ls_historical_kv_migration_bytes)
        .def_readonly("ls_preempted_sequence_ids", &ScheduleResult::ls_preempted_sequence_ids)
        .def_readonly("ls_preemption_reasons", &ScheduleResult::ls_preemption_reasons)
        .def_readonly("ls_planning_latency_ms", &ScheduleResult::ls_planning_latency_ms)
        .def_readonly("ls_kv_consolidation_candidate", &ScheduleResult::ls_kv_consolidation_candidate)
        .def_readonly("ls_kv_consolidation_group_id", &ScheduleResult::ls_kv_consolidation_group_id)
        .def_readonly("ls_kv_consolidation_source_rank", &ScheduleResult::ls_kv_consolidation_source_rank)
        .def_readonly("ls_kv_consolidation_target_dop", &ScheduleResult::ls_kv_consolidation_target_dop)
        .def_readonly("ls_kv_consolidation_stable_steps", &ScheduleResult::ls_kv_consolidation_stable_steps)
        .def_readonly("ls_kv_consolidation_group_util", &ScheduleResult::ls_kv_consolidation_group_util)
        .def_readonly("ls_kv_consolidation_decision_reason",
                      &ScheduleResult::ls_kv_consolidation_decision_reason);  // Bind the Scheduler class
    py::class_<Scheduler, std::shared_ptr<Scheduler>>(m, "Scheduler")
        .def(py::init([](const std::string& engine_id,
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
                         int                ls_kv_consolidation_max_source_blocks_per_event) {
                 return std::make_shared<Scheduler>(engine_id,
                                                    loop_count,
                                                    max_num_seqs,
                                                    max_num_batched_tokens,
                                                    max_num_recv_seqs,
                                                    eos,
                                                    attention_dp,
                                                    attention_sp,
                                                    num_kvcache_blocks,
                                                    kvcache_block_size,
                                                    mode,
                                                    reserved_blocks_per_req,
                                                    segment_size,
                                                    enable_dynamic_sp_size,
                                                    use_new_decode_dynamic_sp_scheduler,
                                                    dynamic_sp_size_strategy,
                                                    dynamic_sp_long_request_threshold,
                                                    dynamic_sp_long_request_size,
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
                                                    sp_debug,
                                                    fixed_sp_size,
                                                    enable_ls_decode_core_scheduler,
                                                    ls_decode_initial_kv_dop,
                                                    ls_decode_batch_per_master,
                                                    ls_decode_enable_memory_scale_up,
                                                    scheduler_mode,
                                                    ls_kv_consolidation_mode,
                                                    ls_kv_consolidation_candidate_util,
                                                    ls_kv_consolidation_target_high_watermark,
                                                    ls_kv_consolidation_stable_steps,
                                                    ls_kv_consolidation_cooldown_steps,
                                                    ls_kv_consolidation_check_interval_steps,
                                                    ls_kv_consolidation_max_source_blocks_per_event);
             }),
             py::arg("engine_id"),
             py::arg("loop_count"),
             py::arg("max_num_seqs"),
             py::arg("max_num_batched_tokens"),
             py::arg("max_num_recv_seqs"),
             py::arg("eos"),
             py::arg("attention_dp"),
             py::arg("attention_sp"),
             py::arg("num_kvcache_blocks"),
             py::arg("kvcache_block_size"),
             py::arg("mode"),
             py::arg("reserved_blocks_per_req"),
             py::arg("segment_size") = 65536,
             py::arg("enable_dynamic_sp_size"),
             py::arg("use_new_decode_dynamic_sp_scheduler") = false,
             py::arg("dynamic_sp_size_strategy")            = "legacy",
             py::arg("dynamic_sp_long_request_threshold")   = 100000,
             py::arg("dynamic_sp_long_request_size")        = 0,
             py::arg("enable_dynamic_sp_bucket_policy")     = false,
             py::arg("dynamic_sp_bucket_policy")            = "",
             py::arg("attention_cost_a")                    = 1.0,
             py::arg("attention_cost_b")                    = 0.0,
             py::arg("q_cost_a")                            = 1.0,
             py::arg("q_cost_b")                            = 0.0,
             py::arg("res_cost_a")                          = 1.0,
             py::arg("res_cost_b")                          = 0.0,
             py::arg("lse_cost_a")                          = 1.0,
             py::arg("lse_cost_b")                          = 0.0,
             py::arg("q_bytes_per_edge")                    = 1,
             py::arg("res_bytes_per_edge")                  = 1,
             py::arg("lse_bytes_per_edge")                  = 1,
             py::arg("enable_non_uniform_split"),
             py::arg("sp_master_selector"),
             py::arg("sp_debug")                                        = false,
             py::arg("fixed_sp_size")                                   = 0,
             py::arg("enable_ls_decode_core_scheduler")                 = false,
             py::arg("ls_decode_initial_kv_dop")                        = 0,
             py::arg("ls_decode_batch_per_master")                      = 64,
             py::arg("ls_decode_enable_memory_scale_up")                = true,
             py::arg("scheduler_mode")                                  = "centralized",
             py::arg("ls_kv_consolidation_mode")                        = "off",
             py::arg("ls_kv_consolidation_candidate_util")              = 0.50,
             py::arg("ls_kv_consolidation_target_high_watermark")       = 0.80,
             py::arg("ls_kv_consolidation_stable_steps")                = 32,
             py::arg("ls_kv_consolidation_cooldown_steps")              = 64,
             py::arg("ls_kv_consolidation_check_interval_steps")        = 8,
             py::arg("ls_kv_consolidation_max_source_blocks_per_event") = 0)

        // Queue management
        .def("add", &Scheduler::add, py::arg("seq"))

        // Scheduling
        .def("schedule", &Scheduler::schedule, py::call_guard<py::gil_scoped_release>())

        // Postprocessing
        .def("postprocess",
             &Scheduler::postprocess,
             py::arg("dp_seqs"),
             py::arg("dp_token_ids"),
             py::arg("update_metrics"),
             py::arg("step_duration_ms"),
             py::arg("loop_count"),
             py::call_guard<py::gil_scoped_release>())

        // State queries
        .def("is_finished", &Scheduler::is_finished)
        .def("get_total_waiting_size", &Scheduler::get_total_waiting_size)
        .def("get_total_waiting_migration_size", &Scheduler::get_total_waiting_migration_size)
        .def("get_ls_pending_batch_ids", &Scheduler::get_ls_pending_batch_ids)
        .def("get_ls_pending_batch_sequence_ids", &Scheduler::get_ls_pending_batch_sequence_ids)
        .def("get_ls_pending_batch_attempts", &Scheduler::get_ls_pending_batch_attempts)
        .def("get_ls_pending_batch_is_recovery", &Scheduler::get_ls_pending_batch_is_recovery)
        .def("get_ls_pending_batch_parent_batch_ids", &Scheduler::get_ls_pending_batch_parent_batch_ids)
        .def("get_ls_group_ids", &Scheduler::get_ls_group_ids)
        .def("get_ls_group_sequence_ids", &Scheduler::get_ls_group_sequence_ids)
        .def("get_ls_group_initial_batch_ids", &Scheduler::get_ls_group_initial_batch_ids)
        .def("get_ls_group_initial_admission_orders", &Scheduler::get_ls_group_initial_admission_orders)
        .def("get_ls_group_initial_sequence_ids", &Scheduler::get_ls_group_initial_sequence_ids)
        .def("get_ls_active_batch_owners", &Scheduler::get_ls_active_batch_owners)
        .def("get_ls_group_allocated_ranks", &Scheduler::get_ls_group_allocated_ranks, py::arg("group_id"))
        .def("plan_ls_kv_scale_down", &Scheduler::plan_ls_kv_scale_down, py::arg("group_id"), py::arg("source_rank"))
        .def("commit_ls_kv_scale_down", &Scheduler::commit_ls_kv_scale_down, py::arg("plan"))
        .def("abort_ls_kv_scale_down", &Scheduler::abort_ls_kv_scale_down, py::arg("plan"))
        .def("set_ls_admission_failure_after_allocations_for_test",
             &Scheduler::set_ls_admission_failure_after_allocations_for_test,
             py::arg("value"))
        .def("set_ls_admission_failure_after_publications_for_test",
             &Scheduler::set_ls_admission_failure_after_publications_for_test,
             py::arg("value"))

        // Preemption
        .def("preempt", &Scheduler::preempt, py::arg("dp_idx"), py::arg("seq"))

        // Migration management
        .def("free_to_be_migrated",
             py::overload_cast<std::shared_ptr<Sequence>>(&Scheduler::free_to_be_migrated),
             py::arg("seq"))
        .def("free_to_be_migrated",
             py::overload_cast<const std::vector<std::shared_ptr<Sequence>>&>(&Scheduler::free_to_be_migrated),
             py::arg("seqs"))

        // Access methods
        .def("running",
             py::overload_cast<int>(&Scheduler::running),
             py::arg("dp_idx"),
             py::return_value_policy::reference_internal)
        .def("block_manager",
             py::overload_cast<int>(&Scheduler::block_manager),
             py::arg("dp_idx"),
             py::return_value_policy::reference_internal)

        // Public member access
        .def_readwrite("waiting", &Scheduler::waiting)
        .def_readwrite("waiting_migration", &Scheduler::waiting_migration)
        .def_readwrite("worker_state", &Scheduler::worker_state)
        .def_readwrite("to_be_migrated", &Scheduler::to_be_migrated)
        .def_readwrite("routing_strategy", &Scheduler::routing_strategy);
}
