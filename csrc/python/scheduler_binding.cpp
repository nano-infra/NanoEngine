#include "nanodeploy/engine/scheduler.h"
#include "nanodeploy/engine/scheduler_utils.h"
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
    // Bind the postprocess_sequences utility function
    m.def("postprocess_sequences",
          &postprocess_sequences,
          py::arg("worker_states"),
          py::arg("dp_sp_seqs"),
          py::arg("dp_sp_token_ids"),
          py::arg("eos_id"),
          py::arg("is_prefill"),
          py::arg("update_metrics") = true,
          py::arg("thread_pool")    = nullptr,
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
        .def_readwrite("dp_seqs", &ScheduleResult::dp_seqs)
        .def_readwrite("dp_sp_seqs", &ScheduleResult::dp_sp_seqs)
        .def_readwrite("filtered_dp_sp_seqs", &ScheduleResult::filtered_dp_sp_seqs)
        .def_readwrite("is_prefill", &ScheduleResult::is_prefill)
        .def_readonly("sp_send_counts", &ScheduleResult::sp_send_counts)
        .def_readonly("sp_recv_counts", &ScheduleResult::sp_recv_counts)
        // .def_readonly("sp_comm_matrix", &ScheduleResult::sp_comm_matrix)
        .def_readonly("sp_q_matrix", &ScheduleResult::sp_q_matrix)
        // .def_readonly("sp_res_matrix", &ScheduleResult::sp_res_matrix);
        .def_readonly("waiting_head_blocks", &ScheduleResult::waiting_head_blocks)
        .def_readonly("waiting_total_blocks", &ScheduleResult::waiting_total_blocks);

    // Bind the Scheduler class
    py::class_<Scheduler, std::shared_ptr<Scheduler>>(m, "Scheduler")
        .def(py::init<const std::string&, int, int, int, int, int, int, int, int, const std::string&, int>(),
             py::arg("engine_id"),
             py::arg("loop_count"),
             py::arg("max_num_seqs"),
             py::arg("max_num_batched_tokens"),
             py::arg("eos"),
             py::arg("attention_dp"),
             py::arg("attention_sp"),
             py::arg("num_kvcache_blocks"),
             py::arg("kvcache_block_size"),
             py::arg("mode"),
             py::arg("segment_size"))

        // Queue management
        .def("add", &Scheduler::add, py::arg("seq"))

        // Scheduling
        .def("schedule", &Scheduler::schedule, py::call_guard<py::gil_scoped_release>())

        // Postprocessing
        .def("postprocess",
             &Scheduler::postprocess,
             py::arg("dp_seqs"),
             py::arg("dp_token_ids"),
             py::arg("update_metrics") = true,
             py::call_guard<py::gil_scoped_release>())

        // State queries
        .def("is_finished", &Scheduler::is_finished)

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
