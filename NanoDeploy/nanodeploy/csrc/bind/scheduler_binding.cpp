#include "nanodeploy/csrc/scheduler/scheduler.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace nanodeploy;

// Make opaque types for Scheduler's containers
PYBIND11_MAKE_OPAQUE(std::deque<std::shared_ptr<Sequence>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<GroupManager>>);
PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string, std::pair<std::shared_ptr<Sequence>, int>>);

void bind_scheduler_utils(py::module_& m)
{
    // Bind the GroupManagerList type
    py::class_<std::vector<std::shared_ptr<GroupManager>>>(m, "GroupManagerList")
        .def(py::init<>())
        .def("__len__", [](const std::vector<std::shared_ptr<GroupManager>>& v) { return v.size(); })
        .def("__getitem__",
             [](std::vector<std::shared_ptr<GroupManager>>& v, size_t idx) -> std::shared_ptr<GroupManager> {
                 if (idx >= v.size())
                     throw py::index_error();
                 return v[idx];
             })
        .def("__setitem__",
             [](std::vector<std::shared_ptr<GroupManager>>& v, size_t idx, std::shared_ptr<GroupManager> val) {
                 if (idx >= v.size())
                     throw py::index_error();
                 v[idx] = val;
             })
        .def(
            "__iter__",
            [](std::vector<std::shared_ptr<GroupManager>>& v) { return py::make_iterator(v.begin(), v.end()); },
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
        .def_readwrite("dp_group_seqs", &ScheduleResult::dp_group_seqs)
        .def_readwrite("filtered_dp_group_seqs", &ScheduleResult::filtered_dp_group_seqs)
        .def_readwrite("is_prefill", &ScheduleResult::is_prefill)
        .def_readonly("group_send_counts", &ScheduleResult::group_send_counts)
        .def_readonly("group_recv_counts", &ScheduleResult::group_recv_counts)
        // .def_readonly("group_comm_matrix", &ScheduleResult::group_comm_matrix)
        .def_readonly("group_q_matrix", &ScheduleResult::group_q_matrix)
        // .def_readonly("group_res_matrix", &ScheduleResult::group_res_matrix);
        .def_readonly("waiting_head_blocks", &ScheduleResult::waiting_head_blocks)
        .def_readonly("waiting_total_blocks", &ScheduleResult::waiting_total_blocks);

    // Bind the Scheduler class
    py::class_<Scheduler, std::shared_ptr<Scheduler>>(m, "Scheduler")
        .def(py::init<const std::string&,
                      int,
                      int,
                      int,
                      int,
                      std::vector<int>,
                      int,
                      int,
                      int,
                      int,
                      const std::string&>(),
             py::arg("engine_id"),
             py::arg("loop_count"),
             py::arg("max_num_seqs"),
             py::arg("max_num_batched_tokens"),
             py::arg("max_model_len"),
             py::arg("eos_ids"),
             py::arg("attention_dp"),
             py::arg("group_size"),
             py::arg("num_kvcache_blocks"),
             py::arg("kvcache_block_size"),
             py::arg("mode"))

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
        .def_readonly("group_size", &Scheduler::group_size_)
        .def_readonly("attention_dp", &Scheduler::attention_dp_)
        .def_readwrite("waiting", &Scheduler::waiting)
        .def_readwrite("waiting_migration", &Scheduler::waiting_migration)
        .def_readwrite("worker_state", &Scheduler::worker_state)
        .def_readwrite("to_be_migrated", &Scheduler::to_be_migrated)
        .def_readwrite("routing_strategy", &Scheduler::routing_strategy);
}
