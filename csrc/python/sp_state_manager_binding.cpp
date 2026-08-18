#include "nanodeploy/scheduler/sp_state_manager.h"
#include "opaque_types.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace nanodeploy;

// Bind the map and deque types
PYBIND11_MAKE_OPAQUE(std::unordered_map<int, std::shared_ptr<BlockManager>>);
PYBIND11_MAKE_OPAQUE(std::deque<std::shared_ptr<Sequence>>);

void bind_sp_state_manager(py::module_& m)
{
    // Bind RoutingStrategy
    py::enum_<RoutingStrategy>(m, "RoutingStrategy")
        .value("RoundRobin", RoutingStrategy::RoundRobin)
        .value("LeastBatch", RoutingStrategy::LeastBatch)
        .value("LeastCache", RoutingStrategy::LeastCache)
        .export_values()
        .def_static("__class_getitem__",
                    [](const std::string& name) {
                        if (name == "RoundRobin")
                            return RoutingStrategy::RoundRobin;
                        if (name == "LeastBatch")
                            return RoutingStrategy::LeastBatch;
                        if (name == "LeastCache")
                            return RoutingStrategy::LeastCache;
                        throw py::key_error(name);
                    })
        .def_property_readonly_static("__members__", [](py::object /* self */) {
            py::dict m;
            m["RoundRobin"] = RoutingStrategy::RoundRobin;
            m["LeastBatch"] = RoutingStrategy::LeastBatch;
            m["LeastCache"] = RoutingStrategy::LeastCache;
            return m;
        });
    py::enum_<SPMasterSelector>(m, "SPMasterSelector")
        .value("RoundRobin", SPMasterSelector::RoundRobin)
        .value("LeastBatch", SPMasterSelector::LeastBatch)
        .value("LeastCache", SPMasterSelector::LeastCache)
        .export_values();

    // Bind BlockManagerMap
    py::bind_map<std::unordered_map<int, std::shared_ptr<BlockManager>>>(m, "BlockManagerMap");

    // Bind SequenceDeque
    py::class_<std::deque<std::shared_ptr<Sequence>>>(m, "SequenceDeque")
        .def(py::init<>())
        .def("append", [](std::deque<std::shared_ptr<Sequence>>& d, std::shared_ptr<Sequence> s) { d.push_back(s); })
        .def("popleft",
             [](std::deque<std::shared_ptr<Sequence>>& d) {
                 if (d.empty())
                     throw py::index_error();
                 auto s = d.front();
                 d.pop_front();
                 return s;
             })
        .def("pop",
             [](std::deque<std::shared_ptr<Sequence>>& d) {
                 if (d.empty())
                     throw py::index_error();
                 auto s = d.back();
                 d.pop_back();
                 return s;
             })
        .def("extendleft",
             [](std::deque<std::shared_ptr<Sequence>>& d, py::object iterable) {
                 for (auto item : iterable) {
                     d.push_front(item.cast<std::shared_ptr<Sequence>>());
                 }
             })
        .def("remove",
             [](std::deque<std::shared_ptr<Sequence>>& d, std::shared_ptr<Sequence> s) {
                 auto it = std::find(d.begin(), d.end(), s);
                 if (it != d.end()) {
                     d.erase(it);
                 }
                 else {
                     throw py::value_error("list.remove(x): x not in list");
                 }
             })
        .def("__getitem__",
             [](const std::deque<std::shared_ptr<Sequence>>& d, int idx) {
                 if (idx < 0)
                     idx += d.size();
                 if (idx < 0 || idx >= (int)d.size())
                     throw py::index_error();
                 return d[idx];
             })
        .def("__setitem__",
             [](std::deque<std::shared_ptr<Sequence>>& d, int idx, std::shared_ptr<Sequence> s) {
                 if (idx < 0)
                     idx += d.size();
                 if (idx < 0 || idx >= (int)d.size())
                     throw py::index_error();
                 d[idx] = s;
             })
        .def("__len__", [](const std::deque<std::shared_ptr<Sequence>>& d) { return d.size(); })
        .def("__bool__", [](const std::deque<std::shared_ptr<Sequence>>& d) { return !d.empty(); })
        .def(
            "__iter__",
            [](std::deque<std::shared_ptr<Sequence>>& d) { return py::make_iterator(d.begin(), d.end()); },
            py::keep_alive<0, 1>());    // Bind SPStateManager
    py::class_<SPStateManager, std::shared_ptr<SPStateManager>>(m, "SPStateManager")
        .def(py::init([](const std::string& engine_id,
                         int                attention_sp,
                         int                num_kvcache_blocks,
                         int                kvcache_block_size,
                         int                max_num_seqs,
                         int                max_num_batched_tokens,
                         int                max_num_recv_seqs,
                         double             reserved_blocks_per_req,
                         int                segment_size,
                         bool               enable_dynamic_sp_size,
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
                         int                fixed_sp_size) {
                 return std::make_shared<SPStateManager>(engine_id,
                                                         attention_sp,
                                                         num_kvcache_blocks,
                                                         kvcache_block_size,
                                                         max_num_seqs,
                                                         max_num_batched_tokens,
                                                         max_num_recv_seqs,
                                                         reserved_blocks_per_req,
                                                         segment_size,
                                                         enable_dynamic_sp_size,
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
                                                         fixed_sp_size);
             }),
             py::arg("engine_id"),
             py::arg("attention_sp"),
             py::arg("num_kvcache_blocks"),
             py::arg("kvcache_block_size"),
             py::arg("max_num_seqs"),
             py::arg("max_num_batched_tokens"),
             py::arg("max_num_recv_seqs"),
             py::arg("reserved_blocks_per_req"),
             py::arg("segment_size") = 65536,
             py::arg("enable_dynamic_sp_size"),
             py::arg("dynamic_sp_size_strategy") = "legacy",
             py::arg("dynamic_sp_long_request_threshold") = 100000,
             py::arg("dynamic_sp_long_request_size") = 0,
             py::arg("enable_dynamic_sp_bucket_policy") = false,
             py::arg("dynamic_sp_bucket_policy") = "",
             py::arg("attention_cost_a") = 1.0,
             py::arg("attention_cost_b") = 0.0,
             py::arg("q_cost_a") = 1.0,
             py::arg("q_cost_b") = 0.0,
             py::arg("res_cost_a") = 1.0,
             py::arg("res_cost_b") = 0.0,
             py::arg("lse_cost_a") = 1.0,
             py::arg("lse_cost_b") = 0.0,
             py::arg("q_bytes_per_edge") = 1,
             py::arg("res_bytes_per_edge") = 1,
             py::arg("lse_bytes_per_edge") = 1,
             py::arg("enable_non_uniform_split"),
             py::arg("sp_master_selector"),
             py::arg("sp_debug") = false,
             py::arg("fixed_sp_size") = 0)

        .def_property_readonly("is_empty", &SPStateManager::is_empty)

        .def("can_append", &SPStateManager::can_append, py::arg("seq"), py::arg("num_tokens") = 1)
        .def("may_append", &SPStateManager::may_append, py::arg("seq"), py::arg("num_tokens") = 1)
        .def("can_fit_lifetime",
             &SPStateManager::can_fit_lifetime,
             py::arg("seq"),
             py::arg("additional_master_tokens"))
        .def("is_control_dummy", &SPStateManager::is_control_dummy, py::arg("seq"))
        .def("num_control_dummy_blocks",
             &SPStateManager::num_control_dummy_blocks,
             py::arg("sp_idx") = -1)
        .def("add_running_tokens",
             &SPStateManager::add_running_tokens,
             py::arg("sp_idx"),
             py::arg("count"))

        .def("can_allocate",
             &SPStateManager::can_allocate,
             py::arg("seq"),
             py::arg("num_seqs"),
             py::arg("num_batched_tokens"))
        .def(
            "apply_planned_placement",
            [](SPStateManager&        state,
               Sequence&              seq,
               int                    master_sp_idx,
               const std::vector<int>& dispatched_tokens) {
                SPStateManager::PlannedPlacement placement;
                placement.master_sp_idx = master_sp_idx;
                placement.num_dispatched_tokens = dispatched_tokens;
                state.apply_planned_placement(seq, placement);
            },
            py::arg("seq"),
            py::arg("master_sp_idx"),
            py::arg("dispatched_tokens"))

        .def("allocate", &SPStateManager::allocate, py::arg("seq"))
        .def("deallocate", &SPStateManager::deallocate, py::arg("seq"), py::arg("slot"))

        .def_readwrite("block_manager", &SPStateManager::block_manager)
        .def_readwrite("running", &SPStateManager::running)
        .def_readwrite("dummy_seqs", &SPStateManager::dummy_seqs)
        .def_property_readonly(
            "num_running_seqs", &SPStateManager::num_running_seqs)
        .def_property_readonly(
            "num_running_tokens", &SPStateManager::num_running_tokens)
        .def_readwrite("routing_strategy", &SPStateManager::routing_strategy);
}
