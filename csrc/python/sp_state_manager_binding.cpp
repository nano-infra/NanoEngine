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
            py::keep_alive<0, 1>());

    // Bind SPStateManager
    py::class_<SPStateManager, std::shared_ptr<SPStateManager>>(m, "SPStateManager")
        .def(py::init<const std::string&, int, int, int, int, int, int, double, int, bool, bool, 
                      const std::string&, const std::string&, float, float, int>(),
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
             py::arg("enable_non_uniform_split"),
             py::arg("sp_master_selector"),
             // SP size policy parameters (thresholds auto-learned)
             py::arg("sp_size_mode") = "segment",
             py::arg("initial_avg_prompt_length") = 1024.0f,
             py::arg("initial_avg_output_length") = 256.0f,
             py::arg("stats_window_size") = 1000)

        .def_property_readonly("is_empty", &SPStateManager::is_empty)

        .def("can_append", &SPStateManager::can_append, py::arg("seq"), py::arg("num_tokens") = 1)
        .def("may_append", &SPStateManager::may_append, py::arg("seq"), py::arg("num_tokens") = 1)

        .def("can_allocate",
             &SPStateManager::can_allocate,
             py::arg("seq"),
             py::arg("num_seqs"),
             py::arg("num_batched_tokens"))

        .def("allocate", &SPStateManager::allocate, py::arg("seq"))
        .def("deallocate", &SPStateManager::deallocate, py::arg("seq"), py::arg("slot"))

        // Load statistics access (for monitoring/debugging)
        .def("get_avg_prompt_length", [](SPStateManager& m) { return m.load_stats().avg_prompt_length(); })
        .def("get_avg_output_length", [](SPStateManager& m) { return m.load_stats().avg_output_length(); })
        .def("get_stats_num_samples", [](SPStateManager& m) { return m.load_stats().num_samples(); })
        .def("get_expected_waiting_requests", [](SPStateManager& m) { return m.load_stats().expected_waiting_requests(); })
        .def("get_kvcache_imbalance", [](SPStateManager& m) { return m.load_stats().kvcache_imbalance_ratio(); })
        .def("get_learned_long_req_threshold", [](SPStateManager& m) { return m.load_stats().long_req_threshold(); })
        .def("get_learned_imbalance_threshold", [](SPStateManager& m) { return m.load_stats().imbalance_threshold(); })

        .def_readwrite("block_manager", &SPStateManager::block_manager)
        .def_readwrite("running", &SPStateManager::running)
        .def_readwrite("dummy_seqs", &SPStateManager::dummy_seqs)
        .def_readwrite("routing_strategy", &SPStateManager::routing_strategy);
}
