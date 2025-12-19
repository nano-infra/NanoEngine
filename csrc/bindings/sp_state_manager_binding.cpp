#include <pybind11/pybind11.h>
#include "opaque_types.h"
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "sp_state_manager.h"
#include "scheduler_utils.h"

namespace py = pybind11;
using namespace nanodeploy;

// Bind the map and deque types
PYBIND11_MAKE_OPAQUE(std::unordered_map<int, std::shared_ptr<BlockManager>>);
PYBIND11_MAKE_OPAQUE(std::deque<std::shared_ptr<Sequence>>);

void bind_sp_state_manager(py::module_& m) {
    // Bind RoutingStrategy
    py::enum_<RoutingStrategy>(m, "RoutingStrategy")
        .value("RoundRobin", RoutingStrategy::RoundRobin)
        .value("LeastToken", RoutingStrategy::LeastToken)
        .value("LeastCache", RoutingStrategy::LeastCache)
        .export_values();

    // Bind BlockManagerMap
    py::bind_map<std::unordered_map<int, std::shared_ptr<BlockManager>>>(m, "BlockManagerMap");

    // Bind SequenceDeque
    py::class_<std::deque<std::shared_ptr<Sequence>>>(m, "SequenceDeque")
        .def(py::init<>())
        .def("append", [](std::deque<std::shared_ptr<Sequence>>& d, std::shared_ptr<Sequence> s) {
            d.push_back(s);
        })
        .def("popleft", [](std::deque<std::shared_ptr<Sequence>>& d) {
            if (d.empty()) throw py::index_error();
            auto s = d.front();
            d.pop_front();
            return s;
        })
        .def("pop", [](std::deque<std::shared_ptr<Sequence>>& d) {
            if (d.empty()) throw py::index_error();
            auto s = d.back();
            d.pop_back();
            return s;
        })
        .def("extendleft", [](std::deque<std::shared_ptr<Sequence>>& d, py::object iterable) {
            for (auto item : iterable) {
                d.push_front(item.cast<std::shared_ptr<Sequence>>());
            }
        })
        .def("remove", [](std::deque<std::shared_ptr<Sequence>>& d, std::shared_ptr<Sequence> s) {
            auto it = std::find(d.begin(), d.end(), s);
            if (it != d.end()) {
                d.erase(it);
            } else {
                throw py::value_error("list.remove(x): x not in list");
            }
        })
        .def("__len__", [](const std::deque<std::shared_ptr<Sequence>>& d) { return d.size(); })
        .def("__bool__", [](const std::deque<std::shared_ptr<Sequence>>& d) { return !d.empty(); })
        .def("__iter__", [](std::deque<std::shared_ptr<Sequence>>& d) {
            return py::make_iterator(d.begin(), d.end());
        }, py::keep_alive<0, 1>());

    // Bind SPStateManager
    py::class_<SPStateManager>(m, "SPStateManager")
        .def(py::init<const std::optional<std::string>&, int, int, int, int, int>(),
             py::arg("engine_id"),
             py::arg("attention_sp"),
             py::arg("num_kvcache_blocks"),
             py::arg("kvcache_block_size"),
             py::arg("max_num_seqs"),
             py::arg("max_num_batched_tokens"))
        
        .def_property_readonly("is_empty", &SPStateManager::is_empty)
        
        .def("can_append", &SPStateManager::can_append,
             py::arg("seq"), py::arg("num_tokens") = 1)
        .def("may_append", &SPStateManager::may_append,
             py::arg("seq"), py::arg("num_tokens") = 1)
        
        .def("can_allocate", &SPStateManager::can_allocate,
             py::arg("seq"),
             py::arg("num_seqs"),
             py::arg("num_batched_tokens"))
             
        .def("allocate", &SPStateManager::allocate, py::arg("seq"))
        .def("deallocate", &SPStateManager::deallocate, py::arg("seq"))
        
        .def_readwrite("block_manager", &SPStateManager::block_manager)
        .def_readwrite("running", &SPStateManager::running)
        .def_readwrite("dummy_seqs", &SPStateManager::dummy_seqs)
        .def_readwrite("routing_strategy", &SPStateManager::routing_strategy);

    m.def("postprocess_sequences", &postprocess_sequences,
        py::arg("worker_states"),
        py::arg("dp_seqs"),
        py::arg("dp_token_ids"),
        py::arg("engine_id"),
        py::arg("eos_id"),
        py::arg("is_prefill"),
        py::arg("to_be_migrated"),
        py::arg("update_metrics")
    );
}
