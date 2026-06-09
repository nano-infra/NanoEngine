#include "nanodeploy/csrc/scheduler/group_manager.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace nanodeploy;

// Bind the map and deque types
PYBIND11_MAKE_OPAQUE(std::unordered_map<int, std::shared_ptr<BlockManager>>);
PYBIND11_MAKE_OPAQUE(std::deque<std::shared_ptr<Sequence>>);

void bind_group_manager(py::module_& m)
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

    // Bind GroupManager
    py::class_<GroupManager, std::shared_ptr<GroupManager>>(m, "GroupManager")
        .def(py::init<const std::string&, int, int, int, int, int>(),
             py::arg("engine_id"),
             py::arg("group_size"),
             py::arg("num_kvcache_blocks"),
             py::arg("kvcache_block_size"),
             py::arg("max_num_seqs"),
             py::arg("max_num_batched_tokens"))

        .def_property_readonly("is_empty", &GroupManager::is_empty)

        .def("can_append", &GroupManager::can_append, py::arg("seq"), py::arg("num_tokens") = 1)
        .def("may_append", &GroupManager::may_append, py::arg("seq"), py::arg("num_tokens") = 1)

        .def("can_allocate",
             &GroupManager::can_allocate,
             py::arg("seq"),
             py::arg("num_seqs"),
             py::arg("num_batched_tokens"))

        .def("allocate", &GroupManager::allocate, py::arg("seq"))
        .def("deallocate", &GroupManager::deallocate, py::arg("seq"), py::arg("slot"))

        .def_readwrite("block_manager", &GroupManager::block_manager)
        .def_readwrite("running", &GroupManager::running)
        .def_readwrite("dummy_seqs", &GroupManager::dummy_seqs)
        .def_readwrite("routing_strategy", &GroupManager::routing_strategy);
}
