#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dlengine/csrc/cache/block_manager.h"
#include "dlengine/csrc/sequence/sequence.h"

namespace py = pybind11;
using namespace dlengine;

void bind_block_manager(py::module_& m)
{
    py::class_<Block>(m, "Block")
        .def(py::init<int>())
        .def("update", py::overload_cast<int64_t, const std::vector<int>&>(&Block::update))
        .def("reset", &Block::reset)
        .def_readwrite("block_id", &Block::block_id)
        .def_readwrite("ref_count", &Block::ref_count)
        .def_readwrite("hash", &Block::hash)
        .def_readwrite("token_ids", &Block::token_ids);

    py::class_<BlockManager, std::shared_ptr<BlockManager>>(m, "BlockManager")
        .def(py::init<const std::string&, int, int, int>(),
             py::arg("engine_id"),
             py::arg("group_id"),
             py::arg("num_blocks"),
             py::arg("block_size"))
        .def_static("compute_hash",
                    py::overload_cast<const std::vector<int>&, int64_t>(&BlockManager::compute_hash),
                    py::arg("token_ids"),
                    py::arg("prefix") = -1)
        .def("can_allocate", &BlockManager::can_allocate)
        .def("allocate", &BlockManager::allocate, py::arg("seq"), py::arg("prefix_hint") = -1)
        .def("deallocate", &BlockManager::deallocate)
        .def("can_append", &BlockManager::can_append, py::arg("seq"), py::arg("num_tokens") = 1)
        .def("may_append", &BlockManager::may_append, py::arg("seq"), py::arg("num_tokens") = 1)
        .def_property_readonly("free_block_ids", &BlockManager::free_block_ids)
        .def_property_readonly("num_free_blocks", &BlockManager::num_free_blocks)
        .def_property_readonly("blocks", &BlockManager::blocks)
        // --- L3 (3FS) tiered KV cache hooks ---
        .def("set_l3_enabled", &BlockManager::set_l3_enabled, py::arg("enabled"))
        .def_property_readonly("l3_enabled", &BlockManager::l3_enabled)
        .def("set_l3_resident_hashes", &BlockManager::set_l3_resident_hashes, py::arg("hashes"))
        .def("mark_l3_resident", &BlockManager::mark_l3_resident, py::arg("hashes"))
        .def("is_l3_resident", &BlockManager::is_l3_resident, py::arg("hash"))
        .def("compute_block_hashes", &BlockManager::compute_block_hashes, py::arg("seq"))
        .def("drain_pending_loads", &BlockManager::drain_pending_loads)
        .def("drain_pending_offloads", &BlockManager::drain_pending_offloads);
}
