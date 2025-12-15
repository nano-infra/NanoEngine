#include "block_manager_core.h"

void bind_block_manager(py::module& m)
{
    py::class_<NDCacheBlock>(m, "Block")
        .def_property_readonly("block_id", [](const NDCacheBlock& b) { return b.block_id; })
        .def_readwrite("ref_count", &NDCacheBlock::ref_count)
        .def_property_readonly("hash", &NDCacheBlock::py_hash)
        .def_readwrite("token_ids", &NDCacheBlock::token_ids);

    py::class_<NDCacheBlockManager>(m, "BlockManager")
        .def(py::init<std::optional<std::string>, int, int, int>(),
             py::arg("engine_id"),
             py::arg("sp_idx"),
             py::arg("num_blocks"),
             py::arg("block_size"))
        .def_property_readonly("engine_id", &NDCacheBlockManager::engine_id)
        .def_property_readonly("sp_idx", &NDCacheBlockManager::sp_idx)
        .def_property_readonly("block_size", &NDCacheBlockManager::block_size)
        .def_property_readonly("free_block_ids", &NDCacheBlockManager::free_block_ids)
        .def("can_allocate", &NDCacheBlockManager::can_allocate, py::arg("seq"))
        .def("allocate",
             &NDCacheBlockManager::allocate,
             py::arg("seq"),
             py::arg("token_idx_from") = -1,
             py::arg("token_idx_to") = -1)
        .def("deallocate", &NDCacheBlockManager::deallocate, py::arg("seq"))
        .def("can_append", &NDCacheBlockManager::can_append, py::arg("seq"), py::arg("num_tokens") = 1)
        .def("may_append", &NDCacheBlockManager::may_append, py::arg("seq"), py::arg("num_tokens") = 1)
        .def_static("compute_hash",
                    [](py::list token_ids, py::object prefix_obj) {
                        std::vector<int> tokens;
                        tokens.reserve(token_ids.size());
                        for (auto item : token_ids) {
                            tokens.push_back(item.cast<int>());
                        }
                        std::optional<uint64_t> prefix;
                        if (!prefix_obj.is_none()) {
                            py::int_ pyi = py::reinterpret_borrow<py::int_>(prefix_obj);
                            // Treat -1 as sentinel meaning "no prefix", otherwise parse as uint64.
                            if (!pyi.equal(py::int_(-1))) {
                                prefix = pyi.cast<uint64_t>();
                            }
                        }
                        return py::int_(NDCacheBlockManager::compute_hash(tokens, prefix));
                    },
                    py::arg("token_ids"),
                    py::arg("prefix") = py::int_(-1));
}
